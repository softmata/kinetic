//! Coordinate frame tree for transform management.
//!
//! A persistent tree of named coordinate frames with SE3 transforms.
//! Equivalent to ROS TF2 but standalone — no ROS dependency.
//!
//! # Usage
//!
//! ```ignore
//! use kinetic_core::frame_tree::FrameTree;
//!
//! let mut tree = FrameTree::new();
//! tree.set_transform("world", "base_link", base_pose, 0.0);
//! tree.set_transform("base_link", "camera_link", camera_cal, 0.0);
//!
//! // Chain resolution: world → base_link → camera_link
//! let cam_in_world = tree.lookup_transform("world", "camera_link")?;
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use nalgebra::Isometry3;

use crate::{KineticError, Pose, Result};

/// A timestamped SE3 transform between two frames.
#[derive(Debug, Clone)]
pub struct StampedTransform {
    /// The transform from parent to child frame.
    pub transform: Isometry3<f64>,
    /// Timestamp in seconds (monotonic or wall-clock, user's choice).
    pub timestamp: f64,
    /// Whether this is a static calibration (never changes from FK).
    pub is_static: bool,
}

/// A tree of coordinate frames with named transforms.
///
/// Thread-safe for concurrent reads via `RwLock`. Supports:
/// - Direct transforms: set parent→child explicitly
/// - Chain resolution: lookup A→C by chaining A→B→C
/// - Inverse resolution: if only B→A exists, infer A→B
/// - FK integration: populate from robot forward kinematics
pub struct FrameTree {
    /// Transforms keyed by (parent_frame, child_frame).
    transforms: RwLock<HashMap<(String, String), StampedTransform>>,
}

impl FrameTree {
    /// Create an empty frame tree.
    pub fn new() -> Self {
        Self {
            transforms: RwLock::new(HashMap::new()),
        }
    }

    /// Set a transform between two frames.
    pub fn set_transform(
        &self,
        parent: &str,
        child: &str,
        transform: Isometry3<f64>,
        timestamp: f64,
    ) {
        let mut map = self.transforms.write().unwrap();
        map.insert(
            (parent.to_string(), child.to_string()),
            StampedTransform {
                transform,
                timestamp,
                is_static: false,
            },
        );
    }

    /// Set a static calibration transform (not overwritten by FK updates).
    pub fn set_static_transform(&self, parent: &str, child: &str, transform: Isometry3<f64>) {
        let mut map = self.transforms.write().unwrap();
        map.insert(
            (parent.to_string(), child.to_string()),
            StampedTransform {
                transform,
                timestamp: 0.0,
                is_static: true,
            },
        );
    }

    /// Look up the transform from `source` frame to `target` frame.
    ///
    /// Chains transforms through intermediate frames if needed.
    /// Inverts transforms when only the reverse direction is stored.
    pub fn lookup_transform(&self, source: &str, target: &str) -> Result<Pose> {
        if source == target {
            return Ok(Pose(Isometry3::identity()));
        }

        let map = self.transforms.read().unwrap();

        // Direct lookup
        if let Some(st) = map.get(&(source.to_string(), target.to_string())) {
            return Ok(Pose(st.transform));
        }

        // Inverse lookup
        if let Some(st) = map.get(&(target.to_string(), source.to_string())) {
            return Ok(Pose(st.transform.inverse()));
        }

        // Chain resolution via BFS
        let frames = self.all_frames_inner(&map);
        let path = self.find_path(source, target, &map, &frames)?;

        let mut result = Isometry3::identity();
        for i in 0..path.len() - 1 {
            let from = &path[i];
            let to = &path[i + 1];

            if let Some(st) = map.get(&(from.clone(), to.clone())) {
                result = result * st.transform;
            } else if let Some(st) = map.get(&(to.clone(), from.clone())) {
                result = result * st.transform.inverse();
            } else {
                return Err(KineticError::PlanningFailed(format!(
                    "No transform between '{}' and '{}'",
                    from, to
                )));
            }
        }

        Ok(Pose(result))
    }

    /// Check if a direct or inverse transform exists between two frames.
    pub fn has_transform(&self, parent: &str, child: &str) -> bool {
        let map = self.transforms.read().unwrap();
        map.contains_key(&(parent.to_string(), child.to_string()))
            || map.contains_key(&(child.to_string(), parent.to_string()))
    }

    /// Check if a transform is static.
    pub fn is_static(&self, parent: &str, child: &str) -> bool {
        let map = self.transforms.read().unwrap();
        map.get(&(parent.to_string(), child.to_string()))
            .map(|st| st.is_static)
            .unwrap_or(false)
    }

    /// List all known frame names.
    pub fn list_frames(&self) -> Vec<String> {
        let map = self.transforms.read().unwrap();
        self.all_frames_inner(&map)
    }

    /// Number of transforms stored.
    pub fn num_transforms(&self) -> usize {
        self.transforms.read().unwrap().len()
    }

    /// Update transforms from robot FK results.
    ///
    /// `link_poses` maps link name → world-frame pose. Only non-static
    /// transforms are overwritten.
    pub fn update_from_fk(&self, link_poses: &HashMap<String, Isometry3<f64>>, timestamp: f64) {
        let mut map = self.transforms.write().unwrap();

        for (link_name, pose) in link_poses {
            let key = ("world".to_string(), link_name.clone());
            // Don't overwrite static transforms
            if let Some(existing) = map.get(&key) {
                if existing.is_static {
                    continue;
                }
            }
            map.insert(
                key,
                StampedTransform {
                    transform: *pose,
                    timestamp,
                    is_static: false,
                },
            );
        }
    }

    /// Clear all non-static transforms.
    pub fn clear_dynamic(&self) {
        let mut map = self.transforms.write().unwrap();
        map.retain(|_, v| v.is_static);
    }

    /// Clear all transforms.
    pub fn clear(&self) {
        self.transforms.write().unwrap().clear();
    }

    // --- Internal helpers ---

    fn all_frames_inner(&self, map: &HashMap<(String, String), StampedTransform>) -> Vec<String> {
        let mut frames = std::collections::HashSet::new();
        for (parent, child) in map.keys() {
            frames.insert(parent.clone());
            frames.insert(child.clone());
        }
        frames.into_iter().collect()
    }

    /// BFS to find a path from source to target through the frame graph.
    fn find_path(
        &self,
        source: &str,
        target: &str,
        map: &HashMap<(String, String), StampedTransform>,
        _frames: &[String],
    ) -> Result<Vec<String>> {
        // Build adjacency list (bidirectional since we can invert transforms)
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (parent, child) in map.keys() {
            adj.entry(parent.clone()).or_default().push(child.clone());
            adj.entry(child.clone()).or_default().push(parent.clone());
        }

        // BFS
        let mut visited: HashMap<String, String> = HashMap::new(); // child → parent in BFS tree
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source.to_string());
        visited.insert(source.to_string(), String::new());

        while let Some(current) = queue.pop_front() {
            if current == target {
                // Reconstruct path
                let mut path = vec![target.to_string()];
                let mut node = target.to_string();
                while node != source {
                    node = visited[&node].clone();
                    path.push(node.clone());
                }
                path.reverse();
                return Ok(path);
            }

            if let Some(neighbors) = adj.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains_key(neighbor) {
                        visited.insert(neighbor.clone(), current.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        Err(KineticError::PlanningFailed(format!(
            "No transform path from '{}' to '{}'",
            source, target
        )))
    }
}

impl Default for FrameTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    fn translation(x: f64, y: f64, z: f64) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    #[test]
    fn identity_lookup() {
        let tree = FrameTree::new();
        let pose = tree.lookup_transform("world", "world").unwrap();
        assert!((pose.0.translation.x).abs() < 1e-10);
    }

    #[test]
    fn direct_lookup() {
        let tree = FrameTree::new();
        tree.set_transform("world", "base", translation(1.0, 0.0, 0.0), 0.0);
        let pose = tree.lookup_transform("world", "base").unwrap();
        assert!((pose.0.translation.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn inverse_lookup() {
        let tree = FrameTree::new();
        tree.set_transform("world", "base", translation(1.0, 2.0, 3.0), 0.0);
        let pose = tree.lookup_transform("base", "world").unwrap();
        assert!((pose.0.translation.x - (-1.0)).abs() < 1e-10);
        assert!((pose.0.translation.y - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn chain_lookup() {
        let tree = FrameTree::new();
        tree.set_transform("world", "base", translation(1.0, 0.0, 0.0), 0.0);
        tree.set_transform("base", "camera", translation(0.0, 0.5, 0.0), 0.0);

        let pose = tree.lookup_transform("world", "camera").unwrap();
        assert!((pose.0.translation.x - 1.0).abs() < 1e-10);
        assert!((pose.0.translation.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn chain_3_hops() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        tree.set_transform("B", "C", translation(0.0, 1.0, 0.0), 0.0);
        tree.set_transform("C", "D", translation(0.0, 0.0, 1.0), 0.0);

        let pose = tree.lookup_transform("A", "D").unwrap();
        assert!((pose.0.translation.x - 1.0).abs() < 1e-10);
        assert!((pose.0.translation.y - 1.0).abs() < 1e-10);
        assert!((pose.0.translation.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn no_path_error() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        let result = tree.lookup_transform("A", "Z");
        assert!(result.is_err());
    }

    #[test]
    fn static_transforms() {
        let tree = FrameTree::new();
        tree.set_static_transform("base", "camera", translation(0.1, 0.0, 0.3));
        assert!(tree.is_static("base", "camera"));
        assert!(!tree.is_static("base", "nonexistent"));
    }

    #[test]
    fn static_survives_clear_dynamic() {
        let tree = FrameTree::new();
        tree.set_static_transform("base", "camera", translation(0.1, 0.0, 0.3));
        tree.set_transform("world", "base", translation(1.0, 0.0, 0.0), 0.0);

        assert_eq!(tree.num_transforms(), 2);
        tree.clear_dynamic();
        assert_eq!(tree.num_transforms(), 1);
        assert!(tree.has_transform("base", "camera"));
        assert!(!tree.has_transform("world", "base"));
    }

    #[test]
    fn update_from_fk() {
        let tree = FrameTree::new();
        tree.set_static_transform("base", "camera", translation(0.1, 0.0, 0.3));

        let mut poses = HashMap::new();
        poses.insert("link1".to_string(), translation(0.0, 0.0, 0.5));
        poses.insert("link2".to_string(), translation(0.0, 0.0, 1.0));

        tree.update_from_fk(&poses, 1.0);

        assert!(tree.has_transform("world", "link1"));
        assert!(tree.has_transform("world", "link2"));
        let pose = tree.lookup_transform("world", "link2").unwrap();
        assert!((pose.0.translation.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn list_frames_includes_all() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        tree.set_transform("B", "C", translation(0.0, 1.0, 0.0), 0.0);

        let frames = tree.list_frames();
        assert!(frames.contains(&"A".to_string()));
        assert!(frames.contains(&"B".to_string()));
        assert!(frames.contains(&"C".to_string()));
    }

    #[test]
    fn concurrent_reads() {
        use std::sync::Arc;
        let tree = Arc::new(FrameTree::new());
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let tree = Arc::clone(&tree);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = tree.lookup_transform("A", "B").unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn default_creates_empty() {
        let tree = FrameTree::default();
        assert_eq!(tree.num_transforms(), 0);
        assert!(tree.list_frames().is_empty());
    }

    #[test]
    fn has_transform_bidirectional() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        assert!(tree.has_transform("A", "B"));
        assert!(tree.has_transform("B", "A")); // reverse
        assert!(!tree.has_transform("A", "C")); // nonexistent
    }

    #[test]
    fn overwrite_transform() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        let p1 = tree.lookup_transform("A", "B").unwrap();
        assert!((p1.0.translation.x - 1.0).abs() < 1e-10);

        tree.set_transform("A", "B", translation(5.0, 0.0, 0.0), 1.0);
        let p2 = tree.lookup_transform("A", "B").unwrap();
        assert!((p2.0.translation.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn static_not_overwritten_by_fk() {
        let tree = FrameTree::new();
        tree.set_static_transform("world", "camera", translation(0.0, 0.0, 0.5));

        let mut poses = HashMap::new();
        poses.insert("camera".to_string(), translation(99.0, 0.0, 0.0));
        tree.update_from_fk(&poses, 1.0);

        // Static transform should NOT be overwritten
        let pose = tree.lookup_transform("world", "camera").unwrap();
        assert!((pose.0.translation.z - 0.5).abs() < 1e-10);
    }

    #[test]
    fn clear_removes_everything() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        tree.set_static_transform("C", "D", translation(2.0, 0.0, 0.0));
        assert_eq!(tree.num_transforms(), 2);
        tree.clear();
        assert_eq!(tree.num_transforms(), 0);
    }

    #[test]
    fn chain_with_mixed_directions() {
        // A→B stored, C→B stored (reverse direction)
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        tree.set_transform("C", "B", translation(0.0, 2.0, 0.0), 0.0);

        // A→C should chain: A→B then B→C (inverted C→B)
        let pose = tree.lookup_transform("A", "C").unwrap();
        assert!((pose.0.translation.x - 1.0).abs() < 1e-10);
        assert!((pose.0.translation.y - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn is_static_false_for_dynamic() {
        let tree = FrameTree::new();
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        assert!(!tree.is_static("A", "B"));
    }

    #[test]
    fn num_transforms_correct() {
        let tree = FrameTree::new();
        assert_eq!(tree.num_transforms(), 0);
        tree.set_transform("A", "B", translation(1.0, 0.0, 0.0), 0.0);
        assert_eq!(tree.num_transforms(), 1);
        tree.set_transform("B", "C", translation(0.0, 1.0, 0.0), 0.0);
        assert_eq!(tree.num_transforms(), 2);
        tree.set_transform("A", "B", translation(2.0, 0.0, 0.0), 1.0); // overwrite
        assert_eq!(tree.num_transforms(), 2); // still 2
    }

    #[test]
    fn update_from_fk_adds_world_prefix() {
        let tree = FrameTree::new();
        let mut poses = HashMap::new();
        poses.insert("link1".to_string(), translation(0.0, 0.0, 0.5));
        tree.update_from_fk(&poses, 0.0);

        // Should create "world" → "link1" transform
        assert!(tree.has_transform("world", "link1"));
        let pose = tree.lookup_transform("world", "link1").unwrap();
        assert!((pose.0.translation.z - 0.5).abs() < 1e-10);
    }

    #[test]
    fn rotation_preserved_in_chain() {
        use nalgebra::{Isometry3, UnitQuaternion, Vector3 as V3};
        let tree = FrameTree::new();

        // A→B: translate + rotate 90° around Z
        let rot = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(V3::z()), std::f64::consts::FRAC_PI_2);
        let ab = Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), rot);
        tree.set_transform("A", "B", ab, 0.0);

        // B→C: translate 1m in X (but B is rotated, so this is 1m in world Y)
        tree.set_transform("B", "C", translation(1.0, 0.0, 0.0), 0.0);

        let ac = tree.lookup_transform("A", "C").unwrap();
        // A→C: should be at ~(1.0, 1.0, 0.0) due to rotation
        assert!((ac.0.translation.x - 1.0).abs() < 0.1);
        assert!((ac.0.translation.y - 1.0).abs() < 0.1);
    }
}
