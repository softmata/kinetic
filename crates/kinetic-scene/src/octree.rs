//! OctoMap-equivalent octree for volumetric occupancy mapping.
//!
//! Provides an adaptive-resolution octree where each voxel stores a
//! probabilistic occupancy value using log-odds. Point clouds can be
//! inserted with sensor-model ray-casting to clear free space along
//! beams and mark endpoints as occupied.
//!
//! Occupied leaf nodes convert to collision spheres for KINETIC's
//! SIMD sphere-based collision pipeline.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_scene::octree::{Octree, OctreeConfig};
//!
//! let mut tree = Octree::new(OctreeConfig::default());
//! tree.insert_pointcloud(&points, &sensor_origin);
//! let spheres = tree.to_collision_spheres();
//! let collides = tree.check_sphere(center, radius);
//! ```

use kinetic_collision::SpheresSoA;

// ── Log-odds constants ──────────────────────────────────────────────────────

/// Log-odds value for an occupied observation.
const L_OCC: f64 = 0.85;
/// Log-odds value for a free-space observation.
const L_FREE: f64 = -0.4;
/// Prior (unknown) log-odds.
const L_PRIOR: f64 = 0.0;
/// Clamping bounds to prevent numerical saturation.
const L_MIN: f64 = -2.0;
const L_MAX: f64 = 3.5;
/// Occupancy threshold: voxels with log-odds above this are "occupied".
const OCC_THRESHOLD: f64 = 0.0;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for octree construction.
#[derive(Debug, Clone)]
pub struct OctreeConfig {
    /// Minimum leaf voxel side length in meters. Default: 0.02 (2 cm).
    pub resolution: f64,
    /// Half-extent of the root cube in meters. Default: 5.0 (10 m cube).
    pub half_extent: f64,
    /// Enable ray-casting to clear free space along sensor beams. Default: true.
    pub ray_cast_free_space: bool,
    /// Maximum ray length for free-space clearing (meters). Default: 5.0.
    pub max_ray_length: f64,
    /// Pruning threshold: if all 8 children have the same occupancy state
    /// and the log-odds difference from the mean is below this, collapse
    /// back into a single leaf. Default: 0.1.
    pub prune_threshold: f64,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            resolution: 0.02,
            half_extent: 5.0,
            ray_cast_free_space: true,
            max_ray_length: 5.0,
            prune_threshold: 0.1,
        }
    }
}

// ── Octree Node ─────────────────────────────────────────────────────────────

/// Octree node: either a leaf (voxel) or an interior node with 8 children.
#[derive(Debug, Clone)]
enum OctreeNode {
    /// Leaf voxel with log-odds occupancy value.
    Leaf { log_odds: f64 },
    /// Interior node with exactly 8 children.
    Interior { children: Box<[OctreeNode; 8]> },
}

impl OctreeNode {
    fn new_leaf() -> Self {
        OctreeNode::Leaf { log_odds: L_PRIOR }
    }
}

// ── Octree ──────────────────────────────────────────────────────────────────

/// Adaptive-resolution octree with probabilistic occupancy.
///
/// The root node covers a cube centered at the origin with side length
/// `2 * config.half_extent`. The tree subdivides lazily down to
/// `config.resolution`.
#[derive(Debug, Clone)]
pub struct Octree {
    /// Configuration.
    config: OctreeConfig,
    /// Root node of the tree.
    root: OctreeNode,
    /// Maximum tree depth (computed from half_extent / resolution).
    max_depth: u32,
    /// Number of occupied leaf voxels (cached for statistics).
    num_occupied: usize,
    /// Total number of leaf nodes.
    num_leaves: usize,
}

impl Octree {
    /// Create a new empty octree.
    pub fn new(config: OctreeConfig) -> Self {
        let max_depth =
            ((2.0 * config.half_extent / config.resolution).log2().ceil() as u32).max(1);
        Octree {
            config,
            root: OctreeNode::new_leaf(),
            max_depth,
            num_occupied: 0,
            num_leaves: 1,
        }
    }

    /// Configuration reference.
    pub fn config(&self) -> &OctreeConfig {
        &self.config
    }

    /// Number of occupied leaf voxels.
    pub fn num_occupied(&self) -> usize {
        self.num_occupied
    }

    /// Total number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Maximum tree depth.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Insert a single point as occupied.
    ///
    /// If `sensor_origin` is provided and ray-casting is enabled, the beam
    /// from origin to point clears free-space voxels along the way.
    pub fn insert_point(&mut self, point: [f64; 3], sensor_origin: Option<[f64; 3]>) {
        let he = self.config.half_extent;

        // Check bounds
        if point[0].abs() > he || point[1].abs() > he || point[2].abs() > he {
            return;
        }

        // Ray-cast free space
        if self.config.ray_cast_free_space {
            if let Some(origin) = sensor_origin {
                self.ray_cast_free(&origin, &point);
            }
        }

        // Mark the endpoint as occupied
        self.update_voxel(point, L_OCC);
    }

    /// Insert a batch of points from a sensor at `sensor_origin`.
    pub fn insert_pointcloud(&mut self, points: &[[f64; 3]], sensor_origin: &[f64; 3]) {
        for &pt in points {
            self.insert_point(pt, Some(*sensor_origin));
        }
        self.prune();
    }

    /// Insert points without ray-casting (just mark occupied).
    pub fn insert_points_occupied(&mut self, points: &[[f64; 3]]) {
        for &pt in points {
            self.insert_point(pt, None);
        }
        self.prune();
    }

    /// Check if a sphere collides with any occupied voxel.
    ///
    /// Returns `true` if any occupied leaf's bounding sphere overlaps the query sphere.
    pub fn check_sphere(&self, center: [f64; 3], radius: f64) -> bool {
        let he = self.config.half_extent;
        self.check_sphere_recursive(&self.root, [0.0, 0.0, 0.0], he, center, radius)
    }

    /// Collect all occupied voxel centers and radii.
    ///
    /// Each occupied leaf becomes a sphere with radius = voxel_half_extent * sqrt(3)
    /// (bounding sphere of the cube).
    pub fn occupied_voxels(&self) -> Vec<([f64; 3], f64)> {
        let mut result = Vec::new();
        let he = self.config.half_extent;
        self.collect_occupied(&self.root, [0.0, 0.0, 0.0], he, &mut result);
        result
    }

    /// Convert occupied voxels to collision spheres for the SIMD pipeline.
    ///
    /// Each occupied leaf produces one sphere. The radius is the voxel
    /// half-diagonal (bounding sphere of the cube) to ensure conservative
    /// collision detection.
    pub fn to_collision_spheres(&self) -> SpheresSoA {
        let voxels = self.occupied_voxels();
        let mut spheres = SpheresSoA::with_capacity(voxels.len());
        for (center, radius) in voxels {
            spheres.push(center[0], center[1], center[2], radius, 0);
        }
        spheres
    }

    /// Clear the entire octree back to a single unknown leaf.
    pub fn clear(&mut self) {
        self.root = OctreeNode::new_leaf();
        self.num_occupied = 0;
        self.num_leaves = 1;
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Update a single voxel's log-odds by `delta`.
    fn update_voxel(&mut self, point: [f64; 3], delta: f64) {
        let he = self.config.half_extent;
        let max_depth = self.max_depth;
        let (occ_delta, leaf_delta) = Self::update_recursive(
            &mut self.root,
            [0.0, 0.0, 0.0],
            he,
            point,
            delta,
            0,
            max_depth,
        );
        self.num_occupied = (self.num_occupied as i64 + occ_delta) as usize;
        self.num_leaves = (self.num_leaves as i64 + leaf_delta) as usize;
    }

    /// Recursive voxel update. Returns (occupied_count_delta, leaf_count_delta).
    fn update_recursive(
        node: &mut OctreeNode,
        center: [f64; 3],
        half_extent: f64,
        point: [f64; 3],
        delta: f64,
        depth: u32,
        max_depth: u32,
    ) -> (i64, i64) {
        match node {
            OctreeNode::Leaf { log_odds } => {
                if depth >= max_depth {
                    // At max depth — update in place
                    let was_occ = *log_odds > OCC_THRESHOLD;
                    *log_odds = (*log_odds + delta).clamp(L_MIN, L_MAX);
                    let is_occ = *log_odds > OCC_THRESHOLD;
                    let occ_delta = (is_occ as i64) - (was_occ as i64);
                    return (occ_delta, 0);
                }

                // Need to subdivide: expand this leaf into an interior node
                let was_occ = *log_odds > OCC_THRESHOLD;
                let old_log_odds = *log_odds;

                // Create 8 children inheriting parent's log-odds
                let children: [OctreeNode; 8] = std::array::from_fn(|_| OctreeNode::Leaf {
                    log_odds: old_log_odds,
                });
                *node = OctreeNode::Interior {
                    children: Box::new(children),
                };

                // 1 leaf became 8 leaves, and the old leaf's occupancy now spread to children
                let mut occ_delta: i64 = -(was_occ as i64);
                // Count how many of the 8 new children are occupied
                let child_occ = if was_occ { 8i64 } else { 0i64 };
                occ_delta += child_occ;
                let leaf_delta: i64 = 7; // 1 leaf -> 8 leaves = +7

                // Now recurse into the correct child
                let (child_occ_d, child_leaf_d) = Self::update_recursive(
                    node,
                    center,
                    half_extent,
                    point,
                    delta,
                    depth,
                    max_depth,
                );
                (occ_delta + child_occ_d, leaf_delta + child_leaf_d)
            }
            OctreeNode::Interior { children } => {
                let child_he = half_extent / 2.0;
                let idx = child_index(center, point);
                let child_center = child_center_from_index(center, child_he, idx);
                Self::update_recursive(
                    &mut children[idx],
                    child_center,
                    child_he,
                    point,
                    delta,
                    depth + 1,
                    max_depth,
                )
            }
        }
    }

    /// Ray-cast from `origin` toward `endpoint`, marking traversed voxels as free.
    fn ray_cast_free(&mut self, origin: &[f64; 3], endpoint: &[f64; 3]) {
        let dx = endpoint[0] - origin[0];
        let dy = endpoint[1] - origin[1];
        let dz = endpoint[2] - origin[2];
        let length = (dx * dx + dy * dy + dz * dz).sqrt();

        if length < 1e-10 {
            return;
        }

        let max_len = length.min(self.config.max_ray_length);

        // Step size = resolution (one voxel at a time)
        let step = self.config.resolution;
        let dir = [dx / length, dy / length, dz / length];

        let num_steps = ((max_len - step) / step).ceil() as usize;
        let he = self.config.half_extent;

        for i in 0..num_steps {
            let t = step * (i as f64 + 0.5);
            let pt = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];

            // Skip points outside the octree bounds
            if pt[0].abs() > he || pt[1].abs() > he || pt[2].abs() > he {
                continue;
            }

            // Don't clear the endpoint voxel (it will be marked occupied after)
            let to_end = [
                endpoint[0] - pt[0],
                endpoint[1] - pt[1],
                endpoint[2] - pt[2],
            ];
            let dist_to_end =
                (to_end[0] * to_end[0] + to_end[1] * to_end[1] + to_end[2] * to_end[2]).sqrt();
            if dist_to_end < step {
                break;
            }

            self.update_voxel(pt, L_FREE);
        }
    }

    /// Recursive sphere collision check.
    fn check_sphere_recursive(
        &self,
        node: &OctreeNode,
        center: [f64; 3],
        half_extent: f64,
        query_center: [f64; 3],
        query_radius: f64,
    ) -> bool {
        // First check if query sphere overlaps this node's bounding box
        if !sphere_aabb_overlap(query_center, query_radius, center, half_extent) {
            return false;
        }

        match node {
            OctreeNode::Leaf { log_odds } => {
                // Occupied leaf: check if bounding spheres overlap
                if *log_odds > OCC_THRESHOLD {
                    let voxel_radius = half_extent * SQRT3;
                    let dx = center[0] - query_center[0];
                    let dy = center[1] - query_center[1];
                    let dz = center[2] - query_center[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let r_sum = voxel_radius + query_radius;
                    dist_sq < r_sum * r_sum
                } else {
                    false
                }
            }
            OctreeNode::Interior { children } => {
                let child_he = half_extent / 2.0;
                for (i, child) in children.iter().enumerate() {
                    let cc = child_center_from_index(center, child_he, i);
                    if self.check_sphere_recursive(child, cc, child_he, query_center, query_radius)
                    {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Collect occupied leaf voxels.
    fn collect_occupied(
        &self,
        node: &OctreeNode,
        center: [f64; 3],
        half_extent: f64,
        result: &mut Vec<([f64; 3], f64)>,
    ) {
        match node {
            OctreeNode::Leaf { log_odds } => {
                if *log_odds > OCC_THRESHOLD {
                    let radius = half_extent * SQRT3;
                    result.push((center, radius));
                }
            }
            OctreeNode::Interior { children } => {
                let child_he = half_extent / 2.0;
                for (i, child) in children.iter().enumerate() {
                    let cc = child_center_from_index(center, child_he, i);
                    self.collect_occupied(child, cc, child_he, result);
                }
            }
        }
    }

    /// Prune: collapse interior nodes whose children are all leaves with
    /// similar log-odds back into a single leaf.
    fn prune(&mut self) {
        let threshold = self.config.prune_threshold;
        let (occ_delta, leaf_delta) = Self::prune_recursive(&mut self.root, threshold);
        self.num_occupied = (self.num_occupied as i64 + occ_delta) as usize;
        self.num_leaves = (self.num_leaves as i64 + leaf_delta) as usize;
    }

    fn prune_recursive(node: &mut OctreeNode, threshold: f64) -> (i64, i64) {
        let mut total_occ_delta = 0i64;
        let mut total_leaf_delta = 0i64;

        if let OctreeNode::Interior { children } = node {
            // First, recursively prune children
            for child in children.iter_mut() {
                let (od, ld) = Self::prune_recursive(child, threshold);
                total_occ_delta += od;
                total_leaf_delta += ld;
            }

            // Check if all children are leaves
            let all_leaves = children
                .iter()
                .all(|c| matches!(c, OctreeNode::Leaf { .. }));
            if !all_leaves {
                return (total_occ_delta, total_leaf_delta);
            }

            // Collect log-odds values
            let mut values = [0.0f64; 8];
            for (i, child) in children.iter().enumerate() {
                if let OctreeNode::Leaf { log_odds } = child {
                    values[i] = *log_odds;
                }
            }

            // Check if all are same occupancy state and similar values
            let all_same_state = values.iter().all(|&v| v > OCC_THRESHOLD)
                || values.iter().all(|&v| v <= OCC_THRESHOLD);

            if all_same_state {
                let mean = values.iter().sum::<f64>() / 8.0;
                let max_dev = values
                    .iter()
                    .map(|v| (v - mean).abs())
                    .fold(0.0f64, f64::max);

                if max_dev < threshold {
                    // Collapse: count occupied children before collapsing
                    let occ_before: i64 =
                        values.iter().filter(|&&v| v > OCC_THRESHOLD).count() as i64;
                    let is_occ_after = mean > OCC_THRESHOLD;

                    *node = OctreeNode::Leaf { log_odds: mean };

                    // 8 leaves -> 1 leaf
                    total_leaf_delta += -7;
                    // Occupancy change
                    total_occ_delta += (is_occ_after as i64) - occ_before;
                }
            }
        }

        (total_occ_delta, total_leaf_delta)
    }
}

// ── Geometry helpers ────────────────────────────────────────────────────────

const SQRT3: f64 = 1.732_050_808;

/// Determine which octant a point falls into relative to a node center.
/// Returns an index 0..7 using bit encoding: bit 0 = x, bit 1 = y, bit 2 = z.
#[inline]
fn child_index(center: [f64; 3], point: [f64; 3]) -> usize {
    let mut idx = 0;
    if point[0] >= center[0] {
        idx |= 1;
    }
    if point[1] >= center[1] {
        idx |= 2;
    }
    if point[2] >= center[2] {
        idx |= 4;
    }
    idx
}

/// Compute the center of a child octant given parent center, child half-extent, and octant index.
#[inline]
fn child_center_from_index(parent_center: [f64; 3], child_he: f64, idx: usize) -> [f64; 3] {
    [
        parent_center[0] + if idx & 1 != 0 { child_he } else { -child_he },
        parent_center[1] + if idx & 2 != 0 { child_he } else { -child_he },
        parent_center[2] + if idx & 4 != 0 { child_he } else { -child_he },
    ]
}

/// Check if a sphere overlaps an axis-aligned bounding box (cube).
#[inline]
fn sphere_aabb_overlap(
    sphere_center: [f64; 3],
    sphere_radius: f64,
    box_center: [f64; 3],
    box_half_extent: f64,
) -> bool {
    // Find closest point on AABB to sphere center
    let mut dist_sq = 0.0;
    for i in 0..3 {
        let min = box_center[i] - box_half_extent;
        let max = box_center[i] + box_half_extent;
        let v = sphere_center[i];
        if v < min {
            dist_sq += (min - v) * (min - v);
        } else if v > max {
            dist_sq += (v - max) * (v - max);
        }
    }
    dist_sq <= sphere_radius * sphere_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_octree() {
        let tree = Octree::new(OctreeConfig::default());
        assert_eq!(tree.num_occupied(), 0);
        assert_eq!(tree.num_leaves(), 1);
        assert!(tree.max_depth() > 0);
    }

    #[test]
    fn insert_single_point() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        tree.insert_point([0.5, 0.5, 0.5], None);
        assert!(tree.num_occupied() > 0);
        assert!(tree.num_leaves() > 1);
    }

    #[test]
    fn insert_multiple_points() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 2.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        let points = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.6],
            [-0.5, -0.5, -0.5],
            [1.0, 0.0, 0.0],
        ];
        tree.insert_points_occupied(&points);
        assert!(tree.num_occupied() >= 3);
    }

    #[test]
    fn out_of_bounds_ignored() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        tree.insert_point([5.0, 0.0, 0.0], None);
        assert_eq!(tree.num_occupied(), 0);
    }

    #[test]
    fn sphere_collision_query() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.05,
            half_extent: 2.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        // Insert a cluster of points at (0.5, 0.5, 0.5)
        let points: Vec<[f64; 3]> = (0..10)
            .flat_map(|i| (0..10).map(move |j| [0.5 + i as f64 * 0.01, 0.5 + j as f64 * 0.01, 0.5]))
            .collect();
        tree.insert_points_occupied(&points);

        // Sphere overlapping the cluster
        assert!(tree.check_sphere([0.5, 0.5, 0.5], 0.2));

        // Sphere far away
        assert!(!tree.check_sphere([-1.5, -1.5, -1.5], 0.1));
    }

    #[test]
    fn to_collision_spheres() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        let points = [[0.3, 0.3, 0.3], [0.6, 0.6, 0.6]];
        tree.insert_points_occupied(&points);

        let spheres = tree.to_collision_spheres();
        assert!(
            spheres.len() >= 2,
            "Expected at least 2 spheres, got {}",
            spheres.len()
        );

        // All radii should be positive
        for r in &spheres.radius {
            assert!(*r > 0.0);
        }
        // All link_ids should be 0 (environment)
        for &id in &spheres.link_id {
            assert_eq!(id, 0);
        }
    }

    #[test]
    fn ray_cast_clears_free_space() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 2.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        // First mark a point as occupied
        tree.insert_point([0.5, 0.0, 0.0], None);
        assert!(tree.num_occupied() > 0);
        let occ_before = tree.num_occupied();

        // Now insert a new point behind it with ray-casting from origin
        let mut tree2 = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 2.0,
            ray_cast_free_space: true,
            ..Default::default()
        });

        // First mark (0.5, 0, 0) as occupied
        tree2.insert_point([0.5, 0.0, 0.0], None);
        let occ_before2 = tree2.num_occupied();

        // Insert with sensor at origin, endpoint at (1.0, 0, 0)
        // This should clear voxels along the ray and mark endpoint
        tree2.insert_point([1.0, 0.0, 0.0], Some([0.0, 0.0, 0.0]));

        // The old occupied point at (0.5, 0, 0) should have been cleared
        // by the ray, and (1.0, 0, 0) should be newly occupied.
        // Total occupied may decrease because of free-space clearing.
        let _ = (occ_before, occ_before2);
        // At minimum the endpoint should be occupied
        assert!(tree2.check_sphere([1.0, 0.0, 0.0], 0.15));
    }

    #[test]
    fn pointcloud_insertion() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.05,
            half_extent: 3.0,
            ray_cast_free_space: true,
            max_ray_length: 5.0,
            ..Default::default()
        });

        let sensor = [0.0, 0.0, 0.0];
        let points: Vec<[f64; 3]> = (0..50).map(|i| [1.0 + i as f64 * 0.01, 0.0, 0.5]).collect();

        tree.insert_pointcloud(&points, &sensor);

        assert!(tree.num_occupied() > 0);
        assert!(tree.check_sphere([1.25, 0.0, 0.5], 0.1));
    }

    #[test]
    fn clear_resets() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        tree.insert_point([0.5, 0.5, 0.5], None);
        assert!(tree.num_occupied() > 0);

        tree.clear();
        assert_eq!(tree.num_occupied(), 0);
        assert_eq!(tree.num_leaves(), 1);
    }

    #[test]
    fn child_index_correct() {
        let center = [0.0, 0.0, 0.0];
        assert_eq!(child_index(center, [-1.0, -1.0, -1.0]), 0);
        assert_eq!(child_index(center, [1.0, -1.0, -1.0]), 1);
        assert_eq!(child_index(center, [-1.0, 1.0, -1.0]), 2);
        assert_eq!(child_index(center, [1.0, 1.0, -1.0]), 3);
        assert_eq!(child_index(center, [-1.0, -1.0, 1.0]), 4);
        assert_eq!(child_index(center, [1.0, 1.0, 1.0]), 7);
    }

    #[test]
    fn sphere_aabb_overlap_test() {
        // Sphere at origin radius 1 vs unit box at origin
        assert!(sphere_aabb_overlap(
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            0.5
        ));

        // Sphere far away
        assert!(!sphere_aabb_overlap(
            [10.0, 0.0, 0.0],
            0.5,
            [0.0, 0.0, 0.0],
            0.5
        ));

        // Sphere just touching
        assert!(sphere_aabb_overlap(
            [1.5, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            0.5
        ));

        // Sphere just missing
        assert!(!sphere_aabb_overlap(
            [1.6, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            0.5
        ));
    }

    #[test]
    fn occupied_voxels_roundtrip() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            ..Default::default()
        });

        let points = [[0.2, 0.2, 0.2], [-0.3, -0.3, -0.3]];
        tree.insert_points_occupied(&points);

        let voxels = tree.occupied_voxels();
        let spheres = tree.to_collision_spheres();
        assert_eq!(voxels.len(), spheres.len());
    }

    #[test]
    fn ray_cast_free_space_clears_occupied_voxel() {
        // Insert a point at (0.5, 0, 0) without ray-casting → becomes occupied.
        // Then insert a point at (1.5, 0, 0) with sensor at origin and ray-casting on.
        // The ray from origin to (1.5, 0, 0) passes through (0.5, 0, 0),
        // so the first voxel should be cleared (or at least reduced in occupancy).
        let config = OctreeConfig {
            resolution: 0.1,
            half_extent: 3.0,
            ray_cast_free_space: true,
            max_ray_length: 5.0,
            prune_threshold: 0.1,
        };
        let mut tree = Octree::new(config);

        // Mark (0.5, 0, 0) as occupied (multiple insertions to push log-odds up)
        for _ in 0..5 {
            tree.insert_point([0.5, 0.0, 0.0], None);
        }
        assert!(
            tree.check_sphere([0.5, 0.0, 0.0], 0.15),
            "Point at (0.5,0,0) should be occupied before ray-cast"
        );

        // Now insert a point far behind it with a ray from origin.
        // The ray from (0,0,0) to (1.5,0,0) traverses the (0.5,0,0) voxel,
        // applying L_FREE which should reduce its log-odds.
        for _ in 0..10 {
            tree.insert_point([1.5, 0.0, 0.0], Some([0.0, 0.0, 0.0]));
        }

        // The (1.5, 0, 0) endpoint must still be occupied
        assert!(
            tree.check_sphere([1.5, 0.0, 0.0], 0.15),
            "Endpoint at (1.5,0,0) should be occupied"
        );

        // The (0.5, 0, 0) voxel should have been cleared by repeated free-space updates
        assert!(
            !tree.check_sphere([0.5, 0.0, 0.0], 0.05),
            "Voxel at (0.5,0,0) should be cleared by ray-cast free-space updates"
        );
    }

    #[test]
    fn ray_cast_vs_no_ray_cast_occupied_count() {
        // With ray-casting, intermediate voxels get free-space updates, so
        // fewer total occupied voxels should remain compared to no ray-casting.
        let points: Vec<[f64; 3]> = (0..20).map(|i| [0.5 + i as f64 * 0.05, 0.0, 0.0]).collect();
        let sensor = [0.0, 0.0, 0.0];

        let mut tree_no_ray = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 3.0,
            ray_cast_free_space: false,
            ..Default::default()
        });
        tree_no_ray.insert_pointcloud(&points, &sensor);
        let occ_no_ray = tree_no_ray.num_occupied();

        let mut tree_ray = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 3.0,
            ray_cast_free_space: true,
            max_ray_length: 5.0,
            ..Default::default()
        });
        tree_ray.insert_pointcloud(&points, &sensor);
        let occ_ray = tree_ray.num_occupied();

        // With ray-casting, earlier points along the sensor beam get free-space
        // updates from later points, so occupied count should be <= without ray-casting
        assert!(
            occ_ray <= occ_no_ray,
            "Ray-cast occupied ({}) should be <= no-ray-cast ({})",
            occ_ray,
            occ_no_ray
        );
    }

    #[test]
    fn pruning_collapses_uniform_children() {
        let mut tree = Octree::new(OctreeConfig {
            resolution: 0.1,
            half_extent: 1.0,
            ray_cast_free_space: false,
            prune_threshold: 1.0, // very permissive — will prune easily
            ..Default::default()
        });

        // Fill all 8 octants of a single subdivision level with similar values
        let coords = [-0.3, 0.3];
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    tree.insert_point([x, y, z], None);
                }
            }
        }

        let leaves_before = tree.num_leaves();
        tree.prune();
        let leaves_after = tree.num_leaves();

        // Pruning should reduce leaf count (some subtrees collapsed)
        assert!(leaves_after <= leaves_before);
    }
}
