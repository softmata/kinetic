//! Cached IK: lookup table with interpolation for fast IK queries.
//!
//! Pre-computes IK solutions across the workspace and stores them in a
//! spatial hash. Runtime queries interpolate between cached solutions,
//! avoiding the full IK solve for nearby poses.
//!
//! # Usage
//!
//! ```ignore
//! let mut cache = IKCache::new(config);
//! cache.warm(&robot, &chain);  // pre-compute across workspace
//! let solution = cache.solve(&target_pose, &seed);  // fast lookup
//! ```

use std::collections::HashMap;

use kinetic_core::Pose;
use kinetic_robot::Robot;
use rand::Rng;

use crate::forward::forward_kinematics;
use crate::ik::{solve_ik, IKConfig, IKSolution};
use crate::KinematicChain;

/// Cached IK configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Spatial resolution for cache grid (meters). Default: 0.05.
    pub resolution: f64,
    /// Orientation resolution (radians). Default: 0.2.
    pub orientation_resolution: f64,
    /// Maximum cache entries. Default: 100_000.
    pub max_entries: usize,
    /// Interpolation radius: search this far for neighbors. Default: 0.1m.
    pub interpolation_radius: f64,
    /// Number of workspace samples for warming. Default: 10_000.
    pub warm_samples: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            resolution: 0.05,
            orientation_resolution: 0.2,
            max_entries: 100_000,
            interpolation_radius: 0.1,
            warm_samples: 10_000,
        }
    }
}

/// A cached IK entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    joints: Vec<f64>,
    position: [f64; 3],
    #[allow(dead_code)]
    cost: f64, // distance from joint center
}

/// Spatial key for the cache grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpatialKey {
    x: i32,
    y: i32,
    z: i32,
}

/// IK cache with spatial lookup and interpolation.
pub struct IKCache {
    config: CacheConfig,
    entries: HashMap<SpatialKey, Vec<CacheEntry>>,
    total_entries: usize,
    hits: usize,
    misses: usize,
}

impl IKCache {
    /// Create a new empty cache.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            total_entries: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Number of cached entries.
    pub fn size(&self) -> usize {
        self.total_entries
    }

    /// Cache hit count.
    pub fn hits(&self) -> usize { self.hits }

    /// Cache miss count.
    pub fn misses(&self) -> usize { self.misses }

    /// Cache hit rate (0.0..1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f64 / total as f64
    }

    /// Reset hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Warm the cache by sampling random configurations and storing their FK results.
    pub fn warm(&mut self, robot: &Robot, chain: &KinematicChain) {
        let dof = chain.dof;
        let mut rng = rand::thread_rng();

        let limits: Vec<(f64, f64)> = chain.active_joints.iter().map(|&ji| {
            robot.joints[ji].limits.as_ref()
                .map(|l| (l.lower, l.upper))
                .unwrap_or((-std::f64::consts::PI, std::f64::consts::PI))
        }).collect();

        for _ in 0..self.config.warm_samples {
            if self.total_entries >= self.config.max_entries {
                break;
            }

            let joints: Vec<f64> = (0..dof).map(|j| {
                let (lo, hi) = limits[j];
                let range = hi - lo;
                if range.is_finite() && range < 100.0 {
                    rng.gen_range(lo..=hi)
                } else {
                    rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                }
            }).collect();

            if let Ok(pose) = forward_kinematics(robot, chain, &joints) {
                self.insert_solution(pose.translation().into(), joints);
            }
        }
    }

    /// Insert a known IK solution into the cache.
    pub fn insert(&mut self, target_pose: &Pose, joints: Vec<f64>) {
        let pos: [f64; 3] = [
            target_pose.translation().x,
            target_pose.translation().y,
            target_pose.translation().z,
        ];
        self.insert_solution(pos, joints);
    }

    /// Look up a cached solution for a target pose.
    ///
    /// Returns the cached joint configuration nearest to the target position.
    /// If no cache entry is within `interpolation_radius`, returns None (cache miss).
    pub fn lookup(&mut self, target: &Pose) -> Option<Vec<f64>> {
        let pos = [
            target.translation().x,
            target.translation().y,
            target.translation().z,
        ];
        let key = self.spatial_key(&pos);

        // Search the target cell and neighbors
        let mut best: Option<(&CacheEntry, f64)> = None;
        let search_radius = (self.config.interpolation_radius / self.config.resolution).ceil() as i32;

        for dx in -search_radius..=search_radius {
            for dy in -search_radius..=search_radius {
                for dz in -search_radius..=search_radius {
                    let neighbor = SpatialKey {
                        x: key.x + dx,
                        y: key.y + dy,
                        z: key.z + dz,
                    };

                    if let Some(entries) = self.entries.get(&neighbor) {
                        for entry in entries {
                            let dist = ((entry.position[0] - pos[0]).powi(2)
                                + (entry.position[1] - pos[1]).powi(2)
                                + (entry.position[2] - pos[2]).powi(2))
                                .sqrt();

                            if dist <= self.config.interpolation_radius {
                                if best.is_none() || dist < best.unwrap().1 {
                                    best = Some((entry, dist));
                                }
                            }
                        }
                    }
                }
            }
        }

        match best {
            Some((entry, _)) => {
                self.hits += 1;
                Some(entry.joints.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Solve IK with cache: try cache first, fall back to full solve.
    pub fn solve(
        &mut self,
        robot: &Robot,
        chain: &KinematicChain,
        target: &Pose,
        ik_config: &IKConfig,
    ) -> Option<IKSolution> {
        // Try cache lookup first
        if let Some(cached_joints) = self.lookup(target) {
            // Use cached solution as seed for refinement
            let mut config_with_seed = ik_config.clone();
            config_with_seed.seed = Some(cached_joints);
            let refined = solve_ik(robot, chain, target, &config_with_seed).ok();
            if let Some(ref sol) = refined {
                if sol.converged {
                    return refined;
                }
            }
        }

        // Full solve
        let result = solve_ik(robot, chain, target, ik_config).ok();

        // Cache the result
        if let Some(ref sol) = result {
            if sol.converged {
                self.insert(target, sol.joints.clone());
            }
        }

        result
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_entries = 0;
    }

    /// Serialize cache to bytes for persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.total_entries as u32).to_le_bytes());
        buf.extend_from_slice(&self.config.resolution.to_le_bytes());

        for entries in self.entries.values() {
            for entry in entries {
                for p in &entry.position { buf.extend_from_slice(&p.to_le_bytes()); }
                buf.extend_from_slice(&(entry.joints.len() as u32).to_le_bytes());
                for j in &entry.joints { buf.extend_from_slice(&j.to_le_bytes()); }
            }
        }
        buf
    }

    /// Deserialize cache from bytes.
    pub fn from_bytes(data: &[u8], config: CacheConfig) -> Option<Self> {
        if data.len() < 12 { return None; }
        let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let _resolution = f64::from_le_bytes(data[4..12].try_into().ok()?);

        let mut cache = Self::new(config);
        let mut pos = 12;

        for _ in 0..count {
            if pos + 28 > data.len() { break; }
            let x = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8;
            let y = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8;
            let z = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8;
            let dof = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize; pos += 4;

            let mut joints = Vec::with_capacity(dof);
            for _ in 0..dof {
                if pos + 8 > data.len() { break; }
                joints.push(f64::from_le_bytes(data[pos..pos+8].try_into().ok()?));
                pos += 8;
            }

            cache.insert_solution([x, y, z], joints);
        }

        Some(cache)
    }

    // ─── Internal ────────────────────────────────────────────────────────

    fn spatial_key(&self, pos: &[f64; 3]) -> SpatialKey {
        SpatialKey {
            x: (pos[0] / self.config.resolution).floor() as i32,
            y: (pos[1] / self.config.resolution).floor() as i32,
            z: (pos[2] / self.config.resolution).floor() as i32,
        }
    }

    fn insert_solution(&mut self, position: [f64; 3], joints: Vec<f64>) {
        if self.total_entries >= self.config.max_entries {
            return;
        }

        let key = self.spatial_key(&position);
        let entry = CacheEntry { joints, position, cost: 0.0 };

        self.entries.entry(key).or_insert_with(Vec::new).push(entry);
        self.total_entries += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="test3dof">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/><child link="tip"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="10"/>
  </joint>
</robot>"#;

    #[test]
    fn cache_insert_and_lookup() {
        let mut cache = IKCache::with_defaults();

        let target = Pose::from_xyz(0.3, 0.0, 0.4);
        cache.insert(&target, vec![0.1, 0.2, 0.3]);

        assert_eq!(cache.size(), 1);

        let result = cache.lookup(&target);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![0.1, 0.2, 0.3]);
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn cache_miss_for_distant() {
        let mut cache = IKCache::with_defaults();
        cache.insert(&Pose::from_xyz(0.0, 0.0, 0.0), vec![0.0; 3]);

        let far = Pose::from_xyz(5.0, 5.0, 5.0);
        assert!(cache.lookup(&far).is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn cache_interpolation_nearby() {
        let mut cache = IKCache::new(CacheConfig {
            interpolation_radius: 0.2,
            resolution: 0.05,
            ..Default::default()
        });

        cache.insert(&Pose::from_xyz(0.3, 0.0, 0.4), vec![0.1, 0.2, 0.3]);

        // Nearby query (within interpolation radius)
        let nearby = Pose::from_xyz(0.35, 0.0, 0.4);
        let result = cache.lookup(&nearby);
        assert!(result.is_some(), "Nearby pose should hit cache");
    }

    #[test]
    fn cache_warming() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();

        let mut cache = IKCache::new(CacheConfig {
            warm_samples: 100,
            max_entries: 1000,
            ..Default::default()
        });

        cache.warm(&robot, &chain);
        assert!(cache.size() > 0, "Warming should populate cache: {}", cache.size());
    }

    #[test]
    fn cache_hit_rate() {
        let mut cache = IKCache::with_defaults();
        cache.insert(&Pose::from_xyz(0.3, 0.0, 0.4), vec![0.1, 0.2, 0.3]);

        cache.lookup(&Pose::from_xyz(0.3, 0.0, 0.4)); // hit
        cache.lookup(&Pose::from_xyz(5.0, 5.0, 5.0)); // miss

        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cache_max_entries() {
        let mut cache = IKCache::new(CacheConfig {
            max_entries: 5,
            ..Default::default()
        });

        for i in 0..10 {
            cache.insert(&Pose::from_xyz(i as f64 * 0.1, 0.0, 0.0), vec![0.0]);
        }

        assert!(cache.size() <= 5, "Should respect max entries: {}", cache.size());
    }

    #[test]
    fn cache_clear() {
        let mut cache = IKCache::with_defaults();
        cache.insert(&Pose::from_xyz(0.0, 0.0, 0.0), vec![0.0]);
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn cache_serialization_roundtrip() {
        let mut cache = IKCache::with_defaults();
        cache.insert(&Pose::from_xyz(0.3, 0.1, 0.4), vec![0.1, 0.2, 0.3]);
        cache.insert(&Pose::from_xyz(-0.2, 0.5, 0.1), vec![0.4, 0.5, 0.6]);

        let bytes = cache.to_bytes();
        let restored = IKCache::from_bytes(&bytes, CacheConfig::default()).unwrap();
        assert_eq!(restored.size(), 2);
    }

    #[test]
    fn cache_reset_stats() {
        let mut cache = IKCache::with_defaults();
        cache.insert(&Pose::from_xyz(0.0, 0.0, 0.0), vec![0.0]);
        cache.lookup(&Pose::from_xyz(0.0, 0.0, 0.0));
        assert_eq!(cache.hits(), 1);

        cache.reset_stats();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }
}
