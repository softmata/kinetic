//! Workspace analysis: reachability maps and dexterity maps.
//!
//! Pre-compute which end-effector positions the robot can reach and
//! how dexterous it is at each point. Useful for robot placement
//! optimization, grasp feasibility checking, and task planning.

use kinetic_core::{KineticError, Result};
use kinetic_robot::Robot;
use rand::Rng;

use crate::{fk, manipulability, KinematicChain};

/// A 3D voxel grid storing reachability and dexterity information.
pub struct ReachabilityMap {
    /// Voxel grid: true if at least one IK solution exists.
    pub reachable: Vec<bool>,
    /// Manipulability index at each reachable voxel (0.0 if unreachable).
    pub dexterity: Vec<f64>,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid origin (minimum corner).
    pub min: [f64; 3],
    /// Voxel resolution in meters.
    pub resolution: f64,
    /// Number of reachable voxels.
    pub num_reachable: usize,
    /// Total samples evaluated.
    pub total_samples: usize,
}

/// Configuration for reachability map computation.
pub struct ReachabilityConfig {
    /// Voxel resolution in meters (default: 0.05 = 5cm).
    pub resolution: f64,
    /// Workspace bounds `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub bounds: [f64; 6],
    /// Number of random joint configs to sample (default: 50000).
    pub num_samples: usize,
}

impl Default for ReachabilityConfig {
    fn default() -> Self {
        Self {
            resolution: 0.05,
            bounds: [-1.5, -1.5, -0.5, 1.5, 1.5, 2.0],
            num_samples: 50000,
        }
    }
}

impl ReachabilityMap {
    /// Compute a reachability map by sampling random joint configurations.
    ///
    /// For each sample: compute FK → get EE position → mark voxel as reachable
    /// and record the manipulability. Uses the maximum manipulability seen
    /// at each voxel.
    pub fn compute(
        robot: &Robot,
        chain: &KinematicChain,
        config: &ReachabilityConfig,
    ) -> Result<Self> {
        let nx = ((config.bounds[3] - config.bounds[0]) / config.resolution).ceil() as usize;
        let ny = ((config.bounds[4] - config.bounds[1]) / config.resolution).ceil() as usize;
        let nz = ((config.bounds[5] - config.bounds[2]) / config.resolution).ceil() as usize;
        let total_voxels = nx * ny * nz;

        if total_voxels == 0 {
            return Err(KineticError::PlanningFailed(
                "Reachability grid has zero voxels".into(),
            ));
        }

        let mut reachable = vec![false; total_voxels];
        let mut dexterity = vec![0.0f64; total_voxels];
        let mut rng = rand::thread_rng();
        let mut num_reachable = 0usize;

        for _ in 0..config.num_samples {
            // Random joint configuration within limits
            let joints: Vec<f64> = chain
                .active_joints
                .iter()
                .map(|&ji| {
                    let joint = &robot.joints[ji];
                    if let Some(limits) = &joint.limits {
                        let range = limits.upper - limits.lower;
                        if range.is_finite() && range < 100.0 {
                            rng.gen_range(limits.lower..=limits.upper)
                        } else {
                            rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                        }
                    } else {
                        rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                    }
                })
                .collect();

            // Compute FK
            let pose = match fk(robot, chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let pos = pose.translation();

            // Map to voxel
            let ix = ((pos.x - config.bounds[0]) / config.resolution) as i64;
            let iy = ((pos.y - config.bounds[1]) / config.resolution) as i64;
            let iz = ((pos.z - config.bounds[2]) / config.resolution) as i64;

            if ix < 0 || iy < 0 || iz < 0 || ix >= nx as i64 || iy >= ny as i64 || iz >= nz as i64 {
                continue;
            }

            let idx = iz as usize * nx * ny + iy as usize * nx + ix as usize;

            // Compute manipulability
            let manip = manipulability(robot, chain, &joints).unwrap_or(0.0);

            if !reachable[idx] {
                reachable[idx] = true;
                num_reachable += 1;
            }
            // Keep maximum manipulability
            if manip > dexterity[idx] {
                dexterity[idx] = manip;
            }
        }

        Ok(Self {
            reachable,
            dexterity,
            nx,
            ny,
            nz,
            min: [config.bounds[0], config.bounds[1], config.bounds[2]],
            resolution: config.resolution,
            num_reachable,
            total_samples: config.num_samples,
        })
    }

    /// Check if a world position is reachable.
    pub fn is_reachable(&self, x: f64, y: f64, z: f64) -> bool {
        if let Some(idx) = self.voxel_index(x, y, z) {
            self.reachable[idx]
        } else {
            false
        }
    }

    /// Get dexterity (manipulability) at a world position.
    pub fn dexterity_at(&self, x: f64, y: f64, z: f64) -> f64 {
        if let Some(idx) = self.voxel_index(x, y, z) {
            self.dexterity[idx]
        } else {
            0.0
        }
    }

    /// Reachability ratio (fraction of workspace that is reachable).
    pub fn reachability_ratio(&self) -> f64 {
        let total = self.nx * self.ny * self.nz;
        if total == 0 {
            0.0
        } else {
            self.num_reachable as f64 / total as f64
        }
    }

    /// Approximate workspace volume in cubic meters (sum of reachable voxels).
    pub fn workspace_volume(&self) -> f64 {
        let voxel_volume = self.resolution * self.resolution * self.resolution;
        self.num_reachable as f64 * voxel_volume
    }

    /// Maximum dexterity value in the map.
    pub fn max_dexterity(&self) -> f64 {
        self.dexterity.iter().copied().fold(0.0f64, f64::max)
    }

    fn voxel_index(&self, x: f64, y: f64, z: f64) -> Option<usize> {
        let ix = ((x - self.min[0]) / self.resolution) as i64;
        let iy = ((y - self.min[1]) / self.resolution) as i64;
        let iz = ((z - self.min[2]) / self.resolution) as i64;

        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.nx as i64
            || iy >= self.ny as i64
            || iz >= self.nz as i64
        {
            None
        } else {
            Some(iz as usize * self.nx * self.ny + iy as usize * self.nx + ix as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reachability_map_basic() {
        let robot = Robot::from_name("ur5e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        let config = ReachabilityConfig {
            resolution: 0.1,
            bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
            num_samples: 5000,
        };

        let map = ReachabilityMap::compute(&robot, &chain, &config).unwrap();

        assert!(map.num_reachable > 0, "UR5e should have reachable voxels");
        assert!(
            map.reachability_ratio() > 0.01,
            "UR5e should reach >1% of workspace"
        );
        assert!(map.workspace_volume() > 0.0);
        assert!(map.max_dexterity() > 0.0);
        assert_eq!(map.total_samples, 5000);
    }

    #[test]
    fn reachability_query() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        let config = ReachabilityConfig {
            resolution: 0.1,
            bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
            num_samples: 10000,
        };

        let map = ReachabilityMap::compute(&robot, &chain, &config).unwrap();

        // Far from robot should be unreachable
        assert!(
            !map.is_reachable(5.0, 5.0, 5.0),
            "5m away should be unreachable"
        );

        // Dexterity at unreachable point should be 0
        assert_eq!(map.dexterity_at(5.0, 5.0, 5.0), 0.0);
    }

    #[test]
    fn reachability_default_config() {
        let config = ReachabilityConfig::default();
        assert_eq!(config.resolution, 0.05);
        assert_eq!(config.num_samples, 50000);
    }
}
