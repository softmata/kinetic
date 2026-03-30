//! CPU fallback for GPU trajectory optimizer.
//!
//! Produces functionally equivalent results to the GPU optimizer using
//! the same seed-based parallel optimization approach, but executed on
//! CPU using existing kinetic primitives. Used when no GPU is available
//! or for equivalence testing.

use kinetic_collision::{RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic_core::Trajectory;
use kinetic_robot::Robot;
use rand::Rng;

use crate::{GpuConfig, GpuError, Result};

/// CPU-based trajectory optimizer — drop-in replacement for `GpuOptimizer`.
///
/// Uses the same seed-based approach as the GPU version but runs entirely
/// on CPU. Slower but works everywhere and produces comparable results.
pub struct CpuOptimizer {
    config: GpuConfig,
}

impl CpuOptimizer {
    /// Create a new CPU optimizer with the given config.
    pub fn new(config: GpuConfig) -> Self {
        Self { config }
    }

    /// Optimize a trajectory from start to goal, avoiding obstacles.
    ///
    /// Same signature as `GpuOptimizer::optimize()`.
    pub fn optimize(
        &self,
        robot: &Robot,
        obstacle_spheres: &SpheresSoA,
        start: &[f64],
        goal: &[f64],
    ) -> Result<Trajectory> {
        let dof = robot.dof;
        let num_seeds = self.config.num_seeds as usize;
        let timesteps = self.config.timesteps as usize;

        if start.len() != dof || goal.len() != dof {
            return Err(GpuError::InvalidConfig(format!(
                "start/goal length ({}/{}) doesn't match robot DOF ({})",
                start.len(),
                goal.len(),
                dof
            )));
        }

        // Generate collision spheres from robot
        let sphere_model = RobotSphereModel::from_robot(robot, &SphereGenConfig::coarse());

        // Generate trajectory seeds (same algorithm as GPU path)
        let mut trajectories = generate_seeds_f64(
            start,
            goal,
            robot,
            num_seeds,
            timesteps,
            self.config.seed_perturbation,
            self.config.warm_start.as_ref(),
        );

        // Build obstacle sphere data for distance queries
        let obs_count = obstacle_spheres.x.len();

        // Iterative optimization
        for _ in 0..self.config.iterations {
            for seed in 0..num_seeds {
                // Compute cost and apply gradient descent
                let base = seed * timesteps * dof;

                // Numerical gradient via finite differences
                let eps = 1e-3;
                let mut gradients = vec![0.0f64; timesteps * dof];

                let current_cost = self.compute_seed_cost(
                    &trajectories[base..base + timesteps * dof],
                    robot,
                    &sphere_model,
                    obstacle_spheres,
                    obs_count,
                    goal,
                    dof,
                    timesteps,
                );

                // Only compute gradients for internal waypoints (endpoints are fixed)
                for t in 1..timesteps - 1 {
                    for j in 0..dof {
                        let idx = base + t * dof + j;
                        let original = trajectories[idx];

                        trajectories[idx] = original + eps;
                        let cost_plus = self.compute_seed_cost(
                            &trajectories[base..base + timesteps * dof],
                            robot,
                            &sphere_model,
                            obstacle_spheres,
                            obs_count,
                            goal,
                            dof,
                            timesteps,
                        );

                        trajectories[idx] = original;
                        gradients[t * dof + j] = (cost_plus - current_cost) / eps;
                    }
                }

                // Apply gradient descent step (internal waypoints only)
                let step = self.config.step_size as f64;
                for t in 1..timesteps - 1 {
                    for j in 0..dof {
                        let idx = base + t * dof + j;
                        trajectories[idx] -= step * gradients[t * dof + j];
                        // Clamp to joint limits
                        trajectories[idx] = trajectories[idx]
                            .clamp(robot.joint_limits[j].lower, robot.joint_limits[j].upper);
                    }
                }
            }
        }

        // Find best seed (lowest cost)
        let mut best_seed = 0;
        let mut best_cost = f64::INFINITY;
        for seed in 0..num_seeds {
            let base = seed * timesteps * dof;
            let cost = self.compute_seed_cost(
                &trajectories[base..base + timesteps * dof],
                robot,
                &sphere_model,
                obstacle_spheres,
                obs_count,
                goal,
                dof,
                timesteps,
            );
            if cost < best_cost {
                best_cost = cost;
                best_seed = seed;
            }
        }

        // Extract best trajectory
        let mut traj = Trajectory::with_dof(dof);
        let base = best_seed * timesteps * dof;
        for t in 0..timesteps {
            let wp: Vec<f64> = (0..dof).map(|j| trajectories[base + t * dof + j]).collect();
            traj.push_waypoint(&wp);
        }

        Ok(traj)
    }

    /// Compute total cost for a single seed trajectory.
    #[allow(clippy::too_many_arguments)]
    fn compute_seed_cost(
        &self,
        seed_data: &[f64],
        robot: &Robot,
        sphere_model: &RobotSphereModel,
        obstacle_spheres: &SpheresSoA,
        obs_count: usize,
        goal: &[f64],
        dof: usize,
        timesteps: usize,
    ) -> f64 {
        let mut collision_cost = 0.0;
        let mut smoothness_cost = 0.0;
        let mut goal_cost = 0.0;

        // Collision cost: check robot spheres against obstacle spheres at each timestep
        if obs_count > 0 {
            for t in 0..timesteps {
                let joints = &seed_data[t * dof..(t + 1) * dof];

                // Compute FK to get world-frame sphere positions
                let world_spheres = compute_world_spheres(robot, sphere_model, joints);

                // Check each robot sphere against each obstacle sphere
                for si in 0..world_spheres.len() / 4 {
                    let sx = world_spheres[si * 4];
                    let sy = world_spheres[si * 4 + 1];
                    let sz = world_spheres[si * 4 + 2];
                    let sr = world_spheres[si * 4 + 3];

                    for oi in 0..obs_count {
                        let ox = obstacle_spheres.x[oi];
                        let oy = obstacle_spheres.y[oi];
                        let oz = obstacle_spheres.z[oi];
                        let or = obstacle_spheres.radius[oi];

                        let dx = sx - ox;
                        let dy = sy - oy;
                        let dz = sz - oz;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt() - sr - or;

                        // Hinge loss: penalize penetration
                        if dist < 0.0 {
                            collision_cost += (-dist) as f64;
                        }
                    }
                }
            }
        }

        // Smoothness cost: jerk (3rd derivative) finite differences
        if timesteps >= 4 {
            for t in 1..timesteps - 2 {
                for j in 0..dof {
                    let q0 = seed_data[(t - 1) * dof + j];
                    let q1 = seed_data[t * dof + j];
                    let q2 = seed_data[(t + 1) * dof + j];
                    let q3 = seed_data[(t + 2) * dof + j];
                    let jerk = q3 - 3.0 * q2 + 3.0 * q1 - q0;
                    smoothness_cost += jerk * jerk;
                }
            }
        }

        // Goal cost: L2 distance from last waypoint to goal
        let last_t = timesteps - 1;
        for j in 0..dof {
            let diff = seed_data[last_t * dof + j] - goal[j];
            goal_cost += diff * diff;
        }

        self.config.collision_weight as f64 * collision_cost
            + self.config.smoothness_weight as f64 * smoothness_cost
            + self.config.goal_weight as f64 * goal_cost
    }

    /// Get the config.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
}

/// Compute world-frame sphere positions for a given joint configuration.
/// Returns flat vec: [x, y, z, radius, x, y, z, radius, ...]
fn compute_world_spheres(
    robot: &Robot,
    sphere_model: &RobotSphereModel,
    joints: &[f64],
) -> Vec<f64> {
    use nalgebra::{Isometry3, UnitQuaternion, Vector3};

    // Compute accumulated transforms for each joint
    let num_joints = robot.joints.len();
    let mut transforms = Vec::with_capacity(num_joints);

    for (ji, joint) in robot.joints.iter().enumerate() {
        let parent_transform = if joint.parent_link == 0 {
            Isometry3::identity()
        } else {
            // Find the joint that has parent_link as its child
            robot
                .joints
                .iter()
                .position(|j| j.child_link == joint.parent_link)
                .and_then(|pi| transforms.get(pi).copied())
                .unwrap_or(Isometry3::identity())
        };

        let joint_val = if ji < joints.len() { joints[ji] } else { 0.0 };

        let local_transform = match joint.joint_type {
            kinetic_robot::JointType::Revolute | kinetic_robot::JointType::Continuous => {
                let axis = Vector3::new(joint.axis.x, joint.axis.y, joint.axis.z);
                let rot = UnitQuaternion::new(axis * joint_val);
                joint.origin.0 * Isometry3::from_parts(Default::default(), rot)
            }
            kinetic_robot::JointType::Prismatic => {
                let axis = Vector3::new(joint.axis.x, joint.axis.y, joint.axis.z);
                let trans = axis * joint_val;
                joint.origin.0
                    * Isometry3::from_parts(
                        nalgebra::Translation3::from(trans),
                        UnitQuaternion::identity(),
                    )
            }
            kinetic_robot::JointType::Fixed => joint.origin.0,
        };

        transforms.push(parent_transform * local_transform);
    }

    // Transform local spheres to world frame
    let mut result = Vec::new();
    for link_idx in 0..sphere_model.num_links {
        let (start, end) = sphere_model.link_ranges[link_idx];
        let joint_idx = robot
            .joints
            .iter()
            .position(|j| j.child_link == link_idx)
            .unwrap_or(0);

        let tf = transforms
            .get(joint_idx)
            .copied()
            .unwrap_or(Isometry3::identity());

        for i in start..end {
            let local_pt = nalgebra::Point3::new(
                sphere_model.local.x[i],
                sphere_model.local.y[i],
                sphere_model.local.z[i],
            );
            let world_pt = tf * local_pt;
            result.push(world_pt.x);
            result.push(world_pt.y);
            result.push(world_pt.z);
            result.push(sphere_model.local.radius[i]);
        }
    }

    result
}

/// Generate trajectory seeds in f64 (CPU path uses f64 throughout).
fn generate_seeds_f64(
    start: &[f64],
    goal: &[f64],
    robot: &Robot,
    num_seeds: usize,
    timesteps: usize,
    perturbation: f32,
    warm_start: Option<&Vec<Vec<f64>>>,
) -> Vec<f64> {
    let dof = start.len();
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0f64; num_seeds * timesteps * dof];

    for seed in 0..num_seeds {
        for t in 0..timesteps {
            let alpha = t as f64 / (timesteps - 1).max(1) as f64;
            for j in 0..dof {
                let base_val = if seed == 0 {
                    if let Some(ws) = warm_start {
                        let ws_alpha = alpha * (ws.len() - 1).max(1) as f64;
                        let ws_idx = ws_alpha.floor() as usize;
                        let ws_frac = ws_alpha - ws_idx as f64;
                        if ws_idx + 1 < ws.len() {
                            ws[ws_idx][j] * (1.0 - ws_frac) + ws[ws_idx + 1][j] * ws_frac
                        } else {
                            ws[ws.len() - 1][j]
                        }
                    } else {
                        start[j] + alpha * (goal[j] - start[j])
                    }
                } else {
                    start[j] + alpha * (goal[j] - start[j])
                };

                let noise = if seed == 0 || t == 0 || t == timesteps - 1 {
                    0.0
                } else {
                    rng.gen_range(-(perturbation as f64)..(perturbation as f64))
                };
                let val = (base_val + noise)
                    .clamp(robot.joint_limits[j].lower, robot.joint_limits[j].upper);
                data[seed * timesteps * dof + t * dof + j] = val;
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_robot() -> Robot {
        Robot::from_name("franka_panda").unwrap()
    }

    #[test]
    fn cpu_optimizer_creates() {
        let opt = CpuOptimizer::new(GpuConfig::default());
        assert_eq!(opt.config().num_seeds, 128);
    }

    #[test]
    fn cpu_optimizer_produces_valid_trajectory() {
        let robot = test_robot();
        let start: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        let goal: Vec<f64> = start.iter().map(|s| s + 0.2).collect();

        let config = GpuConfig {
            num_seeds: 4,
            timesteps: 8,
            iterations: 10,
            ..Default::default()
        };
        let opt = CpuOptimizer::new(config);
        let empty_obs = SpheresSoA::default();
        let traj = opt.optimize(&robot, &empty_obs, &start, &goal).unwrap();

        assert_eq!(traj.len(), 8);
        assert_eq!(traj.dof, robot.dof);

        // Start and goal should match
        let first = traj.waypoint(0);
        let last = traj.waypoint(traj.len() - 1);
        for j in 0..robot.dof {
            assert!(
                (first.positions[j] - start[j]).abs() < 1e-4,
                "Start mismatch joint {}",
                j
            );
            assert!(
                (last.positions[j] - goal[j]).abs() < 0.1,
                "Goal not reached joint {}",
                j
            );
        }
    }

    #[test]
    fn cpu_optimizer_wrong_dof_errors() {
        let robot = test_robot();
        let opt = CpuOptimizer::new(GpuConfig::default());
        let empty_obs = SpheresSoA::default();
        let result = opt.optimize(&robot, &empty_obs, &[0.0, 0.0], &[1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_optimizer_with_obstacles() {
        let robot = test_robot();
        let start: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        let goal: Vec<f64> = start.iter().map(|s| s + 0.15).collect();

        // Place obstacle sphere near the robot
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.4, 0.0, 0.4, 0.05, 0);

        let config = GpuConfig {
            num_seeds: 8,
            timesteps: 8,
            iterations: 20,
            ..Default::default()
        };
        let opt = CpuOptimizer::new(config);
        let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

        // Should produce a valid trajectory
        assert_eq!(traj.len(), 8);
        assert_eq!(traj.dof, robot.dof);
    }
}
