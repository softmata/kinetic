//! Comprehensive GPU crate tests using CPU fallback (no GPU hardware needed).
//!
//! Tests cover CpuOptimizer, SignedDistanceField (CPU path), CpuCollisionChecker,
//! and batch FK data preparation via the public API.

use kinetic_collision::{RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic_gpu::{CpuCollisionChecker, CpuOptimizer, GpuConfig, SignedDistanceField};
use kinetic_robot::Robot;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a robot by name, panicking with a clear message on failure.
fn load_robot(name: &str) -> Robot {
    Robot::from_name(name).unwrap_or_else(|e| panic!("failed to load robot '{}': {}", name, e))
}

/// Return joint-space midpoint for a robot (safe starting configuration).
fn mid_joints(robot: &Robot) -> Vec<f64> {
    robot
        .joint_limits
        .iter()
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect()
}

/// Offset every joint by `delta` from the midpoint, clamped to limits.
fn offset_joints(robot: &Robot, delta: f64) -> Vec<f64> {
    robot
        .joint_limits
        .iter()
        .map(|l| {
            let mid = (l.lower + l.upper) / 2.0;
            (mid + delta).clamp(l.lower, l.upper)
        })
        .collect()
}

/// Build a small, fast config suitable for testing.
fn fast_config(num_seeds: u32, timesteps: u32, iterations: u32) -> GpuConfig {
    GpuConfig {
        num_seeds,
        timesteps,
        iterations,
        step_size: 0.01,
        seed_perturbation: 0.3,
        collision_weight: 100.0,
        smoothness_weight: 1.0,
        goal_weight: 50.0,
        ..Default::default()
    }
}

// ===========================================================================
// 1. CpuOptimizer tests
// ===========================================================================

mod cpu_optimizer {
    use super::*;

    #[test]
    fn plan_no_obstacles_smooth_linear_trajectory() {
        let robot = load_robot("franka_panda");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.2);
        let empty_obs = SpheresSoA::default();

        let config = fast_config(4, 16, 20);
        let opt = CpuOptimizer::new(config);
        let traj = opt.optimize(&robot, &empty_obs, &start, &goal).unwrap();

        assert_eq!(traj.len(), 16);
        assert_eq!(traj.dof, robot.dof);

        // Start should match exactly (endpoints are fixed).
        let first = traj.waypoint(0);
        for j in 0..robot.dof {
            assert!(
                (first.positions[j] - start[j]).abs() < 1e-4,
                "start mismatch at joint {}: expected {}, got {}",
                j,
                start[j],
                first.positions[j]
            );
        }

        // Without obstacles the trajectory should be roughly monotonic in each
        // joint (linear interpolation + optimization smoothness). Check that
        // successive differences are reasonably small, indicating smoothness.
        for j in 0..robot.dof {
            let mut max_jerk = 0.0_f64;
            for t in 1..traj.len().saturating_sub(2) {
                let q_prev = traj.waypoint(t - 1).positions[j];
                let q_curr = traj.waypoint(t).positions[j];
                let q_next = traj.waypoint(t + 1).positions[j];
                let accel_change = (q_next - 2.0 * q_curr + q_prev).abs();
                max_jerk = max_jerk.max(accel_change);
            }
            // Max acceleration-change should be small for a smooth trajectory.
            assert!(
                max_jerk < 0.5,
                "joint {} has high jerk {}, trajectory is not smooth",
                j,
                max_jerk
            );
        }
    }

    #[test]
    fn plan_with_obstacles_produces_valid_trajectory() {
        let robot = load_robot("ur5e");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.15);

        // Place obstacle spheres in the robot's workspace.
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.4, 0.0, 0.4, 0.05, 0);
        obstacles.push(-0.3, 0.2, 0.3, 0.08, 0);

        let config = fast_config(8, 16, 30);
        let opt = CpuOptimizer::new(config);
        let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

        assert_eq!(traj.len(), 16);
        assert_eq!(traj.dof, robot.dof);

        // Verify start and goal are respected.
        let first = traj.waypoint(0);
        let last = traj.waypoint(traj.len() - 1);
        for j in 0..robot.dof {
            assert!(
                (first.positions[j] - start[j]).abs() < 1e-4,
                "start mismatch at joint {}",
                j
            );
            // Goal tolerance is looser because optimization may not fully converge
            // with few iterations, but should still be close.
            assert!(
                (last.positions[j] - goal[j]).abs() < 0.3,
                "goal not reached at joint {}: expected {}, got {}",
                j,
                goal[j],
                last.positions[j]
            );
        }
    }

    #[test]
    fn different_num_seeds_produce_valid_results() {
        let robot = load_robot("franka_panda");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.1);
        let empty_obs = SpheresSoA::default();

        for &seeds in &[4u32, 16, 64] {
            let config = fast_config(seeds, 8, 10);
            let opt = CpuOptimizer::new(config);
            let traj = opt
                .optimize(&robot, &empty_obs, &start, &goal)
                .unwrap_or_else(|e| panic!("failed with {} seeds: {}", seeds, e));

            assert_eq!(traj.len(), 8, "trajectory length with {} seeds", seeds);
            assert_eq!(traj.dof, robot.dof, "DOF mismatch with {} seeds", seeds);

            // All waypoints should have valid (finite) joint values.
            for t in 0..traj.len() {
                let wp = traj.waypoint(t);
                for j in 0..robot.dof {
                    assert!(
                        wp.positions[j].is_finite(),
                        "non-finite value at seed_count={}, t={}, j={}",
                        seeds,
                        t,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn different_timesteps_produce_valid_results() {
        let robot = load_robot("ur5e");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.15);
        let empty_obs = SpheresSoA::default();

        for &ts in &[8u32, 16, 32] {
            let config = fast_config(4, ts, 10);
            let opt = CpuOptimizer::new(config);
            let traj = opt
                .optimize(&robot, &empty_obs, &start, &goal)
                .unwrap_or_else(|e| panic!("failed with {} timesteps: {}", ts, e));

            assert_eq!(
                traj.len(),
                ts as usize,
                "trajectory length with {} timesteps",
                ts
            );
            assert_eq!(traj.dof, robot.dof, "DOF mismatch with {} timesteps", ts);

            // Joint values should respect limits.
            for t in 0..traj.len() {
                let wp = traj.waypoint(t);
                for j in 0..robot.dof {
                    let lower = robot.joint_limits[j].lower;
                    let upper = robot.joint_limits[j].upper;
                    assert!(
                        wp.positions[j] >= lower - 1e-4 && wp.positions[j] <= upper + 1e-4,
                        "joint {} at t={} out of limits [{}, {}]: got {} (timesteps={})",
                        j,
                        t,
                        lower,
                        upper,
                        wp.positions[j],
                        ts
                    );
                }
            }
        }
    }

    #[test]
    fn warm_start_is_used_as_seed_zero() {
        let robot = load_robot("franka_panda");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.2);

        // Build a custom warm-start trajectory: three waypoints that deviate
        // from the linear interpolation.
        let mid_wp: Vec<f64> = start
            .iter()
            .zip(goal.iter())
            .enumerate()
            .map(|(j, (&s, &g))| {
                // Add a deliberate "bump" to mid-waypoint.
                let linear_mid = (s + g) / 2.0;
                (linear_mid + 0.1).clamp(robot.joint_limits[j].lower, robot.joint_limits[j].upper)
            })
            .collect();

        let warm_start = vec![start.clone(), mid_wp.clone(), goal.clone()];

        let config = GpuConfig {
            num_seeds: 1, // Only seed 0 (the warm-start seed).
            timesteps: 8,
            iterations: 0, // Zero iterations so the warm-start is returned as-is.
            warm_start: Some(warm_start.clone()),
            ..fast_config(1, 8, 0)
        };

        let opt = CpuOptimizer::new(config);
        let empty_obs = SpheresSoA::default();
        let traj = opt.optimize(&robot, &empty_obs, &start, &goal).unwrap();

        assert_eq!(traj.len(), 8);

        // With zero iterations the result is the warm-start resampled to 8 timesteps.
        // The midpoint of the trajectory should be close to our custom mid_wp.
        let mid_t = traj.len() / 2;
        let traj_mid = traj.waypoint(mid_t);
        let linear_mid: Vec<f64> = start
            .iter()
            .zip(goal.iter())
            .map(|(&s, &g)| (s + g) / 2.0)
            .collect();

        // The warm-start mid-waypoint has an intentional +0.1 bump; the trajectory
        // midpoint should be closer to the warm-start value than to linear interp.
        let mut closer_to_warm = 0;
        for j in 0..robot.dof {
            let dist_warm = (traj_mid.positions[j] - mid_wp[j]).abs();
            let dist_linear = (traj_mid.positions[j] - linear_mid[j]).abs();
            if dist_warm <= dist_linear {
                closer_to_warm += 1;
            }
        }
        // At least half the joints should be closer to the warm-start.
        assert!(
            closer_to_warm > robot.dof / 2,
            "warm-start not reflected: only {}/{} joints are closer to warm-start",
            closer_to_warm,
            robot.dof
        );
    }

    #[test]
    fn goal_reached_within_tolerance() {
        let robot = load_robot("xarm6");
        let start = mid_joints(&robot);
        let goal = offset_joints(&robot, 0.1);
        let empty_obs = SpheresSoA::default();

        // More iterations for better convergence.
        let config = fast_config(16, 16, 40);
        let opt = CpuOptimizer::new(config);
        let traj = opt.optimize(&robot, &empty_obs, &start, &goal).unwrap();

        let last = traj.waypoint(traj.len() - 1);
        let mut l2_dist = 0.0_f64;
        for j in 0..robot.dof {
            let diff = last.positions[j] - goal[j];
            l2_dist += diff * diff;
        }
        l2_dist = l2_dist.sqrt();

        assert!(
            l2_dist < 0.2,
            "goal not reached: L2 distance = {} (tolerance 0.2)",
            l2_dist
        );
    }
}

// ===========================================================================
// 2. SDF tests
// ===========================================================================

mod sdf {
    use super::*;

    #[test]
    fn single_sphere_center_is_negative() {
        let mut spheres = SpheresSoA::new();
        spheres.push(0.5, 0.5, 0.5, 0.2, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.02,
        )
        .unwrap();

        let at_center = sdf.query(0.5, 0.5, 0.5);
        assert!(
            at_center < 0.0,
            "query at sphere center should be negative (inside), got {}",
            at_center
        );

        // Value at center should be approximately -radius.
        assert!(
            (at_center - (-0.2)).abs() < 0.05,
            "expected ~-0.2 at center, got {}",
            at_center
        );
    }

    #[test]
    fn multiple_spheres_min_distance_is_correct() {
        let mut spheres = SpheresSoA::new();
        // Sphere A at (0.3, 0.5, 0.5) radius 0.1
        spheres.push(0.3, 0.5, 0.5, 0.1, 0);
        // Sphere B at (0.7, 0.5, 0.5) radius 0.15
        spheres.push(0.7, 0.5, 0.5, 0.15, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.02,
        )
        .unwrap();

        // Query at (0.5, 0.5, 0.5) - equidistant from both sphere centers.
        // Distance to A center: 0.2, signed dist to A surface: 0.2 - 0.1 = 0.1
        // Distance to B center: 0.2, signed dist to B surface: 0.2 - 0.15 = 0.05
        // Min signed distance should be ~0.05 (closer to sphere B surface).
        let val = sdf.query(0.5, 0.5, 0.5);
        assert!(
            (val - 0.05).abs() < 0.04,
            "expected ~0.05 at midpoint, got {}",
            val
        );

        // Inside sphere B.
        let inside_b = sdf.query(0.7, 0.5, 0.5);
        assert!(
            inside_b < 0.0,
            "should be inside sphere B, got {}",
            inside_b
        );

        // Inside sphere A.
        let inside_a = sdf.query(0.3, 0.5, 0.5);
        assert!(
            inside_a < 0.0,
            "should be inside sphere A, got {}",
            inside_a
        );
    }

    #[test]
    fn gradient_points_away_from_obstacle_center() {
        let mut spheres = SpheresSoA::new();
        spheres.push(0.5, 0.5, 0.5, 0.15, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.02,
        )
        .unwrap();

        // Query gradient at a point to the +x side of the sphere.
        let grad = sdf.gradient(0.8, 0.5, 0.5);
        assert!(
            grad[0] > 0.0,
            "gradient x-component should be positive (pointing away in +x), got {}",
            grad[0]
        );
        // It should be primarily in the x direction.
        assert!(
            grad[0].abs() > grad[1].abs() && grad[0].abs() > grad[2].abs(),
            "gradient should be primarily in x: {:?}",
            grad
        );

        // Query gradient at a point to the -y side.
        let grad_neg_y = sdf.gradient(0.5, 0.2, 0.5);
        assert!(
            grad_neg_y[1] < 0.0,
            "gradient y-component should be negative (pointing away in -y), got {}",
            grad_neg_y[1]
        );

        // Gradient at a point to the +z side.
        let grad_pos_z = sdf.gradient(0.5, 0.5, 0.85);
        assert!(
            grad_pos_z[2] > 0.0,
            "gradient z-component should be positive (pointing away in +z), got {}",
            grad_pos_z[2]
        );
    }

    #[test]
    fn trilinear_is_smoother_than_nearest_neighbor() {
        // Create an SDF with a sphere and sample along a line.
        // Trilinear interpolation should produce smoother (smaller max
        // step-to-step differences) output than nearest-neighbor.
        let mut spheres = SpheresSoA::new();
        spheres.push(0.5, 0.5, 0.5, 0.2, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.05, // Coarse resolution to make NN artifacts visible.
        )
        .unwrap();

        let num_samples = 50;
        let mut trilinear_vals = Vec::with_capacity(num_samples);
        let mut nearest_vals = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let x = 0.1 + 0.8 * (i as f64) / (num_samples as f64 - 1.0);
            trilinear_vals.push(sdf.query(x, 0.5, 0.5));
            nearest_vals.push(sdf.query_nearest(x, 0.5, 0.5));
        }

        // Compute max successive difference for each.
        let max_diff = |vals: &[f64]| -> f64 {
            vals.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0_f64, f64::max)
        };

        let tri_max = max_diff(&trilinear_vals);
        let nn_max = max_diff(&nearest_vals);

        // Trilinear should have smaller max jumps (smoother).
        assert!(
            tri_max <= nn_max + 1e-6,
            "trilinear max step ({}) should be <= nearest-neighbor max step ({})",
            tri_max,
            nn_max
        );
    }

    #[test]
    fn from_depth_cpu_produces_valid_sdf() {
        // Create a synthetic 8x8 depth image: uniform depth at 1.0m.
        let width = 8u32;
        let height = 8u32;
        let depth: Vec<f32> = vec![1.0; (width * height) as usize];

        // Identity camera pose (camera at origin, looking along +z).
        #[rustfmt::skip]
        let camera_pose: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let fx = 4.0;
        let fy = 4.0;
        let cx = 4.0;
        let cy = 4.0;

        let sdf = SignedDistanceField::from_depth_cpu(
            &depth,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            &camera_pose,
            0.05,  // point radius
            5.0,   // max depth
            0.1,   // resolution
            [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
        )
        .unwrap();

        // Should have voxels.
        assert!(sdf.num_voxels() > 0);

        // Query near (0, 0, 1) where the depth points project to -- should
        // be close to or inside the point spheres.
        let val = sdf.query(0.0, 0.0, 1.0);
        assert!(
            val < 0.2,
            "near projected depth points, distance should be small, got {}",
            val
        );

        // Query far from the depth cloud -- should be large.
        let far = sdf.query(0.0, 0.0, -1.5);
        assert!(
            far > 0.5,
            "far from depth cloud should have large distance, got {}",
            far
        );

        // Depth image size mismatch should error.
        let bad_depth: Vec<f32> = vec![1.0; 5]; // wrong size
        let err = SignedDistanceField::from_depth_cpu(
            &bad_depth,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            &camera_pose,
            0.05,
            5.0,
            0.1,
            [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
        );
        assert!(err.is_err(), "mismatched depth buffer should error");
    }

    #[test]
    fn empty_spheres_sdf_returns_large_positive() {
        let empty = SpheresSoA::default();
        let sdf = SignedDistanceField::from_spheres_cpu(
            &empty,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            0.1,
        )
        .unwrap();

        let val = sdf.query(0.0, 0.0, 0.0);
        assert!(
            val > 1e9,
            "empty SDF should return large positive value, got {}",
            val
        );
    }

    #[test]
    fn sdf_dimensions_and_resolution() {
        let empty = SpheresSoA::default();
        let sdf = SignedDistanceField::from_spheres_cpu(
            &empty,
            [-0.5, -0.5, 0.0],
            [0.5, 0.5, 1.0],
            0.1,
        )
        .unwrap();

        let (nx, ny, nz) = sdf.dimensions();
        assert_eq!(nx, 10);
        assert_eq!(ny, 10);
        assert_eq!(nz, 10);
        assert!((sdf.resolution() - 0.1).abs() < 1e-6);
        assert_eq!(sdf.min_corner(), [-0.5, -0.5, 0.0]);
        assert_eq!(sdf.num_voxels(), 1000);
    }

    #[test]
    fn sdf_query_outside_bounds_returns_large() {
        let mut spheres = SpheresSoA::new();
        spheres.push(0.5, 0.5, 0.5, 0.1, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.05,
        )
        .unwrap();

        assert!(sdf.query(-5.0, 0.5, 0.5) > 1e9);
        assert!(sdf.query(0.5, 10.0, 0.5) > 1e9);
        assert!(sdf.query(0.5, 0.5, -3.0) > 1e9);
    }
}

// ===========================================================================
// 3. CpuCollisionChecker tests
// ===========================================================================

mod cpu_collision_checker {
    use super::*;

    #[test]
    fn no_obstacles_nothing_in_collision() {
        let empty_obs = SpheresSoA::default();
        let checker = CpuCollisionChecker::from_spheres(
            &empty_obs,
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            0.1,
        )
        .unwrap();

        let robot = load_robot("franka_panda");
        let sphere_model = RobotSphereModel::from_robot_default(&robot);
        let config = mid_joints(&robot);

        let (colliding, _min_dist) = checker.check(&robot, &config, &sphere_model);
        // With no obstacles, nothing should collide (or distance should be very large).
        // Some robots may have zero collision geometry, in which case min_dist is inf
        // and colliding is false.
        assert!(
            !colliding,
            "no obstacles: robot should not be in collision"
        );
    }

    #[test]
    fn no_obstacles_multiple_configs_clear() {
        let empty_obs = SpheresSoA::default();
        let checker = CpuCollisionChecker::from_spheres(
            &empty_obs,
            [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
            0.1,
        )
        .unwrap();

        let robot = load_robot("ur5e");
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());

        // Check several configurations.
        for delta in &[0.0, 0.1, -0.1, 0.3] {
            let config = offset_joints(&robot, *delta);
            let (colliding, _) = checker.check(&robot, &config, &sphere_model);
            assert!(
                !colliding,
                "no obstacles: config with delta {} should be clear",
                delta
            );
        }
    }

    #[test]
    fn large_obstacle_causes_collision() {
        // Place a very large sphere at the origin that should engulf any robot
        // configuration near the base.
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.3, 1.5, 0); // huge sphere

        let checker = CpuCollisionChecker::from_spheres(
            &obstacles,
            [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
            0.05,
        )
        .unwrap();

        let robot = load_robot("ur5e");
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let config = mid_joints(&robot);

        let (colliding, min_dist) = checker.check(&robot, &config, &sphere_model);

        // If the robot has collision geometry, the huge obstacle should cause a collision.
        if sphere_model.total_spheres() > 0 {
            assert!(
                colliding,
                "large obstacle should cause collision (min_dist={})",
                min_dist
            );
            assert!(
                min_dist < 0.0,
                "min_distance should be negative for collision, got {}",
                min_dist
            );
        }
    }

    #[test]
    fn min_distance_decreases_as_obstacle_approaches() {
        let robot = load_robot("ur5e");
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let config = mid_joints(&robot);

        // Skip test if robot has no collision spheres.
        if sphere_model.total_spheres() == 0 {
            return;
        }

        let mut distances = Vec::new();
        // Move obstacle closer to origin in steps.
        for radius_offset in &[3.0_f64, 2.0, 1.0, 0.5] {
            let mut obs = SpheresSoA::new();
            obs.push(0.0, 0.0, *radius_offset, 0.1, 0);

            let checker = CpuCollisionChecker::from_spheres(
                &obs,
                [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0],
                0.05,
            )
            .unwrap();

            let (_colliding, min_dist) = checker.check(&robot, &config, &sphere_model);
            distances.push(min_dist);
        }

        // As the obstacle moves closer, min_distance should generally decrease
        // (or stay about the same if the obstacle path doesn't intersect).
        // Check the trend: last value should be <= first value.
        let first = distances[0];
        let last = distances[distances.len() - 1];
        assert!(
            last <= first + 0.01,
            "min_distance should decrease as obstacle approaches: first={}, last={}",
            first,
            last
        );
    }

    #[test]
    fn sdf_accessible_from_checker() {
        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.5, 0.1, 0);

        let checker =
            CpuCollisionChecker::from_spheres(&obs, [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], 0.05)
                .unwrap();

        // The underlying SDF should be queryable.
        let sdf = checker.sdf();
        let val = sdf.query(0.0, 0.0, 0.5);
        assert!(
            val < 0.0,
            "SDF at obstacle center should be negative, got {}",
            val
        );
    }
}

// ===========================================================================
// 4. Batch FK / Robot data tests (public API only)
// ===========================================================================

mod batch_fk_robot_data {
    use super::*;

    /// Test that Robot::from_name works for multiple robots and the DOF is correct.
    /// This validates that prepare_robot_data (internal) would work, since it
    /// depends on robot.joints and robot.dof being consistent.
    #[test]
    fn load_multiple_robots_correct_dof() {
        let cases = [
            ("ur5e", 6),
            ("franka_panda", 7),
            ("xarm6", 6),
        ];

        for (name, expected_dof) in &cases {
            let robot = load_robot(name);
            assert_eq!(
                robot.dof, *expected_dof,
                "robot '{}' should have DOF {}",
                name, expected_dof
            );
        }
    }

    /// Verify that joint type counts match expected DOF for several robots.
    /// Active (non-fixed) joints should equal robot.dof, and all joints
    /// should have valid types.
    #[test]
    fn joint_type_counts_match_expected_dof() {
        for name in &["ur5e", "franka_panda", "xarm6"] {
            let robot = load_robot(name);

            let active_count = robot
                .joints
                .iter()
                .filter(|j| {
                    matches!(
                        j.joint_type,
                        kinetic_robot::JointType::Revolute
                            | kinetic_robot::JointType::Prismatic
                            | kinetic_robot::JointType::Continuous
                    )
                })
                .count();

            assert_eq!(
                active_count, robot.dof,
                "robot '{}': active joint count ({}) != dof ({})",
                name, active_count, robot.dof
            );

            // Verify total joints = active + fixed.
            let fixed_count = robot
                .joints
                .iter()
                .filter(|j| matches!(j.joint_type, kinetic_robot::JointType::Fixed))
                .count();

            assert_eq!(
                active_count + fixed_count,
                robot.joints.len(),
                "robot '{}': joint type counts don't add up",
                name
            );
        }
    }

    /// Verify that RobotSphereModel can be built for each robot and that the
    /// data structure is consistent (link_ranges cover all spheres, etc.).
    #[test]
    fn sphere_model_consistency_multiple_robots() {
        for name in &["ur5e", "franka_panda", "xarm6"] {
            let robot = load_robot(name);
            let model = RobotSphereModel::from_robot_default(&robot);

            assert_eq!(
                model.num_links,
                robot.links.len(),
                "robot '{}': num_links mismatch",
                name
            );
            assert_eq!(
                model.link_ranges.len(),
                robot.links.len(),
                "robot '{}': link_ranges length mismatch",
                name
            );

            // Verify link_ranges are non-overlapping and cover [0, total_spheres).
            let mut covered = 0;
            for (i, &(start, end)) in model.link_ranges.iter().enumerate() {
                assert!(
                    start <= end,
                    "robot '{}': link {} has inverted range [{}, {})",
                    name,
                    i,
                    start,
                    end
                );
                covered += end - start;
            }
            assert_eq!(
                covered,
                model.total_spheres(),
                "robot '{}': link_ranges total ({}) != total_spheres ({})",
                name,
                covered,
                model.total_spheres()
            );
        }
    }

    /// Verify that the robot has valid joint limits (lower < upper) for all
    /// active joints -- a prerequisite for the GPU optimizer seed generation.
    #[test]
    fn joint_limits_valid_for_multiple_robots() {
        for name in &["ur5e", "franka_panda", "xarm6"] {
            let robot = load_robot(name);

            assert_eq!(
                robot.joint_limits.len(),
                robot.dof,
                "robot '{}': joint_limits length != dof",
                name
            );

            for (j, lim) in robot.joint_limits.iter().enumerate() {
                assert!(
                    lim.lower < lim.upper,
                    "robot '{}': joint {} has invalid limits [{}, {}]",
                    name,
                    j,
                    lim.lower,
                    lim.upper
                );
                assert!(
                    lim.lower.is_finite() && lim.upper.is_finite(),
                    "robot '{}': joint {} has non-finite limits",
                    name,
                    j
                );
            }
        }
    }

    /// CpuOptimizer can optimize for each of the test robots without panicking.
    #[test]
    fn cpu_optimizer_works_for_multiple_robots() {
        let empty_obs = SpheresSoA::default();
        let config = fast_config(4, 8, 5);

        for name in &["ur5e", "franka_panda", "xarm6"] {
            let robot = load_robot(name);
            let start = mid_joints(&robot);
            let goal = offset_joints(&robot, 0.1);

            let opt = CpuOptimizer::new(config.clone());
            let traj = opt
                .optimize(&robot, &empty_obs, &start, &goal)
                .unwrap_or_else(|e| panic!("optimize failed for '{}': {}", name, e));

            assert_eq!(traj.len(), 8, "robot '{}'", name);
            assert_eq!(traj.dof, robot.dof, "robot '{}'", name);
        }
    }
}
