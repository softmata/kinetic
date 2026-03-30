//! Joint limit boundary and trajectory edge-case tests.
//!
//! Tests exact boundary conditions: at-limit, epsilon-over-limit, single-waypoint
//! trajectories, start==goal planning, collision margin extremes, and degenerate paths.

use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig, IKSolver, KinematicChain};
use kinetic::prelude::*;
use kinetic::trajectory::{cubic_spline_time, jerk_limited, totp, trapezoidal};
use std::time::Duration;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── Joint limit boundary tests ─────────────────────────────────────────────

#[test]
fn fk_at_exact_upper_limits() {
    let (robot, chain) = ur5e_robot_and_chain();
    let upper: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| robot.joints[ji].limits.as_ref().map_or(3.14, |l| l.upper))
        .collect();
    let result = forward_kinematics(&robot, &chain, &upper);
    assert!(result.is_ok(), "FK at exact upper limits should succeed");
    let t = result.unwrap().translation();
    assert!(t[0].is_finite() && t[1].is_finite() && t[2].is_finite());
}

#[test]
fn fk_at_exact_lower_limits() {
    let (robot, chain) = ur5e_robot_and_chain();
    let lower: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| robot.joints[ji].limits.as_ref().map_or(-3.14, |l| l.lower))
        .collect();
    let result = forward_kinematics(&robot, &chain, &lower);
    assert!(result.is_ok(), "FK at exact lower limits should succeed");
    let t = result.unwrap().translation();
    assert!(t[0].is_finite() && t[1].is_finite() && t[2].is_finite());
}

#[test]
fn fk_beyond_upper_limits() {
    let (robot, chain) = ur5e_robot_and_chain();
    // FK doesn't enforce limits — it computes the transform regardless
    let beyond: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            robot.joints[ji]
                .limits
                .as_ref()
                .map_or(10.0, |l| l.upper + 0.1)
        })
        .collect();
    let result = forward_kinematics(&robot, &chain, &beyond);
    // FK should still work — it's just math, not limit checking
    assert!(result.is_ok(), "FK beyond limits should still compute");
    let t = result.unwrap().translation();
    assert!(t[0].is_finite() && t[1].is_finite() && t[2].is_finite());
}

#[test]
fn ik_rejects_solution_beyond_limits() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Use a target that's reachable but the DLS should enforce limits
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        check_limits: true,
        max_iterations: 200,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    if let Ok(sol) = result {
        // All joints must be within limits
        for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                assert!(
                    sol.joints[i] >= limits.lower - 1e-6,
                    "Joint {} below lower limit: {} < {}",
                    i,
                    sol.joints[i],
                    limits.lower
                );
                assert!(
                    sol.joints[i] <= limits.upper + 1e-6,
                    "Joint {} above upper limit: {} > {}",
                    i,
                    sol.joints[i],
                    limits.upper
                );
            }
        }
    }
}

#[test]
fn ik_without_limit_check_may_exceed() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        check_limits: false, // Disable limit checking
        max_iterations: 200,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    // Should still not panic — limits just aren't enforced
    match result {
        Ok(sol) => {
            for &v in &sol.joints {
                assert!(v.is_finite(), "Output joints should be finite");
            }
        }
        Err(_) => {}
    }
}

#[test]
fn mid_configuration_within_limits() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mid = robot.mid_configuration();
    for (i, &v) in mid.iter().enumerate() {
        let limits = &robot.joint_limits[i];
        assert!(
            v >= limits.lower - 1e-10 && v <= limits.upper + 1e-10,
            "Mid config joint {} ({}) outside limits [{}, {}]",
            i,
            v,
            limits.lower,
            limits.upper
        );
    }
}

// ─── Single-waypoint trajectory (all parameterizers) ────────────────────────

#[test]
fn trapezoidal_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    let result = trapezoidal(&path, 2.0, 5.0).unwrap();
    assert_eq!(result.waypoints.len(), 1);
    assert_eq!(result.duration, Duration::ZERO);
    assert_eq!(
        result.waypoints[0].positions,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert!(result.waypoints[0].velocities.iter().all(|&v| v == 0.0));
}

#[test]
fn totp_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0]];
    let vel = vec![2.0; 3];
    let acc = vec![5.0; 3];
    let result = totp(&path, &vel, &acc, 0.001).unwrap();
    assert_eq!(result.waypoints.len(), 1);
    assert_eq!(result.duration, Duration::ZERO);
}

#[test]
fn jerk_limited_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0]];
    let result = jerk_limited(&path, 2.0, 5.0, 50.0).unwrap();
    assert_eq!(result.waypoints.len(), 1);
    assert_eq!(result.duration, Duration::ZERO);
}

#[test]
fn cubic_spline_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0]];
    let result = cubic_spline_time(&path, None, None);
    // Single waypoint: may return a degenerate trajectory or error
    match result {
        Ok(traj) => {
            assert!(traj.waypoints.len() <= 1);
        }
        Err(_) => {} // Also acceptable
    }
}

// ─── Start == Goal planning ─────────────────────────────────────────────────

#[test]
fn plan_start_equals_goal() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(200),
        max_iterations: 100,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });
    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(start.clone()));
    let result = planner.plan(&start, &goal);

    match result {
        Ok(plan) => {
            // Planner found a path — even start==goal may produce a non-trivial
            // RRT path due to tree exploration. Just verify it doesn't crash
            // and the path is valid.
            assert!(plan.num_waypoints() >= 1);
        }
        Err(_) => {} // Also acceptable to reject trivial plan
    }
}

// ─── Duplicate waypoints in trajectory ──────────────────────────────────────

#[test]
fn trapezoidal_duplicate_waypoints() {
    let path = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0], // duplicate of start
        vec![1.0, 1.0, 1.0],
    ];
    let result = trapezoidal(&path, 2.0, 5.0);
    // Must not panic — duplicate waypoints create zero-length segments
    let _ = result;
}

#[test]
fn totp_duplicate_waypoints() {
    let path = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let vel = vec![2.0; 3];
    let acc = vec![5.0; 3];
    let result = totp(&path, &vel, &acc, 0.001);
    let _ = result;
}

#[test]
fn jerk_limited_duplicate_waypoints() {
    let path = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let result = jerk_limited(&path, 2.0, 5.0, 50.0);
    let _ = result;
}

// ─── Very long trajectory ───────────────────────────────────────────────────

#[test]
fn trapezoidal_many_waypoints() {
    let dof = 6;
    let n = 500;
    let path: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..dof)
                .map(|j| (i as f64 * 0.01) + (j as f64 * 0.001))
                .collect()
        })
        .collect();
    let result = trapezoidal(&path, 2.0, 5.0);
    assert!(result.is_ok(), "500-waypoint trajectory should succeed");
    let traj = result.unwrap();
    assert!(traj.waypoints.len() >= n);
    assert!(traj.duration > Duration::ZERO);
}

#[test]
fn totp_many_waypoints() {
    let dof = 3;
    let n = 200;
    let path: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..dof)
                .map(|j| (i as f64 * 0.02) + (j as f64 * 0.01))
                .collect()
        })
        .collect();
    let vel = vec![2.0; dof];
    let acc = vec![5.0; dof];
    let result = totp(&path, &vel, &acc, 0.01);
    assert!(result.is_ok(), "200-waypoint TOTP should succeed");
    let traj = result.unwrap();
    assert!(traj.duration > Duration::ZERO);
}

// ─── Collision margin boundary ──────────────────────────────────────────────

#[test]
fn planner_with_zero_collision_margin() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(500),
        collision_margin: 0.0,
        ..PlannerConfig::default()
    });
    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);
    // Should work — zero margin means surfaces touching is OK
    match result {
        Ok(plan) => {
            assert!(plan.num_waypoints() >= 2);
        }
        Err(_) => {} // May not find path within timeout
    }
}

#[test]
fn planner_with_tiny_collision_margin() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(500),
        collision_margin: 1e-10,
        ..PlannerConfig::default()
    });
    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);
    // Tiny margin should behave like zero
    match result {
        Ok(plan) => {
            assert!(plan.num_waypoints() >= 2);
        }
        Err(_) => {}
    }
}

#[test]
fn planner_with_large_collision_margin() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(200),
        max_iterations: 500,
        collision_margin: 10.0, // 10 meter margin — impossible to satisfy
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });
    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);
    // With 10m margin, self-collision is almost certainly detected everywhere
    // Should either fail or produce a very short path
    let _ = result; // Must not panic
}

// ─── Zero-DOF robot (all fixed joints) ──────────────────────────────────────

#[test]
fn zero_dof_robot_mid_configuration() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="all_fixed">
  <link name="base"/>
  <link name="part1"/>
  <link name="part2"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="part1"/>
    <origin xyz="0 0 0.1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="part1"/>
    <child link="part2"/>
    <origin xyz="0 0 0.2"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(urdf).unwrap();
    assert_eq!(robot.dof, 0);
    let mid = robot.mid_configuration();
    assert!(mid.is_empty());
}

// ─── Trajectory with identical start and end ────────────────────────────────

#[test]
fn trapezoidal_identical_endpoints() {
    let path = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0], // same as start
    ];
    let result = trapezoidal(&path, 2.0, 5.0);
    // Zero displacement — should produce zero-duration or minimal trajectory
    match result {
        Ok(traj) => {
            // Duration should be zero or very small
            assert!(traj.duration.as_secs_f64() < 0.001);
        }
        Err(_) => {} // Also acceptable
    }
}

#[test]
fn jerk_limited_identical_endpoints() {
    let path = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
    let result = jerk_limited(&path, 2.0, 5.0, 50.0);
    match result {
        Ok(traj) => {
            assert!(traj.duration.as_secs_f64() < 0.001);
        }
        Err(_) => {}
    }
}

// ─── Very tiny motion ───────────────────────────────────────────────────────

#[test]
fn trapezoidal_epsilon_motion() {
    let path = vec![
        vec![0.0, 0.0, 0.0],
        vec![1e-15, 1e-15, 1e-15], // near-zero displacement
    ];
    let result = trapezoidal(&path, 2.0, 5.0);
    // Must not panic — should produce near-zero duration
    match result {
        Ok(traj) => {
            assert!(traj.duration.as_secs_f64() < 1.0);
        }
        Err(_) => {}
    }
}

// ─── Planner config extremes ────────────────────────────────────────────────

#[test]
fn planner_realtime_config() {
    let config = PlannerConfig::realtime();
    assert!(config.timeout <= Duration::from_millis(20));
    assert!(config.max_iterations > 0);
}

#[test]
fn planner_default_config_values() {
    let config = PlannerConfig::default();
    assert!(config.timeout > Duration::ZERO);
    assert!(config.collision_margin >= 0.0);
    assert!(config.max_iterations > 0);
}
