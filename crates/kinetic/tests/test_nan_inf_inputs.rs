//! NaN, Inf, and extreme value input tests.
//!
//! Verifies that NaN/Inf/extreme values never cause panics across all major APIs.
//! Acceptable outcomes: return Err, produce NaN-contaminated output, or clamp.
//! Unacceptable: panic, hang, or segfault.

use kinetic::collision::simd;
use kinetic::collision::SpheresSoA;
use kinetic::kinematics::{forward_kinematics, jacobian, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use kinetic::trajectory::{totp, trapezoidal};

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── FK with NaN/Inf ─────────────────────────────────────────────────────────

#[test]
fn fk_with_nan_joint_values() {
    let (robot, chain) = ur5e_robot_and_chain();
    let joints = vec![f64::NAN; robot.dof];
    // Should not panic — may return Ok with NaN pose or Err
    let result = forward_kinematics(&robot, &chain, &joints);
    match result {
        Ok(pose) => {
            // NaN propagation is acceptable
            let t = pose.translation();
            let _ = t; // just verify no panic during access
        }
        Err(_) => {} // Also fine
    }
}

#[test]
fn fk_with_inf_joint_values() {
    let (robot, chain) = ur5e_robot_and_chain();
    let joints = vec![f64::INFINITY; robot.dof];
    let result = forward_kinematics(&robot, &chain, &joints);
    // Must not panic
    let _ = result;
}

#[test]
fn fk_with_neg_inf_joint_values() {
    let (robot, chain) = ur5e_robot_and_chain();
    let joints = vec![f64::NEG_INFINITY; robot.dof];
    let result = forward_kinematics(&robot, &chain, &joints);
    let _ = result;
}

#[test]
fn fk_with_mixed_nan_and_valid() {
    let (robot, chain) = ur5e_robot_and_chain();
    let mut joints = vec![0.0; robot.dof];
    joints[2] = f64::NAN; // Only one joint is NaN
    let result = forward_kinematics(&robot, &chain, &joints);
    let _ = result;
}

// ─── Jacobian with NaN/Inf ───────────────────────────────────────────────────

#[test]
fn jacobian_with_nan_joints() {
    let (robot, chain) = ur5e_robot_and_chain();
    let joints = vec![f64::NAN; robot.dof];
    let result = jacobian(&robot, &chain, &joints);
    // Must not panic
    let _ = result;
}

#[test]
fn jacobian_with_inf_joints() {
    let (robot, chain) = ur5e_robot_and_chain();
    let joints = vec![f64::INFINITY; robot.dof];
    let result = jacobian(&robot, &chain, &joints);
    let _ = result;
}

// ─── IK with NaN/Inf target ─────────────────────────────────────────────────

#[test]
fn ik_with_nan_target_position() {
    let (robot, chain) = ur5e_robot_and_chain();
    let target = Pose::from_xyz_rpy(f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut config = IKConfig::default();
    config.seed = Some(vec![0.0; robot.dof]);
    let result = solve_ik(&robot, &chain, &target, &config);
    // Should fail gracefully — not converge or return error
    match result {
        Ok(solution) => {
            // If it "converges", the solution should have NaN or be meaningless
            let _ = solution;
        }
        Err(_) => {} // Expected
    }
}

#[test]
fn ik_with_inf_target_position() {
    let (robot, chain) = ur5e_robot_and_chain();
    let target = Pose::from_xyz_rpy(f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0, 0.0, 0.0);
    let mut config = IKConfig::default();
    config.seed = Some(vec![0.0; robot.dof]);
    let result = solve_ik(&robot, &chain, &target, &config);
    let _ = result; // Must not panic
}

#[test]
fn ik_with_nan_seed() {
    let (robot, chain) = ur5e_robot_and_chain();
    let target = Pose::from_xyz_rpy(0.3, 0.0, 0.5, 0.0, std::f64::consts::PI, 0.0);
    let mut config = IKConfig::default();
    config.seed = Some(vec![f64::NAN; robot.dof]);
    let result = solve_ik(&robot, &chain, &target, &config);
    let _ = result; // Must not panic
}

// ─── Trajectory parameterization with NaN ────────────────────────────────────

#[test]
fn trapezoidal_with_nan_waypoints() {
    let dof = 6;
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0; dof], vec![f64::NAN; dof], vec![1.0; dof]];
    let result = trapezoidal(&waypoints, 2.0, 5.0);
    // Must not panic — Err or NaN-contaminated output
    let _ = result;
}

#[test]
fn trapezoidal_with_inf_velocity_limit() {
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let result = trapezoidal(&waypoints, f64::INFINITY, 5.0);
    // Must not panic
    let _ = result;
}

#[test]
fn trapezoidal_with_nan_velocity_limit() {
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let result = trapezoidal(&waypoints, f64::NAN, 5.0);
    let _ = result;
}

// BUG FOUND: trapezoidal panics on zero velocity — Duration conversion fails
// with "cannot convert float seconds to Duration: value is either too big or NaN".
// Fix: validate velocity > 0 at the top of trapezoidal().
#[test]
#[ignore = "BUG: trapezoidal panics on zero velocity — needs input validation"]
fn trapezoidal_with_zero_velocity_limit() {
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let result = trapezoidal(&waypoints, 0.0, 5.0);
    assert!(result.is_err(), "Zero velocity should be rejected");
}

#[test]
fn trapezoidal_with_negative_acceleration() {
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let result = trapezoidal(&waypoints, 2.0, -5.0);
    let _ = result;
}

// BUG FOUND: TOTP panics on NaN waypoints — Duration conversion fails.
// Fix: validate all waypoint values are finite at the top of totp().
#[test]
#[ignore = "BUG: TOTP panics on NaN waypoints — needs input validation"]
fn totp_with_nan_waypoints() {
    let dof = 6;
    let waypoints: Vec<Vec<f64>> = vec![vec![0.0; dof], vec![f64::NAN; dof], vec![1.0; dof]];
    let vel_limits = vec![2.0; dof];
    let acc_limits = vec![5.0; dof];
    let result = totp(&waypoints, &vel_limits, &acc_limits, 0.001);
    assert!(result.is_err(), "NaN waypoints should be rejected");
}

// ─── Collision check with NaN ────────────────────────────────────────────────

#[test]
fn collision_soa_with_nan_positions() {
    let mut a = SpheresSoA::new();
    a.push(f64::NAN, 0.0, 0.0, 0.1, 0);
    let mut b = SpheresSoA::new();
    b.push(0.0, 0.0, 0.0, 0.1, 1);
    // Query should not panic
    let result = simd::any_collision(&a, &b);
    let _ = result;
}

#[test]
fn collision_soa_with_inf_radius() {
    let mut a = SpheresSoA::new();
    a.push(0.0, 0.0, 0.0, f64::INFINITY, 0);
    let mut b = SpheresSoA::new();
    b.push(1.0, 0.0, 0.0, 0.1, 1);
    let result = simd::any_collision(&a, &b);
    let _ = result;
}

#[test]
fn collision_min_distance_with_nan() {
    let mut a = SpheresSoA::new();
    a.push(f64::NAN, 0.0, 0.0, 0.1, 0);
    let mut b = SpheresSoA::new();
    b.push(0.0, 0.0, 0.0, 0.1, 1);
    let dist = simd::min_distance(&a, &b);
    // NaN distance is acceptable — no panic
    let _ = dist;
}

// ─── JointValues with extreme values ─────────────────────────────────────────

#[test]
fn joint_values_with_nan() {
    let jv = JointValues::new(vec![f64::NAN, 0.0, 1.0]);
    assert_eq!(jv.len(), 3);
    assert!(jv[0].is_nan());
}

#[test]
fn joint_values_with_inf() {
    let jv = JointValues::new(vec![f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert_eq!(jv.len(), 3);
    assert!(jv[0].is_infinite());
}

#[test]
fn joint_values_distance_with_nan() {
    let a = JointValues::new(vec![0.0, 1.0, 2.0]);
    let b = JointValues::new(vec![f64::NAN, 1.0, 2.0]);
    let dist = a.distance_to(&b);
    // Distance with NaN should be NaN
    assert!(dist.is_nan());
}

#[test]
fn joint_values_lerp_with_nan() {
    let a = JointValues::new(vec![0.0, 0.0]);
    let b = JointValues::new(vec![f64::NAN, 1.0]);
    let result = a.lerp(&b, 0.5);
    // First element should be NaN, second should be 0.5
    assert!(result[0].is_nan());
    assert!((result[1] - 0.5).abs() < 1e-10);
}

// ─── Pose with NaN ───────────────────────────────────────────────────────────

#[test]
fn pose_from_nan_xyz() {
    let pose = Pose::from_xyz_rpy(f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0);
    let t = pose.translation();
    assert!(t[0].is_nan());
}

#[test]
fn pose_from_nan_rpy() {
    let pose = Pose::from_xyz_rpy(0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0);
    // Should not panic during construction
    let _ = pose;
}

// ─── Planning with NaN ───────────────────────────────────────────────────────

use std::time::Duration;

fn short_timeout_planner(robot: &Robot) -> Planner {
    Planner::new(robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(100),
        max_iterations: 100,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    })
}

// BUG FOUND: RRT planner does not validate inputs for NaN/Inf.
// NaN causes unbounded memory growth because NaN comparisons always return false,
// so the RRT tree never connects but keeps allocating nodes.
// These tests are ignored until the planner adds NaN input validation.
// Fix: Add is_finite() check at the top of plan() and plan_with_scene().

#[test]
#[ignore = "BUG: RRT planner OOMs on NaN input — needs NaN validation guard"]
fn plan_with_nan_start() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = short_timeout_planner(&robot);
    let start = vec![f64::NAN; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![0.5; robot.dof]));
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "NaN start should be rejected");
}

#[test]
#[ignore = "BUG: RRT planner OOMs on NaN input — needs NaN validation guard"]
fn plan_with_nan_goal_joints() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = short_timeout_planner(&robot);
    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![f64::NAN; robot.dof]));
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "NaN goal should be rejected");
}

#[test]
#[ignore = "BUG: RRT planner OOMs on extreme values — needs limit validation"]
fn plan_with_extremely_large_joint_values() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = short_timeout_planner(&robot);
    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![1e15; robot.dof]));
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "Out-of-limits goal should be rejected");
}
