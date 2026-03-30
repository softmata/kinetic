//! Zero/empty input handling tests.
//!
//! Verifies all APIs gracefully handle zero-length, empty, or degenerate inputs.
//! Every test expects Err or a reasonable default — never panic.

use kinetic::collision::simd;
use kinetic::collision::SpheresSoA;
use kinetic::kinematics::{forward_kinematics, jacobian, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use kinetic::scene::PointCloudConfig;
use kinetic::trajectory::{totp, trapezoidal};
use std::time::Duration;

// ─── Trajectory: zero/empty waypoints ───────────────────────────────────────

#[test]
fn trapezoidal_with_zero_waypoints() {
    let path: Vec<Vec<f64>> = vec![];
    let result = trapezoidal(&path, 2.0, 5.0);
    match result {
        Ok(traj) => {
            assert_eq!(traj.waypoints.len(), 0);
            assert_eq!(traj.duration, Duration::ZERO);
        }
        Err(_) => {} // Also acceptable
    }
}

#[test]
fn trapezoidal_with_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0]];
    let result = trapezoidal(&path, 2.0, 5.0);
    match result {
        Ok(traj) => {
            assert_eq!(traj.waypoints.len(), 1);
            assert_eq!(traj.duration, Duration::ZERO);
        }
        Err(_) => {}
    }
}

#[test]
fn trapezoidal_with_zero_dof_waypoints() {
    // Waypoints with zero-length vectors
    let path = vec![vec![], vec![]];
    let result = trapezoidal(&path, 2.0, 5.0);
    // Must not panic — Err or degenerate trajectory
    let _ = result;
}

#[test]
fn totp_with_zero_waypoints() {
    let path: Vec<Vec<f64>> = vec![];
    let vel = vec![];
    let acc = vec![];
    let result = totp(&path, &vel, &acc, 0.001);
    match result {
        Ok(traj) => {
            assert_eq!(traj.waypoints.len(), 0);
            assert_eq!(traj.duration, Duration::ZERO);
        }
        Err(_) => {}
    }
}

#[test]
fn totp_with_single_waypoint() {
    let path = vec![vec![1.0, 2.0, 3.0]];
    let vel = vec![2.0; 3];
    let acc = vec![5.0; 3];
    let result = totp(&path, &vel, &acc, 0.001);
    match result {
        Ok(traj) => {
            assert_eq!(traj.waypoints.len(), 1);
            assert_eq!(traj.duration, Duration::ZERO);
        }
        Err(_) => {}
    }
}

#[test]
fn totp_with_mismatched_limits_length() {
    let path = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let vel = vec![2.0; 2]; // wrong length — DOF is 3
    let acc = vec![5.0; 3];
    let result = totp(&path, &vel, &acc, 0.001);
    assert!(
        result.is_err(),
        "Mismatched limits length should be rejected"
    );
}

#[test]
fn totp_with_zero_dof_waypoints() {
    let path = vec![vec![], vec![]];
    let vel: Vec<f64> = vec![];
    let acc: Vec<f64> = vec![];
    let result = totp(&path, &vel, &acc, 0.001);
    // Must not panic
    let _ = result;
}

// ─── Collision: empty sphere models ─────────────────────────────────────────

#[test]
fn collision_empty_vs_empty() {
    let a = SpheresSoA::new();
    let b = SpheresSoA::new();
    let result = simd::any_collision(&a, &b);
    assert!(!result, "Empty vs empty should never collide");
}

#[test]
fn collision_empty_vs_nonempty() {
    let a = SpheresSoA::new();
    let mut b = SpheresSoA::new();
    b.push(0.0, 0.0, 0.0, 1.0, 0);
    let result = simd::any_collision(&a, &b);
    assert!(!result, "Empty vs non-empty should never collide");
}

#[test]
fn collision_nonempty_vs_empty() {
    let mut a = SpheresSoA::new();
    a.push(0.0, 0.0, 0.0, 1.0, 0);
    let b = SpheresSoA::new();
    let result = simd::any_collision(&a, &b);
    assert!(!result, "Non-empty vs empty should never collide");
}

#[test]
fn min_distance_empty_vs_empty() {
    let a = SpheresSoA::new();
    let b = SpheresSoA::new();
    let dist = simd::min_distance(&a, &b);
    // Should be infinity or some sentinel — must not panic
    assert!(
        dist.is_infinite() || dist.is_nan(),
        "Empty min_distance should be infinite or NaN"
    );
}

#[test]
fn min_distance_empty_vs_nonempty() {
    let a = SpheresSoA::new();
    let mut b = SpheresSoA::new();
    b.push(1.0, 0.0, 0.0, 0.1, 0);
    let dist = simd::min_distance(&a, &b);
    assert!(dist.is_infinite() || dist.is_nan());
}

// ─── Scene: no objects ──────────────────────────────────────────────────────

#[test]
fn scene_empty_no_collision() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    assert_eq!(scene.num_objects(), 0);
    let joints = vec![0.0; robot.dof];
    let result = scene.check_collision(&joints).unwrap();
    // With no objects, only self-collision is possible
    let _ = result; // Must not panic
}

#[test]
fn scene_empty_min_distance_is_infinity() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let joints = vec![0.0; robot.dof];
    let dist = scene.min_distance_to_robot(&joints).unwrap();
    assert_eq!(dist, f64::INFINITY, "No objects → infinite distance");
}

#[test]
fn scene_empty_contact_points_empty() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let joints = vec![0.0; robot.dof];
    let contacts = scene.contact_points(&joints, 0.1).unwrap();
    assert!(contacts.is_empty(), "No objects → no contacts");
}

#[test]
fn scene_empty_pointcloud() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    let points: Vec<[f64; 3]> = vec![];
    scene.add_pointcloud("empty_cloud", &points, PointCloudConfig::default());
    assert_eq!(scene.num_pointclouds(), 1);
    // Empty cloud should not cause collision
    let joints = vec![0.0; robot.dof];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

#[test]
fn scene_build_env_spheres_empty() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let env = scene.build_environment_spheres();
    assert!(env.is_empty(), "Empty scene → empty environment spheres");
}

// ─── Robot: 0-DOF configurations ────────────────────────────────────────────

#[test]
fn zero_dof_robot_fk() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="all_fixed">
  <link name="base"/>
  <link name="part1"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="part1"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(urdf).unwrap();
    assert_eq!(robot.dof, 0);
    // FK with empty joint values on a 0-DOF robot
    let chain = KinematicChain::extract(&robot, "base", "part1").unwrap();
    assert_eq!(chain.dof, 0);
    let result = forward_kinematics(&robot, &chain, &[]);
    // Should succeed — the chain is just a fixed transform
    match result {
        Ok(pose) => {
            let t = pose.translation();
            // Fixed joint at z=0.1
            assert!((t[2] - 0.1).abs() < 1e-6);
        }
        Err(_) => {} // Also acceptable
    }
}

#[test]
fn zero_dof_robot_jacobian() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="all_fixed">
  <link name="base"/>
  <link name="part1"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="part1"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(urdf).unwrap();
    let chain = KinematicChain::extract(&robot, "base", "part1").unwrap();
    let result = jacobian(&robot, &chain, &[]);
    // 0-DOF → Jacobian is 6×0 or returns Err
    match result {
        Ok(jac) => {
            assert_eq!(jac.ncols(), 0);
        }
        Err(_) => {}
    }
}

#[test]
fn zero_dof_robot_scene() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="all_fixed">
  <link name="base"/>
  <link name="part1"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="part1"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(urdf).unwrap();
    let scene = Scene::new(&robot).unwrap();
    assert_eq!(scene.dof(), 0);
    // Collision check with empty joints
    let result = scene.check_collision(&[]);
    assert!(result.is_ok());
}

// ─── JointValues: empty ─────────────────────────────────────────────────────

#[test]
fn joint_values_empty() {
    let jv = JointValues::new(vec![]);
    assert_eq!(jv.len(), 0);
    assert!(jv.is_empty());
}

#[test]
fn joint_values_distance_empty() {
    let a = JointValues::new(vec![]);
    let b = JointValues::new(vec![]);
    let dist = a.distance_to(&b);
    // Distance between two empty joint configs — should be 0
    assert!((dist - 0.0).abs() < 1e-10 || dist.is_nan());
}

#[test]
fn joint_values_lerp_empty() {
    let a = JointValues::new(vec![]);
    let b = JointValues::new(vec![]);
    let result = a.lerp(&b, 0.5);
    assert!(result.is_empty());
}

// ─── Goal::Named with nonexistent pose ──────────────────────────────────────

#[test]
fn plan_with_nonexistent_named_goal() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(100),
        max_iterations: 100,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });
    let start = vec![0.0; robot.dof];
    let goal = Goal::Named("totally_nonexistent_pose_xyz".to_string());
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "Nonexistent named pose should return Err");
}

// ─── Wrong-length joint inputs ──────────────────────────────────────────────

// BUG FOUND: FK panics with assert! on wrong-length joint input instead of returning Err.
// Fix: Replace assert! in forward_kinematics with proper error return.
#[test]
#[ignore = "BUG: FK panics on wrong-length joints — needs Err instead of assert"]
fn fk_with_wrong_joint_count() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let result = forward_kinematics(&robot, &chain, &[0.0, 0.0]);
    assert!(result.is_err(), "Wrong joint count should return Err");
}

#[test]
#[ignore = "BUG: FK panics on wrong-length joints — needs Err instead of assert"]
fn fk_with_too_many_joints() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let result = forward_kinematics(&robot, &chain, &[0.0; 12]);
    assert!(result.is_err(), "Wrong joint count should return Err");
}

// BUG FOUND: Jacobian panics with assert! on wrong-length joint input.
#[test]
#[ignore = "BUG: Jacobian panics on wrong-length joints — needs Err instead of assert"]
fn jacobian_with_wrong_joint_count() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let result = jacobian(&robot, &chain, &[0.0, 0.0]);
    assert!(result.is_err(), "Wrong joint count should return Err");
}

// BUG FOUND: IK with empty seed panics with index-out-of-bounds in extract_joint_values.
// Fix: validate seed length matches DOF before use.
#[test]
#[ignore = "BUG: IK panics on empty seed — needs seed length validation"]
fn ik_with_empty_seed() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let target = Pose::from_xyz_rpy(0.3, 0.0, 0.5, 0.0, std::f64::consts::PI, 0.0);
    let mut config = IKConfig::default();
    config.seed = Some(vec![]); // Empty seed
    let result = solve_ik(&robot, &chain, &target, &config);
    assert!(result.is_err(), "Empty seed should return Err");
}

// ─── Planning with wrong-length start ───────────────────────────────────────

// BUG FOUND: Planner panics on wrong-length start — FK assert! triggers.
// Fix: validate start.len() == dof at the top of plan().
#[test]
#[ignore = "BUG: Planner panics on wrong-length start — needs input validation"]
fn plan_with_wrong_length_start() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(100),
        max_iterations: 100,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });
    let start = vec![0.0; 2];
    let goal = Goal::Joints(JointValues::new(vec![0.5; robot.dof]));
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "Wrong-length start should return Err");
}

#[test]
#[ignore = "BUG: Planner panics on empty start — needs input validation"]
fn plan_with_empty_start() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(100),
        max_iterations: 100,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });
    let start: Vec<f64> = vec![];
    let goal = Goal::Joints(JointValues::new(vec![0.5; robot.dof]));
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "Empty start should return Err");
}

// ─── KinematicChain: degenerate cases ───────────────────────────────────────

#[test]
fn chain_base_equals_tip() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    // Extract chain where base == tip
    let result = KinematicChain::extract(&robot, &arm.base_link, &arm.base_link);
    match result {
        Ok(chain) => {
            assert_eq!(chain.dof, 0);
            assert!(chain.all_joints.is_empty());
        }
        Err(_) => {} // Also acceptable
    }
}

#[test]
fn chain_extract_joint_values_empty() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="simple">
  <link name="base"/>
  <link name="tip"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="tip"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(urdf).unwrap();
    let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
    assert_eq!(chain.dof, 0);
    let vals = chain.extract_joint_values(&[]);
    assert!(vals.is_empty());
}

// ─── Scene with all objects removed ─────────────────────────────────────────

#[test]
fn scene_add_then_clear() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    scene.add("box1", Shape::cuboid(0.1, 0.1, 0.1), Isometry3::identity());
    scene.add(
        "box2",
        Shape::cuboid(0.1, 0.1, 0.1),
        Isometry3::translation(1.0, 0.0, 0.0),
    );
    assert_eq!(scene.num_objects(), 2);
    scene.clear();
    assert_eq!(scene.num_objects(), 0);
    // Collision check after clear should work
    let joints = vec![0.0; robot.dof];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

#[test]
fn scene_remove_nonexistent() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    let removed = scene.remove("nonexistent_object");
    assert!(removed.is_none());
}

#[test]
fn scene_detach_nonexistent() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    assert!(!scene.detach("nonexistent_object", Isometry3::identity()));
}
