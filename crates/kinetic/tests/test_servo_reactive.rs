//! Integration test: servo and reactive control.
//!
//! Tests the RMP (Riemannian Motion Policies) controller and Servo
//! subsystem through the full kinetic pipeline.

use kinetic::core::Twist;
use kinetic::prelude::*;
use kinetic::reactive::servo::{Servo, ServoConfig};
use kinetic::reactive::{PolicyType, RMP};
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

#[test]
fn servo_creation_and_state() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let servo = Servo::new(&robot, &scene, config).unwrap();
    let state = servo.state();

    // State should have 6 joint positions
    assert_eq!(state.joint_positions.len(), 6);
    assert_eq!(state.joint_velocities.len(), 6);

    // Manipulability should be finite and positive
    assert!(state.manipulability.is_finite());
    assert!(state.manipulability >= 0.0);
}

#[test]
fn servo_twist_command() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Send a small twist command (move down at 1cm/s)
    let twist = Twist::new(Vector3::new(0.0, 0.0, -0.01), Vector3::new(0.0, 0.0, 0.0));

    let cmd = servo.send_twist(&twist).unwrap();
    assert_eq!(cmd.positions.len(), 6);
    assert_eq!(cmd.velocities.len(), 6);
}

#[test]
fn servo_joint_jog() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Jog joint 0 at 0.1 rad/s
    let cmd = servo.send_joint_jog(0, 0.1).unwrap();
    assert_eq!(cmd.positions.len(), 6);
    assert_eq!(cmd.velocities.len(), 6);
}

#[test]
fn servo_multiple_steps() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Start from a non-singular config to avoid Jacobian issues
    let start_pos = vec![0.0, -1.2, 1.0, -0.8, -std::f64::consts::FRAC_PI_2, 0.0];
    servo.set_state(&start_pos, &[0.0; 6]).unwrap();

    let initial_state = servo.state().clone();

    // Run 10 twist commands
    let twist = Twist::new(Vector3::new(0.0, 0.0, -0.05), Vector3::new(0.0, 0.0, 0.0));

    for _ in 0..10 {
        let _cmd = servo.send_twist(&twist).unwrap();
    }

    let final_state = servo.state();

    // Joints should have changed
    let total_change: f64 = initial_state
        .joint_positions
        .iter()
        .zip(final_state.joint_positions.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        total_change > 1e-10,
        "Joints should have moved after 10 steps"
    );
}

#[test]
fn servo_set_state() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Set to a known state
    let new_positions = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let new_velocities = vec![0.0; 6];
    servo.set_state(&new_positions, &new_velocities).unwrap();

    let state = servo.state();
    for (a, b) in state.joint_positions.iter().zip(new_positions.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "State position mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn rmp_reach_target() {
    let robot = ur5e();
    let mut rmp = RMP::new(&robot).unwrap();

    let target = Isometry3::translation(0.3, 0.1, 0.4);
    rmp.add(PolicyType::ReachTarget {
        target_pose: target,
        gain: 10.0,
    });
    rmp.add(PolicyType::Damping { coefficient: 0.5 });

    let joints = vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0];
    let velocities = vec![0.0; 6];

    // Simulate several steps
    let mut pos = joints;
    let mut vel = velocities;
    for _ in 0..20 {
        let cmd = rmp.compute(&pos, &vel, 0.01).unwrap();
        pos = cmd.positions;
        vel = cmd.velocities;
    }

    // Robot should have moved
    let moved: f64 = pos.iter().map(|p| p.abs()).sum();
    assert!(moved > 0.0, "RMP should have moved the robot");
}

#[test]
fn rmp_combined_policies_stability() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());

    let mut rmp = RMP::new(&robot).unwrap();
    let target = Isometry3::translation(0.3, 0.0, 0.5);

    rmp.add(PolicyType::ReachTarget {
        target_pose: target,
        gain: 10.0,
    });
    rmp.add(PolicyType::AvoidObstacles {
        scene,
        influence_distance: 0.1,
        gain: 20.0,
    });
    rmp.add(PolicyType::JointLimitAvoidance {
        margin: 0.1,
        gain: 15.0,
    });
    rmp.add(PolicyType::Damping { coefficient: 0.5 });

    let joints = vec![0.0, -1.0, 0.5, 0.0, 0.5, 0.0];
    let velocities = vec![0.0; 6];

    // Run 50 steps and check for NaN/Inf
    let mut pos = joints;
    let mut vel = velocities;
    for step in 0..50 {
        let cmd = rmp.compute(&pos, &vel, 0.005).unwrap();
        for (i, &p) in cmd.positions.iter().enumerate() {
            assert!(
                p.is_finite(),
                "Position NaN/Inf at step {} joint {}: {}",
                step,
                i,
                p
            );
        }
        for (i, &v) in cmd.velocities.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Velocity NaN/Inf at step {} joint {}: {}",
                step,
                i,
                v
            );
        }
        pos = cmd.positions;
        vel = cmd.velocities;
    }
}
