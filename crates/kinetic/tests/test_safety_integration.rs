//! Safety integration tests for real-robot deployment.
//!
//! These tests verify the COMPLETE safety chain end-to-end.
//! Every test here represents a scenario that, if it fails,
//! could cause physical damage to a real robot or its environment.

use kinetic::collision::{CollisionPointTree, AABB};
use kinetic::execution::{CommandSink, ExecutionConfig, RealTimeExecutor};
use kinetic::kinematics::{forward_kinematics, KinematicChain};
use kinetic::planning::Planner;
use kinetic::prelude::*;

use std::sync::Arc;

// ─── Test helpers ────────────────────────────────────────────────────────────

struct RecordingSink {
    commands: Vec<(Vec<f64>, Vec<f64>)>,
}

impl RecordingSink {
    fn new() -> Self {
        Self { commands: vec![] }
    }
}

impl CommandSink for RecordingSink {
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> std::result::Result<(), String> {
        self.commands
            .push((positions.to_vec(), velocities.to_vec()));
        Ok(())
    }
}

// ─── Test 1: CAPT broadphase covers full workspace ──────────────────────────

#[test]
fn safety_capt_full_workspace_coverage() {
    // Tests the fix for the old 2m influence radius bug.
    // Obstacle at x=3m should still affect cells near it.
    let mut obstacles = kinetic::collision::SpheresSoA::new();
    obstacles.push(3.0, 0.0, 0.0, 0.5, 0); // Obstacle at x=3, radius=0.5

    let bounds = AABB::new(-4.0, -4.0, -4.0, 4.0, 4.0, 4.0);
    let capt = CollisionPointTree::build(&obstacles, 0.1, bounds);

    // Clearance at x=2.6 should be ~0.1m (obstacle surface at x=2.5)
    let clearance = capt.clearance_at(2.6, 0.0, 0.0);
    assert!(
        clearance < 0.2,
        "SAFETY: CAPT should report clearance < 0.2m at x=2.6, got {:.3}m. \
         Obstacle at x=3 (r=0.5) has surface at x=2.5.",
        clearance
    );

    // Should NOT be collision-free for a sphere of radius 0.2 at x=2.6
    let is_free = capt.check_point(2.6, 0.0, 0.0, 0.2);
    assert!(
        !is_free,
        "SAFETY: CAPT reports collision-free for r=0.2 sphere at x=2.6, \
         but obstacle surface is at x=2.5 (only 0.1m clearance)"
    );
}

// ─── Test 2: Collision detection near end-effector ──────────────────────────

#[test]
fn safety_collision_detected_near_ee() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

    let mut scene = Scene::new(&robot).unwrap();
    let joints = vec![0.0; chain.dof];

    // Get EE position at zero config
    let ee_pose = forward_kinematics(&robot, &chain, &joints).unwrap();
    let ee_pos = ee_pose.translation();

    // Place obstacle sphere 3cm from EE
    scene.add(
        "near_obstacle",
        Shape::sphere(0.02),
        Isometry3::translation(ee_pos.x + 0.03, ee_pos.y, ee_pos.z),
    );

    // Should detect as very close (even if not technically colliding with spheres)
    let distance = scene
        .min_distance_to_robot(&joints)
        .unwrap_or(f64::INFINITY);

    assert!(
        distance < 0.15,
        "SAFETY: Obstacle 3cm from EE should report small distance, got {:.4}m",
        distance
    );
}

// ─── Test 3: Out-of-limits trajectory rejected ──────────────────────────────

#[test]
fn safety_executor_rejects_out_of_limits() {
    let robot = Robot::from_name("ur5e").unwrap();
    let dof = robot.dof;

    // Build trajectory with wildly out-of-bounds joint
    let start = vec![0.0; dof];
    let mut goal = vec![0.0; dof];
    goal[0] = 99.0; // Way beyond any limit

    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let traj = kinetic::trajectory::trapezoidal_per_joint(
        &vec![start, goal],
        &vel_limits,
        &accel_limits,
    )
    .unwrap();

    let limits: Vec<(f64, f64)> = robot
        .joint_limits
        .iter()
        .map(|l| (l.lower, l.upper))
        .collect();

    let executor = RealTimeExecutor::new(ExecutionConfig {
        joint_limits: Some(limits),
        rate_hz: 100.0,
        ..Default::default()
    });

    let mut sink = RecordingSink::new();
    let result = executor.execute(&traj, &mut sink);

    assert!(
        result.is_err(),
        "SAFETY: Executor must reject trajectory with joint at 99 rad"
    );
    assert_eq!(
        sink.commands.len(),
        0,
        "SAFETY: No commands should be sent before validation"
    );
}

// ─── Test 4: Trajectory at exact limit accepted ─────────────────────────────

#[test]
fn safety_executor_accepts_at_exact_limit() {
    let robot = Robot::from_name("ur5e").unwrap();
    let dof = robot.dof;

    let start = vec![0.0; dof];
    let mut goal = vec![0.0; dof];
    goal[0] = robot.joint_limits[0].upper; // Exact upper limit

    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let traj = kinetic::trajectory::trapezoidal_per_joint(
        &vec![start, goal],
        &vel_limits,
        &accel_limits,
    )
    .unwrap();

    let limits: Vec<(f64, f64)> = robot
        .joint_limits
        .iter()
        .map(|l| (l.lower, l.upper))
        .collect();

    let executor = RealTimeExecutor::new(ExecutionConfig {
        joint_limits: Some(limits),
        rate_hz: 100.0,
        ..Default::default()
    });

    let mut sink = RecordingSink::new();
    let result = executor.execute(&traj, &mut sink);

    assert!(
        result.is_ok(),
        "Trajectory at exact limit should be accepted: {:?}",
        result.err()
    );
    assert!(sink.commands.len() > 0, "Commands should be sent");
}

// ─── Test 5: Require feedback blocks blind execution ────────────────────────

#[test]
fn safety_require_feedback_blocks_blind() {
    let robot = Robot::from_name("ur5e").unwrap();
    let dof = robot.dof;

    let path = vec![vec![0.0; dof], vec![0.1; dof]];
    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let traj =
        kinetic::trajectory::trapezoidal_per_joint(&path, &vel_limits, &accel_limits).unwrap();

    let executor = RealTimeExecutor::new(ExecutionConfig {
        require_feedback: true,
        rate_hz: 100.0,
        ..Default::default()
    });

    let mut sink = RecordingSink::new();
    // execute() has no feedback — should be rejected when require_feedback=true
    let result = executor.execute(&traj, &mut sink);

    assert!(
        result.is_err(),
        "SAFETY: Must reject execution when require_feedback=true and no feedback"
    );
    assert_eq!(sink.commands.len(), 0, "No commands should be sent");
}

// ─── Test 6: Planned path is collision-free ─────────────────────────────────

#[test]
fn safety_planned_path_collision_free() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    // Plan a simple motion (no obstacles — tests self-collision avoidance)
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal = vec![0.5, -1.0, 0.3, -1.0, -0.3, 0.2];

    let planner = Planner::new(&robot).unwrap();
    let result = planner
        .plan(&start, &Goal::Joints(JointValues(goal)))
        .unwrap();

    assert!(result.waypoints.len() >= 2, "Should have at least start+goal");

    // Verify all waypoints are within joint limits (safety invariant)
    for (i, wp) in result.waypoints.iter().enumerate() {
        for (j, &pos) in wp.iter().enumerate() {
            assert!(
                pos >= robot.joint_limits[j].lower - 0.01
                    && pos <= robot.joint_limits[j].upper + 0.01,
                "SAFETY: Planned waypoint {} joint {} = {:.4} outside limits [{:.4}, {:.4}]",
                i, j, pos, robot.joint_limits[j].lower, robot.joint_limits[j].upper
            );
        }
    }

    // Verify no self-collision at any waypoint
    for (i, wp) in result.waypoints.iter().enumerate() {
        let in_collision = scene.check_collision(wp).unwrap_or(true);
        assert!(
            !in_collision,
            "SAFETY: Planned waypoint {} is in self-collision!",
            i
        );
    }
}

// ─── Test 7: Servo outputs are finite ───────────────────────────────────────

#[test]
fn safety_servo_finite_outputs() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    let config = kinetic::reactive::ServoConfig::default();

    let robot = Arc::new(robot);
    let scene = Arc::new(scene);
    let mut servo = kinetic::reactive::Servo::new(
        &robot,
        &scene,
        config,
    )
    .unwrap();

    let twist = kinetic::core::Twist::new(
        nalgebra::Vector3::new(0.1, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.1),
    );

    for tick in 0..20 {
        match servo.send_twist(&twist) {
            Ok(cmd) => {
                for (j, &p) in cmd.positions.iter().enumerate() {
                    assert!(
                        p.is_finite(),
                        "SAFETY: Servo tick {} joint {} produced non-finite position: {}",
                        tick, j, p
                    );
                }
                for (j, &v) in cmd.velocities.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "SAFETY: Servo tick {} joint {} produced non-finite velocity: {}",
                        tick, j, v
                    );
                }
            }
            Err(_e) => {
                // Servo errors (e.g., singularity lockup) are acceptable —
                // they're the safety system working. The key is no NaN/Inf.
                break;
            }
        }
    }
}
