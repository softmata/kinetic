//! Tests that Rust error types produce human-readable, contextual messages
//! that will propagate correctly through PyO3 `.to_string()` conversion.
//!
//! These tests exercise the same error paths that the Python bindings use,
//! verifying that:
//! 1. Each KineticError variant has a meaningful Display implementation
//! 2. Error messages include actionable context (joint names, limits, residuals)
//! 3. Errors from public API functions contain enough information to diagnose issues
//! 4. Error classification (is_retryable, is_input_error) is correct

use std::time::Duration;

use std::sync::Arc;

use kinetic::prelude::*;

// ── KineticError Display messages ───────────────────────────────────────

#[test]
fn error_urdf_parse_includes_detail() {
    let e = KineticError::UrdfParse("unexpected element <foo> at line 42".into());
    let msg = e.to_string();
    assert!(msg.contains("URDF"), "Should mention URDF: {msg}");
    assert!(
        msg.contains("line 42"),
        "Should include parse detail: {msg}"
    );
}

#[test]
fn error_ik_not_converged_includes_iterations_and_residual() {
    let e = KineticError::IKNotConverged {
        iterations: 200,
        residual: 0.0154,
    };
    let msg = e.to_string();
    assert!(msg.contains("200"), "Should include iteration count: {msg}");
    assert!(msg.contains("0.015"), "Should include residual: {msg}");
    assert!(
        msg.contains("converge"),
        "Should mention convergence: {msg}"
    );
}

#[test]
fn error_planning_timeout_includes_elapsed_and_iterations() {
    let e = KineticError::PlanningTimeout {
        elapsed: Duration::from_millis(5000),
        iterations: 15000,
    };
    let msg = e.to_string();
    assert!(msg.contains("15000"), "Should include iterations: {msg}");
    assert!(msg.contains("5"), "Should include elapsed time: {msg}");
    assert!(msg.contains("timed out"), "Should say timed out: {msg}");
}

#[test]
fn error_joint_limit_includes_joint_name_value_and_bounds() {
    let e = KineticError::JointLimitViolation {
        name: "panda_joint3".into(),
        value: -3.5,
        min: -3.0718,
        max: -0.0698,
    };
    let msg = e.to_string();
    assert!(
        msg.contains("panda_joint3"),
        "Should include joint name: {msg}"
    );
    assert!(
        msg.contains("-3.5"),
        "Should include offending value: {msg}"
    );
    assert!(msg.contains("-3.0718"), "Should include lower limit: {msg}");
    assert!(msg.contains("-0.0698"), "Should include upper limit: {msg}");
}

#[test]
fn error_start_in_collision_is_clear() {
    let msg = KineticError::StartInCollision.to_string();
    assert!(
        msg.contains("Start") && msg.contains("collision"),
        "Should be descriptive: {msg}"
    );
}

#[test]
fn error_goal_in_collision_is_clear() {
    let msg = KineticError::GoalInCollision.to_string();
    assert!(
        msg.contains("Goal") && msg.contains("collision"),
        "Should be descriptive: {msg}"
    );
}

#[test]
fn error_goal_unreachable_is_clear() {
    let msg = KineticError::GoalUnreachable.to_string();
    assert!(
        msg.contains("unreachable"),
        "Should mention unreachable: {msg}"
    );
}

#[test]
fn error_no_ik_solution_is_clear() {
    let msg = KineticError::NoIKSolution.to_string();
    assert!(
        msg.contains("IK") && msg.contains("solution"),
        "Should mention IK: {msg}"
    );
}

#[test]
fn error_robot_config_not_found_includes_name() {
    let e = KineticError::RobotConfigNotFound("nonexistent_robot_xyz".into());
    let msg = e.to_string();
    assert!(
        msg.contains("nonexistent_robot_xyz"),
        "Should include robot name: {msg}"
    );
}

#[test]
fn error_cartesian_path_incomplete_includes_fraction() {
    let e = KineticError::CartesianPathIncomplete { fraction: 45.2 };
    let msg = e.to_string();
    assert!(msg.contains("45.2"), "Should include fraction: {msg}");
}

#[test]
fn error_collision_detected_includes_waypoint() {
    let e = KineticError::CollisionDetected { waypoint_index: 7 };
    let msg = e.to_string();
    assert!(msg.contains("7"), "Should include waypoint index: {msg}");
    assert!(msg.contains("waypoint"), "Should mention waypoint: {msg}");
}

#[test]
fn error_trajectory_limit_includes_detail() {
    let e = KineticError::TrajectoryLimitExceeded {
        waypoint_index: 3,
        detail: "joint 2 velocity 5.2 > limit 3.14".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("waypoint 3"), "Should include waypoint: {msg}");
    assert!(
        msg.contains("velocity"),
        "Should include violation detail: {msg}"
    );
}

#[test]
fn error_no_links_is_clear() {
    let msg = KineticError::NoLinks.to_string();
    assert!(msg.contains("no links"), "Should be clear: {msg}");
}

#[test]
fn error_link_not_found_includes_name() {
    let e = KineticError::LinkNotFound("end_effector_link".into());
    let msg = e.to_string();
    assert!(
        msg.contains("end_effector_link"),
        "Should include link name: {msg}"
    );
}

#[test]
fn error_joint_not_found_includes_name() {
    let e = KineticError::JointNotFound("shoulder_pan_joint".into());
    let msg = e.to_string();
    assert!(
        msg.contains("shoulder_pan_joint"),
        "Should include joint name: {msg}"
    );
}

#[test]
fn error_named_config_not_found_includes_name() {
    let e = KineticError::NamedConfigNotFound("home".into());
    let msg = e.to_string();
    assert!(msg.contains("home"), "Should include config name: {msg}");
}

#[test]
fn error_chain_extraction_includes_detail() {
    let e = KineticError::ChainExtraction("no path from base_link to ee_link".into());
    let msg = e.to_string();
    assert!(
        msg.contains("base_link"),
        "Should include chain details: {msg}"
    );
}

#[test]
fn error_planning_failed_includes_reason() {
    let e = KineticError::PlanningFailed("RRT tree could not extend".into());
    let msg = e.to_string();
    assert!(msg.contains("RRT"), "Should include failure reason: {msg}");
}

// ── Error classification ────────────────────────────────────────────────

#[test]
fn retryable_errors_are_correctly_classified() {
    // These should be retryable
    assert!(KineticError::PlanningTimeout {
        elapsed: Duration::from_millis(100),
        iterations: 1000,
    }
    .is_retryable());
    assert!(KineticError::IKNotConverged {
        iterations: 50,
        residual: 0.01,
    }
    .is_retryable());
    assert!(KineticError::CartesianPathIncomplete { fraction: 80.0 }.is_retryable());

    // These should NOT be retryable
    assert!(!KineticError::StartInCollision.is_retryable());
    assert!(!KineticError::GoalInCollision.is_retryable());
    assert!(!KineticError::GoalUnreachable.is_retryable());
    assert!(!KineticError::NoIKSolution.is_retryable());
    assert!(!KineticError::NoLinks.is_retryable());
    assert!(!KineticError::UrdfParse("bad".into()).is_retryable());
    assert!(!KineticError::RobotConfigNotFound("x".into()).is_retryable());
}

#[test]
fn input_errors_are_correctly_classified() {
    // These should be input errors
    assert!(KineticError::UrdfParse("bad".into()).is_input_error());
    assert!(KineticError::SrdfParse("bad".into()).is_input_error());
    assert!(KineticError::GoalUnreachable.is_input_error());
    assert!(KineticError::NoLinks.is_input_error());
    assert!(KineticError::LinkNotFound("x".into()).is_input_error());
    assert!(KineticError::JointNotFound("x".into()).is_input_error());
    assert!(KineticError::NamedConfigNotFound("x".into()).is_input_error());
    assert!(KineticError::ChainExtraction("x".into()).is_input_error());
    assert!(KineticError::IncompatibleKinematics("x".into()).is_input_error());
    assert!(KineticError::UnsupportedGoal("x".into()).is_input_error());
    assert!(KineticError::RobotConfigNotFound("x".into()).is_input_error());
    assert!(KineticError::JointLimitViolation {
        name: "j".into(),
        value: 0.0,
        min: -1.0,
        max: 1.0,
    }
    .is_input_error());

    // These should NOT be input errors
    assert!(!KineticError::PlanningTimeout {
        elapsed: Duration::from_millis(10),
        iterations: 100,
    }
    .is_input_error());
    assert!(!KineticError::IKNotConverged {
        iterations: 50,
        residual: 0.01,
    }
    .is_input_error());
    assert!(!KineticError::StartInCollision.is_input_error());
    assert!(!KineticError::GoalInCollision.is_input_error());
    assert!(!KineticError::CollisionDetected { waypoint_index: 0 }.is_input_error());
}

// ── Error is Send + Sync (required for PyO3) ───────────────────────────

#[test]
fn kinetic_error_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<KineticError>();
    assert_sync::<KineticError>();
}

#[test]
fn kinetic_error_is_debug() {
    let e = KineticError::IKNotConverged {
        iterations: 100,
        residual: 0.01,
    };
    let debug = format!("{e:?}");
    assert!(
        debug.contains("IKNotConverged"),
        "Debug should include variant name: {debug}"
    );
    assert!(
        debug.contains("100"),
        "Debug should include fields: {debug}"
    );
}

// ── Real API error paths (same paths Python bindings exercise) ──────────

#[test]
fn invalid_robot_name_produces_descriptive_error() {
    let result = Robot::from_name("nonexistent_robot_xyz");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent_robot_xyz") || msg.contains("not found"),
        "Error should mention the robot name or that it wasn't found: {msg}"
    );
}

#[test]
fn invalid_urdf_path_produces_descriptive_error() {
    let result = Robot::from_urdf("/tmp/definitely_not_a_real_urdf.urdf");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    // Should mention URDF or file-not-found
    assert!(!msg.is_empty(), "Error message should not be empty: {msg}");
}

#[test]
fn fk_wrong_dof_panics_with_descriptive_message() {
    // FK uses assert! for DOF checking — panics rather than Result.
    // In PyO3, panics are caught and converted to RuntimeError.
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = KinematicChain::auto_detect(&robot).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        forward_kinematics(&robot, &chain, &[0.0, 0.0, 0.0])
    }));
    assert!(result.is_err(), "FK with wrong DOF should panic");
}

#[test]
fn ik_unreachable_pose_returns_error() {
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = KinematicChain::auto_detect(&robot).unwrap();

    // Target far outside workspace
    let unreachable_pose = Pose(Isometry3::translation(100.0, 100.0, 100.0));
    let config = IKConfig {
        max_iterations: 50,
        ..IKConfig::default()
    };

    let result = solve_ik(&robot, &chain, &unreachable_pose, &config);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        !msg.is_empty(),
        "IK failure should produce descriptive error: {msg}"
    );
}

#[test]
fn planner_plan_with_wrong_dof_start_panics_or_errors() {
    // Planner may panic (via FK assert) or return error depending on code path.
    // In PyO3, both are caught and converted to RuntimeError.
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let start = vec![0.0, 0.0, 0.0];
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| planner.plan(&start, &goal)));
    // Either it panicked or returned Err — both are acceptable
    match result {
        Err(_) => {}     // panic caught
        Ok(Err(_)) => {} // returned error
        Ok(Ok(_)) => panic!("Wrong DOF start should fail"),
    }
}

#[test]
fn planner_plan_with_wrong_dof_goal_panics_or_errors() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let start = vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0];
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0]));

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| planner.plan(&start, &goal)));
    match result {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Wrong DOF goal should fail"),
    }
}

#[test]
fn scene_check_collision_wrong_dof_panics_or_errors() {
    // Scene collision check may panic (via FK assert) with wrong DOF.
    // In PyO3, panics are caught and converted to RuntimeError.
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scene.check_collision(&[0.0, 0.0, 0.0])
    }));
    match result {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Wrong DOF collision check should fail"),
    }
}

#[test]
fn scene_min_distance_wrong_dof_panics_or_errors() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scene.min_distance_to_robot(&[0.0])
    }));
    match result {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("Wrong DOF min_distance should fail"),
    }
}

#[test]
fn servo_invalid_joint_index_returns_error() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = kinetic::reactive::servo::ServoConfig::default();
    let mut servo = kinetic::reactive::servo::Servo::new(&robot, &scene, config).unwrap();

    // Set valid initial state
    let init_pos = vec![0.0, -1.2, 1.0, -0.8, -1.57, 0.0];
    let init_vel = vec![0.0; 6];
    servo.set_state(&init_pos, &init_vel).unwrap();

    // Joint index 99 doesn't exist
    let result = servo.send_joint_jog(99, 0.1);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("99") || msg.contains("joint") || msg.contains("DOF"),
        "Should mention invalid joint index: {msg}"
    );
}

#[test]
fn servo_wrong_dof_set_state_returns_error() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = kinetic::reactive::servo::ServoConfig::default();
    let mut servo = kinetic::reactive::servo::Servo::new(&robot, &scene, config).unwrap();

    // Wrong number of joints
    let result = servo.set_state(&[0.0, 0.0], &[0.0, 0.0]);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(!msg.is_empty(), "Wrong DOF set_state should error: {msg}");
}

// ── Error message consistency ───────────────────────────────────────────

#[test]
fn all_error_variants_produce_nonempty_messages() {
    let errors: Vec<KineticError> = vec![
        KineticError::UrdfParse("test".into()),
        KineticError::IKNotConverged {
            iterations: 1,
            residual: 0.1,
        },
        KineticError::PlanningTimeout {
            elapsed: Duration::from_millis(1),
            iterations: 1,
        },
        KineticError::StartInCollision,
        KineticError::GoalInCollision,
        KineticError::GoalUnreachable,
        KineticError::NoIKSolution,
        KineticError::JointLimitViolation {
            name: "j".into(),
            value: 0.0,
            min: -1.0,
            max: 1.0,
        },
        KineticError::RobotConfigNotFound("r".into()),
        KineticError::CartesianPathIncomplete { fraction: 50.0 },
        KineticError::CollisionDetected { waypoint_index: 0 },
        KineticError::TrajectoryLimitExceeded {
            waypoint_index: 0,
            detail: "test".into(),
        },
        KineticError::NoLinks,
        KineticError::LinkNotFound("l".into()),
        KineticError::JointNotFound("j".into()),
        KineticError::NamedConfigNotFound("c".into()),
        KineticError::SrdfParse("s".into()),
        KineticError::ChainExtraction("c".into()),
        KineticError::IncompatibleKinematics("k".into()),
        KineticError::UnsupportedGoal("g".into()),
        KineticError::PlanningFailed("f".into()),
        KineticError::Other("o".into()),
    ];

    for e in &errors {
        let msg = e.to_string();
        assert!(
            !msg.is_empty(),
            "Error variant {:?} produced empty message",
            e
        );
        // Other is a pass-through, so it can be short
        if !matches!(e, KineticError::Other(_)) {
            assert!(
                msg.len() > 3,
                "Error variant {:?} message too short: '{}'",
                e,
                msg
            );
        }
    }
}

// ── GPU error propagation ───────────────────────────────────────────────

#[test]
fn gpu_error_variants_produce_descriptive_messages() {
    use kinetic::gpu::GpuError;

    let errors: Vec<GpuError> = vec![
        GpuError::NoAdapter,
        GpuError::BufferMapping,
        GpuError::InvalidConfig("test config error".into()),
    ];

    for e in &errors {
        let msg = e.to_string();
        assert!(!msg.is_empty(), "GpuError {:?} empty message", e);
    }

    // Specific checks
    let no_adapter = GpuError::NoAdapter.to_string();
    assert!(
        no_adapter.contains("GPU") || no_adapter.contains("adapter"),
        "NoAdapter should mention GPU: {no_adapter}"
    );

    let invalid = GpuError::InvalidConfig("DOF mismatch".into()).to_string();
    assert!(
        invalid.contains("DOF mismatch"),
        "InvalidConfig should include detail: {invalid}"
    );
}

// ── Servo error propagation ─────────────────────────────────────────────

#[test]
fn servo_error_variants_produce_descriptive_messages() {
    use kinetic::reactive::servo::ServoError;

    let e = ServoError::InvalidJointIndex { index: 10, dof: 6 };
    let msg = e.to_string();
    assert!(msg.contains("10"), "Should include index: {msg}");
    assert!(msg.contains("6"), "Should include DOF: {msg}");

    let e2 = ServoError::EmergencyStop {
        distance: 0.01,
        stop: 0.05,
    };
    let msg2 = e2.to_string();
    assert!(
        msg2.contains("Emergency") || msg2.contains("stop"),
        "Should mention emergency stop: {msg2}"
    );
}
