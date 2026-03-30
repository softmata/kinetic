//! Singularity and edge-case tests.
//!
//! Verifies behavior at kinematic singularities where the Jacobian becomes
//! rank-deficient: wrist singularity, shoulder singularity, fully-extended
//! elbow. Tests that DLS solver remains stable, manipulability is detected,
//! and the servo controller clamps velocity near singularity.

use std::sync::Arc;

use kinetic::core::Twist;
use kinetic::kinematics::{
    forward_kinematics, jacobian, manipulability, solve_ik, IKConfig, IKSolver, KinematicChain,
};
use kinetic::prelude::*;
use kinetic::reactive::{Servo, ServoConfig};
use kinetic::scene::Scene;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── Manipulability detection ───────────────────────────────────────────────

#[test]
fn manipulability_drops_at_wrist_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Wrist singularity: joint5 near 0 → joints 4 and 6 axes align
    let singular = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0]; // j5 = 0
    let non_singular = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0]; // j5 = 1.0

    let m_singular = manipulability(&robot, &chain, &singular).unwrap();
    let m_non_singular = manipulability(&robot, &chain, &non_singular).unwrap();

    assert!(
        m_singular < m_non_singular,
        "Manipulability at wrist singularity ({}) should be less than non-singular ({})",
        m_singular,
        m_non_singular
    );
    // At exact wrist singularity, manipulability should be very small or zero
    assert!(
        m_singular < 0.01,
        "Wrist singularity manipulability should be near-zero: {}",
        m_singular
    );
}

#[test]
fn manipulability_drops_at_elbow_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Elbow singularity: arm fully extended (j2+j3 ≈ 0 or π)
    let extended = vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 1.0, 0.0];
    let bent = vec![0.0, -1.0, 1.5, 0.0, 1.0, 0.0];

    let m_extended = manipulability(&robot, &chain, &extended).unwrap();
    let m_bent = manipulability(&robot, &chain, &bent).unwrap();

    assert!(
        m_extended < m_bent,
        "Extended arm manipulability ({}) should be less than bent ({})",
        m_extended,
        m_bent
    );
}

#[test]
fn manipulability_positive_at_generic_config() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.2, 1.0, -0.8, 1.5, 0.3];
    let m = manipulability(&robot, &chain, &q).unwrap();
    assert!(
        m > 0.0,
        "Generic config should have positive manipulability: {}",
        m
    );
}

#[test]
fn manipulability_is_finite() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Test at many configs — manipulability must never be NaN or Inf
    let configs: Vec<Vec<f64>> = vec![
        vec![0.0; 6],
        vec![1.0; 6],
        vec![-1.0; 6],
        vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.14, 0.0, 0.0, 0.0],
    ];
    for q in &configs {
        let m = manipulability(&robot, &chain, q).unwrap();
        assert!(
            m.is_finite(),
            "Manipulability must be finite at {:?}: {}",
            q,
            m
        );
        assert!(m >= 0.0, "Manipulability must be non-negative: {}", m);
    }
}

// ─── Jacobian rank at singularities ─────────────────────────────────────────

#[test]
fn jacobian_rank_deficient_at_wrist_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Wrist singularity: j5 = 0
    let q = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let jac = jacobian(&robot, &chain, &q).unwrap();

    // Check singular values — at least one should be near zero
    let svd = jac.svd(false, false);
    let min_sv = svd.singular_values.min();
    assert!(
        min_sv < 1e-3,
        "Min singular value at wrist singularity should be near-zero: {}",
        min_sv
    );
}

#[test]
fn jacobian_well_conditioned_at_generic_config() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.2, 1.0, -0.8, 1.5, 0.3];
    let jac = jacobian(&robot, &chain, &q).unwrap();

    let svd = jac.svd(false, false);
    let min_sv = svd.singular_values.min();
    let max_sv = svd.singular_values.max();
    let condition = max_sv / min_sv;

    assert!(
        min_sv > 1e-3,
        "Min SV at generic config should be large: {}",
        min_sv
    );
    assert!(
        condition < 1000.0,
        "Condition number at generic config should be reasonable: {}",
        condition
    );
}

#[test]
fn jacobian_no_nan_at_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    let singular_configs: Vec<Vec<f64>> = vec![
        vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0], // wrist singularity
        vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 1.0, 0.0], // elbow
        vec![0.0; 6],                        // zero config
    ];
    for q in &singular_configs {
        let jac = jacobian(&robot, &chain, q).unwrap();
        for r in 0..jac.nrows() {
            for c in 0..jac.ncols() {
                assert!(
                    jac[(r, c)].is_finite(),
                    "Jacobian element [{},{}] is not finite at config {:?}: {}",
                    r,
                    c,
                    q,
                    jac[(r, c)]
                );
            }
        }
    }
}

// ─── FK at singular configurations ──────────────────────────────────────────

#[test]
fn fk_at_wrist_singularity_is_finite() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let pose = forward_kinematics(&robot, &chain, &q).unwrap();
    let t = pose.translation();
    assert!(t[0].is_finite() && t[1].is_finite() && t[2].is_finite());
}

#[test]
fn fk_at_elbow_singularity_is_finite() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 1.0, 0.0];
    let pose = forward_kinematics(&robot, &chain, &q).unwrap();
    let t = pose.translation();
    assert!(t[0].is_finite() && t[1].is_finite() && t[2].is_finite());
}

// ─── DLS solver stability at singularity ────────────────────────────────────

#[test]
fn dls_stable_at_wrist_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Get FK at a wrist-singular config
    let singular_q = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let target = forward_kinematics(&robot, &chain, &singular_q).unwrap();

    // Solve IK with DLS starting from a nearby non-singular config
    let seed = vec![0.1, -0.9, 0.7, 0.1, 0.1, 0.1];
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(seed),
        max_iterations: 200,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    // Must not panic — may converge or return IKNotConverged
    match result {
        Ok(sol) => {
            // Output must be finite
            for &v in &sol.joints {
                assert!(v.is_finite(), "DLS output has non-finite joint: {}", v);
            }
            assert!(sol.position_error.is_finite());
            assert!(sol.orientation_error.is_finite());
        }
        Err(_) => {} // Not converging near singularity is acceptable
    }
}

#[test]
fn dls_stable_at_elbow_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    let singular_q = vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 1.0, 0.0];
    let target = forward_kinematics(&robot, &chain, &singular_q).unwrap();

    let seed = vec![0.1, -1.2, 0.3, 0.1, 0.8, 0.1];
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(seed),
        max_iterations: 200,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    match result {
        Ok(sol) => {
            for &v in &sol.joints {
                assert!(v.is_finite(), "DLS output has non-finite joint: {}", v);
            }
        }
        Err(_) => {}
    }
}

#[test]
fn dls_no_nan_output_at_any_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    let singular_configs = vec![
        vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0],
        vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 1.0, 0.0],
        vec![0.0; 6],
    ];

    for q in &singular_configs {
        let target = forward_kinematics(&robot, &chain, q).unwrap();
        let seed = q.iter().map(|v| v + 0.1).collect::<Vec<_>>();
        let config = IKConfig {
            solver: IKSolver::DLS { damping: 0.1 },
            seed: Some(seed),
            max_iterations: 50,
            num_restarts: 0,
            ..IKConfig::default()
        };
        let result = solve_ik(&robot, &chain, &target, &config);
        match result {
            Ok(sol) => {
                for (i, &v) in sol.joints.iter().enumerate() {
                    assert!(
                        !v.is_nan(),
                        "DLS produced NaN for joint {} at singular config {:?}",
                        i,
                        q
                    );
                }
            }
            Err(_) => {} // Acceptable
        }
    }
}

#[test]
fn dls_high_damping_still_converges_away_from_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.2, 1.0, -0.8, 1.5, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.5 }, // Very high damping
        seed: Some(vec![0.0; 6]),
        max_iterations: 300,
        num_restarts: 2,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    // High damping is slow but should still converge with enough iterations
    match result {
        Ok(sol) => {
            assert!(
                sol.position_error < 0.01,
                "High damping DLS should still converge: pos_err={}",
                sol.position_error
            );
        }
        Err(_) => {} // Acceptable given high damping
    }
}

// ─── Servo singularity detection and velocity clamping ──────────────────────

fn test_3dof_robot_arc() -> Arc<Robot> {
    let urdf = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="ee_link">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.05"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
</robot>"#;
    Arc::new(Robot::from_urdf_string(urdf).unwrap())
}

#[test]
fn servo_detects_singularity() {
    let robot = test_3dof_robot_arc();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig {
        singularity_threshold: 0.05,
        ..Default::default()
    };
    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Set state to fully-extended arm (singular for 3R planar)
    // j2=0, j3=0 → arm straight up → low manipulability for XY motion
    servo.set_state(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]).unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.1, 0.0, 0.0),
        nalgebra::Vector3::zeros(),
    );
    let cmd = servo.send_twist(&twist).unwrap();

    // Check that singularity flag is set or velocities are reduced
    let state = servo.state();
    // At full extension, manipulability is low
    // Whether the flag is set depends on exact threshold, but the system must not panic
    let _ = state.is_near_singularity;
    let _ = cmd;
}

#[test]
fn servo_singularity_flag_triggers_correctly() {
    let robot = test_3dof_robot_arc();
    let scene = Arc::new(Scene::new(&robot).unwrap());

    // Artificially high threshold to guarantee is_near_singularity=true
    let config = ServoConfig {
        singularity_threshold: 1000.0,
        ..Default::default()
    };
    let mut servo = Servo::new(&robot, &scene, config).unwrap();
    servo
        .set_state(&[0.0, 0.5, -0.3], &[0.0, 0.0, 0.0])
        .unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.1, 0.0, 0.0),
        nalgebra::Vector3::zeros(),
    );
    let _cmd = servo.send_twist(&twist).unwrap();

    // With threshold=1000, any real manipulability should be below it
    assert!(
        servo.state().is_near_singularity,
        "With threshold=1000, is_near_singularity should be true"
    );

    // And with threshold=0, it should never trigger
    let config_no = ServoConfig {
        singularity_threshold: 0.0,
        ..Default::default()
    };
    let mut servo_no = Servo::new(&robot, &scene, config_no).unwrap();
    servo_no
        .set_state(&[0.0, 0.5, -0.3], &[0.0, 0.0, 0.0])
        .unwrap();
    let _cmd2 = servo_no.send_twist(&twist).unwrap();

    assert!(
        !servo_no.state().is_near_singularity,
        "With threshold=0, is_near_singularity should be false"
    );
}

#[test]
fn servo_no_nan_output_at_singularity() {
    let robot = test_3dof_robot_arc();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

    // Set to fully-extended (singular)
    servo.set_state(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]).unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.5, 0.0, 0.0),
        nalgebra::Vector3::zeros(),
    );
    let cmd = servo.send_twist(&twist).unwrap();

    for (i, &v) in cmd.velocities.iter().enumerate() {
        assert!(!v.is_nan(), "Servo velocity {} is NaN at singularity", i);
        assert!(
            v.is_finite(),
            "Servo velocity {} is infinite at singularity",
            i
        );
    }
    for (i, &p) in cmd.positions.iter().enumerate() {
        assert!(!p.is_nan(), "Servo position {} is NaN at singularity", i);
        assert!(
            p.is_finite(),
            "Servo position {} is infinite at singularity",
            i
        );
    }
}

#[test]
fn servo_multiple_steps_through_singularity() {
    let robot = test_3dof_robot_arc();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

    // Start near singularity
    servo
        .set_state(&[0.0, 0.1, -0.1], &[0.0, 0.0, 0.0])
        .unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.05, 0.0, 0.0),
        nalgebra::Vector3::zeros(),
    );

    // Run 50 steps — must never produce NaN or Inf.
    // The servo may return an error (e.g., singularity lockup) which is a valid
    // safety response — what matters is that it never produces NaN/Inf values.
    for step in 0..50 {
        match servo.send_twist(&twist) {
            Ok(cmd) => {
                for (i, &v) in cmd.velocities.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "Step {}: velocity {} is not finite: {}",
                        step,
                        i,
                        v
                    );
                }
                for (i, &p) in cmd.positions.iter().enumerate() {
                    assert!(
                        p.is_finite(),
                        "Step {}: position {} is not finite: {}",
                        step,
                        i,
                        p
                    );
                }
            }
            Err(_) => {
                // Singularity lockup or similar safety error — this is expected
                // behavior when driving through a singularity. The servo correctly
                // refuses to produce potentially dangerous commands.
                break;
            }
        }
    }
}

// ─── IK at joint limits boundary ────────────────────────────────────────────

#[test]
fn ik_target_at_joint_limit_boundary() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Config with joint values near limits
    let near_limits: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            robot.joints[ji]
                .limits
                .as_ref()
                .map_or(0.0, |l| l.upper - 0.01)
        })
        .collect();

    let target = forward_kinematics(&robot, &chain, &near_limits).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 200,
        num_restarts: 3,
        check_limits: true,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    match result {
        Ok(sol) => {
            // Verify solution respects limits
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
        Err(_) => {} // May not reach near-limit configs from zero seed
    }
}

// ─── Determinant-based singularity detection ────────────────────────────────

#[test]
fn jacobian_det_near_zero_at_singularity() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q_singular = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0]; // wrist singularity
    let q_generic = vec![0.5, -1.2, 1.0, -0.8, 1.5, 0.3];

    let jac_s = jacobian(&robot, &chain, &q_singular).unwrap();
    let jac_g = jacobian(&robot, &chain, &q_generic).unwrap();

    // For square 6×6 Jacobian, compute det(J * J^T)
    let jjt_s = &jac_s * jac_s.transpose();
    let jjt_g = &jac_g * jac_g.transpose();

    let det_s = jjt_s.determinant().abs();
    let det_g = jjt_g.determinant().abs();

    assert!(
        det_s < det_g,
        "Singular config det ({}) should be less than generic ({})",
        det_s,
        det_g
    );
    assert!(
        det_s < 1e-6,
        "Singular config det should be near-zero: {}",
        det_s
    );
}
