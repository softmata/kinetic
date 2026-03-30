//! Acceptance tests: 07 numerical_robustness
//! Spec: doc_tests/07_NUMERICAL_ROBUSTNESS.md
//!
//! NaN/Inf handling, proptest never-panic suites, edge-value inputs.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::prelude::*;
use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig};

// ─── NaN inputs never panic ─────────────────────────────────────────────────

#[test]
fn nan_joints_fk_does_not_panic() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let nan_joints = vec![f64::NAN; chain.dof];
        // Must not panic — Err or weird pose is fine
        let _ = forward_kinematics(&robot, &chain, &nan_joints);
    }
}

#[test]
fn nan_joints_ik_does_not_panic() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let target = Pose::from_xyz(0.3, 0.0, 0.4);
        let config = IKConfig {
            seed: Some(vec![f64::NAN; chain.dof]),
            num_restarts: 0,
            max_iterations: 10,
            ..Default::default()
        };
        let _ = solve_ik(&robot, &chain, &target, &config);
    }
}

#[test]
fn nan_target_ik_does_not_panic() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let nan_target = Pose::from_xyz(f64::NAN, 0.0, 0.0);
        let config = IKConfig { num_restarts: 0, max_iterations: 10, ..Default::default() };
        let _ = solve_ik(&robot, &chain, &nan_target, &config);
    }
}

// ─── Inf inputs never panic ─────────────────────────────────────────────────

#[test]
fn inf_joints_fk_does_not_panic() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let inf_joints = vec![f64::INFINITY; chain.dof];
        let _ = forward_kinematics(&robot, &chain, &inf_joints);
        let neg_inf_joints = vec![f64::NEG_INFINITY; chain.dof];
        let _ = forward_kinematics(&robot, &chain, &neg_inf_joints);
    }
}

// ─── Extreme values never panic ─────────────────────────────────────────────

#[test]
fn extreme_values_fk_does_not_panic() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let max_joints = vec![f64::MAX; chain.dof];
        let _ = forward_kinematics(&robot, &chain, &max_joints);
        let min_joints = vec![f64::MIN; chain.dof];
        let _ = forward_kinematics(&robot, &chain, &min_joints);
    }
}

#[test]
fn zero_joints_fk_produces_finite_result() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let zeros = vec![0.0; chain.dof];
        let result = forward_kinematics(&robot, &chain, &zeros);
        assert!(result.is_ok(), "{name}: FK at zero should succeed");
        let pose = result.unwrap();
        let t = pose.translation();
        assert!(t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
            "{name}: FK at zero should be finite");
    }
}

// ─── Large-scale proptest: FK never panics ──────────────────────────────────

#[test]
fn proptest_fk_never_panics_1000() {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let robot = load_robot("ur5e");
    let chain = load_chain(&robot);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut ok_count = 0;
    let mut err_count = 0;

    for _ in 0..1000 {
        let joints: Vec<f64> = (0..chain.dof)
            .map(|_| {
                match rng.gen_range(0..5) {
                    0 => rng.gen_range(-10.0..10.0),
                    1 => rng.gen_range(-1e6..1e6),
                    2 => 0.0,
                    3 => if rng.gen_bool(0.5) { f64::INFINITY } else { f64::NEG_INFINITY },
                    _ => f64::NAN,
                }
            })
            .collect();
        // Must not panic — Ok or Err both fine
        match forward_kinematics(&robot, &chain, &joints) {
            Ok(_) => ok_count += 1,
            Err(_) => err_count += 1,
        }
    }
    // At least some should succeed (normal range inputs)
    assert!(ok_count > 0, "at least some FK calls should succeed: ok={ok_count}, err={err_count}");
    eprintln!("proptest FK: {ok_count} ok, {err_count} err out of 1000");
}

#[test]
fn proptest_ik_never_panics_200() {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let robot = load_robot("ur5e");
    let chain = load_chain(&robot);
    let mut rng = ChaCha8Rng::seed_from_u64(99);

    for _ in 0..200 {
        let target = Pose::from_xyz(
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-1.0..2.0),
        );
        let config = IKConfig { num_restarts: 1, max_iterations: 20, ..Default::default() };
        let _ = solve_ik(&robot, &chain, &target, &config);
    }
}

// ─── check_limits never panics with extreme inputs ──────────────────────────

#[test]
fn check_limits_extreme_inputs_no_panic() {
    let robot = load_robot("ur5e");
    let extremes: Vec<Vec<f64>> = vec![
        vec![f64::NAN; 6],
        vec![f64::INFINITY; 6],
        vec![f64::NEG_INFINITY; 6],
        vec![f64::MAX; 6],
        vec![f64::MIN; 6],
        vec![0.0; 6],
        vec![], // empty
    ];

    for vals in &extremes {
        let jv = JointValues::new(vals.clone());
        let _ = robot.check_limits(&jv); // must not panic
    }
}
