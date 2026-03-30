# Test Conventions & Shared Infrastructure

## Crate Structure

All acceptance tests live in `crates/kinetic/tests/acceptance/`. The `kinetic` meta-crate
re-exports everything, so tests import from `kinetic::*`.

## Required Dependencies (add to kinetic's dev-dependencies)

```toml
[dev-dependencies]
rand = "0.8"
rand_chacha = "0.3"
proptest = "1.4"
approx = "0.5"
```

## Standard Imports

Every test file starts with:

```rust
use kinetic::prelude::*;
use kinetic::kinematics::{
    forward_kinematics, fk, fk_all_links, solve_ik,
    IKConfig, IKSolution, IKSolver, IKMode, NullSpace,
    KinematicChain, jacobian, manipulability,
};
use kinetic::robot::Robot;
use kinetic::collision::{RobotSphereModel, SphereGenConfig};
use kinetic::scene::Scene;
use kinetic::planning::{Planner, PlannerConfig, PlannerType, PlanningResult};
use kinetic::trajectory::{
    trapezoidal, trapezoidal_per_joint, totp, jerk_limited,
    cubic_spline_time, blend, TrajectoryValidator, ValidationConfig,
    ViolationType, TimedTrajectory,
};
use kinetic::execution::{
    CommandSink, FeedbackSource, TrajectoryExecutor,
    RealTimeExecutor, SimExecutor, ExecutionConfig, ExecutionResult,
    ExecutionState, ExecutionError,
};
use kinetic::reactive::{Servo, ServoConfig, InputType, RMP, PolicyType, JointCommand};
use kinetic::core::{
    Pose, JointValues, Goal, Constraint, KineticError,
    Trajectory, Waypoint, TimedWaypoint,
};

use std::sync::Arc;
use std::time::Duration;
use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_4};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
```

## helpers.rs — Shared Test Utilities

Create this file with the following helpers. All test files import from here.

```rust
// File: crates/kinetic/tests/acceptance/helpers.rs

use kinetic::prelude::*;
use kinetic::kinematics::{forward_kinematics, KinematicChain, IKConfig, solve_ik};
use kinetic::robot::Robot;
use std::sync::Arc;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ─── All 52 Robot Names ───────────────────────────────────────────
// Format: (name, total_dof, arm_chain_dof)
pub const ALL_ROBOTS: &[(&str, usize, usize)] = &[
    // UR family
    ("ur3e", 6, 6), ("ur5e", 6, 6), ("ur10e", 6, 6),
    ("ur16e", 6, 6), ("ur20", 6, 6), ("ur30", 6, 6),
    // Franka
    ("franka_panda", 7, 7),
    // KUKA
    ("kuka_iiwa7", 7, 7), ("kuka_iiwa14", 7, 7), ("kuka_kr6", 6, 6),
    // ABB
    ("abb_irb1200", 6, 6), ("abb_irb4600", 6, 6),
    ("abb_yumi_left", 7, 7), ("abb_yumi_right", 7, 7),
    // Fanuc
    ("fanuc_crx10ia", 6, 6), ("fanuc_lr_mate_200id", 6, 6),
    // Yaskawa
    ("yaskawa_gp7", 6, 6), ("yaskawa_hc10", 6, 6),
    // Kinova
    ("kinova_gen3", 7, 7), ("kinova_gen3_lite", 6, 6), ("jaco2_6dof", 6, 6),
    // xArm
    ("xarm5", 5, 5), ("xarm6", 6, 6), ("xarm7", 7, 7),
    // Rethink
    ("sawyer", 7, 7), ("baxter_left", 7, 7), ("baxter_right", 7, 7),
    // ALOHA bimanual
    ("aloha_left", 6, 6), ("aloha_right", 6, 6),
    // Other cobots
    ("dobot_cr5", 6, 6), ("flexiv_rizon4", 7, 7), ("meca500", 6, 6),
    ("mycobot_280", 6, 6), ("techman_tm5_700", 6, 6), ("elite_ec66", 6, 6),
    ("niryo_ned2", 6, 6), ("denso_vs068", 6, 6), ("staubli_tx260", 6, 6),
    // Trossen
    ("viperx_300", 5, 5), ("widowx_250", 5, 5),
    ("trossen_px100", 4, 4), ("trossen_rx150", 5, 5), ("trossen_wx250s", 5, 5),
    // Mobile manipulators (total_dof > arm_dof)
    ("fetch", 8, 7), ("tiago", 7, 7), ("pr2", 7, 7), ("stretch_re2", 4, 4),
    // Open-source / education
    ("so_arm100", 5, 5), ("koch_v1", 6, 6),
    ("open_manipulator_x", 4, 4), ("lerobot_so100", 5, 5),
    ("robotis_open_manipulator_p", 6, 6),
];

// ─── Robot Subsets by DOF ─────────────────────────────────────────
pub const ROBOTS_6DOF: &[&str] = &[
    "ur3e", "ur5e", "ur10e", "ur16e", "ur20", "ur30",
    "kuka_kr6", "abb_irb1200", "abb_irb4600",
    "fanuc_crx10ia", "fanuc_lr_mate_200id",
    "yaskawa_gp7", "yaskawa_hc10",
    "kinova_gen3_lite", "jaco2_6dof",
    "xarm6", "aloha_left", "aloha_right",
    "dobot_cr5", "meca500", "mycobot_280",
    "techman_tm5_700", "elite_ec66", "niryo_ned2",
    "denso_vs068", "staubli_tx260", "koch_v1",
    "robotis_open_manipulator_p",
];

pub const ROBOTS_7DOF: &[&str] = &[
    "franka_panda", "kuka_iiwa7", "kuka_iiwa14",
    "abb_yumi_left", "abb_yumi_right",
    "kinova_gen3", "xarm7", "sawyer",
    "baxter_left", "baxter_right", "flexiv_rizon4",
    "tiago", "pr2",
];

pub const ROBOTS_5DOF: &[&str] = &[
    "xarm5", "viperx_300", "widowx_250",
    "trossen_rx150", "trossen_wx250s",
    "so_arm100", "lerobot_so100",
];

pub const ROBOTS_4DOF: &[&str] = &[
    "trossen_px100", "open_manipulator_x", "stretch_re2",
];

pub const BIMANUAL_PAIRS: &[(&str, &str)] = &[
    ("aloha_left", "aloha_right"),
    ("baxter_left", "baxter_right"),
    ("abb_yumi_left", "abb_yumi_right"),
];

// Robots known to have collision geometry in their URDFs
pub const ROBOTS_WITH_COLLISION: &[&str] = &[
    "ur5e", "ur10e", "franka_panda", "kuka_iiwa7", "kuka_iiwa14",
    "kinova_gen3", "xarm6", "xarm7",
];

// Representative subset for expensive tests (1 per DOF class + 1 industrial)
pub const REPRESENTATIVE_ROBOTS: &[&str] = &[
    "trossen_px100",   // 4-DOF
    "xarm5",           // 5-DOF
    "ur5e",            // 6-DOF industrial
    "franka_panda",    // 7-DOF research
    "fetch",           // mobile manipulator
];

// ─── Deterministic RNG ────────────────────────────────────────────
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

// ─── Robot Loading Helpers ────────────────────────────────────────
pub fn load_robot(name: &str) -> Arc<Robot> {
    Arc::new(Robot::from_name(name).unwrap_or_else(|e| {
        panic!("Failed to load robot '{}': {}", name, e)
    }))
}

pub fn load_robot_and_chain(name: &str) -> (Arc<Robot>, KinematicChain) {
    let robot = load_robot(name);
    let chain = KinematicChain::auto_detect(&robot).unwrap_or_else(|e| {
        panic!("Failed to extract chain for '{}': {}", name, e)
    });
    (robot, chain)
}

// ─── Joint Generation ─────────────────────────────────────────────
/// Random joint values within limits, with `margin` fraction inset from edges.
/// margin=0.05 means 5% away from each limit boundary.
pub fn random_joints_within_limits(
    robot: &Robot,
    chain: &KinematicChain,
    rng: &mut impl Rng,
    margin: f64,
) -> Vec<f64> {
    chain.active_joints.iter().map(|&ji| {
        let joint = &robot.joints[ji];
        if let Some(ref limits) = joint.limits {
            let range = limits.upper - limits.lower;
            let lo = limits.lower + range * margin;
            let hi = limits.upper - range * margin;
            if lo < hi {
                rng.gen_range(lo..hi)
            } else {
                (limits.lower + limits.upper) / 2.0
            }
        } else {
            rng.gen_range(-PI..PI)
        }
    }).collect()
}

/// Joint values AT the limits (alternating lower/upper).
pub fn joints_at_limits(robot: &Robot, chain: &KinematicChain) -> Vec<Vec<f64>> {
    let dof = chain.dof;
    let mut configs = Vec::new();

    // All lower limits
    let lower: Vec<f64> = chain.active_joints.iter().map(|&ji| {
        robot.joints[ji].limits.as_ref().map_or(-PI, |l| l.lower)
    }).collect();
    configs.push(lower);

    // All upper limits
    let upper: Vec<f64> = chain.active_joints.iter().map(|&ji| {
        robot.joints[ji].limits.as_ref().map_or(PI, |l| l.upper)
    }).collect();
    configs.push(upper);

    // Each joint at its lower limit, others at mid
    for i in 0..dof {
        let mut config = mid_joints(robot, chain);
        config[i] = chain.active_joints.get(i).map(|&ji| {
            robot.joints[ji].limits.as_ref().map_or(-PI, |l| l.lower)
        }).unwrap_or(-PI);
        configs.push(config);
    }

    configs
}

/// Joint values at the midpoint of each limit range.
pub fn mid_joints(robot: &Robot, chain: &KinematicChain) -> Vec<f64> {
    chain.active_joints.iter().map(|&ji| {
        let joint = &robot.joints[ji];
        if let Some(ref limits) = joint.limits {
            (limits.lower + limits.upper) / 2.0
        } else {
            0.0
        }
    }).collect()
}

// ─── Assertion Helpers ────────────────────────────────────────────
/// Assert all values are finite (not NaN, not Inf).
pub fn assert_all_finite(values: &[f64], context: &str) {
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{}: value[{}] = {} is not finite", context, i, v);
    }
}

/// Assert joint values are within robot limits (with epsilon tolerance).
pub fn assert_within_limits(
    robot: &Robot,
    chain: &KinematicChain,
    joints: &[f64],
    epsilon: f64,
    context: &str,
) {
    for (i, &ji) in chain.active_joints.iter().enumerate() {
        if let Some(ref limits) = robot.joints[ji].limits {
            let name = &robot.joints[ji].name;
            assert!(
                joints[i] >= limits.lower - epsilon && joints[i] <= limits.upper + epsilon,
                "{}: joint '{}' (idx {}) = {:.6} outside limits [{:.6}, {:.6}] (eps={})",
                context, name, i, joints[i], limits.lower, limits.upper, epsilon
            );
        }
    }
}

/// Assert two poses are close.
pub fn assert_poses_close(
    a: &Pose, b: &Pose,
    pos_tol: f64, ori_tol: f64,
    context: &str,
) {
    let pos_err = a.translation_distance(b);
    let ori_err = a.rotation_distance(b);
    assert!(
        pos_err < pos_tol,
        "{}: position error {:.6} m exceeds tolerance {:.6} m", context, pos_err, pos_tol
    );
    assert!(
        ori_err < ori_tol,
        "{}: orientation error {:.6} rad exceeds tolerance {:.6} rad", context, ori_err, ori_tol
    );
}

/// Assert a trajectory has monotonically increasing timestamps.
pub fn assert_trajectory_monotonic(traj: &TimedTrajectory, context: &str) {
    for (i, pair) in traj.waypoints.windows(2).enumerate() {
        assert!(
            pair[1].time >= pair[0].time,
            "{}: timestamps not monotonic at index {}: {} > {}",
            context, i, pair[0].time, pair[1].time
        );
    }
}

// ─── Tolerances ───────────────────────────────────────────────────
pub mod tol {
    /// Position accuracy for IK solutions (meters)
    pub const IK_POSITION: f64 = 1e-3;        // 1mm
    /// Orientation accuracy for IK solutions (radians)
    pub const IK_ORIENTATION: f64 = 1e-2;     // ~0.57 degrees
    /// Tight position accuracy for analytical solvers
    pub const IK_POSITION_TIGHT: f64 = 1e-4;  // 0.1mm
    /// Tight orientation accuracy for analytical solvers
    pub const IK_ORIENTATION_TIGHT: f64 = 1e-3; // ~0.057 degrees
    /// Joint limit enforcement epsilon
    pub const JOINT_LIMIT_EPS: f64 = 1e-6;
    /// FK numerical precision
    pub const FK_PRECISION: f64 = 1e-10;
    /// Jacobian finite-difference vs analytical tolerance
    pub const JACOBIAN_TOL: f64 = 1e-5;
    /// Trajectory position jump threshold (rad)
    pub const TRAJ_MAX_JUMP: f64 = 0.5;
    /// SIMD vs scalar collision distance tolerance
    pub const SIMD_SCALAR_TOL: f64 = 1e-8;
    /// Velocity limit safety factor (5% headroom)
    pub const VEL_SAFETY_FACTOR: f64 = 1.05;
}

// ─── Test Counting ────────────────────────────────────────────────
/// Run a parameterized test across all robots, collecting failures.
/// Returns (passed, failed, total).
pub fn run_for_all_robots<F: Fn(&str, usize, usize) -> Result<(), String>>(
    test_fn: F,
) -> (usize, usize, usize) {
    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();

    for &(name, total_dof, arm_dof) in ALL_ROBOTS {
        match test_fn(name, total_dof, arm_dof) {
            Ok(()) => passed += 1,
            Err(msg) => {
                failed += 1;
                failures.push(format!("  {} — {}", name, msg));
            }
        }
    }

    if !failures.is_empty() {
        eprintln!("\nFailed robots ({}/{}):", failed, passed + failed);
        for f in &failures {
            eprintln!("{}", f);
        }
    }

    (passed, failed, passed + failed)
}

/// Same but for a robot subset.
pub fn run_for_robots<F: Fn(&str) -> Result<(), String>>(
    robots: &[&str],
    test_fn: F,
) -> (usize, usize, usize) {
    let mut passed = 0;
    let mut failed = 0;

    for &name in robots {
        match test_fn(name) {
            Ok(()) => passed += 1,
            Err(msg) => {
                failed += 1;
                eprintln!("  FAIL {}: {}", name, msg);
            }
        }
    }

    (passed, failed, passed + failed)
}
```

## Naming Convention

- Test functions: `p0_category_specific_thing` or `p1_category_specific_thing`
- The `p0_`/`p1_`/`p2_` prefix allows filtering by priority via `cargo test -- p0_`
- Parameterized tests use the `run_for_all_robots` helper and assert 0 failures

## Test Attributes

```rust
#[test]
fn p0_fk_roundtrip_all_robots() { ... }

#[test]
#[ignore = "slow: runs 26000 IK solves"]
fn p0_ik_exhaustive_all_solvers_all_robots() { ... }
```

Mark tests that take >30s as `#[ignore]`. CI runs them with `cargo test -- --include-ignored`.

## Error Handling in Tests

Tests must distinguish between:
1. **Expected failure** — planner correctly returns `Err(GoalUnreachable)` → test PASSES
2. **Unexpected failure** — planner panics or returns wrong error → test FAILS
3. **Silent wrong answer** — planner returns `Ok` but trajectory is invalid → test FAILS (worst case)

Always validate the CONTENTS of `Ok` results, not just that they're `Ok`.
