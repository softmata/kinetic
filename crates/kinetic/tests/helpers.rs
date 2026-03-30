//! Shared test helpers for kinetic acceptance tests.
//!
//! Include in acceptance test files via:
//! ```ignore
//! #[path = "helpers.rs"]
//! mod helpers;
//! use helpers::*;
//! ```

#![allow(dead_code)]

use kinetic::kinematics::KinematicChain;
use kinetic::robot::Robot;

// ─── Robot Lists ────────────────────────────────────────────────────────────

/// All 52 robots: (name, expected_dof)
pub const ALL_ROBOTS: &[(&str, usize)] = &[
    ("ur3e", 6), ("ur5e", 6), ("ur10e", 6), ("ur16e", 6), ("ur20", 6), ("ur30", 6),
    ("franka_panda", 7),
    ("kuka_iiwa7", 7), ("kuka_iiwa14", 7), ("kuka_kr6", 6),
    ("abb_irb1200", 6), ("abb_irb4600", 6), ("abb_yumi_left", 7), ("abb_yumi_right", 7),
    ("fanuc_crx10ia", 6), ("fanuc_lr_mate_200id", 6),
    ("yaskawa_gp7", 6), ("yaskawa_hc10", 6),
    ("kinova_gen3", 7), ("kinova_gen3_lite", 6), ("jaco2_6dof", 6),
    ("xarm5", 5), ("xarm6", 6), ("xarm7", 7),
    ("sawyer", 7), ("baxter_left", 7), ("baxter_right", 7),
    ("aloha_left", 6), ("aloha_right", 6),
    ("dobot_cr5", 6), ("flexiv_rizon4", 7), ("meca500", 6),
    ("mycobot_280", 6), ("techman_tm5_700", 6), ("elite_ec66", 6),
    ("niryo_ned2", 6), ("denso_vs068", 6), ("staubli_tx260", 6),
    ("viperx_300", 5), ("widowx_250", 5),
    ("trossen_px100", 4), ("trossen_rx150", 5), ("trossen_wx250s", 5),
    ("fetch", 8), ("tiago", 7), ("pr2", 7), ("stretch_re2", 4),
    ("so_arm100", 5), ("koch_v1", 6),
    ("open_manipulator_x", 4), ("lerobot_so100", 5), ("robotis_open_manipulator_p", 6),
];

/// 5 representative robots for expensive parameterized tests.
pub const SAFETY_ROBOTS: &[&str] = &[
    "ur5e", "franka_panda", "kuka_iiwa7", "xarm6", "kinova_gen3",
];

// ─── Robot Loading ──────────────────────────────────────────────────────────

pub fn load_robot(name: &str) -> Robot {
    Robot::from_name(name)
        .unwrap_or_else(|e| panic!("failed to load robot '{}': {}", name, e))
}

pub fn load_chain(robot: &Robot) -> KinematicChain {
    if let Some((_, group)) = robot.groups.iter().next() {
        KinematicChain::extract(robot, &group.base_link, &group.tip_link)
            .unwrap_or_else(|e| panic!("failed to extract chain: {}", e))
    } else {
        KinematicChain::auto_detect(robot)
            .unwrap_or_else(|e| panic!("failed to auto-detect chain: {}", e))
    }
}

// ─── Configuration Generators ───────────────────────────────────────────────

pub fn mid_joints(robot: &Robot) -> Vec<f64> {
    robot.joint_limits.iter().map(|l| (l.lower + l.upper) / 2.0).collect()
}

pub fn zero_joints(robot: &Robot) -> Vec<f64> {
    vec![0.0; robot.dof]
}

pub fn random_joints(robot: &Robot, seed: u64) -> Vec<f64> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    robot.joint_limits.iter().map(|l| {
        let range = l.upper - l.lower;
        if range.is_finite() && range < 100.0 {
            rng.gen_range(l.lower..=l.upper)
        } else {
            rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
        }
    }).collect()
}

pub fn boundary_joints(robot: &Robot, which: &str) -> Vec<f64> {
    match which {
        "lower" => robot.joint_limits.iter().map(|l| l.lower).collect(),
        "upper" => robot.joint_limits.iter().map(|l| l.upper).collect(),
        _ => mid_joints(robot),
    }
}

// ─── Assertion Helpers ──────────────────────────────────────────────────────

pub fn assert_within_limits(robot: &Robot, joints: &[f64], context: &str) {
    for (j, &val) in joints.iter().enumerate() {
        if j >= robot.joint_limits.len() { break; }
        let lo = robot.joint_limits[j].lower;
        let hi = robot.joint_limits[j].upper;
        assert!(
            val >= lo - 1e-6 && val <= hi + 1e-6,
            "{}: joint {} = {:.6} outside [{:.4}, {:.4}]",
            context, j, val, lo, hi
        );
    }
}

// ─── Recording Sink ─────────────────────────────────────────────────────────

pub struct RecordingSink {
    pub commands: Vec<(Vec<f64>, Vec<f64>)>,
}

impl RecordingSink {
    pub fn new() -> Self { Self { commands: vec![] } }
    pub fn count(&self) -> usize { self.commands.len() }
}

impl kinetic::execution::CommandSink for RecordingSink {
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> std::result::Result<(), String> {
        self.commands.push((positions.to_vec(), velocities.to_vec()));
        Ok(())
    }
}
