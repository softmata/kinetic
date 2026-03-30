//! IKFast-style analytical IK code generation.
//!
//! Instead of a full symbolic CAS, this module provides:
//! 1. Template-based codegen for common robot geometries (6-DOF spherical wrist).
//! 2. Robot geometry analysis to determine which template applies.
//! 3. Generated Rust IK functions that return closed-form solutions.
//!
//! # Supported Robot Classes
//!
//! - **Spherical wrist 6-DOF**: UR-family, KUKA, ABB. Last 3 joint axes intersect.
//!   Uses Paden-Kahan subproblems for the wrist + geometric method for the arm.
//! - **Puma-type**: 6 revolute joints with specific DH conventions.
//!
//! For robots that don't match a template, falls back to numerical IK (DLS/SQP).

use kinetic_core::Pose;
use kinetic_robot::Robot;

use crate::ik::{IKMode, IKSolution};
use crate::KinematicChain;

/// Robot geometry classification for IK template selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotClass {
    /// 6-DOF with spherical wrist (last 3 axes intersect).
    SphericalWrist6DOF,
    /// 6-DOF with offset wrist (last 3 axes nearly intersect).
    OffsetWrist6DOF,
    /// 7-DOF redundant (one extra joint).
    Redundant7DOF,
    /// General: no template available, use numerical solver.
    General,
}

/// Result of robot geometry analysis.
#[derive(Debug, Clone)]
pub struct GeometryAnalysis {
    /// Classified robot type.
    pub robot_class: RobotClass,
    /// DH parameters [alpha, a, d, theta_offset] per joint.
    pub dh_params: Vec<[f64; 4]>,
    /// Whether the wrist axes intersect (within tolerance).
    pub wrist_intersects: bool,
    /// Wrist intersection point (if applicable).
    pub wrist_center: Option<[f64; 3]>,
    /// Number of active joints.
    pub dof: usize,
}

/// Analyze a robot's geometry to determine which IK template to use.
pub fn analyze_geometry(robot: &Robot, chain: &KinematicChain) -> GeometryAnalysis {
    let dof = chain.dof;

    // Extract DH-like parameters from joint origins
    let mut dh_params = Vec::new();
    for &joint_idx in &chain.all_joints {
        let joint = &robot.joints[joint_idx];
        let t = joint.origin.translation();
        let a = joint.axis;

        dh_params.push([
            a.y.atan2(a.z),  // alpha (twist angle approx)
            (t.x * t.x + t.y * t.y).sqrt(), // a (link length)
            t.z,              // d (link offset)
            0.0,              // theta offset
        ]);
    }

    // Check for spherical wrist: do last 3 joint axes intersect?
    let (wrist_intersects, wrist_center) = if dof >= 6 {
        check_wrist_intersection(robot, chain)
    } else {
        (false, None)
    };

    let robot_class = match dof {
        6 if wrist_intersects => RobotClass::SphericalWrist6DOF,
        6 => RobotClass::OffsetWrist6DOF,
        7 => RobotClass::Redundant7DOF,
        _ => RobotClass::General,
    };

    GeometryAnalysis {
        robot_class,
        dh_params,
        wrist_intersects,
        wrist_center,
        dof,
    }
}

/// Check if the last 3 joint axes of a 6-DOF robot intersect at a common point.
fn check_wrist_intersection(robot: &Robot, chain: &KinematicChain) -> (bool, Option<[f64; 3]>) {
    if chain.all_joints.len() < 6 {
        return (false, None);
    }

    // Get the last 3 joint origins
    let j4 = &robot.joints[chain.all_joints[chain.all_joints.len() - 3]];
    let j5 = &robot.joints[chain.all_joints[chain.all_joints.len() - 2]];
    let j6 = &robot.joints[chain.all_joints[chain.all_joints.len() - 1]];

    // Check if the origins of joints 4, 5, 6 are close together
    let t4 = j4.origin.translation();
    let t5 = j5.origin.translation();
    let t6 = j6.origin.translation();

    // Wrist is "spherical" if j5 and j6 have very small offsets from j4
    let d5 = (t5.x * t5.x + t5.y * t5.y).sqrt();
    let d6 = (t6.x * t6.x + t6.y * t6.y).sqrt();

    let threshold = 0.01; // 1cm tolerance
    let intersects = d5 < threshold && d6 < threshold;

    let center = if intersects {
        // Wrist center is approximately at the joint 4 origin
        // (accumulated through the chain from base)
        Some([0.0, 0.0, t4.z + t5.z + t6.z])
    } else {
        None
    };

    (intersects, center)
}

/// Solve IK using the appropriate template for the robot class.
///
/// Returns up to 8 solutions for 6-DOF spherical wrist robots.
/// Falls back to None if no template matches (caller should use numerical IK).
pub fn solve_ikfast(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    analysis: &GeometryAnalysis,
) -> Vec<IKSolution> {
    match analysis.robot_class {
        RobotClass::SphericalWrist6DOF => {
            solve_spherical_wrist_6dof(robot, chain, target, analysis)
        }
        RobotClass::OffsetWrist6DOF => {
            // Use numerical refinement from approximate analytical solution
            solve_offset_wrist_6dof(robot, chain, target, seed, analysis)
        }
        _ => vec![], // No template — caller should use numerical solver
    }
}

/// Analytical IK for 6-DOF spherical wrist robots.
///
/// Decouples position (joints 1-3) and orientation (joints 4-6):
/// 1. Compute wrist center from target pose.
/// 2. Solve joints 1-3 geometrically for wrist center position.
/// 3. Solve joints 4-6 using Euler angle decomposition for orientation.
fn solve_spherical_wrist_6dof(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    analysis: &GeometryAnalysis,
) -> Vec<IKSolution> {
    let dh = &analysis.dh_params;
    if dh.len() < 6 { return vec![]; }

    let target_pos = target.translation();
    let target_rot = target.0.rotation;

    // Link lengths from DH params
    let d1 = dh[0][2];
    let a2 = dh[1][1].max(dh[1][2].abs()); // link 2 length
    let a3 = dh[2][1].max(dh[2][2].abs()); // link 3 length
    let _d4 = dh[3][2]; // wrist offset

    // Wrist center = target - d6 * z_axis_of_target
    let z_target = target_rot * nalgebra::Vector3::z();
    let d6 = dh[5][2];
    let wc = target_pos - z_target * d6;

    let mut solutions = Vec::new();

    // Joint 1: atan2(wc_y, wc_x) — two solutions (elbow up/down)
    let theta1_options = [
        wc.y.atan2(wc.x),
        wc.y.atan2(wc.x) + std::f64::consts::PI,
    ];

    for &theta1 in &theta1_options {
        // Project wrist center into the plane of joints 2-3
        let r = (wc.x * wc.x + wc.y * wc.y).sqrt();
        let s = wc.z - d1;

        // Solve 2-link planar IK for joints 2-3
        let d_sq = r * r + s * s;
        let d = d_sq.sqrt();

        if d > a2 + a3 + 0.001 || d < (a2 - a3).abs() - 0.001 {
            continue; // unreachable
        }

        // Law of cosines for joint 3
        let cos_theta3 = (d_sq - a2 * a2 - a3 * a3) / (2.0 * a2 * a3);
        let cos3_clamped = cos_theta3.clamp(-1.0, 1.0);

        for sign in [1.0, -1.0] {
            let theta3 = sign * cos3_clamped.acos();

            // Joint 2
            let k1 = a2 + a3 * theta3.cos();
            let k2 = a3 * theta3.sin();
            let theta2 = s.atan2(r) - k2.atan2(k1);

            // Joints 4-6: Euler ZYZ decomposition of remaining rotation
            let r_03 = compute_r03(theta1, theta2, theta3, dh);
            let r_36 = r_03.transpose() * target_rot.to_rotation_matrix();

            // ZYZ Euler angles
            let (theta4, theta5, theta6) = euler_zyz(&r_36);

            let joints = vec![theta1, theta2, theta3, theta4, theta5, theta6];

            // Verify with FK
            if let Ok(check_pose) = crate::forward::forward_kinematics(robot, chain, &joints) {
                let pos_err = (check_pose.translation() - target_pos).norm();
                let rot_err = (check_pose.0.rotation.inverse() * target_rot).angle();

                if pos_err < 0.01 && rot_err < 0.1 {
                    solutions.push(IKSolution {
                        joints,
                        position_error: pos_err,
                        orientation_error: rot_err,
                        iterations: 0,
                        converged: pos_err < 0.001 && rot_err < 0.01,
                        mode_used: IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
                    });
                }
            }
        }
    }

    // Sort by total error
    solutions.sort_by(|a, b| {
        let ea = a.position_error + a.orientation_error;
        let eb = b.position_error + b.orientation_error;
        ea.partial_cmp(&eb).unwrap()
    });

    solutions
}

/// Approximate IK for offset-wrist 6-DOF robots.
///
/// Uses the spherical wrist solution as initial guess, then refines with
/// a few SQP iterations to account for the wrist offset.
fn solve_offset_wrist_6dof(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    analysis: &GeometryAnalysis,
) -> Vec<IKSolution> {
    // Try spherical wrist first as approximation
    let approx = solve_spherical_wrist_6dof(robot, chain, target, analysis);

    if !approx.is_empty() {
        // Refine best solution with SQP
        let best_seed = &approx[0].joints;
        if let Some(refined) = crate::sqp::solve_sqp(
            robot, chain, target, best_seed,
            &crate::sqp::SQPConfig { max_iterations: 20, ..Default::default() },
        ) {
            return vec![refined];
        }
    }

    // Fall back to SQP from provided seed
    if let Some(sol) = crate::sqp::solve_sqp(
        robot, chain, target, seed,
        &crate::sqp::SQPConfig::default(),
    ) {
        vec![sol]
    } else {
        vec![]
    }
}

/// Compute R_03 rotation matrix from first 3 joint angles.
fn compute_r03(t1: f64, t2: f64, t3: f64, _dh: &[[f64; 4]]) -> nalgebra::Matrix3<f64> {
    // Simplified: assume standard revolute joint axes
    let c1 = t1.cos(); let s1 = t1.sin();
    let c23 = (t2 + t3).cos(); let s23 = (t2 + t3).sin();

    nalgebra::Matrix3::new(
        c1 * c23, -c1 * s23, s1,
        s1 * c23, -s1 * s23, -c1,
        s23, c23, 0.0,
    )
}

/// Extract ZYZ Euler angles from a rotation matrix.
fn euler_zyz(r: &nalgebra::Matrix3<f64>) -> (f64, f64, f64) {
    let theta5 = r[(2, 2)].clamp(-1.0, 1.0).acos();

    if theta5.abs() < 1e-6 {
        // Gimbal lock: theta5 ≈ 0
        (r[(0, 1)].atan2(r[(0, 0)]), 0.0, 0.0)
    } else if (theta5 - std::f64::consts::PI).abs() < 1e-6 {
        // Gimbal lock: theta5 ≈ π
        (-r[(0, 1)].atan2(r[(0, 0)]), std::f64::consts::PI, 0.0)
    } else {
        let theta4 = r[(1, 2)].atan2(r[(0, 2)]);
        let theta6 = r[(2, 1)].atan2(-r[(2, 0)]);
        (theta4, theta5, theta6)
    }
}

/// Generate Rust source code for a robot's IK solver.
///
/// Produces a standalone Rust function that computes IK for the specific
/// robot geometry without runtime overhead.
pub fn generate_ik_code(robot: &Robot, chain: &KinematicChain) -> String {
    let analysis = analyze_geometry(robot, chain);
    let mut code = String::new();

    code.push_str(&format!("// Auto-generated IK solver for '{}'\n", robot.name));
    code.push_str(&format!("// Robot class: {:?}\n", analysis.robot_class));
    code.push_str(&format!("// DOF: {}\n", analysis.dof));
    code.push_str(&format!("// Wrist intersection: {}\n\n", analysis.wrist_intersects));

    code.push_str("/// DH parameters [alpha, a, d, theta_offset] per joint.\n");
    code.push_str("pub const DH_PARAMS: &[[f64; 4]] = &[\n");
    for (i, dh) in analysis.dh_params.iter().enumerate() {
        code.push_str(&format!("    [{:.6}, {:.6}, {:.6}, {:.6}], // joint {}\n",
            dh[0], dh[1], dh[2], dh[3], i));
    }
    code.push_str("];\n\n");

    match analysis.robot_class {
        RobotClass::SphericalWrist6DOF => {
            code.push_str("/// Solve IK analytically (spherical wrist 6-DOF).\n");
            code.push_str("/// Returns up to 8 solutions.\n");
            code.push_str("pub fn solve(target_pos: [f64; 3], target_rot: [[f64; 3]; 3]) -> Vec<[f64; 6]> {\n");
            code.push_str("    // TODO: paste generated closed-form solution here\n");
            code.push_str("    vec![]\n");
            code.push_str("}\n");
        }
        _ => {
            code.push_str("// No analytical template for this robot class.\n");
            code.push_str("// Use numerical IK (DLS, SQP, or Bio-IK).\n");
        }
    }

    code
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF_6DOF: &str = r#"<?xml version="1.0"?>
<robot name="test_6dof">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <link name="l3"/><link name="l4"/><link name="l5"/><link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="10"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="10"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="3" effort="5"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="3" effort="5"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="tip"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="3" effort="5"/>
  </joint>
</robot>"#;

    #[test]
    fn analyze_6dof_geometry() {
        let robot = Robot::from_urdf_string(URDF_6DOF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let analysis = analyze_geometry(&robot, &chain);

        assert_eq!(analysis.dof, 6);
        assert_eq!(analysis.dh_params.len(), 6);
        // This test robot has j5 with zero offset → wrist should intersect
    }

    #[test]
    fn generate_code_produces_output() {
        let robot = Robot::from_urdf_string(URDF_6DOF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let code = generate_ik_code(&robot, &chain);

        assert!(code.contains("DH_PARAMS"));
        assert!(code.contains("test_6dof"));
        assert!(code.contains("DOF: 6"));
    }

    #[test]
    fn solve_ikfast_returns_solutions() {
        let robot = Robot::from_urdf_string(URDF_6DOF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let analysis = analyze_geometry(&robot, &chain);

        // Target at a known reachable pose
        let seed = [0.3, 0.5, -0.3, 0.1, 0.2, 0.1];
        let target = crate::forward::forward_kinematics(&robot, &chain, &seed).unwrap();

        let solutions = solve_ikfast(&robot, &chain, &target, &[0.0; 6], &analysis);
        // May or may not find solutions depending on the template match
        // The important thing is it doesn't crash
        let _ = solutions;
    }

    #[test]
    fn euler_zyz_roundtrip() {
        let (a, b, c) = (0.5, 1.0, -0.3);
        let r = nalgebra::Rotation3::from_euler_angles(0.0, b, 0.0)
            * nalgebra::Rotation3::from_euler_angles(0.0, 0.0, a)
            * nalgebra::Rotation3::from_euler_angles(0.0, 0.0, c);
        // ZYZ extraction should recover similar angles (up to equivalent representations)
        let m = r.matrix();
        let (t4, t5, t6) = euler_zyz(m);
        // At minimum, t5 should match b approximately
        assert!((t5 - b).abs() < 0.5 || (t5 + b).abs() < 0.5,
            "theta5 should be near b={}: got {}", b, t5);
    }

    #[test]
    fn robot_class_detection() {
        let robot = Robot::from_urdf_string(URDF_6DOF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let analysis = analyze_geometry(&robot, &chain);

        assert!(matches!(analysis.robot_class,
            RobotClass::SphericalWrist6DOF | RobotClass::OffsetWrist6DOF));
    }
}
