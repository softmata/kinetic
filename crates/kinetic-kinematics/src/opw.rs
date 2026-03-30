//! OPW (Ortho-Parallel-Wrist) analytical IK for 6-DOF spherical wrist robots.
//!
//! Provides closed-form inverse kinematics for common industrial robot
//! geometries: 6 revolute joints where the last 3 axes approximately
//! intersect at a wrist center point. Covers UR, xArm6, and similar arms.
//!
//! # Algorithm
//!
//! 1. Compute wrist center from target pose using a pre-computed tool offset.
//! 2. Position IK: solve joints 1-3 using arm-plane geometry and cosine law.
//! 3. Orientation IK: compute R_03 via FK, then YZY Euler decomposition for
//!    joints 4-6 (matching Y,Z,Y wrist axis convention).
//! 4. Up to 2×2×2 = 8 solutions.
//!
//! # Performance
//!
//! Target: <50 µs per solve (returns all 8 solutions). Typical: <10 µs.

use std::f64::consts::PI;

use kinetic_core::{Pose, Result};
use kinetic_robot::Robot;

use crate::chain::KinematicChain;
use crate::forward::{forward_kinematics_all, forward_kinematics};
use crate::ik::{pose_error, IKConfig, IKMode, IKSolution};

/// OPW kinematic parameters for a 6-DOF robot.
///
/// These encode the arm geometry extracted from the URDF. The parameters
/// are defined in the arm plane after the shoulder (J1) rotation.
#[derive(Debug, Clone)]
pub struct OPWParameters {
    /// Height from base to shoulder (J2) axis along Z.
    pub shoulder_height: f64,
    /// Upper arm length: distance from J2 to J3 origin.
    pub upper_arm_length: f64,
    /// Forearm effective length: distance from J3 to wrist center (J4 origin).
    pub forearm_length: f64,
    /// Forearm angle offset: deviation of forearm from the pure arm-plane direction,
    /// caused by perpendicular offsets (d4). Computed as atan2(z_component, x_component).
    pub forearm_phi: f64,
    /// Offset from wrist center to tool0 in the tool frame at zero config.
    /// Used to compute wrist center from target pose.
    pub wrist_to_tool_in_tool: nalgebra::Vector3<f64>,
}

impl OPWParameters {
    /// Extract OPW parameters from a robot model and kinematic chain.
    ///
    /// Computes arm geometry by evaluating FK at zero configuration and
    /// measuring link-to-link distances.
    pub fn from_robot_chain(robot: &Robot, chain: &KinematicChain) -> Result<Self> {
        let zero = vec![0.0; chain.dof];
        let link_poses = forward_kinematics_all(robot, chain, &zero)?;

        // link_poses: [base, after_J1, after_J2, after_J3, after_J4, after_J5, after_J6, tool0]
        // For 6-DOF + fixed ee: 8 poses (7 joints including the fixed one)
        // Indices:
        //   0: base_link
        //   1: shoulder_link (after J1)
        //   2: upper_arm_link (after J2)
        //   3: forearm_link (after J3 = elbow)
        //   4: wrist_1_link (after J4 = wrist center)
        //   5: wrist_2_link (after J5)
        //   6: wrist_3_link (after J6)
        //   7: tool0 (after fixed joint)

        if link_poses.len() < 5 {
            return Err(kinetic_core::KineticError::IncompatibleKinematics(
                "OPW requires at least 5 link poses (6-DOF chain)".into(),
            ));
        }

        let p_shoulder = link_poses[1].translation();
        let p_elbow = link_poses[3].translation();
        let p_wrist = link_poses[4].translation(); // J4 origin = wrist center
        let p_tool = link_poses.last().unwrap().translation();
        let r_tool = link_poses.last().unwrap().rotation().to_rotation_matrix();

        let shoulder_height = p_shoulder.z;

        let upper_arm_length = (p_elbow - p_shoulder).norm();

        // Forearm vector from elbow to wrist center
        let forearm_vec = p_wrist - p_elbow;
        let forearm_length = forearm_vec.norm();

        // Forearm angle: in the arm plane (XZ after J1=0), the angle of the
        // forearm vector from the radial (-X) direction toward +Z.
        // At J1=0, J2=0, J3=0: the arm extends along -X.
        // forearm_vec at zero: (-a3, 0, d4) in world coords.
        let forearm_phi = forearm_vec.z.atan2(-forearm_vec.x);

        // Wrist center to tool offset in the tool frame
        let wrist_to_tool_world = p_tool - p_wrist;
        let wrist_to_tool_in_tool = r_tool.matrix().transpose() * wrist_to_tool_world;

        Ok(Self {
            shoulder_height,
            upper_arm_length,
            forearm_length,
            forearm_phi,
            wrist_to_tool_in_tool,
        })
    }

    /// Pre-built OPW parameters for the UR5e.
    pub fn ur5e(robot: &Robot, chain: &KinematicChain) -> Result<Self> {
        Self::from_robot_chain(robot, chain)
    }

    /// Pre-built OPW parameters for the UR10e.
    pub fn ur10e(robot: &Robot, chain: &KinematicChain) -> Result<Self> {
        Self::from_robot_chain(robot, chain)
    }

    /// Pre-built OPW parameters for the xArm6.
    pub fn xarm6(robot: &Robot, chain: &KinematicChain) -> Result<Self> {
        Self::from_robot_chain(robot, chain)
    }
}

/// Solve OPW analytical IK, returning up to 8 solutions.
///
/// Each solution is a `[f64; 6]` array of joint values. Solutions that
/// are geometrically impossible (NaN from acos/asin) are filtered out.
pub fn solve_opw(
    params: &OPWParameters,
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
) -> Vec<[f64; 6]> {
    let mut solutions = Vec::with_capacity(8);

    // Step 1: Compute wrist center from target pose.
    // wc = p_target - R_target * (tool_to_wrist_in_tool)
    // Note: wrist_to_tool_in_tool = p_tool - p_wrist, so tool_to_wrist = -wrist_to_tool
    let r_target = target.rotation().to_rotation_matrix();
    let tool_to_wrist = -params.wrist_to_tool_in_tool;
    let wc = target.translation() + r_target.matrix() * tool_to_wrist;

    let wx = wc.x;
    let wy = wc.y;
    let wz = wc.z;

    // Step 2: Solve J1 (shoulder pan) — two solutions
    let j1_options = solve_j1(wx, wy);

    for &j1 in &j1_options {
        // Step 3: Solve J2, J3 (shoulder lift + elbow) — two solutions per J1
        let j23_options = solve_j23(params, j1, wx, wy, wz);

        for &(j2, j3) in &j23_options {
            // Step 4: Solve J4, J5, J6 (wrist) using FK-based R_03
            let j456_options = solve_j456(robot, chain, j1, j2, j3, target);

            for &(j4, j5, j6) in &j456_options {
                let q = [j1, j2, j3, j4, j5, j6];

                // Filter out NaN solutions
                if q.iter().all(|v| v.is_finite()) {
                    solutions.push(q);
                }
            }
        }
    }

    solutions
}

/// Solve for joint 1 (base rotation toward wrist center).
fn solve_j1(wx: f64, wy: f64) -> Vec<f64> {
    // The arm extends in the direction determined by J1.
    // At J1=0, the arm extends along -X.
    // The wrist center is at (wx, wy) in the XY plane.
    // J1 = atan2(-wy, -wx) to point the arm toward the wrist center.
    let j1_a = (-wy).atan2(-wx);
    let j1_b = normalize_angle(j1_a + PI);
    vec![j1_a, j1_b]
}

/// Solve for joints 2 and 3 given joint 1 and wrist center position.
fn solve_j23(params: &OPWParameters, j1: f64, wx: f64, wy: f64, wz: f64) -> Vec<(f64, f64)> {
    let mut solutions = Vec::with_capacity(2);

    let c1j1 = j1.cos();
    let s1j1 = j1.sin();

    // Radial and height in arm plane
    let r = wx * c1j1 + wy * s1j1;
    let h = wz - params.shoulder_height;

    // Distance from shoulder to wrist center
    let d_sq = r * r + h * h;
    let d = d_sq.sqrt();

    let l1 = params.upper_arm_length;
    let l2 = params.forearm_length;
    let phi = params.forearm_phi;

    if d < 1e-12 {
        return solutions; // wrist center at shoulder — degenerate
    }

    // Cosine law for the "elbow" angle
    let cos_elbow = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);

    if cos_elbow.abs() > 1.0 {
        return solutions; // unreachable
    }

    let elbow_angle = cos_elbow.acos();

    // Two elbow configurations
    for &sign in &[1.0, -1.0] {
        let ea = sign * elbow_angle;

        // Joint 3 (URDF angle) = elbow_angle - phi_forearm
        // The elbow angle is the supplement of the interior triangle angle,
        // offset by the forearm's built-in angle.
        let j3 = normalize_angle(ea - phi);

        // Compute arm geometry for this j3
        let arm_x = -l1 - l2 * (j3 + phi).cos();
        let arm_z = l2 * (j3 + phi).sin();

        // Joint 2: angle from arm geometry to target direction
        let j2 = normalize_angle(arm_z.atan2(arm_x) - h.atan2(r));

        solutions.push((j2, j3));
    }

    solutions
}

/// Solve for joints 4, 5, 6 using FK-based R_03 computation.
///
/// Computes R_03 from FK at [j1, j2, j3, 0, 0, 0], then decomposes
/// R_03^{-1} * R_target into YZY Euler angles (matching Y, Z, Y wrist axes).
fn solve_j456(
    robot: &Robot,
    chain: &KinematicChain,
    j1: f64,
    j2: f64,
    j3: f64,
    target: &Pose,
) -> Vec<(f64, f64, f64)> {
    let mut solutions = Vec::with_capacity(2);

    // Compute R at [j1, j2, j3, 0, 0, 0]
    let q_partial = vec![j1, j2, j3, 0.0, 0.0, 0.0];
    let pose_partial = match forward_kinematics(robot, chain, &q_partial) {
        Ok(p) => p,
        Err(_) => return solutions,
    };

    // R_partial = R_03 * R_wrist_at_zero * R_tool_offset
    // At wrist zero, all wrist rotations are identity (no RPY on wrist joint origins),
    // but there may be a tool offset rotation.
    // R_target = R_03 * R_wrist(j4,j5,j6) * R_tool_offset
    // R_partial = R_03 * R_tool_offset (wrist at zero = identity rotations)
    // So: R_partial^{-1} * R_target = R_tool_offset^{-1} * R_wrist(j4,j5,j6) * R_tool_offset
    //
    // If R_tool_offset = Identity (no RPY on wrist/tool origins):
    // R_partial^{-1} * R_target = R_wrist(j4,j5,j6)
    //
    // For URDFs with no RPY rotations on joint origins (like our UR5e/UR10e),
    // this simplifies to direct decomposition.

    let r_partial = pose_partial.rotation().to_rotation_matrix();
    let r_target = target.rotation().to_rotation_matrix();

    // R_wrist = R_partial^T * R_target
    let r_wrist = r_partial.matrix().transpose() * r_target.matrix();

    // Decompose R_wrist as Ry(j4) * Rz(j5) * Ry(j6)
    // Matrix elements of Ry(a) * Rz(b) * Ry(c):
    // M = [[ca*cb*cc - sa*sc,  -ca*sb,  ca*cb*sc + sa*cc],
    //      [sb*cc,              cb,      sb*sc],
    //      [-sa*cb*cc - ca*sc,  sa*sb,  -sa*cb*sc + ca*cc]]
    //
    // M[1][1] = cos(j5)
    // M[1][0] = sin(j5)*cos(j6)
    // M[1][2] = sin(j5)*sin(j6)
    // M[0][1] = -cos(j4)*sin(j5)
    // M[2][1] = sin(j4)*sin(j5)

    let cb = r_wrist[(1, 1)].clamp(-1.0, 1.0);
    let sb_sq = 1.0 - cb * cb;

    if sb_sq > 1e-10 {
        let sb = sb_sq.sqrt();

        // Solution 1: j5 positive
        let j5_a = normalize_angle(sb.atan2(cb));
        let j6_a = normalize_angle(r_wrist[(1, 2)].atan2(r_wrist[(1, 0)]));
        let j4_a = normalize_angle(r_wrist[(2, 1)].atan2(-r_wrist[(0, 1)]));
        solutions.push((j4_a, j5_a, j6_a));

        // Solution 2: j5 negative (flipped wrist)
        let j5_b = normalize_angle((-sb).atan2(cb));
        let j6_b = normalize_angle((-r_wrist[(1, 2)]).atan2(-r_wrist[(1, 0)]));
        let j4_b = normalize_angle((-r_wrist[(2, 1)]).atan2(r_wrist[(0, 1)]));
        solutions.push((j4_b, j5_b, j6_b));
    } else {
        // Wrist singularity: j5 ≈ 0 or π
        let j5 = if cb > 0.0 { 0.0 } else { PI };
        // j4 + j6 (or j4 - j6) is determined; split arbitrarily
        let sum = r_wrist[(0, 0)].atan2(-r_wrist[(2, 0)]);
        solutions.push((sum, j5, 0.0));
    }

    solutions
}

/// Normalize angle to [-π, π].
fn normalize_angle(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Filter OPW solutions by joint limits.
pub fn filter_by_limits(
    solutions: &[[f64; 6]],
    robot: &Robot,
    chain: &KinematicChain,
) -> Vec<[f64; 6]> {
    solutions
        .iter()
        .filter(|q| {
            chain
                .active_joints
                .iter()
                .enumerate()
                .all(|(i, &joint_idx)| {
                    if let Some(limits) = &robot.joints[joint_idx].limits {
                        q[i] >= limits.lower && q[i] <= limits.upper
                    } else {
                        true
                    }
                })
        })
        .copied()
        .collect()
}

/// Solve OPW IK and return the best solution as an IKSolution.
///
/// Uses the OPW analytical solver to get initial solutions, then refines
/// with DLS if needed (the UR5e's non-spherical wrist causes ~10cm error
/// in the analytical solution, which DLS cleans up in 1-5 iterations).
pub fn solve_opw_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &IKConfig,
) -> Result<IKSolution> {
    let params = OPWParameters::from_robot_chain(robot, chain)?;

    let all_solutions = solve_opw(&params, robot, chain, target);

    // Filter by joint limits
    let valid = if config.check_limits {
        filter_by_limits(&all_solutions, robot, chain)
    } else {
        all_solutions
    };

    if valid.is_empty() {
        return Err(kinetic_core::KineticError::IKNotConverged {
            iterations: 0,
            residual: f64::INFINITY,
        });
    }

    // Find solution closest to seed (in joint-space distance)
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;

    for (i, q) in valid.iter().enumerate() {
        let dist: f64 = q
            .iter()
            .zip(seed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let q = valid[best_idx].to_vec();

    // Compute actual error via FK
    let actual_pose = forward_kinematics(robot, chain, &q)?;
    let (pos_err, orient_err, _) = pose_error(&actual_pose, target);

    let converged =
        pos_err < config.position_tolerance && orient_err < config.orientation_tolerance;

    if converged {
        return Ok(IKSolution {
            joints: q,
            position_error: pos_err,
            orientation_error: orient_err,
            converged: true,
            iterations: 1,
            mode_used: IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
        });
    }

    // Refine with DLS from the best analytical seed.
    // This handles non-spherical wrist robots (UR5e, etc.) where the
    // analytical solution has ~10cm error due to d5/d6 offsets.
    let refine_config = IKConfig {
        solver: crate::ik::IKSolver::DLS { damping: 0.01 },
        mode: config.mode,
        max_iterations: 30,
        position_tolerance: config.position_tolerance,
        orientation_tolerance: config.orientation_tolerance,
        check_limits: config.check_limits,
        seed: Some(q),
        null_space: None,
        num_restarts: 0,
    };

    crate::ik::solve_ik(robot, chain, target, &refine_config)
}

/// Check if a robot has OPW-compatible geometry.
///
/// Validates structural requirements for the OPW analytical solver:
/// 1. The chain has exactly 6 DOF
/// 2. J1 axis is approximately along Z (vertical base rotation)
/// 3. OPW parameters can be extracted with reasonable arm geometry
///    (upper arm and forearm > 3cm each — filters out tiny hobby arms
///    where DLS-only performs better)
pub fn is_opw_compatible(robot: &Robot, chain: &KinematicChain) -> bool {
    if chain.dof != 6 {
        return false;
    }

    // Check J1 axis — OPW assumes base rotation around Z
    let j1_idx = chain.active_joints[0];
    let j1_axis = robot.joints[j1_idx].axis.normalize();
    let z_axis = nalgebra::Vector3::z();
    let y_axis = nalgebra::Vector3::y();
    let j1_z_alignment = j1_axis.dot(&z_axis).abs();
    if j1_z_alignment < 0.95 {
        return false;
    }

    // Check wrist axis convention: OPW uses YZY Euler decomposition for J4-J5-J6.
    // The local URDF axes must follow approximately Y, Z, Y for the wrist joints.
    let j4_idx = chain.active_joints[3];
    let j5_idx = chain.active_joints[4];
    let j6_idx = chain.active_joints[5];
    let j4_axis = robot.joints[j4_idx].axis.normalize();
    let j5_axis = robot.joints[j5_idx].axis.normalize();
    let j6_axis = robot.joints[j6_idx].axis.normalize();

    // J4 should be approximately Y
    if j4_axis.dot(&y_axis).abs() < 0.9 {
        return false;
    }
    // J5 should be approximately Z
    if j5_axis.dot(&z_axis).abs() < 0.9 {
        return false;
    }
    // J6 should be approximately Y
    if j6_axis.dot(&y_axis).abs() < 0.9 {
        return false;
    }

    // Check wrist compactness: J4 and J6 origins should be close to each other
    // (approximately intersecting wrist axes). For spread-out wrists (d46 > 5cm
    // relative to arm length), the OPW closed-form solution is inaccurate.
    let zero = vec![0.0; chain.dof];
    let link_poses = match crate::forward::forward_kinematics_all(robot, chain, &zero) {
        Ok(p) => p,
        Err(_) => return false,
    };
    if link_poses.len() >= 7 {
        let p4 = link_poses[4].translation();
        let p6 = link_poses[6].translation();
        let d46 = (p6 - p4).norm();
        // Allow up to 10cm wrist spread — beyond this DLS refinement struggles
        if d46 > 0.10 {
            return false;
        }
    }

    // Try to extract OPW parameters — fails if geometry is degenerate
    let params = match OPWParameters::from_robot_chain(robot, chain) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Validate arm lengths are reasonable (> 3cm each)
    if params.upper_arm_length < 0.03 || params.forearm_length < 0.03 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::forward_kinematics;

    #[test]
    fn opw_ur5e_geometry() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        assert!((params.shoulder_height - 0.1625).abs() < 0.001);
        assert!((params.upper_arm_length - 0.425).abs() < 0.001);
    }

    #[test]
    fn opw_ur5e_analytical_solutions() {
        // Test that the analytical solver finds reasonable (approximate) solutions.
        // Due to the UR5e's non-spherical wrist (d5/d6 offsets), the raw
        // analytical solutions have ~10cm error. The solve_opw_ik function
        // refines them with DLS to get sub-millimeter accuracy.
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let q_original = [0.5, -1.0, 0.8, -0.5, 1.2, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();
        let solutions = solve_opw(&params, &robot, &chain, &target);

        assert!(
            !solutions.is_empty(),
            "OPW should find at least one solution"
        );

        // Analytical solutions should be in the right ballpark (<0.2m)
        let mut best_err = f64::INFINITY;
        for q in &solutions {
            if let Ok(pose) = forward_kinematics(&robot, &chain, q) {
                let (pos_err, _, _) = pose_error(&pose, &target);
                if pos_err < best_err {
                    best_err = pos_err;
                }
            }
        }
        assert!(
            best_err < 0.2,
            "Best analytical solution should be within 20cm (got {:.4}m)",
            best_err
        );
    }

    #[test]
    fn opw_ur5e_roundtrip() {
        // Full roundtrip test using the OPW+DLS hybrid solver.
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let q_original = [0.5, -1.0, 0.8, -0.5, 1.2, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let config = IKConfig {
            solver: crate::ik::IKSolver::OPW,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.01,
            check_limits: true,
            seed: Some(q_original.to_vec()),
            ..Default::default()
        };

        let solution = solve_opw_ik(&robot, &chain, &target, &q_original, &config).unwrap();

        assert!(
            solution.converged,
            "OPW+DLS should converge: pos_err={}, orient_err={}",
            solution.position_error, solution.orientation_error
        );
        assert!(
            solution.position_error < 1e-3,
            "Position error too large: {}",
            solution.position_error
        );
    }

    #[test]
    fn opw_ur5e_ik_interface() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let q_target = [0.3, -1.2, 0.5, -0.8, 1.0, -0.2];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let config = IKConfig {
            solver: crate::ik::IKSolver::OPW,
            position_tolerance: 0.01,
            orientation_tolerance: 0.05,
            check_limits: true,
            ..Default::default()
        };

        let solution = solve_opw_ik(&robot, &chain, &target, &q_target, &config).unwrap();

        assert!(
            solution.position_error < 0.02,
            "OPW IK position error too large: {}",
            solution.position_error
        );
    }

    #[test]
    fn opw_multiple_solutions() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();
        let target = Pose::from_xyz_rpy(0.3, 0.1, 0.4, 0.0, PI / 2.0, 0.0);
        let solutions = solve_opw(&params, &robot, &chain, &target);

        assert!(
            !solutions.is_empty(),
            "Expected at least 1 OPW solution, got {}",
            solutions.len()
        );
    }

    #[test]
    fn opw_compatibility_check() {
        let panda = Robot::from_name("franka_panda").unwrap();
        let panda_arm = &panda.groups["arm"];
        let panda_chain =
            KinematicChain::extract(&panda, &panda_arm.base_link, &panda_arm.tip_link).unwrap();
        assert!(
            !is_opw_compatible(&panda, &panda_chain),
            "Panda is 7-DOF, not OPW"
        );

        let ur5e = Robot::from_name("ur5e").unwrap();
        let ur5e_arm = &ur5e.groups["arm"];
        let ur5e_chain =
            KinematicChain::extract(&ur5e, &ur5e_arm.base_link, &ur5e_arm.tip_link).unwrap();
        assert!(
            is_opw_compatible(&ur5e, &ur5e_chain),
            "UR5e should be OPW compatible"
        );

        let xarm6 = Robot::from_name("xarm6").unwrap();
        let xarm6_arm = &xarm6.groups["arm"];
        let xarm6_chain =
            KinematicChain::extract(&xarm6, &xarm6_arm.base_link, &xarm6_arm.tip_link).unwrap();
        assert!(
            is_opw_compatible(&xarm6, &xarm6_chain),
            "xArm6 should be OPW compatible"
        );
    }

    #[test]
    fn opw_filter_by_limits() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();
        let target = Pose::from_xyz_rpy(0.3, 0.1, 0.4, 0.0, PI / 2.0, 0.0);
        let all_solutions = solve_opw(&params, &robot, &chain, &target);
        let filtered = filter_by_limits(&all_solutions, &robot, &chain);

        // Filtered should be a subset
        assert!(filtered.len() <= all_solutions.len());

        // All filtered solutions should be within limits
        for q in &filtered {
            for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
                if let Some(limits) = &robot.joints[joint_idx].limits {
                    assert!(
                        q[i] >= limits.lower && q[i] <= limits.upper,
                        "Joint {} = {} outside [{}, {}]",
                        i,
                        q[i],
                        limits.lower,
                        limits.upper
                    );
                }
            }
        }
    }

    // ─── New edge case tests below ───

    /// Wrist singularity: j5 ≈ 0 causes gimbal lock in YZY decomposition.
    /// Verify solutions don't contain NaN and solver doesn't panic.
    #[test]
    fn opw_wrist_singularity_j5_near_zero() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Config with j5 = 0.001 (near singularity)
        let q_singular = [0.5, -1.0, 0.8, -0.5, 0.001, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_singular).unwrap();
        let solutions = solve_opw(&params, &robot, &chain, &target);

        // No solution should contain NaN
        for (i, q) in solutions.iter().enumerate() {
            for (j, &val) in q.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Solution {i} joint {j} is not finite: {val}"
                );
            }
        }
        // Should still find at least one solution
        assert!(
            !solutions.is_empty(),
            "should find solutions near wrist singularity"
        );
    }

    /// Wrist singularity: j5 ≈ π (another singularity case).
    #[test]
    fn opw_wrist_singularity_j5_near_pi() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Config with j5 near π
        let q_singular = [0.3, -0.8, 0.5, 0.2, PI - 0.001, 0.1];
        let target = forward_kinematics(&robot, &chain, &q_singular).unwrap();
        let solutions = solve_opw(&params, &robot, &chain, &target);

        for q in &solutions {
            assert!(
                q.iter().all(|v| v.is_finite()),
                "NaN in solution at j5≈π singularity: {:?}",
                q
            );
        }
    }

    /// Arm fully extended: target at maximum reach (cosine law degenerate, cos_elbow ≈ 1).
    #[test]
    fn opw_arm_fully_extended() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Max reach = upper_arm + forearm from shoulder height
        let max_reach = params.upper_arm_length + params.forearm_length;
        // Place target at nearly max reach along -X (arm extension direction at j1=0)
        let target = Pose::from_xyz_rpy(
            -(max_reach * 0.999), // just barely reachable
            0.0,
            params.shoulder_height,
            0.0,
            PI / 2.0,
            0.0,
        );

        let solutions = solve_opw(&params, &robot, &chain, &target);
        // Should find solutions (or empty if cos_elbow rounds above 1.0 due to floating point)
        // Key: no NaN, no panic
        for q in &solutions {
            assert!(
                q.iter().all(|v| v.is_finite()),
                "NaN in fully-extended solution: {:?}",
                q
            );
        }
    }

    /// Target just barely unreachable — beyond max reach.
    /// Verify empty solutions, no NaN, no panic.
    #[test]
    fn opw_target_unreachable() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Target at 10 meters — far beyond any arm's reach
        let target = Pose::from_xyz_rpy(-10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let solutions = solve_opw(&params, &robot, &chain, &target);

        // Should return empty (cosine law fails: cos_elbow > 1)
        assert!(
            solutions.is_empty(),
            "unreachable target should produce 0 solutions, got {}",
            solutions.len()
        );
    }

    /// Arm fully retracted: wrist center at shoulder position (d ≈ 0).
    #[test]
    fn opw_arm_fully_retracted() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Target at the shoulder origin — wrist center would need to be at (0, 0, shoulder_height)
        // This is degenerate: d ≈ 0
        let target = Pose::from_xyz_rpy(0.0, 0.0, params.shoulder_height, 0.0, 0.0, 0.0);
        let solutions = solve_opw(&params, &robot, &chain, &target);

        // No panic, no NaN — may return empty due to d < 1e-12 guard
        for q in &solutions {
            assert!(
                q.iter().all(|v| v.is_finite()),
                "NaN in fully-retracted solution: {:?}",
                q
            );
        }
    }

    /// OPWParameters::from_robot_chain with a chain that produces < 5 link poses
    /// should return an error.
    #[test]
    fn opw_from_robot_chain_short_chain_error() {
        // Use a 2-link robot to get a short kinematic chain
        let urdf = r#"<?xml version="1.0"?>
<robot name="two_link">
  <link name="base"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="tip"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let result = OPWParameters::from_robot_chain(&robot, &chain);
        assert!(
            result.is_err(),
            "OPW should fail for chain with < 5 link poses"
        );
    }

    /// All 8 solutions at a non-singular configuration.
    /// OPW produces 2 J1 × 2 (J2,J3) × 2 (J4,J5,J6) = 8 solutions maximum.
    #[test]
    fn opw_eight_solutions_enumeration() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Use a generic config away from singularities
        let q_generic = [0.5, -1.0, 0.8, -0.5, 1.2, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_generic).unwrap();
        let solutions = solve_opw(&params, &robot, &chain, &target);

        // Should have multiple solutions (ideally 8, but some may be filtered by NaN)
        assert!(
            solutions.len() >= 2,
            "expected at least 2 solutions at non-singular config, got {}",
            solutions.len()
        );

        // Verify solutions are distinct (not duplicates)
        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let diff: f64 = solutions[i]
                    .iter()
                    .zip(solutions[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                assert!(
                    diff > 1e-6,
                    "solutions {} and {} are duplicates (diff={})",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    /// solve_opw_ik returns IKNotConverged when no valid solution exists.
    #[test]
    fn opw_solve_ik_unreachable_returns_error() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let target = Pose::from_xyz_rpy(-10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let seed = vec![0.0; 6];
        let config = IKConfig {
            solver: crate::ik::IKSolver::OPW,
            ..Default::default()
        };
        let result = solve_opw_ik(&robot, &chain, &target, &seed, &config);
        assert!(result.is_err(), "unreachable target should return error");
    }

    /// normalize_angle produces values in [-π, π].
    #[test]
    fn opw_normalize_angle_range() {
        assert!((normalize_angle(0.0) - 0.0).abs() < 1e-12);
        assert!((normalize_angle(PI) - PI).abs() < 1e-12);
        assert!((normalize_angle(-PI) - (-PI)).abs() < 1e-12);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < 1e-10);
        assert!((normalize_angle(-3.0 * PI) - (-PI)).abs() < 1e-10);
        assert!((normalize_angle(2.0 * PI) - 0.0).abs() < 1e-10);
    }

    // ─── is_opw_compatible edge cases ───

    /// 3-DOF chain: not OPW compatible (needs exactly 6 DOF).
    #[test]
    fn opw_not_compatible_3dof() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="three_link">
  <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
  <joint name="j1" type="revolute">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="b"/><child link="c"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="c"/><child link="d"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "a", "d").unwrap();
        assert_eq!(chain.dof, 3);
        assert!(
            !is_opw_compatible(&robot, &chain),
            "3-DOF should not be OPW compatible"
        );
    }

    /// 6-DOF but J1 axis is along X (not Z): OPW requires vertical base.
    #[test]
    fn opw_not_compatible_horizontal_base() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="horiz_base">
  <link name="l0"/><link name="l1"/><link name="l2"/>
  <link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/>
  <joint name="j1" type="revolute">
    <parent link="l0"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.08"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "l0", "l6").unwrap();
        assert_eq!(chain.dof, 6);
        assert!(
            !is_opw_compatible(&robot, &chain),
            "Horizontal J1 (X axis) should not be OPW compatible"
        );
    }

    /// UR10e: should be OPW compatible (similar geometry to UR5e).
    #[test]
    fn opw_ur10e_compatible() {
        let robot = Robot::from_name("ur10e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        assert!(
            is_opw_compatible(&robot, &chain),
            "UR10e should be OPW compatible"
        );
    }

    /// OPW parameter extraction for UR10e: arm lengths should be reasonable.
    #[test]
    fn opw_ur10e_parameter_extraction() {
        let robot = Robot::from_name("ur10e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // UR10e: upper arm ~0.6127m, forearm ~0.5716m, shoulder ~0.1807m
        assert!(
            params.upper_arm_length > 0.5 && params.upper_arm_length < 0.7,
            "UR10e upper_arm_length={} out of expected range",
            params.upper_arm_length
        );
        assert!(
            params.forearm_length > 0.3 && params.forearm_length < 0.7,
            "UR10e forearm_length={} out of expected range",
            params.forearm_length
        );
        assert!(
            params.shoulder_height > 0.1 && params.shoulder_height < 0.3,
            "UR10e shoulder_height={} out of expected range",
            params.shoulder_height
        );
    }

    /// xArm6: should be OPW compatible and parameters should be extractable.
    #[test]
    fn opw_xarm6_parameter_extraction() {
        let robot = Robot::from_name("xarm6").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        assert!(is_opw_compatible(&robot, &chain));

        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();
        assert!(
            params.upper_arm_length > 0.03,
            "xArm6 upper arm too short: {}",
            params.upper_arm_length
        );
        assert!(
            params.forearm_length > 0.03,
            "xArm6 forearm too short: {}",
            params.forearm_length
        );
    }

    /// solve_opw_ik with check_limits=false: allows out-of-limit solutions.
    #[test]
    fn opw_solve_ik_no_limit_check() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let q_original = [0.5, -1.0, 0.8, -0.5, 1.2, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let config = IKConfig {
            solver: crate::ik::IKSolver::OPW,
            position_tolerance: 0.01,
            orientation_tolerance: 0.05,
            check_limits: false, // disabled
            seed: Some(q_original.to_vec()),
            ..Default::default()
        };

        let solution = solve_opw_ik(&robot, &chain, &target, &q_original, &config).unwrap();
        assert!(solution.position_error.is_finite());
        assert_eq!(solution.joints.len(), 6);
    }

    /// solve_opw with target on the base axis (wx=0, wy=0): J1 is degenerate.
    #[test]
    fn opw_target_on_base_axis() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let params = OPWParameters::from_robot_chain(&robot, &chain).unwrap();

        // Target directly above the base (wx=wy=0)
        let target = Pose::from_xyz_rpy(0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        let solutions = solve_opw(&params, &robot, &chain, &target);

        // No NaN, no panic — may have solutions depending on arm geometry
        for q in &solutions {
            for &v in q {
                assert!(v.is_finite(), "NaN in on-axis solution: {:?}", q);
            }
        }
    }

    /// filter_by_limits with an empty solution set: should return empty.
    #[test]
    fn opw_filter_by_limits_empty() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let empty: Vec<[f64; 6]> = vec![];
        let result = filter_by_limits(&empty, &robot, &chain);
        assert!(result.is_empty());
    }

    /// filter_by_limits: a solution with a value at the exact boundary should pass.
    #[test]
    fn opw_filter_by_limits_boundary_values() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        // Build a solution at the exact lower limit for all joints
        let mut sol = [0.0f64; 6];
        for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                sol[i] = limits.lower; // exact boundary
            }
        }

        let result = filter_by_limits(&[sol], &robot, &chain);
        assert_eq!(result.len(), 1, "Boundary values should pass the filter");
    }

    /// OPWParameters::ur5e, ur10e, xarm6 convenience constructors all work.
    #[test]
    fn opw_convenience_constructors() {
        let ur5e = Robot::from_name("ur5e").unwrap();
        let ur5e_arm = &ur5e.groups["arm"];
        let ur5e_chain =
            KinematicChain::extract(&ur5e, &ur5e_arm.base_link, &ur5e_arm.tip_link).unwrap();
        let p1 = OPWParameters::ur5e(&ur5e, &ur5e_chain).unwrap();

        // The convenience constructors delegate to from_robot_chain,
        // so they should produce the same parameters.
        let p2 = OPWParameters::from_robot_chain(&ur5e, &ur5e_chain).unwrap();
        assert!((p1.shoulder_height - p2.shoulder_height).abs() < 1e-10);
        assert!((p1.upper_arm_length - p2.upper_arm_length).abs() < 1e-10);
    }
}
