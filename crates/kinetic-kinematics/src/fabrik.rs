//! FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.
//!
//! A fast iterative IK solver that works on the position level:
//! 1. Forward pass: starting from the end-effector, move each joint
//!    toward the target while maintaining link lengths.
//! 2. Backward pass: starting from the base, correct positions to
//!    maintain the root constraint.
//! 3. Repeat until convergence.
//!
//! FABRIK is typically faster than Jacobian-based methods for simple
//! chains but doesn't directly handle orientation constraints.
//! This implementation uses a hybrid approach: FABRIK for position,
//! then a small DLS refinement for orientation.

use kinetic_core::{Pose, Result};
use kinetic_robot::Robot;
use nalgebra::Vector3;

use crate::chain::KinematicChain;
use crate::forward::{forward_kinematics_all, forward_kinematics};
use crate::ik::{pose_error, IKConfig, IKMode, IKSolution};

/// Solve IK using FABRIK.
///
/// First converges on position using the FABRIK algorithm,
/// then refines orientation using a small Jacobian-based step.
pub fn solve_fabrik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &IKConfig,
) -> Result<IKSolution> {
    let dof = chain.dof;
    let mut q: Vec<f64> = seed.to_vec();
    if q.len() != dof {
        q.resize(dof, 0.0);
    }

    // Get link positions at seed configuration
    let link_poses = forward_kinematics_all(robot, chain, &q)?;
    let num_points = link_poses.len();

    // Extract link positions
    let mut points: Vec<Vector3<f64>> = link_poses.iter().map(|p| p.translation()).collect();

    // Compute link lengths (distances between consecutive points)
    let mut link_lengths: Vec<f64> = Vec::with_capacity(num_points - 1);
    for i in 0..num_points - 1 {
        link_lengths.push((points[i + 1] - points[i]).norm());
    }

    let target_pos = target.translation();
    let base_pos = points[0];

    // Check reachability
    let total_reach: f64 = link_lengths.iter().sum();
    let target_dist = (target_pos - base_pos).norm();
    if target_dist > total_reach * 1.05 {
        // Target is beyond reach — stretch toward it
        // (FABRIK will get as close as possible)
    }

    let mut iterations = 0;
    let tol = config.position_tolerance;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Check convergence (position only for FABRIK)
        let ee_error = (points[num_points - 1] - target_pos).norm();
        if ee_error < tol {
            break;
        }

        // --- Forward pass ---
        // Start from end-effector, move toward target
        points[num_points - 1] = target_pos;

        for i in (0..num_points - 1).rev() {
            let direction = points[i] - points[i + 1];
            let dist = direction.norm();
            if dist > 1e-12 {
                points[i] = points[i + 1] + direction * (link_lengths[i] / dist);
            }
        }

        // --- Backward pass ---
        // Start from base, maintain root position
        points[0] = base_pos;

        for i in 0..num_points - 1 {
            let direction = points[i + 1] - points[i];
            let dist = direction.norm();
            if dist > 1e-12 {
                points[i + 1] = points[i] + direction * (link_lengths[i] / dist);
            }
        }
    }

    // Convert point positions back to joint angles
    // This is the key step: find joint angles that produce the FABRIK positions
    q = points_to_joints(robot, chain, &points, &q)?;

    // Clamp to limits
    if config.check_limits {
        for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                q[i] = q[i].clamp(limits.lower, limits.upper);
            }
        }
    }

    // Refine with small DLS steps for orientation
    let orient_iters = config.max_iterations.min(20);
    for _ in 0..orient_iters {
        let current = forward_kinematics(robot, chain, &q)?;
        let (pos_err, orient_err, error_6d) = pose_error(&current, target);

        if pos_err < config.position_tolerance && orient_err < config.orientation_tolerance {
            break;
        }

        // Small DLS step
        let j = crate::forward::jacobian(robot, chain, &q)?;
        let jjt = &j * j.transpose();
        let damping = 0.1;
        let damping_mat = nalgebra::DMatrix::identity(6, 6) * (damping * damping);
        let jjt_damped = jjt + damping_mat;

        if let Some(y) = jjt_damped.lu().solve(&error_6d) {
            let dq = j.transpose() * y;
            let step_scale = 0.5; // conservative step
            for i in 0..dof {
                q[i] += dq[i] * step_scale;
            }

            if config.check_limits {
                for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
                    if let Some(limits) = &robot.joints[joint_idx].limits {
                        q[i] = q[i].clamp(limits.lower, limits.upper);
                    }
                }
            }
        }
    }

    // Final error computation
    let final_pose = forward_kinematics(robot, chain, &q)?;
    let (final_pos_err, final_orient_err, _) = pose_error(&final_pose, target);

    let converged = final_pos_err < config.position_tolerance
        && final_orient_err < config.orientation_tolerance;

    Ok(IKSolution {
        joints: q,
        position_error: final_pos_err,
        orientation_error: final_orient_err,
        converged,
        iterations,
        mode_used: IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
    })
}

/// Convert FABRIK point positions back to joint angles.
///
/// For each joint, computes the angle that would move the child link
/// toward the desired position. Uses the current joint angles as
/// a starting point and applies incremental corrections.
fn points_to_joints(
    robot: &Robot,
    chain: &KinematicChain,
    target_points: &[Vector3<f64>],
    current_q: &[f64],
) -> Result<Vec<f64>> {
    let mut q = current_q.to_vec();

    // Iterative refinement: adjust each joint to minimize position error
    let num_refinements = 5;

    for _ in 0..num_refinements {
        let current_poses = forward_kinematics_all(robot, chain, &q)?;
        let mut active_idx = 0;

        for (link_idx, &joint_idx) in chain.all_joints.iter().enumerate() {
            let joint = &robot.joints[joint_idx];
            if !joint.is_active() {
                continue;
            }

            // Current and target positions for the next link
            let next_link = link_idx + 1;
            if next_link >= target_points.len() {
                active_idx += 1;
                continue;
            }

            let current_pos = current_poses[next_link].translation();
            let target_pos = target_points[next_link];
            let error = target_pos - current_pos;

            if error.norm() < 1e-8 {
                active_idx += 1;
                continue;
            }

            // Compute numerical gradient: how does this joint affect this link's position?
            let eps = 1e-5;
            let mut q_plus = q.clone();
            q_plus[active_idx] += eps;
            let poses_plus = forward_kinematics_all(robot, chain, &q_plus)?;
            let pos_plus = poses_plus[next_link].translation();
            let gradient = (pos_plus - current_pos) / eps;

            // Project error onto gradient direction
            let grad_norm_sq = gradient.dot(&gradient);
            if grad_norm_sq > 1e-12 {
                let step = gradient.dot(&error) / grad_norm_sq;
                q[active_idx] += step.clamp(-0.5, 0.5);
            }

            active_idx += 1;
        }
    }

    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::forward_kinematics;

    const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

    /// 10-DOF chain for high-DOF testing.
    const TEN_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_10dof">
  <link name="l0"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="l6"/><link name="l7"/>
  <link name="l8"/><link name="l9"/><link name="l10"/>

  <joint name="j1" type="revolute">
    <parent link="l0"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.12"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.12"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j7" type="revolute">
    <parent link="l6"/><child link="l7"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j8" type="revolute">
    <parent link="l7"/><child link="l8"/>
    <origin xyz="0 0 0.08"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j9" type="revolute">
    <parent link="l8"/><child link="l9"/>
    <origin xyz="0 0 0.08"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j10" type="revolute">
    <parent link="l9"/><child link="l10"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
</robot>
"#;

    fn default_config() -> IKConfig {
        IKConfig {
            max_iterations: 200,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.1,
            check_limits: true,
            ..Default::default()
        }
    }

    #[test]
    fn fabrik_3dof_position() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_original = vec![0.3, 0.5, -0.2];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &default_config()).unwrap();

        assert!(
            solution.position_error < 1e-2,
            "FABRIK position error too large: {}",
            solution.position_error
        );
    }

    #[test]
    fn fabrik_near_target() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_original = vec![0.2, 0.3, -0.1];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.1, 0.2, 0.0];
        let solution = solve_fabrik(&robot, &chain, &target, &seed, &default_config()).unwrap();

        assert!(
            solution.position_error < 1e-2,
            "FABRIK should converge from nearby seed: pos_err={}",
            solution.position_error
        );
    }

    // ─── Convergence stalling ────────────────────────────────────────────────

    #[test]
    fn fabrik_convergence_stalling_tight_tolerance() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_target = vec![0.8, 1.2, -0.5];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        // Extremely tight tolerance + few iterations → should not converge
        let config = IKConfig {
            max_iterations: 3,
            position_tolerance: 1e-12,
            orientation_tolerance: 1e-12,
            check_limits: true,
            ..Default::default()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        // Must return a valid result (no panic), even if not converged
        assert!(
            !solution.converged,
            "Should not converge with 3 iterations and 1e-12 tolerance"
        );
        assert!(solution.iterations <= 3, "Should respect iteration limit");
        assert!(!solution.position_error.is_nan(), "Error should not be NaN");
    }

    // ─── Unreachable goal ────────────────────────────────────────────────────

    #[test]
    fn fabrik_unreachable_goal() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Total reach: 0.1 + 0.3 + 0.25 = 0.65m. Place target at 10m.
        let target = Pose::from_xyz(10.0, 0.0, 0.0);

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &default_config()).unwrap();

        // Must not panic, must not converge
        assert!(
            !solution.converged,
            "Should not converge on unreachable goal"
        );
        assert!(
            solution.position_error > 9.0,
            "Error should be large for 10m target: {}",
            solution.position_error
        );
        assert!(!solution.position_error.is_nan());
        assert!(!solution.orientation_error.is_nan());
    }

    // ─── Joint limit enforcement ─────────────────────────────────────────────

    #[test]
    fn fabrik_joint_limits_enforced() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Target that requires extreme joint angles
        let q_extreme = vec![2.5, 1.8, -2.0];
        let target = forward_kinematics(&robot, &chain, &q_extreme).unwrap();

        let config = IKConfig {
            check_limits: true,
            ..default_config()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        // Verify joint limits: j1 ±3.14, j2 ±2.0, j3 ±2.5
        let limits = [(-3.14, 3.14), (-2.0, 2.0), (-2.5, 2.5)];
        for (i, (lo, hi)) in limits.iter().enumerate() {
            assert!(
                solution.joints[i] >= *lo - 1e-6 && solution.joints[i] <= *hi + 1e-6,
                "Joint {} = {} exceeds limits [{}, {}]",
                i,
                solution.joints[i],
                lo,
                hi
            );
        }
    }

    #[test]
    fn fabrik_without_limit_checking() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_target = vec![0.5, 0.8, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let config = IKConfig {
            check_limits: false,
            ..default_config()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        // Should still return valid result
        assert!(!solution.position_error.is_nan());
        for &v in &solution.joints {
            assert!(v.is_finite(), "Joint value should be finite");
        }
    }

    // ─── High-DOF chain ──────────────────────────────────────────────────────

    #[test]
    fn fabrik_10dof_chain() {
        let robot = Robot::from_urdf_string(TEN_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "l0", "l10").unwrap();
        assert_eq!(chain.dof, 10);

        // Use a reachable target from known config
        let q_known = vec![0.1, 0.2, -0.1, 0.15, -0.1, 0.2, 0.05, -0.1, 0.1, -0.05];
        let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

        let seed = vec![0.0; 10];
        let solution = solve_fabrik(&robot, &chain, &target, &seed, &default_config()).unwrap();

        assert!(!solution.position_error.is_nan());
        assert_eq!(solution.joints.len(), 10);
        // Position-level convergence should work well for redundant chains
        assert!(
            solution.position_error < 0.05,
            "10-DOF FABRIK position error: {}",
            solution.position_error
        );
    }

    // ─── Zero-motion goal (goal at current EE) ──────────────────────────────

    #[test]
    fn fabrik_goal_at_current_ee() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let seed = vec![0.3, 0.5, -0.2];
        let target = forward_kinematics(&robot, &chain, &seed).unwrap();

        // Solve with the seed as both start and target
        let solution = solve_fabrik(&robot, &chain, &target, &seed, &default_config()).unwrap();

        assert!(
            solution.converged,
            "Should converge trivially when goal equals current EE"
        );
        assert!(
            solution.position_error < 1e-3,
            "Error should be tiny: {}",
            solution.position_error
        );
    }

    // ─── Large joint rotation target ─────────────────────────────────────────

    #[test]
    fn fabrik_large_rotation_target() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Target from config with joint1 near its limit (close to π)
        let q_target = vec![3.0, -1.5, 2.0];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &default_config()).unwrap();

        // FABRIK should still produce a valid (if imperfect) result
        assert!(!solution.position_error.is_nan());
        assert!(solution.position_error.is_finite());
        assert_eq!(solution.joints.len(), 3);
    }

    // ─── Seed shorter/longer than DOF ────────────────────────────────────────

    #[test]
    fn fabrik_seed_length_mismatch() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let target = Pose::from_xyz(0.1, 0.1, 0.4);

        // Short seed (1 element for 3-DOF chain) — should be padded
        let solution = solve_fabrik(&robot, &chain, &target, &[0.5], &default_config()).unwrap();
        assert_eq!(solution.joints.len(), 3);
        assert!(!solution.position_error.is_nan());

        // Empty seed — should be padded to zeros
        let solution = solve_fabrik(&robot, &chain, &target, &[], &default_config()).unwrap();
        assert_eq!(solution.joints.len(), 3);
        assert!(!solution.position_error.is_nan());
    }

    // ─── Multiple targets convergence ────────────────────────────────────────

    #[test]
    fn fabrik_multiple_targets_consistent() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let targets = vec![
            vec![0.1, 0.3, -0.1],
            vec![0.5, 0.8, -0.3],
            vec![-0.2, 0.1, 0.4],
            vec![1.0, -1.0, 0.5],
        ];

        for (i, q_target) in targets.iter().enumerate() {
            let target = forward_kinematics(&robot, &chain, q_target).unwrap();
            let solution =
                solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &default_config()).unwrap();

            assert!(!solution.position_error.is_nan(), "Target {i}: NaN error");
            assert!(
                solution.position_error < 0.05,
                "Target {i}: error too large: {}",
                solution.position_error
            );
        }
    }

    // ─── IKSolution fields validation ────────────────────────────────────────

    #[test]
    fn fabrik_solution_fields_populated() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_target = vec![0.2, 0.3, -0.1];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &default_config()).unwrap();

        assert_eq!(solution.joints.len(), 3);
        assert!(solution.position_error >= 0.0);
        assert!(solution.orientation_error >= 0.0);
        assert!(solution.iterations > 0);
        assert_eq!(solution.mode_used, IKMode::Full6D);
    }

    // ─── Orientation refinement path ─────────────────────────────────────────

    /// FABRIK with orientation refinement: the DLS refinement loop at the end
    /// should improve orientation. Use a 3-DOF chain where full 6D convergence
    /// is impossible (under-determined), but the refinement code path executes.
    #[test]
    fn fabrik_orientation_refinement_executes() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Create a target with non-trivial orientation
        let q_target = vec![0.4, 0.6, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let config = IKConfig {
            max_iterations: 100,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.05,
            check_limits: true,
            ..Default::default()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        // Position should be good; the orientation refinement DLS loop ran
        assert!(
            solution.position_error < 0.05,
            "Position error: {}",
            solution.position_error
        );
        assert!(solution.orientation_error.is_finite());
    }

    // ─── Zero-length link edge case ─────────────────────────────────────────

    /// FABRIK with co-located joints (link length = 0): the forward/backward
    /// pass should handle direction vectors of zero length without NaN.
    #[test]
    fn fabrik_colocated_joints_no_nan() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="colocated">
  <link name="base"/>
  <link name="mid"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="mid"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="mid"/><child link="tip"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        assert_eq!(chain.dof, 2);

        let target = Pose::from_xyz(0.1, 0.1, 0.2);
        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0], &default_config()).unwrap();

        for &v in &solution.joints {
            assert!(v.is_finite(), "Joint should be finite with co-located joints");
        }
        assert!(!solution.position_error.is_nan());
    }

    // ─── Single iteration limit ─────────────────────────────────────────────

    /// With max_iterations=1, FABRIK should still produce a valid result.
    #[test]
    fn fabrik_single_iteration() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_target = vec![0.5, 0.8, -0.3];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let config = IKConfig {
            max_iterations: 1,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.1,
            check_limits: true,
            ..Default::default()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        assert_eq!(solution.iterations, 1);
        assert!(!solution.position_error.is_nan());
        assert_eq!(solution.joints.len(), 3);
    }

    // ─── 10-DOF with far seed ───────────────────────────────────────────────

    /// 10-DOF chain with a seed far from the target: FABRIK should still
    /// produce finite results and make meaningful progress.
    #[test]
    fn fabrik_10dof_far_seed() {
        let robot = Robot::from_urdf_string(TEN_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "l0", "l10").unwrap();

        let q_known = vec![0.3, 0.5, -0.2, 0.4, -0.3, 0.5, 0.1, -0.3, 0.2, -0.1];
        let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

        // Seed very far from target
        let seed = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

        let config = IKConfig {
            max_iterations: 200,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.1,
            check_limits: true,
            ..Default::default()
        };

        let solution = solve_fabrik(&robot, &chain, &target, &seed, &config).unwrap();

        assert_eq!(solution.joints.len(), 10);
        for &v in &solution.joints {
            assert!(v.is_finite());
        }
        assert!(solution.position_error.is_finite());
    }

    // ─── DLS refinement failure path ─────────────────────────────────────────

    /// When the DLS refinement LU solve fails (jjt_damped is singular),
    /// the code should gracefully skip the DLS step. Test by checking
    /// that even with an odd configuration, no NaN leaks through.
    #[test]
    fn fabrik_dls_refinement_graceful_on_singular() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Target at the base — causes degenerate Jacobian
        let target = Pose::from_xyz(0.0, 0.0, 0.1);

        let config = IKConfig {
            max_iterations: 50,
            position_tolerance: 1e-3,
            orientation_tolerance: 0.1,
            check_limits: true,
            ..Default::default()
        };

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        for &v in &solution.joints {
            assert!(v.is_finite(), "Joint should be finite even with degenerate target");
        }
        assert!(!solution.position_error.is_nan());
    }

    // ─── Convergence consistency ────────────────────────────────────────────

    /// If the solver reports converged=true, the errors must be below
    /// the configured tolerances.
    #[test]
    fn fabrik_convergence_implies_tolerance() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let q_target = vec![0.3, 0.4, -0.2];
        let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

        let config = IKConfig {
            max_iterations: 200,
            position_tolerance: 0.01,
            orientation_tolerance: 0.1,
            check_limits: true,
            ..Default::default()
        };

        let solution =
            solve_fabrik(&robot, &chain, &target, &[0.0, 0.0, 0.0], &config).unwrap();

        if solution.converged {
            assert!(
                solution.position_error < config.position_tolerance,
                "Converged but pos_err {} >= tol {}",
                solution.position_error,
                config.position_tolerance
            );
            assert!(
                solution.orientation_error < config.orientation_tolerance,
                "Converged but orient_err {} >= tol {}",
                solution.orientation_error,
                config.orientation_tolerance
            );
        }
    }
}
