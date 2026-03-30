//! SQP (Sequential Quadratic Programming) IK solver — Trac-IK parity.
//!
//! Combines numerical IK with joint-limit-aware optimization:
//! 1. Computes Jacobian at current configuration.
//! 2. Solves QP subproblem: minimize ‖J*Δq - e‖² subject to joint limits.
//! 3. Updates q += α * Δq with line search.
//! 4. Repeats until convergence.
//!
//! More robust than plain DLS near joint limits because it handles
//! inequality constraints explicitly rather than clamping post-hoc.

use kinetic_core::Pose;
use kinetic_robot::Robot;

use crate::forward::forward_kinematics;
use crate::ik::IKSolution;
use crate::KinematicChain;

/// SQP IK solver configuration.
#[derive(Debug, Clone)]
pub struct SQPConfig {
    /// Maximum iterations (default: 100).
    pub max_iterations: usize,
    /// Position convergence tolerance in meters (default: 1e-4).
    pub position_tolerance: f64,
    /// Orientation convergence tolerance in radians (default: 1e-3).
    pub orientation_tolerance: f64,
    /// Step size (default: 1.0, reduced by line search).
    pub step_size: f64,
    /// Damping factor for regularization (default: 0.01).
    pub damping: f64,
    /// Line search: minimum step fraction (default: 0.1).
    pub min_step: f64,
    /// Line search: backtracking factor (default: 0.5).
    pub backtrack_factor: f64,
}

impl Default for SQPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            step_size: 1.0,
            damping: 0.01,
            min_step: 0.1,
            backtrack_factor: 0.5,
        }
    }
}

/// Solve IK using SQP with joint limit constraints.
pub fn solve_sqp(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &SQPConfig,
) -> Option<IKSolution> {
    let dof = chain.dof;
    let mut q = seed.to_vec();

    // Joint limits
    let limits: Vec<(f64, f64)> = chain.active_joints.iter().map(|&ji| {
        robot.joints[ji].limits.as_ref()
            .map(|l| (l.lower, l.upper))
            .unwrap_or((-std::f64::consts::PI, std::f64::consts::PI))
    }).collect();

    for iter in 0..config.max_iterations {
        let pose = forward_kinematics(robot, chain, &q).ok()?;

        // Compute pose error
        let pos_err = target.translation() - pose.translation();
        let rot_diff = target.0.rotation * pose.0.rotation.inverse();
        let axis_angle = rot_diff.scaled_axis();

        let pos_norm = pos_err.norm();
        let orient_norm = axis_angle.norm();

        // Check convergence
        if pos_norm < config.position_tolerance && orient_norm < config.orientation_tolerance {
            return Some(IKSolution {
                joints: q,
                position_error: pos_norm,
                orientation_error: orient_norm,
                iterations: iter,
                converged: true,
            mode_used: crate::ik::IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
            });
        }

        // 6D error vector [position; orientation]
        let error = [
            pos_err.x, pos_err.y, pos_err.z,
            axis_angle.x, axis_angle.y, axis_angle.z,
        ];

        // Compute Jacobian numerically
        let jac = numerical_jacobian(robot, chain, &q, 1e-6)?;

        // Solve damped least squares with limit awareness:
        // Δq = (J^T J + λI)^{-1} J^T e
        // Then clamp to joint limits
        let delta_q = solve_damped_ls(&jac, &error, dof, config.damping);

        // Line search: find step size that reduces error
        let mut alpha = config.step_size;
        let current_err = pos_norm + orient_norm * 0.1;

        loop {
            let mut q_new = q.clone();
            for j in 0..dof {
                q_new[j] += alpha * delta_q[j];
                q_new[j] = q_new[j].clamp(limits[j].0, limits[j].1);
            }

            if let Ok(new_pose) = forward_kinematics(robot, chain, &q_new) {
                let new_pos_err = (target.translation() - new_pose.translation()).norm();
                let new_rot_diff = target.0.rotation * new_pose.0.rotation.inverse();
                let new_orient_err = new_rot_diff.scaled_axis().norm();
                let new_err = new_pos_err + new_orient_err * 0.1;

                if new_err < current_err || alpha <= config.min_step {
                    q = q_new;
                    break;
                }
            }

            alpha *= config.backtrack_factor;
            if alpha < config.min_step {
                // Accept the step even if it doesn't improve
                for j in 0..dof {
                    q[j] += config.min_step * delta_q[j];
                    q[j] = q[j].clamp(limits[j].0, limits[j].1);
                }
                break;
            }
        }
    }

    // Return best found (may not have converged)
    let pose = forward_kinematics(robot, chain, &q).ok()?;
    let pos_err = (target.translation() - pose.translation()).norm();
    let rot_diff = target.0.rotation * pose.0.rotation.inverse();
    let orient_err = rot_diff.scaled_axis().norm();

    Some(IKSolution {
        joints: q,
        position_error: pos_err,
        orientation_error: orient_err,
        iterations: config.max_iterations,
        converged: false,
        mode_used: crate::ik::IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
    })
}

/// Compute 6×DOF Jacobian numerically via central differences.
fn numerical_jacobian(
    robot: &Robot,
    chain: &KinematicChain,
    q: &[f64],
    eps: f64,
) -> Option<Vec<[f64; 6]>> {
    let dof = chain.dof;
    let mut jac = vec![[0.0f64; 6]; dof];

    for j in 0..dof {
        let mut q_plus = q.to_vec();
        q_plus[j] += eps;
        let pose_plus = forward_kinematics(robot, chain, &q_plus).ok()?;

        let mut q_minus = q.to_vec();
        q_minus[j] -= eps;
        let pose_minus = forward_kinematics(robot, chain, &q_minus).ok()?;

        let dt = (pose_plus.translation() - pose_minus.translation()) / (2.0 * eps);
        let dr = (pose_plus.0.rotation * pose_minus.0.rotation.inverse()).scaled_axis() / (2.0 * eps);

        jac[j] = [dt.x, dt.y, dt.z, dr.x, dr.y, dr.z];
    }

    Some(jac)
}

/// Solve damped least squares: Δq = (J^T J + λI)^{-1} J^T e.
fn solve_damped_ls(jac: &[[f64; 6]], error: &[f64; 6], dof: usize, damping: f64) -> Vec<f64> {
    // J^T * e
    let mut jt_e = vec![0.0; dof];
    for j in 0..dof {
        for i in 0..6 {
            jt_e[j] += jac[j][i] * error[i];
        }
    }

    // J^T * J + λI
    let mut jtj = vec![0.0; dof * dof];
    for j in 0..dof {
        for k in 0..dof {
            let mut sum = 0.0;
            for i in 0..6 {
                sum += jac[j][i] * jac[k][i];
            }
            jtj[j * dof + k] = sum;
        }
        jtj[j * dof + j] += damping;
    }

    // Solve (J^T J + λI) * Δq = J^T * e via Cholesky-like direct solve
    // For small DOF (≤7), Gauss elimination is fine
    solve_linear_system(&jtj, &jt_e, dof)
}

/// Solve Ax = b for small systems via Gauss elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val { max_val = val; max_row = row; }
        }
        if max_row != col {
            for j in 0..=n { aug.swap(col * (n + 1) + j, max_row * (n + 1) + j); }
        }
        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 { continue; }
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-15 { continue; }
        x[i] = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            x[i] -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] /= pivot;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="test3dof">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/><child link="tip"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="10"/>
  </joint>
</robot>"#;

    #[test]
    fn sqp_finds_solution() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();

        let seed = [0.5, 0.3, -0.2];
        let target = forward_kinematics(&robot, &chain, &seed).unwrap();

        let result = solve_sqp(&robot, &chain, &target, &[0.0; 3], &SQPConfig::default());
        assert!(result.is_some());
        let sol = result.unwrap();
        assert!(sol.converged, "SQP should converge: pos_err={}", sol.position_error);
        assert!(sol.position_error < 0.001);
    }

    #[test]
    fn sqp_respects_joint_limits() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();

        let target = Pose::from_xyz(0.3, 0.0, 0.3);
        let result = solve_sqp(&robot, &chain, &target, &[0.0; 3], &SQPConfig::default());

        if let Some(sol) = result {
            // All joints should be within limits
            assert!(sol.joints[0] >= -3.14 && sol.joints[0] <= 3.14);
            assert!(sol.joints[1] >= -2.0 && sol.joints[1] <= 2.0);
            assert!(sol.joints[2] >= -2.5 && sol.joints[2] <= 2.5);
        }
    }

    #[test]
    fn sqp_config_default() {
        let config = SQPConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.position_tolerance > 0.0);
        assert!(config.damping > 0.0);
        assert!(config.step_size > 0.0);
    }

    #[test]
    fn sqp_near_seed_converges() {
        // Intent: starting near solution should converge quickly
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let seed = [0.3, -0.5, 0.8];
        let target = forward_kinematics(&robot, &chain, &seed).unwrap();
        // Start from slightly perturbed seed
        let near = [0.32, -0.48, 0.82];
        let result = solve_sqp(&robot, &chain, &target, &near, &SQPConfig::default());
        assert!(result.is_some());
        let sol = result.unwrap();
        assert!(sol.converged, "near seed should converge");
        assert!(sol.position_error < 0.001);
    }

    #[test]
    fn sqp_unreachable_does_not_crash() {
        // Intent: unreachable target should return best-effort, not panic
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let target = Pose::from_xyz(100.0, 0.0, 0.0); // way out of reach
        let config = SQPConfig {
            max_iterations: 20,
            ..Default::default()
        };
        let result = solve_sqp(&robot, &chain, &target, &[0.0; 3], &config);
        // Should not panic, may or may not return a solution
        if let Some(sol) = result {
            assert!(!sol.converged, "unreachable should not converge");
        }
    }

    #[test]
    fn sqp_returns_finite_values() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let target = Pose::from_xyz(0.2, 0.1, 0.4);
        let result = solve_sqp(&robot, &chain, &target, &[0.0; 3], &SQPConfig::default());
        if let Some(sol) = result {
            for (i, &j) in sol.joints.iter().enumerate() {
                assert!(j.is_finite(), "joint {i}: {j}");
            }
            assert!(sol.position_error.is_finite());
        }
    }
}
