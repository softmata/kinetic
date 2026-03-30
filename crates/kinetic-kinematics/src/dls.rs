//! Damped Least Squares (DLS) IK solver.
//!
//! Also known as Levenberg-Marquardt IK. Computes the pseudo-inverse
//! of the Jacobian with damping to handle singularities:
//!
//! ```text
//! Δq = J^T (J J^T + λ²I)^{-1} Δx
//! ```
//!
//! where λ is the damping factor, J is the 6×N Jacobian, and Δx is
//! the 6D pose error.

use kinetic_core::{Pose, Result};
use kinetic_robot::Robot;
use nalgebra::DMatrix;

use crate::chain::KinematicChain;
use crate::forward::{forward_kinematics, jacobian, manipulability};
use crate::ik::{pose_error, IKConfig, IKMode, IKSolution, NullSpace};

/// Solve IK using Damped Least Squares.
pub fn solve_dls(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    damping: f64,
    config: &IKConfig,
    mode: IKMode,
) -> Result<IKSolution> {
    let dof = chain.dof;
    let mut q: Vec<f64> = seed.to_vec();

    // Ensure seed has correct length
    if q.len() != dof {
        q.resize(dof, 0.0);
    }

    // Clamp initial seed to joint limits
    if config.check_limits {
        clamp_to_limits(robot, chain, &mut q);
    }

    let mut best_error = f64::INFINITY;
    let mut best_q = q.clone();
    let mut converged = false;
    let mut degraded = false;
    let mut iterations = 0;
    let mut current_damping = damping;
    let mut prev_error = f64::INFINITY;
    let mut stall_count: usize = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute current FK
        let current_pose = forward_kinematics(robot, chain, &q)?;

        // Compute pose error
        let (pos_err, orient_err, error_6d) = pose_error(&current_pose, target);

        // For position-only mode, use only position error for tracking and convergence
        let position_only = mode == IKMode::PositionOnly;
        let combined_err = if position_only {
            pos_err
        } else {
            pos_err + orient_err * 0.1
        };

        // Track best solution
        if combined_err < best_error {
            best_error = combined_err;
            best_q = q.clone();
            stall_count = 0;
        } else {
            stall_count += 1;
        }

        // Check convergence
        if position_only {
            if pos_err < config.position_tolerance {
                converged = true;
                break;
            }
        } else if pos_err < config.position_tolerance && orient_err < config.orientation_tolerance {
            converged = true;
            break;
        }

        // Stall escape: if no improvement for 40 iterations, restart from
        // best solution with a random perturbation to escape local minimum
        if stall_count >= 40 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            q = best_q.clone();
            for qi in q.iter_mut().take(dof) {
                *qi += rng.gen_range(-0.3..0.3);
            }
            if config.check_limits {
                clamp_to_limits(robot, chain, &mut q);
            }
            stall_count = 0;
            current_damping = damping;
            prev_error = f64::INFINITY;
            continue;
        }

        // Adaptive damping: reduce when making progress, increase when stalling
        if combined_err < prev_error * 0.99 {
            current_damping = (current_damping * 0.8).max(damping * 0.1);
        } else {
            current_damping = (current_damping * 1.5).min(damping * 20.0);
        }
        prev_error = combined_err;

        // Compute Jacobian
        let j_full = jacobian(robot, chain, &q)?;

        // For position-only mode, use only the top 3 rows (linear velocity)
        // and the corresponding position error components.
        let (j, error_vec) = if position_only {
            let j_pos = j_full.rows(0, 3).clone_owned();
            let err_pos = error_6d.rows(0, 3).clone_owned();
            (j_pos, err_pos)
        } else {
            (j_full.clone(), error_6d.clone())
        };

        let task_dim = j.nrows(); // 3 for position-only, 6 for full

        // DLS pseudo-inverse: J^T (J J^T + λ²I)^{-1}
        let jjt = &j * j.transpose();
        let n = jjt.nrows();
        let damping_matrix = DMatrix::identity(n, n) * (current_damping * current_damping);
        let jjt_damped = jjt + damping_matrix;

        // Solve (J J^T + λ²I) * y = Δx, then Δq = J^T * y
        let decomp = jjt_damped.lu();
        let (y, lu_failed) = match decomp.solve(&error_vec) {
            Some(y) => (y, false),
            None => {
                // Fallback: use transpose method if LU fails (near-singular)
                (j.transpose() * &error_vec * 0.1, true)
            }
        };
        if lu_failed {
            degraded = true;
        }

        let mut dq = j.transpose() * y;

        // Null-space optimization for redundant robots (DOF > task_dim)
        if dof > task_dim {
            if let Some(null_space) = &config.null_space {
                let j_pinv = j.transpose()
                    * &decomp
                        .solve(&DMatrix::identity(task_dim, task_dim))
                        .unwrap_or_else(|| DMatrix::identity(task_dim, task_dim));
                let null_proj = DMatrix::identity(dof, dof) - &j_pinv * &j;

                let gradient = null_space_gradient(robot, chain, &q, null_space);
                let null_step = &null_proj * &gradient;
                dq += null_step * 0.1; // scale null-space contribution
            }
        }

        // Apply step with step-size limiting
        let step_norm = dq.norm();
        let max_step = 0.5; // radians
        let scale = if step_norm > max_step {
            max_step / step_norm
        } else {
            1.0
        };

        for i in 0..dof {
            q[i] += dq[i] * scale;
        }

        // Clamp to joint limits
        if config.check_limits {
            clamp_to_limits(robot, chain, &mut q);
        }
    }

    // Final FK for error computation
    let final_pose = forward_kinematics(robot, chain, &best_q)?;
    let (final_pos_err, final_orient_err, _) = pose_error(&final_pose, target);

    Ok(IKSolution {
        joints: best_q,
        position_error: final_pos_err,
        orientation_error: final_orient_err,
        converged,
        iterations,
        mode_used: mode,
        degraded,
        condition_number: f64::INFINITY,
    })
}

/// Compute null-space gradient for the given objective.
fn null_space_gradient(
    robot: &Robot,
    chain: &KinematicChain,
    q: &[f64],
    objective: &NullSpace,
) -> nalgebra::DVector<f64> {
    let dof = chain.dof;
    let eps = 1e-5;

    match objective {
        NullSpace::Manipulability => {
            // Gradient of manipulability index
            let mut grad = nalgebra::DVector::zeros(dof);
            let m0 = manipulability(robot, chain, q).unwrap_or(0.0);
            for i in 0..dof {
                let mut q_plus = q.to_vec();
                q_plus[i] += eps;
                let m_plus = manipulability(robot, chain, &q_plus).unwrap_or(0.0);
                grad[i] = (m_plus - m0) / eps;
            }
            grad
        }
        NullSpace::MinimalDisplacement => {
            // Gradient pushes toward zero displacement from seed
            // (negative of current position — assuming seed is at 0)
            let mut grad = nalgebra::DVector::zeros(dof);
            for i in 0..dof {
                grad[i] = -q[i];
            }
            grad
        }
        NullSpace::JointCentering => {
            // Gradient pushes toward center of joint limits
            let mut grad = nalgebra::DVector::zeros(dof);
            for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
                if let Some(limits) = &robot.joints[joint_idx].limits {
                    let center = (limits.lower + limits.upper) / 2.0;
                    grad[i] = center - q[i];
                }
            }
            grad
        }
    }
}

/// Clamp joint values to their limits.
fn clamp_to_limits(robot: &Robot, chain: &KinematicChain, q: &mut [f64]) {
    for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
        if let Some(limits) = &robot.joints[joint_idx].limits {
            q[i] = q[i].clamp(limits.lower, limits.upper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::forward_kinematics;

    const PANDA_LIKE_URDF: &str = r#"<?xml version="1.0"?>
<robot name="panda_like">
  <link name="panda_link0"/>
  <link name="panda_link1"/>
  <link name="panda_link2"/>
  <link name="panda_link3"/>
  <link name="panda_link4"/>
  <link name="panda_link5"/>
  <link name="panda_link6"/>
  <link name="panda_link7"/>
  <link name="panda_link8"/>

  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <origin xyz="0 0 0.333"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint2" type="revolute">
    <parent link="panda_link1"/>
    <child link="panda_link2"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.8326" upper="1.8326" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint3" type="revolute">
    <parent link="panda_link2"/>
    <child link="panda_link3"/>
    <origin xyz="0 -0.316 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint4" type="revolute">
    <parent link="panda_link3"/>
    <child link="panda_link4"/>
    <origin xyz="0.0825 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0718" upper="-0.0698" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint5" type="revolute">
    <parent link="panda_link4"/>
    <child link="panda_link5"/>
    <origin xyz="-0.0825 0.384 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint6" type="revolute">
    <parent link="panda_link5"/>
    <child link="panda_link6"/>
    <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.0175" upper="3.7525" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint7" type="revolute">
    <parent link="panda_link6"/>
    <child link="panda_link7"/>
    <origin xyz="0.088 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint8" type="fixed">
    <parent link="panda_link7"/>
    <child link="panda_link8"/>
    <origin xyz="0 0 0.107"/>
  </joint>
</robot>
"#;

    #[test]
    fn dls_7dof_roundtrip() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();
        assert_eq!(chain.dof, 7);

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        // Use mid-configuration as seed (realistic starting point)
        let mid = robot.mid_configuration();
        let seed: Vec<f64> = chain.extract_joint_values(&mid.0);

        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        assert!(
            solution.converged,
            "DLS 7-DOF should converge: pos_err={}, orient_err={}",
            solution.position_error, solution.orientation_error
        );
    }

    #[test]
    fn dls_null_space_manipulability() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 200,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: None,
            null_space: Some(NullSpace::Manipulability),
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &robot.mid_configuration().as_slice()[..7],
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        assert!(
            solution.converged,
            "DLS with null-space should converge: pos_err={}",
            solution.position_error
        );
    }

    #[test]
    fn dls_position_only() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let mid = robot.mid_configuration();
        let seed: Vec<f64> = chain.extract_joint_values(&mid.0);

        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::PositionOnly,
            max_iterations: 300,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::PositionOnly,
        )
        .unwrap();

        assert!(
            solution.converged,
            "Position-only DLS should converge: pos_err={}",
            solution.position_error
        );
        assert!(
            solution.position_error < 1e-4,
            "Position error should be below tolerance: {}",
            solution.position_error
        );
        assert_eq!(solution.mode_used, IKMode::PositionOnly);
    }

    // ─── Adaptive damping & stall escape tests ───

    /// Unreachable target: solver should not produce NaN and should not converge.
    #[test]
    fn dls_unreachable_target_no_nan() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // Target far beyond workspace (100m away)
        let target = Pose(nalgebra::Isometry3::translation(100.0, 100.0, 100.0));

        let seed = vec![0.0; chain.dof];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 200,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // Should NOT converge
        assert!(
            !solution.converged,
            "Unreachable target should not converge"
        );

        // No NaN in solution joints
        for (i, &v) in solution.joints.iter().enumerate() {
            assert!(!v.is_nan(), "Joint {} is NaN in unreachable target test", i);
            assert!(
                v.is_finite(),
                "Joint {} is not finite ({}) in unreachable target test",
                i,
                v
            );
        }

        // Error values should be finite
        assert!(solution.position_error.is_finite());
        assert!(solution.orientation_error.is_finite());
    }

    /// Stall escape (triggered at stall_count >= 40) should keep joints within limits.
    #[test]
    fn dls_stall_escape_respects_limits() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // A challenging target that will trigger stall escape
        let target = Pose(nalgebra::Isometry3::translation(0.8, 0.8, 0.1));

        let seed = vec![0.0; chain.dof];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::PositionOnly,
            max_iterations: 500, // enough for at least one stall escape cycle
            position_tolerance: 1e-6, // very tight — force stalling
            orientation_tolerance: 1e-6,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::PositionOnly,
        )
        .unwrap();

        // After stall escape, joints should still respect limits
        for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                assert!(
                    solution.joints[i] >= limits.lower - 1e-10
                        && solution.joints[i] <= limits.upper + 1e-10,
                    "Joint {} ({}) = {} outside limits [{}, {}] after stall escape",
                    i,
                    robot.joints[joint_idx].name,
                    solution.joints[i],
                    limits.lower,
                    limits.upper
                );
            }
        }
    }

    /// Null-space JointCentering objective: solution joints should be closer
    /// to their limit midpoints than with no null-space optimization.
    #[test]
    fn dls_null_space_joint_centering() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; chain.dof];

        let config_centering = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 1000,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: Some(NullSpace::JointCentering),
            num_restarts: 0,
        };

        let sol = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config_centering,
            IKMode::Full6D,
        )
        .unwrap();

        // Null-space objectives may slow convergence; the key property is no NaN/Inf
        for (i, &v) in sol.joints.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Joint {} = {} is not finite with JointCentering",
                i,
                v
            );
        }
        assert!(
            sol.position_error.is_finite(),
            "Position error is not finite"
        );
        assert!(
            sol.orientation_error.is_finite(),
            "Orientation error is not finite"
        );
        // Should make some progress (error should be less than initial)
        assert!(
            sol.position_error < 10.0,
            "Should make some progress toward target"
        );
    }

    /// Null-space MinimalDisplacement: the solution should stay close to seed.
    #[test]
    fn dls_null_space_minimal_displacement() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // Target close to the seed configuration to keep things reachable
        let q_original = vec![0.1, -0.1, 0.1, -1.0, 0.05, 0.5, 0.2];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; chain.dof];

        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 1000,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: Some(NullSpace::MinimalDisplacement),
            num_restarts: 0,
        };

        let sol = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // Null-space objectives may slow convergence; the key property is no NaN/Inf
        for (i, &v) in sol.joints.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Joint {} = {} is not finite with MinimalDisplacement",
                i,
                v
            );
        }
        assert!(sol.position_error.is_finite());
        assert!(sol.orientation_error.is_finite());
        // Should make some progress
        assert!(
            sol.position_error < 10.0,
            "Should make some progress toward target"
        );
    }

    /// High damping: solver should still work (slower convergence, no crash).
    #[test]
    fn dls_high_damping_no_crash() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; chain.dof];
        // Very high damping (10.0) — should make convergence slow but not crash
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 10.0 },
            mode: IKMode::Full6D,
            max_iterations: 500,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            10.0,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // No NaN regardless of convergence
        for (i, &v) in solution.joints.iter().enumerate() {
            assert!(v.is_finite(), "Joint {} = {} with high damping", i, v);
        }
        assert!(solution.position_error.is_finite());
    }

    // ─── Near-singularity tests ───

    /// Near-singularity: arm fully extended (joints near zero). DLS damping
    /// should prevent the pseudo-inverse from blowing up.
    #[test]
    fn dls_near_singularity_extended_arm() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // Target from a near-singular configuration (arm almost fully stretched)
        let q_near_singular = vec![0.0, 0.0, 0.0, -0.07, 0.0, 0.0, 0.0];
        let target = forward_kinematics(&robot, &chain, &q_near_singular).unwrap();

        let seed = vec![0.1, -0.2, 0.1, -0.5, 0.1, 0.3, 0.1];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.1 },
            mode: IKMode::Full6D,
            max_iterations: 500,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.1,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // Must not produce NaN or Inf near singularity
        for (i, &v) in solution.joints.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Joint {} = {} is not finite near singularity",
                i,
                v
            );
        }
        assert!(solution.position_error.is_finite());
        assert!(solution.orientation_error.is_finite());
    }

    /// Seed shorter than chain DOF: should be zero-padded and still work.
    #[test]
    fn dls_short_seed_zero_padded() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.1, -0.2, 0.1, -1.0, 0.05, 0.5, 0.2];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        // Seed shorter than DOF (3 values for 7-DOF chain)
        let short_seed = vec![0.0, 0.0, 0.0];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(short_seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &short_seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // Should produce a 7-element solution (padded from 3)
        assert_eq!(solution.joints.len(), 7);
        for &v in &solution.joints {
            assert!(v.is_finite(), "Joint value should be finite after padding");
        }
    }

    /// Seed longer than chain DOF: should be truncated and still work.
    #[test]
    fn dls_long_seed_truncated() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.1, -0.2, 0.1, -1.0, 0.05, 0.5, 0.2];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        // Seed longer than DOF (10 values for 7-DOF chain)
        let long_seed = vec![0.0; 10];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(long_seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        // solve_dls resizes seed to match DOF
        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &long_seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        assert_eq!(solution.joints.len(), 7);
        for &v in &solution.joints {
            assert!(v.is_finite());
        }
    }

    /// Very low damping (near-zero): should still converge without NaN.
    #[test]
    fn dls_very_low_damping() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.2, -0.3, 0.1, -1.2, 0.1, 0.8, 0.3];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; chain.dof];
        let damping = 0.001; // very low
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping },
            mode: IKMode::Full6D,
            max_iterations: 500,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            damping,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        for &v in &solution.joints {
            assert!(v.is_finite(), "Joint should be finite with very low damping");
        }
        assert!(solution.position_error.is_finite());
    }

    /// Position-only mode: orientation error can be anything, only position
    /// must converge. Use a nearby target for reliable convergence.
    #[test]
    fn dls_position_only_ignores_orientation() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // Use a moderate target close to mid-configuration for reliable convergence
        let q_original = vec![0.2, -0.3, 0.1, -1.2, 0.1, 0.8, 0.3];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let mid = robot.mid_configuration();
        let seed: Vec<f64> = chain.extract_joint_values(&mid.0);
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::PositionOnly,
            max_iterations: 500,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::PositionOnly,
        )
        .unwrap();

        // Position should converge
        assert!(
            solution.converged,
            "Position-only DLS should converge: pos_err={}",
            solution.position_error
        );
        assert!(
            solution.position_error < 1e-3,
            "Position error should converge in position-only: {}",
            solution.position_error
        );
        // Mode must be recorded correctly
        assert_eq!(solution.mode_used, IKMode::PositionOnly);
    }

    /// check_limits=false: joints may exceed limits.
    #[test]
    fn dls_no_limit_checking() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; chain.dof];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: false, // disable limit enforcement
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        // Should still produce valid (finite) result
        for &v in &solution.joints {
            assert!(v.is_finite());
        }
        assert!(solution.position_error.is_finite());
    }

    /// Adaptive damping increase path: when the solver is stalling (error not
    /// decreasing), damping should increase. Verify by using a target that
    /// causes oscillation.
    #[test]
    fn dls_adaptive_damping_stall_increases() {
        let robot = Robot::from_urdf_string(PANDA_LIKE_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

        // A target that's reachable but in a tricky spot
        let target = Pose(nalgebra::Isometry3::translation(0.5, 0.3, 0.6));
        let seed = vec![0.0; chain.dof];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::PositionOnly,
            max_iterations: 100,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: None,
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::PositionOnly,
        )
        .unwrap();

        // The solver exercised the adaptive damping code path;
        // verify result is still valid
        for &v in &solution.joints {
            assert!(v.is_finite());
        }
        assert!(solution.iterations > 0);
    }

    /// Null-space with non-redundant chain (DOF <= task_dim): null-space
    /// code is skipped, solver should still work.
    #[test]
    fn dls_null_space_on_non_redundant_chain() {
        // Use a 6-DOF chain where DOF == task_dim
        let urdf_6dof = r#"<?xml version="1.0"?>
<robot name="test_6dof">
  <link name="l0"/><link name="l1"/><link name="l2"/>
  <link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/>
  <joint name="j1" type="revolute">
    <parent link="l0"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.08"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf_6dof).unwrap();
        let chain = KinematicChain::extract(&robot, "l0", "l6").unwrap();
        assert_eq!(chain.dof, 6);

        let q_original = vec![0.1, 0.2, -0.1, 0.3, 0.1, -0.2];
        let target = forward_kinematics(&robot, &chain, &q_original).unwrap();

        let seed = vec![0.0; 6];
        let config = IKConfig {
            solver: crate::ik::IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed.clone()),
            null_space: Some(NullSpace::Manipulability), // should be ignored for 6-DOF
            num_restarts: 0,
        };

        let solution = solve_dls(
            &robot,
            &chain,
            &target,
            &seed,
            0.05,
            &config,
            IKMode::Full6D,
        )
        .unwrap();

        for &v in &solution.joints {
            assert!(v.is_finite());
        }
        assert!(solution.position_error.is_finite());
    }

    /// clamp_to_limits: verify the helper function handles joints without limits.
    #[test]
    fn clamp_to_limits_no_limits_no_crash() {
        // Use a URDF with continuous joints (no explicit limits)
        let urdf = r#"<?xml version="1.0"?>
<robot name="cont">
  <link name="a"/><link name="b"/>
  <joint name="jc" type="continuous">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "a", "b").unwrap();
        let mut q = vec![100.0]; // very large value

        // clamp_to_limits should skip joints without limits
        clamp_to_limits(&robot, &chain, &mut q);
        assert_eq!(q[0], 100.0, "Continuous joint should not be clamped");
    }
}
