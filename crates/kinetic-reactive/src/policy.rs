//! Per-policy evaluators for RMP.
//!
//! Each function computes a (joint_acceleration, joint_metric) pair for one
//! [`PolicyType`] variant in joint space. The metrics are combined by [`RMP::compute`]
//! via the metric-weighted-average formula described in the crate-level docs.

use nalgebra::{DMatrix, DVector, Isometry3};

use kinetic_kinematics::KinematicChain;
use kinetic_robot::Robot;
use kinetic_scene::Scene;

use crate::RobotState;

/// Reach-target policy: attract end-effector toward `target_pose` with critically-damped gain.
pub(crate) fn evaluate_reach_target(
    state: &RobotState,
    target_pose: &Isometry3<f64>,
    gain: f64,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let jac = &state.jacobian;
    let ee_iso = &state.ee_pose.0;

    // Position error
    let pos_err = target_pose.translation.vector - ee_iso.translation.vector;

    // Orientation error (log map of rotation error)
    let rot_err_mat = target_pose.rotation * ee_iso.rotation.inverse();
    let angle = rot_err_mat.angle();
    let ori_err = if angle.abs() > 1e-10 {
        rot_err_mat
            .axis()
            .map_or(nalgebra::Vector3::zeros(), |ax| ax.into_inner() * angle)
    } else {
        nalgebra::Vector3::zeros()
    };

    // 6D task-space acceleration
    let mut task_accel = DVector::zeros(6);
    let g = gain;
    let g_sqrt = g.sqrt();
    let vel_dv = DVector::from_column_slice(&state.joint_velocities);
    let task_vel = jac * &vel_dv;
    for k in 0..3 {
        task_accel[k] = g * pos_err[k] - 2.0 * g_sqrt * task_vel[k];
    }
    for k in 0..3 {
        task_accel[3 + k] = g * ori_err[k] - 2.0 * g_sqrt * task_vel[3 + k];
    }

    // Task-space metric: identity scaled by gain
    let task_metric = DMatrix::<f64>::identity(6, 6) * gain;

    // Pull back to joint space: a_joint = J^T * a_task, M_joint = J^T * M_task * J
    let jt = jac.transpose();
    let joint_accel = &jt * &task_accel;
    let joint_metric = &jt * &task_metric * jac;

    Ok((joint_accel, joint_metric))
}

/// Avoid-obstacles policy: repel from scene obstacles within `influence_distance`.
pub(crate) fn evaluate_avoid_obstacles(
    state: &RobotState,
    scene: &Scene,
    influence_distance: f64,
    gain: f64,
    dof: usize,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let min_dist = scene
        .min_distance_to_robot(&state.joint_positions)
        .unwrap_or(f64::INFINITY);

    let mut joint_accel = DVector::zeros(dof);
    let mut joint_metric = DMatrix::zeros(dof, dof);

    if min_dist < influence_distance && min_dist > 1e-6 {
        // Repulsive field strength: increases as distance decreases
        let alpha = 1.0 - min_dist / influence_distance;
        let strength = gain * alpha * alpha;

        // Use Jacobian to map repulsion to joint space
        // Approximate: push in direction that increases distance
        // Use gradient of distance w.r.t. joint angles (finite difference)
        let eps = 1e-4;
        let mut grad = DVector::zeros(dof);
        for j in 0..dof {
            let mut q_plus = state.joint_positions.clone();
            q_plus[j] += eps;
            let d_plus = scene.min_distance_to_robot(&q_plus).unwrap_or(min_dist);
            grad[j] = (d_plus - min_dist) / eps;
        }

        // Acceleration: push along gradient (increase distance)
        joint_accel = &grad * strength;

        // Metric: rank-1 update along gradient direction
        let grad_norm = grad.norm();
        if grad_norm > 1e-10 {
            let n = &grad / grad_norm;
            joint_metric = &n * n.transpose() * strength;
        }
    }

    Ok((joint_accel, joint_metric))
}

/// Avoid-self-collision policy: simple joint-space velocity damping.
pub(crate) fn evaluate_avoid_self_collision(
    state: &RobotState,
    gain: f64,
    dof: usize,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    // Simple joint-space self-collision avoidance using velocity damping
    let joint_metric = DMatrix::<f64>::identity(dof, dof) * (gain * 0.1);

    // Use velocity damping scaled by proximity to self-collision
    let vel = DVector::from_column_slice(&state.joint_velocities);
    let joint_accel = -vel * (gain * 0.1);

    Ok((joint_accel, joint_metric))
}

/// Joint-limit-avoidance policy: repulsive force when near or past joint limits.
pub(crate) fn evaluate_joint_limit_avoidance(
    state: &RobotState,
    robot: &Robot,
    chain: &KinematicChain,
    margin: f64,
    gain: f64,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let dof = chain.dof;
    let mut joint_accel = DVector::zeros(dof);
    let mut joint_metric = DMatrix::zeros(dof, dof);

    for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
        if i >= dof {
            break;
        }
        let joint = &robot.joints[joint_idx];
        let limits = match &joint.limits {
            Some(l) => l,
            None => continue,
        };
        let q = state.joint_positions[i];
        let lower = limits.lower;
        let upper = limits.upper;

        // Distance to lower limit
        let dist_lower = q - lower;
        // Distance to upper limit
        let dist_upper = upper - q;

        // Repulsive acceleration when near limits
        if dist_lower < margin && dist_lower > 0.0 {
            let alpha = 1.0 - dist_lower / margin;
            joint_accel[i] += gain * alpha * alpha;
            joint_metric[(i, i)] += gain * alpha;
        }
        if dist_upper < margin && dist_upper > 0.0 {
            let alpha = 1.0 - dist_upper / margin;
            joint_accel[i] -= gain * alpha * alpha;
            joint_metric[(i, i)] += gain * alpha;
        }

        // Push back if already past limits
        if dist_lower <= 0.0 {
            joint_accel[i] += gain * 2.0;
            joint_metric[(i, i)] += gain * 2.0;
        }
        if dist_upper <= 0.0 {
            joint_accel[i] -= gain * 2.0;
            joint_metric[(i, i)] += gain * 2.0;
        }
    }

    Ok((joint_accel, joint_metric))
}

/// Singularity-avoidance policy: damping when manipulability falls below `threshold`.
pub(crate) fn evaluate_singularity_avoidance(
    state: &RobotState,
    threshold: f64,
    gain: f64,
    dof: usize,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let joint_accel = DVector::zeros(dof);
    let mut joint_metric = DMatrix::<f64>::zeros(dof, dof);

    if state.manipulability < threshold {
        // Near singularity: add strong damping to slow down
        let alpha = 1.0 - state.manipulability / threshold;
        let damping = gain * alpha * alpha;
        joint_metric = DMatrix::<f64>::identity(dof, dof) * damping;
    }

    Ok((joint_accel, joint_metric))
}

/// Damping policy: pure velocity damping, a = -c * v.
pub(crate) fn evaluate_damping(
    state: &RobotState,
    coefficient: f64,
    dof: usize,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let vel = DVector::from_column_slice(&state.joint_velocities);
    let joint_accel = -vel * coefficient;
    let joint_metric = DMatrix::<f64>::identity(dof, dof) * coefficient;

    Ok((joint_accel, joint_metric))
}
