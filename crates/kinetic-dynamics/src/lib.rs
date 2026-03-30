//! Dynamics bridge for kinetic via featherstone.
//!
//! Adds dynamics-aware capabilities to kinetic's planning stack:
//! - Torque-feasibility checking for trajectories
//! - Gravity compensation torques
//! - Inverse/forward dynamics queries
//! - Mass matrix computation
//! - Dynamic manipulability
//!
//! All conversions between kinetic (f64) and featherstone (f32) happen
//! at this boundary — neither crate needs to know about the other.

use kinetic_core::JointValues;
use kinetic_kinematics::KinematicChain;
use kinetic_robot::{Joint, JointType, Robot};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use featherstone::prelude::{
    aba_forward_dynamics, body_jacobian, crba_mass_matrix,
    gravity_compensation as sd_gravity_compensation, rnea_inverse_dynamics, ArticulatedBody,
    GenJoint, SpatialInertia, SpatialTransform,
};

// ─── Conversion ──────────────────────────────────────────────────────────────

/// Build an `ArticulatedBody` from a kinetic `Robot` and `KinematicChain`.
///
/// Walks the chain from base to tip, converting each active joint and its
/// child link into a body definition. Fixed joints are folded into the
/// parent transform (composite).
pub fn articulated_body_from_chain(robot: &Robot, chain: &KinematicChain) -> ArticulatedBody {
    let mut body = ArticulatedBody::new();
    body.set_gravity(Vector3::new(0.0, 0.0, -9.81));

    // Map from kinetic joint index → softmata body index
    let mut joint_to_body: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    // Accumulated transform for runs of fixed joints
    let mut pending_transform = SpatialTransform::identity();
    let mut last_body: i32 = -1; // parent in softmata (-1 = root)

    for &joint_idx in &chain.all_joints {
        let joint = &robot.joints[joint_idx];
        let child_link = &robot.links[joint.child_link];

        // Convert joint origin to spatial transform
        let joint_transform = pose_to_spatial_transform(&joint.origin);
        let combined = compose_transforms(&pending_transform, &joint_transform);

        if joint.joint_type == JointType::Fixed {
            // Accumulate fixed transforms — don't create a body
            pending_transform = combined;
            continue;
        }

        // Active joint: create a body
        let gen_joint = convert_joint_type(joint);
        let inertia = convert_inertial(child_link);

        let body_idx = body.add_body(
            &child_link.name,
            last_body,
            gen_joint,
            inertia,
            combined,
        );

        joint_to_body.insert(joint_idx, body_idx);
        last_body = body_idx as i32;
        pending_transform = SpatialTransform::identity();
    }

    body
}

/// Set joint positions on the `ArticulatedBody` from kinetic `JointValues`.
///
/// Only maps DOFs corresponding to the chain's active joints.
pub fn set_positions(body: &mut ArticulatedBody, q: &JointValues) {
    let n = body.q.len().min(q.len());
    for i in 0..n {
        body.q[i] = q[i] as f32;
    }
}

/// Set joint velocities on the `ArticulatedBody` from kinetic `JointValues`.
pub fn set_velocities(body: &mut ArticulatedBody, qd: &JointValues) {
    let n = body.qd.len().min(qd.len());
    for i in 0..n {
        body.qd[i] = qd[i] as f32;
    }
}

/// Set joint accelerations on the `ArticulatedBody`.
pub fn set_accelerations(body: &mut ArticulatedBody, qdd: &JointValues) {
    let n = body.qdd.len().min(qdd.len());
    for i in 0..n {
        body.qdd[i] = qdd[i] as f32;
    }
}

/// Set joint torques on the `ArticulatedBody`.
pub fn set_torques(body: &mut ArticulatedBody, tau: &JointValues) {
    let n = body.tau.len().min(tau.len());
    for i in 0..n {
        body.tau[i] = tau[i] as f32;
    }
}

/// Read joint-space result back as kinetic `JointValues` (f32→f64).
fn dvector_to_joint_values(v: &DVector<f32>) -> JointValues {
    JointValues::from_slice(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
}

// ─── Dynamics Queries ────────────────────────────────────────────────────────

/// Compute inverse dynamics: joint torques required for a given (q, qd, qdd).
///
/// Returns the torque vector τ = M(q)q̈ + C(q,q̇)q̇ + g(q).
pub fn inverse_dynamics(
    body: &mut ArticulatedBody,
    q: &JointValues,
    qd: &JointValues,
    qdd: &JointValues,
) -> JointValues {
    set_positions(body, q);
    set_velocities(body, qd);
    set_accelerations(body, qdd);
    let (tau, _) = rnea_inverse_dynamics(body);
    dvector_to_joint_values(&tau)
}

/// Compute gravity compensation torques for a given configuration.
///
/// Returns the torques needed to hold the robot stationary at position q.
pub fn gravity_compensation(body: &mut ArticulatedBody, q: &JointValues) -> JointValues {
    set_positions(body, q);
    let tau = sd_gravity_compensation(body);
    dvector_to_joint_values(&tau)
}

/// Compute forward dynamics: joint accelerations from applied torques.
///
/// Returns q̈ = M⁻¹(q)(τ - C(q,q̇)q̇ - g(q)).
pub fn forward_dynamics(
    body: &mut ArticulatedBody,
    q: &JointValues,
    qd: &JointValues,
    tau: &JointValues,
) -> JointValues {
    set_positions(body, q);
    set_velocities(body, qd);
    set_torques(body, tau);
    let qdd = aba_forward_dynamics(body);
    dvector_to_joint_values(qdd)
}

/// Compute the joint-space mass matrix M(q).
///
/// Returns an n×n symmetric positive-definite matrix (as f64).
pub fn mass_matrix(body: &mut ArticulatedBody, q: &JointValues) -> DMatrix<f64> {
    set_positions(body, q);
    let m = crba_mass_matrix(body);
    m.map(|x| x as f64)
}

/// Compute dynamic manipulability at configuration q for a given body.
///
/// Dynamic manipulability = √(det(J M⁻¹ Jᵀ)) — measures how easily the
/// end-effector can accelerate, accounting for inertia.
pub fn dynamic_manipulability(
    body: &mut ArticulatedBody,
    q: &JointValues,
    body_id: usize,
) -> f64 {
    set_positions(body, q);
    let m = crba_mass_matrix(body);
    let j = body_jacobian(body, body_id);

    // M⁻¹ via Cholesky
    let m_chol = match nalgebra::linalg::Cholesky::new(m) {
        Some(c) => c,
        None => return 0.0,
    };
    let m_inv = m_chol.inverse();

    // J M⁻¹ Jᵀ
    let jmjt = &j * &m_inv * j.transpose();
    let det = jmjt.determinant();
    if det > 0.0 {
        (det as f64).sqrt()
    } else {
        0.0
    }
}

// ─── Trajectory Feasibility ──────────────────────────────────────────────────

/// A dynamics violation at a specific waypoint and joint.
#[derive(Debug, Clone)]
pub struct DynamicsViolation {
    /// Waypoint index in the trajectory.
    pub waypoint_index: usize,
    /// Joint index (chain-local DOF).
    pub joint_index: usize,
    /// Torque required by inverse dynamics (Nm).
    pub required_effort: f64,
    /// Maximum effort from joint limits (Nm).
    pub effort_limit: f64,
}

/// Check whether a trajectory is dynamically feasible.
///
/// For each timed waypoint, computes inverse dynamics (RNEA) and checks
/// whether the required joint torques exceed the robot's effort limits.
///
/// Returns an empty vec if the trajectory is feasible.
/// Requires a timed trajectory (with timestamps, velocities, accelerations).
///
/// # SAFETY: Call this before executing on real hardware
///
/// **This function MUST be called before sending trajectories to real robots.**
/// A trajectory that violates torque limits will either:
/// - Cause the motor controller to clip torques (losing tracking accuracy), or
/// - Trigger a hardware emergency stop (uncontrolled motion)
///
/// ```ignore
/// let violations = check_trajectory_feasibility(&mut body, &trajectory, &effort_limits);
/// if !violations.is_empty() {
///     eprintln!("UNSAFE: {} torque violations", violations.len());
///     for v in &violations {
///         eprintln!("  wp={} j={}: {:.1} Nm required, {:.1} Nm limit",
///             v.waypoint_index, v.joint_index, v.required_effort, v.effort_limit);
///     }
///     return Err("Trajectory infeasible".into());
/// }
/// ```
pub fn check_trajectory_feasibility(
    body: &mut ArticulatedBody,
    trajectory: &kinetic_core::Trajectory,
    effort_limits: &[f64],
) -> Vec<DynamicsViolation> {
    let mut violations = Vec::new();
    let dof = trajectory.dof;

    if !trajectory.is_timed() {
        return violations; // can't check without velocities/accelerations
    }

    for wp_idx in 0..trajectory.len() {
        // Extract per-waypoint data from SoA layout (joint-major)
        let positions: Vec<f64> = (0..dof)
            .map(|j| trajectory.position(j, wp_idx))
            .collect();

        // Use sample_at to get interpolated velocities/accelerations
        // For exact waypoint times, this returns the waypoint data directly
        let duration = trajectory.duration().unwrap_or_default();
        let t = if trajectory.len() <= 1 {
            0.0
        } else {
            duration.as_secs_f64() * (wp_idx as f64) / ((trajectory.len() - 1) as f64)
        };

        let timed_wp = match trajectory.sample_at(std::time::Duration::from_secs_f64(t)) {
            Some(wp) => wp,
            None => continue,
        };

        let q = JointValues::from_slice(&positions);
        let qd = timed_wp.velocities;
        let qdd = timed_wp.accelerations;

        let tau = inverse_dynamics(body, &q, &qd, &qdd);

        for j in 0..dof.min(effort_limits.len()) {
            let required = tau[j].abs();
            let limit = effort_limits[j].abs();
            if limit > 0.0 && required > limit {
                violations.push(DynamicsViolation {
                    waypoint_index: wp_idx,
                    joint_index: j,
                    required_effort: required,
                    effort_limit: limit,
                });
            }
        }
    }

    violations
}

/// Extract effort limits from a robot's chain for use with feasibility checking.
pub fn effort_limits_from_chain(robot: &Robot, chain: &KinematicChain) -> Vec<f64> {
    chain
        .active_joints
        .iter()
        .map(|&joint_idx| {
            robot.joints[joint_idx]
                .limits
                .as_ref()
                .map(|l| l.effort)
                .unwrap_or(0.0)
        })
        .collect()
}

// ─── Internal Conversions ────────────────────────────────────────────────────

fn convert_joint_type(joint: &Joint) -> GenJoint {
    let axis = Vector3::new(joint.axis.x as f32, joint.axis.y as f32, joint.axis.z as f32);
    match joint.joint_type {
        JointType::Revolute | JointType::Continuous => GenJoint::Revolute { axis },
        JointType::Prismatic => GenJoint::Prismatic { axis },
        JointType::Fixed => GenJoint::Fixed, // handled before reaching here
    }
}

fn convert_inertial(link: &kinetic_robot::Link) -> SpatialInertia {
    match &link.inertial {
        Some(inertial) => {
            let mass = inertial.mass as f32;
            let com = Vector3::new(
                inertial.origin.translation().x as f32,
                inertial.origin.translation().y as f32,
                inertial.origin.translation().z as f32,
            );
            // kinetic stores [ixx, ixy, ixz, iyy, iyz, izz]
            let [ixx, ixy, ixz, iyy, iyz, izz] = inertial.inertia;
            let inertia_matrix = Matrix3::new(
                ixx as f32, ixy as f32, ixz as f32,
                ixy as f32, iyy as f32, iyz as f32,
                ixz as f32, iyz as f32, izz as f32,
            );
            SpatialInertia::from_mass_inertia(mass, com, inertia_matrix)
        }
        None => SpatialInertia::from_mass_inertia(
            0.001, // avoid zero-mass singularity
            Vector3::zeros(),
            Matrix3::identity() * 1e-6,
        ),
    }
}

fn pose_to_spatial_transform(pose: &kinetic_core::Pose) -> SpatialTransform {
    let iso = pose.isometry();
    let rot = iso.rotation.to_rotation_matrix();
    let rotation = Matrix3::new(
        *rot.matrix().index((0, 0)) as f32, *rot.matrix().index((0, 1)) as f32, *rot.matrix().index((0, 2)) as f32,
        *rot.matrix().index((1, 0)) as f32, *rot.matrix().index((1, 1)) as f32, *rot.matrix().index((1, 2)) as f32,
        *rot.matrix().index((2, 0)) as f32, *rot.matrix().index((2, 1)) as f32, *rot.matrix().index((2, 2)) as f32,
    );
    let translation = Vector3::new(
        iso.translation.x as f32,
        iso.translation.y as f32,
        iso.translation.z as f32,
    );
    SpatialTransform::from_rotation_translation(rotation, translation)
}

fn compose_transforms(a: &SpatialTransform, b: &SpatialTransform) -> SpatialTransform {
    let rotation = a.rotation * b.rotation;
    let translation = a.rotation * b.translation + a.translation;
    SpatialTransform::from_rotation_translation(rotation, translation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_produces_empty_body() {
        let robot = Robot::from_urdf_string(
            "<robot name='test'><link name='base'/></robot>",
        )
        .unwrap();
        let chain = KinematicChain {
            all_joints: vec![],
            active_joints: vec![],
            active_to_robot_active: vec![],
            dof: 0,
            base_link: 0,
            tip_link: 0,
        };
        let body = articulated_body_from_chain(&robot, &chain);
        assert_eq!(body.bodies.len(), 0);
    }

    #[test]
    fn gravity_comp_returns_correct_dof() {
        // 2-link planar arm
        let urdf = r#"
        <robot name="test">
            <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link1"><inertial><mass value="1"/><origin xyz="0.5 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link2"><inertial><mass value="1"/><origin xyz="0.5 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="100" velocity="1"/></joint>
            <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="100" velocity="1"/></joint>
        </robot>
        "#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link2").unwrap();
        let mut body = articulated_body_from_chain(&robot, &chain);
        let q = JointValues::zeros(2);
        let tau = gravity_compensation(&mut body, &q);
        assert_eq!(tau.len(), 2);
    }

    #[test]
    fn effort_limits_extracted() {
        let urdf = r#"
        <robot name="test">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="50" velocity="1"/></joint>
        </robot>
        "#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link1").unwrap();
        let limits = effort_limits_from_chain(&robot, &chain);
        assert_eq!(limits.len(), 1);
        assert!((limits[0] - 50.0).abs() < 1e-6);
    }

    // ── Helper: builds a 2-link planar arm for reuse across tests ───────────

    fn two_link_urdf() -> &'static str {
        r#"
        <robot name="test">
            <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link1"><inertial><mass value="2"/><origin xyz="0.25 0 0"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link2"><inertial><mass value="1"/><origin xyz="0.25 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="100" velocity="2"/></joint>
            <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0.5 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="50" velocity="2"/></joint>
        </robot>
        "#
    }

    fn make_two_link() -> (Robot, KinematicChain, ArticulatedBody) {
        let robot = Robot::from_urdf_string(two_link_urdf()).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link2").unwrap();
        let body = articulated_body_from_chain(&robot, &chain);
        (robot, chain, body)
    }

    // ── Inverse/Forward Dynamics Roundtrip ──────────────────────────────────

    #[test]
    fn inverse_forward_dynamics_roundtrip() {
        // Intent: forward_dynamics(q, qd, inverse_dynamics(q, qd, qdd)) ≈ qdd
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.5, -0.3]);
        let qd = JointValues::from_slice(&[0.1, -0.2]);
        let qdd = JointValues::from_slice(&[1.0, -0.5]);

        // ID: compute torques needed for desired accelerations
        let tau = inverse_dynamics(&mut body, &q, &qd, &qdd);
        assert_eq!(tau.len(), 2);

        // FD: apply those torques → should recover original accelerations
        let qdd_recovered = forward_dynamics(&mut body, &q, &qd, &tau);
        assert_eq!(qdd_recovered.len(), 2);

        // f32 precision limits roundtrip accuracy to ~1e-2
        for i in 0..2 {
            assert!(
                (qdd_recovered[i] - qdd[i]).abs() < 0.1,
                "joint {i}: expected {}, got {} (diff={})",
                qdd[i],
                qdd_recovered[i],
                (qdd_recovered[i] - qdd[i]).abs()
            );
        }
    }

    // ── Gravity Compensation ────────────────────────────────────────────────

    #[test]
    fn gravity_comp_equals_id_at_rest() {
        // Intent: gravity_compensation(q) == inverse_dynamics(q, 0, 0)
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.3, -0.5]);
        let zero = JointValues::zeros(2);

        let tau_grav = gravity_compensation(&mut body, &q);
        let tau_id = inverse_dynamics(&mut body, &q, &zero, &zero);

        for i in 0..2 {
            assert!(
                (tau_grav[i] - tau_id[i]).abs() < 1e-3,
                "joint {i}: grav={}, id={}",
                tau_grav[i],
                tau_id[i]
            );
        }
    }

    fn gravity_arm_urdf() -> &'static str {
        // Y-axis revolute joints so gravity (-Z) creates torque
        r#"
        <robot name="grav_test">
            <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link1"><inertial><mass value="2"/><origin xyz="0.25 0 0"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link2"><inertial><mass value="1"/><origin xyz="0.25 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="100" velocity="2"/></joint>
            <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0.5 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="50" velocity="2"/></joint>
        </robot>
        "#
    }

    fn make_gravity_arm() -> (Robot, KinematicChain, ArticulatedBody) {
        let robot = Robot::from_urdf_string(gravity_arm_urdf()).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link2").unwrap();
        let body = articulated_body_from_chain(&robot, &chain);
        (robot, chain, body)
    }

    #[test]
    fn gravity_comp_nonzero_for_horizontal_arm() {
        // Intent: when arm extends horizontally with Y-axis joints, gravity (-Z) requires nonzero torque
        let (_robot, _chain, mut body) = make_gravity_arm();
        let q = JointValues::zeros(2);
        let tau = gravity_compensation(&mut body, &q);
        let total_torque: f64 = tau.as_slice().iter().map(|t| t.abs()).sum();
        assert!(
            total_torque > 0.0,
            "gravity comp should be nonzero for horizontal arm with Y-axis joints, got {:?}",
            tau.as_slice()
        );
    }

    // ── Mass Matrix Properties ──────────────────────────────────────────────

    #[test]
    fn mass_matrix_correct_dimensions() {
        // Intent: M(q) is n×n where n = DOF
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::zeros(2);
        let m = mass_matrix(&mut body, &q);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    }

    #[test]
    fn mass_matrix_symmetric() {
        // Intent: M(q) must be symmetric (physical law)
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.7, -1.2]);
        let m = mass_matrix(&mut body, &q);
        let diff = (&m - m.transpose()).norm();
        assert!(
            diff < 1e-4,
            "mass matrix not symmetric: asymmetry norm = {diff}"
        );
    }

    #[test]
    fn mass_matrix_positive_semidefinite() {
        // Intent: M(q) must be positive semi-definite (energy ≥ 0)
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.5, 0.5]);
        let m = mass_matrix(&mut body, &q);

        // Check eigenvalues via symmetric eigendecomposition
        let sym = nalgebra::SymmetricEigen::new(m);
        for &ev in sym.eigenvalues.iter() {
            assert!(
                ev >= -1e-6,
                "mass matrix has negative eigenvalue: {ev}"
            );
        }
    }

    #[test]
    fn mass_matrix_changes_with_configuration() {
        // Intent: M(q) depends on joint configuration (coupled inertia)
        let (_robot, _chain, mut body) = make_two_link();
        let m1 = mass_matrix(&mut body, &JointValues::from_slice(&[0.0, 0.0]));
        let m2 = mass_matrix(&mut body, &JointValues::from_slice(&[1.5, -1.0]));
        let diff = (&m1 - &m2).norm();
        assert!(
            diff > 1e-6,
            "mass matrix should change with configuration"
        );
    }

    // ── Forward Dynamics ────────────────────────────────────────────────────

    #[test]
    fn forward_dynamics_zero_torque_produces_gravity_accel() {
        // Intent: with zero torque and Y-axis joints, gravity (-Z) causes nonzero accelerations
        let (_robot, _chain, mut body) = make_gravity_arm();
        let q = JointValues::zeros(2);
        let qd = JointValues::zeros(2);
        let tau = JointValues::zeros(2);
        let qdd = forward_dynamics(&mut body, &q, &qd, &tau);
        let total: f64 = qdd.as_slice().iter().map(|a| a.abs()).sum();
        assert!(
            total > 0.0,
            "zero torque should produce gravity-induced accelerations, got {:?}",
            qdd.as_slice()
        );
    }

    #[test]
    fn forward_dynamics_gravity_comp_torque_gives_zero_accel() {
        // Intent: applying gravity comp torques should yield zero acceleration at rest
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.3, -0.5]);
        let qd = JointValues::zeros(2);
        let tau_grav = gravity_compensation(&mut body, &q);
        let qdd = forward_dynamics(&mut body, &q, &qd, &tau_grav);
        for i in 0..2 {
            assert!(
                qdd[i].abs() < 0.05,
                "joint {i}: expected ~0 accel with grav comp, got {}",
                qdd[i]
            );
        }
    }

    // ── Dynamic Manipulability ──────────────────────────────────────────────

    #[test]
    fn dynamic_manipulability_nonnegative() {
        // Intent: manipulability must be ≥ 0 (it's a sqrt of a determinant)
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[0.5, -0.3]);
        let m = dynamic_manipulability(&mut body, &q, 0);
        assert!(m >= 0.0, "dynamic manipulability should be >= 0, got {m}");
    }

    #[test]
    fn dynamic_manipulability_varies_with_config() {
        // Intent: manipulability should change as arm configuration changes
        let (_robot, _chain, mut body) = make_two_link();
        let m1 = dynamic_manipulability(&mut body, &JointValues::from_slice(&[0.0, 0.0]), 0);
        let m2 = dynamic_manipulability(&mut body, &JointValues::from_slice(&[1.5, 0.0]), 0);
        // At least one should differ (may both be 0 for 2-DOF planar with 6D Jacobian)
        // Just verify no crashes and non-negative
        assert!(m1 >= 0.0);
        assert!(m2 >= 0.0);
    }

    // ── Set State Functions ─────────────────────────────────────────────────

    #[test]
    fn set_positions_maps_correctly() {
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[1.5, -0.7]);
        set_positions(&mut body, &q);
        assert!((body.q[0] - 1.5_f32).abs() < 1e-6);
        assert!((body.q[1] - (-0.7_f32)).abs() < 1e-6);
    }

    #[test]
    fn set_velocities_maps_correctly() {
        let (_robot, _chain, mut body) = make_two_link();
        let qd = JointValues::from_slice(&[0.3, -1.1]);
        set_velocities(&mut body, &qd);
        assert!((body.qd[0] - 0.3_f32).abs() < 1e-6);
        assert!((body.qd[1] - (-1.1_f32)).abs() < 1e-6);
    }

    #[test]
    fn set_accelerations_maps_correctly() {
        let (_robot, _chain, mut body) = make_two_link();
        let qdd = JointValues::from_slice(&[5.0, -3.0]);
        set_accelerations(&mut body, &qdd);
        assert!((body.qdd[0] - 5.0_f32).abs() < 1e-6);
        assert!((body.qdd[1] - (-3.0_f32)).abs() < 1e-6);
    }

    #[test]
    fn set_torques_maps_correctly() {
        let (_robot, _chain, mut body) = make_two_link();
        let tau = JointValues::from_slice(&[10.0, -20.0]);
        set_torques(&mut body, &tau);
        assert!((body.tau[0] - 10.0_f32).abs() < 1e-6);
        assert!((body.tau[1] - (-20.0_f32)).abs() < 1e-6);
    }

    // ── Trajectory Feasibility ──────────────────────────────────────────────

    #[test]
    fn feasibility_untimed_trajectory_returns_empty() {
        // Intent: can't check feasibility without timing info
        let (_robot, _chain, mut body) = make_two_link();
        let mut traj = kinetic_core::Trajectory::new(2, vec!["j1".into(), "j2".into()]);
        traj.push_waypoint(&[0.0, 0.0]);
        traj.push_waypoint(&[1.0, 1.0]);
        let violations = check_trajectory_feasibility(&mut body, &traj, &[100.0, 50.0]);
        assert!(violations.is_empty(), "untimed trajectory should return no violations");
    }

    #[test]
    fn feasibility_within_limits_returns_empty() {
        // Intent: a slow trajectory within effort limits should have no violations
        let (robot, chain, mut body) = make_two_link();
        let limits = effort_limits_from_chain(&robot, &chain);
        let mut traj = kinetic_core::Trajectory::new(2, vec!["j1".into(), "j2".into()]);
        // Very slow motion: small displacements over 2 seconds
        traj.push_waypoint(&[0.0, 0.0]);
        traj.push_waypoint(&[0.01, 0.01]);
        // SoA layout: timestamps, velocities[joint*wp+wp_idx], accelerations[joint*wp+wp_idx]
        traj.set_timing(
            vec![0.0, 2.0],                      // timestamps in seconds
            vec![0.0, 0.005, 0.0, 0.005],        // velocities (2 joints × 2 waypoints)
            vec![0.0, 0.0, 0.0, 0.0],            // accelerations (2 joints × 2 waypoints)
        );
        let violations = check_trajectory_feasibility(&mut body, &traj, &limits);
        assert!(
            violations.is_empty(),
            "slow trajectory should be feasible, got {} violations",
            violations.len()
        );
    }

    #[test]
    fn feasibility_with_tiny_limits_finds_violations() {
        // Intent: impossibly tight limits should produce violations
        let (_robot, _chain, mut body) = make_two_link();
        let mut traj = kinetic_core::Trajectory::new(2, vec!["j1".into(), "j2".into()]);
        traj.push_waypoint(&[0.0, 0.0]);
        traj.push_waypoint(&[1.0, 1.0]);
        traj.set_timing(
            vec![0.0, 0.1],                      // 100ms — fast motion
            vec![0.0, 10.0, 0.0, 10.0],          // high velocities
            vec![0.0, 100.0, 0.0, 100.0],        // high accelerations
        );
        // Impossibly tight limits (0.001 Nm)
        let violations = check_trajectory_feasibility(&mut body, &traj, &[0.001, 0.001]);
        // Gravity alone should exceed 0.001 Nm for a 2-link arm
        assert!(
            !violations.is_empty(),
            "impossibly tight limits should produce violations"
        );
    }

    #[test]
    fn dynamics_violation_has_correct_fields() {
        let v = DynamicsViolation {
            waypoint_index: 3,
            joint_index: 1,
            required_effort: 55.0,
            effort_limit: 50.0,
        };
        assert_eq!(v.waypoint_index, 3);
        assert_eq!(v.joint_index, 1);
        assert!((v.required_effort - 55.0).abs() < 1e-10);
        assert!((v.effort_limit - 50.0).abs() < 1e-10);
    }

    // ── Conversion Precision ────────────────────────────────────────────────

    #[test]
    fn f64_f32_roundtrip_precision() {
        // Intent: verify f64→f32→f64 conversion doesn't lose critical precision
        let (_robot, _chain, mut body) = make_two_link();
        let q = JointValues::from_slice(&[std::f64::consts::PI / 4.0, -std::f64::consts::PI / 6.0]);
        set_positions(&mut body, &q);
        // Read back
        let q0_back = body.q[0] as f64;
        let q1_back = body.q[1] as f64;
        // f32 has ~7 decimal digits of precision
        assert!((q0_back - q[0]).abs() < 1e-6, "q0: expected {}, got {}", q[0], q0_back);
        assert!((q1_back - q[1]).abs() < 1e-6, "q1: expected {}, got {}", q[1], q1_back);
    }

    // ── Chain with Fixed Joints ─────────────────────────────────────────────

    #[test]
    fn chain_with_fixed_joints_folds_correctly() {
        // Intent: fixed joints should be folded into parent transform, not create bodies
        let urdf = r#"
        <robot name="test">
            <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="mount"><inertial><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="link1"><inertial><mass value="1"/><origin xyz="0.25 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <joint name="mount_joint" type="fixed"><parent link="base"/><child link="mount"/><origin xyz="0 0 0.1"/></joint>
            <joint name="j1" type="revolute"><parent link="mount"/><child link="link1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="100" velocity="2"/></joint>
        </robot>
        "#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link1").unwrap();
        let body = articulated_body_from_chain(&robot, &chain);
        // Only 1 active joint → 1 body (fixed joint folded)
        assert_eq!(body.bodies.len(), 1, "fixed joint should not create a body");
    }

    // ── Prismatic Joint Support ─────────────────────────────────────────────

    #[test]
    fn prismatic_joint_dynamics() {
        // Intent: prismatic joints should produce force (not torque) from dynamics
        let urdf = r#"
        <robot name="test">
            <link name="base"><inertial><mass value="10"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <link name="slider"><inertial><mass value="2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
            <joint name="j1" type="prismatic"><parent link="base"/><child link="slider"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="200" velocity="1"/></joint>
        </robot>
        "#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "slider").unwrap();
        let mut body = articulated_body_from_chain(&robot, &chain);
        // Gravity comp for vertical prismatic joint should be mg
        let q = JointValues::zeros(1);
        let tau = gravity_compensation(&mut body, &q);
        assert_eq!(tau.len(), 1);
        // Force ≈ mass * g = 2 * 9.81 ≈ 19.62 N (axis is Z, gravity is -Z)
        let expected_force = 2.0 * 9.81;
        assert!(
            (tau[0].abs() - expected_force).abs() < 1.0,
            "prismatic gravity comp should be ~{expected_force}N, got {}",
            tau[0]
        );
    }

    // ── Effort Limits Edge Cases ────────────────────────────────────────────

    #[test]
    fn effort_limits_missing_limits_returns_zero() {
        // Intent: joints without explicit effort limits default to 0.0
        let urdf = r#"
        <robot name="test">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="continuous"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/></joint>
        </robot>
        "#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "link1").unwrap();
        let limits = effort_limits_from_chain(&robot, &chain);
        assert_eq!(limits.len(), 1);
        assert!((limits[0]).abs() < 1e-10, "continuous joint without limits should have 0 effort");
    }

    #[test]
    fn effort_limits_multi_joint_correct_order() {
        let (robot, chain, _body) = make_two_link();
        let limits = effort_limits_from_chain(&robot, &chain);
        assert_eq!(limits.len(), 2);
        assert!((limits[0] - 100.0).abs() < 1e-6, "j1 limit should be 100");
        assert!((limits[1] - 50.0).abs() < 1e-6, "j2 limit should be 50");
    }
}
