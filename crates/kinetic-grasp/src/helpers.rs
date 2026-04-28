//! Cross-cutting geometric helpers: gripper-pose construction and
//! collision/reachability filters used by every grasp strategy.

use nalgebra::{Isometry3, UnitQuaternion, Vector3};

use kinetic_kinematics::{solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;
use kinetic_scene::Scene;

/// Create a rotation where the local -Z axis points along `approach` and Y is near `up`.
pub(crate) fn rotation_from_approach(
    approach: &Vector3<f64>,
    up_hint: &Vector3<f64>,
) -> UnitQuaternion<f64> {
    let z = -approach.normalize();
    let x = up_hint.cross(&z);
    let x_norm = x.norm();
    if x_norm < 1e-10 {
        // approach is parallel to up — pick arbitrary perpendicular
        let alt_up = if approach.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let x = alt_up.cross(&z).normalize();
        let y = z.cross(&x);
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            nalgebra::Matrix3::from_columns(&[x, y, z]),
        ))
    } else {
        let x = x / x_norm;
        let y = z.cross(&x);
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            nalgebra::Matrix3::from_columns(&[x, y, z]),
        ))
    }
}

/// Check if a grasp pose would cause collision with the scene.
///
/// Approximates the gripper as a small sphere at the TCP and checks
/// overlap against all scene environment spheres. This avoids needing
/// joint values (which require IK) for a quick geometric check.
pub(crate) fn grasp_in_collision(scene: &Scene, grasp_pose: &Isometry3<f64>) -> bool {
    let tcp = grasp_pose.translation.vector;
    let gripper_radius = 0.03; // approximate TCP sphere radius

    let env_spheres = scene.build_environment_spheres();
    for i in 0..env_spheres.len() {
        let dx = tcp.x - env_spheres.x[i];
        let dy = tcp.y - env_spheres.y[i];
        let dz = tcp.z - env_spheres.z[i];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < gripper_radius + env_spheres.radius[i] {
            return true;
        }
    }
    false
}

/// Check if a grasp pose is reachable via IK.
pub(crate) fn is_reachable(
    robot: &Robot,
    chain: &KinematicChain,
    grasp_pose: &Isometry3<f64>,
) -> bool {
    let pose = kinetic_core::Pose(*grasp_pose);
    let config = IKConfig {
        num_restarts: 2,
        max_iterations: 50,
        ..Default::default()
    };
    solve_ik(robot, chain, &pose, &config).is_ok()
}
