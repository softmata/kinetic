//! Pose (SE3) type with convenience constructors.

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// A 6-DOF pose in 3D space (position + orientation).
///
/// Wraps `nalgebra::Isometry3<f64>` with ergonomic constructors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pose(pub Isometry3<f64>);

impl Pose {
    /// Identity pose (origin, no rotation).
    pub fn identity() -> Self {
        Self(Isometry3::identity())
    }

    /// Pose from translation only (no rotation).
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        Self(Isometry3::from_parts(
            Translation3::new(x, y, z),
            UnitQuaternion::identity(),
        ))
    }

    /// Pose from position and roll-pitch-yaw Euler angles (radians).
    ///
    /// Rotation order: Z (yaw) * Y (pitch) * X (roll), applied intrinsically.
    pub fn from_xyz_rpy(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Self {
        let rotation = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self(Isometry3::from_parts(Translation3::new(x, y, z), rotation))
    }

    /// Pose from position and quaternion (qx, qy, qz, qw).
    ///
    /// The quaternion is normalized internally.
    pub fn from_xyz_quat(x: f64, y: f64, z: f64, qx: f64, qy: f64, qz: f64, qw: f64) -> Self {
        let q = nalgebra::Quaternion::new(qw, qx, qy, qz);
        let rotation = UnitQuaternion::from_quaternion(q);
        Self(Isometry3::from_parts(Translation3::new(x, y, z), rotation))
    }

    /// Pose from a 4x4 homogeneous transformation matrix.
    pub fn from_matrix(mat: &nalgebra::Matrix4<f64>) -> Self {
        let iso = Isometry3::from_parts(
            Translation3::new(mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]),
            UnitQuaternion::from_matrix(&mat.fixed_view::<3, 3>(0, 0).into_owned()),
        );
        Self(iso)
    }

    /// Camera-style "look at" pose.
    ///
    /// The pose is positioned at `eye`, looking toward `target`, with `up` defining the vertical.
    pub fn look_at(eye: &Vector3<f64>, target: &Vector3<f64>, up: &Vector3<f64>) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(up).normalize();
        let actual_up = right.cross(&forward);

        let rotation_matrix = nalgebra::Matrix3::from_columns(&[right, actual_up, -forward]);
        let rotation = UnitQuaternion::from_matrix(&rotation_matrix);

        Self(Isometry3::from_parts(
            Translation3::new(eye.x, eye.y, eye.z),
            rotation,
        ))
    }

    /// Get the translation component.
    pub fn translation(&self) -> Vector3<f64> {
        self.0.translation.vector
    }

    /// Get the rotation as a unit quaternion.
    pub fn rotation(&self) -> &UnitQuaternion<f64> {
        &self.0.rotation
    }

    /// Get the roll, pitch, yaw Euler angles (radians).
    pub fn rpy(&self) -> (f64, f64, f64) {
        self.0.rotation.euler_angles()
    }

    /// Get the underlying Isometry3.
    pub fn isometry(&self) -> &Isometry3<f64> {
        &self.0
    }

    /// Convert to a 4x4 homogeneous transformation matrix.
    pub fn to_matrix(&self) -> nalgebra::Matrix4<f64> {
        self.0.to_homogeneous()
    }

    /// Compose two poses: self * other.
    pub fn compose(&self, other: &Pose) -> Pose {
        Pose(self.0 * other.0)
    }

    /// Inverse of this pose.
    pub fn inverse(&self) -> Pose {
        Pose(self.0.inverse())
    }

    /// Euclidean distance between two pose translations.
    pub fn translation_distance(&self, other: &Pose) -> f64 {
        (self.translation() - other.translation()).norm()
    }

    /// Angular distance between two pose orientations (radians).
    pub fn rotation_distance(&self, other: &Pose) -> f64 {
        self.0.rotation.angle_to(&other.0.rotation)
    }
}

impl From<Isometry3<f64>> for Pose {
    fn from(iso: Isometry3<f64>) -> Self {
        Self(iso)
    }
}

impl From<Pose> for Isometry3<f64> {
    fn from(pose: Pose) -> Self {
        pose.0
    }
}

impl std::ops::Deref for Pose {
    type Target = Isometry3<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self::identity()
    }
}

// softmata-core conversions
impl From<softmata_core::geometry::Pose3D> for Pose {
    fn from(p: softmata_core::geometry::Pose3D) -> Self {
        Self(p.to_isometry3())
    }
}

impl Pose {
    /// Convert to softmata-core Pose3D.
    pub fn to_core(&self) -> softmata_core::geometry::Pose3D {
        softmata_core::geometry::Pose3D::from_isometry3(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identity_is_origin() {
        let p = Pose::identity();
        assert_eq!(p.translation(), Vector3::zeros());
        assert!((p.rotation().angle() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn from_xyz_translation_only() {
        let p = Pose::from_xyz(1.0, 2.0, 3.0);
        let t = p.translation();
        assert!((t.x - 1.0).abs() < 1e-10);
        assert!((t.y - 2.0).abs() < 1e-10);
        assert!((t.z - 3.0).abs() < 1e-10);
        assert!((p.rotation().angle() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn from_xyz_rpy_roundtrip() {
        let roll = 0.3;
        let pitch = 0.5;
        let yaw = 1.2;
        let p = Pose::from_xyz_rpy(1.0, 2.0, 3.0, roll, pitch, yaw);
        let (r, pi, y) = p.rpy();
        assert!((r - roll).abs() < 1e-10);
        assert!((pi - pitch).abs() < 1e-10);
        assert!((y - yaw).abs() < 1e-10);
    }

    #[test]
    fn from_xyz_quat_normalized() {
        // Non-unit quaternion should be normalized
        let p = Pose::from_xyz_quat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0);
        assert!((p.rotation().norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compose_and_inverse() {
        let a = Pose::from_xyz_rpy(1.0, 0.0, 0.0, 0.0, 0.0, PI / 2.0);
        let b = Pose::from_xyz(0.0, 1.0, 0.0);

        let c = a.compose(&b);
        let recovered_b = a.inverse().compose(&c);

        assert!(recovered_b.translation_distance(&b) < 1e-10);
    }

    #[test]
    fn translation_distance() {
        let a = Pose::from_xyz(0.0, 0.0, 0.0);
        let b = Pose::from_xyz(3.0, 4.0, 0.0);
        assert!((a.translation_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_roundtrip() {
        let p = Pose::from_xyz_rpy(1.0, 2.0, 3.0, 0.3, 0.5, 1.2);
        let mat = p.to_matrix();
        let recovered = Pose::from_matrix(&mat);
        assert!(p.translation_distance(&recovered) < 1e-10);
        assert!(p.rotation_distance(&recovered) < 1e-10);
    }
}
