//! Math re-exports and convenience types.

pub use nalgebra::{
    Isometry3, Matrix3, Matrix4, Matrix6, Point3, Quaternion, Rotation3, Translation3,
    UnitQuaternion, Vector3, Vector6,
};

/// 3D vector type alias (f64).
pub type Vec3 = Vector3<f64>;

/// Axis-aligned unit vectors.
pub mod axis {
    use super::Vec3;

    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const NEG_X: Vec3 = Vec3::new(-1.0, 0.0, 0.0);
    pub const NEG_Y: Vec3 = Vec3::new(0.0, -1.0, 0.0);
    pub const NEG_Z: Vec3 = Vec3::new(0.0, 0.0, -1.0);
}

/// Axis enum for constraint specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// Get the unit vector for this axis.
    pub fn unit_vector(self) -> Vec3 {
        match self {
            Axis::X => axis::X,
            Axis::Y => axis::Y,
            Axis::Z => axis::Z,
        }
    }
}
