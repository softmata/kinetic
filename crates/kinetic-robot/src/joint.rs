//! Joint types and limits for robot models.

use kinetic_core::{Pose, Vec3};
use serde::{Deserialize, Serialize};

/// Type of a URDF joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointType {
    /// Rotates around an axis within limits.
    Revolute,
    /// Translates along an axis within limits.
    Prismatic,
    /// Rotates around an axis without limits.
    Continuous,
    /// No motion — rigidly connects parent to child.
    Fixed,
}

/// Physical limits on a joint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    /// Lower position limit (radians or meters).
    pub lower: f64,
    /// Upper position limit (radians or meters).
    pub upper: f64,
    /// Maximum velocity (rad/s or m/s).
    pub velocity: f64,
    /// Maximum effort (Nm or N).
    pub effort: f64,
    /// Maximum acceleration (not always in URDF).
    pub acceleration: Option<f64>,
}

impl JointLimits {
    /// Check if a value is within position limits.
    pub fn in_range(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Clamp a value to position limits.
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.lower, self.upper)
    }

    /// Position range.
    pub fn range(&self) -> f64 {
        self.upper - self.lower
    }
}

impl Default for JointLimits {
    fn default() -> Self {
        Self {
            lower: -std::f64::consts::PI,
            upper: std::f64::consts::PI,
            velocity: 2.0,
            effort: 100.0,
            acceleration: None,
        }
    }
}

/// A joint in the kinematic tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    /// Joint name from URDF.
    pub name: String,
    /// Type of motion.
    pub joint_type: JointType,
    /// Index of the parent link.
    pub parent_link: usize,
    /// Index of the child link.
    pub child_link: usize,
    /// Transform from parent link to joint frame.
    pub origin: Pose,
    /// Rotation/translation axis (unit vector).
    pub axis: Vec3,
    /// Position/velocity/effort limits.
    pub limits: Option<JointLimits>,
}

impl Joint {
    /// Whether this joint is actuated (non-fixed).
    pub fn is_active(&self) -> bool {
        self.joint_type != JointType::Fixed
    }

    /// Whether this joint has position limits.
    pub fn is_limited(&self) -> bool {
        matches!(self.joint_type, JointType::Revolute | JointType::Prismatic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_limits_in_range() {
        let limits = JointLimits {
            lower: -1.0,
            upper: 1.0,
            velocity: 2.0,
            effort: 100.0,
            acceleration: None,
        };
        assert!(limits.in_range(0.5));
        assert!(!limits.in_range(1.5));
        assert!(!limits.in_range(-1.5));
    }

    #[test]
    fn joint_limits_clamp() {
        let limits = JointLimits {
            lower: -1.0,
            upper: 1.0,
            velocity: 2.0,
            effort: 100.0,
            acceleration: None,
        };
        assert!((limits.clamp(0.5) - 0.5).abs() < 1e-10);
        assert!((limits.clamp(2.0) - 1.0).abs() < 1e-10);
        assert!((limits.clamp(-2.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn joint_active() {
        let revolute = Joint {
            name: "j1".into(),
            joint_type: JointType::Revolute,
            parent_link: 0,
            child_link: 1,
            origin: Pose::identity(),
            axis: nalgebra::Vector3::z(),
            limits: Some(JointLimits::default()),
        };
        assert!(revolute.is_active());

        let fixed = Joint {
            name: "j_fixed".into(),
            joint_type: JointType::Fixed,
            parent_link: 0,
            child_link: 1,
            origin: Pose::identity(),
            axis: nalgebra::Vector3::z(),
            limits: None,
        };
        assert!(!fixed.is_active());
    }
}
