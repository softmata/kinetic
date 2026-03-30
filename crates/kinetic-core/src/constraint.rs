//! Motion planning constraints.

use serde::{Deserialize, Serialize};

use crate::math::{Axis, Vec3};

/// A constraint applied during motion planning.
///
/// Constraints restrict the feasible configuration space. The planner must
/// ensure every waypoint satisfies all active constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Keep a link's orientation within `tolerance` (radians) of a reference
    /// axis direction throughout the motion.
    ///
    /// Common use: keeping a cup upright while moving.
    Orientation {
        /// Link name to constrain.
        link: String,
        /// Reference axis direction (world frame).
        axis: Vec3,
        /// Maximum angular deviation in radians.
        tolerance: f64,
    },

    /// Keep a link's position bounded along a world-frame axis.
    ///
    /// Common use: keeping the end-effector above a table surface.
    PositionBound {
        /// Link name to constrain.
        link: String,
        /// Which axis to bound.
        axis: Axis,
        /// Lower bound (meters).
        min: f64,
        /// Upper bound (meters).
        max: f64,
    },

    /// Restrict a joint to a custom range during this motion.
    ///
    /// Tighter than the URDF limits — useful for avoiding singularities.
    Joint {
        /// Joint index in the planning group.
        joint_index: usize,
        /// Minimum joint value (radians or meters).
        min: f64,
        /// Maximum joint value (radians or meters).
        max: f64,
    },

    /// Keep a target point visible from a sensor link.
    ///
    /// The sensor's forward axis must point within `cone_angle` of the target.
    Visibility {
        /// Sensor link name.
        sensor_link: String,
        /// Target point in world frame.
        target: Vec3,
        /// Half-angle of the visibility cone in radians.
        cone_angle: f64,
    },
}

impl Constraint {
    /// Create an orientation constraint.
    pub fn orientation(link: impl Into<String>, axis: Vec3, tolerance: f64) -> Self {
        Self::Orientation {
            link: link.into(),
            axis,
            tolerance,
        }
    }

    /// Create a position bound constraint.
    pub fn position_bound(link: impl Into<String>, axis: Axis, min: f64, max: f64) -> Self {
        Self::PositionBound {
            link: link.into(),
            axis,
            min,
            max,
        }
    }

    /// Create a joint range constraint.
    pub fn joint(joint_index: usize, min: f64, max: f64) -> Self {
        Self::Joint {
            joint_index,
            min,
            max,
        }
    }

    /// Create a visibility constraint.
    pub fn visibility(sensor_link: impl Into<String>, target: Vec3, cone_angle: f64) -> Self {
        Self::Visibility {
            sensor_link: sensor_link.into(),
            target,
            cone_angle,
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn orientation_constraint() {
        let c = Constraint::orientation("ee_link", Vector3::z(), 0.1);
        match &c {
            Constraint::Orientation {
                link,
                axis,
                tolerance,
            } => {
                assert_eq!(link, "ee_link");
                assert!((axis.z - 1.0).abs() < 1e-10);
                assert!((*tolerance - 0.1).abs() < 1e-10);
            }
            _ => panic!("Expected Orientation"),
        }
    }

    #[test]
    fn position_bound_constraint() {
        let c = Constraint::position_bound("wrist_link", Axis::Z, 0.3, 1.5);
        match &c {
            Constraint::PositionBound {
                link,
                axis,
                min,
                max,
            } => {
                assert_eq!(link, "wrist_link");
                assert!(matches!(axis, Axis::Z));
                assert!((*min - 0.3).abs() < 1e-10);
                assert!((*max - 1.5).abs() < 1e-10);
            }
            _ => panic!("Expected PositionBound"),
        }
    }

    #[test]
    fn joint_constraint() {
        let c = Constraint::joint(3, -1.0, 1.0);
        match &c {
            Constraint::Joint {
                joint_index,
                min,
                max,
            } => {
                assert_eq!(*joint_index, 3);
                assert!((*min - (-1.0)).abs() < 1e-10);
                assert!((*max - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Joint"),
        }
    }

    #[test]
    fn visibility_constraint() {
        let target = Vector3::new(1.0, 0.0, 0.5);
        let c = Constraint::visibility("camera_link", target, 0.5);
        match &c {
            Constraint::Visibility {
                sensor_link,
                target: t,
                cone_angle,
            } => {
                assert_eq!(sensor_link, "camera_link");
                assert!((t.x - 1.0).abs() < 1e-10);
                assert!((*cone_angle - 0.5).abs() < 1e-10);
            }
            _ => panic!("Expected Visibility"),
        }
    }

    #[test]
    fn constraint_clone() {
        let c = Constraint::joint(0, -3.14, 3.14);
        let c2 = c.clone();
        match (&c, &c2) {
            (
                Constraint::Joint { joint_index: a, .. },
                Constraint::Joint { joint_index: b, .. },
            ) => assert_eq!(a, b),
            _ => panic!("Clone mismatch"),
        }
    }
}
