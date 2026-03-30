//! Planning goal types.

use serde::{Deserialize, Serialize};

use crate::joint_values::JointValues;
use crate::math::Vec3;
use crate::pose::Pose;

/// What the planner should achieve.
///
/// Goals can be specified in joint space, Cartesian space, by name, or as
/// relative motions. The planner or IK solver resolves the goal into a
/// concrete joint configuration before (or during) planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Goal {
    /// Target joint configuration (no IK needed).
    Joints(JointValues),
    /// Target end-effector pose (IK resolved internally).
    Pose(Pose),
    /// Named pose from robot configuration (e.g., `"home"`, `"ready"`).
    Named(String),
    /// Relative motion in end-effector frame.
    Relative(Vec3),
}

impl Goal {
    /// Create a joint-space goal from any type convertible to [`JointValues`].
    ///
    /// Accepts `Vec<f64>`, `[f64; N]`, or `JointValues` directly.
    ///
    /// ```
    /// # use kinetic_core::Goal;
    /// let g = Goal::joints(vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0]);
    /// let g = Goal::joints([0.0; 6]); // fixed array
    /// ```
    pub fn joints(values: impl Into<JointValues>) -> Self {
        Goal::Joints(values.into())
    }

    /// Create a named pose goal (e.g., `"home"`, `"ready"`).
    ///
    /// ```
    /// # use kinetic_core::Goal;
    /// let g = Goal::named("home");
    /// ```
    pub fn named(name: impl Into<String>) -> Self {
        Goal::Named(name.into())
    }

    /// Create a Cartesian pose goal.
    pub fn pose(pose: Pose) -> Self {
        Goal::Pose(pose)
    }

    /// Create a relative motion goal in end-effector frame.
    pub fn relative(offset: Vec3) -> Self {
        Goal::Relative(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn goal_joints() {
        let g = Goal::Joints(JointValues::zeros(7));
        match &g {
            Goal::Joints(jv) => assert_eq!(jv.len(), 7),
            _ => panic!("Expected Goal::Joints"),
        }
    }

    #[test]
    fn goal_pose() {
        let g = Goal::Pose(Pose::identity());
        match &g {
            Goal::Pose(p) => {
                let t = p.translation();
                assert!((t.x.powi(2) + t.y.powi(2) + t.z.powi(2)).sqrt() < 1e-10);
            }
            _ => panic!("Expected Goal::Pose"),
        }
    }

    #[test]
    fn goal_named() {
        let g = Goal::Named("home".to_string());
        match &g {
            Goal::Named(name) => assert_eq!(name, "home"),
            _ => panic!("Expected Goal::Named"),
        }
    }

    #[test]
    fn goal_relative() {
        let g = Goal::Relative(Vector3::new(0.1, 0.0, -0.05));
        match &g {
            Goal::Relative(v) => {
                assert!((v.x - 0.1).abs() < 1e-10);
                assert!((v.z - (-0.05)).abs() < 1e-10);
            }
            _ => panic!("Expected Goal::Relative"),
        }
    }

    #[test]
    fn goal_clone() {
        let g = Goal::Named("ready".to_string());
        let g2 = g.clone();
        match (&g, &g2) {
            (Goal::Named(a), Goal::Named(b)) => assert_eq!(a, b),
            _ => panic!("Clone mismatch"),
        }
    }

    #[test]
    fn goal_joints_from_vec() {
        let g = Goal::joints(vec![0.0, 1.0, 2.0]);
        match &g {
            Goal::Joints(jv) => assert_eq!(jv.len(), 3),
            _ => panic!("Expected Goal::Joints"),
        }
    }

    #[test]
    fn goal_joints_from_array() {
        let g = Goal::joints([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        match &g {
            Goal::Joints(jv) => assert_eq!(jv.len(), 6),
            _ => panic!("Expected Goal::Joints"),
        }
    }

    #[test]
    fn goal_named_from_str() {
        let g = Goal::named("home");
        match &g {
            Goal::Named(n) => assert_eq!(n, "home"),
            _ => panic!("Expected Goal::Named"),
        }
    }

    #[test]
    fn goal_pose_constructor() {
        let g = Goal::pose(Pose::identity());
        match &g {
            Goal::Pose(_) => {}
            _ => panic!("Expected Goal::Pose"),
        }
    }

    #[test]
    fn goal_relative_constructor() {
        let g = Goal::relative(Vector3::new(0.1, 0.0, 0.0));
        match &g {
            Goal::Relative(v) => assert!((v.x - 0.1).abs() < 1e-10),
            _ => panic!("Expected Goal::Relative"),
        }
    }
}
