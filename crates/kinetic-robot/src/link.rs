//! Link types for robot models.

use kinetic_core::Pose;
use serde::{Deserialize, Serialize};

/// Primitive geometry shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryShape {
    /// Box with half-extents (x, y, z).
    Box { x: f64, y: f64, z: f64 },
    /// Cylinder with radius and length.
    Cylinder { radius: f64, length: f64 },
    /// Sphere with radius.
    Sphere { radius: f64 },
    /// Mesh file reference.
    Mesh { filename: String, scale: [f64; 3] },
}

/// A geometry instance with origin offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry {
    /// Shape of the geometry.
    pub shape: GeometryShape,
    /// Transform from link origin to geometry origin.
    pub origin: Pose,
}

/// Inertial properties of a link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inertial {
    /// Mass in kg.
    pub mass: f64,
    /// Transform from link origin to center of mass.
    pub origin: Pose,
    /// Inertia tensor [ixx, ixy, ixz, iyy, iyz, izz].
    pub inertia: [f64; 6],
}

/// A link (rigid body) in the kinematic tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    /// Link name from URDF.
    pub name: String,
    /// Index of the joint connecting this link to its parent (None for root).
    pub parent_joint: Option<usize>,
    /// Indices of joints connecting this link to children.
    pub child_joints: Vec<usize>,
    /// Visual geometry (for rendering).
    pub visual_geometry: Vec<Geometry>,
    /// Collision geometry (for planning).
    pub collision_geometry: Vec<Geometry>,
    /// Inertial properties (for dynamics).
    pub inertial: Option<Inertial>,
}

impl Link {
    /// Whether this is the root link (no parent joint).
    pub fn is_root(&self) -> bool {
        self.parent_joint.is_none()
    }

    /// Whether this is a leaf link (no child joints).
    pub fn is_leaf(&self) -> bool {
        self.child_joints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_link() {
        let link = Link {
            name: "base_link".into(),
            parent_joint: None,
            child_joints: vec![0],
            visual_geometry: vec![],
            collision_geometry: vec![],
            inertial: None,
        };
        assert!(link.is_root());
        assert!(!link.is_leaf());
    }

    #[test]
    fn leaf_link() {
        let link = Link {
            name: "ee_link".into(),
            parent_joint: Some(5),
            child_joints: vec![],
            visual_geometry: vec![],
            collision_geometry: vec![],
            inertial: None,
        };
        assert!(!link.is_root());
        assert!(link.is_leaf());
    }
}
