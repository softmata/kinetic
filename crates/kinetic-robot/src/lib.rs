//! Robot model loading and configuration for KINETIC.
//!
//! Handles URDF parsing, joint tree construction, planning groups,
//! end-effector definitions, and robot configuration via TOML files.
//!
//! # Usage
//!
//! ```ignore
//! use kinetic_robot::Robot;
//!
//! let robot = Robot::from_urdf("path/to/robot.urdf")?;
//! println!("Loaded {} with {} DOF", robot.name, robot.dof);
//! ```

pub mod config;
pub mod joint;
pub mod link;
pub mod mjcf_loader;
pub mod robot;
pub mod sdf_loader;
pub mod srdf;
pub mod urdf_loader;

pub use config::{CollisionPreference, IkPreference, RobotConfig};
pub use joint::{Joint, JointLimits, JointType};
pub use link::{Geometry, GeometryShape, Inertial, Link};
pub use robot::{EndEffector, PlanningGroup, Robot};
pub use srdf::{DisabledCollision, GroupState, SrdfChain, SrdfEndEffector, SrdfGroup, SrdfModel};
