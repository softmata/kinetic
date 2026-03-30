//! Core types for the KINETIC motion planning stack.
//!
//! Provides fundamental types used across all kinetic crates:
//! - [`Pose`] — SE(3) transform with ergonomic constructors
//! - [`Twist`] — 6-DOF velocity (linear + angular)
//! - [`Wrench`] — 6-DOF force/torque
//! - [`JointValues`] — Named joint-position vector
//! - [`Trajectory`] — SoA waypoint trajectory
//! - [`Goal`] — Planning target specification
//! - [`Constraint`] — Motion constraints
//! - [`KineticError`] — Unified error type
//! - [`PlannerConfig`] — Shared planner configuration
//! - [`Axis`] — Axis enum for constraint specification
//! - Math re-exports from nalgebra (f64 throughout)

pub mod config;
pub mod constraint;
pub mod error;
pub mod frame_tree;
pub mod goal;
pub mod joint_values;
pub mod math;
pub mod pose;
pub mod trajectory;
pub mod twist;
pub mod wrench;

// Re-export primary types at crate root
pub use config::PlannerConfig;
pub use constraint::Constraint;
pub use error::{KineticError, Result};
pub use frame_tree::{FrameTree, StampedTransform};
pub use goal::Goal;
pub use joint_values::JointValues;
pub use math::{Axis, Vec3};
pub use pose::Pose;
pub use trajectory::{TimedWaypoint, Trajectory, Waypoint};
pub use twist::Twist;
pub use wrench::Wrench;

// Re-export essential nalgebra types
pub use nalgebra::{Isometry3, Matrix4, Point3, UnitQuaternion, Vector3};

// Re-export softmata-core for canonical type access
pub use softmata_core;
