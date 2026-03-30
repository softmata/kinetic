//! HORUS bridge for KINETIC.
//!
//! Wraps KINETIC planners, servo, and scene as HORUS nodes with
//! zero-copy shared-memory IPC communication.
//!
//! # Architecture
//!
//! This crate provides three HORUS nodes:
//! - [`PlannerNode`] — Receives planning requests, publishes trajectories
//! - [`ServoNode`] — Receives twist/jog commands, publishes joint commands at 500Hz
//! - [`SceneNode`] — Manages the collision world, publishes scene updates
//!
//! # Without HORUS (default)
//!
//! When built without the `horus-ipc` feature, the nodes expose a channel-based
//! API that can be used standalone or integrated with any message transport.
//!
//! # With HORUS IPC
//!
//! Enable the `horus-ipc` feature to get full HORUS `Node` trait implementations
//! with zero-copy shared-memory topics.
//!
//! ```ignore
//! use horus_kinetic::{PlannerNode, ServoNode, SceneNode};
//!
//! let mut planner = PlannerNode::new("ur5e")?;
//! let mut servo = ServoNode::new("ur5e")?;
//! let mut scene = SceneNode::new("ur5e")?;
//! ```

pub mod messages;
pub mod planner_node;
pub mod scene_node;
pub mod servo_node;

pub use messages::*;
pub use planner_node::PlannerNode;
pub use scene_node::SceneNode;
pub use servo_node::ServoNode;
