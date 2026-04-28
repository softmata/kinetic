//! Data types for the controller subsystem.
//!
//! Pulled out of `controller.rs` so the manager file holds just the
//! [`ControllerManager`] state machine. All types here are re-exported
//! from `controller` so external paths stay stable
//! (`kinetic_execution::controller::ControllerType` etc).

use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Controller Types
// ═══════════════════════════════════════════════════════════════════════════

/// Type of hardware controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControllerType {
    /// Joint trajectory controller (arm).
    JointTrajectory,
    /// Gripper controller (open/close/position).
    Gripper,
    /// Head/pan-tilt controller.
    PanTilt,
    /// Custom controller type.
    Custom,
}

/// State of a registered controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerState {
    /// Controller is registered but not active.
    Inactive,
    /// Controller is active and ready to receive commands.
    Active,
    /// Controller is currently executing a command.
    Busy,
    /// Controller reported an error.
    Error,
    /// Controller is in emergency stop state.
    EStopped,
}

/// A registered controller in the manager.
#[derive(Debug, Clone)]
pub struct ControllerInfo {
    /// Unique name (e.g., "arm_controller", "gripper_left").
    pub name: String,
    /// Controller type.
    pub controller_type: ControllerType,
    /// Current state.
    pub state: ControllerState,
    /// Joint names this controller manages.
    pub joint_names: Vec<String>,
    /// DOF of this controller.
    pub dof: usize,
    /// Whether this controller supports velocity scaling.
    pub supports_velocity_scaling: bool,
    /// Current velocity scale factor (0.0..=1.0).
    pub velocity_scale: f64,
    /// Timestamp of last successful command.
    pub last_command_time: Option<Instant>,
    /// Cumulative error count.
    pub error_count: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Gripper Command
// ═══════════════════════════════════════════════════════════════════════════

/// Gripper command types.
#[derive(Debug, Clone)]
pub enum GripperCommand {
    /// Open the gripper fully.
    Open,
    /// Close the gripper (optionally with max effort).
    Close { max_effort: Option<f64> },
    /// Move to a specific position (0.0 = closed, 1.0 = open).
    MoveTo { position: f64, max_effort: Option<f64> },
}

/// Result of a gripper command.
#[derive(Debug, Clone)]
pub struct GripperResult {
    /// Whether the gripper reached the target.
    pub reached_goal: bool,
    /// Final gripper position (0.0..1.0).
    pub position: f64,
    /// Whether an object was detected (stalled before reaching goal).
    pub stalled: bool,
    /// Effort at final position.
    pub effort: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution Actions
// ═══════════════════════════════════════════════════════════════════════════

/// Action handle for tracking in-progress execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionHandle(pub(crate) u64);

/// State of an execution action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionState {
    /// Action is queued but not yet started.
    Pending,
    /// Action is currently executing.
    Active,
    /// Action completed successfully.
    Succeeded,
    /// Action was preempted by a new action.
    Preempted,
    /// Action was cancelled by the user.
    Cancelled,
    /// Action failed with an error.
    Failed,
}

/// An execution action (trajectory or gripper command).
#[derive(Debug, Clone)]
pub struct Action {
    pub handle: ActionHandle,
    pub controller_name: String,
    pub state: ActionState,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub error: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Recovery
// ═══════════════════════════════════════════════════════════════════════════

/// Recovery strategy when deviation is detected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryStrategy {
    /// Stop execution immediately.
    Stop,
    /// Replan from current position to original goal.
    Replan,
    /// Splice a new trajectory from current position onto the remaining path.
    Splice,
    /// Reduce velocity and continue.
    SlowDown { scale: f64 },
}

/// Deviation monitoring configuration.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Position deviation threshold before recovery triggers (radians).
    pub position_threshold: f64,
    /// Velocity deviation threshold (rad/s).
    pub velocity_threshold: f64,
    /// Recovery strategy.
    pub recovery: RecoveryStrategy,
    /// Check interval (every N commands).
    pub check_interval: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            position_threshold: 0.05,
            velocity_threshold: 0.3,
            recovery: RecoveryStrategy::Stop,
            check_interval: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Log
// ═══════════════════════════════════════════════════════════════════════════

/// Event types for the execution log.
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Controller registered.
    ControllerRegistered { name: String, controller_type: ControllerType },
    /// Controller activated/deactivated.
    ControllerStateChanged { name: String, from: ControllerState, to: ControllerState },
    /// Action created.
    ActionCreated { handle: ActionHandle, controller: String },
    /// Action state changed.
    ActionStateChanged { handle: ActionHandle, from: ActionState, to: ActionState },
    /// Deviation detected.
    DeviationDetected { controller: String, deviation: f64, threshold: f64 },
    /// Recovery triggered.
    RecoveryTriggered { controller: String, strategy: RecoveryStrategy },
    /// Emergency stop.
    EmergencyStop { reason: String },
    /// Velocity scale changed.
    VelocityScaleChanged { controller: String, old_scale: f64, new_scale: f64 },
}

/// Timestamped event.
#[derive(Debug, Clone)]
pub struct TimestampedEvent {
    pub time: Instant,
    pub event: ExecutionEvent,
}
