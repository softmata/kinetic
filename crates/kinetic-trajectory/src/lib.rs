//! Trajectory processing and time parameterization for KINETIC.
//!
//! Converts geometric paths into timed trajectories with velocity,
//! acceleration profiles: TOTP, trapezoidal, jerk-limited S-curve,
//! cubic spline, and trajectory blending.

pub mod blend;
pub mod export;
pub mod jerk_limited;
pub mod monitor;
pub mod spline;
pub mod totp;
pub mod trapezoidal;
pub mod validation;

pub use blend::{blend, blend_sequence};
pub use export::{
    trajectory_from_csv, trajectory_from_json, trajectory_to_csv, trajectory_to_csv_file,
    trajectory_to_json, trajectory_to_json_file,
};
pub use jerk_limited::{jerk_limited, jerk_limited_per_joint};
pub use monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
pub use spline::{cubic_spline_time, cubic_spline_time_clamped};
pub use totp::totp;
pub use trapezoidal::{trapezoidal, trapezoidal_per_joint, TimedTrajectory, TimedWaypoint};
pub use validation::{TrajectoryValidator, TrajectoryViolation, ValidationConfig, ViolationType};
