//! Shared planner configuration.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration shared across all motion planners.
///
/// Controls timeouts, iteration limits, collision margins, and post-processing.
/// Each planner may interpret these slightly differently, but the semantics
/// are consistent:
///
/// - `timeout` — hard wall-clock cutoff; planner returns best-so-far or error.
/// - `max_iterations` — upper bound on sampling/expansion iterations.
/// - `collision_margin` — minimum clearance (meters) from obstacles.
/// - `shortcut_iterations` — number of shortcutting passes on the raw path.
/// - `smooth` — whether to apply B-spline smoothing after shortcutting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// Maximum planning time before timeout.
    #[serde(with = "duration_millis")]
    pub timeout: Duration,
    /// Maximum number of planner iterations.
    pub max_iterations: usize,
    /// Minimum collision clearance in meters.
    pub collision_margin: f64,
    /// Number of random-shortcut passes for path optimization.
    pub shortcut_iterations: usize,
    /// Whether to apply B-spline smoothing after shortcutting.
    pub smooth: bool,
    /// Optional workspace bounds `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// If set, the planner validates that the end-effector stays within these
    /// Cartesian bounds at every waypoint. Waypoints outside bounds trigger
    /// an error. Prevents plans that reach the workspace edge (low manipulability,
    /// risky for execution).
    #[serde(skip)]
    pub workspace_bounds: Option<[f64; 6]>,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(50),
            max_iterations: 10_000,
            collision_margin: 0.02,
            shortcut_iterations: 100,
            smooth: true,
            workspace_bounds: None,
        }
    }
}

impl PlannerConfig {
    /// Create a config with custom timeout (other values default).
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Self::default()
        }
    }

    /// Create a fast config for real-time planning (10ms timeout).
    pub fn realtime() -> Self {
        Self {
            timeout: Duration::from_millis(10),
            max_iterations: 2_000,
            collision_margin: 0.01,
            shortcut_iterations: 20,
            smooth: false,
            workspace_bounds: None,
        }
    }

    /// Create a thorough config for offline planning (500ms timeout).
    pub fn offline() -> Self {
        Self {
            timeout: Duration::from_millis(500),
            max_iterations: 100_000,
            collision_margin: 0.02,
            shortcut_iterations: 500,
            smooth: true,
            workspace_bounds: None,
        }
    }
}

/// Serde helper: serialize Duration as integer milliseconds.
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(dur: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(dur.as_millis() as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = PlannerConfig::default();
        assert_eq!(cfg.timeout, Duration::from_millis(50));
        assert_eq!(cfg.max_iterations, 10_000);
        assert!((cfg.collision_margin - 0.02).abs() < 1e-10);
        assert_eq!(cfg.shortcut_iterations, 100);
        assert!(cfg.smooth);
    }

    #[test]
    fn realtime_config() {
        let cfg = PlannerConfig::realtime();
        assert_eq!(cfg.timeout, Duration::from_millis(10));
        assert!(!cfg.smooth);
    }

    #[test]
    fn offline_config() {
        let cfg = PlannerConfig::offline();
        assert_eq!(cfg.timeout, Duration::from_millis(500));
        assert_eq!(cfg.max_iterations, 100_000);
        assert!(cfg.smooth);
    }

    #[test]
    fn with_timeout() {
        let cfg = PlannerConfig::with_timeout(Duration::from_millis(200));
        assert_eq!(cfg.timeout, Duration::from_millis(200));
        // Other values should be default
        assert_eq!(cfg.max_iterations, 10_000);
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = PlannerConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: PlannerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.timeout, cfg2.timeout);
        assert_eq!(cfg.max_iterations, cfg2.max_iterations);
        assert!((cfg.collision_margin - cfg2.collision_margin).abs() < 1e-10);
    }

    #[test]
    fn clone_config() {
        let cfg = PlannerConfig::realtime();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.timeout, cfg2.timeout);
        assert_eq!(cfg.max_iterations, cfg2.max_iterations);
    }
}
