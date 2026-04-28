//! Controller Manager: multi-controller registry, execution actions, and recovery.
//!
//! Manages multiple hardware controllers (arm, gripper, head) with:
//! - Controller registration and switching
//! - FollowJointTrajectory action (stream trajectory to arm controller)
//! - GripperCommand action (open/close/move gripper)
//! - Execution monitoring with deviation-triggered replanning
//! - Preemption/cancel of in-progress actions
//! - Trajectory splicing for recovery
//! - Emergency stop propagation
//! - Velocity scaling during execution
//! - Controller health monitoring
//! - Event logging

use std::collections::HashMap;
use std::time::Instant;

use kinetic_trajectory::TimedTrajectory;

// Types live in controller_types.rs; re-exported here so external paths
// `kinetic_execution::controller::ControllerType` etc. stay stable.
pub use crate::controller_types::*;

// ═══════════════════════════════════════════════════════════════════════════
// Controller Manager
// ═══════════════════════════════════════════════════════════════════════════

/// Controller Manager: central registry for all hardware controllers.
///
/// Manages controller lifecycle, action dispatch, monitoring, and recovery.
pub struct ControllerManager {
    controllers: HashMap<String, ControllerInfo>,
    actions: HashMap<ActionHandle, Action>,
    next_action_id: u64,
    monitor_config: MonitorConfig,
    event_log: Vec<TimestampedEvent>,
    /// Global emergency stop flag.
    e_stopped: bool,
    /// Maximum event log size (default: 10000).
    pub max_log_size: usize,
}

impl ControllerManager {
    /// Create a new controller manager.
    pub fn new(monitor_config: MonitorConfig) -> Self {
        Self {
            controllers: HashMap::new(),
            actions: HashMap::new(),
            next_action_id: 1,
            monitor_config,
            event_log: Vec::new(),
            e_stopped: false,
            max_log_size: 10000,
        }
    }

    /// Create with default monitoring configuration.
    pub fn with_defaults() -> Self {
        Self::new(MonitorConfig::default())
    }

    // ─── Controller Registration ─────────────────────────────────────────

    /// Register a controller.
    pub fn register(
        &mut self,
        name: &str,
        controller_type: ControllerType,
        joint_names: Vec<String>,
        supports_velocity_scaling: bool,
    ) {
        let dof = joint_names.len();
        self.controllers.insert(
            name.to_string(),
            ControllerInfo {
                name: name.to_string(),
                controller_type,
                state: ControllerState::Inactive,
                joint_names,
                dof,
                supports_velocity_scaling,
                velocity_scale: 1.0,
                last_command_time: None,
                error_count: 0,
            },
        );
        self.log_event(ExecutionEvent::ControllerRegistered {
            name: name.to_string(),
            controller_type,
        });
    }

    /// Activate a controller (make it ready to receive commands).
    pub fn activate(&mut self, name: &str) -> bool {
        if let Some(ctrl) = self.controllers.get_mut(name) {
            let old = ctrl.state;
            ctrl.state = ControllerState::Active;
            self.log_event(ExecutionEvent::ControllerStateChanged {
                name: name.to_string(), from: old, to: ControllerState::Active,
            });
            true
        } else {
            false
        }
    }

    /// Deactivate a controller.
    pub fn deactivate(&mut self, name: &str) -> bool {
        if let Some(ctrl) = self.controllers.get_mut(name) {
            let old = ctrl.state;
            ctrl.state = ControllerState::Inactive;
            self.log_event(ExecutionEvent::ControllerStateChanged {
                name: name.to_string(), from: old, to: ControllerState::Inactive,
            });
            true
        } else {
            false
        }
    }

    /// Get controller info.
    pub fn controller(&self, name: &str) -> Option<&ControllerInfo> {
        self.controllers.get(name)
    }

    /// List all registered controllers.
    pub fn controllers(&self) -> impl Iterator<Item = &ControllerInfo> {
        self.controllers.values()
    }

    /// Number of registered controllers.
    pub fn num_controllers(&self) -> usize {
        self.controllers.len()
    }

    /// Switch active controller: deactivate current, activate new.
    ///
    /// Returns false if the new controller doesn't exist or the old
    /// controller is currently busy.
    pub fn switch(&mut self, from: &str, to: &str) -> bool {
        if let Some(ctrl) = self.controllers.get(from) {
            if ctrl.state == ControllerState::Busy {
                return false; // can't switch while busy
            }
        }
        self.deactivate(from);
        self.activate(to)
    }

    // ─── Action Dispatch ─────────────────────────────────────────────────

    /// Create a FollowJointTrajectory action.
    ///
    /// Returns a handle for tracking. The action starts in Pending state.
    pub fn follow_joint_trajectory(
        &mut self,
        controller_name: &str,
    ) -> Option<ActionHandle> {
        let ctrl = self.controllers.get(controller_name)?;
        if !matches!(ctrl.state, ControllerState::Active | ControllerState::Busy) {
            return None;
        }
        if self.e_stopped {
            return None;
        }

        // Preempt any existing active action on this controller
        self.preempt_active_on(controller_name);

        let handle = ActionHandle(self.next_action_id);
        self.next_action_id += 1;

        self.actions.insert(handle, Action {
            handle,
            controller_name: controller_name.to_string(),
            state: ActionState::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            error: None,
        });

        self.log_event(ExecutionEvent::ActionCreated {
            handle,
            controller: controller_name.to_string(),
        });

        Some(handle)
    }

    /// Start executing an action (transition Pending → Active).
    pub fn start_action(&mut self, handle: ActionHandle) -> bool {
        if let Some(action) = self.actions.get_mut(&handle) {
            if action.state != ActionState::Pending { return false; }

            let old = action.state;
            action.state = ActionState::Active;
            action.started_at = Some(Instant::now());

            if let Some(ctrl) = self.controllers.get_mut(&action.controller_name) {
                ctrl.state = ControllerState::Busy;
            }

            self.log_event(ExecutionEvent::ActionStateChanged {
                handle, from: old, to: ActionState::Active,
            });
            true
        } else {
            false
        }
    }

    /// Complete an action successfully.
    pub fn complete_action(&mut self, handle: ActionHandle) -> bool {
        self.transition_action(handle, ActionState::Succeeded)
    }

    /// Fail an action with an error message.
    pub fn fail_action(&mut self, handle: ActionHandle, error: &str) -> bool {
        if let Some(action) = self.actions.get_mut(&handle) {
            let old = action.state;
            action.state = ActionState::Failed;
            action.completed_at = Some(Instant::now());
            action.error = Some(error.to_string());

            if let Some(ctrl) = self.controllers.get_mut(&action.controller_name) {
                ctrl.state = ControllerState::Error;
                ctrl.error_count += 1;
            }

            self.log_event(ExecutionEvent::ActionStateChanged {
                handle, from: old, to: ActionState::Failed,
            });
            true
        } else {
            false
        }
    }

    /// Cancel an action.
    pub fn cancel_action(&mut self, handle: ActionHandle) -> bool {
        self.transition_action(handle, ActionState::Cancelled)
    }

    /// Preempt an action (replaced by a newer action).
    pub fn preempt_action(&mut self, handle: ActionHandle) -> bool {
        self.transition_action(handle, ActionState::Preempted)
    }

    /// Get action info.
    pub fn action(&self, handle: ActionHandle) -> Option<&Action> {
        self.actions.get(&handle)
    }

    /// Get the active action on a controller.
    pub fn active_action_on(&self, controller_name: &str) -> Option<&Action> {
        self.actions.values().find(|a| {
            a.controller_name == controller_name && a.state == ActionState::Active
        })
    }

    // ─── Monitoring ──────────────────────────────────────────────────────

    /// Check deviation and trigger recovery if needed.
    ///
    /// `actual_positions`: current robot joint positions.
    /// `expected_positions`: positions from the trajectory at current time.
    ///
    /// Returns the recovery strategy if deviation exceeds threshold, or None.
    pub fn check_deviation(
        &mut self,
        controller_name: &str,
        actual_positions: &[f64],
        expected_positions: &[f64],
    ) -> Option<RecoveryStrategy> {
        let max_dev = actual_positions.iter()
            .zip(expected_positions.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f64, f64::max);

        if max_dev > self.monitor_config.position_threshold {
            self.log_event(ExecutionEvent::DeviationDetected {
                controller: controller_name.to_string(),
                deviation: max_dev,
                threshold: self.monitor_config.position_threshold,
            });
            self.log_event(ExecutionEvent::RecoveryTriggered {
                controller: controller_name.to_string(),
                strategy: self.monitor_config.recovery,
            });
            Some(self.monitor_config.recovery)
        } else {
            None
        }
    }

    // ─── Velocity Scaling ────────────────────────────────────────────────

    /// Set velocity scaling for a controller (0.0..=1.0).
    pub fn set_velocity_scale(&mut self, controller_name: &str, scale: f64) -> bool {
        let scale = scale.clamp(0.0, 1.0);
        if let Some(ctrl) = self.controllers.get_mut(controller_name) {
            if !ctrl.supports_velocity_scaling { return false; }
            let old = ctrl.velocity_scale;
            ctrl.velocity_scale = scale;
            self.log_event(ExecutionEvent::VelocityScaleChanged {
                controller: controller_name.to_string(),
                old_scale: old,
                new_scale: scale,
            });
            true
        } else {
            false
        }
    }

    /// Get current velocity scale for a controller.
    pub fn velocity_scale(&self, controller_name: &str) -> Option<f64> {
        self.controllers.get(controller_name).map(|c| c.velocity_scale)
    }

    // ─── Emergency Stop ──────────────────────────────────────────────────

    /// Trigger emergency stop on all controllers.
    ///
    /// All active actions are failed, all controllers set to EStopped.
    pub fn emergency_stop(&mut self, reason: &str) {
        self.e_stopped = true;

        self.log_event(ExecutionEvent::EmergencyStop {
            reason: reason.to_string(),
        });

        // Fail all active actions
        let active_handles: Vec<ActionHandle> = self.actions.values()
            .filter(|a| a.state == ActionState::Active || a.state == ActionState::Pending)
            .map(|a| a.handle)
            .collect();

        for handle in active_handles {
            self.fail_action(handle, &format!("E-STOP: {}", reason));
        }

        // Set all controllers to EStopped
        for ctrl in self.controllers.values_mut() {
            ctrl.state = ControllerState::EStopped;
        }
    }

    /// Clear emergency stop and reset controllers to Inactive.
    pub fn clear_estop(&mut self) {
        self.e_stopped = false;
        for ctrl in self.controllers.values_mut() {
            if ctrl.state == ControllerState::EStopped {
                ctrl.state = ControllerState::Inactive;
            }
        }
    }

    /// Whether the system is in emergency stop.
    pub fn is_estopped(&self) -> bool {
        self.e_stopped
    }

    // ─── Health ──────────────────────────────────────────────────────────

    /// Check if a controller is healthy (Active or Busy, no recent errors).
    pub fn is_healthy(&self, name: &str) -> bool {
        self.controllers.get(name).map_or(false, |c| {
            matches!(c.state, ControllerState::Active | ControllerState::Busy)
        })
    }

    /// Get all unhealthy controllers.
    pub fn unhealthy_controllers(&self) -> Vec<&ControllerInfo> {
        self.controllers.values().filter(|c| {
            matches!(c.state, ControllerState::Error | ControllerState::EStopped)
        }).collect()
    }

    // ─── Event Log ───────────────────────────────────────────────────────

    /// Get the event log.
    pub fn event_log(&self) -> &[TimestampedEvent] {
        &self.event_log
    }

    /// Number of events logged.
    pub fn event_count(&self) -> usize {
        self.event_log.len()
    }

    /// Clear the event log.
    pub fn clear_log(&mut self) {
        self.event_log.clear();
    }

    // ─── Trajectory Splicing ─────────────────────────────────────────────

    /// Compute a splice point for recovery: given the current position and
    /// remaining trajectory, find the nearest waypoint to splice from.
    ///
    /// Returns the waypoint index in the trajectory to resume from.
    pub fn find_splice_point(
        current_positions: &[f64],
        trajectory: &TimedTrajectory,
    ) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;

        for i in 0..trajectory.len() {
            let wp = &trajectory.waypoints[i];
            let dist: f64 = current_positions.iter()
                .zip(wp.positions.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    // ─── Internal ────────────────────────────────────────────────────────

    fn transition_action(&mut self, handle: ActionHandle, new_state: ActionState) -> bool {
        if let Some(action) = self.actions.get_mut(&handle) {
            let old = action.state;
            action.state = new_state;
            action.completed_at = Some(Instant::now());

            if let Some(ctrl) = self.controllers.get_mut(&action.controller_name) {
                if matches!(new_state, ActionState::Succeeded | ActionState::Cancelled | ActionState::Preempted) {
                    ctrl.state = ControllerState::Active;
                    ctrl.last_command_time = Some(Instant::now());
                }
            }

            self.log_event(ExecutionEvent::ActionStateChanged {
                handle, from: old, to: new_state,
            });
            true
        } else {
            false
        }
    }

    fn preempt_active_on(&mut self, controller_name: &str) {
        let active_handles: Vec<ActionHandle> = self.actions.values()
            .filter(|a| a.controller_name == controller_name && a.state == ActionState::Active)
            .map(|a| a.handle)
            .collect();

        for handle in active_handles {
            self.preempt_action(handle);
        }
    }

    fn log_event(&mut self, event: ExecutionEvent) {
        if self.event_log.len() >= self.max_log_size {
            self.event_log.drain(0..self.max_log_size / 2);
        }
        self.event_log.push(TimestampedEvent {
            time: Instant::now(),
            event,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ControllerManager {
        let mut mgr = ControllerManager::with_defaults();
        mgr.register(
            "arm",
            ControllerType::JointTrajectory,
            vec!["j1".into(), "j2".into(), "j3".into(), "j4".into(), "j5".into(), "j6".into()],
            true,
        );
        mgr.register(
            "gripper",
            ControllerType::Gripper,
            vec!["finger_left".into(), "finger_right".into()],
            false,
        );
        mgr
    }

    // ─── Registration tests ───

    #[test]
    fn register_controllers() {
        let mgr = setup();
        assert_eq!(mgr.num_controllers(), 2);
        assert!(mgr.controller("arm").is_some());
        assert!(mgr.controller("gripper").is_some());
        assert!(mgr.controller("nonexistent").is_none());
    }

    #[test]
    fn controller_info_correct() {
        let mgr = setup();
        let arm = mgr.controller("arm").unwrap();
        assert_eq!(arm.controller_type, ControllerType::JointTrajectory);
        assert_eq!(arm.dof, 6);
        assert_eq!(arm.state, ControllerState::Inactive);
        assert!(arm.supports_velocity_scaling);
    }

    #[test]
    fn activate_deactivate() {
        let mut mgr = setup();
        assert!(mgr.activate("arm"));
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Active);

        assert!(mgr.deactivate("arm"));
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Inactive);
    }

    #[test]
    fn switch_controller() {
        let mut mgr = setup();
        mgr.activate("arm");
        assert!(mgr.switch("arm", "gripper"));
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Inactive);
        assert_eq!(mgr.controller("gripper").unwrap().state, ControllerState::Active);
    }

    // ─── Action tests ───

    #[test]
    fn create_and_start_action() {
        let mut mgr = setup();
        mgr.activate("arm");

        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Pending);

        assert!(mgr.start_action(handle));
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Active);
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Busy);
    }

    #[test]
    fn complete_action() {
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        assert!(mgr.complete_action(handle));
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Succeeded);
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Active);
    }

    #[test]
    fn cancel_action() {
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        assert!(mgr.cancel_action(handle));
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Cancelled);
    }

    #[test]
    fn preempt_on_new_action() {
        let mut mgr = setup();
        mgr.activate("arm");

        let h1 = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(h1);

        // New action preempts the old one
        let h2 = mgr.follow_joint_trajectory("arm").unwrap();
        assert_eq!(mgr.action(h1).unwrap().state, ActionState::Preempted);
        assert_eq!(mgr.action(h2).unwrap().state, ActionState::Pending);
    }

    #[test]
    fn action_on_inactive_controller_fails() {
        let mut mgr = setup();
        // arm is Inactive
        assert!(mgr.follow_joint_trajectory("arm").is_none());
    }

    #[test]
    fn active_action_on_controller() {
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        let active = mgr.active_action_on("arm");
        assert!(active.is_some());
        assert_eq!(active.unwrap().handle, handle);
    }

    // ─── Monitoring tests ───

    #[test]
    fn deviation_within_threshold() {
        let mut mgr = setup();
        let actual = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01];

        let recovery = mgr.check_deviation("arm", &actual, &expected);
        assert!(recovery.is_none(), "Small deviation should not trigger recovery");
    }

    #[test]
    fn deviation_exceeds_threshold() {
        let mut mgr = setup();
        let actual = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]; // 0.1 rad deviation

        let recovery = mgr.check_deviation("arm", &actual, &expected);
        assert!(recovery.is_some(), "Large deviation should trigger recovery");
        assert_eq!(recovery.unwrap(), RecoveryStrategy::Stop);
    }

    // ─── Velocity scaling tests ───

    #[test]
    fn velocity_scaling() {
        let mut mgr = setup();
        assert_eq!(mgr.velocity_scale("arm"), Some(1.0));

        assert!(mgr.set_velocity_scale("arm", 0.5));
        assert_eq!(mgr.velocity_scale("arm"), Some(0.5));

        // Clamp to [0, 1]
        mgr.set_velocity_scale("arm", 1.5);
        assert_eq!(mgr.velocity_scale("arm"), Some(1.0));
    }

    #[test]
    fn velocity_scaling_unsupported() {
        let mut mgr = setup();
        // Gripper doesn't support velocity scaling
        assert!(!mgr.set_velocity_scale("gripper", 0.5));
    }

    // ─── Emergency stop tests ───

    #[test]
    fn emergency_stop_all() {
        let mut mgr = setup();
        mgr.activate("arm");
        mgr.activate("gripper");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        mgr.emergency_stop("test emergency");

        assert!(mgr.is_estopped());
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::EStopped);
        assert_eq!(mgr.controller("gripper").unwrap().state, ControllerState::EStopped);
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Failed);
    }

    #[test]
    fn clear_estop() {
        let mut mgr = setup();
        mgr.activate("arm");
        mgr.emergency_stop("test");

        mgr.clear_estop();
        assert!(!mgr.is_estopped());
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Inactive);
    }

    #[test]
    fn estop_blocks_new_actions() {
        let mut mgr = setup();
        mgr.activate("arm");
        mgr.emergency_stop("test");

        assert!(mgr.follow_joint_trajectory("arm").is_none());
    }

    // ─── Health tests ───

    #[test]
    fn controller_health() {
        let mut mgr = setup();
        assert!(!mgr.is_healthy("arm")); // Inactive

        mgr.activate("arm");
        assert!(mgr.is_healthy("arm"));

        mgr.emergency_stop("test");
        assert!(!mgr.is_healthy("arm"));
        assert_eq!(mgr.unhealthy_controllers().len(), 2);
    }

    // ─── Event log tests ───

    #[test]
    fn event_log_records() {
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);
        mgr.complete_action(handle);

        assert!(mgr.event_count() >= 4); // register*2 + activate + create + start + complete
    }

    #[test]
    fn event_log_truncation() {
        let mut mgr = ControllerManager::with_defaults();
        mgr.max_log_size = 10;

        for i in 0..20 {
            mgr.register(&format!("ctrl_{}", i), ControllerType::Custom, vec![], false);
        }

        assert!(mgr.event_count() <= 15, "Log should be truncated: {}", mgr.event_count());
    }

    // ─── Trajectory splicing tests ───

    #[test]
    fn find_splice_point_nearest() {
        let traj = TimedTrajectory {
            duration: std::time::Duration::from_secs(2),
            dof: 2,
            waypoints: vec![
                kinetic_trajectory::TimedWaypoint {
                    positions: vec![0.0, 0.0],
                    velocities: vec![0.0, 0.0],
                    accelerations: vec![0.0, 0.0],
                    time: 0.0,
                },
                kinetic_trajectory::TimedWaypoint {
                    positions: vec![1.0, 1.0],
                    velocities: vec![0.5, 0.5],
                    accelerations: vec![0.0, 0.0],
                    time: 1.0,
                },
                kinetic_trajectory::TimedWaypoint {
                    positions: vec![2.0, 2.0],
                    velocities: vec![0.0, 0.0],
                    accelerations: vec![0.0, 0.0],
                    time: 2.0,
                },
            ],
        };

        // Current position near waypoint 1
        let idx = ControllerManager::find_splice_point(&[0.9, 0.9], &traj);
        assert_eq!(idx, 1, "Should splice from nearest waypoint");
    }

    #[test]
    fn switch_blocks_during_busy() {
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        // Can't switch away from busy controller
        assert!(!mgr.switch("arm", "gripper"));
    }

    // ─── Integration test ───

    #[test]
    fn full_execution_lifecycle() {
        let mut mgr = ControllerManager::new(MonitorConfig {
            recovery: RecoveryStrategy::Replan,
            ..Default::default()
        });

        mgr.register("arm", ControllerType::JointTrajectory,
            (0..6).map(|i| format!("j{}", i)).collect(), true);
        mgr.register("gripper", ControllerType::Gripper,
            vec!["finger".into()], false);

        // Activate arm
        mgr.activate("arm");
        assert!(mgr.is_healthy("arm"));

        // Start trajectory execution
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Busy);

        // Monitor during execution (no deviation)
        let recovery = mgr.check_deviation("arm", &[0.0; 6], &[0.01; 6]);
        assert!(recovery.is_none());

        // Complete
        mgr.complete_action(handle);
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Active);

        // Now use gripper
        mgr.activate("gripper");
        assert_eq!(mgr.num_controllers(), 2);

        // Check event log
        assert!(mgr.event_count() > 5);
    }

    // ─── Edge case: nonexistent controllers ───

    #[test]
    fn activate_nonexistent_returns_false() {
        let mut mgr = setup();
        assert!(!mgr.activate("nonexistent"));
    }

    #[test]
    fn deactivate_nonexistent_returns_false() {
        let mut mgr = setup();
        assert!(!mgr.deactivate("nonexistent"));
    }

    #[test]
    fn switch_nonexistent_returns_false() {
        let mut mgr = setup();
        mgr.activate("arm");
        assert!(!mgr.switch("arm", "nonexistent"));
    }

    #[test]
    fn velocity_scale_nonexistent_returns_none() {
        let mgr = setup();
        assert_eq!(mgr.velocity_scale("nonexistent"), None);
    }

    #[test]
    fn set_velocity_scale_nonexistent_returns_false() {
        let mut mgr = setup();
        assert!(!mgr.set_velocity_scale("nonexistent", 0.5));
    }

    #[test]
    fn health_nonexistent_returns_false() {
        let mgr = setup();
        assert!(!mgr.is_healthy("nonexistent"));
    }

    // ─── Edge case: action state transitions ───

    #[test]
    fn fail_action_directly() {
        // Intent: fail_action() should set action state to Failed and controller to Error
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        assert!(mgr.fail_action(handle, "motor fault"));
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Failed);
        assert_eq!(mgr.controller("arm").unwrap().state, ControllerState::Error);
    }

    #[test]
    fn preempt_action_directly() {
        // Intent: preempt_action() should mark action as Preempted
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);

        assert!(mgr.preempt_action(handle));
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Preempted);
    }

    #[test]
    fn complete_action_already_completed_is_idempotent() {
        // Discovery: complete_action on already-completed action returns true (idempotent)
        let mut mgr = setup();
        mgr.activate("arm");
        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);
        assert!(mgr.complete_action(handle));
        // Second complete is accepted (idempotent behavior)
        let _ = mgr.complete_action(handle);
        assert_eq!(mgr.action(handle).unwrap().state, ActionState::Succeeded);
    }

    // ─── Edge case: clear_log ───

    #[test]
    fn clear_log_empties_events() {
        let mut mgr = setup();
        mgr.activate("arm");
        assert!(mgr.event_count() > 0);
        mgr.clear_log();
        assert_eq!(mgr.event_count(), 0);
    }

    // ─── Edge case: multiple unhealthy ───

    #[test]
    fn multiple_unhealthy_controllers() {
        let mut mgr = setup();
        mgr.activate("arm");
        mgr.activate("gripper");

        let handle = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(handle);
        mgr.fail_action(handle, "motor fault");

        mgr.emergency_stop("test");

        let unhealthy = mgr.unhealthy_controllers();
        assert_eq!(unhealthy.len(), 2, "both should be unhealthy");
    }

    // ─── Edge case: deviation on nonexistent controller ───

    #[test]
    fn deviation_check_on_registered_controller() {
        // check_deviation works on any registered controller, even if inactive
        let mut mgr = setup();
        // arm is registered but not activated — check deviation with large error
        let result = mgr.check_deviation("arm", &[0.0; 6], &[1.0; 6]);
        // Should detect deviation since 1.0 > default threshold (0.05)
        assert!(result.is_some(), "large deviation should trigger recovery");
    }

    // ─── Edge case: action on busy controller ───

    #[test]
    fn follow_trajectory_on_busy_controller() {
        // Intent: creating action on busy controller preempts old action
        let mut mgr = setup();
        mgr.activate("arm");
        let h1 = mgr.follow_joint_trajectory("arm").unwrap();
        mgr.start_action(h1);

        // New action should preempt the old one
        let h2 = mgr.follow_joint_trajectory("arm");
        if let Some(h2) = h2 {
            assert_eq!(mgr.action(h1).unwrap().state, ActionState::Preempted);
            assert_eq!(mgr.action(h2).unwrap().state, ActionState::Pending);
        }
    }
}
