//! Independent safety watchdog for trajectory execution.
//!
//! Runs on a separate thread, monitoring heartbeats from the executor loop.
//! If the executor hangs, deadlocks, or crashes, the watchdog fires after
//! a configurable timeout and sends a zero-velocity command to the hardware.
//!
//! This is defense-in-depth: the deviation detection runs INSIDE the executor
//! loop, so if the loop stops, detection stops too. The watchdog runs
//! INDEPENDENTLY on its own thread.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Action taken when the watchdog fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogAction {
    /// Send zero velocity to all joints (safest default).
    ZeroVelocity,
    /// Send the last known safe command (hold position).
    HoldPosition,
    /// Just log and return — caller handles the rest.
    Abort,
}

impl Default for WatchdogAction {
    fn default() -> Self {
        Self::ZeroVelocity
    }
}

/// Configuration for the safety watchdog.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// Maximum time between heartbeats before watchdog fires.
    /// Default: 50ms (for 500Hz control loop, this allows 25 missed cycles).
    pub heartbeat_timeout: Duration,
    /// Action to take when watchdog fires.
    pub on_timeout: WatchdogAction,
    /// Number of joints (DOF) — needed for zero-velocity command size.
    pub dof: usize,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout: Duration::from_millis(50),
            on_timeout: WatchdogAction::ZeroVelocity,
            dof: 6,
        }
    }
}

/// Independent safety watchdog thread.
///
/// # Usage
///
/// ```ignore
/// let sink = Arc::new(Mutex::new(my_sink));
/// let watchdog = SafetyWatchdog::start(sink.clone(), WatchdogConfig::default());
///
/// // In your control loop:
/// loop {
///     watchdog.heartbeat(); // Signal that the loop is alive
///     // ... compute and send commands ...
/// }
///
/// // When done, drop the watchdog (thread terminates automatically)
/// ```
pub struct SafetyWatchdog {
    heartbeat_tx: mpsc::Sender<()>,
    handle: Option<std::thread::JoinHandle<bool>>,
}

impl SafetyWatchdog {
    /// Start the watchdog on a new thread.
    ///
    /// The `sink` is used to send the recovery command if the watchdog fires.
    /// The `last_safe_positions` are sent as the position component of the
    /// zero-velocity command (hold the robot at the last known safe place).
    pub fn start(
        sink: Arc<Mutex<dyn crate::CommandSink + Send>>,
        last_safe_positions: Arc<Mutex<Vec<f64>>>,
        config: WatchdogConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel();

        let handle = std::thread::Builder::new()
            .name("kinetic-safety-watchdog".into())
            .spawn(move || {
                loop {
                    match rx.recv_timeout(config.heartbeat_timeout) {
                        Ok(()) => continue, // Heartbeat received — executor is alive
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // WATCHDOG FIRED: executor is stuck
                            eprintln!(
                                "[KINETIC SAFETY WATCHDOG] No heartbeat for {:?} — executor may be stuck!",
                                config.heartbeat_timeout
                            );

                            match config.on_timeout {
                                WatchdogAction::ZeroVelocity => {
                                    let positions = last_safe_positions.lock()
                                        .map(|p| p.clone())
                                        .unwrap_or_else(|_| vec![0.0; config.dof]);
                                    let zero_vel = vec![0.0; positions.len()];
                                    if let Ok(mut s) = sink.lock() {
                                        let _ = s.send_command(&positions, &zero_vel);
                                    }
                                }
                                WatchdogAction::HoldPosition => {
                                    let positions = last_safe_positions.lock()
                                        .map(|p| p.clone())
                                        .unwrap_or_else(|_| vec![0.0; config.dof]);
                                    let zero_vel = vec![0.0; positions.len()];
                                    if let Ok(mut s) = sink.lock() {
                                        let _ = s.send_command(&positions, &zero_vel);
                                    }
                                }
                                WatchdogAction::Abort => {
                                    // Caller will handle — just signal fired
                                }
                            }

                            return true; // Watchdog fired
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            // Sender dropped (executor finished normally)
                            return false; // Clean shutdown
                        }
                    }
                }
            })
            .expect("failed to spawn watchdog thread");

        Self {
            heartbeat_tx: tx,
            handle: Some(handle),
        }
    }

    /// Signal that the executor loop is alive.
    ///
    /// Call this every control cycle (e.g., every 2ms at 500Hz).
    /// If this isn't called within `heartbeat_timeout`, the watchdog fires.
    pub fn heartbeat(&self) {
        let _ = self.heartbeat_tx.send(());
    }

    /// Check if the watchdog has fired (non-blocking).
    pub fn has_fired(&self) -> bool {
        self.handle.as_ref().map(|h| h.is_finished()).unwrap_or(true)
    }

    /// Wait for the watchdog thread to finish and return whether it fired.
    pub fn join(mut self) -> bool {
        self.handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or(false)
    }
}

impl Drop for SafetyWatchdog {
    fn drop(&mut self) {
        // Dropping the sender disconnects the channel, which terminates
        // the watchdog thread gracefully (RecvTimeoutError::Disconnected).
        // The thread will exit on the next recv_timeout call.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingSink {
        commands: Vec<(Vec<f64>, Vec<f64>)>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self { commands: vec![] }
        }
    }

    impl crate::CommandSink for RecordingSink {
        fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
            self.commands.push((positions.to_vec(), velocities.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn watchdog_fires_on_missing_heartbeats() {
        let sink = Arc::new(Mutex::new(RecordingSink::new()));
        let positions = Arc::new(Mutex::new(vec![1.0, 2.0, 3.0]));

        let config = WatchdogConfig {
            heartbeat_timeout: Duration::from_millis(30),
            on_timeout: WatchdogAction::ZeroVelocity,
            dof: 3,
        };

        let watchdog = SafetyWatchdog::start(
            sink.clone() as Arc<Mutex<dyn crate::CommandSink + Send>>,
            positions,
            config,
        );

        // DON'T send heartbeats — simulate stuck executor
        std::thread::sleep(Duration::from_millis(100));

        // Watchdog should have fired
        assert!(watchdog.has_fired(), "watchdog should have fired");

        // Should have sent a zero-velocity command
        let cmds = sink.lock().unwrap();
        assert!(!cmds.commands.is_empty(), "watchdog should have sent recovery command");
        let (pos, vel) = &cmds.commands[0];
        assert_eq!(pos, &[1.0, 2.0, 3.0], "should use last safe positions");
        assert!(vel.iter().all(|&v| v == 0.0), "velocities should be zero");
    }

    #[test]
    fn watchdog_does_not_fire_with_regular_heartbeats() {
        let sink = Arc::new(Mutex::new(RecordingSink::new()));
        let positions = Arc::new(Mutex::new(vec![0.0; 3]));

        let config = WatchdogConfig {
            heartbeat_timeout: Duration::from_millis(50),
            on_timeout: WatchdogAction::ZeroVelocity,
            dof: 3,
        };

        let watchdog = SafetyWatchdog::start(
            sink.clone() as Arc<Mutex<dyn crate::CommandSink + Send>>,
            positions,
            config,
        );

        // Send heartbeats regularly for 200ms
        for _ in 0..20 {
            watchdog.heartbeat();
            std::thread::sleep(Duration::from_millis(10));
        }

        // Watchdog should NOT have fired
        assert!(!watchdog.has_fired(), "watchdog should NOT fire with regular heartbeats");

        // No recovery commands should have been sent
        let cmds = sink.lock().unwrap();
        assert!(cmds.commands.is_empty(), "no recovery commands expected");
    }

    #[test]
    fn watchdog_fires_within_2x_timeout() {
        let sink = Arc::new(Mutex::new(RecordingSink::new()));
        let positions = Arc::new(Mutex::new(vec![0.0; 6]));

        let timeout = Duration::from_millis(20);
        let config = WatchdogConfig {
            heartbeat_timeout: timeout,
            on_timeout: WatchdogAction::ZeroVelocity,
            dof: 6,
        };

        let start = std::time::Instant::now();
        let watchdog = SafetyWatchdog::start(
            sink.clone() as Arc<Mutex<dyn crate::CommandSink + Send>>,
            positions,
            config,
        );

        // Wait for fire
        let fired = watchdog.join();
        let elapsed = start.elapsed();

        assert!(fired, "watchdog should fire");
        assert!(
            elapsed < timeout * 3,
            "should fire within ~2x timeout: elapsed {:?}, timeout {:?}",
            elapsed,
            timeout
        );
    }

    #[test]
    fn watchdog_clean_shutdown_on_drop() {
        let sink = Arc::new(Mutex::new(RecordingSink::new()));
        let positions = Arc::new(Mutex::new(vec![0.0; 3]));

        let config = WatchdogConfig {
            heartbeat_timeout: Duration::from_secs(10), // Very long — won't fire naturally
            dof: 3,
            ..Default::default()
        };

        let watchdog = SafetyWatchdog::start(
            sink.clone() as Arc<Mutex<dyn crate::CommandSink + Send>>,
            positions,
            config,
        );

        // Drop the watchdog — should terminate the thread cleanly
        drop(watchdog);
        std::thread::sleep(Duration::from_millis(50));

        // No commands should have been sent (clean shutdown, not a timeout)
        let cmds = sink.lock().unwrap();
        assert!(cmds.commands.is_empty(), "clean shutdown should not send commands");
    }

    #[test]
    fn watchdog_abort_action_sends_no_command() {
        let sink = Arc::new(Mutex::new(RecordingSink::new()));
        let positions = Arc::new(Mutex::new(vec![0.0; 3]));

        let config = WatchdogConfig {
            heartbeat_timeout: Duration::from_millis(20),
            on_timeout: WatchdogAction::Abort,
            dof: 3,
        };

        let watchdog = SafetyWatchdog::start(
            sink.clone() as Arc<Mutex<dyn crate::CommandSink + Send>>,
            positions,
            config,
        );

        let fired = watchdog.join();
        assert!(fired, "watchdog should fire");

        // Abort action doesn't send a command
        let cmds = sink.lock().unwrap();
        assert!(cmds.commands.is_empty(), "Abort action should not send commands");
    }
}
