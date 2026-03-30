# 06 — Execution Safety Tests

**Priority**: P0 — the execution layer sends commands to real motors. Bugs here move the robot.
**File**: `crates/kinetic/tests/acceptance/test_execution_safety.rs`
**Estimated cases**: ~100

---

## PRINCIPLE

The execution layer is the LAST software barrier before physical motion. It must:
1. Send commands at a stable rate
2. Detect when the real robot deviates from the plan
3. Stop immediately on error
4. Never send commands outside joint limits
5. Start and end at zero velocity (robot at rest)
6. Handle all failure modes gracefully

---

## SHARED TEST INFRASTRUCTURE

```rust
/// Records all commands sent for verification.
struct RecordingSink {
    commands: Vec<(Vec<f64>, Vec<f64>)>, // (positions, velocities)
    timestamps: Vec<std::time::Instant>,
    start: std::time::Instant,
}
impl RecordingSink {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Self { commands: Vec::new(), timestamps: Vec::new(), start: now }
    }
}
impl CommandSink for RecordingSink {
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
        self.commands.push((positions.to_vec(), velocities.to_vec()));
        self.timestamps.push(std::time::Instant::now());
        Ok(())
    }
}

/// Simulates a robot that follows commands perfectly.
struct PerfectFeedback {
    last_commanded: std::sync::Mutex<Option<Vec<f64>>>,
}
impl FeedbackSource for PerfectFeedback {
    fn read_positions(&self) -> Option<Vec<f64>> {
        self.last_commanded.lock().unwrap().clone()
    }
}

/// Simulates a robot that's stuck (never moves from initial position).
struct StuckFeedback {
    stuck_at: Vec<f64>,
}
impl FeedbackSource for StuckFeedback {
    fn read_positions(&self) -> Option<Vec<f64>> {
        Some(self.stuck_at.clone())
    }
}

/// Simulates feedback dropout.
struct DroppingFeedback {
    positions: Vec<f64>,
    call_count: std::sync::atomic::AtomicUsize,
    drop_every_n: usize,
}
impl FeedbackSource for DroppingFeedback {
    fn read_positions(&self) -> Option<Vec<f64>> {
        let n = self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % self.drop_every_n == 0 { None } else { Some(self.positions.clone()) }
    }
}

/// CommandSink that fails after N commands.
struct FailingSink {
    fail_after: usize,
    sent: usize,
}
impl CommandSink for FailingSink {
    fn send_command(&mut self, _: &[f64], _: &[f64]) -> Result<(), String> {
        self.sent += 1;
        if self.sent >= self.fail_after {
            Err("hardware fault".to_string())
        } else {
            Ok(())
        }
    }
}
```

---

## TEST 1: SimExecutor Traverses All Waypoints

**Function**: `p0_sim_executor_traverses_all`

For `REPRESENTATIVE_ROBOTS`:
1. Create a timed trajectory (plan → trapezoidal)
2. Execute with `SimExecutor` + `RecordingSink`
3. Verify:
   - `result.state == ExecutionState::Completed`
   - At least `waypoints.len()` commands were sent
   - First command matches start position
   - Last command matches goal position

---

## TEST 2: RealTimeExecutor Command Rate

**Function**: `p0_realtime_executor_command_rate`

1. Create a 2-second trajectory for "ur5e"
2. Execute with `RealTimeExecutor { rate_hz: 500.0, .. }` + `RecordingSink`
3. Verify:
   - ~1000 commands sent (500Hz × 2s, ±10%)
   - Time between consecutive commands: mean ~2ms, max <5ms (jitter bound)

```rust
let executor = RealTimeExecutor::new(ExecutionConfig {
    rate_hz: 500.0,
    ..ExecutionConfig::default()
});

let mut sink = RecordingSink::new();
let result = executor.execute(&traj, &mut sink).unwrap();

let expected_commands = (traj.duration.as_secs_f64() * 500.0) as usize;
assert!(sink.commands.len() >= expected_commands * 90 / 100,
    "too few commands: {} (expected ~{})", sink.commands.len(), expected_commands);

// Check timing jitter
for pair in sink.timestamps.windows(2) {
    let dt = pair[1].duration_since(pair[0]);
    assert!(dt < Duration::from_millis(5),
        "command gap {:?} exceeds 5ms", dt);
}
```

---

## TEST 3: Deviation Detection — Stuck Robot

**Function**: `p0_deviation_detection_stuck_robot`

1. Create trajectory that moves the arm
2. Provide `StuckFeedback` (robot doesn't move)
3. Execute
4. Must return `Err(DeviationExceeded { .. })` within a few control cycles
5. Must NOT continue sending commands after detection

```rust
let stuck = StuckFeedback { stuck_at: start.clone() };
let result = executor.execute_with_feedback(&traj, &mut sink, &stuck);

match result {
    Err(ExecutionError::DeviationExceeded { deviation, tolerance, .. }) => {
        assert!(deviation > tolerance, "deviation {} should exceed tolerance {}", deviation, tolerance);
    }
    Err(ExecutionError::Timeout { .. }) => {} // Also acceptable
    Ok(_) => panic!("should have detected stuck robot"),
    Err(other) => panic!("unexpected error: {:?}", other),
}
```

---

## TEST 4: Deviation Detection — Drifting Robot

**Function**: `p0_deviation_detection_drifting`

1. Provide feedback that slowly drifts from the commanded position
2. Once drift exceeds `position_tolerance`, execution must abort

```rust
struct DriftingFeedback {
    drift_per_call: f64,
    base: Vec<f64>,
    calls: std::sync::atomic::AtomicUsize,
}
impl FeedbackSource for DriftingFeedback {
    fn read_positions(&self) -> Option<Vec<f64>> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as f64;
        Some(self.base.iter().map(|&v| v + n * self.drift_per_call).collect())
    }
}
```

---

## TEST 5: Hardware Fault Handling

**Function**: `p0_hardware_fault_handling`

1. Use `FailingSink` that returns `Err` after 10 commands
2. Execute
3. Must return `Err(CommandFailed(_))`
4. Must stop immediately — no more commands sent after the failure

```rust
let mut sink = FailingSink { fail_after: 10, sent: 0 };
let result = executor.execute(&traj, &mut sink);

assert!(matches!(result, Err(ExecutionError::CommandFailed(_))),
    "should report command failure");
assert_eq!(sink.sent, 10, "should have stopped at failure point");
```

---

## TEST 6: Timeout Handling

**Function**: `p0_execution_timeout`

1. Create a very long trajectory (60 seconds)
2. Set `timeout_factor: 0.1` (timeout = 6 seconds, less than trajectory duration)
3. Execute
4. Must return `Err(Timeout { .. })` after ~6 seconds

---

## TEST 7: Commands Start and End at Zero Velocity

**Function**: `p0_commands_zero_start_end_velocity`

For `REPRESENTATIVE_ROBOTS`:
1. Execute any trajectory with `RecordingSink`
2. First command: velocity components must be ~0 (within 1e-6)
3. Last command: velocity components must be ~0 (within 1e-6)

Robot must not start or stop with nonzero velocity — it must be at rest.

---

## TEST 8: Commands Within Joint Limits

**Function**: `p0_all_commands_within_limits`

For `REPRESENTATIVE_ROBOTS`:
1. Execute trajectory with `RecordingSink`
2. Every single commanded position must be within joint limits

```rust
for (ci, (positions, _)) in sink.commands.iter().enumerate() {
    for (ji, &pos) in positions.iter().enumerate() {
        let joint_idx = chain.active_joints[ji];
        if let Some(ref lim) = robot.joints[joint_idx].limits {
            assert!(pos >= lim.lower - 1e-6 && pos <= lim.upper + 1e-6,
                "command {} joint {}: {} outside [{}, {}]",
                ci, ji, pos, lim.lower, lim.upper);
        }
    }
}
```

---

## TEST 9: Commands Are Monotonically Interpolated

**Function**: `p1_commands_monotonic_interpolation`

1. Execute with RecordingSink
2. Between consecutive commands, position change must be smooth:
   `|cmd[i+1].pos[j] - cmd[i].pos[j]| < vel_limit[j] * dt * 1.5`
3. No sudden jumps in commanded position

---

## TEST 10: ExecutionResult Correctness

**Function**: `p1_execution_result_fields`

After successful execution:
```rust
assert_eq!(result.state, ExecutionState::Completed);
assert!(result.actual_duration >= traj.duration * 0.9); // at least 90% of planned
assert!(result.actual_duration <= traj.duration * 1.2); // at most 120% of planned
assert!(result.commands_sent > 0);
assert_eq!(result.final_positions.len(), dof);
```

---

## TEST 11: ExecutionMonitor Warning Level

**Function**: `p1_execution_monitor_warning_level`

Using `ExecutionMonitor` directly:
1. Create monitor with `position_tolerance: 0.1, warning_fraction: 0.7`
2. Feed positions that are 0.05 rad off (below warning threshold) → `DeviationLevel::Normal`
3. Feed positions that are 0.075 rad off (above 70% of 0.1) → `DeviationLevel::Warning`
4. Feed positions that are 0.15 rad off (above tolerance) → `DeviationLevel::Abort`

---

## TEST 12: LogExecutor Records All Commands

**Function**: `p1_log_executor_records`

1. Execute with `LogExecutor` to a temp file
2. Verify the file contains all waypoints
3. Verify the file is parseable (e.g., CSV or JSON)

---

## TEST 13: Concurrent Execution Safety

**Function**: `p1_concurrent_execution_no_race`

1. Create two independent executors for two different trajectories
2. Execute both in parallel (separate threads)
3. Both must complete without data races or panics
4. Each executor's RecordingSink must contain only its own trajectory's commands

---

## TEST 14: Empty Trajectory Handling

**Function**: `p0_empty_trajectory_execution`

1. Create a trajectory with 0 waypoints (or 1 waypoint at start)
2. Execute
3. Must return `Ok(Completed)` with 0 or 1 commands, OR `Err(InvalidTrajectory)`
4. Must NOT panic or hang

---

## TEST 15: Graceful Process Interruption

**Function**: `p1_graceful_interruption`

1. Start execution in a separate thread
2. After 100ms, signal cancellation (drop the executor or set a cancel flag)
3. Verify the last command sent was a safe state (zero velocity or hold position)
4. Verify the thread exits cleanly

---

## SUMMARY

| Test | Priority | What It Catches |
|------|----------|-----------------|
| SimExecutor traversal | P0 | Missed waypoints |
| RealTime command rate | P0 | Timing instability |
| Stuck robot detection | P0 | Undetected mechanical jam |
| Drifting detection | P0 | Gradual divergence |
| Hardware fault | P0 | Continued commands after fault |
| Timeout | P0 | Infinite execution |
| Zero start/end velocity | P0 | Violent start/stop |
| Commands within limits | P0 | Joint damage |
| Monotonic interpolation | P1 | Command jumps |
| Result correctness | P1 | Wrong metadata |
| Monitor warning level | P1 | Incorrect thresholds |
| Log executor | P1 | Missing recording |
| Concurrent safety | P1 | Race conditions |
| Empty trajectory | P0 | Edge case panic |
| Graceful interruption | P1 | Unsafe shutdown |
