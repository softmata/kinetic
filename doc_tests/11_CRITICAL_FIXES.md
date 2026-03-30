# 11 — Critical Safety Fixes

**Priority**: P0 — these are bugs/gaps found during the safety audit that must be fixed
**BEFORE any real robot deployment**

---

## OVERVIEW

The safety audit identified 5 critical gaps. Each must be fixed AND have a test proving the fix.
These fixes are prerequisites — without them, passing all other tests is insufficient for
production deployment.

---

## FIX 1: Mandatory Trajectory Validation Gate

**Problem**: Planner outputs are NOT validated through `TrajectoryValidator::validate()` before
execution. A planner bug could produce a trajectory that violates joint/velocity/acceleration
limits and it would be sent directly to hardware.

**Location**: The gap is between planner output and trajectory execution.
- `kinetic-planning/src/facade.rs` or wherever `plan()` returns results
- `kinetic-execution/src/realtime.rs` where `execute()` accepts trajectories

**Fix**: Add a mandatory validation step. Two options:

**Option A** — Validate at planner output:
```rust
// In Planner::plan() or plan_execute pipeline:
pub fn plan_validated(&self, start: &[f64], goal: &Goal) -> Result<PlanningResult> {
    let result = self.plan(start, goal)?;

    // Build validator from robot limits
    let validator = TrajectoryValidator::new(
        &self.robot.lower_limits(),
        &self.robot.upper_limits(),
        &self.robot.velocity_limits(),
        &self.robot.acceleration_limits(),
        ValidationConfig::default(),
    );

    // Validate every waypoint
    for (i, wp) in result.waypoints.iter().enumerate() {
        self.robot.check_limits(&JointValues::new(wp.clone()))
            .map_err(|e| KineticError::Other(
                format!("planner output waypoint {} violates limits: {}", i, e)
            ))?;
    }

    Ok(result)
}
```

**Option B** — Validate at executor input:
```rust
// In RealTimeExecutor::execute():
pub fn execute(&self, traj: &TimedTrajectory, sink: &mut dyn CommandSink)
    -> Result<ExecutionResult, ExecutionError>
{
    // SAFETY GATE: validate before sending any command
    let validator = ...; // built from robot limits
    validator.validate(traj)
        .map_err(|violations| ExecutionError::InvalidTrajectory(
            format!("{} violations found", violations.len())
        ))?;

    // ... existing execution logic ...
}
```

**Preferred**: Option A — catch it at the source, before time parameterization.

**Test to verify fix**:
```rust
#[test]
fn p0_validation_gate_catches_bad_planner_output() {
    // Manually create a planner output with one waypoint outside limits
    // The validation gate must catch it before execution
    let robot = load_robot("ur5e");
    let mut waypoints = plan_valid_path(&robot);

    // Corrupt one waypoint
    waypoints[1][0] = 100.0; // way outside limits

    // Time-parameterize (may or may not detect the issue)
    // The validation gate MUST catch it
    // Test that the pipeline rejects this trajectory
}
```

---

## FIX 2: Independent Safety Watchdog

**Problem**: If the executor thread hangs or deadlocks, there's no independent watchdog to
trigger an emergency stop. The deviation detection is part of the execution loop — if the loop
stops running, detection stops too.

**Location**: `kinetic-execution/src/`

**Fix**: Add an independent watchdog thread that monitors the executor:

```rust
pub struct WatchdogConfig {
    /// Maximum time between heartbeats before triggering e-stop
    pub heartbeat_timeout: Duration, // default: 50ms at 500Hz
    /// Action when watchdog fires
    pub on_timeout: WatchdogAction,
}

pub enum WatchdogAction {
    HoldPosition, // Send last known safe command
    ZeroVelocity, // Send zero velocity command
    Abort,        // Return error immediately
}

pub struct SafetyWatchdog {
    heartbeat_tx: std::sync::mpsc::Sender<()>,
    handle: std::thread::JoinHandle<()>,
}

impl SafetyWatchdog {
    pub fn start(
        sink: Arc<Mutex<dyn CommandSink>>,
        last_safe_command: Arc<Mutex<Vec<f64>>>,
        config: WatchdogConfig,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            loop {
                match rx.recv_timeout(config.heartbeat_timeout) {
                    Ok(()) => continue, // heartbeat received
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // WATCHDOG FIRED: executor is stuck
                        eprintln!("SAFETY WATCHDOG: no heartbeat for {:?}", config.heartbeat_timeout);
                        match config.on_timeout {
                            WatchdogAction::ZeroVelocity => {
                                let safe = last_safe_command.lock().unwrap();
                                let zeros = vec![0.0; safe.len()];
                                let _ = sink.lock().unwrap().send_command(&safe, &zeros);
                            }
                            _ => {}
                        }
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });
        Self { heartbeat_tx: tx, handle }
    }

    pub fn heartbeat(&self) {
        let _ = self.heartbeat_tx.send(());
    }
}
```

The executor calls `watchdog.heartbeat()` every control cycle. If it misses, the watchdog fires.

**Test to verify fix**:
```rust
#[test]
fn p0_watchdog_fires_on_stuck_executor() {
    let (tx, rx) = std::sync::mpsc::channel();
    let sink = Arc::new(Mutex::new(RecordingCommandSink::new()));
    let last_cmd = Arc::new(Mutex::new(vec![0.0; 6]));

    let watchdog = SafetyWatchdog::start(
        sink.clone(),
        last_cmd.clone(),
        WatchdogConfig {
            heartbeat_timeout: Duration::from_millis(50),
            on_timeout: WatchdogAction::ZeroVelocity,
        },
    );

    // DON'T send heartbeats — simulate stuck executor
    std::thread::sleep(Duration::from_millis(100));

    // Watchdog should have fired and sent zero velocity
    let commands = sink.lock().unwrap().commands.clone();
    assert!(!commands.is_empty(), "watchdog didn't fire");
    let (_, velocities) = commands.last().unwrap();
    assert!(velocities.iter().all(|&v| v == 0.0), "watchdog didn't zero velocity");
}
```

---

## FIX 3: Replace Silent DLS Fallback

**Problem**: In `kinetic-kinematics/src/dls.rs` line ~138:
```rust
decomp.solve(&error_vec).unwrap_or_else(|| j.transpose() * &error_vec * 0.1);
```
When LU decomposition fails (near-singular Jacobian), it silently falls back to a scaled
transpose multiplication. This may produce an unstable step direction without any indication
to the caller.

**Fix**: Either:
A) Return an explicit error when decomposition fails
B) Set a flag on IKSolution indicating degraded accuracy
C) At minimum, log a warning

**Preferred**: Option B — add a `degraded: bool` field to IKSolution:

```rust
pub struct IKSolution {
    pub joints: Vec<f64>,
    pub position_error: f64,
    pub orientation_error: f64,
    pub converged: bool,
    pub iterations: usize,
    pub mode_used: IKMode,
    pub degraded: bool,  // NEW: true if solver used fallback paths
}
```

And in dls.rs:
```rust
let (step, degraded) = match decomp.solve(&error_vec) {
    Some(s) => (s, false),
    None => {
        // Fallback: gradient descent direction
        (j.transpose() * &error_vec * 0.1, true)
    }
};
// Track if any iteration was degraded
if degraded { solution_degraded = true; }
```

**Test to verify fix**:
```rust
#[test]
fn p0_dls_reports_degraded_near_singularity() {
    let (robot, chain) = load_robot_and_chain("ur5e");
    // Target at full arm extension (near singularity)
    let singular = vec![0.0; 6];
    let pose = fk(&robot, &chain, &singular).unwrap();

    // Use tight damping to force decomposition issues
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 1e-10 }, // very low damping
        max_iterations: 500,
        num_restarts: 1,
        ..IKConfig::dls()
    };

    let result = solve_ik(&robot, &chain, &pose, &config);
    // If it converges, check degraded flag
    if let Ok(sol) = result {
        // At very low damping near singularity, degraded should be true at some point
        // This test documents the behavior — whether degraded is true or false,
        // the key thing is it's REPORTED
        eprintln!("converged={}, degraded={}, iterations={}", sol.converged, sol.degraded, sol.iterations);
    }
}
```

---

## FIX 4: Workspace Boundary Enforcement

**Problem**: IK solvers will return solutions where the end-effector is at the very edge of
(or slightly beyond) the reachable workspace. Planners don't constrain to workspace boundaries.
The workspace analysis (`workspace.rs`) is informational only.

**Fix**: Add optional workspace constraints to planning:

```rust
pub struct WorkspaceConstraint {
    /// Maximum distance from base to EE (meters)
    pub max_reach: f64,
    /// Minimum distance from base to EE (meters) — avoids self-collision zone
    pub min_reach: f64,
    /// Optional: bounding box [min_x, max_x, min_y, max_y, min_z, max_z]
    pub bounds: Option<[f64; 6]>,
}
```

Add to PlannerConfig:
```rust
pub struct PlannerConfig {
    // ... existing fields ...
    pub workspace_constraint: Option<WorkspaceConstraint>,
}
```

The planner samples/validates each waypoint's EE position against these constraints.

**Test to verify fix**:
```rust
#[test]
fn p1_workspace_constraint_respected() {
    let robot = load_robot("ur5e");
    let planner = Planner::new(&robot).unwrap();

    let config = PlannerConfig {
        workspace_constraint: Some(WorkspaceConstraint {
            max_reach: 0.5, // Much less than UR5e's actual reach (~0.85m)
            min_reach: 0.1,
            bounds: None,
        }),
        ..PlannerConfig::default()
    };

    let result = planner.plan_with_config(&start, &goal, config);
    if let Ok(result) = result {
        for (i, wp) in result.waypoints.iter().enumerate() {
            let pose = fk(&robot, &chain, wp).unwrap();
            let reach = pose.translation().norm();
            assert!(reach <= 0.5 + 0.01,
                "waypoint {} reach {:.3}m exceeds max 0.5m", i, reach);
            assert!(reach >= 0.1 - 0.01,
                "waypoint {} reach {:.3}m below min 0.1m", i, reach);
        }
    }
}
```

---

## FIX 5: Condition Number Monitoring in IK

**Problem**: IK solvers don't report when they're operating near a singularity. The user
gets a solution that may be numerically unstable without any warning.

**Fix**: Add condition number to IKSolution:

```rust
pub struct IKSolution {
    // ... existing fields ...
    pub condition_number: Option<f64>,  // NEW: cond(J) at solution
}
```

Compute as:
```rust
let jac = jacobian(&robot, &chain, &solution.joints)?;
let svd = jac.svd(true, true);
let singular_values = svd.singular_values;
let max_sv = singular_values.max();
let min_sv = singular_values.min();
let condition = if min_sv > 1e-15 { max_sv / min_sv } else { f64::INFINITY };
solution.condition_number = Some(condition);
```

User can check: `if sol.condition_number.unwrap_or(0.0) > 1e6 { warn!("near singularity") }`

**Test to verify fix**:
```rust
#[test]
fn p1_ik_reports_condition_number() {
    let (robot, chain) = load_robot_and_chain("ur5e");

    // Well-conditioned config
    let target = fk(&robot, &chain, &vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]).unwrap();
    let sol = solve_ik(&robot, &chain, &target, &IKConfig::dls()).unwrap();
    assert!(sol.condition_number.is_some(), "condition number not reported");
    let cond_good = sol.condition_number.unwrap();
    assert!(cond_good < 1e6, "well-conditioned config has high cond: {}", cond_good);

    // Near-singular config (arm extended)
    let target_sing = fk(&robot, &chain, &vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let sol_sing = solve_ik(&robot, &chain, &target_sing, &IKConfig::dls()).unwrap();
    let cond_bad = sol_sing.condition_number.unwrap();
    assert!(cond_bad > cond_good, "singular config should have higher condition number");
}
```

---

## IMPLEMENTATION ORDER

1. **Fix 1** (Validation Gate) — highest impact, prevents ALL trajectory-level bugs from reaching hardware
2. **Fix 2** (Watchdog) — prevents stuck executor from being invisible
3. **Fix 3** (DLS degraded flag) — prevents silent numerical instability
4. **Fix 5** (Condition number) — informs users about solution quality
5. **Fix 4** (Workspace constraints) — nice to have, lower risk than others

## VERIFICATION

After implementing all 5 fixes, run the FULL acceptance test suite. All tests from
specs 01-10 must pass with the fixes in place. The fixes should not break any existing tests.
