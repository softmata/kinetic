# Production Deployment

Checklist and best practices for deploying kinetic on real robots.

## Pre-Deployment Checklist

Complete all 10 items before running kinetic on physical hardware.

- [ ] **1. Validate URDF accuracy.** Run FK at known joint configurations and
  compare the computed EE pose against physical measurements. Position error
  should be under 1mm.

- [ ] **2. Verify joint limits.** Confirm that the URDF joint limits match the
  actual hardware limits. Wrong limits cause collisions or unreachable goals.

- [ ] **3. Test IK convergence.** Run IK for 100+ random reachable poses and
  confirm convergence rate is above 95%. Check `solution.condition_number` --
  values above 100 indicate near-singularity configurations.

- [ ] **4. Validate collision model.** Add all fixed obstacles (tables, walls,
  mounting plates) to the Scene. Run the planner in simulation to verify
  collision avoidance before hardware.

- [ ] **5. Use ExecutionConfig::safe().** This auto-populates joint limits,
  enables feedback monitoring, and configures a safety watchdog.

- [ ] **6. Set velocity scaling.** Start at 10% speed (`velocity_scale: 0.1`)
  and increase gradually. Never run a new trajectory at full speed.

- [ ] **7. Implement CommandSink with hardware timeout.** Your `send_command`
  callback should timeout and return `Err` if the hardware does not respond
  within the configured `command_timeout_ms`.

- [ ] **8. Enable the safety watchdog.** Configure `WatchdogConfig` to fire
  `ZeroVelocity` if the control loop stalls for more than 50ms.

- [ ] **9. Test emergency stop.** Verify that the robot stops within one control
  cycle when the watchdog fires or when `send_command` returns an error.

- [ ] **10. Log everything.** Use `LogExecutor` in parallel to record all
  commanded positions for post-incident analysis.

## Error Handling Patterns

All kinetic errors are returned through `KineticError`. Use the built-in
classification methods to decide how to respond.

```rust
use kinetic::prelude::*;

match planner.plan(&start, &goal) {
    Ok(result) => execute(result),
    Err(e) if e.is_retryable() => {
        // PlanningTimeout, IKNotConverged, CartesianPathIncomplete
        // Try again with relaxed parameters
        let config = PlannerConfig::offline();
        let result = planner.plan_with_config(&start, &goal, config)?;
        execute(result);
    }
    Err(e) if e.is_input_error() => {
        // Bad URDF, wrong DOF, unreachable goal
        // Fix the input, do not retry
        log::error!("Input error: {e}");
    }
    Err(e) => {
        log::error!("Unexpected error: {e}");
    }
}
```

## Trajectory Validation

Always validate trajectories before executing on hardware.

```rust
use kinetic::trajectory::{TrajectoryValidator, ValidationConfig};

let validator = TrajectoryValidator::new(ValidationConfig {
    max_velocity: robot.velocity_limits(),
    max_acceleration: robot.acceleration_limits(),
    joint_limits: robot.joint_limits.clone(),
    ..Default::default()
});

let violations = validator.validate(&timed_trajectory);
if !violations.is_empty() {
    for v in &violations {
        log::warn!("Trajectory violation: {:?}", v);
    }
    // Do NOT execute. Replan or adjust time parameterization.
}
```

## Monitoring Manipulability

Track manipulability during execution to detect approaching singularities.

```rust
use kinetic::kinematics::manipulability;

let m = manipulability(&robot, &chain, &current_joints)?;
if m < 0.02 {
    log::warn!("Low manipulability ({m:.4}), near singularity");
    // Consider replanning away from this configuration
}
```

## The degraded Flag

IK solutions include a `degraded` flag indicating whether the solver fell
back to a less accurate method (e.g., scaled transpose instead of full
pseudoinverse near a singularity).

```rust
let solution = solve_ik(&robot, &chain, &target, &config)?;
if solution.degraded {
    log::warn!(
        "IK solution is degraded (condition={:.0}), verify before executing",
        solution.condition_number
    );
}
```

## Recovery Strategies

### Replan on Failure

```rust
use kinetic::prelude::*;

let loop_config = PlanExecuteConfig {
    max_replans: 3,
    replan_strategy: ReplanStrategy::FromCurrent,
    recovery: RecoveryStrategy::RetryWithRelaxedConfig,
    ..Default::default()
};

let pel = PlanExecuteLoop::new(&robot, loop_config);
let result = pel.run(&start, &goal, &scene)?;
```

### Graceful Degradation

When the primary planner fails, fall back to simpler strategies:

1. Retry with `PlannerConfig::offline()` (more iterations)
2. Try a different planner type (`PlannerType::EST` for narrow passages)
3. Use waypoint decomposition (break the path into segments)
4. Alert the operator

### Execution Monitoring

```rust
use kinetic::trajectory::{ExecutionMonitor, MonitorConfig, DeviationLevel};

let monitor = ExecutionMonitor::new(MonitorConfig::default());
// During execution, feed actual vs commanded joint positions
let level = monitor.check_deviation(&commanded, &actual);
match level {
    DeviationLevel::Normal => {}
    DeviationLevel::Warning => {
        log::warn!("Position deviation detected");
    }
    DeviationLevel::Critical => {
        log::error!("Critical deviation, emergency stop");
    }
}
```

## Safe Execution Configuration

```rust
use kinetic::execution::{ExecutionConfig, RealTimeExecutor};

// Auto-configures from robot model:
//   rate_hz: 500
//   position_tolerance: 0.05 rad
//   velocity_tolerance: 0.3 rad/s
//   joint_limits: from URDF
//   command_timeout_ms: 50
//   require_feedback: true
//   watchdog: 50ms timeout, ZeroVelocity action
let config = ExecutionConfig::safe(&robot);
let executor = RealTimeExecutor::new(config);
```

## Production Architecture

```
Sensors --> SceneNode --> Scene
                           |
Operator --> Goal -----> Planner --> Trajectory --> Executor --> Robot
                           |                          |
                    PlanExecuteLoop            SafetyWatchdog
```

Every component has a failure mode and a recovery path.
No single failure should cause uncontrolled robot motion.
