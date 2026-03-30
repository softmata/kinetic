# Hardware Integration

Connecting kinetic trajectories to real robot hardware.

## Trajectory Format

Kinetic produces `TimedTrajectory` objects containing timestamped waypoints
with positions, velocities, and optional accelerations.

```rust
use kinetic::trajectory::{TimedTrajectory, TimedWaypoint};

// TimedWaypoint structure:
// - time: f64 (seconds from trajectory start)
// - positions: Vec<f64> (joint angles in radians)
// - velocities: Vec<f64> (joint velocities in rad/s)
// - accelerations: Option<Vec<f64>> (joint accelerations in rad/s^2)
```

Time parameterization converts a geometric path (waypoints without timing)
into a timed trajectory that respects velocity and acceleration limits:

```rust
use kinetic::prelude::*;

let result = planner.plan(&start, &goal)?;
let vel_limits = robot.velocity_limits();
let accel_limits = robot.acceleration_limits();

// Trapezoidal (fast, per-joint limits)
let timed = trapezoidal(&result.waypoints, &vel_limits, &accel_limits)?;

// TOTP (time-optimal, tighter trajectory)
let timed = totp(&result.waypoints, &vel_limits, &accel_limits)?;
```

## The CommandSink Trait

`CommandSink` is the interface between kinetic and your robot hardware.
Implement it once for each robot driver.

```rust
use kinetic::execution::CommandSink;

struct MyRobotDriver {
    connection: TcpStream,
}

impl CommandSink for MyRobotDriver {
    fn send_command(
        &mut self,
        positions: &[f64],
        velocities: &[f64],
    ) -> Result<(), String> {
        // Format and send to your controller
        // Return Err("...") on hardware fault
        Ok(())
    }
}
```

The executor calls `send_command` at the configured rate (default: 500 Hz),
interpolating between trajectory waypoints.

## 500 Hz Control Loop Pattern

The `RealTimeExecutor` handles the timing loop. You provide the `CommandSink`,
it handles interpolation, timing, and error detection.

```rust
use kinetic::execution::{RealTimeExecutor, ExecutionConfig};

let config = ExecutionConfig {
    rate_hz: 500.0,
    position_tolerance: 0.05,
    velocity_tolerance: 0.3,
    ..ExecutionConfig::default()
};

let executor = RealTimeExecutor::new(config);
let result = executor.execute(&timed_trajectory, &mut my_driver)?;

println!("Executed in {:.2}s", result.actual_duration.as_secs_f64());
println!("Max deviation: {:.4} rad", result.max_deviation);
```

For feedback-based monitoring, implement `FeedbackSource`:

```rust
use kinetic::execution::FeedbackSource;

impl FeedbackSource for MyRobotDriver {
    fn read_positions(&self) -> Option<Vec<f64>> {
        // Read actual joint positions from encoders
        Some(self.read_encoders())
    }
}
```

## Connecting to Universal Robots via URScript

UR robots accept URScript commands over TCP port 30002 (secondary interface)
or 30003 (real-time interface at 500 Hz).

```rust
use std::io::Write;
use std::net::TcpStream;
use kinetic::execution::CommandSink;

struct URDriver {
    stream: TcpStream,
}

impl URDriver {
    fn new(ip: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(format!("{ip}:30003"))?;
        Ok(Self { stream })
    }
}

impl CommandSink for URDriver {
    fn send_command(
        &mut self,
        positions: &[f64],
        velocities: &[f64],
    ) -> Result<(), String> {
        // URScript servoj command
        let cmd = format!(
            "servoj([{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}], t=0.002)\n",
            positions[0], positions[1], positions[2],
            positions[3], positions[4], positions[5],
        );
        self.stream.write_all(cmd.as_bytes())
            .map_err(|e| format!("UR send failed: {e}"))
    }
}
```

## General Driver Pattern

For any robot, follow this pattern:

1. **Connect** to the robot controller (TCP, serial, shared memory, etc.)
2. **Implement `CommandSink`** with your protocol's command format
3. **Optionally implement `FeedbackSource`** for deviation monitoring
4. **Create an `ExecutionConfig`** appropriate for your robot
5. **Execute** with `RealTimeExecutor`

```rust
use kinetic::prelude::*;
use kinetic::execution::*;

// 1. Plan
let robot = Robot::from_name("ur5e")?;
let planner = Planner::new(&robot)?;
let result = planner.plan(&start, &goal)?;

// 2. Time-parameterize
let timed = trapezoidal(
    &result.waypoints,
    &robot.velocity_limits(),
    &robot.acceleration_limits(),
)?;

// 3. Validate
let validator = kinetic::trajectory::TrajectoryValidator::new(Default::default());
let violations = validator.validate(&timed);
assert!(violations.is_empty(), "Trajectory has violations");

// 4. Execute
let config = ExecutionConfig::safe(&robot);
let executor = RealTimeExecutor::new(config);
let mut driver = MyRobotDriver::connect("192.168.1.100")?;
let exec_result = executor.execute(&timed, &mut driver)?;
```

## Trajectory Export

For robots that accept trajectory files instead of streaming commands,
use `trajectory_to_csv_file` or `trajectory_to_json_file`. Import with
`trajectory_from_csv` and `trajectory_from_json`.

See the Production Deployment guide for safety watchdog configuration.
