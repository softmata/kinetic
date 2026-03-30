# Servo Control

## What You'll Learn
- Create a `Servo` controller for real-time Cartesian velocity control
- Send twist (linear + angular velocity) commands at 100 Hz
- Monitor manipulability and singularity proximity
- Execute multi-phase motions with collision awareness

## Prerequisites
- [Reactive Control](../../concepts/reactive-control.md)
- [Forward Kinematics](../../concepts/forward-kinematics.md)
- [Collision Detection](../../concepts/collision-detection.md)

## Overview

Servo control lets you command a robot's end-effector velocity in real time,
useful for teleoperation, visual servoing, and force-guided assembly. Kinetic's
`Servo` controller converts Cartesian twist commands into joint velocity commands
at each timestep, while monitoring singularity proximity and collision distances.
This tutorial runs a simulated 2-second servo loop with two motion phases:
forward along X, then sideways along Y.

## Step 1: Load Robot and Create Scene

```rust
use std::sync::Arc;
use kinetic::core::Twist;
use kinetic::prelude::*;
use kinetic::reactive::servo::{Servo, ServoConfig};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let robot = Arc::new(Robot::from_urdf_string(ARM_URDF)?);
    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.4, 0.01], [0.6, 0.0, 0.3]);
    let scene = Arc::new(scene);
```

**What this does:** Loads a 6-DOF arm and creates a scene with a table obstacle. Both are wrapped in `Arc` because `Servo` needs shared ownership for internal threading.

**Why:** Servo control runs at high frequency (100+ Hz) and needs fast access to robot kinematics and collision data. `Arc` enables safe shared access without copying the robot model or scene at every timestep.

## Step 2: Configure and Initialize Servo

```rust
    let config = ServoConfig {
        rate_hz: 100.0,
        singularity_threshold: 0.005,
        ..Default::default()
    };
    let mut servo = Servo::new(&robot, &scene, config)?;

    let initial_joints = vec![0.0, -0.8, 1.0, 0.0, -0.5, 0.0];
    let initial_vel = vec![0.0; robot.dof];
    servo.set_state(&initial_joints, &initial_vel)?;
```

**What this does:** Creates a `Servo` controller at 100 Hz. `singularity_threshold` controls when the controller starts scaling down velocities near singular configurations. `set_state` initializes the joint positions and velocities.

**Why:** The rate determines the timestep for integration (dt = 1/rate_hz = 10 ms). A higher rate gives smoother motion but requires faster computation. The singularity threshold is a manipulability value below which the controller degrades gracefully instead of producing large joint velocities.

## Step 3: Phase 1 — Forward Motion

```rust
    let forward_twist = Twist::new(
        Vector3::new(0.05, 0.0, 0.0),  // 5 cm/s forward
        Vector3::zeros(),               // no rotation
    );

    for step in 0..100 {
        let cmd = servo.send_twist(&forward_twist)?;
        let state = servo.state();

        if step % 20 == 0 {
            print!("  t={:.2}s  joints=[", step as f64 / 100.0);
            for (i, p) in cmd.positions.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:.3}", p);
            }
            println!("]");
            println!(
                "          manip={:.4}  near_singularity={}  near_collision={}",
                state.manipulability, state.is_near_singularity, state.is_near_collision,
            );
        }
    }
```

**What this does:** Sends 100 twist commands (1 second at 100 Hz), each requesting 5 cm/s forward motion in the X direction. `send_twist` returns a `JointCommand` with the computed joint positions. The state reports manipulability (a scalar measuring how "dexterous" the current configuration is) and collision/singularity proximity flags.

**Why:** Each `send_twist` call does: (1) compute Jacobian at current configuration, (2) invert it to get joint velocities from Cartesian velocity, (3) integrate joint positions forward by dt, (4) check for singularity and collision. The returned `cmd.positions` are what you would send to a real robot controller.

## Step 4: Phase 2 — Sideways Motion

```rust
    let sideways_twist = Twist::new(
        Vector3::new(0.0, 0.05, 0.0),  // 5 cm/s sideways
        Vector3::zeros(),
    );

    for step in 0..100 {
        let cmd = servo.send_twist(&sideways_twist)?;
        // ... same logging pattern
    }
```

**What this does:** Changes the twist direction to the Y axis and runs another 100 steps. The motion seamlessly transitions from forward to sideways without stopping.

**Why:** Phase-based motion demonstrates how to compose servo commands in sequence. In teleoperation, the twist comes from a joystick or spacemouse. In visual servoing, it comes from an image-based controller. The servo loop does not care about the source — it converts whatever twist you provide.

## Step 5: Inspect Final State

```rust
    let final_state = servo.state();
    let ee = final_state.ee_pose.translation.vector;

    println!("Final EE position: ({:.4}, {:.4}, {:.4})", ee.x, ee.y, ee.z);
    println!("Manipulability: {:.4}", final_state.manipulability);
    println!("Near singularity: {}", final_state.is_near_singularity);
    println!("Near collision: {}", final_state.is_near_collision);
    println!("Min obstacle distance: {:.4} m", final_state.min_obstacle_distance);

    Ok(())
}
```

**What this does:** Reads the final servo state after both motion phases. The state includes the end-effector pose, manipulability, singularity and collision flags, and the minimum distance to any obstacle.

**Why:** `min_obstacle_distance` is critical for safety monitoring. If it drops below your safety threshold, you should reduce speed or stop. `manipulability` trending toward zero warns of an approaching singularity where the robot loses a degree of freedom and small Cartesian motions require large joint velocities.

## Complete Code

```rust
use std::sync::Arc;
use kinetic::core::Twist;
use kinetic::prelude::*;
use kinetic::reactive::servo::{Servo, ServoConfig};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let robot = Arc::new(Robot::from_urdf_string(ARM_URDF)?);
    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.4, 0.01], [0.6, 0.0, 0.3]);
    let scene = Arc::new(scene);

    let config = ServoConfig { rate_hz: 100.0, singularity_threshold: 0.005, ..Default::default() };
    let mut servo = Servo::new(&robot, &scene, config)?;
    servo.set_state(&[0.0, -0.8, 1.0, 0.0, -0.5, 0.0], &vec![0.0; robot.dof])?;

    // Phase 1: Forward (X axis, 1 second)
    let fwd = Twist::new(Vector3::new(0.05, 0.0, 0.0), Vector3::zeros());
    for _ in 0..100 { servo.send_twist(&fwd)?; }

    // Phase 2: Sideways (Y axis, 1 second)
    let side = Twist::new(Vector3::new(0.0, 0.05, 0.0), Vector3::zeros());
    for _ in 0..100 { servo.send_twist(&side)?; }

    let state = servo.state();
    let ee = state.ee_pose.translation.vector;
    println!("EE: ({:.4}, {:.4}, {:.4}), manip={:.4}", ee.x, ee.y, ee.z, state.manipulability);

    Ok(())
}
```

## What You Learned
- `Servo::new(&robot, &scene, config)` creates a real-time Cartesian velocity controller
- `Twist::new(linear, angular)` specifies desired end-effector velocity
- `send_twist()` converts Cartesian velocity to joint commands at each timestep
- Servo state reports `manipulability`, `is_near_singularity`, `is_near_collision`, and `min_obstacle_distance`
- Phase-based motion is achieved by changing the twist command between loop iterations
- `ServoConfig::rate_hz` determines the integration timestep (dt = 1/rate)

## Try This
- Add angular velocity to the twist: `Twist::new(linear, Vector3::new(0.0, 0.0, 0.1))` for rotation about Z
- Reduce `singularity_threshold` to 0.001 and drive the arm toward a singular configuration to observe the scaling behavior
- Move the table obstacle closer to the arm and observe `is_near_collision` and `min_obstacle_distance`
- Increase `rate_hz` to 500 and compare the smoothness of the end-effector trajectory

## Next
- [Grasp Planning](grasp-planning.md) — generating grasp candidates for objects
- [Pick and Place](pick-and-place.md) — combining servo with task planning
