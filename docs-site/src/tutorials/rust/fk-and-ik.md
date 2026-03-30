# FK and IK Roundtrip

## What You'll Learn
- Load a robot from a URDF string
- Extract a kinematic chain between two links
- Compute forward kinematics (FK) to get end-effector pose
- Solve inverse kinematics (IK) to recover joint angles
- Verify FK/IK roundtrip accuracy

## Prerequisites
- [Forward Kinematics](../../concepts/forward-kinematics.md)
- [Inverse Kinematics](../../concepts/inverse-kinematics.md)
- [Robots and URDF](../../concepts/robots-and-urdf.md)

## Overview

Forward kinematics maps joint angles to an end-effector pose. Inverse kinematics
does the reverse: given a desired pose, find joint angles that reach it. This
tutorial walks through both directions using a 7-DOF Panda-like arm, showing the
complete FK-to-IK roundtrip and verifying sub-millimeter accuracy.

## Step 1: Load the Robot

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    println!("Loaded '{}' — {} DOF", robot.name, robot.dof);
```

**What this does:** Parses a URDF XML string into a `Robot` struct containing links, joints, and joint limits.

**Why:** `Robot` is the central data structure in kinetic. It holds all the geometric and kinematic information needed for FK and IK. In production, you would use `Robot::from_urdf("path/to/robot.urdf")` or `Robot::from_name("franka_panda")` to load from a file or the built-in robot library.

## Step 2: Extract the Kinematic Chain

```rust
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;
```

**What this does:** Walks the robot's link tree from `panda_link0` (base) to `panda_link8` (end-effector flange) and extracts the ordered list of active joints along that path.

**Why:** A robot may have multiple chains (e.g., a dual-arm robot). `KinematicChain` isolates the joints relevant to one specific base-to-tip path, so FK and IK operate on exactly the right degrees of freedom.

## Step 3: Compute Forward Kinematics

```rust
    let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let start = std::time::Instant::now();
    let pose = forward_kinematics(&robot, &chain, &q)?;
    let fk_time = start.elapsed();

    let t = pose.translation();
    println!(
        "FK → position: ({:.4}, {:.4}, {:.4})  [{:?}]",
        t.x, t.y, t.z, fk_time
    );
```

**What this does:** Multiplies the chain of homogeneous transforms (one per joint) to compute the 6D pose (position + orientation) of the end-effector. The result is a `Pose` (an `Isometry3<f64>`) in the base frame.

**Why:** FK is the foundation of robotics — it tells you where the end-effector is given the current joint state. Kinetic's FK runs in microseconds, making it fast enough for real-time control loops.

## Step 4: Solve Inverse Kinematics

```rust
    let config = IKConfig::dls()
        .with_seed(robot.mid_configuration().to_vec())
        .with_max_iterations(300);

    let start = std::time::Instant::now();
    let solution = solve_ik(&robot, &chain, &pose, &config)?;
    let ik_time = start.elapsed();

    println!(
        "IK → converged in {} iters, pos_err={:.2e}, orient_err={:.2e}  [{:?}]",
        solution.iterations, solution.position_error, solution.orientation_error, ik_time
    );
```

**What this does:** Uses Damped Least Squares (DLS) to iteratively find joint angles that place the end-effector at the target `pose`. The seed — `mid_configuration()` — gives the solver a starting guess at the midpoint of each joint's range.

**Why:** DLS is the general-purpose workhorse IK solver. It handles arbitrary DOF robots and converges reliably with appropriate damping. The `IKSolution` struct reports convergence status, iteration count, and residual errors so you can decide whether to trust the result before commanding hardware.

## Step 5: Verify the Roundtrip

```rust
    let recovered_pose = forward_kinematics(&robot, &chain, &solution.joints)?;
    let rt = recovered_pose.translation();
    println!("Roundtrip FK → ({:.4}, {:.4}, {:.4})", rt.x, rt.y, rt.z);

    let pos_diff = (t - rt).norm();
    println!("Position roundtrip error: {:.2e} m", pos_diff);

    Ok(())
}
```

**What this does:** Runs FK on the IK solution to get a second pose, then computes the Euclidean distance between the original pose and the recovered pose.

**Why:** The roundtrip test is the gold standard for IK validation. A position error below 1e-4 meters (0.1 mm) confirms the solver converged to a valid solution. Always verify before sending commands to real hardware.

## Complete Code

```rust
use kinetic::prelude::*;

const PANDA_URDF: &str = r#"<?xml version="1.0"?>
<robot name="panda_like">
  <!-- 7 revolute joints from panda_link0 to panda_link8 -->
  <!-- (full URDF omitted for brevity — see examples/hello_fk_ik.rs) -->
</robot>
"#;

fn main() -> kinetic::core::Result<()> {
    // 1. Load robot
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    println!("Loaded '{}' — {} DOF", robot.name, robot.dof);

    // 2. Extract kinematic chain
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

    // 3. FK: joint angles → end-effector pose
    let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let pose = forward_kinematics(&robot, &chain, &q)?;
    let t = pose.translation();
    println!("FK → ({:.4}, {:.4}, {:.4})", t.x, t.y, t.z);

    // 4. IK: end-effector pose → joint angles
    let config = IKConfig::dls()
        .with_seed(robot.mid_configuration().to_vec())
        .with_max_iterations(300);
    let solution = solve_ik(&robot, &chain, &pose, &config)?;
    println!("IK → {} iters, err={:.2e}", solution.iterations, solution.position_error);

    // 5. Verify roundtrip
    let recovered = forward_kinematics(&robot, &chain, &solution.joints)?;
    let pos_diff = (t - recovered.translation()).norm();
    println!("Roundtrip error: {:.2e} m", pos_diff);

    Ok(())
}
```

## What You Learned
- `Robot::from_urdf_string()` parses a URDF into kinetic's internal representation
- `KinematicChain::extract()` isolates a base-to-tip joint chain
- `forward_kinematics()` computes end-effector pose in microseconds
- `IKConfig::dls()` configures the Damped Least Squares solver with seed and iteration limits
- `solve_ik()` returns an `IKSolution` with convergence info and residual errors
- FK/IK roundtrip error should be below 1e-4 m for a valid solution

## Try This
- Change the joint angles `q` and observe how the FK pose changes
- Try `IKConfig::fabrik()` instead of `IKConfig::dls()` and compare convergence speed
- Use `robot.mid_configuration()` vs `vec![0.0; 7]` as the seed and see how it affects iteration count
- Load a built-in robot with `Robot::from_name("ur5e")` instead of inline URDF

## Next
- [IK Solver Selection](ik-solver-selection.md) — choosing the right solver for your robot
- [Multiple IK Solutions](multiple-ik-solutions.md) — finding diverse solutions with random seeds
