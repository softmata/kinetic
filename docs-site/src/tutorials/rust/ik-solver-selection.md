# IK Solver Selection

## What You'll Learn
- Understand when to use each of kinetic's five IK solvers
- Apply the decision flowchart based on robot DOF and wrist geometry
- Configure OPW, Subproblem, Subproblem7DOF, FABRIK, and DLS solvers
- Use `IKSolver::Auto` for automatic solver selection

## Prerequisites
- [Inverse Kinematics](../../concepts/inverse-kinematics.md)
- [FK and IK Tutorial](fk-and-ik.md)

## Overview

Kinetic ships five IK solvers, each optimized for different robot geometries.
Choosing the right solver can mean the difference between a 2-microsecond
analytical solution and a 200-microsecond iterative one. This tutorial explains
the decision flowchart and shows code for each solver. In most cases,
`IKSolver::Auto` makes the right choice, but understanding the options helps
when you need to override.

## Decision Flowchart

```
Is it a 6-DOF robot with a spherical wrist?
├── YES: Does it match OPW kinematic parameters?
│   ├── YES → OPW (closed-form, ~2 µs, up to 8 solutions)
│   └── NO  → Subproblem (analytical, ~10 µs, up to 16 solutions)
└── NO:
    Is it a 7-DOF robot with intersecting wrist axes?
    ├── YES → Subproblem7DOF (sweep + analytical, ~50 µs)
    └── NO:
        Do you only need position (not orientation)?
        ├── YES → FABRIK (fast, position-focused)
        └── NO  → DLS (general-purpose, any DOF)
```

`IKSolver::Auto` implements this exact logic. Override it only when you know the
geometry better than the auto-detector.

## Step 1: OPW — 6-DOF Spherical Wrist Robots

```rust
use kinetic::prelude::*;

// OPW works for UR5e, ABB IRB, KUKA KR, Fanuc — standard industrial arms
let robot = Robot::from_name("ur5e")?;
let chain = KinematicChain::from_robot(&robot)?;

let config = IKConfig::opw();
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
// Closed-form: converges in 1 iteration, ~2 µs
```

**What this does:** Uses the Ortho-Parallel-Wrist decomposition to compute a closed-form solution. No iterations needed.

**Why:** OPW is the fastest solver in kinetic. It exploits the fact that most 6-DOF industrial arms have three intersecting wrist axes, which decouples the position and orientation sub-problems. Use it for any robot whose wrist axes meet at a single point (UR, ABB, KUKA, Fanuc, etc.).

## Step 2: Subproblem — General 6-DOF Robots

```rust
let config = IKConfig {
    solver: IKSolver::Subproblem,
    ..Default::default()
};
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
// Analytical: returns up to 16 solutions
```

**What this does:** Decomposes the IK problem into Paden-Kahan subproblems (rotations about known axes). Returns all analytical solutions, ranked by proximity to the seed.

**Why:** Use Subproblem when your 6-DOF robot has intersecting wrist axes but doesn't match OPW's stricter kinematic parameter constraints. It is still analytical (no iterative convergence) and returns multiple solutions.

## Step 3: Subproblem7DOF — 7-DOF Redundant Robots

```rust
let config = IKConfig {
    solver: IKSolver::Subproblem7DOF { num_samples: 36 },
    ..Default::default()
};
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
// Sweeps 36 values of the redundant joint, solves analytically at each
```

**What this does:** Sweeps the redundant (7th) joint across its range in `num_samples` steps. At each sample, it locks that joint and solves the remaining 6-DOF system analytically via subproblem decomposition.

**Why:** A 7-DOF robot like the Franka Panda has infinite IK solutions for most targets (one extra degree of freedom). This solver discretizes the redundancy and finds the best analytical solution at each sample. Increase `num_samples` for finer resolution at the cost of compute time.

## Step 4: FABRIK — Position-Only IK

```rust
let config = IKConfig::fabrik()
    .with_mode(IKMode::PositionOnly)
    .with_max_iterations(100);
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
// Fast convergence for position, ignores orientation
```

**What this does:** Forward And Backward Reaching IK iterates by reaching each joint toward the target (forward pass) then pulling the chain back to the base (backward pass). Naturally converges to position targets.

**Why:** FABRIK is geometrically intuitive and fast for position-only problems (e.g., moving a tooltip to a point without caring about wrist orientation). It struggles with full 6D pose constraints because orientation is not part of its core algorithm.

## Step 5: DLS — General-Purpose Solver

```rust
let config = IKConfig::dls()
    .with_seed(vec![0.0; robot.dof])
    .with_max_iterations(300);
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
// Iterative: works for any DOF, any geometry
```

**What this does:** Damped Least Squares computes the Jacobian pseudo-inverse with a damping term, stepping toward the target each iteration. The damping factor prevents instability near singularities.

**Why:** DLS is the fallback solver that works for any robot — 3-DOF, 6-DOF, 7-DOF, or more. It handles full 6D pose targets on arbitrary kinematic structures. The trade-off is speed (iterative, ~100-500 µs) and the risk of local minima. Use `with_restarts(5)` to improve convergence on difficult targets.

## Step 6: Auto Selection

```rust
// Let kinetic choose the best solver
let config = IKConfig::default(); // solver: IKSolver::Auto
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
```

**What this does:** `IKSolver::Auto` checks the robot's DOF, wrist geometry, and optional `ik_preference` field in the robot config. It selects OPW > Subproblem > Subproblem7DOF > DLS in order of preference.

**Why:** Auto selection is the recommended default. It picks the fastest applicable solver and falls back to DLS when no analytical solver matches. Override it only when profiling reveals a better choice for your specific use case.

## Complete Code

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_name("ur5e")?;
    let chain = KinematicChain::from_robot(&robot)?;
    let target = forward_kinematics(&robot, &chain, &[0.5, -0.8, 0.5, 0.1, -0.1, 0.3])?;

    // Compare solvers
    for (name, config) in [
        ("Auto",       IKConfig::default()),
        ("OPW",        IKConfig::opw()),
        ("DLS",        IKConfig::dls().with_max_iterations(300)),
        ("FABRIK",     IKConfig::fabrik().with_max_iterations(200)),
    ] {
        let start = std::time::Instant::now();
        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) => println!(
                "{:10}: {} iters, err={:.2e}, {:?}",
                name, sol.iterations, sol.position_error, start.elapsed()
            ),
            Err(e) => println!("{:10}: failed — {}", name, e),
        }
    }

    Ok(())
}
```

## What You Learned
- OPW is fastest for 6-DOF spherical-wrist robots (closed-form, ~2 us)
- Subproblem handles general 6-DOF with intersecting wrist axes (up to 16 solutions)
- Subproblem7DOF sweeps the redundant joint for 7-DOF robots
- FABRIK excels at position-only targets but struggles with full 6D pose
- DLS is the reliable general-purpose fallback for any robot
- `IKSolver::Auto` selects the best solver automatically

## Try This
- Load a Franka Panda (`Robot::from_name("franka_panda")`) and compare Auto vs DLS timing
- Benchmark OPW vs DLS on a UR5e over 10,000 solves to measure the speed difference
- Try `IKConfig::with_fallback()` which attempts Full6D then falls back to PositionOnly
- Set `IKConfig { solver: IKSolver::DLS { damping: 0.001 }, .. }` vs `damping: 0.1` and observe convergence behavior near singularities

## Next
- [Multiple IK Solutions](multiple-ik-solutions.md) — finding diverse configurations
- [Servo Control](servo-control.md) — real-time IK in a control loop
