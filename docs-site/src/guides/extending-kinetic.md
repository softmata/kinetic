# Extending Kinetic

Adding custom planners, IK solvers, robot configurations, tests, and benchmarks.

## Adding a Custom Planner

### Step 1: Implement the PlannerPlugin Trait

The `PlannerPlugin` trait is the extension point for the planning pipeline.

```rust
use kinetic_planning::pipeline::{PlannerPlugin, PlanningRequest};
use kinetic_core::Trajectory;

pub struct MyPlanner {
    // Your planner state
}

impl PlannerPlugin for MyPlanner {
    fn plan(&self, request: &PlanningRequest) -> Option<Trajectory> {
        // Implement your planning algorithm
        // Return None if no solution found within timeout
        let mut traj = Trajectory::with_dof(request.start.len());
        traj.push_waypoint(&request.start);
        // ... add waypoints ...
        Some(traj)
    }

    fn id(&self) -> &str {
        "my_planner"
    }

    fn description(&self) -> &str {
        "My custom planner: does something special"
    }
}
```

### Step 2: Register in the Pipeline

```rust
use kinetic_planning::pipeline::PlanningPipeline;

let mut pipeline = PlanningPipeline::new();
pipeline.add_planner(Box::new(MyPlanner::new()));
pipeline.default_planner = "my_planner".into();
```

### Step 3: Use via the Facade (Optional)

To make your planner available through the `Planner` facade, add a
variant to `PlannerType` and handle it in the dispatch match:

```rust
// In kinetic-planning/src/facade.rs
pub enum PlannerType {
    // ... existing variants ...
    MyPlanner,
}

// In Planner::plan_with_config, add the dispatch:
PlannerType::MyPlanner => {
    let planner = MyPlanner::new(/* params */);
    let r = planner.plan(&chain_start, &goal_resolved)?;
    (r.waypoints, r.planning_time, r.iterations, r.tree_size)
}
```

## Adding a Custom IK Solver

### Step 1: Implement the Solver Function

Follow the pattern in `kinetic-kinematics/src/dls.rs`:

```rust
use kinetic_core::{Pose, Result};
use kinetic_kinematics::{KinematicChain, IKConfig, IKSolution, IKMode};
use kinetic_robot::Robot;

pub fn solve_my_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &IKConfig,
    mode: IKMode,
) -> Result<IKSolution> {
    // Your IK algorithm here
    // Must return IKSolution with:
    //   - joints: Vec<f64>
    //   - position_error: f64
    //   - orientation_error: f64
    //   - converged: bool
    //   - iterations: usize
    //   - mode_used: IKMode
    //   - degraded: bool
    //   - condition_number: f64
    todo!()
}
```

### Step 2: Register in the Solver Dispatch

Add a variant to `IKSolver` in `kinetic-kinematics/src/ik.rs` and
handle it in `solve_once`:

```rust
pub enum IKSolver {
    // ... existing variants ...
    MySolver { param: f64 },
}

// In solve_once:
IKSolver::MySolver { param } => {
    my_solver::solve_my_ik(robot, chain, target, seed, config, mode)
}
```

## Adding a Robot Configuration

1. Create `robot_configs/<name>/` with `kinetic.toml` and the URDF file
2. Follow the structure from an existing config (see `robot_configs/ur5e/`)
3. Add a test to verify FK/IK:

```rust
#[test]
fn my_robot_fk_ik_roundtrip() {
    let robot = Robot::from_name("my_robot").unwrap();
    let planner = Planner::new(&robot).unwrap();

    let home = robot.named_pose("home").unwrap();
    let pose = planner.fk(&home).unwrap();
    let ik_joints = planner.ik(&pose).unwrap();

    let recovered = planner.fk(&ik_joints).unwrap();
    let err = (pose.translation() - recovered.translation()).norm();
    assert!(err < 0.001, "FK/IK roundtrip error: {err}");
}
```

## Adding Tests

### Acceptance Test Template

Every new planner or solver should have an acceptance test that verifies
correctness on a real robot model.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_core::Goal;
    use kinetic_robot::Robot;

    #[test]
    fn planner_finds_path_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap()
            .with_planner_type(PlannerType::MyPlanner);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.num_waypoints() >= 2);

        // Verify start and end match
        let first = result.start().unwrap();
        let last = result.end().unwrap();
        for i in 0..6 {
            assert!((first[i] - start[i]).abs() < 1e-6);
        }
    }

}
```

Run tests with `cargo test` (or `cargo test -p kinetic-planning` for a
specific crate). Use `xvfb-run cargo test` for visual features.

## Adding Benchmarks

Use Criterion for micro-benchmarks. Add to `benches/` in the relevant crate.
Run with `cargo bench --bench planning_benchmarks`.

## Code Conventions

- All `pub` items must have `///` doc comments
- Use `kinetic_core::Result<T>` for fallible functions, `thiserror` for errors
- Run `cargo fmt` and `cargo clippy -D warnings` before submitting
- Target 85% test coverage for new code
