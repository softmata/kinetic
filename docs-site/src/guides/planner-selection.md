# Planner Selection

How to pick the right planning algorithm for your problem.

## Decision Flowchart

Follow this text-based flowchart from top to bottom. Start with your
planning scenario and follow the arrows to the recommended planner.

```
START: What does your problem look like?
 |
 +-- Is the workspace mostly open (few obstacles)?
 |    YES --> Use RRT-Connect (fastest general-purpose planner)
 |
 +-- Are there narrow passages or tight spaces?
 |    YES --> Use EST or KPIECE (designed for constrained spaces)
 |
 +-- Do you need the shortest/optimal path?
 |    YES --> Do you have time (>100ms)?
 |             YES --> Use RRT* or BiRRT*
 |             NO  --> Use RRT-Connect + shortcutting (default)
 |
 +-- Do you want cost-aware planning (avoid joint limits, singularities)?
 |    YES --> Use BiTRRT (cost-guided transition test)
 |
 +-- Will you solve many queries in the same environment?
 |    YES --> Use PRM (build roadmap once, query many times)
 |
 +-- Do you need globally optimal trajectories?
 |    YES --> Use GCS (pre-compute convex decomposition, then solve)
 |
 +-- Do you need straight-line Cartesian motion?
 |    YES --> Use Cartesian planner (LERP position, SLERP orientation)
 |
 +-- Are you planning for two arms simultaneously?
      YES --> Use DualArmPlanner (combined C-space, inter-arm avoidance)
```

## Planner Comparison Table

| Planner | Speed | Optimality | Best For |
|---------|-------|------------|----------|
| **RRT-Connect** | Fast (<50ms) | Feasible | General pick-and-place, open spaces |
| **RRT\*** | Slow (100ms+) | Asymptotically optimal | Offline planning, quality matters |
| **BiRRT\*** | Medium (50-200ms) | Asymptotically optimal | Faster RRT* convergence |
| **BiTRRT** | Medium (50-200ms) | Cost-aware | Avoiding singularities, joint limits |
| **EST** | Medium | Feasible | Narrow passages, constrained spaces |
| **KPIECE** | Medium | Feasible | High-DOF, narrow passages |
| **PRM** | Build: slow / Query: fast | Feasible | Multi-query same environment |
| **GCS** | Build: slow / Query: fast | Globally optimal | Repeated queries, convex environments |
| **Cartesian** | Fast (<10ms) | N/A (follows path) | Linear/arc EE motion |
| **DualArmPlanner** | Slow (200ms+) | Feasible | Bimanual coordination |
| **ConstrainedRRT** | Medium-slow | Feasible | Orientation/position constraints |
| **CHOMP** | Medium | Locally optimal | Trajectory optimization |
| **STOMP** | Medium | Locally optimal | Stochastic optimization |
| **GpuOptimizer** | Fast (GPU) | Locally optimal | Parallel seed optimization |

## Setting the Planner Type

```rust
use kinetic::prelude::*;

let robot = Robot::from_name("ur5e")?;

// Default (Auto selects RRT-Connect)
let planner = Planner::new(&robot)?;

// Explicit planner selection
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::RRTStar);

// With custom config
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::BiTRRT)
    .with_config(PlannerConfig::offline());
```

```python
import kinetic

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
# Python uses the default (Auto/RRT-Connect) planner
```

## "My Plan Is Slow" Diagnosis

If planning is taking too long, work through these steps in order.

**1. Check if start or goal is in collision.**
The planner spends its entire time budget searching before returning
`StartInCollision` or `GoalInCollision`. Validate both before calling `plan()`.

```rust
if planner.is_in_collision(&start) {
    // Move the robot or adjust the start configuration
}
```

**2. Reduce the iteration count.**
Default is 10,000 iterations. For real-time use, try `PlannerConfig::realtime()`
which uses 2,000 iterations and a 10ms timeout.

**3. Disable post-processing.**
Shortcutting and smoothing add time after the path is found.
Disable them for latency-critical paths.

```rust
let config = PlannerConfig {
    shortcut_iterations: 0,
    smooth: false,
    ..PlannerConfig::default()
};
```

**4. Narrow the workspace bounds.**
Setting workspace bounds eliminates exploration of unreachable regions.

**5. Use GPU optimization for complex environments.**
When the environment has many obstacles, `GpuOptimizer` runs hundreds of
parallel seeds on the GPU and can find solutions faster than tree-based planners.

**6. Switch to PRM for repeated queries.**
If you plan many paths in the same environment, build a PRM roadmap once and
query it repeatedly. Amortized query time drops significantly.

**7. Profile with the benchmark suite.**
Run `cargo bench --bench planning_benchmarks` to compare planners on your
specific robot and environment.

## When to Use GPU Optimization

The `GpuOptimizer` is not a replacement for sampling-based planners.
Use it when:

- The environment has many obstacles (SDF is more efficient than sphere checks)
- You need smooth trajectories (optimizer minimizes jerk natively)
- You can warm-start from an RRT solution
- A Vulkan/Metal-capable GPU is available

Do not use GPU optimization when:

- The environment is simple (RRT-Connect will be faster)
- You need strict completeness guarantees
- No GPU is available (CPU fallback exists but is slower than RRT)

## Planner Combinations

Planners can be composed for better results:

1. **RRT-Connect + GPU warm-start**: find a feasible path fast, then
   optimize it on the GPU for smoothness.
2. **PRM + Cartesian**: use PRM for free-space transit, switch to
   Cartesian for approach/retreat motions.
3. **PlanExecuteLoop**: wraps planning with automatic replanning and
   fallback chains when execution deviates from the plan.
