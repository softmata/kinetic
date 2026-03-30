# Planner Selection Guide

KINETIC provides 9 planners. This guide helps you pick the right one.

## Quick Reference

| Planner | Type | Optimal? | Multi-query? | Best for |
|---------|------|----------|--------------|----------|
| **RRT-Connect** | Sampling | No | No | General purpose, fast first solution |
| **RRT\*** | Sampling | Yes (asymptotic) | No | High-quality paths, open spaces |
| **BiRRT\*** | Sampling | Yes (asymptotic) | No | Faster convergence than RRT* |
| **BiTRRT** | Sampling | Cost-aware | No | Cost-sensitive planning without full RRT* overhead |
| **EST** | Sampling | No | No | Narrow passages, undersampled regions |
| **KPIECE** | Sampling | No | No | High-dimensional spaces (7+ DOF) |
| **PRM/PRM\*** | Roadmap | PRM*: yes | Yes | Many queries in same environment |
| **Cartesian** | Interpolation | N/A | No | Straight-line task-space motions |
| **GCS** | Optimization | Yes (global) | No | Pre-computed convex decompositions |

## Decision Flowchart

```
                    Start Here
                        |
            +-----------+-----------+
            |                       |
    Multiple queries         Single query
    same environment?        (one-shot)?
            |                       |
            v                       |
       Use PRM/PRM*                |
       (build once,            +---+---+
        query many)            |       |
                        Need optimal   Just need
                        path quality?  a path?
                               |       |
                          +----+       |
                          |            |
                     Yes: Use      +---+---+
                     RRT* or       |       |
                     BiRRT*    Narrow    Open/general
                               passages?  environment?
                               |          |
                          Use EST      +--+--+
                          or KPIECE    |     |
                                    Cost   No cost
                                    aware?  preference?
                                    |       |
                                Use       Use
                                BiTRRT    RRT-Connect
                                          (default)
```

## Planner Details

### RRT-Connect (default)

**Algorithm**: Bidirectional rapidly-exploring random tree with greedy connect heuristic.

**Strengths**:
- Fast first solution (sub-second for most problems)
- Works for any robot, any DOF
- Simple, predictable behavior

**Weaknesses**:
- Path quality varies (random exploration)
- No optimality guarantee
- Can struggle with narrow passages

**Config**: `PlannerType::RRTConnect`

**When to use**: Default choice. Start here, switch if path quality or narrow passages are issues.

---

### RRT* (Asymptotically Optimal)

**Algorithm**: Single-tree RRT with cost tracking and rewiring. Continues improving after first solution.

**Strengths**:
- Asymptotically optimal (path quality improves with time)
- Anytime behavior: returns best-so-far on timeout
- Informed sampling after first solution (ellipsoidal focus)

**Weaknesses**:
- Slower than RRT-Connect for first solution
- Rewiring overhead grows with tree size
- Single-tree: slower to find initial path

**Config**: `PlannerType::RRTStar`, configure via `RRTStarConfig { gamma, anytime, informed }`

**When to use**: When path quality matters more than planning speed. Good for pre-computed trajectories where you can afford planning time.

---

### BiRRT* (Bidirectional RRT*)

**Algorithm**: Two-tree RRT* with rewiring in both trees.

**Strengths**:
- Faster initial solution than single-tree RRT*
- Still asymptotically optimal
- Better convergence rate

**Weaknesses**:
- Higher memory than RRT-Connect (two trees + cost tracking)
- More complex than basic RRT*

**Config**: `PlannerType::BiRRTStar`

**When to use**: When you want RRT* quality with faster convergence.

---

### BiTRRT (Transition-based RRT)

**Algorithm**: Bidirectional RRT with Boltzmann transition test. Accepts cost-increasing moves with probability `exp(-delta_cost / T)`, temperature anneals down.

**Strengths**:
- Cost-aware without full RRT* overhead
- Flexible: plug in any cost function via `CostFn` trait
- Frustration mechanism escapes local minima
- Better paths than RRT-Connect, faster than RRT*

**Weaknesses**:
- Not asymptotically optimal
- Requires tuning temperature schedule for best results
- Cost function design matters

**Config**: `PlannerType::BiTRRT`, configure via `BiTRRTConfig { initial_temperature, alpha }`

**When to use**: When you have a meaningful cost function (e.g., distance from obstacles, joint-limit avoidance, manipulability) and want cost-aware paths without the full RRT* machinery.

---

### EST (Expansive Space Trees)

**Algorithm**: Density-biased node selection. Nodes in sparse regions get higher selection probability: `weight = 1 / (1 + neighbor_count)`.

**Strengths**:
- Excellent for narrow passages (sparse regions get exploration priority)
- Natural frontier detection without explicit grid
- Adaptive: automatically focuses on unexplored areas

**Weaknesses**:
- O(n^2) density computation (mitigated by periodic refresh)
- No optimality guarantee
- May over-explore in open spaces

**Config**: `PlannerType::EST`, configure via `ESTConfig { density_radius, expansion_range }`

**When to use**: Environments with narrow passages, tight corridors, or areas that RRT-Connect struggles with.

---

### KPIECE (Interior-Exterior Cell Exploration)

**Algorithm**: Projects C-space onto a low-dimensional grid. Tracks interior (well-explored) vs exterior (frontier) cells. Expansion biased toward frontier.

**Strengths**:
- Efficient in high-dimensional spaces (projects to 2-3D grid)
- Systematic frontier tracking
- Cell-based importance prevents redundant exploration

**Weaknesses**:
- Grid resolution is a tuning parameter
- Projection loses information (may miss some paths)
- Not optimal

**Config**: `PlannerType::KPIECE`, configure via `KPIECEConfig { cell_size, projection_dims }`

**When to use**: High-DOF robots (7+ joints) where other planners waste time exploring already-covered regions.

---

### PRM / PRM* (Probabilistic Roadmap)

**Algorithm**: Two-phase: (1) Build roadmap of collision-free configs and edges. (2) Query via A* search.

**PRM**: Fixed K-nearest connections. Fast construction, may miss optimal paths.

**PRM\***: Adaptive K = `ceil(e * (1 + 1/d) * ln(n))`. Asymptotically optimal connectivity.

**Strengths**:
- Multi-query: build once, query many times instantly
- Roadmap persistence: save/load to disk
- Incremental growth: add samples to improve coverage
- Lazy mode: defer collision checks to query time

**Weaknesses**:
- Upfront construction cost
- Not suitable for dynamic environments (need revalidation)
- Memory scales with roadmap size

**Config**: Use `PRM` struct directly (not via Planner facade — requires `build_roadmap()`).

**When to use**: Many planning queries in the same static environment (e.g., production line, fixed workcell). Build roadmap once at startup, query per pick/place cycle.

---

### Cartesian Planner

**Algorithm**: Interpolate in task-space (position LERP + orientation SLERP), solve IK at each step.

**Config**: Via `CartesianPlanner` or `Planner::plan` with `Goal::Pose`.

**When to use**: Straight-line end-effector motions (welding seams, painting, assembly approach).

---

### GCS (Graph of Convex Sets)

**Algorithm**: Globally optimal planning over pre-computed convex decomposition of free space.

**Config**: Via `GCSPlanner`.

**When to use**: When you can pre-compute a convex decomposition (e.g., known shelf layout). Produces globally optimal paths within the decomposition.

---

## Performance Characteristics

| Planner | First Solution | Path Quality | Memory | Best DOF Range |
|---------|---------------|-------------|--------|----------------|
| RRT-Connect | Fast | Variable | Low | 3-12 |
| RRT* | Moderate | High (improves) | Medium | 3-7 |
| BiRRT* | Moderate-Fast | High (improves) | Medium | 3-7 |
| BiTRRT | Fast | Good (cost-aware) | Low | 3-12 |
| EST | Moderate | Variable | Medium | 3-12 |
| KPIECE | Moderate | Variable | Low | 6-20+ |
| PRM (query) | Instant | Depends on roadmap | High (roadmap) | 3-12 |
| Cartesian | Instant | Optimal (straight) | Minimal | Any |
| GCS | Instant | Optimal (regions) | Medium | 3-7 |

## Usage Examples

```rust
use kinetic::prelude::*;

// Default (RRT-Connect)
let planner = Planner::new(&robot)?;
let result = planner.plan(&start, &goal)?;

// Optimal paths
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::RRTStar);

// Cost-aware
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::BiTRRT);

// Narrow passages
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::EST);

// High-DOF
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::KPIECE);

// Multi-query (use PRM directly)
let mut prm = PRM::new(robot, chain, env, config, PRMConfig::prm_star(1000));
prm.build_roadmap();
let r1 = prm.query(&start1, &goal1)?;
let r2 = prm.query(&start2, &goal2)?;
```
