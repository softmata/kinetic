# 05 — Planner Correctness Tests

**Priority**: P1 (P0 for collision-related planner tests — see 03_COLLISION_SAFETY.md)
**File**: `crates/kinetic/tests/acceptance/test_planner_correctness.rs`
**Estimated parameterized cases**: ~2,500

---

## PRINCIPLE

A motion planner must:
1. Return a valid path or a clear error — never garbage, never hang
2. Start at the requested start configuration
3. End at the requested goal (within tolerance)
4. Respect collision constraints when given a scene
5. Complete within timeout
6. Be deterministic with the same seed

---

## TEST 1: RRT-Connect Basic Planning

**Function**: `p1_rrt_connect_basic_all_robots`

For every robot × 5 random start/goal pairs (seed=100):
1. Both start and goal within joint limits (margin=0.1)
2. `planner.plan(&start, &Goal::Joints(JointValues::new(goal)))`
3. Verify:
   - Path has ≥2 waypoints
   - First waypoint matches start (within 1e-6)
   - Last waypoint matches goal (within IK tolerance for Pose goals, exact for Joint goals)
   - planning_time > 0
   - All waypoints have correct DOF

```rust
let result = planner.plan(&start, &Goal::Joints(JointValues::new(goal.clone()))).unwrap();

assert!(result.waypoints.len() >= 2, "{}: path too short", name);

// Start matches
for (j, (&planned, &actual)) in start.iter().zip(result.waypoints[0].iter()).enumerate() {
    assert!((planned - actual).abs() < 1e-6,
        "{}: start mismatch joint {}", name, j);
}

// Goal matches
let last = result.waypoints.last().unwrap();
for (j, (&planned, &actual)) in goal.iter().zip(last.iter()).enumerate() {
    assert!((planned - actual).abs() < 1e-3,
        "{}: goal mismatch joint {}: {} vs {}", name, j, planned, actual);
}
```

**Pass criteria**: ≥90% of plans succeed (some may timeout on difficult configurations).

---

## TEST 2: Cartesian Planning

**Function**: `p1_cartesian_planning`

For `REPRESENTATIVE_ROBOTS`:
1. Compute FK at start → start_pose
2. Create a target 10cm ahead in EE X direction: `target = start_pose * Pose::from_xyz(0.1, 0.0, 0.0)`
3. Plan Cartesian path
4. Verify:
   - All intermediate EE poses are on a straight line from start to target
   - Maximum deviation from the line < 1mm
   - Orientation stays constant (within 0.01 rad)

```rust
// Verify linearity
for (i, wp) in result.waypoints.iter().enumerate() {
    let pose = fk(&robot, &chain, wp).unwrap();
    let pos = pose.translation();

    // Project onto the line from start_pos to target_pos
    let line_dir = (target_pos - start_pos).normalize();
    let to_point = pos - start_pos;
    let projection = to_point.dot(&line_dir);
    let perpendicular = to_point - line_dir * projection;
    let deviation = perpendicular.norm();

    assert!(deviation < 0.001,
        "{}: Cartesian deviation {:.4}m at waypoint {} (max 1mm)", name, deviation, i);
}
```

---

## TEST 3: Pose Goal (IK + Planning)

**Function**: `p1_pose_goal_planning`

For `REPRESENTATIVE_ROBOTS`:
1. Target = arbitrary reachable Pose
2. `planner.plan(&start, &Goal::Pose(target))`
3. Verify:
   - FK at last waypoint is close to target (position < 2mm, orientation < 0.02 rad)
   - All waypoints within joint limits

---

## TEST 4: Named Goal

**Function**: `p1_named_goal_planning`

For robots with named poses (check `robot.named_poses`):
1. Plan from current config to `Goal::Named("home".to_string())`
2. Verify last waypoint matches the named pose configuration

---

## TEST 5: Timeout Behavior

**Function**: `p0_planner_timeout_respected`

For `REPRESENTATIVE_ROBOTS`:
1. Set `PlannerConfig { timeout: Duration::from_millis(10), .. }` (very short)
2. Plan a difficult problem (large joint-space distance)
3. Measure wall-clock time
4. Must complete within `2 × timeout` (20ms) — allows some overhead
5. Must return `Err(PlanningTimeout { .. })` or `Ok` (if it solved it fast)

```rust
let start = std::time::Instant::now();
let result = planner.plan_with_config(&start_joints, &goal, PlannerConfig {
    timeout: Duration::from_millis(10),
    ..PlannerConfig::default()
});
let elapsed = start.elapsed();

assert!(elapsed < Duration::from_millis(50),
    "{}: planner took {:?} despite 10ms timeout", name, elapsed);

match result {
    Ok(_) => {} // solved within timeout — great
    Err(KineticError::PlanningTimeout { .. }) => {} // correct timeout
    Err(other) => {} // other errors are OK (GoalUnreachable, etc.)
}
```

---

## TEST 6: Planner Determinism

**Function**: `p1_planner_determinism`

For `REPRESENTATIVE_ROBOTS`:
1. Plan the SAME problem 5 times with the SAME PlannerConfig (same RNG seed)
2. All 5 results must have identical waypoints

```rust
let mut results = Vec::new();
for _ in 0..5 {
    let result = planner.plan_with_config(&start, &goal, config.clone()).unwrap();
    results.push(result);
}

for i in 1..results.len() {
    assert_eq!(results[0].waypoints.len(), results[i].waypoints.len(),
        "{}: path lengths differ between runs", name);
    for (wi, (a, b)) in results[0].waypoints.iter().zip(results[i].waypoints.iter()).enumerate() {
        for (ji, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(va, vb, "{}: waypoint {} joint {} differs: {} vs {}",
                name, wi, ji, va, vb);
        }
    }
}
```

---

## TEST 7: All Planner Types Produce Valid Output

**Function**: `p1_all_planner_types_valid`

For "ur5e" (well-tested robot) × each `PlannerType`:
1. Plan the same simple problem (small joint-space motion)
2. For planners that succeed, verify:
   - Path starts at start, ends at goal
   - All waypoints have correct DOF
   - All waypoints within joint limits

```rust
let planner_types = [
    PlannerType::RRTConnect,
    PlannerType::RRTStar,
    PlannerType::BiRRTStar,
    PlannerType::BiTRRT,
    PlannerType::EST,
    PlannerType::KPIECE,
    PlannerType::PRM,
    // Skip CHOMP, STOMP (optimization-based, may need special setup)
    // Skip Cartesian (tested separately)
    // Skip GCS (tested separately)
];

for pt in &planner_types {
    let config = PlannerConfig {
        planner_type: pt.clone(),
        timeout: Duration::from_secs(5),
        ..PlannerConfig::default()
    };
    match planner.plan_with_config(&start, &goal, config) {
        Ok(result) => {
            // Validate output
            assert!(result.waypoints.len() >= 2, "{:?}: too few waypoints", pt);
            assert_eq!(result.waypoints[0].len(), dof, "{:?}: wrong DOF", pt);
            // ... full validation ...
        }
        Err(KineticError::PlanningTimeout { .. }) => {
            eprintln!("{:?}: timed out (acceptable)", pt);
        }
        Err(other) => {
            panic!("{:?}: unexpected error: {}", pt, other);
        }
    }
}
```

---

## TEST 8: RRT* Produces Better Paths Than RRT-Connect

**Function**: `p2_rrt_star_path_quality`

For "ur5e" × 10 problems × 5 runs each:
1. Plan with RRT-Connect (fast, suboptimal)
2. Plan with RRT* (slower, asymptotically optimal) with higher iteration count
3. Measure path length: `sum of |waypoint[i+1] - waypoint[i]|` in joint space
4. RRT* path length should be ≤ RRT-Connect path length in ≥60% of cases

This is a statistical test — RRT* should tend to find shorter paths.

---

## TEST 9: GCS Planning

**Function**: `p1_gcs_planning_basic`

For "ur5e":
1. Set up a simple obstacle in C-space
2. Build GCS planner with IRIS regions
3. Plan around the obstacle
4. Verify path is collision-free
5. Verify path starts at start and ends at goal

---

## TEST 10: Replanning When Goal Changes

**Function**: `p1_replan_goal_change`

For `REPRESENTATIVE_ROBOTS`:
1. Plan to goal_1 → get trajectory_1
2. Change goal to goal_2 (nearby)
3. Replan from current position along trajectory_1
4. New plan must be valid (collision-free, within limits)
5. New plan should start from the replanning point, not from the original start

---

## TEST 11: Start In Collision Error

**Function**: `p0_start_in_collision_error`

For `REPRESENTATIVE_ROBOTS`:
1. Place obstacle at the start config's EE position
2. Attempt to plan
3. Must return `Err(StartInCollision)` — NOT an `Ok` with garbage path

---

## TEST 12: Goal In Collision Error

**Function**: `p0_goal_in_collision_error`

Same as TEST 11 but obstacle at goal config's EE position.
Must return `Err(GoalInCollision)`.

---

## TEST 13: Path Shortcutting Preserves Validity

**Function**: `p1_shortcut_preserves_validity`

1. Plan a path (may be long/suboptimal)
2. Apply shortcutting/smoothing
3. Shortened path must still:
   - Start at start, end at goal
   - Be collision-free (if scene was provided)
   - Have all waypoints within limits
   - Be SHORTER (or equal) in joint-space length

---

## TEST 14: Planning With Constraints

**Function**: `p1_constrained_planning`

For "franka_panda" (7-DOF, has redundancy for constraints):
1. Add orientation constraint: `Constraint::Orientation { link: "ee", axis: Z, tolerance: 0.1 }`
   (keep the EE pointing up)
2. Plan with constraint
3. Verify every waypoint satisfies the constraint:
   - FK → check EE Z-axis direction is within tolerance of vertical

---

## TEST 15: Plan Function One-Liner API

**Function**: `p1_plan_one_liner_api`

Test the convenience `plan()` function (not the `Planner` struct):
```rust
let result = kinetic::planning::plan("ur5e", &start, &Goal::Joints(JointValues::new(goal)));
assert!(result.is_ok());
// Same validation as TEST 1
```

---

## SUMMARY

| Test | Robots | Plans | Priority |
|------|--------|-------|----------|
| RRT-Connect basic | 52 | 5 each = 260 | P1 |
| Cartesian | 5 | 5 | P1 |
| Pose goal | 5 | 5 | P1 |
| Named goal | ~10 | 1 each | P1 |
| Timeout | 5 | 5 | P0 |
| Determinism | 5 | 5×5 = 25 | P1 |
| All planner types | 1 | ~8 types | P1 |
| RRT* quality | 1 | 10×5 = 50 | P2 |
| GCS | 1 | 5 | P1 |
| Replan | 5 | 5 | P1 |
| Start in collision | 5 | 1 | P0 |
| Goal in collision | 5 | 1 | P0 |
| Shortcut validity | 5 | 5 | P1 |
| Constrained | 1 | 5 | P1 |
| One-liner API | 1 | 1 | P1 |
