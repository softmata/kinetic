# 09 — Multi-Robot & Dual-Arm Tests

**Priority**: P1 — dual-arm coordination bugs cause arm-arm collisions
**File**: `crates/kinetic/tests/acceptance/test_multi_robot.rs`
**Estimated cases**: ~75

---

## PRINCIPLE

Multi-robot and dual-arm planning must:
1. Avoid inter-robot collisions (arm A doesn't hit arm B)
2. Maintain independent joint limit enforcement per arm
3. Handle shared workspace coordination
4. Not corrupt state between concurrent planners

---

## TEST 1: Dual-Arm Planning Basic

**Function**: `p1_dual_arm_planning_basic`

For each pair in `BIMANUAL_PAIRS` (ALOHA, Baxter, YuMi):
1. Load both robots
2. Plan for each arm independently (non-conflicting goals)
3. Both plans must succeed
4. Both plans must respect their own joint limits

---

## TEST 2: Dual-Arm Collision Avoidance

**Function**: `p1_dual_arm_collision_avoidance`

For `BIMANUAL_PAIRS`:
1. Plan left arm to reach into right arm's workspace
2. Plan right arm to reach into left arm's workspace
3. If using `DualArmPlanner`:
   - Plans must avoid inter-arm collision
   - If collision-free plan exists, it should find it
   - If no collision-free plan exists, return appropriate error
4. If planning independently:
   - Check all left waypoints against right arm's geometry at each time step
   - At least detect the collision (even if planning doesn't coordinate)

---

## TEST 3: Coordinated Dual-Arm Motion

**Function**: `p1_coordinated_dual_arm`

If `DualArmPlanner` is available:
1. Set up a task requiring both arms (e.g., both reach to the same area)
2. Plan coordinated motion
3. Verify:
   - Both arms arrive at goals
   - No inter-arm collision at any time step
   - Timing is coordinated (both finish within same time window)

---

## TEST 4: Independent Planning Isolation

**Function**: `p1_independent_planning_isolation`

1. Create two `Planner` instances for different robots
2. Plan simultaneously in two threads
3. Both must succeed independently
4. No shared state corruption (verify via determinism — same result as single-threaded)

```rust
let robot_a = load_robot("ur5e");
let robot_b = load_robot("franka_panda");

let handle_a = std::thread::spawn(move || {
    let planner = Planner::new(&robot_a).unwrap();
    planner.plan(&start_a, &goal_a).unwrap()
});
let handle_b = std::thread::spawn(move || {
    let planner = Planner::new(&robot_b).unwrap();
    planner.plan(&start_b, &goal_b).unwrap()
});

let result_a = handle_a.join().unwrap();
let result_b = handle_b.join().unwrap();

// Verify both are valid
assert!(result_a.waypoints.len() >= 2);
assert!(result_b.waypoints.len() >= 2);
```

---

## TEST 5: Multiple Planners Same Robot

**Function**: `p1_multiple_planners_same_robot`

1. Create two `Planner` instances for the SAME robot (shared `Arc<Robot>`)
2. Plan different goals simultaneously
3. Both must succeed
4. Results must be independent (different goals → different paths)

---

## TEST 6: Servo Independence

**Function**: `p1_servo_independence`

If Servo supports multi-robot:
1. Create two Servo controllers for left/right arms
2. Send twist commands to both simultaneously
3. Each servo must track its own arm independently
4. No state leakage between servos

---

## TEST 7: Concurrent Scene Access

**Function**: `p1_concurrent_scene_access`

1. Create a shared `Scene`
2. Multiple threads plan with the same scene
3. No data races (Scene must be thread-safe or cloned per thread)
4. All plans are collision-free

---

## TEST 8: Resource Contention Under Load

**Function**: `p2_resource_contention_stress`

1. Launch 10 concurrent planners for different robots
2. All with 1-second timeout
3. All must either succeed or timeout gracefully
4. No panics, no deadlocks, no corrupted results

```rust
let handles: Vec<_> = (0..10).map(|i| {
    let name = REPRESENTATIVE_ROBOTS[i % REPRESENTATIVE_ROBOTS.len()];
    let robot = load_robot(name);
    std::thread::spawn(move || {
        let planner = Planner::new(&robot).unwrap();
        let start = vec![0.0; robot.dof];
        let goal = vec![0.5; robot.dof];
        let config = PlannerConfig {
            timeout: Duration::from_secs(1),
            ..PlannerConfig::default()
        };
        planner.plan_with_config(&start, &Goal::Joints(JointValues::new(goal)), config)
    })
}).collect();

let mut panicked = 0;
for (i, h) in handles.into_iter().enumerate() {
    match h.join() {
        Ok(Ok(_)) => {}
        Ok(Err(KineticError::PlanningTimeout { .. })) => {}
        Ok(Err(e)) => eprintln!("planner {} error: {}", i, e),
        Err(_) => { panicked += 1; }
    }
}
assert_eq!(panicked, 0, "{} threads panicked", panicked);
```

---

## SUMMARY

| Test | Pairs/Threads | Priority |
|------|---------------|----------|
| Dual-arm basic | 3 pairs | P1 |
| Dual-arm collision | 3 pairs | P1 |
| Coordinated motion | 3 pairs | P1 |
| Independent isolation | 2 threads | P1 |
| Same robot multi-planner | 2 threads | P1 |
| Servo independence | 2 servos | P1 |
| Concurrent scene | 4 threads | P1 |
| Stress contention | 10 threads | P2 |
