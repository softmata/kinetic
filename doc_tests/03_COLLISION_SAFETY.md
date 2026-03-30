# 03 — Collision Safety Tests

**Priority**: P0 — collision with environment or self can destroy sensors, damage robot, injure humans
**File**: `crates/kinetic/tests/acceptance/test_collision_safety.rs`
**Estimated parameterized cases**: ~63,000

---

## PRINCIPLE

Collision checking has two failure modes:
1. **False negative** (misses a collision) — CATASTROPHIC. Robot hits something.
2. **False positive** (reports collision when none exists) — annoying but safe.

We test heavily for false negatives. False positive rate should be <5% but a false positive is NEVER a test failure — only false negatives fail.

---

## TEST 1: Self-Collision at Known Collision Configs

**Function**: `p0_self_collision_known_configs`

For `ROBOTS_WITH_COLLISION`:
1. Create known self-collision configurations. These are configs where links physically overlap.
2. Check self-collision — MUST return `true`

**Known self-collision configs** (implement per robot):
```
UR5e: fold arm back on itself
  joints = [0.0, -0.1, 2.5, -2.5, 0.0, 0.0]  // elbow hits base

Franka Panda: wrist folded into forearm
  joints = [0.0, 0.5, 0.0, -0.3, 0.0, 3.0, 0.0]  // link7 hits link4

KUKA IIWA7: similar fold-back
  joints = [0.0, 1.5, 0.0, -0.3, 0.0, 2.5, 0.0]
```

The implementing Claude should:
1. For each robot in `ROBOTS_WITH_COLLISION`, find configs where links physically overlap
2. Use `RobotSphereModel` to verify the collision geometry reports collision
3. If no obvious collision config exists for a robot, skip it (but document why)

**Strategy to find collision configs**:
```rust
// Brute force: sample 10,000 random configs, find ones where self-collision is true
let model = RobotSphereModel::from_robot_default(&robot);
let mut rng = seeded_rng(42);
let mut collision_configs = Vec::new();
for _ in 0..10_000 {
    let joints = random_joints_within_limits(&robot, &chain, &mut rng, 0.0);
    if model.check_self_collision(&robot, &joints) {
        collision_configs.push(joints);
    }
}
// Store at least 5 per robot for regression testing
```

**Pass criteria**: All known collision configs are detected. Zero false negatives.

---

## TEST 2: Self-Collision at Home/Zero is False

**Function**: `p0_no_self_collision_at_home`

For every robot:
1. Home position: `robot.home_position()` or zeros
2. Check self-collision — MUST return `false`
3. A robot that self-collides at home is mis-configured

**Pass criteria**: 0 robots report self-collision at home.

---

## TEST 3: Environment Collision — Box Obstacle

**Function**: `p0_env_collision_box_obstacle`

For `REPRESENTATIVE_ROBOTS`:
1. Place a large box obstacle at the robot's end-effector position at mid-config
2. The EE is INSIDE the box → collision must be detected
3. Move the box 10m away → collision must NOT be detected

```rust
let (robot, chain) = load_robot_and_chain(name);
let mid = mid_joints(&robot, &chain);
let ee_pose = fk(&robot, &chain, &mid).unwrap();
let ee_pos = ee_pose.translation();

// Box at EE position — MUST collide
let mut scene = Scene::new(&robot).unwrap();
scene.add("box", Shape::box_shape(0.5, 0.5, 0.5),
    Isometry3::translation(ee_pos.x, ee_pos.y, ee_pos.z));
assert!(scene.check_collision(&mid).unwrap(),
    "{}: EE inside box but no collision detected", name);

// Box 10m away — MUST NOT collide
let mut scene2 = Scene::new(&robot).unwrap();
scene2.add("far_box", Shape::box_shape(0.5, 0.5, 0.5),
    Isometry3::translation(10.0, 10.0, 10.0));
assert!(!scene2.check_collision(&mid).unwrap(),
    "{}: false positive with box 10m away", name);
```

**Pass criteria**: Collision detected when EE is inside obstacle. No false positive when far away.

---

## TEST 4: Environment Collision — Sphere Obstacle

**Function**: `p0_env_collision_sphere_obstacle`

Same pattern as TEST 3 but with `Shape::sphere(0.3)`. Tests a different collision primitive.

---

## TEST 5: Environment Collision — Cylinder Obstacle

**Function**: `p0_env_collision_cylinder_obstacle`

Same pattern with `Shape::cylinder(0.2, 0.5)`.

---

## TEST 6: Collision Along Trajectory

**Function**: `p0_trajectory_collision_check`

For `REPRESENTATIVE_ROBOTS`:
1. Plan a valid collision-free trajectory (start → goal, no obstacles)
2. Add an obstacle AFTER planning, positioned to intersect the planned path
3. Check collision at every waypoint
4. At least one waypoint MUST report collision (the obstacle intersects the path)

This tests that post-hoc collision checking works — critical for dynamic environments.

```rust
// Plan without obstacles
let result = planner.plan(&start, &goal).unwrap();

// Add obstacle that intersects the path
// Strategy: take the midpoint waypoint, compute its EE position, place obstacle there
let mid_waypoint = &result.waypoints[result.waypoints.len() / 2];
let mid_pose = fk(&robot, &chain, mid_waypoint).unwrap();
let mid_pos = mid_pose.translation();

let mut scene = Scene::new(&robot).unwrap();
scene.add("blocker", Shape::sphere(0.3),
    Isometry3::translation(mid_pos.x, mid_pos.y, mid_pos.z));

// At least one waypoint must be in collision
let any_collision = result.waypoints.iter()
    .any(|wp| scene.check_collision(wp).unwrap_or(false));
assert!(any_collision, "{}: obstacle at path midpoint not detected", name);
```

---

## TEST 7: Planning Avoids Obstacles

**Function**: `p0_planner_avoids_obstacles`

For `REPRESENTATIVE_ROBOTS`:
1. Set up a scene with a sphere obstacle at a known position
2. Plan with the scene active
3. Check EVERY waypoint of the result against the scene
4. ZERO waypoints should be in collision

```rust
let mut scene = Scene::new(&robot).unwrap();
scene.add("obstacle", Shape::sphere(0.3),
    Isometry3::translation(0.3, 0.0, 0.5));

let result = plan_with_scene(name, &start, &goal, &scene).unwrap();

for (i, wp) in result.waypoints.iter().enumerate() {
    assert!(!scene.check_collision(wp).unwrap(),
        "{}: planner output waypoint {} in collision", name, i);
}
```

**Pass criteria**: Planner NEVER outputs a collision path when given a scene.

---

## TEST 8: Planning Reports Failure for Blocked Paths

**Function**: `p0_planner_fails_for_blocked_goal`

For `REPRESENTATIVE_ROBOTS`:
1. Place a very large obstacle that completely encases the goal position
2. Attempt to plan
3. Must return `Err(GoalInCollision)` or `Err(PlanningFailed)` or `Err(PlanningTimeout)`
4. Must NOT return `Ok` with a collision path
5. Must NOT hang — must complete within timeout

```rust
// Giant obstacle at goal
let goal_joints = random_joints_within_limits(&robot, &chain, &mut rng, 0.05);
let goal_pose = fk(&robot, &chain, &goal_joints).unwrap();
let goal_pos = goal_pose.translation();

let mut scene = Scene::new(&robot).unwrap();
scene.add("wall", Shape::box_shape(5.0, 5.0, 5.0),
    Isometry3::translation(goal_pos.x, goal_pos.y, goal_pos.z));

let config = PlannerConfig {
    timeout: Duration::from_secs(2),
    ..PlannerConfig::default()
};
let result = planner.plan_with_config(&start, &Goal::Joints(JointValues::new(goal_joints)), config);

match result {
    Err(KineticError::GoalInCollision) => {} // correct
    Err(KineticError::PlanningTimeout { .. }) => {} // acceptable
    Err(KineticError::PlanningFailed(_)) => {} // acceptable
    Ok(r) => {
        // If Ok, every waypoint must be collision-free (shouldn't happen but verify)
        for wp in &r.waypoints {
            assert!(!scene.check_collision(wp).unwrap(),
                "{}: planner returned Ok but path is in collision", name);
        }
    }
    Err(other) => panic!("{}: unexpected error: {}", name, other),
}
```

---

## TEST 9: Continuous Collision Detection (CCD)

**Function**: `p0_ccd_catches_fast_sweeps`

For `REPRESENTATIVE_ROBOTS`:
1. Create two configs where the arm passes through an obstacle during interpolation
   (start is on one side, goal on the other, obstacle in between)
2. Discrete check at start and goal: both NOT in collision
3. CCD between start and goal: MUST detect collision

This is the critical test — discrete checking misses "sweep-through" collisions.

```rust
// Start: arm on left side of obstacle
// Goal: arm on right side of obstacle
// The arm sweeps through the obstacle between waypoints

// Discrete check at start: no collision
assert!(!scene.check_collision(&start).unwrap());
// Discrete check at goal: no collision
assert!(!scene.check_collision(&goal).unwrap());

// CCD: MUST detect collision
let ccd_result = scene.check_continuous_collision(&start, &goal);
assert!(ccd_result.collision_detected,
    "{}: CCD missed sweep-through collision", name);
```

**Pass criteria**: CCD detects all sweep-through collisions that discrete misses.

---

## TEST 10: Allowed Collision Matrix (ACM)

**Function**: `p1_acm_correctness`

For robots with SRDF files (Franka Panda, UR5e):
1. Load robot with SRDF: `Robot::from_urdf_srdf(urdf, srdf)`
2. ACM should disable self-collision for adjacent links
3. With ACM: configs that touch adjacent links → NOT reported as collision
4. Without ACM: same configs → reported as collision

**Pass criteria**: ACM correctly filters expected link-pair contacts.

---

## TEST 11: Collision Margin / Padding

**Function**: `p1_collision_margin_conservative`

For `REPRESENTATIVE_ROBOTS`:
1. Find a config where the EE is 5cm from an obstacle (just outside contact)
2. Check collision with margin=0.0 → NOT in collision
3. Check collision with margin=0.1 (10cm) → IN collision (because 5cm < 10cm margin)
4. The margin makes checking more conservative — never less

```rust
// EE at 5cm from obstacle
scene.add("near_obstacle", Shape::sphere(0.1),
    Isometry3::translation(ee_pos.x + 0.15, ee_pos.y, ee_pos.z));

// No margin: no collision (5cm gap)
assert!(!scene.check_collision(&joints).unwrap());

// With 10cm margin: collision (5cm < 10cm)
assert!(scene.check_collision_with_margin(&joints, 0.1).unwrap());
```

---

## TEST 12: SIMD vs Scalar Consistency

**Function**: `p1_simd_scalar_collision_consistency`

For `REPRESENTATIVE_ROBOTS` × 1000 random configs:
1. Check self-collision using SIMD path
2. Check self-collision using scalar fallback
3. Results MUST agree (same `bool` result)
4. Distance values must agree within `tol::SIMD_SCALAR_TOL` (1e-8)

This ensures the SIMD optimization doesn't change collision results.

---

## TEST 13: Collision with Multiple Obstacles

**Function**: `p1_multi_obstacle_scene`

For `REPRESENTATIVE_ROBOTS`:
1. Add 20 random obstacles to the scene
2. Plan through the cluttered environment
3. Verify every waypoint is collision-free against ALL 20 obstacles
4. The planner must respect every obstacle, not just the first one

---

## TEST 14: Dynamic Scene — Add/Remove Obstacles

**Function**: `p1_dynamic_scene_add_remove`

1. Plan a trajectory in empty scene (succeeds)
2. Add obstacle that blocks the planned path
3. Check collision on the planned trajectory — must now detect collision
4. Remove the obstacle
5. Check collision again — must now be clear

Tests that scene modifications are reflected in collision checks.

---

## TEST 15: Empty Scene — No False Positives

**Function**: `p0_empty_scene_no_false_positives`

For every robot × 100 random configs:
1. Scene is empty (no obstacles)
2. Environment collision check must return `false`
3. (Self-collision may be true — that's correct. Only environment collision must be false.)

**Pass criteria**: ZERO false positives in empty scene environment checks.

---

## TEST 16: Sphere Model Coverage

**Function**: `p1_sphere_model_covers_geometry`

For `ROBOTS_WITH_COLLISION`:
1. Generate `RobotSphereModel::from_robot_default(&robot)`
2. The sphere model must have at least 1 sphere per link
3. Coarse model: fewer spheres than fine model
4. All sphere radii must be positive and finite
5. All sphere centers must be finite

```rust
let default = RobotSphereModel::from_robot_default(&robot);
let coarse = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
let fine = RobotSphereModel::from_robot(&robot, &SphereGenConfig::fine());

// Fine must have more or equal spheres than coarse
assert!(fine.num_spheres() >= coarse.num_spheres(),
    "{}: fine ({}) has fewer spheres than coarse ({})",
    name, fine.num_spheres(), coarse.num_spheres());
```

---

## TEST 17: Planner + Scene End-to-End

**Function**: `p0_e2e_plan_with_scene_collision_free`

For `REPRESENTATIVE_ROBOTS`:
1. Set up scene with 5 obstacles at various positions
2. Plan 10 trajectories (random start/goal, both collision-free)
3. Time-parameterize each trajectory
4. Sample at 1ms intervals
5. At EVERY sample point, check collision against the scene

**Pass criteria**: ZERO collisions at any sampled point. This is the gold standard test.

```rust
let dt = 0.001;
let mut t = 0.0;
while t <= traj.duration.as_secs_f64() {
    let sample = traj.sample_at(Duration::from_secs_f64(t));
    assert!(!scene.check_collision(&sample.positions).unwrap(),
        "{}: collision at t={:.3}s in planned trajectory", name, t);
    t += dt;
}
```

---

## SUMMARY

| Test | Robots | Configs | Total Checks | Priority |
|------|--------|---------|--------------|----------|
| Known self-collision | ~8 | ~5 each | ~40 | P0 |
| No self-collision at home | 52 | 1 | 52 | P0 |
| Box obstacle | 5 | 2 | 10 | P0 |
| Sphere obstacle | 5 | 2 | 10 | P0 |
| Cylinder obstacle | 5 | 2 | 10 | P0 |
| Trajectory collision | 5 | 5 | 25 | P0 |
| Planner avoids | 5 | 5 | 25×20wp | P0 |
| Blocked path | 5 | 5 | 25 | P0 |
| CCD sweep-through | 5 | 5 | 25 | P0 |
| ACM correctness | 2 | ~10 | 20 | P1 |
| Collision margin | 5 | 5 | 25 | P1 |
| SIMD vs scalar | 5 | 1000 | 5,000 | P1 |
| Multi-obstacle | 5 | 5 | 25×20obs | P1 |
| Dynamic scene | 5 | 1 | 5 | P1 |
| Empty scene FP | 52 | 100 | 5,200 | P0 |
| Sphere model | ~8 | 1 | 8 | P1 |
| E2E scene collision | 5 | 10×1000samp | 50,000 | P0 |
