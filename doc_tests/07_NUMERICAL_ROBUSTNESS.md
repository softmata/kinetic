# 07 — Numerical Robustness Tests

**Priority**: P1 — silent numerical errors produce wrong trajectories; NaN propagation is insidious
**File**: `crates/kinetic/tests/acceptance/test_numerical_robustness.rs`
**Estimated parameterized cases**: ~12,000

---

## PRINCIPLE

Every public API function must handle degenerate inputs gracefully:
- NaN → return `Err`, never propagate silently
- Inf → return `Err`, never propagate silently
- Near-zero denominators → bounded result via damping/fallback
- Extreme values → bounded result or clear error
- NEVER panic on any numerical input

---

## TEST 1: NaN Joint Values → FK

**Function**: `p1_nan_joints_fk`

For `REPRESENTATIVE_ROBOTS`:
1. Create joint values with NaN at each position (one at a time)
2. Call `fk(&robot, &chain, &joints)`
3. Must return `Err` OR return a Pose where NaN is detectable
4. Must NEVER panic

```rust
for j in 0..dof {
    let mut joints = mid_joints(&robot, &chain);
    joints[j] = f64::NAN;
    let result = fk(&robot, &chain, &joints);
    match result {
        Err(_) => {} // Good: caught the NaN
        Ok(pose) => {
            // If it returned Ok, the pose must contain NaN (not a valid pose)
            // OR we flag this as a test failure because NaN was silently consumed
            let t = pose.translation();
            if t.x.is_finite() && t.y.is_finite() && t.z.is_finite() {
                panic!("{}: FK consumed NaN joint {} and produced finite pose — silent corruption",
                    name, j);
            }
        }
    }
}
```

---

## TEST 2: NaN Joint Values → IK

**Function**: `p1_nan_joints_ik`

For `REPRESENTATIVE_ROBOTS`:
1. Create a valid target Pose
2. Set IK seed to contain NaN: `config.seed = Some(vec_with_nan)`
3. Solve IK
4. Must return `Err` or converge to a valid (all-finite) solution
5. Must NEVER return an `Ok(IKSolution)` with NaN in the joints

```rust
let mut seed = mid_joints(&robot, &chain);
seed[0] = f64::NAN;
let config = IKConfig { seed: Some(seed), ..IKConfig::dls() };
let result = solve_ik(&robot, &chain, &target, &config);
if let Ok(sol) = result {
    assert_all_finite(&sol.joints, &format!("{}: IK solution with NaN seed", name));
}
```

---

## TEST 3: NaN Target Pose → IK

**Function**: `p1_nan_target_ik`

1. Create target Pose with NaN translation: `Pose::from_xyz(f64::NAN, 0.0, 0.5)`
2. Solve IK
3. Must return `Err`
4. Must NEVER return Ok with a valid-looking solution

---

## TEST 4: Inf Joint Values → FK

**Function**: `p1_inf_joints_fk`

Same as TEST 1 but with `f64::INFINITY` and `f64::NEG_INFINITY`.
Must not panic, must not produce finite-looking poses from infinite inputs.

---

## TEST 5: Inf Joint Values → IK

**Function**: `p1_inf_joints_ik`

Same as TEST 2 but with Inf in seed. Must not diverge forever.

---

## TEST 6: f64::MAX / f64::MIN Joint Values

**Function**: `p1_extreme_joints`

For `REPRESENTATIVE_ROBOTS`:
1. Joints = `[f64::MAX; dof]`
2. FK: must not panic, result must not be finite (or must return Err)
3. IK with these as seed: must not panic, may return Err or converge

Also test `f64::MIN_POSITIVE` (subnormal-adjacent).

---

## TEST 7: Near-Singular Jacobian

**Function**: `p1_near_singular_jacobian`

For `REPRESENTATIVE_ROBOTS`:
1. Find a near-singular config (arm fully extended)
2. Compute `jacobian(&robot, &chain, &joints)`
3. All elements must be finite (no NaN, no Inf)
4. Condition number may be huge — that's OK as long as it's finite

```rust
// UR5e near-singular: arm fully extended
let singular = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let jac = jacobian(&robot, &chain, &singular).unwrap();
for i in 0..jac.nrows() {
    for j in 0..jac.ncols() {
        assert!(jac[(i, j)].is_finite(),
            "Jacobian[{},{}] = {} at singular config", i, j, jac[(i, j)]);
    }
}
```

---

## TEST 8: IK at Singularity Doesn't Hang

**Function**: `p0_ik_at_singularity_terminates`

For `REPRESENTATIVE_ROBOTS`:
1. Target pose at a known singular configuration
2. IK with max_iterations=500
3. Must terminate within reasonable time (5 seconds wall clock)
4. May return Ok or Err — but must not hang

```rust
let start = std::time::Instant::now();
let config = IKConfig {
    max_iterations: 500,
    num_restarts: 3,
    ..IKConfig::dls()
};
let _ = solve_ik(&robot, &chain, &singular_pose, &config);
assert!(start.elapsed() < Duration::from_secs(5),
    "{}: IK took {:?} at singularity", name, start.elapsed());
```

---

## TEST 9: Trajectory from Near-Zero Motion

**Function**: `p1_trajectory_near_zero_motion`

1. Waypoints where each joint moves by only 1e-12 rad
2. `trapezoidal(&waypoints, 1.0, 2.0)`
3. Must not produce NaN or Inf in velocities
4. Duration may be very short or zero — both are OK

---

## TEST 10: Collision with NaN Joints

**Function**: `p1_collision_nan_joints`

1. Check collision with NaN in joint values
2. Must return `Err` or `false` (conservative) — never `Ok(true)` claiming collision where it can't compute
3. Must NEVER panic

---

## TEST 11: Collision with Coincident Spheres

**Function**: `p1_collision_coincident_spheres`

1. Two spheres at exactly the same position and same radius
2. Distance = 0, depth = 2×radius
3. Must report collision, must not divide by zero when computing contact normal

---

## TEST 12: Quaternion Edge Cases in Pose

**Function**: `p1_quaternion_edge_cases`

Test Pose construction with edge-case quaternions:
1. Identity quaternion: `(0, 0, 0, 1)` → `Pose::from_xyz_quat(0,0,0, 0,0,0,1)` — must work
2. 180-degree rotation: `(1, 0, 0, 0)` — must work
3. Zero quaternion: `(0, 0, 0, 0)` — must return Err or normalize to identity
4. Non-unit quaternion: `(1, 1, 1, 1)` — must normalize or reject
5. NaN quaternion: `(NaN, 0, 0, 1)` — must reject

```rust
// Identity
let p = Pose::from_xyz_quat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
assert!(p.rotation().angle() < 1e-10, "identity rotation has angle");

// 180 degrees about X
let p = Pose::from_xyz_quat(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
assert!((p.rotation().angle() - PI).abs() < 1e-10, "180 rotation wrong angle");

// Zero quaternion — should not produce valid pose without normalization
let p = Pose::from_xyz_quat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
// Must either: normalize to something, OR the rotation norm is NaN
// In either case: must not panic
```

---

## TEST 13: Large Translation Values

**Function**: `p1_large_translations`

1. `Pose::from_xyz(1e10, 1e10, 1e10)` — valid
2. `Pose::from_xyz(1e308, 0.0, 0.0)` — near f64::MAX
3. Compose two large-translation poses: `a.compose(&b)`
4. Result translation = sum of translations — check for overflow
5. Must not produce Inf from finite inputs that happen to overflow

---

## TEST 14: Proptest — FK Never Panics

**Function**: `p1_proptest_fk_no_panic`

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn p1_proptest_fk_no_panic(
        joints in prop::collection::vec(-100.0f64..100.0, 6..=6)
    ) {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        // Must not panic — Ok or Err both fine
        let _ = fk(&robot, &chain, &joints);
    }
}
```

---

## TEST 15: Proptest — IK Never Panics

**Function**: `p1_proptest_ik_no_panic`

```rust
proptest! {
    #[test]
    fn p1_proptest_ik_no_panic(
        x in -10.0f64..10.0,
        y in -10.0f64..10.0,
        z in -10.0f64..10.0,
    ) {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        let target = Pose::from_xyz(x, y, z);
        let _ = solve_ik(&robot, &chain, &target, &IKConfig::dls());
    }
}
```

---

## TEST 16: Proptest — Planner Never Panics

**Function**: `p1_proptest_planner_no_panic`

```rust
proptest! {
    #[test]
    fn p1_proptest_planner_no_panic(
        start in prop::collection::vec(-3.14f64..3.14, 6..=6),
        goal in prop::collection::vec(-3.14f64..3.14, 6..=6),
    ) {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let planner = Planner::new(&robot).unwrap();
        let config = PlannerConfig {
            timeout: Duration::from_millis(100),
            ..PlannerConfig::default()
        };
        let _ = planner.plan_with_config(&start,
            &Goal::Joints(JointValues::new(goal)), config);
    }
}
```

---

## TEST 17: Proptest — Trapezoidal Never Panics

**Function**: `p1_proptest_trapezoidal_no_panic`

```rust
proptest! {
    #[test]
    fn p1_proptest_trapezoidal_no_panic(
        waypoint_count in 1usize..20,
        dof in 1usize..8,
        max_vel in 0.01f64..100.0,
        max_accel in 0.01f64..100.0,
    ) {
        let waypoints: Vec<Vec<f64>> = (0..waypoint_count)
            .map(|i| (0..dof).map(|j| (i * dof + j) as f64 * 0.1).collect())
            .collect();
        let _ = trapezoidal(&waypoints, max_vel, max_accel);
    }
}
```

---

## TEST 18: DLS Solver Condition Number Awareness

**Function**: `p1_dls_condition_number`

For "ur5e":
1. At well-conditioned config: verify IK converges quickly (<50 iterations)
2. At ill-conditioned config (near singularity): verify IK still terminates
3. IK solution from ill-conditioned start: check `solution.iterations` is higher (more work needed)
4. IK solution quality may be worse at singularity — that's OK, but must not be silently wrong

---

## SUMMARY

| Test | Inputs | Total Cases | Priority |
|------|--------|-------------|----------|
| NaN → FK | 5 robots × dof | ~35 | P1 |
| NaN → IK | 5 × dof | ~35 | P1 |
| NaN target → IK | 5 × 3 axes | 15 | P1 |
| Inf → FK | 5 × dof × 2 | ~70 | P1 |
| Inf → IK | 5 × dof × 2 | ~70 | P1 |
| f64::MAX | 5 × 2 | 10 | P1 |
| Near-singular Jacobian | 5 × 3 configs | 15 | P1 |
| IK at singularity | 5 × 3 | 15 | P0 |
| Near-zero motion | 5 × 1 | 5 | P1 |
| Collision NaN | 5 × dof | ~35 | P1 |
| Coincident spheres | 1 | 1 | P1 |
| Quaternion edges | 5 cases | 5 | P1 |
| Large translations | 3 cases | 3 | P1 |
| Proptest FK | 10,000 | 10,000 | P1 |
| Proptest IK | 1,000 | 1,000 | P1 |
| Proptest planner | 500 | 500 | P1 |
| Proptest trapezoidal | 500 | 500 | P1 |
| DLS condition | 1 × 3 | 3 | P1 |
