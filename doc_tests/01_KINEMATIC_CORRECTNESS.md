# 01 — Kinematic Correctness Tests

**Priority**: P0 — wrong FK/IK can cause robot collision with environment or self
**File**: `crates/kinetic/tests/acceptance/test_kinematic_correctness.rs`
**Estimated parameterized cases**: ~33,000

---

## TEST 1: FK Determinism

**Function**: `p0_fk_determinism_all_robots`

For every robot in `ALL_ROBOTS`:
1. Load robot, extract chain via `KinematicChain::auto_detect`
2. Generate 20 random joint configs using `random_joints_within_limits(margin=0.05)` with seed `42`
3. For each config, call `forward_kinematics(&robot, &chain, &joints)` **twice**
4. Assert the two poses are **identical** (not just close — bitwise equal `f64`)

**Pass criteria**: 0 robots fail. FK is a pure function — same input MUST yield same output.

```rust
#[test]
fn p0_fk_determinism_all_robots() {
    let (passed, failed, _) = run_for_all_robots(|name, _, _| {
        let (robot, chain) = load_robot_and_chain(name);
        let mut rng = seeded_rng(42);
        for i in 0..20 {
            let joints = random_joints_within_limits(&robot, &chain, &mut rng, 0.05);
            let pose1 = forward_kinematics(&robot, &chain, &joints)
                .map_err(|e| format!("FK failed at config {}: {}", i, e))?;
            let pose2 = forward_kinematics(&robot, &chain, &joints)
                .map_err(|e| format!("FK failed at config {} (2nd call): {}", i, e))?;
            let pos_diff = pose1.translation_distance(&pose2);
            let ori_diff = pose1.rotation_distance(&pose2);
            if pos_diff != 0.0 || ori_diff != 0.0 {
                return Err(format!("config {}: pos_diff={}, ori_diff={}", i, pos_diff, ori_diff));
            }
        }
        Ok(())
    });
    assert_eq!(failed, 0, "FK determinism: {}/{} robots failed", failed, passed + failed);
}
```

---

## TEST 2: FK Produces Finite Poses

**Function**: `p0_fk_finite_all_robots`

For every robot:
1. Test FK at: zero config, home config, mid config, 100 random configs
2. Assert ALL components of the resulting Pose are finite (no NaN, no Inf)
3. Assert the rotation quaternion has unit norm (within 1e-10)

**Pass criteria**: 0 robots fail.

```
For each robot:
  configs = [zeros(dof), home, mid] + 100 random configs (seed=123)
  For each config:
    pose = fk(robot, chain, config)
    assert pose.translation().x.is_finite()
    assert pose.translation().y.is_finite()
    assert pose.translation().z.is_finite()
    quat = pose.rotation()
    assert (quat.norm() - 1.0).abs() < 1e-10
```

---

## TEST 3: FK at Joint Limits

**Function**: `p0_fk_at_joint_limits`

For every robot:
1. Generate configs from `joints_at_limits()` (all lower, all upper, each joint at limit)
2. FK must succeed and produce finite poses at every limit configuration
3. This tests that FK doesn't break at mechanical stops

**Pass criteria**: 0 failures. The robot physically reaches its limits — FK must handle them.

---

## TEST 4: FK Produces Distinct Poses

**Function**: `p0_fk_distinct_poses`

For every robot:
1. Generate 10 random configs with seed=456
2. Compute FK for each
3. Assert that no two configs >0.01 rad apart in joint space produce identical poses
4. (Two configs that are very close in joint space MAY produce nearly identical poses — that's fine)

**Pass criteria**: All distinct joint configs produce distinct Cartesian poses. If they don't, the FK is degenerate (e.g., ignoring a joint).

```
For i, j where joint_distance(config_i, config_j) > 0.01:
  assert pose_i.translation_distance(pose_j) > 1e-6
      OR pose_i.rotation_distance(pose_j) > 1e-6
```

---

## TEST 5: FK → IK Roundtrip (All Robots, All Solvers)

**Function**: `p0_ik_roundtrip_all_robots_all_solvers`

This is the most critical kinematic test. For every robot × every applicable solver:

1. Generate 30 random configs (seed=789)
2. Compute FK to get target pose
3. Solve IK from a **different** seed (mid-config)
4. Compute FK on the IK solution
5. Compare recovered pose to target pose

**Solver applicability**:
- `DLS`: all robots (4-7 DOF)
- `FABRIK`: all robots (4-7 DOF)
- `OPW`: only robots where `is_opw_compatible()` returns true (specific 6-DOF geometries)
- `Subproblem`: only 6-DOF robots where `is_subproblem_compatible()` returns true
- `Subproblem7DOF`: only 7-DOF robots

**IK Config**:
```rust
IKConfig {
    solver: <varies>,
    max_iterations: 500,
    num_restarts: 10,
    position_tolerance: 1e-4,
    orientation_tolerance: 1e-3,
    check_limits: true,
    seed: Some(mid_joints(&robot, &chain)),
    ..IKConfig::default()
}
```

**Pass criteria per robot per solver**:
- DLS: ≥20/30 solves converge (66%)
- FABRIK: ≥15/30 converge (50%)
- OPW: ≥25/30 converge (83%) — analytical should be high
- Subproblem: ≥25/30 converge (83%)
- Subproblem7DOF: ≥15/30 converge (50%)

**For converged solutions**:
- Position error < 1mm (1e-3 m)
- Orientation error < 0.01 rad (~0.57 degrees)
- All joints within limits (epsilon = 1e-6)

**Total**: 52 robots × ~3 applicable solvers × 30 configs = ~4,680 IK solves

---

## TEST 6: IK Solution Joint Limits

**Function**: `p0_ik_solutions_within_limits`

For every robot × every IK solve in TEST 5 that converges:
1. Check every joint value against the robot's limits
2. Epsilon = 1e-6 (accounts for floating point)

**Pass criteria**: ZERO violations. A single joint value outside limits = FAIL.

```
For each converged IK solution:
  For each joint i:
    assert solution.joints[i] >= limits[i].lower - 1e-6
    assert solution.joints[i] <= limits[i].upper + 1e-6
```

This is separate from TEST 5 because it's a harder requirement — even if the pose is slightly off, joints MUST be within limits.

---

## TEST 7: IK with Seed at Solution

**Function**: `p0_ik_warm_start_convergence`

For every robot:
1. Generate random config → FK → get target pose
2. Solve IK with `seed = Some(original_config)` (warm start from the answer)
3. IK MUST converge since we're starting at the solution

**Pass criteria**: 100% convergence rate across all robots.
**Tolerance**: position < 1e-4, orientation < 1e-3.

This tests that the IK solver doesn't diverge from a known solution.

---

## TEST 8: IK at Workspace Boundary

**Function**: `p0_ik_workspace_boundary`

For every robot:
1. Compute workspace radius: FK at max-reach config (all joints at 0 or fully extended)
2. Create targets at 95%, 100%, and 105% of max reach (along +X axis)
3. Solve IK for each

**Pass criteria**:
- 95% reach: IK should converge (inside workspace)
- 100% reach: IK may or may not converge (boundary)
- 105% reach: IK MUST return `Err(NoIKSolution)` or `Err(GoalUnreachable)`, never `Ok` with garbage
- NO panics at any reach distance

---

## TEST 9: Jacobian Accuracy

**Function**: `p1_jacobian_finite_difference_vs_analytical`

For `REPRESENTATIVE_ROBOTS` (5 robots) × 100 random configs:
1. Compute analytical Jacobian: `jacobian(&robot, &chain, &joints)`
2. Compute numerical Jacobian via finite differences (step = 1e-7)
3. Compare element-wise

**Finite difference Jacobian computation**:
```
For each joint j (column):
  joints_plus = joints.clone(); joints_plus[j] += h
  joints_minus = joints.clone(); joints_minus[j] -= h
  pose_plus = fk(robot, chain, &joints_plus)
  pose_minus = fk(robot, chain, &joints_minus)
  J[:, j] = (pose_plus - pose_minus) / (2*h)  // central difference
```

**Pass criteria**: Every element matches within `tol::JACOBIAN_TOL` (1e-5).

---

## TEST 10: Manipulability at Singularity

**Function**: `p1_manipulability_near_singularity`

For `REPRESENTATIVE_ROBOTS`:
1. Find a near-singular configuration (e.g., elbow fully extended for UR5e)
2. Compute `manipulability(&robot, &chain, &joints)`
3. Assert it's a small positive number (close to 0), not NaN or negative

**Known singular configs** (implement per robot):
- UR5e: `[0, 0, 0, 0, 0, 0]` (fully extended)
- Franka Panda: `[0, 0, 0, -PI/2, 0, PI/2, 0]` (wrist singularity)

**Pass criteria**: `manipulability >= 0.0` and `manipulability.is_finite()` for all configs.

---

## TEST 11: FK Batch Consistency

**Function**: `p1_fk_batch_matches_individual`

For `REPRESENTATIVE_ROBOTS`:
1. Generate 50 random configs
2. Compute individual FK for each: `fk(&robot, &chain, &config_i)`
3. Compute batch FK: `fk_batch(&robot, &chain, &all_configs)`
4. Assert each batch result matches individual result exactly

**Pass criteria**: Bitwise identical f64 values. Batch is an optimization — must not change results.

---

## TEST 12: FK All Links

**Function**: `p1_fk_all_links_consistency`

For every robot:
1. Random config
2. `fk_all_links(&robot, &joints)` → Vec<Pose> for every link
3. The tip link's pose must match `fk(&robot, &chain, &joints)`
4. The base link's pose must be identity (or the robot's base transform)
5. All poses must be finite

---

## TEST 13: IK Mode Fallback

**Function**: `p1_ik_position_fallback_mode`

For `REPRESENTATIVE_ROBOTS`:
1. Create a target pose at ~80% reach
2. Solve with `IKMode::Full6D` — may or may not converge
3. Solve with `IKMode::PositionOnly` — should converge more often
4. Solve with `IKMode::PositionFallback` — should converge at least as often as PositionOnly
5. For PositionOnly solutions: verify `solution.position_error < 1e-3` (orientation may be wrong)

**Pass criteria**: `PositionFallback` convergence rate ≥ `PositionOnly` convergence rate.

---

## TEST 14: IK Null Space (7-DOF)

**Function**: `p1_ik_null_space_7dof`

For `ROBOTS_7DOF` only:
1. Target pose at ~60% reach (many solutions exist)
2. Solve IK with `NullSpace::Manipulability` — should maximize manipulability
3. Solve IK with `NullSpace::JointCentering` — should be closer to mid-config
4. Both solutions must be valid (FK recovers target pose within tolerance)

**Pass criteria**:
- Manipulability solution: `manipulability(sol_manip) >= manipulability(sol_center)` in ≥60% of cases
- JointCentering solution: `distance_to_mid(sol_center) <= distance_to_mid(sol_manip)` in ≥60% of cases

---

## TEST 15: IK Determinism with Seed

**Function**: `p0_ik_deterministic_with_seed`

For every robot:
1. Fixed target pose, fixed IK config with `seed = Some([0.1; dof])`
2. Solve IK 10 times
3. All 10 solutions must be **identical** (same seed = same result)

**Pass criteria**: 0 differences across repeated solves.

---

## SUMMARY

| Test | Robots | Configs/Robot | Total Cases | Priority |
|------|--------|---------------|-------------|----------|
| FK determinism | 52 | 20 | 1,040 | P0 |
| FK finite | 52 | 103 | 5,356 | P0 |
| FK at limits | 52 | ~8 | ~416 | P0 |
| FK distinct | 52 | 10 | 520 | P0 |
| IK roundtrip | 52 | 30×3 solvers | ~4,680 | P0 |
| IK limits | 52 | (from above) | ~4,680 | P0 |
| IK warm start | 52 | 10 | 520 | P0 |
| IK workspace boundary | 52 | 3 | 156 | P0 |
| Jacobian accuracy | 5 | 100 | 500 | P1 |
| Manipulability | 5 | ~3 | ~15 | P1 |
| FK batch | 5 | 50 | 250 | P1 |
| FK all links | 52 | 1 | 52 | P1 |
| IK mode fallback | 5 | 10 | 50 | P1 |
| IK null space | 13 | 10 | 130 | P1 |
| IK determinism | 52 | 10 | 520 | P0 |
| **TOTAL** | | | **~18,885** | |
