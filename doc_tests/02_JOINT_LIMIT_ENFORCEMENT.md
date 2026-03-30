# 02 — Joint Limit Enforcement Tests

**Priority**: P0 — a joint driven past its limit can strip gears, break the robot, or injure nearby humans
**File**: `crates/kinetic/tests/acceptance/test_joint_limits.rs`
**Estimated parameterized cases**: ~36,000

---

## PRINCIPLE

Joint limits are the LAST line of defense. Every layer must enforce them independently:
1. **IK solver** — solutions must be within limits
2. **Planner** — every waypoint must be within limits
3. **Trajectory parameterizer** — interpolated positions must stay within limits
4. **Executor** — commanded positions must be clamped to limits
5. **Validator** — independent check catches any layer that fails

If ANY layer passes an out-of-limits value, the test FAILS.

---

## TEST 1: Robot Limits Are Valid

**Function**: `p0_robot_limits_valid`

For every robot in `ALL_ROBOTS`:
1. Load robot
2. For each active joint, check:
   - `limits.lower < limits.upper` (not inverted)
   - `limits.lower.is_finite()` and `limits.upper.is_finite()` (for non-continuous joints)
   - `limits.velocity > 0.0` (positive velocity limit)
   - `limits.effort >= 0.0` (non-negative effort)
   - For continuous joints: limits may be absent or ±MAX — that's OK

**Pass criteria**: 0 invalid limit definitions.

```rust
for &ji in &chain.active_joints {
    let joint = &robot.joints[ji];
    if joint.joint_type != JointType::Continuous {
        if let Some(ref lim) = joint.limits {
            assert!(lim.lower < lim.upper,
                "'{}' joint '{}': lower {} >= upper {}",
                name, joint.name, lim.lower, lim.upper);
            assert!(lim.lower.is_finite(), "'{}' joint '{}': lower is not finite", name, joint.name);
            assert!(lim.upper.is_finite(), "'{}' joint '{}': upper is not finite", name, joint.name);
        }
    }
}
```

---

## TEST 2: check_limits() Catches Violations

**Function**: `p0_check_limits_catches_violations`

For every robot:
1. Generate a config where each joint is 0.1 rad PAST its upper limit
2. Call `robot.check_limits(&joints)` — MUST return `Err(JointLimitViolation { .. })`
3. Generate a config where each joint is 0.1 rad BELOW its lower limit
4. Call `robot.check_limits(&joints)` — MUST return `Err(JointLimitViolation { .. })`
5. Generate a valid mid-config
6. Call `robot.check_limits(&joints)` — MUST return `Ok(())`

**Pass criteria**: check_limits correctly identifies all violations and accepts all valid configs.

---

## TEST 3: clamp_to_limits() Works

**Function**: `p0_clamp_to_limits_correctness`

For every robot:
1. Create joints 1.0 rad past upper limit on every joint
2. Call `robot.clamp_to_limits(&mut joints)`
3. Assert every joint is now at exactly the upper limit (within f64 epsilon)
4. Create joints 1.0 rad below lower limit on every joint
5. Clamp again
6. Assert every joint is now at exactly the lower limit

```rust
// After clamping past-upper:
for (i, &ji) in chain.active_joints.iter().enumerate() {
    if let Some(ref lim) = robot.joints[ji].limits {
        assert!((joints[i] - lim.upper).abs() < 1e-15,
            "joint {} not clamped to upper: {} vs {}", i, joints[i], lim.upper);
    }
}
```

---

## TEST 4: IK Never Returns Out-of-Limits Solutions

**Function**: `p0_ik_output_within_limits_exhaustive`

For every robot × DLS solver × 100 random target poses (seed=1000):
1. Solve IK with `check_limits: true`
2. If `Ok(solution)`:
   - Assert EVERY joint is within `[lower - 1e-6, upper + 1e-6]`
   - A single violation = FAIL

**Total**: 52 robots × 100 configs = 5,200 IK solves

**Pass criteria**: ZERO joint limit violations across all converged solutions.

```rust
if let Ok(ref sol) = result {
    assert_within_limits(&robot, &chain, &sol.joints, tol::JOINT_LIMIT_EPS,
        &format!("{} config {}", name, i));
}
```

---

## TEST 5: IK with check_limits=false Still Reports Violations

**Function**: `p1_ik_unchecked_limits_flagged`

For `REPRESENTATIVE_ROBOTS`:
1. Solve IK with `check_limits: false`
2. If the solution has joints outside limits, verify we can DETECT this via `robot.check_limits()`
3. This tests that even if IK doesn't enforce limits, the user can catch it

**Pass criteria**: `robot.check_limits()` catches every out-of-limits solution.

---

## TEST 6: Planner Output Within Limits

**Function**: `p0_planner_output_within_limits`

For every robot × 10 random start/goal pairs (both within limits):
1. Plan with `Planner::new(&robot).plan(&start, &goal)`
2. For EVERY waypoint in the result:
   - Assert every joint is within limits (epsilon = 1e-6)

**Total**: 52 robots × 10 plans × ~20 waypoints/plan = ~10,400 waypoint checks

**Pass criteria**: ZERO violations.

```rust
for (wi, waypoint) in result.waypoints.iter().enumerate() {
    for (ji, &val) in waypoint.iter().enumerate() {
        let joint_idx = chain.active_joints[ji];
        if let Some(ref lim) = robot.joints[joint_idx].limits {
            assert!(
                val >= lim.lower - 1e-6 && val <= lim.upper + 1e-6,
                "{}: waypoint {} joint {} = {:.6} outside [{:.6}, {:.6}]",
                name, wi, ji, val, lim.lower, lim.upper
            );
        }
    }
}
```

---

## TEST 7: Trajectory Parameterization Respects Velocity Limits

**Function**: `p0_trajectory_velocity_within_limits`

For `REPRESENTATIVE_ROBOTS`:
1. Plan a trajectory (10 random plans per robot)
2. Time-parameterize with `trapezoidal_per_joint` using robot's velocity/acceleration limits
3. Sample the trajectory at 1ms intervals across its full duration
4. At EVERY sample point, check:
   - `|velocity[j]| <= vel_limit[j]` for all joints j

**Sampling**:
```rust
let dt = 0.001; // 1ms
let mut t = 0.0;
while t <= traj.duration.as_secs_f64() {
    let sample = traj.sample_at(Duration::from_secs_f64(t));
    for (j, &vel) in sample.velocities.iter().enumerate() {
        assert!(vel.abs() <= vel_limits[j] * tol::VEL_SAFETY_FACTOR,
            "{}: t={:.3}s joint {} velocity {:.4} > limit {:.4}",
            name, t, j, vel.abs(), vel_limits[j]);
    }
    t += dt;
}
```

**Pass criteria**: ZERO velocity violations at any sample point.

---

## TEST 8: Trajectory Parameterization Respects Acceleration Limits

**Function**: `p0_trajectory_acceleration_within_limits`

Same as TEST 7 but for accelerations:
1. Sample at 1ms intervals
2. Check `|acceleration[j]| <= accel_limit[j] * 1.05` for all joints

**Pass criteria**: ZERO acceleration violations.

---

## TEST 9: Trajectory Interpolation Stays Within Position Limits

**Function**: `p0_trajectory_interpolation_within_position_limits`

This catches a subtle bug: even if waypoints are within limits, linear interpolation BETWEEN waypoints might exceed limits if the path crosses a limit boundary.

For `REPRESENTATIVE_ROBOTS` × 10 plans:
1. Time-parameterize
2. Sample at 0.5ms intervals
3. At every sample: `position[j] ∈ [lower[j] - 1e-6, upper[j] + 1e-6]`

**Pass criteria**: ZERO position limit violations at any interpolated point.

---

## TEST 10: Continuous Joint Handling

**Function**: `p0_continuous_joint_no_discontinuity`

For robots with continuous joints (check `JointType::Continuous`):
1. Plan a trajectory where a continuous joint wraps around ±PI
2. The trajectory should use the SHORTEST path (e.g., +3.0 → -3.0 should go through ±PI, not through 0)
3. Between adjacent waypoints: position jump < PI (no sudden ±2PI jump)

**Pass criteria**: No position discontinuity > PI between adjacent waypoints on continuous joints.

---

## TEST 11: Fuzz — Random f64 Values

**Function**: `p0_fuzz_check_limits_never_panics`

Using proptest or manual fuzzing:
1. Generate 10,000 random `Vec<f64>` of the correct DOF
2. Include: NaN, Inf, -Inf, f64::MAX, f64::MIN, subnormals, 0.0
3. Call `robot.check_limits(&joints)`
4. MUST return `Ok` or `Err` — NEVER panic

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn p0_fuzz_check_limits_never_panics(
        joints in prop::collection::vec(prop::num::f64::ANY, 6..=7)
    ) {
        let robot = Robot::from_name("ur5e").unwrap();
        // Must not panic — Ok or Err both acceptable
        let _ = robot.check_limits(&JointValues::new(joints));
    }
}
```

**Pass criteria**: 0 panics across 10,000 random inputs.

---

## TEST 12: Executor Clamps Commands

**Function**: `p0_executor_clamps_out_of_limits_commands`

1. Create a `TimedTrajectory` with waypoints slightly OUTSIDE limits (limits + 0.05 rad)
2. Execute with a `RecordingSink` (custom `CommandSink` that records all sent commands)
3. Assert every command sent by the executor is within limits

This tests that the executor is the FINAL safety net.

```rust
struct RecordingSink {
    commands: Vec<(Vec<f64>, Vec<f64>)>,
}
impl CommandSink for RecordingSink {
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
        self.commands.push((positions.to_vec(), velocities.to_vec()));
        Ok(())
    }
}

// After execution:
for (ci, (positions, _)) in sink.commands.iter().enumerate() {
    for (ji, &pos) in positions.iter().enumerate() {
        // Must be within limits
        assert!(pos >= lower[ji] - 1e-6 && pos <= upper[ji] + 1e-6,
            "Command {} joint {} = {} outside limits", ci, ji, pos);
    }
}
```

**Pass criteria**: ZERO out-of-limits commands sent to the sink.

---

## TEST 13: TrajectoryValidator Catches Violations

**Function**: `p0_validator_catches_limit_violations`

1. Create a valid trajectory
2. Manually inject one waypoint with joint 0 at `upper + 0.1`
3. Run `TrajectoryValidator::validate()`
4. Assert it returns `Err` containing a `ViolationType::PositionLimit` violation

Also test:
- Velocity violation: inject `velocity[0] = vel_limit[0] * 2.0`
- Acceleration violation: inject `acceleration[0] = accel_limit[0] * 2.0`
- Each must be caught and reported with the correct `ViolationType`

**Pass criteria**: Validator catches all injected violations. Zero false negatives.

---

## TEST 14: End-to-End Limit Enforcement Pipeline

**Function**: `p0_e2e_limits_plan_to_execution`

For `REPRESENTATIVE_ROBOTS`:
1. Plan: `planner.plan(&start, &goal)` → waypoints
2. Validate waypoints: check all within limits
3. Time-parameterize: `trapezoidal_per_joint(waypoints, vel_limits, accel_limits)`
4. Validate trajectory: `TrajectoryValidator::validate(&traj)`
5. Execute with `RecordingSink`
6. Validate all commands sent

At EVERY stage, joint limits must hold. This is the integration test that proves the full pipeline is safe.

**Pass criteria**: 0 violations at any stage.

---

## SUMMARY

| Test | Robots | Configs | Total Checks | Priority |
|------|--------|---------|--------------|----------|
| Limits valid | 52 | all joints | ~300 | P0 |
| check_limits catches | 52 | 3 | 156 | P0 |
| clamp_to_limits | 52 | 2 | 104 | P0 |
| IK within limits | 52 | 100 | 5,200 | P0 |
| IK unchecked detection | 5 | 30 | 150 | P1 |
| Planner within limits | 52 | 10×20wp | 10,400 | P0 |
| Velocity limits | 5 | 10×1000samples | 50,000 | P0 |
| Acceleration limits | 5 | 10×1000samples | 50,000 | P0 |
| Position interpolation | 5 | 10×2000samples | 100,000 | P0 |
| Continuous joints | ~5 | varies | ~50 | P0 |
| Fuzz | 1 | 10,000 | 10,000 | P0 |
| Executor clamps | 5 | 1 | 5 | P0 |
| Validator catches | 5 | 3 types | 15 | P0 |
| E2E pipeline | 5 | 5 | 25 | P0 |
