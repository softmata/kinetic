# 04 — Trajectory Safety Tests

**Priority**: P0 — a discontinuous trajectory causes violent jerks that can damage the robot or fling objects
**File**: `crates/kinetic/tests/acceptance/test_trajectory_safety.rs`
**Estimated parameterized cases**: ~5,000

---

## PRINCIPLE

A trajectory is a time-stamped sequence of joint positions, velocities, and accelerations.
For safe execution, trajectories must satisfy:
1. **Monotonic time** — time never goes backward
2. **Position continuity** — no teleportation between waypoints
3. **Velocity continuity** — no infinite acceleration spikes
4. **Bounded velocity** — never exceed joint velocity limits
5. **Bounded acceleration** — never exceed joint acceleration limits
6. **Within joint limits** — every position within mechanical stops
7. **Valid dimensions** — all waypoints have correct DOF

---

## TEST 1: Trapezoidal Produces Valid Trajectory

**Function**: `p0_trapezoidal_valid_output`

For `REPRESENTATIVE_ROBOTS` × 10 random plans each:
1. Plan: waypoints = `planner.plan(&start, &goal).waypoints`
2. Time-parameterize: `trapezoidal(&waypoints, 1.0, 2.0)`
3. Validate the output:

```rust
let traj = trapezoidal(&waypoints, 1.0, 2.0).unwrap();

// Duration is positive
assert!(traj.duration > Duration::ZERO, "zero duration");

// At least 2 waypoints
assert!(traj.waypoints.len() >= 2, "fewer than 2 waypoints");

// Monotonic timestamps
assert_trajectory_monotonic(&traj, name);

// All positions are finite
for (i, wp) in traj.waypoints.iter().enumerate() {
    assert_all_finite(&wp.positions.as_slice(), &format!("{} wp{} positions", name, i));
    assert_all_finite(&wp.velocities.as_slice(), &format!("{} wp{} velocities", name, i));
}

// First waypoint matches start
for (j, (&planned, &traj_pos)) in waypoints[0].iter().zip(traj.waypoints[0].positions.as_slice().iter()).enumerate() {
    assert!((planned - traj_pos).abs() < 1e-10,
        "{}: start mismatch joint {}: {} vs {}", name, j, planned, traj_pos);
}

// Last waypoint matches goal
let last = traj.waypoints.last().unwrap();
for (j, (&planned, &traj_pos)) in waypoints.last().unwrap().iter().zip(last.positions.as_slice().iter()).enumerate() {
    assert!((planned - traj_pos).abs() < 1e-10,
        "{}: goal mismatch joint {}: {} vs {}", name, j, planned, traj_pos);
}

// Start and end velocities are zero (robot starts and stops at rest)
for (j, &v) in traj.waypoints[0].velocities.as_slice().iter().enumerate() {
    assert!(v.abs() < 1e-6, "{}: start velocity joint {} = {} (should be ~0)", name, j, v);
}
for (j, &v) in last.velocities.as_slice().iter().enumerate() {
    assert!(v.abs() < 1e-6, "{}: end velocity joint {} = {} (should be ~0)", name, j, v);
}
```

---

## TEST 2: TOTP Produces Valid Trajectory

**Function**: `p0_totp_valid_output`

Same checks as TEST 1, but using `totp(&waypoints, &vel_limits, &accel_limits, 0.01)`.

Additionally:
- TOTP should produce **shorter or equal duration** compared to trapezoidal for the same path
- `totp_duration <= trapezoidal_duration * 1.01` (1% tolerance for numerical differences)

---

## TEST 3: Per-Joint Time Parameterization

**Function**: `p0_per_joint_trajectory_valid`

For `REPRESENTATIVE_ROBOTS`:
1. Get robot's actual velocity and acceleration limits
2. `trapezoidal_per_joint(&waypoints, &vel_limits, &accel_limits)`
3. Sample at 1ms intervals, check EVERY joint at every sample:
   - `|velocity[j]| <= vel_limits[j]`
   - `|acceleration[j]| <= accel_limits[j]`

This uses the ROBOT-SPECIFIC limits, not generic ones.

---

## TEST 4: Position Continuity

**Function**: `p0_trajectory_position_continuity`

For every trajectory produced in TEST 1-3:
1. Between adjacent waypoints, the maximum joint-space step must be bounded
2. `|position[j][i+1] - position[j][i]| / dt <= vel_limits[j] * 1.1` (velocity-implied continuity)

Also check interpolated samples:
```rust
let dt = 0.0005; // 0.5ms — finer than waypoint spacing
let mut prev_positions: Option<Vec<f64>> = None;
let mut t = 0.0;
while t <= traj.duration.as_secs_f64() {
    let sample = traj.sample_at(Duration::from_secs_f64(t));
    if let Some(ref prev) = prev_positions {
        for (j, (&curr, &prev_j)) in sample.positions.iter().zip(prev.iter()).enumerate() {
            let jump = (curr - prev_j).abs();
            assert!(jump < tol::TRAJ_MAX_JUMP,
                "{}: position jump {:.4} rad at t={:.4} joint {}", name, jump, t, j);
        }
    }
    prev_positions = Some(sample.positions.to_vec());
    t += dt;
}
```

---

## TEST 5: Velocity Continuity

**Function**: `p0_trajectory_velocity_continuity`

For trajectories from TEST 1-3:
1. Sample velocities at 1ms intervals
2. Between adjacent samples, acceleration implied by velocity change must be bounded:
   `|(vel[t+dt] - vel[t]) / dt| <= accel_limit * 1.1`
3. No infinite acceleration spikes

```rust
let mut prev_velocities: Option<Vec<f64>> = None;
let mut prev_t: f64 = 0.0;
let dt = 0.001;
let mut t = 0.0;
while t <= traj.duration.as_secs_f64() {
    let sample = traj.sample_at(Duration::from_secs_f64(t));
    if let Some(ref prev_vel) = prev_velocities {
        let delta_t = t - prev_t;
        if delta_t > 1e-10 {
            for (j, (&curr_v, &prev_v)) in sample.velocities.iter().zip(prev_vel.iter()).enumerate() {
                let implied_accel = (curr_v - prev_v).abs() / delta_t;
                assert!(implied_accel <= accel_limits[j] * 1.5,
                    "{}: velocity discontinuity at t={:.4}s joint {}: implied accel {:.2} > limit {:.2}",
                    name, t, j, implied_accel, accel_limits[j]);
            }
        }
    }
    prev_velocities = Some(sample.velocities.to_vec());
    prev_t = t;
    t += dt;
}
```

---

## TEST 6: Jerk-Limited Trajectory

**Function**: `p1_jerk_limited_valid_output`

For `REPRESENTATIVE_ROBOTS`:
1. `jerk_limited(&waypoints, 1.0, 2.0, 10.0)` (max_jerk = 10 rad/s³)
2. Sample at 0.5ms intervals
3. Verify jerk (rate of change of acceleration) is bounded:
   `|(accel[t+dt] - accel[t]) / dt| <= max_jerk * 1.1`

Jerk-limited trajectories are smoother — important for delicate manipulation.

---

## TEST 7: Zero-Length Trajectory (Start == Goal)

**Function**: `p0_zero_length_trajectory`

For `REPRESENTATIVE_ROBOTS`:
1. `waypoints = vec![start.clone(), start.clone()]` (start == goal)
2. `trapezoidal(&waypoints, 1.0, 2.0)`
3. Must NOT panic
4. Duration should be 0 or near-zero
5. Output positions should match start

---

## TEST 8: Single-Waypoint Trajectory

**Function**: `p0_single_waypoint_trajectory`

1. `waypoints = vec![start.clone()]` (just one point)
2. `trapezoidal(&waypoints, 1.0, 2.0)`
3. Must return `Ok` with duration = 0 and 1 waypoint, OR return an appropriate `Err`
4. Must NOT panic

---

## TEST 9: Very Long Path (>500 waypoints)

**Function**: `p1_long_path_trajectory`

1. Generate a path with 500+ waypoints (e.g., by planning multiple segments)
2. Time-parameterize
3. Must succeed without OOM
4. Must satisfy all continuity/limit requirements

---

## TEST 10: Trajectory Sampling Accuracy

**Function**: `p1_trajectory_sampling_at_waypoints`

For trajectories from TEST 1:
1. Sample at each waypoint's exact timestamp
2. Sampled position must match the waypoint's position within 1e-10

```rust
for wp in &traj.waypoints {
    let sample = traj.sample_at(Duration::from_secs_f64(wp.time));
    for (j, (&sampled, &expected)) in sample.positions.iter().zip(wp.positions.as_slice().iter()).enumerate() {
        assert!((sampled - expected).abs() < 1e-10,
            "Sample at t={} joint {}: {} vs {}", wp.time, j, sampled, expected);
    }
}
```

---

## TEST 11: Trajectory Sampling Beyond Duration

**Function**: `p1_trajectory_sampling_beyond_duration`

1. Sample at `traj.duration + 1.0s`
2. Must return the LAST waypoint's positions (clamped, not extrapolated)
3. Must NOT panic or return NaN

---

## TEST 12: Trajectory Sampling at t=0

**Function**: `p0_trajectory_sampling_at_zero`

1. Sample at `t = 0.0`
2. Must return the FIRST waypoint's positions exactly
3. Velocity should be zero (starting from rest)

---

## TEST 13: TrajectoryValidator Comprehensive

**Function**: `p0_validator_catches_all_violation_types`

Create synthetic trajectories with specific violations and verify the validator catches each:

```rust
let robot = load_robot("ur5e");
let vel_limits = robot.velocity_limits();
let accel_limits = robot.acceleration_limits();
let joint_limits = ... // extract lower/upper

let validator = TrajectoryValidator::new(
    &lower_limits, &upper_limits,
    &vel_limits, &accel_limits,
    ValidationConfig::default(),
);

// 1. Position limit violation
let mut bad_traj = valid_traj.clone();
bad_traj.waypoints[1].positions[0] = upper_limits[0] + 0.5; // 0.5 rad past limit
let result = validator.validate(&bad_traj);
assert!(result.is_err());
assert!(result.unwrap_err().iter().any(|v| matches!(v.violation_type, ViolationType::PositionLimit)));

// 2. Velocity limit violation
let mut bad_traj = valid_traj.clone();
bad_traj.waypoints[1].velocities[0] = vel_limits[0] * 2.0;
let result = validator.validate(&bad_traj);
assert!(result.is_err());
assert!(result.unwrap_err().iter().any(|v| matches!(v.violation_type, ViolationType::VelocityLimit)));

// 3. Acceleration limit violation
let mut bad_traj = valid_traj.clone();
bad_traj.waypoints[1].accelerations[0] = accel_limits[0] * 2.0;
let result = validator.validate(&bad_traj);
assert!(result.is_err());
assert!(result.unwrap_err().iter().any(|v| matches!(v.violation_type, ViolationType::AccelerationLimit)));

// 4. Position jump
let mut bad_traj = valid_traj.clone();
bad_traj.waypoints[1].positions[0] += 5.0; // 5 rad jump
let result = validator.validate(&bad_traj);
assert!(result.is_err());
assert!(result.unwrap_err().iter().any(|v| matches!(v.violation_type, ViolationType::PositionJump)));

// 5. Dimension mismatch
let mut bad_traj = valid_traj.clone();
bad_traj.waypoints[1].positions = JointValues::new(vec![0.0]); // wrong DOF
let result = validator.validate(&bad_traj);
assert!(result.is_err());

// 6. Valid trajectory passes
let result = validator.validate(&valid_traj);
assert!(result.is_ok(), "valid trajectory flagged: {:?}", result);
```

---

## TEST 14: Blending Produces Smooth Transitions

**Function**: `p1_blend_smooth_transitions`

1. Create two trajectory segments that meet at a shared waypoint
2. `blend_sequence(&[&seg1, &seg2], 0.1)` (blend radius = 0.1)
3. Verify the blended path has no position discontinuities
4. Verify velocity continuity at the blend point

---

## TEST 15: Cubic Spline Time Parameterization

**Function**: `p1_cubic_spline_valid`

1. `cubic_spline_time(&waypoints, 2.0)` (2 second duration)
2. Sample at 1ms intervals
3. Verify positions are smooth (no jumps)
4. Verify start and end positions match

---

## TEST 16: MANDATORY Validation Gate (CRITICAL FIX)

**Function**: `p0_validation_gate_in_pipeline`

This test verifies that the CRITICAL FIX from `11_CRITICAL_FIXES.md` is implemented.

The planner or the plan-execute pipeline MUST call `TrajectoryValidator::validate()` on every
trajectory before execution. Test this by:

1. Create a planner
2. Plan a trajectory
3. Verify that `validate()` passes on the output
4. (If the pipeline has a validation gate, test that it catches bad trajectories)

Currently this validation is NOT called automatically. The fix is to add it.
After the fix, this test verifies it works.

---

## SUMMARY

| Test | Robots | Trajectories | Samples | Priority |
|------|--------|--------------|---------|----------|
| Trapezoidal valid | 5 | 10 | - | P0 |
| TOTP valid | 5 | 10 | - | P0 |
| Per-joint valid | 5 | 10 | 10k | P0 |
| Position continuity | 5 | 10 | 20k | P0 |
| Velocity continuity | 5 | 10 | 20k | P0 |
| Jerk-limited | 5 | 10 | 20k | P1 |
| Zero-length | 5 | 1 | - | P0 |
| Single waypoint | 5 | 1 | - | P0 |
| Long path | 5 | 1 | - | P1 |
| Sampling accuracy | 5 | 10 | ~200 | P1 |
| Sampling beyond | 5 | 1 | 1 | P1 |
| Sampling at zero | 5 | 1 | 1 | P0 |
| Validator comprehensive | 1 | 6 types | - | P0 |
| Blending | 5 | 2 | - | P1 |
| Cubic spline | 5 | 5 | 5k | P1 |
| Validation gate | 5 | 5 | - | P0 |
