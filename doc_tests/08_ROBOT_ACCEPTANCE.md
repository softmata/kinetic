# 08 — Robot-Specific Acceptance Tests

**Priority**: P0 — each robot has unique geometry, limits, and kinematics that must be validated
**File**: `crates/kinetic/tests/acceptance/test_robot_acceptance.rs`
**Estimated cases**: ~520 (52 robots × 10 tests each)

---

## PRINCIPLE

Every built-in robot configuration must be validated independently. A universal test
may pass on 51/52 robots and fail on the one the researcher is using. These tests catch
robot-specific issues: wrong URDF, swapped axes, incorrect limits, broken chain extraction.

---

## TEST 1: Every Robot Loads Successfully

**Function**: `p0_every_robot_loads`

```rust
#[test]
fn p0_every_robot_loads() {
    let (passed, failed, _) = run_for_all_robots(|name, expected_dof, _| {
        let robot = Robot::from_name(name)
            .map_err(|e| format!("failed to load: {}", e))?;
        if robot.dof != expected_dof {
            return Err(format!("expected {} DOF, got {}", expected_dof, robot.dof));
        }
        Ok(())
    });
    assert_eq!(failed, 0, "{} robots failed to load", failed);
}
```

**Pass criteria**: ALL 52 robots load. Zero failures.

---

## TEST 2: Every Robot Has a Valid Kinematic Chain

**Function**: `p0_every_robot_has_chain`

For every robot:
1. `KinematicChain::auto_detect(&robot)` must succeed
2. Chain DOF must match `arm_chain_dof` from `ALL_ROBOTS`
3. Chain must have a valid base_link and tip_link

```rust
let chain = KinematicChain::auto_detect(&robot)
    .map_err(|e| format!("chain extraction failed: {}", e))?;
if chain.dof != expected_arm_dof {
    return Err(format!("chain DOF {} != expected {}", chain.dof, expected_arm_dof));
}
```

---

## TEST 3: Every Robot Has Reasonable Workspace

**Function**: `p0_every_robot_reasonable_workspace`

For every robot:
1. FK at zero config → must produce a finite pose with reasonable translation
2. Max reach estimate: `||fk(zeros).translation()|| < 5.0` meters (no robot has 5m reach)
3. Min reach: `||fk(zeros).translation()|| > 0.01` meters (must have some extent)

This catches URDF scale errors (meters vs millimeters) and broken transforms.

```rust
let zeros = vec![0.0; chain.dof];
let pose = fk(&robot, &chain, &zeros)
    .map_err(|e| format!("FK at zeros failed: {}", e))?;
let reach = pose.translation().norm();
if reach > 5.0 {
    return Err(format!("unreasonable reach {:.2}m — URDF scale error?", reach));
}
if reach < 0.01 {
    return Err(format!("tiny reach {:.6}m — degenerate URDF?", reach));
}
```

---

## TEST 4: Every Robot FK is Non-Degenerate

**Function**: `p0_fk_non_degenerate`

For every robot:
1. Generate 5 well-separated random configs
2. FK must produce 5 distinct poses (no two within 1mm and 0.01 rad of each other)
3. If all poses are identical, the FK is ignoring joints

---

## TEST 5: Joint Names Are Unique

**Function**: `p0_joint_names_unique`

For every robot:
1. All active joint names must be unique
2. No empty joint names

```rust
let names: Vec<&str> = robot.active_joints.iter()
    .map(|&ji| robot.joints[ji].name.as_str())
    .collect();
let mut seen = std::collections::HashSet::new();
for name in &names {
    assert!(!name.is_empty(), "{}: empty joint name", robot_name);
    assert!(seen.insert(name), "{}: duplicate joint name '{}'", robot_name, name);
}
```

---

## TEST 6: Link Tree Is Acyclic

**Function**: `p0_link_tree_acyclic`

For every robot:
1. Starting from root link, traverse parent→child relationships
2. Must never visit the same link twice (no cycles)
3. All links must be reachable from root

---

## TEST 7: Home Position Is Within Limits

**Function**: `p0_home_within_limits`

For every robot:
1. `robot.home_position()` must have correct DOF
2. Every joint value must be within limits

```rust
let home = robot.home_position();
assert_eq!(home.len(), robot.dof, "{}: home has wrong DOF", name);
for (i, &val) in home.iter().enumerate() {
    let ji = robot.active_joints[i];
    if let Some(ref lim) = robot.joints[ji].limits {
        assert!(val >= lim.lower - 1e-6 && val <= lim.upper + 1e-6,
            "{}: home joint {} = {} outside [{}, {}]",
            name, i, val, lim.lower, lim.upper);
    }
}
```

---

## TEST 8: Mid Configuration Is Within Limits

**Function**: `p0_mid_config_within_limits`

Same as TEST 7 but for `robot.mid_configuration()`.

---

## TEST 9: Named Poses Are Within Limits

**Function**: `p0_named_poses_within_limits`

For every robot:
1. For each named pose in `robot.named_poses`:
   - DOF must match robot.dof
   - All values within joint limits
   - FK must produce a finite pose

---

## TEST 10: Planning Group Validity

**Function**: `p0_planning_group_valid`

For every robot that has planning groups:
1. Each group's `base_link` must exist in the robot
2. Each group's `tip_link` must exist in the robot
3. `joint_indices` must all be valid indices into robot.joints
4. Extracting a chain from the group must succeed

---

## TEST 11: Velocity and Acceleration Limits Are Positive

**Function**: `p0_vel_accel_limits_positive`

For every robot:
1. `robot.velocity_limits()` — all values must be > 0
2. `robot.acceleration_limits()` — all values must be > 0
3. These are used for trajectory parameterization — negative/zero would break it

---

## TEST 12: Known Robot Properties

**Function**: `p0_known_robot_properties`

Verify specific known properties for well-documented robots. These are regression tests
against published datasheets.

```rust
// UR5e: 6 DOF, ±2π joint limits (approximately), 850mm reach
{
    let robot = Robot::from_name("ur5e").unwrap();
    assert_eq!(robot.dof, 6);
    let zeros_pose = fk(&robot, &chain, &vec![0.0; 6]).unwrap();
    let reach = zeros_pose.translation().norm();
    assert!((reach - 0.85).abs() < 0.15,
        "UR5e reach {:.3}m (expected ~0.85m)", reach);
}

// Franka Panda: 7 DOF, ~855mm reach
{
    let robot = Robot::from_name("franka_panda").unwrap();
    assert_eq!(robot.dof, 7);
}

// xArm5: 5 DOF
{
    let robot = Robot::from_name("xarm5").unwrap();
    assert_eq!(robot.dof, 5);
}

// Trossen PX100: 4 DOF
{
    let robot = Robot::from_name("trossen_px100").unwrap();
    assert_eq!(robot.dof, 4);
}
```

---

## TEST 13: Mobile Manipulator DOF Split

**Function**: `p0_mobile_manipulator_dof`

For mobile manipulators (fetch, tiago, pr2):
1. Total robot DOF > arm chain DOF (base has extra joints)
2. Chain extraction produces the ARM chain, not the whole robot
3. Planning uses the arm DOF, not total DOF

```rust
// Fetch: 8 total (1 lift + 7 arm), arm chain = 7
let (robot, chain) = load_robot_and_chain("fetch");
assert_eq!(robot.dof, 8);
assert_eq!(chain.dof, 7);
```

---

## TEST 14: Bimanual Robots Are Independent

**Function**: `p0_bimanual_independent`

For each pair in `BIMANUAL_PAIRS`:
1. Load left and right robots independently
2. They must have the same DOF
3. Their joint limits may differ (asymmetric mounting)
4. FK at identical joints should produce DIFFERENT poses (they're mirrored)

```rust
for &(left_name, right_name) in BIMANUAL_PAIRS {
    let left = Robot::from_name(left_name).unwrap();
    let right = Robot::from_name(right_name).unwrap();
    assert_eq!(left.dof, right.dof,
        "{}/{}: DOF mismatch ({} vs {})", left_name, right_name, left.dof, right.dof);

    let left_chain = KinematicChain::auto_detect(&left).unwrap();
    let right_chain = KinematicChain::auto_detect(&right).unwrap();

    // Same config → different poses (they're different arms)
    let joints = vec![0.1; left.dof];
    let left_pose = fk(&left, &left_chain, &joints).unwrap();
    let right_pose = fk(&right, &right_chain, &joints).unwrap();
    let pos_diff = left_pose.translation_distance(&right_pose);
    assert!(pos_diff > 0.001,
        "{}/{}: identical poses — are they the same URDF?", left_name, right_name);
}
```

---

## TEST 15: IK Solver Auto-Selection

**Function**: `p1_ik_solver_auto_selection`

For every robot:
1. Use `IKSolver::Auto`
2. The auto-selector should pick:
   - OPW for compatible 6-DOF robots
   - Subproblem7DOF for compatible 7-DOF robots
   - DLS as fallback
3. Verify the auto-selected solver matches the robot's `ik_preference` (if set in kinetic.toml)

---

## SUMMARY

| Test | Robots | Priority |
|------|--------|----------|
| Load all | 52 | P0 |
| Chain extraction | 52 | P0 |
| Reasonable workspace | 52 | P0 |
| FK non-degenerate | 52 | P0 |
| Unique joint names | 52 | P0 |
| Acyclic link tree | 52 | P0 |
| Home within limits | 52 | P0 |
| Mid within limits | 52 | P0 |
| Named poses valid | 52 | P0 |
| Planning group valid | ~20 | P0 |
| Vel/accel positive | 52 | P0 |
| Known properties | 4 | P0 |
| Mobile DOF split | 3 | P0 |
| Bimanual independent | 3 pairs | P0 |
| IK auto-selection | 52 | P1 |
