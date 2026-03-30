# Error Reference

Every `KineticError` variant with display message, cause, fix, and code example.

All kinetic operations return `kinetic_core::Result<T>`, which is an alias
for `std::result::Result<T, KineticError>`. The error type is `#[non_exhaustive]`,
so match arms should include a wildcard.

## Error Classification

```rust
let err: KineticError = ...;

// Retryable errors: try again with different parameters
if err.is_retryable() { /* PlanningTimeout, IKNotConverged, CartesianPathIncomplete */ }

// Input errors: fix the input, do not retry
if err.is_input_error() { /* UrdfParse, DimensionMismatch, GoalUnreachable, ... */ }
```

## UrdfParse

**Display:** `URDF parse error: {detail}`

**Cause:** The URDF file contains malformed XML, missing required tags,
or invalid attribute values.

**Fix:** Validate the URDF with an XML linter. Ensure every `<joint>` has
`type`, `<parent>`, `<child>`, `<origin>`, `<axis>`, and `<limit>` tags.

## IKNotConverged

**Display:** `IK did not converge after {iterations} iterations (residual: {residual:.6})`

**Cause:** Iterative solver did not reach tolerance. Common near workspace
boundaries or with poor seeds.

**Fix:** Increase `num_restarts`, provide a better seed, lower damping,
or use `IKConfig::with_fallback()`.

## PlanningTimeout

**Display:** `Planning timed out after {elapsed:?} ({iterations} iterations)`

**Cause:** Planner exhausted time budget. Constrained environments or short timeouts.

**Fix:** Use `PlannerConfig::offline()`, try EST/KPIECE, verify start/goal
are not in collision.

## StartInCollision / GoalInCollision

**StartInCollision:** Start joints collide with scene or self.
Check `planner.is_in_collision(&start)` before planning.

**GoalInCollision:** Resolved goal collides with scene.
Verify goal is collision-free. Try different IK seeds for pose goals.

## GoalUnreachable

**Display:** `Goal is unreachable`

**Cause:** Goal joints violate URDF limits, or IK cannot find a valid
solution for a pose goal within joint limits.

**Fix:** Check that goal values are within `robot.joint_limits`.

## NoIKSolution

**Display:** `No valid IK solution found for target pose`

**Cause:** Target is outside the robot's workspace or no solution exists within limits.

**Fix:** Verify the target is reachable. Use `ReachabilityMap` for workspace analysis.

## JointLimitViolation

**Display:** `Joint '{name}' value {value} outside limits [{min}, {max}]`

**Cause:** A joint value exceeds URDF-defined limits.

**Fix:** Clamp or adjust the offending joint value.

## RobotConfigNotFound

**Display:** `Robot config not found: {name}`

**Cause:** No config directory for the given robot name.

**Fix:** Check spelling. Verify `robot_configs/{name}/kinetic.toml` exists.

## CartesianPathIncomplete

**Display:** `Cartesian path only achieved {fraction:.1}% of requested path`

**Cause:** Cartesian planner hit IK failures, collisions, or joint jumps.

**Fix:** Lower `max_step` in `CartesianConfig` or break into shorter segments.

## CollisionDetected

**Display:** `Collision detected at waypoint {waypoint_index}`

**Fix:** Replan with updated scene. Increase `collision_margin`.

## TrajectoryLimitExceeded

**Display:** `Trajectory limit exceeded at waypoint {waypoint_index}: {detail}`

**Fix:** Re-run time parameterization with correct limits. Lower velocity scaling.

## DimensionMismatch

**Display:** `Dimension mismatch in {context}: expected {expected}, got {got}`

**Cause:** Array length does not match the expected DOF. Common when
mixing full robot DOF with chain DOF on mobile manipulators.

**Fix:** Check `robot.dof` and `chain.dof`. For mobile manipulators
(Fetch, TIAGo), `chain.dof` may be less than `robot.dof`.

```rust
let chain = KinematicChain::extract(&robot, "base_link", "tool0")?;
// Use chain.dof for joint arrays passed to plan/fk/ik
let joints = vec![0.0; chain.dof];
```

## SingularityLockup

**Display:** `Singularity lockup: pseudoinverse failed {consecutive_failures} consecutive times`

**Cause:** The servo controller's pseudoinverse computation failed
repeatedly because the robot is at or near a kinematic singularity.

**Fix:** Move the robot away from the singular configuration. Increase
`singularity_damping` in `ServoConfig`. Check `condition_number` on
IK solutions to detect approaching singularities early.

## PlannerOutputInvalid

**Display:** `Planner output invalid at waypoint {waypoint}: {reason}`

**Cause:** The planner's output failed internal safety validation. Either
a waypoint violates joint limits (safety gate 1) or the end-effector
exits the configured workspace bounds (safety gate 2).

**Fix:** If joint limits: verify URDF limit accuracy. If workspace bounds:
widen the bounds or remove `workspace_bounds` from the config. If this
error persists, it may indicate a planner bug -- please report it.

```rust
match planner.plan(&start, &goal) {
    Err(KineticError::PlannerOutputInvalid { waypoint, reason }) => {
        eprintln!("Safety gate failed at waypoint {waypoint}: {reason}");
    }
    _ => {}
}
```

## NoLinks / LinkNotFound / JointNotFound

**Display:** `Robot has no links` / `Link '{name}' not found` / `Joint '{name}' not found`

**Cause:** Empty URDF, or a typo in a link/joint name string.

**Fix:** Print `robot.links` or `robot.joints` to see available names.

## NamedConfigNotFound

**Display:** `Named configuration '{name}' not found`

**Cause:** The named pose does not exist in `kinetic.toml`.

**Fix:** Check the `[named_poses]` section. Available names can be
queried via `robot.named_pose_names()`.

## Other

**Display:** `{message}`

**Cause:** Catch-all for errors that do not fit a specific variant.

**Fix:** Read the message for context. Common cases include non-finite
joint values (NaN/Infinity) passed to the planner.
