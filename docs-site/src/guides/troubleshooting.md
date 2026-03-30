# Troubleshooting

Systematic diagnosis of common errors and failures. Every `KineticError`
variant is listed with its message, common cause, and fix.

## IK Failures

### IKNotConverged

**Message:** `IK did not converge after {iterations} iterations (residual: {residual})`

**Common causes:**
- Target pose is at or near the workspace boundary
- Seed configuration is far from the solution
- Solver damping is too high (DLS)

**Fix:**
1. Increase `num_restarts` to 8 or higher
2. Provide a better seed via `IKConfig::with_seed(current_joints)`
3. Lower DLS damping from 0.05 to 0.01
4. Try `IKConfig::with_fallback()` for position-only fallback
5. Verify the target is reachable by checking workspace bounds

### NoIKSolution

**Message:** `No valid IK solution found for target pose`

**Common causes:**
- Target is physically outside the robot's workspace
- Joint limits prevent reaching the target orientation
- Wrong kinematic chain (base_link or tip_link mismatch)

**Fix:**
1. Verify the target is within the robot's reachable workspace
2. Check that `base_link` and `tip_link` in the config match your URDF
3. Use `IKMode::PositionOnly` if orientation does not matter

### SingularityLockup

**Message:** `Singularity lockup: pseudoinverse failed {n} consecutive times`

**Common causes:**
- Robot is at a singular configuration (e.g., arm fully extended)
- Servo mode is driving through a singularity

**Fix:**
1. Move the robot away from the singular configuration manually
2. In servo mode, increase `singularity_damping` (e.g., 0.1)
3. Check `solution.condition_number` -- values above 100 indicate trouble

## Planning Failures

### PlanningTimeout

**Message:** `Planning timed out after {elapsed} ({iterations} iterations)`

**Common causes:**
- Environment is highly constrained (narrow passages)
- Timeout is too short for the problem complexity
- Start or goal is near an obstacle

**Fix:**
1. Increase timeout: `PlannerConfig::offline()` uses 500ms
2. Try EST or KPIECE for narrow passages
3. Verify start and goal are not in collision before planning
4. Reduce `collision_margin` if the robot is being too conservative

### StartInCollision

**Message:** `Start configuration is in collision`

**Common causes:**
- Robot's current joint state touches a scene obstacle
- Self-collision due to tight collision model
- Wrong joint values (e.g., degrees instead of radians)

**Fix:**
1. Check `planner.is_in_collision(&start)` before planning
2. Verify joint values are in radians
3. Add offending link pairs to `skip_pairs` if they are false positives
4. Reduce `collision_margin`

### GoalInCollision / GoalUnreachable

**GoalInCollision:** Target joints collide with the scene. Verify the goal
is collision-free and adjust if needed.

**GoalUnreachable:** Goal joints violate URDF limits or pose has no IK
solution. Check that values are within `robot.joint_limits`.

### PlanningFailed

**Message:** `Planning failed: {reason}`

**Common causes:**
- No collision-free path exists between start and goal
- Planner exhausted all iterations without connecting trees

**Fix:**
1. Increase iterations and timeout
2. Try a different planner (EST, KPIECE for constrained spaces)
3. Decompose the problem into intermediate waypoints

## Collision Issues

### CollisionDetected

**Message:** `Collision detected at waypoint {index}`

**Common causes:**
- Post-planning validation found a collision (shortcutting artifact)
- Scene changed between planning and validation

**Fix:**
1. Replan with the current scene
2. Increase `collision_margin` to build in safety buffer
3. Disable shortcutting if it is creating invalid shortcuts

### PlannerOutputInvalid

**Message:** `Planner output invalid at waypoint {waypoint}: {reason}`

**Common causes:**
- A waypoint violates joint limits (safety gate 1)
- End-effector position exceeds workspace bounds (safety gate 2)

**Fix:**
1. If joint limits: check URDF limit accuracy
2. If workspace bounds: widen bounds or remove `workspace_bounds` config
3. This error indicates a planner bug -- report it if reproducible

## Trajectory Violations

### TrajectoryLimitExceeded

**Message:** `Trajectory limit exceeded at waypoint {index}: {detail}`

**Common causes:**
- Velocity or acceleration exceeds robot limits
- Time parameterization produced infeasible timing

**Fix:**
1. Use `trapezoidal_per_joint` which respects per-joint limits
2. Lower `velocity_scale` in the execution config
3. Increase the time parameterization's speed limits

## Robot Loading Errors

### UrdfParse

**Message:** `URDF parse error: {detail}`

**Common causes:**
- Malformed XML in the URDF file
- Missing required tags (`joint`, `link`, `origin`)
- Encoding issues (non-UTF-8 characters)

**Fix:**
1. Validate your URDF with `check_urdf` (from ROS) or an XML validator
2. Ensure all joints have `type`, `parent`, `child`, `limit` tags

### RobotConfigNotFound

**Message:** `Robot config not found: {name}`

**Common causes:**
- Typo in the robot name
- Config directory not in `robot_configs/`
- Missing `kinetic.toml` in the config directory

**Fix:**
1. Check spelling: `Robot::from_name("ur5e")` (not `UR5e` or `ur-5e`)
2. Verify `robot_configs/{name}/kinetic.toml` exists
3. Use `Robot::from_path()` for custom locations

### DimensionMismatch

**Message:** `Dimension mismatch in {context}: expected {expected}, got {got}`

**Common causes:**
- Wrong number of joint values for the robot
- Mixing full robot DOF with chain DOF
- Array indexing error in user code

**Fix:**
1. Check `robot.dof` and `chain.dof` and match your arrays accordingly
2. For mobile manipulators (Fetch, TIAGo), the chain DOF may be less than robot DOF

**LinkNotFound / JointNotFound:** Typo in link or joint name.
**NamedConfigNotFound:** Pose not in `kinetic.toml` `[named_poses]`.
**ChainExtraction:** Links not connected by joints in the URDF.
