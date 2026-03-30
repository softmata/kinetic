# Glossary

Alphabetical definitions of key terms used throughout kinetic. Each entry gives a concise definition followed by a short explanation. Cross-references point to the concept page where the term is discussed in depth.

---

**ACM (Allowed Collision Matrix)** — A symmetric boolean matrix specifying which link pairs can safely be ignored during collision checking. Adjacent links in the kinematic tree always collide geometrically, so the ACM lets planners skip those known-benign pairs. Built from SRDF `<disable_collisions>` entries. See [Collision Detection](./collision-detection.md).

**C-space (Configuration Space)** — The space of all possible joint-value vectors for a robot. Each point in C-space is a vector of DOF joint values that fully determines the robot's posture. A 6-DOF arm lives in a 6-dimensional C-space. Motion planning searches for collision-free paths through C-space. See [Motion Planning](./motion-planning.md).

**CCD (Continuous Collision Detection)** — Collision checking that considers the swept volume between two configurations, not just the start and end. Prevents the robot from passing through thin obstacles between waypoints. See [Collision Detection](./collision-detection.md).

**Configuration** — A specific set of joint values that fully determines every link's pose. In kinetic, represented as `&[f64]` of length `robot.dof` or as a `JointValues` struct. The zero configuration has all joints at 0; the mid-configuration centers each joint between its limits. See [Forward Kinematics](./forward-kinematics.md).

**Constraint** — A requirement imposed on a planned motion. Position constraints restrict the end-effector to a region of space. Orientation constraints restrict which way it faces. Joint constraints restrict individual joint ranges. In kinetic, the `Constraint` type is used by planners. See [Motion Planning](./motion-planning.md).

**DOF (Degrees of Freedom)** — The number of independent joint values that define a configuration. A typical industrial arm has 6 DOF (exactly enough for arbitrary 6D pose). A 7-DOF arm (e.g., Franka Panda) has one redundant DOF, giving a null space. Stored as `robot.dof`. See [Robots and URDF](./robots-and-urdf.md).

**DLS (Damped Least Squares)** — An iterative IK solver that uses the pseudo-inverse of the Jacobian with an added damping term (Levenberg-Marquardt style). The damping prevents instability near singularities at the cost of slightly slower convergence. In kinetic: `IKSolver::DLS { damping: 0.05 }`. See [Inverse Kinematics](./inverse-kinematics.md).

**EE (End-Effector)** — The functional tip of a kinematic chain, typically where a tool or gripper is attached. Defined by a parent link and an optional grasp-frame offset. In kinetic: `Robot::end_effectors` and the `EndEffector` struct. See [Robots and URDF](./robots-and-urdf.md).

**FABRIK (Forward And Backward Reaching Inverse Kinematics)** — An iterative IK solver that alternates between reaching forward from the base and backward from the target. Works position-first, then refines orientation. Fast and intuitive but less precise on orientation than DLS. In kinetic: `IKSolver::FABRIK`. See [Inverse Kinematics](./inverse-kinematics.md).

**FK (Forward Kinematics)** — Computing the end-effector pose from joint values by chaining transforms through the kinematic tree. The fundamental operation in robotics: given angles, where is the tool? In kinetic: `forward_kinematics()`, `robot.fk()`. See [Forward Kinematics](./forward-kinematics.md).

**FrameTree** — A named-frame graph that stores SE(3) transforms between coordinate frames. Equivalent to ROS2 TF2 but standalone. Supports chain resolution (A to C via A to B to C) and automatic inversion. In kinetic: `FrameTree::new()`, `set_transform()`, `lookup_transform()`. See [Coordinate Frames](./coordinate-frames.md).

**Goal** — The target specification for a motion plan. Can be a Cartesian pose, a joint configuration, a named pose, or a set of constraints. In kinetic, the `Goal` enum specifies what the planner is trying to reach. See [Motion Planning](./motion-planning.md).

**IK (Inverse Kinematics)** — Finding joint values that place the end-effector at a desired pose. The inverse of FK. Often has multiple solutions, no solution, or infinite solutions. In kinetic: `solve_ik()`, `robot.ik()`. See [Inverse Kinematics](./inverse-kinematics.md).

**Isometry3** — The nalgebra type `Isometry3<f64>` representing a rigid-body transform: rotation (as `UnitQuaternion`) plus translation. Preserves distances and angles. The mathematical representation of SE(3) used throughout kinetic. See [Coordinate Frames](./coordinate-frames.md).

**Jacobian** — A 6-by-DOF matrix mapping joint velocities to end-effector spatial velocity. Rows 0-2 are linear velocity (m/s), rows 3-5 are angular velocity (rad/s). Used for IK, manipulability analysis, and velocity control. In kinetic: `jacobian()`. See [Forward Kinematics](./forward-kinematics.md).

**Joint (revolute)** — A joint that rotates around an axis within position limits. The most common joint type in robot arms. Each revolute joint contributes one rotational DOF. In kinetic: `JointType::Revolute`. See [Forward Kinematics](./forward-kinematics.md).

**Joint (prismatic)** — A joint that translates along an axis within position limits. Common in linear stages and Cartesian robots. Each prismatic joint contributes one translational DOF. In kinetic: `JointType::Prismatic`. See [Forward Kinematics](./forward-kinematics.md).

**Joint (continuous)** — A revolute joint with no position limits. It can spin indefinitely (e.g., a wheel). Treated identically to revolute for FK/IK but limit checking is skipped. In kinetic: `JointType::Continuous`. See [Robots and URDF](./robots-and-urdf.md).

**Joint (fixed)** — A non-actuated joint that rigidly attaches two links. Contributes zero DOF. Used for sensor mounts, tool flanges, and structural connections. In kinetic: `JointType::Fixed`. See [Robots and URDF](./robots-and-urdf.md).

**Joint Limits** — Position, velocity, effort, and acceleration bounds on a joint. Position limits define the range of legal values (radians or meters). Stored in `JointLimits` and enforced by planners and IK solvers. See [Robots and URDF](./robots-and-urdf.md).

**KinematicChain** — An ordered subset of joints connecting a base link to a tip link. Extracted from the robot's kinematic tree for FK/IK computation. In kinetic: `KinematicChain::extract(&robot, "base_link", "ee_link")`. See [Forward Kinematics](./forward-kinematics.md).

**Manipulability** — A scalar measuring how dexterous the robot is at a given configuration. Computed as `sqrt(det(J * J^T))` (Yoshikawa's measure). Zero at singularities, high when the robot can move freely in all directions. In kinetic: `manipulability()`. See [Forward Kinematics](./forward-kinematics.md).

**Null Space** — The set of joint velocities that produce zero end-effector velocity. Exists only for redundant robots (DOF > 6). Null-space motion lets the robot reconfigure internally without moving the tool. Used for secondary objectives like joint centering. In kinetic: `NullSpace` enum. See [Inverse Kinematics](./inverse-kinematics.md).

**OPW (Ortho-Parallel-Wrist)** — An analytical IK solver for 6-DOF robots with a spherical wrist (last three axes intersect). Returns up to 8 closed-form solutions in under 50 microseconds. Covers UR, ABB, KUKA, Fanuc, and similar industrial arms. In kinetic: `IKSolver::OPW`. See [Inverse Kinematics](./inverse-kinematics.md).

**Planning Group** — A named subset of joints used for planning. Separates the arm from the gripper, or the left arm from the right. Defined in a TOML config or SRDF file. In kinetic: `PlanningGroup` with `base_link`, `tip_link`, and `joint_indices`. See [Robots and URDF](./robots-and-urdf.md).

**Pose** — A position plus an orientation in 3D space; a point in SE(3). In kinetic, the `Pose` struct wraps `Isometry3<f64>` and provides constructors like `Pose::from_xyz_rpy()` and `Pose::from_xyz_quat()`. See [Coordinate Frames](./coordinate-frames.md).

**Quaternion** — A 4-component representation of 3D rotation (qx, qy, qz, qw). Unit quaternions avoid gimbal lock, interpolate smoothly, and compose efficiently. In kinetic, `UnitQuaternion<f64>` from nalgebra is the primary rotation representation. See [Coordinate Frames](./coordinate-frames.md).

**RMP (Riemannian Motion Policy)** — A reactive control framework that combines multiple motion policies on a Riemannian manifold. Each policy pulls the robot toward a goal, away from obstacles, or along a preferred direction. See [Reactive Control](./reactive-control.md).

**RRT (Rapidly-exploring Random Tree)** — A sampling-based motion planner that grows a tree from the start configuration toward random C-space samples. Probabilistically complete: guaranteed to find a path if one exists, given enough time. In kinetic: multiple RRT variants. See [Motion Planning](./motion-planning.md).

**SE(3)** — The Special Euclidean group in 3D: the set of all rigid-body transforms (rotation + translation). Every pose in the physical world is a point in SE(3). Mathematically, SE(3) = SO(3) x R^3. In kinetic, represented by `Isometry3<f64>` and the `Pose` wrapper. See [Coordinate Frames](./coordinate-frames.md).

**Self-Collision** — A collision between two links of the same robot. Must be checked during planning to prevent the arm from hitting itself. The ACM (from SRDF) disables expected self-collision pairs. See [Collision Detection](./collision-detection.md).

**Singularity** — A configuration where the Jacobian loses rank, meaning the robot cannot move in some direction. At a singularity, manipulability is zero and IK becomes ill-conditioned. Common examples: fully extended arm, wrist axes aligned. See [Forward Kinematics](./forward-kinematics.md), [Inverse Kinematics](./inverse-kinematics.md).

**SRDF (Semantic Robot Description Format)** — An XML companion to URDF that adds planning-level semantics: planning groups, disabled collision pairs, end-effectors, and named poses. Originated from MoveIt. In kinetic: `SrdfModel::from_file()`, `Robot::from_urdf_srdf()`. See [Robots and URDF](./robots-and-urdf.md).

**TOTP (Time-Optimal Trajectory Parameterization)** — An algorithm that computes the fastest timing for a geometric path while respecting velocity and acceleration limits. Turns a path (sequence of waypoints) into a time-parameterized trajectory. See [Trajectory Generation](./trajectory-generation.md).

**Trajectory** — A time-parameterized sequence of waypoints: each waypoint has joint values plus a timestamp. Distinguished from a path (no timing). In kinetic, `Trajectory` stores timed waypoints in struct-of-arrays layout. See [Trajectory Generation](./trajectory-generation.md).

**URDF (Unified Robot Description Format)** — An XML format describing a robot's links, joints, geometry, and limits. The standard robot model format in ROS and the robotics industry. In kinetic: `Robot::from_urdf()`, `Robot::from_urdf_string()`. See [Robots and URDF](./robots-and-urdf.md).

**Waypoint** — A single configuration (joint values) along a planned path. A trajectory is built from waypoints plus timing information. In kinetic, `Waypoint` holds positions, and `TimedWaypoint` adds a timestamp. See [Trajectory Generation](./trajectory-generation.md).

**Workspace** — The set of all positions (and orientations) the end-effector can reach. The workspace is bounded by arm geometry and joint limits. Points outside the workspace have no IK solution. In kinetic: `ReachabilityMap` samples the workspace. See [Inverse Kinematics](./inverse-kinematics.md).

**Bio-IK** — An evolutionary IK solver using a population-based strategy (CMA-ES inspired). Maintains a population of candidate joint configurations, selects the fittest, and mutates. Effective for highly redundant robots (7+ DOF) and multi-objective IK. See [Inverse Kinematics](./inverse-kinematics.md).

**CHOMP (Covariant Hamiltonian Optimization for Motion Planning)** — A trajectory optimizer that minimizes a cost functional over the entire trajectory simultaneously, using covariant gradient descent. Smooths paths while avoiding obstacles. See [Motion Planning](./motion-planning.md).

**Condition Number** — The ratio of the largest to smallest singular value of the Jacobian. Measures proximity to singularity. Below 50 is good; above 100 is marginal; above 1000 is near-singular. Reported in `IKSolution::condition_number`. See [Inverse Kinematics](./inverse-kinematics.md).

**JointValues** — A named vector of joint positions. Wraps `Vec<f64>` with indexing, arithmetic, and interop with nalgebra. Length equals `robot.dof`. The primary type for passing joint configurations through the kinetic API. See [Forward Kinematics](./forward-kinematics.md).

**PRM (Probabilistic Road Map)** — A sampling-based planner that pre-builds a graph of collision-free configurations connected by local paths. Multi-query: the graph is reusable across different start/goal pairs. Efficient when the robot repeatedly plans in the same environment. See [Motion Planning](./motion-planning.md).

**Seed Configuration** — The starting joint values for an iterative IK solver. The solver converges toward the nearest solution to the seed. Choosing a good seed avoids unnecessary arm reconfiguration and helps escape local minima. In kinetic: `IKConfig::seed`. See [Inverse Kinematics](./inverse-kinematics.md).

**STOMP (Stochastic Trajectory Optimization for Motion Planning)** — A trajectory optimizer that explores the cost landscape by sampling noisy trajectories around a seed. Does not require gradient information, making it robust to non-smooth cost functions. See [Motion Planning](./motion-planning.md).

**Twist** — A 6-DOF velocity vector: three linear components (m/s) and three angular components (rad/s). Represents spatial velocity of a rigid body. In kinetic: the `Twist` struct. Related to the Jacobian: `twist = J * joint_velocities`. See [Forward Kinematics](./forward-kinematics.md).

**Wrench** — A 6-DOF force/torque vector: three force components (N) and three torque components (Nm). Represents forces and moments applied to a rigid body. In kinetic: the `Wrench` struct. Dual of Twist under the power pairing. See [Glossary](./glossary.md).

## See Also

- [Robots and URDF](./robots-and-urdf.md) — how robots are loaded and configured
- [Coordinate Frames](./coordinate-frames.md) — SE(3), poses, and the FrameTree
- [Forward Kinematics](./forward-kinematics.md) — computing end-effector poses
- [Inverse Kinematics](./inverse-kinematics.md) — finding joint angles for a target pose
- [Motion Planning](./motion-planning.md) — collision-free path search
- [Trajectory Generation](./trajectory-generation.md) — timing and smoothing of paths
- [Collision Detection](./collision-detection.md) — self-collision and environment checking
- [Reactive Control](./reactive-control.md) — real-time motion policies
