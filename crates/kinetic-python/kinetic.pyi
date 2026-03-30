"""Type stubs for the kinetic Python module (PyO3 bindings)."""

from __future__ import annotations

from typing import Callable

import numpy

# ── Module-level function ──────────────────────────────────────────────

def plan(robot_name: str, start_joints: numpy.ndarray, goal: Goal) -> Trajectory:
    """One-liner motion planning: load robot, plan, return time-parameterized trajectory."""
    ...

# ── Core Types ─────────────────────────────────────────────────────────

class Robot:
    """Robot model loaded from URDF, MJCF, SDF, or built-in config."""

    def __init__(self, name: str) -> None:
        """Load a built-in robot by name (e.g., 'ur5e', 'franka_panda')."""
        ...
    @staticmethod
    def from_urdf(path: str) -> Robot:
        """Load robot from a URDF file."""
        ...
    @staticmethod
    def from_config(name: str) -> Robot:
        """Load a built-in robot by config name (alias for Robot(name))."""
        ...
    @staticmethod
    def from_urdf_srdf(urdf_path: str, srdf_path: str) -> Robot:
        """Load robot from URDF + SRDF (MoveIt-compatible)."""
        ...
    @property
    def name(self) -> str: ...
    @property
    def dof(self) -> int: ...
    @property
    def num_joints(self) -> int: ...
    @property
    def num_links(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    @property
    def velocity_limits(self) -> list[float]: ...
    @property
    def acceleration_limits(self) -> list[float]: ...
    @property
    def named_poses(self) -> list[str]: ...
    def fk(self, joint_values: numpy.ndarray) -> numpy.ndarray:
        """Forward kinematics. Returns 4x4 SE(3) homogeneous matrix."""
        ...
    def jacobian(self, joint_values: numpy.ndarray) -> numpy.ndarray:
        """Compute 6xDOF geometric Jacobian."""
        ...
    def ik(self, target_pose: numpy.ndarray, seed: numpy.ndarray | None = None) -> numpy.ndarray:
        """Inverse kinematics (DLS, 8 restarts). Returns joint values."""
        ...
    def ik_config(
        self,
        target_pose: numpy.ndarray,
        solver: str = "auto",
        mode: str = "full6d",
        seed: numpy.ndarray | None = None,
        null_space: str | None = None,
        max_iterations: int = 300,
        num_restarts: int = 10,
    ) -> dict:
        """IK with full configuration. Returns dict with 'joints', 'converged', 'position_error', etc."""
        ...
    def batch_fk(self, configs: numpy.ndarray) -> list[numpy.ndarray]:
        """Batch FK: (N, DOF) array -> list of N (4,4) pose matrices."""
        ...
    def batch_ik(
        self,
        target_poses: list[numpy.ndarray],
        solver: str = "auto",
        num_restarts: int = 10,
    ) -> list[dict | None]:
        """Batch IK: list of (4,4) targets -> list of solution dicts (None for failures)."""
        ...
    def manipulability(self, joint_values: numpy.ndarray) -> float:
        """Yoshikawa manipulability index (0 = singular)."""
        ...
    def named_pose(self, name: str) -> numpy.ndarray | None:
        """Get a named joint configuration (e.g., 'home')."""
        ...

class Goal:
    """Planning goal specification."""

    @staticmethod
    def joints(values: numpy.ndarray) -> Goal:
        """Joint-space goal (no IK needed)."""
        ...
    @staticmethod
    def pose(target: numpy.ndarray) -> Goal:
        """Cartesian pose goal (4x4 SE3 matrix, IK resolved internally)."""
        ...
    @staticmethod
    def named(name: str) -> Goal:
        """Named pose goal from robot config (e.g., 'home')."""
        ...

class Constraint:
    """Motion constraint for constrained planning."""

    @staticmethod
    def orientation(link: str, axis: list[float], tolerance: float) -> Constraint:
        """Keep link axis within tolerance of specified direction."""
        ...
    @staticmethod
    def position_bound(link: str, axis: str, min: float, max: float) -> Constraint:
        """Bound link position along axis ('x', 'y', or 'z')."""
        ...
    @staticmethod
    def joint(joint_index: int, min: float, max: float) -> Constraint:
        """Restrict joint angle to [min, max] range."""
        ...
    @staticmethod
    def visibility(sensor_link: str, target: list[float], cone_angle: float) -> Constraint:
        """Maintain line-of-sight from sensor to target within cone angle."""
        ...

class CartesianConfig:
    """Configuration for Cartesian (straight-line) planning."""

    def __init__(
        self,
        max_step: float = 0.005,
        jump_threshold: float = 1.4,
        avoid_collisions: bool = True,
        collision_margin: float = 0.02,
    ) -> None: ...

class CartesianResult:
    """Result of Cartesian planning."""

    trajectory: Trajectory
    fraction: float

# ── Planning ───────────────────────────────────────────────────────────

class Planner:
    """Motion planner with optional scene awareness and planner type selection."""

    def __init__(
        self,
        robot: Robot,
        scene: Scene | None = None,
        timeout: float | None = None,
        planner_type: str | None = None,
    ) -> None:
        """Create a planner. planner_type: 'rrt_connect', 'rrt_star', 'bi_rrt_star', 'bitrrt', 'est', 'kpiece', 'prm', 'gcs'."""
        ...
    def plan(
        self,
        start_joints: numpy.ndarray,
        goal: Goal,
        time_parameterize: bool = True,
    ) -> Trajectory:
        """Plan a collision-free path. Auto time-parameterizes by default."""
        ...
    def plan_constrained(
        self,
        start_joints: numpy.ndarray,
        goal: Goal,
        constraints: list[Constraint],
        time_parameterize: bool = True,
    ) -> Trajectory:
        """Plan with motion constraints."""
        ...
    def plan_cartesian(
        self,
        start_joints: numpy.ndarray,
        goal: Goal,
        config: CartesianConfig | None = None,
        time_parameterize: bool = True,
    ) -> CartesianResult:
        """Plan a Cartesian (straight-line) path. Returns trajectory + achieved fraction."""
        ...

class DualArmPlanner:
    """Dual-arm motion planner for bimanual coordination."""

    def __init__(
        self,
        robot: Robot,
        left_group: str,
        right_group: str,
        mode: str = "synchronized",
    ) -> None:
        """Create a dual-arm planner. mode: 'independent', 'synchronized', 'coordinated'."""
        ...
    @property
    def left_dof(self) -> int: ...
    @property
    def right_dof(self) -> int: ...
    def plan(
        self,
        start_left: numpy.ndarray,
        start_right: numpy.ndarray,
        goal_left: Goal,
        goal_right: Goal,
    ) -> dict:
        """Plan dual-arm motion. Returns dict with 'left_trajectory', 'right_trajectory', 'planning_time'."""
        ...

class MoveGroup:
    """MoveIt-style high-level planning interface."""

    def __init__(self, urdf: str) -> None:
        """Create MoveGroup from URDF string."""
        ...
    @property
    def dof(self) -> int: ...
    @property
    def robot_name(self) -> str: ...
    def set_joint_state(self, joints: numpy.ndarray) -> None: ...
    def get_joint_state(self) -> numpy.ndarray: ...
    def set_joint_target(self, target: numpy.ndarray) -> None: ...
    def set_pose_target(self, position: list[float], orientation: list[float]) -> None:
        """Set pose target. orientation is [x, y, z, w] quaternion."""
        ...
    def set_named_target(self, name: str) -> None: ...
    def remember_pose(self, name: str, joints: numpy.ndarray) -> None: ...
    def get_named_poses(self) -> list[str]: ...
    def set_planning_time(self, seconds: float) -> None: ...
    def set_max_velocity_scaling_factor(self, factor: float) -> None: ...
    def set_max_acceleration_scaling_factor(self, factor: float) -> None: ...
    def set_planner_id(self, id: str) -> None: ...
    def plan(self) -> Trajectory | None: ...
    def go(self) -> bool:
        """Plan and execute. Returns True on success."""
        ...
    def stop(self) -> None: ...
    def attach_object(self, object_name: str, link_name: str) -> None: ...
    def detach_object(self, object_name: str) -> None: ...
    def get_attached_objects(self) -> list[tuple[str, str]]: ...
    def get_joint_names(self) -> list[str]: ...
    def get_joint_limits(self) -> list[tuple[float, float]]: ...

# ── Trajectory ─────────────────────────────────────────────────────────

class Trajectory:
    """Time-parameterized joint trajectory."""

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        ...
    @property
    def num_waypoints(self) -> int: ...
    @property
    def dof(self) -> int: ...
    def sample(self, t: float) -> numpy.ndarray:
        """Interpolate joint values at time t (clamped to [0, duration])."""
        ...
    def to_numpy(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Export as (times, positions, velocities) numpy arrays."""
        ...
    def positions(self) -> list[list[float]]:
        """Get all waypoint positions as nested lists."""
        ...
    def validate(
        self,
        position_lower: numpy.ndarray,
        position_upper: numpy.ndarray,
        velocity_limits: numpy.ndarray,
        acceleration_limits: numpy.ndarray,
    ) -> list[dict]:
        """Check trajectory against limits. Returns list of violation dicts (empty if valid)."""
        ...
    def time_parameterize(
        self,
        profile: str,
        velocity_limits: numpy.ndarray,
        acceleration_limits: numpy.ndarray,
        jerk_limits: numpy.ndarray | None = None,
    ) -> Trajectory:
        """Apply time parameterization. Profiles: 'trapezoidal', 'jerk_limited', 'totp', 'cubic_spline'."""
        ...
    def blend(self, other: Trajectory, blend_duration: float) -> Trajectory:
        """Smooth blend between this trajectory and another."""
        ...
    def to_json(self) -> str:
        """Export trajectory as JSON string."""
        ...
    def to_csv(self) -> str:
        """Export trajectory as CSV string."""
        ...
    @staticmethod
    def from_json(json_str: str) -> Trajectory:
        """Import trajectory from JSON string."""
        ...
    @staticmethod
    def from_csv(csv_str: str) -> Trajectory:
        """Import trajectory from CSV string."""
        ...
    def __len__(self) -> int: ...

# ── Scene ──────────────────────────────────────────────────────────────

class Shape:
    """Collision shape for scene objects."""

    @staticmethod
    def cuboid(half_x: float, half_y: float, half_z: float) -> Shape: ...
    @staticmethod
    def sphere(radius: float) -> Shape: ...
    @staticmethod
    def cylinder(radius: float, half_height: float) -> Shape: ...

class Scene:
    """Planning scene with collision objects."""

    def __init__(self, robot: Robot) -> None: ...
    def add(self, name: str, shape: Shape, pose: numpy.ndarray) -> None:
        """Add obstacle with full 4x4 pose."""
        ...
    def remove(self, name: str) -> bool: ...
    def clear(self) -> None: ...
    def attach(self, name: str, shape: Shape, grasp_transform: numpy.ndarray, link_name: str) -> None:
        """Attach object to robot link (e.g., grasped object)."""
        ...
    def detach(self, name: str, place_pose: numpy.ndarray) -> bool:
        """Detach object and place at pose."""
        ...
    def check_collision(self, joint_values: numpy.ndarray) -> bool: ...
    def min_distance(self, joint_values: numpy.ndarray) -> float:
        """Minimum distance to nearest obstacle (meters)."""
        ...
    @property
    def num_objects(self) -> int: ...
    @property
    def num_attached(self) -> int: ...
    @property
    def num_octrees(self) -> int: ...
    def update_octree(self, name: str, points: numpy.ndarray, sensor_origin: numpy.ndarray) -> None: ...
    def add_pointcloud(
        self, name: str, points: numpy.ndarray, sphere_radius: float = 0.01, max_points: int = 100000
    ) -> None: ...
    def update_from_depth(
        self,
        name: str,
        depth: numpy.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_pose: numpy.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        sphere_radius: float = 0.01,
    ) -> None: ...
    def allow_collision(self, name_a: str, name_b: str) -> None: ...
    def disallow_collision(self, name_a: str, name_b: str) -> None: ...

# ── Reactive Control ───────────────────────────────────────────────────

class Policy:
    """Riemannian Motion Policy for reactive control."""

    @staticmethod
    def reach_target(target: numpy.ndarray, gain: float) -> Policy: ...
    @staticmethod
    def avoid_obstacles(scene: Scene, influence_distance: float, gain: float) -> Policy: ...
    @staticmethod
    def avoid_self_collision(gain: float) -> Policy: ...
    @staticmethod
    def joint_limit_avoidance(margin: float, gain: float) -> Policy: ...
    @staticmethod
    def singularity_avoidance(threshold: float, gain: float) -> Policy: ...
    @staticmethod
    def damping(coefficient: float) -> Policy: ...

class RMP:
    """Riemannian Motion Policy combiner (multi-policy blending)."""

    def __init__(self, robot: Robot) -> None: ...
    def add(self, policy: Policy) -> None: ...
    def clear(self) -> None: ...
    @property
    def num_policies(self) -> int: ...
    @property
    def dof(self) -> int: ...
    def compute(self, current_joints: numpy.ndarray, current_velocities: numpy.ndarray, dt: float) -> dict:
        """Blend all policies. Returns dict with 'positions', 'velocities', 'accelerations'."""
        ...

class Servo:
    """Real-time servo teleoperation controller."""

    def __init__(self, robot: Robot, scene: Scene, rate_hz: float = 500.0) -> None: ...
    def send_twist(self, twist: numpy.ndarray) -> dict:
        """Send 6D twist [vx, vy, vz, wx, wy, wz]. Returns dict with 'positions', 'velocities'."""
        ...
    def send_joint_jog(self, joint_index: int, velocity: float) -> dict: ...
    def state(self) -> dict:
        """Get servo state: joint_positions, ee_pose, manipulability, near_singularity, etc."""
        ...
    def set_state(self, positions: numpy.ndarray, velocities: numpy.ndarray) -> None: ...
    @property
    def rate_hz(self) -> float: ...

# ── Dynamics ──────────────────────────────────────────────────────────

class Dynamics:
    """Robot dynamics via featherstone (inverse/forward dynamics, gravity compensation)."""

    def __init__(self, robot: Robot) -> None: ...
    def gravity_compensation(self, joint_positions: numpy.ndarray) -> numpy.ndarray:
        """Torques to hold robot stationary at given configuration."""
        ...
    def inverse_dynamics(
        self, q: numpy.ndarray, qd: numpy.ndarray, qdd: numpy.ndarray
    ) -> numpy.ndarray:
        """Compute torques: tau = M(q)qdd + C(q,qd)qd + g(q)."""
        ...
    def forward_dynamics(
        self, q: numpy.ndarray, qd: numpy.ndarray, tau: numpy.ndarray
    ) -> numpy.ndarray:
        """Compute accelerations: qdd = M^-1(tau - C*qd - g)."""
        ...
    def mass_matrix(self, joint_positions: numpy.ndarray) -> numpy.ndarray:
        """Joint-space mass matrix M(q). Returns (DOF, DOF) symmetric positive-definite."""
        ...

# ── Task Planning ──────────────────────────────────────────────────────

class GripperType:
    """Gripper specification for grasp generation."""

    @staticmethod
    def parallel(max_opening: float, finger_depth: float) -> GripperType: ...
    @staticmethod
    def suction(cup_radius: float) -> GripperType: ...

class Grasp:
    """Generated grasp candidate."""

    quality_score: float
    approach_vector: list[float]
    def pose(self) -> numpy.ndarray:
        """4x4 grasp pose matrix."""
        ...

class GraspGenerator:
    """Geometric grasp generator."""

    def __init__(self, gripper: GripperType) -> None: ...
    def from_shape(
        self,
        shape_type: str,
        dimensions: list[float],
        object_pose: numpy.ndarray,
        num_candidates: int = 100,
    ) -> list[Grasp]: ...

class Approach:
    """Approach/retreat motion specification."""

    def __init__(self, direction: list[float], distance: float) -> None: ...

class TaskSolution:
    """Result of task planning."""

    num_stages: int
    total_duration: float
    total_planning_time: float
    stage_names: list[str]

class Task:
    """High-level task composition (pick, place, move, gripper, sequence)."""

    @staticmethod
    def move_to(robot: Robot, goal: Goal) -> Task: ...
    @staticmethod
    def gripper(width: float) -> Task: ...
    @staticmethod
    def pick(
        robot: Robot,
        scene: Scene,
        object_name: str,
        grasp_poses: list[numpy.ndarray],
        approach: Approach,
        retreat: Approach,
        gripper_open: float = 0.08,
        gripper_close: float = 0.04,
    ) -> Task:
        """Create a pick task with explicit grasp candidates."""
        ...
    @staticmethod
    def place(
        robot: Robot,
        scene: Scene,
        object_name: str,
        target_pose: numpy.ndarray,
        approach: Approach,
        retreat: Approach,
        gripper_open: float = 0.08,
    ) -> Task:
        """Create a place task."""
        ...
    @staticmethod
    def sequence(tasks: list[Task]) -> Task:
        """Compose tasks into a sequence. Consumes the input tasks."""
        ...
    def plan(self, start_joints: numpy.ndarray) -> TaskSolution: ...

# ── Execution ──────────────────────────────────────────────────────────

class SimExecutor:
    """Simulated trajectory executor (instant, no real-time)."""

    def __init__(self, rate_hz: float = 500.0) -> None: ...
    def execute(self, trajectory: Trajectory) -> dict: ...

class LogExecutor:
    """Logging trajectory executor (records all commands)."""

    def __init__(self, rate_hz: float = 500.0) -> None: ...
    def execute(self, trajectory: Trajectory) -> dict: ...
    def commands(self) -> list[dict]: ...
    @property
    def num_commands(self) -> int: ...
    def clear(self) -> None: ...

class RealTimeExecutor:
    """Real-time executor that streams commands to hardware via Python callback."""

    def __init__(
        self,
        rate_hz: float = 500.0,
        position_tolerance: float = 0.1,
        command_timeout_ms: int = 100,
        timeout_factor: float = 2.0,
        require_feedback: bool = False,
    ) -> None: ...
    @staticmethod
    def safe(robot: Robot, rate_hz: float = 500.0) -> RealTimeExecutor:
        """Create a safe executor with joint limit validation and required feedback."""
        ...
    def execute(
        self,
        trajectory: Trajectory,
        command_callback: Callable[[numpy.ndarray, numpy.ndarray], None],
        feedback: Callable[[], list[float] | None] | None = None,
    ) -> dict:
        """Execute trajectory. command_callback(positions, velocities) called at rate_hz.
        Returns dict with state, actual_duration, commands_sent, final_positions, max_deviation."""
        ...

class FrameTree:
    """Coordinate frame transformation tree (TF2-like)."""

    def __init__(self) -> None: ...
    def set_transform(self, parent: str, child: str, transform: numpy.ndarray, timestamp: float) -> None: ...
    def set_static(self, parent: str, child: str, transform: numpy.ndarray) -> None:
        """Set a static calibration transform."""
        ...
    def lookup(self, source: str, target: str) -> numpy.ndarray:
        """Look up transform from source to target frame. Returns 4x4 matrix."""
        ...
    def has_transform(self, parent: str, child: str) -> bool: ...
    def list_frames(self) -> list[str]: ...
    @property
    def num_transforms(self) -> int: ...
    def update_from_robot(self, robot: Robot, joints: numpy.ndarray, timestamp: float) -> None:
        """Populate frame tree from robot FK at given joint configuration."""
        ...
    def clear_dynamic(self) -> None: ...
    def clear(self) -> None: ...

# ── GPU Acceleration ──────────────────────────────────────────────────

class GpuOptimizer:
    """GPU-accelerated trajectory optimizer (cuRobo-style parallel seeds)."""

    def __init__(
        self,
        robot: Robot,
        preset: str = "balanced",
        num_seeds: int | None = None,
        iterations: int | None = None,
    ) -> None:
        """Create optimizer. preset: 'balanced', 'speed', 'quality'. Falls back to CPU if no GPU."""
        ...
    @property
    def is_gpu(self) -> bool:
        """Whether running on GPU (vs CPU fallback)."""
        ...
    @staticmethod
    def gpu_available() -> bool:
        """Check if any GPU is available on this system."""
        ...
    def optimize(
        self,
        start: numpy.ndarray,
        goal: numpy.ndarray,
        scene: Scene | None = None,
        obstacle_spheres: numpy.ndarray | None = None,
    ) -> Trajectory:
        """Optimize trajectory. obstacle_spheres: (N, 4) array of [x, y, z, radius]."""
        ...

class GpuCollisionChecker:
    """GPU-accelerated batch collision checker."""

    def __init__(self, robot: Robot, scene: Scene, resolution: float = 0.02) -> None:
        """Create from scene. Falls back to CPU if no GPU."""
        ...
    @property
    def is_gpu(self) -> bool: ...
    def check_batch(self, configs: numpy.ndarray) -> dict:
        """Check (N, DOF) configs. Returns {'in_collision': list[bool], 'min_distances': ndarray}."""
        ...
    def check_single(self, joint_values: numpy.ndarray) -> tuple[bool, float]:
        """Check single config. Returns (in_collision, min_distance)."""
        ...
