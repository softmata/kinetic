//! **KINETIC** — Fast, Rust-native motion planning for robotics.
//!
//! A complete motion planning stack providing forward/inverse kinematics,
//! SIMD-vectorized collision detection, trajectory generation, and time
//! parameterization.
//!
//! # Quick Start
//!
//! ```ignore
//! use kinetic::prelude::*;
//!
//! let robot = Robot::from_urdf("panda.urdf")?;
//! let planner = Planner::new(&robot)?;
//! let result = planner.plan(&start_joints, &Goal::Joints(goal_joints))?;
//! ```
//!
//! # Crates
//!
//! | Crate | Purpose |
//! |-------|---------|
//! | `kinetic-core` | Pose, JointValues, Trajectory, Goal, Constraint |
//! | `kinetic-robot` | URDF parser, Robot model, config TOML |
//! | `kinetic-kinematics` | FK, Jacobian, IK (DLS, FABRIK, OPW) |
//! | `kinetic-collision` | SIMD sphere collision, CAPT broadphase |
//! | `kinetic-planning` | RRT-Connect, Cartesian, shortcutting, smoothing |
//! | `kinetic-trajectory` | TOTP, trapezoidal, S-curve, spline, blending |
//!
//! # Performance Targets
//!
//! | Operation | Target | Tier |
//! |-----------|--------|------|
//! | FK (6-DOF) | <1 us | V1 |
//! | Jacobian (6-DOF) | <2 us | V1 |
//! | IK (DLS, 7-DOF) | <500 us | V1 |
//! | Collision check (SIMD) | <500 ns | V1 |
//! | RRT-Connect (simple) | <100 us | V1 |

pub mod benchmark;

// Re-export sub-crates as modules
pub use kinetic_collision as collision;
pub use kinetic_core as core;
pub use kinetic_execution as execution;
pub use kinetic_gpu as gpu;
pub use kinetic_grasp as grasp;
pub use kinetic_kinematics as kinematics;
pub use kinetic_planning as planning;
pub use kinetic_reactive as reactive;
pub use kinetic_robot as robot;
pub use kinetic_scene as scene;
pub use kinetic_task as task;
pub use kinetic_trajectory as trajectory;

// Top-level convenience: one-line planning
pub use kinetic_planning::plan;
pub use kinetic_planning::plan_with_scene;

/// Convenient prelude — essentials for getting started quickly.
///
/// For advanced features, import from the sub-crate modules directly:
/// - `kinetic::collision::*` — SIMD collision internals, sphere models
/// - `kinetic::kinematics::*` — OPW, subproblem solvers, Paden-Kahan
/// - `kinetic::trajectory::*` — all time-parameterization algorithms
/// - `kinetic::scene::*` — depth, octree, point cloud processing
/// - `kinetic::reactive::*` — servo, RMP reactive control
/// - `kinetic::task::*` — pick/place task planning
/// - `kinetic::gpu::*` — GPU trajectory optimization
/// - `kinetic::grasp::*` — grasp generation
pub mod prelude {
    // Core types
    pub use kinetic_core::{
        Constraint, Goal, JointValues, KineticError, PlannerConfig, Pose, Result, Trajectory,
        Waypoint,
    };

    // Robot model
    pub use kinetic_robot::{Joint, JointType, Link, Robot, RobotConfig};

    // Kinematics
    #[allow(deprecated)]
    pub use kinetic_kinematics::fk_all_links;
    pub use kinetic_kinematics::{
        forward_kinematics, forward_kinematics_all, jacobian, solve_ik, IKConfig, IKSolution,
        IKSolver, KinematicChain, RobotKinematics,
    };

    // Collision (high-level only)
    pub use kinetic_collision::{AllowedCollisionMatrix, CollisionResult, AABB};

    // Planning
    pub use kinetic_planning::{
        plan, plan_with_scene, DualArmMode, DualArmPlanner, DualGoal, Planner, PlannerType,
        PlanningResult, RRTConfig,
    };

    // Plan-Execute Loop
    pub use kinetic_planning::{
        PlanExecuteConfig, PlanExecuteLoop, PlanExecuteResult, RecoveryStrategy, ReplanStrategy,
    };

    // Execution
    pub use kinetic_execution::{
        CommandSink, ExecutionConfig, ExecutionResult, ExecutionState, FeedbackSource, LogExecutor,
        RealTimeExecutor, SimExecutor, TrajectoryExecutor,
    };

    // Trajectory (most-used algorithms + export)
    pub use kinetic_trajectory::{
        totp, trajectory_from_csv, trajectory_from_json, trajectory_to_csv, trajectory_to_json,
        trapezoidal, TimedTrajectory, TimedWaypoint,
    };

    // Scene
    pub use kinetic_scene::{Scene, SceneObject, Shape};

    // Frame Tree
    pub use kinetic_core::{FrameTree, StampedTransform};

    // GPU (convenience re-exports)
    pub use kinetic_gpu::{CpuOptimizer, GpuConfig, GpuOptimizer, SignedDistanceField};

    // Workspace analysis
    pub use kinetic_kinematics::workspace::{ReachabilityConfig, ReachabilityMap};

    // Math re-exports
    pub use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
}
