//! Motion planners for KINETIC.
//!
//! SIMD-vectorized RRT-Connect (<100µs simple scenes), Cartesian planner,
//! constrained RRT, path shortcutting, and cubic spline smoothing.
//!
//! # Quick Start
//!
//! ```ignore
//! use kinetic_planning::{plan, Planner};
//! use kinetic_core::{Goal, JointValues};
//!
//! // One-liner
//! let result = plan("ur5e", &start, &Goal::Joints(goal))?;
//!
//! // Or with more control
//! let planner = Planner::new(&robot)?;
//! let result = planner.plan(&start, &goal)?;
//! ```

pub mod bi_rrt_star;
pub mod bitrrt;
pub mod cartesian;
pub mod chomp;
pub mod constrained_rrt;
pub mod constraint;
pub mod cost;
pub mod dual_arm;
pub mod est;
pub mod facade;
pub mod gcs;
pub mod iris;
pub mod kpiece;
pub mod pipeline;
pub mod plan_execute;
pub mod prm;
pub mod rrt;
pub mod rrt_star;
pub mod shortcut;
pub mod smooth;
pub mod stomp;

pub use bi_rrt_star::BiRRTStar;
pub use bitrrt::{BiTRRT, BiTRRTConfig};
pub use cartesian::{CartesianConfig, CartesianPlanner, CartesianResult};
pub use chomp::{CHOMPConfig, CHOMP};
pub use constrained_rrt::{ConstrainedPlanningResult, ConstrainedRRT};
pub use dual_arm::{DualArmMode, DualArmPlanner, DualArmResult, DualGoal};
pub use est::{ESTConfig, EST};
pub use facade::PlanningResult;
pub use facade::{plan, plan_with_scene, Planner, PlannerType};
pub use gcs::GCSPlanner;
pub use iris::{ConvexDecomposition, ConvexRegion, IrisConfig};
pub use kpiece::{KPIECEConfig, KPIECE};
pub use plan_execute::{
    PlanExecuteConfig, PlanExecuteLoop, PlanExecuteResult, RecoveryStrategy, ReplanStrategy,
};
pub use prm::PRM;
pub use rrt::RRTConfig;
pub use rrt_star::{RRTStar, RRTStarConfig};
pub use shortcut::{path_length, shortcut, CollisionChecker};
pub use smooth::{smooth_bspline, smooth_cubic_spline, spline_derivatives, SmoothedPath};
pub use stomp::{STOMPConfig, STOMP};
