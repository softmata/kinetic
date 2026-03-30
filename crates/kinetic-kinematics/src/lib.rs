//! Forward and inverse kinematics for KINETIC.
//!
//! Provides FK, Jacobian computation, kinematic chain extraction,
//! batch FK for planning, and IK solvers (DLS, FABRIK).

pub mod chain;
pub mod dls;
pub mod fabrik;
pub mod forward;
pub mod ik;
pub mod opw;
pub mod paden_kahan;
pub mod subproblem;
pub mod workspace;

pub use chain::KinematicChain;
#[allow(deprecated)]
pub use forward::fk_all_links;
pub use forward::{fk, fk_batch, forward_kinematics, forward_kinematics_all, jacobian, manipulability};
pub use ik::{solve_ik, solve_ik_batch, IKConfig, IKMode, IKSolution, IKSolver, NullSpace};
pub use opw::{is_opw_compatible, solve_opw, solve_opw_ik, OPWParameters};
pub use paden_kahan::{
    axis_angle_rotation, euler_zyz_decompose, subproblem1, subproblem2, subproblem3,
};
pub use subproblem::{
    is_subproblem_7dof_compatible, is_subproblem_compatible, solve_subproblem_7dof_ik,
    solve_subproblem_ik, SubproblemIK, SubproblemIK7DOF,
};
pub use workspace::{ReachabilityConfig, ReachabilityMap};

use kinetic_core::{Pose, Result};
use kinetic_robot::Robot;

/// Convenience extension trait adding FK/IK methods directly on [`Robot`].
///
/// Auto-detects the kinematic chain from planning groups or URDF tree.
///
/// ```ignore
/// use kinetic::prelude::*;
/// let robot = Robot::from_name("ur5e")?;
/// let pose = robot.fk(&joints)?;
/// let solution = robot.ik(&pose)?;
/// ```
pub trait RobotKinematics {
    /// Compute forward kinematics for the robot's default kinematic chain.
    fn fk(&self, joints: &[f64]) -> Result<Pose>;

    /// Compute forward kinematics for all links along the chain.
    fn fk_all(&self, joints: &[f64]) -> Result<Vec<Pose>>;

    /// Solve inverse kinematics for the robot's default kinematic chain.
    ///
    /// Uses DLS with 8 random restarts. For more control, use
    /// [`ik_config`](Self::ik_config) or [`solve_ik`] with a custom [`IKConfig`].
    fn ik(&self, target: &Pose) -> Result<Vec<f64>>;

    /// Solve inverse kinematics with a custom configuration.
    ///
    /// Returns the full [`IKSolution`] with convergence info, condition number, etc.
    fn ik_config(&self, target: &Pose, config: &IKConfig) -> Result<IKSolution>;
}

impl RobotKinematics for Robot {
    fn fk(&self, joints: &[f64]) -> Result<Pose> {
        let chain = KinematicChain::auto_detect(self)?;
        forward_kinematics(self, &chain, joints)
    }

    fn fk_all(&self, joints: &[f64]) -> Result<Vec<Pose>> {
        let chain = KinematicChain::auto_detect(self)?;
        forward_kinematics_all(self, &chain, joints)
    }

    fn ik(&self, target: &Pose) -> Result<Vec<f64>> {
        let chain = KinematicChain::auto_detect(self)?;
        let config = IKConfig {
            num_restarts: 8,
            ..Default::default()
        };
        solve_ik(self, &chain, target, &config).map(|sol| sol.joints)
    }

    fn ik_config(&self, target: &Pose, config: &IKConfig) -> Result<IKSolution> {
        let chain = KinematicChain::auto_detect(self)?;
        solve_ik(self, &chain, target, config)
    }
}
