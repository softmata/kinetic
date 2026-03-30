//! Inverse kinematics API — unified interface for all IK solvers.
//!
//! Provides [`solve_ik`] as the main entry point, with configurable
//! solver selection, convergence parameters, and null-space objectives.

use kinetic_core::{KineticError, Pose, Result};
use kinetic_robot::Robot;

use crate::chain::KinematicChain;
use crate::dls;
use crate::fabrik;
use crate::opw;
use crate::subproblem;

/// IK solver selection.
#[derive(Debug, Clone, Default)]
pub enum IKSolver {
    /// Auto-select based on robot DOF and geometry.
    #[default]
    Auto,
    /// Damped Least Squares (Levenberg-Marquardt style).
    DLS {
        /// Damping factor. Higher values → more stable near singularities,
        /// but slower convergence. Default: 0.05.
        damping: f64,
    },
    /// Forward And Backward Reaching IK.
    FABRIK,
    /// OPW analytical solver for 6-DOF spherical-wrist robots.
    /// Returns closed-form solution in ~1 iteration (<5 µs).
    OPW,
    /// Paden-Kahan subproblem decomposition for 6-DOF robots with
    /// intersecting wrist axes. Analytical, returns up to 16 solutions.
    Subproblem,
    /// 7-DOF subproblem decomposition: sweeps the redundant joint and
    /// solves analytically at each sample. `num_samples` controls the
    /// sweep resolution (default 36 = every 10 degrees).
    Subproblem7DOF {
        /// Number of samples across the redundant joint's range.
        num_samples: usize,
    },
}

/// IK solving mode controlling which components of pose error are minimized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IKMode {
    /// Full 6D IK: minimize both position and orientation error.
    #[default]
    Full6D,
    /// Position-only IK: minimize position error, ignore orientation.
    PositionOnly,
    /// Try Full6D first, fall back to PositionOnly if it fails to converge.
    PositionFallback,
}

/// Null-space objective for redundant robots (DOF > 6).
#[derive(Debug, Clone)]
pub enum NullSpace {
    /// Maximize manipulability in the null space.
    Manipulability,
    /// Stay close to seed configuration.
    MinimalDisplacement,
    /// Bias toward mid-range joint values.
    JointCentering,
}

/// Configuration for IK solving.
#[derive(Debug, Clone)]
pub struct IKConfig {
    /// Which solver to use.
    pub solver: IKSolver,
    /// IK mode: Full6D, PositionOnly, or PositionFallback.
    pub mode: IKMode,
    /// Maximum iterations for iterative solvers.
    pub max_iterations: usize,
    /// Position tolerance in meters.
    pub position_tolerance: f64,
    /// Orientation tolerance in radians.
    pub orientation_tolerance: f64,
    /// Whether to enforce joint limits.
    pub check_limits: bool,
    /// Starting configuration for iterative solvers.
    /// If None, uses robot mid-configuration.
    pub seed: Option<Vec<f64>>,
    /// Null-space objective for redundant robots.
    pub null_space: Option<NullSpace>,
    /// Number of random restarts (for escaping local minima).
    pub num_restarts: usize,
}

impl Default for IKConfig {
    fn default() -> Self {
        Self {
            solver: IKSolver::Auto,
            mode: IKMode::Full6D,
            max_iterations: 100,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            check_limits: true,
            seed: None,
            null_space: None,
            num_restarts: 0,
        }
    }
}

impl IKConfig {
    /// Config for DLS solver with default parameters.
    pub fn dls() -> Self {
        Self {
            solver: IKSolver::DLS { damping: 0.05 },
            ..Default::default()
        }
    }

    /// Config for FABRIK solver.
    pub fn fabrik() -> Self {
        Self {
            solver: IKSolver::FABRIK,
            ..Default::default()
        }
    }

    /// Config for OPW analytical solver (6-DOF spherical wrist robots).
    pub fn opw() -> Self {
        Self {
            solver: IKSolver::OPW,
            ..Default::default()
        }
    }

    /// Set seed configuration.
    pub fn with_seed(mut self, seed: Vec<f64>) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set position tolerance.
    pub fn with_position_tolerance(mut self, tol: f64) -> Self {
        self.position_tolerance = tol;
        self
    }

    /// Set number of random restarts.
    pub fn with_restarts(mut self, n: usize) -> Self {
        self.num_restarts = n;
        self
    }

    /// Set IK mode.
    pub fn with_mode(mut self, mode: IKMode) -> Self {
        self.mode = mode;
        self
    }

    /// Shorthand for position-only mode.
    pub fn position_only() -> Self {
        Self {
            mode: IKMode::PositionOnly,
            ..Default::default()
        }
    }

    /// Shorthand for position-fallback mode (try full 6D, then position-only).
    pub fn with_fallback() -> Self {
        Self {
            mode: IKMode::PositionFallback,
            ..Default::default()
        }
    }
}

/// Result of an IK solve attempt.
#[derive(Debug, Clone)]
pub struct IKSolution {
    /// Joint values (length = chain.dof).
    pub joints: Vec<f64>,
    /// Position error (Euclidean distance) in meters.
    pub position_error: f64,
    /// Orientation error in radians.
    pub orientation_error: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations used.
    pub iterations: usize,
    /// Which IK mode produced this solution.
    pub mode_used: IKMode,
    /// Whether the solver used a fallback/degraded path during solving.
    ///
    /// True if DLS solver's LU decomposition failed and it fell back to
    /// scaled transpose (J^T * error * 0.1). Solutions with `degraded=true`
    /// may have lower accuracy near singularities. Check this flag before
    /// executing on real hardware.
    pub degraded: bool,
    /// Jacobian condition number at the solution configuration.
    ///
    /// Ratio of largest to smallest singular value of the 6×N Jacobian.
    /// High values indicate proximity to singularity:
    /// - `< 50`: good, far from singularity
    /// - `50-100`: marginal, approaching singularity
    /// - `> 100`: near-singular, solution may be unreliable
    /// - `> 1000`: at or very near singularity
    ///
    /// Set to `f64::INFINITY` if Jacobian computation fails.
    pub condition_number: f64,
}

/// Solve inverse kinematics.
///
/// Returns the best solution found. Use `config.num_restarts > 0` for
/// multiple attempts with random seeds to escape local minima.
pub fn solve_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    config: &IKConfig,
) -> Result<IKSolution> {
    // Auto-select solver based on robot config preference, DOF, and geometry
    let effective_solver = match &config.solver {
        IKSolver::Auto => {
            // Check robot config's IK preference first
            let preferred = robot
                .ik_preference
                .as_ref()
                .map(|p| p.solver.as_str())
                .unwrap_or("auto");

            match preferred {
                "opw" if opw::is_opw_compatible(robot, chain) => IKSolver::OPW,
                "subproblem" if subproblem::is_subproblem_compatible(robot, chain) => {
                    IKSolver::Subproblem
                }
                "subproblem7dof" if subproblem::is_subproblem_7dof_compatible(robot, chain) => {
                    IKSolver::Subproblem7DOF { num_samples: 36 }
                }
                "fabrik" => IKSolver::FABRIK,
                "dls" => IKSolver::DLS { damping: 0.05 },
                // "auto" or preference didn't validate — use geometry-based selection
                _ => {
                    if opw::is_opw_compatible(robot, chain) {
                        IKSolver::OPW
                    } else if subproblem::is_subproblem_compatible(robot, chain) {
                        IKSolver::Subproblem
                    } else if subproblem::is_subproblem_7dof_compatible(robot, chain) {
                        IKSolver::Subproblem7DOF { num_samples: 36 }
                    } else {
                        IKSolver::DLS { damping: 0.05 }
                    }
                }
            }
        }
        other => other.clone(),
    };

    // Get initial seed
    let initial_seed = config
        .seed
        .clone()
        .unwrap_or_else(|| robot.mid_configuration().to_vec());

    // Extract chain-local seed
    let seed = chain.extract_joint_values(&initial_seed);

    // Determine the effective mode for this solve
    let effective_mode = match config.mode {
        IKMode::PositionFallback => IKMode::Full6D, // try full first
        other => other,
    };

    // Solve with the selected solver — don't short-circuit on error so
    // restarts and fallback solvers get a chance.
    let mut best = match solve_once(
        robot,
        chain,
        target,
        &seed,
        &effective_solver,
        config,
        effective_mode,
    ) {
        Ok(sol) => sol,
        Err(_) => IKSolution {
            joints: seed.to_vec(),
            position_error: f64::INFINITY,
            orientation_error: f64::INFINITY,
            converged: false,
            iterations: 0,
            mode_used: effective_mode,
        degraded: false,
        condition_number: f64::INFINITY,
        },
    };

    if best.converged {
        return Ok(best);
    }

    // For analytical solvers (OPW/Subproblem/Subproblem7DOF), fall back to DLS
    // for restarts since the analytical solver may not suit this robot's geometry.
    let restart_solver = match &effective_solver {
        IKSolver::OPW | IKSolver::Subproblem | IKSolver::Subproblem7DOF { .. } => {
            IKSolver::DLS { damping: 0.05 }
        }
        other => other.clone(),
    };

    // Random restarts with DLS fallback
    if config.num_restarts > 0 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..config.num_restarts {
            // Generate random seed within joint limits
            let random_seed: Vec<f64> = chain
                .active_joints
                .iter()
                .map(|&joint_idx| {
                    let joint = &robot.joints[joint_idx];
                    if let Some(limits) = &joint.limits {
                        let range = limits.upper - limits.lower;
                        if range.is_finite() && range < 100.0 {
                            rng.gen_range(limits.lower..=limits.upper)
                        } else {
                            // Continuous/very-wide joint
                            rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                        }
                    } else {
                        rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                    }
                })
                .collect();

            if let Ok(solution) = solve_once(
                robot,
                chain,
                target,
                &random_seed,
                &restart_solver,
                config,
                effective_mode,
            ) {
                if solution.converged || solution.position_error < best.position_error {
                    best = solution;
                    if best.converged {
                        break;
                    }
                }
            }
        }
    }

    // Position-only fallback: if Full6D failed and mode is PositionFallback,
    // retry with PositionOnly mode
    if !best.converged && config.mode == IKMode::PositionFallback {
        let fallback_result = solve_once(
            robot,
            chain,
            target,
            &seed,
            &restart_solver,
            config,
            IKMode::PositionOnly,
        );
        if let Ok(sol) = fallback_result {
            if sol.converged || sol.position_error < best.position_error {
                best = sol;
            }
        }
    }

    if !best.converged {
        return Err(KineticError::IKNotConverged {
            iterations: best.iterations,
            residual: best.position_error,
        });
    }

    // Compute Jacobian condition number at solution
    best.condition_number = match crate::jacobian(robot, chain, &best.joints) {
        Ok(j) => {
            let svd = j.svd(false, false);
            let singular_values = &svd.singular_values;
            if singular_values.is_empty() {
                f64::INFINITY
            } else {
                let s_max = singular_values.iter().copied().fold(0.0_f64, f64::max);
                let s_min = singular_values.iter().copied().fold(f64::INFINITY, f64::min);
                if s_min > 1e-10 {
                    // Normal case: well-conditioned Jacobian
                    s_max / s_min
                } else if s_max > 1e-10 {
                    // Near-singular: smallest SV is near zero but largest isn't
                    // Report a very high but finite condition number
                    s_max / 1e-10
                } else {
                    // Degenerate: all singular values near zero
                    f64::INFINITY
                }
            }
        }
        Err(_) => f64::INFINITY,
    };

    Ok(best)
}

/// Batch IK: solve inverse kinematics for multiple target poses.
///
/// Returns `Vec<Option<IKSolution>>` — `None` for targets where IK failed.
/// Does not short-circuit on failure; all targets are attempted.
///
/// Uses the same config for all targets. If you need different configs per
/// target, call [`solve_ik`] individually.
pub fn solve_ik_batch(
    robot: &Robot,
    chain: &KinematicChain,
    targets: &[Pose],
    config: &IKConfig,
) -> Vec<Option<IKSolution>> {
    targets
        .iter()
        .map(|target| solve_ik(robot, chain, target, config).ok())
        .collect()
}

/// Single IK solve attempt.
fn solve_once(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    solver: &IKSolver,
    config: &IKConfig,
    mode: IKMode,
) -> Result<IKSolution> {
    match solver {
        IKSolver::DLS { damping } => {
            dls::solve_dls(robot, chain, target, seed, *damping, config, mode)
        }
        IKSolver::FABRIK => {
            // FABRIK is inherently position-focused; pass through mode for reporting
            let mut sol = fabrik::solve_fabrik(robot, chain, target, seed, config)?;
            sol.mode_used = mode;
            if mode == IKMode::PositionOnly {
                // For position-only, consider converged if position is within tolerance
                sol.converged = sol.position_error < config.position_tolerance;
            }
            Ok(sol)
        }
        IKSolver::OPW => {
            let mut sol = opw::solve_opw_ik(robot, chain, target, seed, config)?;
            sol.mode_used = mode;
            Ok(sol)
        }
        IKSolver::Subproblem => {
            let mut sol =
                subproblem::solve_subproblem_ik_as_solution(robot, chain, target, seed, config)?;
            sol.mode_used = mode;
            Ok(sol)
        }
        IKSolver::Subproblem7DOF { num_samples } => {
            let mut sol = subproblem::solve_subproblem_7dof_ik(
                robot,
                chain,
                target,
                seed,
                config,
                *num_samples,
            )?;
            sol.mode_used = mode;
            Ok(sol)
        }
        IKSolver::Auto => unreachable!("Auto should be resolved before calling solve_once"),
    }
}

/// Compute the 6D pose error between current and target.
///
/// Returns (position_error, orientation_error, error_vector_6d).
pub(crate) fn pose_error(current: &Pose, target: &Pose) -> (f64, f64, nalgebra::DVector<f64>) {
    let pos_err = target.translation() - current.translation();
    let pos_error_norm = pos_err.norm();

    // Orientation error as axis-angle
    let rot_err = target.rotation() * current.rotation().inverse();
    let angle = rot_err.angle();
    let orientation_error = angle;

    // Build 6D error vector [linear; angular]
    let mut error = nalgebra::DVector::zeros(6);
    error[0] = pos_err.x;
    error[1] = pos_err.y;
    error[2] = pos_err.z;

    if angle > 1e-10 {
        let axis = rot_err
            .axis()
            .unwrap_or(nalgebra::Unit::new_normalize(nalgebra::Vector3::z()));
        error[3] = axis.x * angle;
        error[4] = axis.y * angle;
        error[5] = axis.z * angle;
    }

    (pos_error_norm, orientation_error, error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::forward_kinematics;

    const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

    fn test_robot_and_chain() -> (Robot, KinematicChain) {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        (robot, chain)
    }

    #[test]
    fn ik_roundtrip_dls() {
        let (robot, chain) = test_robot_and_chain();
        let original_joints = vec![0.3, 0.5, -0.2];

        // FK to get target pose
        let target = forward_kinematics(&robot, &chain, &original_joints).unwrap();

        // IK to recover joints
        let config = IKConfig::dls().with_seed(vec![0.0, 0.0, 0.0]);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();

        assert!(
            solution.converged,
            "DLS should converge: pos_err={}, orient_err={}",
            solution.position_error, solution.orientation_error
        );
        assert!(
            solution.position_error < 1e-3,
            "Position error too large: {}",
            solution.position_error
        );
    }

    #[test]
    fn ik_roundtrip_fabrik() {
        let (robot, chain) = test_robot_and_chain();
        let original_joints = vec![0.3, 0.5, -0.2];

        let target = forward_kinematics(&robot, &chain, &original_joints).unwrap();

        let config = IKConfig::fabrik()
            .with_seed(vec![0.0, 0.0, 0.0])
            .with_max_iterations(200);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();

        assert!(
            solution.converged,
            "FABRIK should converge: pos_err={}",
            solution.position_error
        );
    }

    #[test]
    fn ik_auto_solver() {
        let (robot, chain) = test_robot_and_chain();
        let target = forward_kinematics(&robot, &chain, &[0.5, -0.3, 0.8]).unwrap();

        let config = IKConfig::default().with_seed(vec![0.0, 0.0, 0.0]);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
        assert!(solution.converged);
    }

    #[test]
    fn ik_with_restarts() {
        let (robot, chain) = test_robot_and_chain();
        let target = forward_kinematics(&robot, &chain, &[1.5, -1.0, 0.5]).unwrap();

        let config = IKConfig::dls()
            .with_seed(vec![0.0, 0.0, 0.0])
            .with_restarts(5);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
        assert!(solution.converged);
    }

    #[test]
    fn ik_unreachable_target() {
        let (robot, chain) = test_robot_and_chain();
        // Target far beyond robot reach
        let target = Pose::from_xyz(10.0, 10.0, 10.0);

        let config = IKConfig::dls();
        let result = solve_ik(&robot, &chain, &target, &config);
        assert!(result.is_err());
    }

    #[test]
    fn ik_solution_within_limits() {
        let (robot, chain) = test_robot_and_chain();
        let target = forward_kinematics(&robot, &chain, &[0.5, 0.5, 0.5]).unwrap();

        let config = IKConfig::dls().with_seed(vec![0.0, 0.0, 0.0]);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();

        // Verify joints are within limits
        for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                assert!(
                    solution.joints[i] >= limits.lower - 1e-6
                        && solution.joints[i] <= limits.upper + 1e-6,
                    "Joint {} value {} outside limits [{}, {}]",
                    i,
                    solution.joints[i],
                    limits.lower,
                    limits.upper
                );
            }
        }
    }

    #[test]
    fn ik_position_only_mode() {
        let (robot, chain) = test_robot_and_chain();
        let target = forward_kinematics(&robot, &chain, &[0.5, 0.5, 0.5]).unwrap();

        let config = IKConfig::position_only().with_seed(vec![0.0, 0.0, 0.0]);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();

        assert!(
            solution.converged,
            "Position-only IK should converge: pos_err={}",
            solution.position_error
        );
        assert_eq!(solution.mode_used, IKMode::PositionOnly);
    }

    #[test]
    fn ik_fallback_mode_converges_full6d() {
        let (robot, chain) = test_robot_and_chain();
        let target = forward_kinematics(&robot, &chain, &[0.3, 0.5, -0.2]).unwrap();

        // Easy target — should converge in Full6D without needing fallback
        let config = IKConfig::with_fallback().with_seed(vec![0.0, 0.0, 0.0]);
        let solution = solve_ik(&robot, &chain, &target, &config).unwrap();

        assert!(solution.converged);
        assert_eq!(solution.mode_used, IKMode::Full6D);
    }

    #[test]
    fn ik_mode_default_is_full6d() {
        let config = IKConfig::default();
        assert_eq!(config.mode, IKMode::Full6D);
    }

    /// Gap 10: For a 7-DOF robot (not OPW-compatible), verify that Auto solver
    /// selection falls back to DLS (not OPW) and still produces a valid solution.
    /// OPW is designed for 6-DOF spherical wrist robots only.
    #[test]
    fn ik_auto_fallback_to_dls_on_7dof() {
        // Load a 7-DOF robot — OPW is designed for 6-DOF spherical wrist only
        let robot = Robot::from_name("franka_panda").unwrap();
        assert_eq!(robot.dof, 7, "Panda should be 7-DOF");

        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        assert_eq!(chain.dof, 7, "Chain should have 7 DOF");

        // Verify OPW is NOT compatible with this 7-DOF robot
        assert!(
            !crate::opw::is_opw_compatible(&robot, &chain),
            "OPW should not be compatible with 7-DOF Panda"
        );

        // Pick a reachable target via FK
        let known_joints = vec![0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854];
        let target = forward_kinematics(&robot, &chain, &known_joints).unwrap();

        // Use Auto solver — it should detect 7-DOF and select DLS (not OPW).
        // With restarts, it should find a solution.
        let config = IKConfig {
            solver: IKSolver::Auto,
            num_restarts: 5,
            ..Default::default()
        };

        let result = solve_ik(&robot, &chain, &target, &config);

        match result {
            Ok(solution) => {
                // DLS fallback succeeded — verify solution quality
                assert!(
                    solution.converged,
                    "Auto solver should converge for reachable target: pos_err={}, orient_err={}",
                    solution.position_error,
                    solution.orientation_error
                );
                assert!(
                    solution.position_error < 0.01,
                    "Position error should be small: {}",
                    solution.position_error
                );
                assert_eq!(
                    solution.joints.len(),
                    chain.dof,
                    "Solution should have {} joints",
                    chain.dof
                );
            }
            Err(e) => {
                panic!(
                    "Auto IK on 7-DOF Panda with reachable target should succeed, got: {}",
                    e
                );
            }
        }
    }
}
