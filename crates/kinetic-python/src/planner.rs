//! PyPlanner, PyGoal, PyConstraint, and PyCartesianConfig — Python bindings for kinetic motion planning.

use std::time::Duration;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use kinetic_core::math::Axis;
use kinetic_core::{Constraint, Goal, JointValues, PlannerConfig, Pose};
use kinetic_planning::{CartesianConfig, CartesianPlanner, Planner, PlannerType};
use nalgebra::Vector3;

use crate::convert;
use crate::robot::PyRobot;
use crate::scene::PyScene;
use crate::trajectory::PyTrajectory;

/// Goal specification for motion planning.
///
/// Usage:
///     goal = kinetic.Goal.joints([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
///     goal = kinetic.Goal.pose(target_4x4)
///     goal = kinetic.Goal.named("home")
#[pyclass(name = "Goal")]
#[derive(Clone)]
pub struct PyGoal {
    pub(crate) inner: Goal,
}

#[pymethods]
impl PyGoal {
    /// Create a joint-space goal.
    #[staticmethod]
    #[pyo3(text_signature = "(values)")]
    fn joints(values: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let v = convert::numpy_to_vec(values)?;
        Ok(PyGoal {
            inner: Goal::Joints(JointValues(v)),
        })
    }

    /// Create a Cartesian pose goal from a 4x4 matrix.
    #[staticmethod]
    #[pyo3(text_signature = "(target)")]
    fn pose(target: numpy::PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let iso = convert::numpy_4x4_to_isometry(target)?;
        Ok(PyGoal {
            inner: Goal::Pose(Pose(iso)),
        })
    }

    /// Create a named goal (e.g., "home").
    #[staticmethod]
    #[pyo3(text_signature = "(name)")]
    fn named(name: &str) -> Self {
        PyGoal {
            inner: Goal::Named(name.to_string()),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Goal::Joints(jv) => format!("Goal.joints({:?})", jv.0),
            Goal::Pose(_) => "Goal.pose(<4x4>)".to_string(),
            Goal::Named(name) => format!("Goal.named('{}')", name),
            Goal::Relative(v) => format!("Goal.relative([{}, {}, {}])", v.x, v.y, v.z),
        }
    }
}

/// Planning constraint for constrained motion planning.
///
/// Usage:
///     c = kinetic.Constraint.orientation("ee_link", [0, 0, 1], 0.1)
///     c = kinetic.Constraint.position_bound("ee_link", "z", 0.3, 1.5)
///     c = kinetic.Constraint.joint(3, -1.0, 1.0)
///     c = kinetic.Constraint.visibility("camera_link", [1, 0, 0.5], 0.5)
#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    pub(crate) inner: Constraint,
}

#[pymethods]
impl PyConstraint {
    /// Keep a link's orientation within tolerance of a reference axis.
    #[staticmethod]
    #[pyo3(text_signature = "(link, axis, tolerance)")]
    fn orientation(link: &str, axis: [f64; 3], tolerance: f64) -> Self {
        PyConstraint {
            inner: Constraint::orientation(link, Vector3::new(axis[0], axis[1], axis[2]), tolerance),
        }
    }

    /// Keep a link's position bounded along an axis.
    ///
    /// axis: "x", "y", or "z"
    #[staticmethod]
    #[pyo3(text_signature = "(link, axis, min, max)")]
    fn position_bound(link: &str, axis: &str, min: f64, max: f64) -> PyResult<Self> {
        let ax = match axis.to_lowercase().as_str() {
            "x" => Axis::X,
            "y" => Axis::Y,
            "z" => Axis::Z,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "axis must be 'x', 'y', or 'z'",
                ))
            }
        };
        Ok(PyConstraint {
            inner: Constraint::position_bound(link, ax, min, max),
        })
    }

    /// Restrict a joint to a custom range during this motion.
    #[staticmethod]
    #[pyo3(text_signature = "(joint_index, min, max)")]
    fn joint(joint_index: usize, min: f64, max: f64) -> Self {
        PyConstraint {
            inner: Constraint::joint(joint_index, min, max),
        }
    }

    /// Keep a target point visible from a sensor link.
    #[staticmethod]
    #[pyo3(text_signature = "(sensor_link, target, cone_angle)")]
    fn visibility(sensor_link: &str, target: [f64; 3], cone_angle: f64) -> Self {
        PyConstraint {
            inner: Constraint::visibility(
                sensor_link,
                Vector3::new(target[0], target[1], target[2]),
                cone_angle,
            ),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Constraint::Orientation { link, tolerance, .. } => {
                format!("Constraint.orientation('{}', tol={:.3})", link, tolerance)
            }
            Constraint::PositionBound {
                link, axis, min, max, ..
            } => format!(
                "Constraint.position_bound('{}', {:?}, {:.3}, {:.3})",
                link, axis, min, max
            ),
            Constraint::Joint {
                joint_index,
                min,
                max,
            } => format!("Constraint.joint({}, {:.3}, {:.3})", joint_index, min, max),
            Constraint::Visibility {
                sensor_link,
                cone_angle,
                ..
            } => format!(
                "Constraint.visibility('{}', cone={:.3})",
                sensor_link, cone_angle
            ),
        }
    }
}

/// Configuration for Cartesian (straight-line) planning.
///
/// Usage:
///     config = kinetic.CartesianConfig(max_step=0.01)
///     config = kinetic.CartesianConfig(max_step=0.005, jump_threshold=1.4)
#[pyclass(name = "CartesianConfig")]
#[derive(Clone)]
pub struct PyCartesianConfig {
    pub(crate) inner: CartesianConfig,
}

#[pymethods]
impl PyCartesianConfig {
    #[new]
    #[pyo3(text_signature = "(max_step=0.005, jump_threshold=1.4, avoid_collisions=True, collision_margin=0.02)")]
    #[pyo3(signature = (max_step=0.005, jump_threshold=1.4, avoid_collisions=true, collision_margin=0.02))]
    fn new(max_step: f64, jump_threshold: f64, avoid_collisions: bool, collision_margin: f64) -> Self {
        PyCartesianConfig {
            inner: CartesianConfig {
                max_step,
                jump_threshold,
                avoid_collisions,
                collision_margin,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CartesianConfig(max_step={:.4}, jump_threshold={:.2})",
            self.inner.max_step, self.inner.jump_threshold
        )
    }
}

/// Result of a Cartesian planning operation.
///
/// Contains the trajectory and the fraction of the path that was achievable.
#[pyclass(name = "CartesianResult")]
pub struct PyCartesianResult {
    /// The planned trajectory.
    #[pyo3(get)]
    pub trajectory: PyTrajectory,
    /// Fraction of the requested path that was achieved (0.0 to 1.0).
    #[pyo3(get)]
    pub fraction: f64,
}

#[pymethods]
impl PyCartesianResult {
    fn __repr__(&self) -> String {
        format!("CartesianResult(fraction={:.3})", self.fraction)
    }
}

/// Motion planner for a robot.
///
/// Usage:
///     planner = kinetic.Planner(robot)
///     planner = kinetic.Planner(robot, timeout=5.0)
///     planner = kinetic.Planner(robot, scene=scene)
///     traj = planner.plan(start_joints, goal)
#[pyclass(name = "Planner")]
pub struct PyPlanner {
    planner: Planner,
    vel_limits: Vec<f64>,
    accel_limits: Vec<f64>,
}

#[pymethods]
impl PyPlanner {
    /// Create a planner for the given robot.
    ///
    /// Args:
    ///     robot: Robot model
    ///     scene: optional Scene for collision-aware planning
    ///     timeout: optional planning timeout in seconds (default: 5.0)
    ///     planner_type: "rrt_connect", "rrt_star", "bi_rrt_star", "bitrrt", "est", "kpiece", "prm" (default: "rrt_connect")
    #[new]
    #[pyo3(text_signature = "(robot, scene=None, timeout=None, planner_type=None)")]
    #[pyo3(signature = (robot, scene=None, timeout=None, planner_type=None))]
    fn new(
        robot: &PyRobot,
        scene: Option<&PyScene>,
        timeout: Option<f64>,
        planner_type: Option<&str>,
    ) -> PyResult<Self> {
        let mut planner = Planner::new(&robot.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if let Some(scene) = scene {
            planner = planner.with_scene(&scene.inner);
        }

        if let Some(timeout_secs) = timeout {
            let config = PlannerConfig {
                timeout: Duration::from_secs_f64(timeout_secs),
                ..PlannerConfig::default()
            };
            planner = planner.with_config(config);
        }

        if let Some(pt) = planner_type {
            planner = planner.with_planner_type(parse_planner_type(pt)?);
        }

        let vel_limits = robot.inner.velocity_limits();
        let accel_limits = robot.inner.acceleration_limits();

        Ok(PyPlanner {
            planner,
            vel_limits,
            accel_limits,
        })
    }

    /// Plan a trajectory from start_joints to goal.
    ///
    /// Args:
    ///     start_joints: numpy array of starting joint positions
    ///     goal: Goal object (Goal.joints, Goal.pose, or Goal.named)
    ///     time_parameterize: if True (default), returns time-parameterized trajectory
    ///
    /// Returns:
    ///     PyTrajectory with timed waypoints
    #[pyo3(text_signature = "(self, start_joints, goal, time_parameterize=True)")]
    #[pyo3(signature = (start_joints, goal, time_parameterize=true))]
    fn plan(
        &self,
        start_joints: PyReadonlyArray1<'_, f64>,
        goal: &PyGoal,
        time_parameterize: bool,
    ) -> PyResult<PyTrajectory> {
        let start = convert::numpy_to_vec(start_joints)?;

        let result = self
            .planner
            .plan(&start, &goal.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if time_parameterize {
            let timed = kinetic_trajectory::trapezoidal_per_joint(
                &result.waypoints,
                &self.vel_limits,
                &self.accel_limits,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            Ok(PyTrajectory::from_timed(timed))
        } else {
            Ok(PyTrajectory::from_path(result.waypoints))
        }
    }

    /// Plan with constraints (e.g., keep orientation upright).
    ///
    /// Args:
    ///     start_joints: numpy array of starting joint positions
    ///     goal: Goal object
    ///     constraints: list of Constraint objects
    ///     time_parameterize: if True (default), returns time-parameterized trajectory
    ///
    /// Returns:
    ///     PyTrajectory with timed waypoints
    #[pyo3(text_signature = "(self, start_joints, goal, constraints, time_parameterize=True)")]
    #[pyo3(signature = (start_joints, goal, constraints, time_parameterize=true))]
    fn plan_constrained(
        &self,
        start_joints: PyReadonlyArray1<'_, f64>,
        goal: &PyGoal,
        constraints: Vec<PyConstraint>,
        time_parameterize: bool,
    ) -> PyResult<PyTrajectory> {
        let start = convert::numpy_to_vec(start_joints)?;
        let rust_constraints: Vec<Constraint> =
            constraints.into_iter().map(|c| c.inner).collect();

        let constrained = self.planner.with_constraints(&rust_constraints);

        let result = constrained
            .plan(&start, &goal.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if time_parameterize {
            let timed = kinetic_trajectory::trapezoidal_per_joint(
                &result.waypoints,
                &self.vel_limits,
                &self.accel_limits,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            Ok(PyTrajectory::from_timed(timed))
        } else {
            Ok(PyTrajectory::from_path(result.waypoints))
        }
    }

    /// Plan a Cartesian (straight-line) path.
    ///
    /// Args:
    ///     start_joints: numpy array of starting joint positions
    ///     goal: Goal.pose() target
    ///     config: optional CartesianConfig
    ///     time_parameterize: if True (default), returns time-parameterized trajectory
    ///
    /// Returns:
    ///     PyCartesianResult with trajectory and fraction achieved
    #[pyo3(text_signature = "(self, start_joints, goal, config=None, time_parameterize=True)")]
    #[pyo3(signature = (start_joints, goal, config=None, time_parameterize=true))]
    fn plan_cartesian(
        &self,
        start_joints: PyReadonlyArray1<'_, f64>,
        goal: &PyGoal,
        config: Option<&PyCartesianConfig>,
        time_parameterize: bool,
    ) -> PyResult<PyCartesianResult> {
        let start = convert::numpy_to_vec(start_joints)?;

        let target_pose = match &goal.inner {
            Goal::Pose(p) => p.clone(),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cartesian planning requires a Goal.pose() target",
                ))
            }
        };

        let cart_config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();

        let cartesian = CartesianPlanner::new(
            std::sync::Arc::new(self.planner.robot().clone()),
            self.planner.chain().clone(),
        );

        let result = cartesian
            .plan_linear(&start, &target_pose, &cart_config)
            .map_err(|e: kinetic_core::KineticError| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;

        let fraction = result.fraction;
        let traj = if time_parameterize && !result.waypoints.is_empty() {
            let timed = kinetic_trajectory::trapezoidal_per_joint(
                &result.waypoints,
                &self.vel_limits,
                &self.accel_limits,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            PyTrajectory::from_timed(timed)
        } else {
            PyTrajectory::from_path(result.waypoints)
        };

        Ok(PyCartesianResult {
            trajectory: traj,
            fraction,
        })
    }

    fn __repr__(&self) -> String {
        "Planner(<robot>)".to_string()
    }
}

fn parse_planner_type(s: &str) -> PyResult<PlannerType> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PlannerType::Auto),
        "rrt_connect" | "rrtconnect" | "rrt-connect" => Ok(PlannerType::RRTConnect),
        "rrt_star" | "rrtstar" | "rrt*" => Ok(PlannerType::RRTStar),
        "bi_rrt_star" | "birrtstar" | "birrt*" => Ok(PlannerType::BiRRTStar),
        "bitrrt" | "bi_trrt" => Ok(PlannerType::BiTRRT),
        "est" => Ok(PlannerType::EST),
        "kpiece" => Ok(PlannerType::KPIECE),
        "prm" => Ok(PlannerType::PRM),
        "gcs" => Ok(PlannerType::GCS),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown planner type '{}'. Use: rrt_connect, rrt_star, bi_rrt_star, bitrrt, est, kpiece, prm, gcs",
            s
        ))),
    }
}
