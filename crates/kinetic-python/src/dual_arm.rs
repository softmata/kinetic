//! PyDualArmPlanner — Python bindings for dual-arm planning.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use kinetic_planning::{DualArmMode, DualArmPlanner, DualGoal};

use crate::convert;
use crate::planner::PyGoal;
use crate::robot::PyRobot;
use crate::trajectory::PyTrajectory;

/// Dual-arm motion planner.
///
/// Usage:
///     planner = kinetic.DualArmPlanner(robot, "left_arm", "right_arm")
///     planner = kinetic.DualArmPlanner(robot, "left_arm", "right_arm", mode="synchronized")
///     result = planner.plan(start_left, start_right, goal_left, goal_right)
#[pyclass(name = "DualArmPlanner")]
pub struct PyDualArmPlanner {
    inner: DualArmPlanner,
    left_dof: usize,
    right_dof: usize,
}

#[pymethods]
impl PyDualArmPlanner {
    /// Create a dual-arm planner.
    ///
    /// Args:
    ///     robot: Robot model with multiple planning groups
    ///     left_group: name of the left arm planning group
    ///     right_group: name of the right arm planning group
    ///     mode: "independent", "synchronized", or "coordinated" (default: "synchronized")
    #[new]
    #[pyo3(text_signature = "(robot, left_group, right_group, mode='synchronized')")]
    #[pyo3(signature = (robot, left_group, right_group, mode="synchronized"))]
    fn new(
        robot: &PyRobot,
        left_group: &str,
        right_group: &str,
        mode: &str,
    ) -> PyResult<Self> {
        let arm_mode = match mode.to_lowercase().as_str() {
            "independent" => DualArmMode::Independent,
            "synchronized" | "sync" => DualArmMode::Synchronized,
            "coordinated" | "coord" => DualArmMode::Coordinated,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown mode '{}'. Use: independent, synchronized, coordinated",
                    mode
                )))
            }
        };

        let planner = DualArmPlanner::new(
            robot.inner.clone(),
            left_group,
            right_group,
            arm_mode,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let left_dof = planner.left_dof();
        let right_dof = planner.right_dof();

        Ok(PyDualArmPlanner {
            inner: planner,
            left_dof,
            right_dof,
        })
    }

    /// Plan dual-arm motion.
    ///
    /// Args:
    ///     start_left: numpy array of left arm joint positions
    ///     start_right: numpy array of right arm joint positions
    ///     goal_left: Goal for left arm
    ///     goal_right: Goal for right arm
    ///
    /// Returns:
    ///     dict with 'left_trajectory', 'right_trajectory', 'planning_time', 'tree_size'
    #[pyo3(text_signature = "(self, start_left, start_right, goal_left, goal_right)")]
    fn plan<'py>(
        &self,
        py: Python<'py>,
        start_left: PyReadonlyArray1<'_, f64>,
        start_right: PyReadonlyArray1<'_, f64>,
        goal_left: &PyGoal,
        goal_right: &PyGoal,
    ) -> PyResult<PyObject> {
        let sl = convert::numpy_to_vec(start_left)?;
        let sr = convert::numpy_to_vec(start_right)?;

        let dual_goal = DualGoal {
            left: goal_left.inner.clone(),
            right: goal_right.inner.clone(),
        };

        let result = self
            .inner
            .plan(&sl, &sr, &dual_goal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        let left_traj = PyTrajectory::from_path(result.left_waypoints);
        let right_traj = PyTrajectory::from_path(result.right_waypoints);
        dict.set_item("left_trajectory", left_traj.into_pyobject(py)?)?;
        dict.set_item("right_trajectory", right_traj.into_pyobject(py)?)?;
        dict.set_item("planning_time", result.planning_time.as_secs_f64())?;
        dict.set_item("tree_size", result.tree_size)?;
        Ok(dict.into_any().unbind())
    }

    /// Left arm DOF.
    #[getter]
    fn left_dof(&self) -> usize {
        self.left_dof
    }

    /// Right arm DOF.
    #[getter]
    fn right_dof(&self) -> usize {
        self.right_dof
    }

    fn __repr__(&self) -> String {
        format!(
            "DualArmPlanner(left_dof={}, right_dof={})",
            self.left_dof, self.right_dof
        )
    }
}
