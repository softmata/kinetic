//! Python bindings for KINETIC via PyO3.
//!
//! Provides `import kinetic` with numpy interop for Robot, Planner,
//! Scene, Trajectory, and Servo classes.
//!
//! # Quick Start (Python)
//!
//! ```python
//! import kinetic
//! import numpy as np
//!
//! robot = kinetic.Robot("ur5e")
//! planner = kinetic.Planner(robot)
//! start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
//! goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
//! traj = planner.plan(start, goal)
//!
//! for t in np.linspace(0, traj.duration, 100):
//!     joints = traj.sample(t)
//! ```

mod convert;
mod dual_arm;
mod dynamics;
mod execution;
mod gpu;
mod move_group;
mod planner;
mod robot;
mod scene;
mod servo;
mod task;
mod trajectory;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use kinetic_planning::Planner;
use kinetic_robot::Robot;

use crate::convert::numpy_to_vec;
use crate::planner::PyGoal;
use crate::trajectory::PyTrajectory;

/// One-liner planning function.
///
/// Usage:
///     traj = kinetic.plan("ur5e", start_joints, kinetic.Goal.joints(goal))
#[pyfunction]
fn plan(
    robot_name: &str,
    start_joints: PyReadonlyArray1<'_, f64>,
    goal: &PyGoal,
) -> PyResult<PyTrajectory> {
    let robot = Robot::from_name(robot_name)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let planner = Planner::new(&robot)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let start = numpy_to_vec(start_joints)?;

    let result = planner
        .plan(&start, &goal.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let timed = kinetic_trajectory::trapezoidal_per_joint(&result.waypoints, &vel_limits, &accel_limits)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(PyTrajectory::from_timed(timed))
}

/// The kinetic Python module.
#[pymodule]
fn kinetic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<robot::PyRobot>()?;
    m.add_class::<planner::PyPlanner>()?;
    m.add_class::<planner::PyGoal>()?;
    m.add_class::<planner::PyConstraint>()?;
    m.add_class::<planner::PyCartesianConfig>()?;
    m.add_class::<planner::PyCartesianResult>()?;
    m.add_class::<scene::PyScene>()?;
    m.add_class::<scene::PyShape>()?;
    m.add_class::<trajectory::PyTrajectory>()?;
    m.add_class::<servo::PyServo>()?;
    m.add_class::<servo::PyRMP>()?;
    m.add_class::<servo::PyPolicy>()?;
    m.add_class::<task::PyTask>()?;
    m.add_class::<task::PyTaskSolution>()?;
    m.add_class::<task::PyGraspGenerator>()?;
    m.add_class::<task::PyGripperType>()?;
    m.add_class::<task::PyGrasp>()?;
    m.add_class::<task::PyApproach>()?;
    m.add_class::<execution::PySimExecutor>()?;
    m.add_class::<execution::PyLogExecutor>()?;
    m.add_class::<execution::PyRealTimeExecutor>()?;
    m.add_class::<execution::PyFrameTree>()?;
    m.add_class::<dynamics::PyDynamics>()?;
    m.add_class::<dual_arm::PyDualArmPlanner>()?;
    m.add_class::<move_group::PyMoveGroup>()?;
    m.add_class::<gpu::PyGpuOptimizer>()?;
    m.add_class::<gpu::PyGpuCollisionChecker>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    Ok(())
}
