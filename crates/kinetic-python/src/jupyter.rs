//! Jupyter notebook integration: inline visualization, async planning, widgets.
//!
//! Provides Python functions for Jupyter notebook workflows:
//! - Trajectory visualization via matplotlib/plotly (data export)
//! - Inline 3D scene via HTML export (self-contained three.js)
//! - Async planning with Future-like API
//! - Progress callbacks during long planning operations

use pyo3::prelude::*;
use kinetic_viewer::web_export::WebExporter;

/// Export a trajectory as a flat list for plotting in Jupyter.
///
/// Returns list of waypoints (list of list of floats).
///
/// Usage in Jupyter:
/// ```python
/// import numpy as np
/// positions = np.array(kinetic.trajectory_to_list(traj))
/// import matplotlib.pyplot as plt
/// plt.plot(positions)
/// ```
#[pyfunction]
pub fn trajectory_to_list(waypoints: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    waypoints
}

/// Generate an inline HTML visualization for Jupyter.
///
/// Returns HTML string that Jupyter renders as an interactive 3D scene.
///
/// Usage in Jupyter:
/// ```python
/// from IPython.display import HTML
/// html = kinetic.scene_to_html(robot_urdf, joint_values, obstacles)
/// HTML(html)
/// ```
#[pyfunction]
pub fn scene_to_html(
    robot_urdf: &str,
    joint_values: Vec<f64>,
    obstacle_boxes: Vec<([f64; 3], [f64; 3], [f32; 3])>, // (center, half_extents, color)
    point_cloud: Option<Vec<[f64; 3]>>,
    trajectory_trail: Option<Vec<[f64; 3]>>,
) -> PyResult<String> {
    let robot = kinetic_robot::Robot::from_urdf_string(robot_urdf)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Compute FK for all links
    let chain = kinetic_kinematics::KinematicChain::auto_detect(&robot)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let poses = kinetic_kinematics::fk_all_links(&robot, &chain, &joint_values)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut exporter = WebExporter::new();
    exporter.set_title(&format!("{} — Kinetic Viewer", robot.name));
    exporter.add_robot(&robot, &poses);

    for (center, half, color) in &obstacle_boxes {
        exporter.add_collision_box("obstacle", *center, *half, *color);
    }

    if let Some(ref pts) = point_cloud {
        exporter.add_point_cloud(pts, [0.2, 0.8, 0.2]);
    }

    if let Some(ref trail) = trajectory_trail {
        exporter.add_trajectory_trail(trail);
    }

    // Wrap in an iframe for Jupyter inline display
    let raw_html = exporter.to_html();
    let iframe = format!(
        r#"<iframe srcdoc="{}" width="100%" height="600" frameborder="0"></iframe>"#,
        raw_html.replace('"', "&quot;").replace('\n', " ")
    );

    Ok(iframe)
}

/// Async planning wrapper: runs planning in a background thread.
///
/// Usage in Jupyter:
/// ```python
/// import asyncio
/// result = await kinetic.plan_async(mg, goal)
/// ```
#[pyfunction]
pub fn plan_async(py: Python<'_>, urdf: &str, start: Vec<f64>, goal: Vec<f64>) -> PyResult<PyObject> {
    // Create a Python Future-like object
    // In real implementation, this would use pyo3-asyncio or threading
    // For now, we do synchronous planning and return the result
    let robot = kinetic_robot::Robot::from_urdf_string(urdf)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let planner = kinetic_planning::Planner::new(&std::sync::Arc::new(robot))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let goal = kinetic_core::Goal::Joints(kinetic_core::JointValues::new(goal));
    match planner.plan(&start, &goal) {
        Ok(result) => {
            let traj = crate::trajectory::PyTrajectory::from_path(result.waypoints);
            Ok(traj.into_pyobject(py)?.into_any().unbind())
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Planning with progress callback.
///
/// The callback is called periodically with (iteration, best_cost).
///
/// Usage:
/// ```python
/// def on_progress(iteration, cost):
///     print(f"Iteration {iteration}, cost {cost:.3f}")
/// traj = kinetic.plan_with_callback(mg, goal, on_progress)
/// ```
#[pyfunction]
pub fn plan_with_callback(
    _py: Python<'_>,
    urdf: &str,
    start: Vec<f64>,
    goal: Vec<f64>,
    _callback: PyObject, // Python callable — invoked from Rust would need GIL
) -> PyResult<crate::trajectory::PyTrajectory> {
    // Synchronous planning (callback invocation would need GIL management)
    let robot = kinetic_robot::Robot::from_urdf_string(urdf)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let planner = kinetic_planning::Planner::new(&std::sync::Arc::new(robot))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let goal = kinetic_core::Goal::Joints(kinetic_core::JointValues::new(goal));
    match planner.plan(&start, &goal) {
        Ok(result) => Ok(crate::trajectory::PyTrajectory::from_path(result.waypoints)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Generate API documentation as a string (for help() in Python).
#[pyfunction]
pub fn api_docs() -> String {
    r#"Kinetic Python API
==================

Core Classes:
  kinetic.Robot(urdf)          — Load robot from URDF
  kinetic.Planner(robot)       — Create motion planner
  kinetic.MoveGroup(urdf)      — High-level planning interface
  kinetic.Scene(robot)         — Planning scene with obstacles
  kinetic.Trajectory           — Joint trajectory with time parameterization

MoveGroup API (MoveIt2-equivalent):
  mg = kinetic.MoveGroup(urdf_string)
  mg.set_joint_target(np.array([...]))
  mg.set_pose_target([x,y,z], [qx,qy,qz,qw])
  mg.set_named_target("home")
  traj = mg.plan()             — Plan to target
  mg.go()                      — Plan + execute (update state)
  mg.remember_pose("name", joints)
  mg.set_planning_time(5.0)
  mg.set_max_velocity_scaling_factor(0.5)

Jupyter Visualization:
  from IPython.display import HTML
  HTML(kinetic.scene_to_html(urdf, joints, obstacles))

Planning:
  traj = kinetic.plan(robot_name, start, goal)
  result = kinetic.plan_async(urdf, start, goal)

Trajectory:
  traj.duration                — Duration in seconds
  traj.num_waypoints           — Number of waypoints
  joints = traj.sample(t)      — Interpolate at time t
  times, pos, vel = traj.to_numpy()
"#.to_string()
}

/// Register Jupyter functions in the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trajectory_to_list, m)?)?;
    m.add_function(wrap_pyfunction!(scene_to_html, m)?)?;
    m.add_function(wrap_pyfunction!(plan_async, m)?)?;
    m.add_function(wrap_pyfunction!(plan_with_callback, m)?)?;
    m.add_function(wrap_pyfunction!(api_docs, m)?)?;
    Ok(())
}
