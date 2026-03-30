//! Python bindings for trajectory execution and frame tree.

use std::collections::HashMap;

use numpy::PyArray1;
use pyo3::prelude::*;

use kinetic_core::frame_tree::FrameTree;
use kinetic_execution::{
    CommandSink, ExecutionConfig, FeedbackSource, LogExecutor, RealTimeExecutor, SimExecutor,
};

use crate::convert;
use crate::robot::PyRobot;
use crate::trajectory::PyTrajectory;

/// Simulated trajectory executor (instant, no real timing).
///
/// Usage:
///     executor = kinetic.SimExecutor()
///     result = executor.execute(trajectory)
#[pyclass(name = "SimExecutor")]
pub struct PySimExecutor {
    inner: SimExecutor,
}

#[pymethods]
impl PySimExecutor {
    #[new]
    #[pyo3(text_signature = "(rate_hz=500.0)")]
    #[pyo3(signature = (rate_hz=500.0))]
    fn new(rate_hz: f64) -> Self {
        PySimExecutor {
            inner: SimExecutor::new(ExecutionConfig {
                rate_hz,
                ..Default::default()
            }),
        }
    }

    /// Execute a trajectory instantly (no real timing).
    ///
    /// Returns dict with: state, actual_duration, commands_sent, final_positions
    #[pyo3(text_signature = "(self, trajectory)")]
    fn execute<'py>(&self, py: Python<'py>, trajectory: &PyTrajectory) -> PyResult<PyObject> {
        let timed = trajectory
            .timed
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Trajectory not time-parameterized",
                )
            })?;

        let result = self
            .inner
            .validate(timed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        execution_result_to_dict(py, &result)
    }

    fn __repr__(&self) -> String {
        "SimExecutor()".to_string()
    }
}

/// Log executor that records all commands.
///
/// Usage:
///     executor = kinetic.LogExecutor()
///     result = executor.execute(trajectory)
///     commands = executor.commands()  # list of (time, positions, velocities)
#[pyclass(name = "LogExecutor")]
pub struct PyLogExecutor {
    inner: LogExecutor,
}

#[pymethods]
impl PyLogExecutor {
    #[new]
    #[pyo3(text_signature = "(rate_hz=500.0)")]
    #[pyo3(signature = (rate_hz=500.0))]
    fn new(rate_hz: f64) -> Self {
        PyLogExecutor {
            inner: LogExecutor::new(ExecutionConfig {
                rate_hz,
                ..Default::default()
            }),
        }
    }

    /// Execute and record all commands.
    #[pyo3(text_signature = "(self, trajectory)")]
    fn execute<'py>(&mut self, py: Python<'py>, trajectory: &PyTrajectory) -> PyResult<PyObject> {
        let timed = trajectory
            .timed
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Trajectory not time-parameterized",
                )
            })?;

        let result = self
            .inner
            .execute_and_log(timed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        execution_result_to_dict(py, &result)
    }

    /// Get recorded commands as list of dicts.
    ///
    /// Each dict has: time (float), positions (numpy), velocities (numpy)
    #[pyo3(text_signature = "(self)")]
    fn commands<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty(py);
        for cmd in self.inner.commands() {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("time", cmd.time)?;
            dict.set_item("positions", PyArray1::from_slice(py, &cmd.positions))?;
            dict.set_item("velocities", PyArray1::from_slice(py, &cmd.velocities))?;
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Number of recorded commands.
    #[getter]
    fn num_commands(&self) -> usize {
        self.inner.len()
    }

    /// Clear the command log.
    #[pyo3(text_signature = "(self)")]
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!("LogExecutor(commands={})", self.inner.len())
    }
}

/// Bridge: wraps a Python callable as a Rust CommandSink.
struct PyCommandSink {
    callback: Py<PyAny>,
}

impl CommandSink for PyCommandSink {
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
        Python::with_gil(|py| {
            let pos = PyArray1::from_slice(py, positions);
            let vel = PyArray1::from_slice(py, velocities);
            self.callback
                .call1(py, (pos, vel))
                .map_err(|e| format!("Python command callback error: {e}"))?;
            Ok(())
        })
    }
}

/// Bridge: wraps a Python callable as a Rust FeedbackSource.
struct PyFeedbackSource {
    callback: Py<PyAny>,
}

impl FeedbackSource for PyFeedbackSource {
    fn read_positions(&self) -> Option<Vec<f64>> {
        Python::with_gil(|py| {
            let result = self.callback.call0(py).ok()?;
            if result.is_none(py) {
                return None;
            }
            // Try to extract as list/numpy array of floats
            result.extract::<Vec<f64>>(py).ok()
        })
    }
}

/// Real-time trajectory executor that streams commands at precise rates.
///
/// Usage:
///     def send(positions, velocities):
///         my_robot.set_joints(positions)
///
///     executor = kinetic.RealTimeExecutor(rate_hz=500)
///     result = executor.execute(trajectory, send)
///
///     # With feedback for deviation monitoring:
///     def read_feedback():
///         return my_robot.get_joint_positions()  # or None
///
///     result = executor.execute(trajectory, send, feedback=read_feedback,
///                               position_tolerance=0.1)
#[pyclass(name = "RealTimeExecutor")]
pub struct PyRealTimeExecutor {
    rate_hz: f64,
    position_tolerance: f64,
    command_timeout_ms: u64,
    timeout_factor: f64,
    joint_limits: Option<Vec<(f64, f64)>>,
    require_feedback: bool,
}

#[pymethods]
impl PyRealTimeExecutor {
    /// Create a real-time executor.
    ///
    /// Args:
    ///     rate_hz: command rate in Hz (default: 500)
    ///     position_tolerance: max allowed deviation in radians (default: 0.1)
    ///     command_timeout_ms: per-command timeout in ms, 0 to disable (default: 100)
    ///     timeout_factor: abort if execution exceeds duration * factor (default: 2.0)
    ///     require_feedback: require feedback source for safety (default: False)
    #[new]
    #[pyo3(text_signature = "(rate_hz=500.0, position_tolerance=0.1, command_timeout_ms=100, timeout_factor=2.0, require_feedback=False)")]
    #[pyo3(signature = (rate_hz=500.0, position_tolerance=0.1, command_timeout_ms=100, timeout_factor=2.0, require_feedback=false))]
    fn new(
        rate_hz: f64,
        position_tolerance: f64,
        command_timeout_ms: u64,
        timeout_factor: f64,
        require_feedback: bool,
    ) -> Self {
        PyRealTimeExecutor {
            rate_hz,
            position_tolerance,
            command_timeout_ms,
            timeout_factor,
            joint_limits: None,
            require_feedback,
        }
    }

    /// Create a safe executor configured from the robot's limits.
    ///
    /// Enables joint limit validation and requires feedback.
    #[staticmethod]
    #[pyo3(text_signature = "(robot, rate_hz=500.0)")]
    #[pyo3(signature = (robot, rate_hz=500.0))]
    fn safe(robot: &PyRobot, rate_hz: f64) -> Self {
        let limits: Vec<(f64, f64)> = robot
            .inner
            .joint_limits
            .iter()
            .map(|l| (l.lower, l.upper))
            .collect();
        PyRealTimeExecutor {
            rate_hz,
            position_tolerance: 0.1,
            command_timeout_ms: 50,
            timeout_factor: 2.0,
            joint_limits: Some(limits),
            require_feedback: true,
        }
    }

    /// Execute a trajectory, streaming commands to a Python callback.
    ///
    /// Args:
    ///     trajectory: time-parameterized Trajectory
    ///     command_callback: callable(positions: np.array, velocities: np.array)
    ///         Called at rate_hz with interpolated joint commands.
    ///         Raise an exception to abort execution (treated as hardware fault).
    ///     feedback: optional callable() -> Optional[list[float]]
    ///         Returns current actual joint positions, or None if unavailable.
    ///         Used for deviation monitoring when provided.
    ///
    /// Returns:
    ///     dict with: state, actual_duration, expected_duration, commands_sent,
    ///                final_positions, max_deviation (if feedback provided)
    ///
    /// Note: This blocks the Python thread for the duration of the trajectory.
    ///       The command_callback is called from within this blocking loop.
    #[pyo3(text_signature = "(self, trajectory, command_callback, feedback=None)")]
    #[pyo3(signature = (trajectory, command_callback, feedback=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        trajectory: &PyTrajectory,
        command_callback: Py<PyAny>,
        feedback: Option<Py<PyAny>>,
    ) -> PyResult<PyObject> {
        let timed = trajectory.timed.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trajectory not time-parameterized")
        })?;

        let config = ExecutionConfig {
            rate_hz: self.rate_hz,
            position_tolerance: self.position_tolerance,
            command_timeout_ms: self.command_timeout_ms,
            timeout_factor: self.timeout_factor,
            joint_limits: self.joint_limits.clone(),
            require_feedback: self.require_feedback,
            ..Default::default()
        };

        let executor = RealTimeExecutor::new(config);
        let mut sink = PyCommandSink {
            callback: command_callback,
        };

        let feedback_source = feedback.map(|cb| PyFeedbackSource { callback: cb });

        // Release GIL during execution loop — callbacks re-acquire it
        let result = py.allow_threads(|| {
            executor.execute_with_feedback(
                timed,
                &mut sink,
                feedback_source.as_ref().map(|f| f as &dyn FeedbackSource),
            )
        });

        let result =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        execution_result_to_dict(py, &result)
    }

    fn __repr__(&self) -> String {
        format!(
            "RealTimeExecutor(rate_hz={}, tolerance={:.3})",
            self.rate_hz, self.position_tolerance
        )
    }
}

/// Coordinate frame tree for transform management.
///
/// Usage:
///     tree = kinetic.FrameTree()
///     tree.set_transform("world", "base", pose_4x4, timestamp=0.0)
///     cam_pose = tree.lookup("world", "camera")
#[pyclass(name = "FrameTree")]
pub struct PyFrameTree {
    inner: FrameTree,
}

#[pymethods]
impl PyFrameTree {
    #[new]
    #[pyo3(text_signature = "()")]
    fn new() -> Self {
        PyFrameTree {
            inner: FrameTree::new(),
        }
    }

    /// Set a transform between two frames.
    #[pyo3(text_signature = "(self, parent, child, transform, timestamp)")]
    fn set_transform(
        &self,
        parent: &str,
        child: &str,
        transform: numpy::PyReadonlyArray2<'_, f64>,
        timestamp: f64,
    ) -> PyResult<()> {
        let iso = convert::numpy_4x4_to_isometry(transform)?;
        self.inner.set_transform(parent, child, iso, timestamp);
        Ok(())
    }

    /// Set a static calibration transform (not overwritten by FK updates).
    #[pyo3(text_signature = "(self, parent, child, transform)")]
    fn set_static(
        &self,
        parent: &str,
        child: &str,
        transform: numpy::PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let iso = convert::numpy_4x4_to_isometry(transform)?;
        self.inner.set_static_transform(parent, child, iso);
        Ok(())
    }

    /// Look up the transform from source to target frame.
    ///
    /// Returns a 4x4 numpy array. Chains through intermediate frames.
    #[pyo3(text_signature = "(self, source, target)")]
    fn lookup<'py>(
        &self,
        py: Python<'py>,
        source: &str,
        target: &str,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let pose = self
            .inner
            .lookup_transform(source, target)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(convert::isometry_to_numpy_4x4(py, &pose.0))
    }

    /// Check if a transform exists between two frames.
    #[pyo3(text_signature = "(self, parent, child)")]
    fn has_transform(&self, parent: &str, child: &str) -> bool {
        self.inner.has_transform(parent, child)
    }

    /// List all known frame names.
    #[pyo3(text_signature = "(self)")]
    fn list_frames(&self) -> Vec<String> {
        self.inner.list_frames()
    }

    /// Number of stored transforms.
    #[getter]
    fn num_transforms(&self) -> usize {
        self.inner.num_transforms()
    }

    /// Update transforms from robot FK.
    ///
    /// Args:
    ///     robot: Robot model
    ///     joints: current joint positions (numpy array)
    ///     timestamp: current time
    #[pyo3(text_signature = "(self, robot, joints, timestamp)")]
    fn update_from_robot(
        &self,
        robot: &PyRobot,
        joints: numpy::PyReadonlyArray1<'_, f64>,
        timestamp: f64,
    ) -> PyResult<()> {
        let joint_vals = convert::numpy_to_vec(joints)?;

        // Compute FK for all links
        let all_poses = kinetic_kinematics::forward_kinematics_all(&robot.inner, &robot.chain, &joint_vals)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut pose_map = HashMap::new();
        for (link_idx, pose) in all_poses.iter().enumerate() {
            if link_idx < robot.inner.links.len() {
                pose_map.insert(robot.inner.links[link_idx].name.clone(), pose.0);
            }
        }

        self.inner.update_from_fk(&pose_map, timestamp);
        Ok(())
    }

    /// Clear all non-static transforms.
    #[pyo3(text_signature = "(self)")]
    fn clear_dynamic(&self) {
        self.inner.clear_dynamic();
    }

    /// Clear all transforms.
    #[pyo3(text_signature = "(self)")]
    fn clear(&self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameTree(transforms={}, frames={})",
            self.inner.num_transforms(),
            self.inner.list_frames().len()
        )
    }
}

fn execution_result_to_dict<'py>(
    py: Python<'py>,
    result: &kinetic_execution::ExecutionResult,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("state", format!("{:?}", result.state))?;
    dict.set_item("actual_duration", result.actual_duration.as_secs_f64())?;
    dict.set_item("expected_duration", result.expected_duration.as_secs_f64())?;
    dict.set_item("commands_sent", result.commands_sent)?;
    dict.set_item(
        "final_positions",
        PyArray1::from_slice(py, &result.final_positions),
    )?;
    if let Some(dev) = result.max_deviation {
        dict.set_item("max_deviation", dev)?;
    }
    Ok(dict.into_any().unbind())
}

