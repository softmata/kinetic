//! PyTrajectory — Python bindings for kinetic timed trajectories.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use kinetic_trajectory::{TimedTrajectory, TrajectoryValidator, ValidationConfig};

/// A time-parameterized trajectory.
///
/// Usage:
///     traj = planner.plan(start, goal)
///     traj.duration          # total time in seconds
///     traj.num_waypoints     # number of waypoints
///     joints = traj.sample(t)         # interpolate at time t
///     times, pos, vel = traj.to_numpy()  # export entire trajectory
#[pyclass(name = "Trajectory")]
#[derive(Clone)]
pub struct PyTrajectory {
    pub(crate) timed: Option<TimedTrajectory>,
    pub(crate) path: Option<Vec<Vec<f64>>>,
}

impl PyTrajectory {
    pub fn from_timed(timed: TimedTrajectory) -> Self {
        PyTrajectory {
            timed: Some(timed),
            path: None,
        }
    }

    pub fn from_path(path: Vec<Vec<f64>>) -> Self {
        PyTrajectory {
            timed: None,
            path: Some(path),
        }
    }
}

#[pymethods]
impl PyTrajectory {
    /// Duration of the trajectory in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        if let Some(t) = &self.timed {
            t.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Number of waypoints in the trajectory.
    #[getter]
    fn num_waypoints(&self) -> usize {
        if let Some(t) = &self.timed {
            t.waypoints.len()
        } else if let Some(p) = &self.path {
            p.len()
        } else {
            0
        }
    }

    /// Number of degrees of freedom per waypoint.
    #[getter]
    fn dof(&self) -> usize {
        if let Some(t) = &self.timed {
            t.waypoints.first().map_or(0, |w| w.positions.len())
        } else if let Some(p) = &self.path {
            p.first().map_or(0, |w| w.len())
        } else {
            0
        }
    }

    /// Sample the trajectory at time t (seconds), returning joint positions.
    ///
    /// Uses linear interpolation between waypoints.
    #[pyo3(text_signature = "(self, t)")]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let timed = self
            .timed
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Trajectory not time-parameterized; call with time_parameterize=True",
                )
            })?;

        if timed.waypoints.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Empty trajectory",
            ));
        }

        // Clamp t to valid range
        let t = t.clamp(0.0, timed.duration.as_secs_f64());

        // Find surrounding waypoints
        let mut i = 0;
        while i + 1 < timed.waypoints.len() && timed.waypoints[i + 1].time < t {
            i += 1;
        }

        if i + 1 >= timed.waypoints.len() {
            // At or past end
            let positions = &timed.waypoints.last().unwrap().positions;
            return Ok(PyArray1::from_slice(py, positions));
        }

        let wp0 = &timed.waypoints[i];
        let wp1 = &timed.waypoints[i + 1];
        let dt = wp1.time - wp0.time;
        let alpha = if dt > 1e-12 {
            (t - wp0.time) / dt
        } else {
            0.0
        };

        let positions: Vec<f64> = wp0
            .positions
            .iter()
            .zip(wp1.positions.iter())
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect();

        Ok(PyArray1::from_slice(py, &positions))
    }

    /// Export the entire trajectory as numpy arrays.
    ///
    /// Returns:
    ///     (times, positions, velocities) where:
    ///         times: (N,) array of timestamps
    ///         positions: (N, DOF) array of joint positions
    ///         velocities: (N, DOF) array of joint velocities
    #[pyo3(text_signature = "(self)")]
    #[allow(clippy::type_complexity)]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let timed = self
            .timed
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Trajectory not time-parameterized",
                )
            })?;

        let n = timed.waypoints.len();
        let dof = timed
            .waypoints
            .first()
            .map_or(0, |w| w.positions.len());

        let times: Vec<f64> = timed.waypoints.iter().map(|w| w.time).collect();

        let mut pos_flat = Vec::with_capacity(n * dof);
        let mut vel_flat = Vec::with_capacity(n * dof);
        for wp in &timed.waypoints {
            pos_flat.extend_from_slice(&wp.positions);
            vel_flat.extend_from_slice(&wp.velocities);
        }

        let times_arr = PyArray1::from_slice(py, &times);
        let pos_arr = PyArray1::from_slice(py, &pos_flat).reshape([n, dof])?;
        let vel_arr = PyArray1::from_slice(py, &vel_flat).reshape([n, dof])?;

        Ok((times_arr, pos_arr, vel_arr))
    }

    /// Get positions as a list of lists (no numpy required).
    #[pyo3(text_signature = "(self)")]
    fn positions(&self) -> Vec<Vec<f64>> {
        if let Some(t) = &self.timed {
            t.waypoints.iter().map(|w| w.positions.clone()).collect()
        } else if let Some(p) = &self.path {
            p.clone()
        } else {
            vec![]
        }
    }

    /// Validate the trajectory against per-joint limits.
    ///
    /// Args:
    ///     position_lower: numpy array of per-joint lower position limits
    ///     position_upper: numpy array of per-joint upper position limits
    ///     velocity_limits: numpy array of per-joint velocity limits
    ///     acceleration_limits: numpy array of per-joint acceleration limits
    ///
    /// Returns:
    ///     list of violation dicts, empty if trajectory is valid.
    ///     Each dict has: {'waypoint': int, 'joint': int, 'type': str, 'value': float, 'limit': float}
    #[pyo3(text_signature = "(self, position_lower, position_upper, velocity_limits, acceleration_limits)")]
    fn validate<'py>(
        &self,
        py: Python<'py>,
        position_lower: PyReadonlyArray1<'_, f64>,
        position_upper: PyReadonlyArray1<'_, f64>,
        velocity_limits: PyReadonlyArray1<'_, f64>,
        acceleration_limits: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyObject> {
        let timed = self
            .timed
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Trajectory not time-parameterized; cannot validate",
                )
            })?;

        let pos_lo = position_lower.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let pos_hi = position_upper.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let vel = velocity_limits.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let acc = acceleration_limits.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;

        let validator = TrajectoryValidator::new(pos_lo, pos_hi, vel, acc, ValidationConfig::default());

        let result_list = pyo3::types::PyList::empty(py);
        if let Err(violations) = validator.validate(timed) {
            for v in &violations {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("waypoint", v.waypoint_index)?;
                dict.set_item("joint", v.joint_index)?;
                dict.set_item(
                    "type",
                    format!("{:?}", v.violation_type),
                )?;
                dict.set_item("value", v.actual_value)?;
                dict.set_item("limit", v.limit_value)?;
                result_list.append(dict)?;
            }
        }

        Ok(result_list.into_any().unbind())
    }

    /// Apply time parameterization to a geometric path.
    ///
    /// Args:
    ///     profile: "trapezoidal", "jerk_limited", "totp", or "cubic_spline"
    ///     velocity_limits: numpy array of per-joint max velocities
    ///     acceleration_limits: numpy array of per-joint max accelerations
    ///     jerk_limits: numpy array of per-joint max jerks (only for "jerk_limited")
    ///
    /// Returns:
    ///     New time-parameterized Trajectory
    #[pyo3(text_signature = "(self, profile, velocity_limits, acceleration_limits, jerk_limits=None)")]
    #[pyo3(signature = (profile, velocity_limits, acceleration_limits, jerk_limits=None))]
    fn time_parameterize(
        &self,
        profile: &str,
        velocity_limits: PyReadonlyArray1<'_, f64>,
        acceleration_limits: PyReadonlyArray1<'_, f64>,
        jerk_limits: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let path = if let Some(p) = &self.path {
            p.clone()
        } else if let Some(t) = &self.timed {
            t.waypoints.iter().map(|w| w.positions.clone()).collect()
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Empty trajectory"));
        };

        let vel = velocity_limits.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let acc = acceleration_limits.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;

        let timed = match profile.to_lowercase().as_str() {
            "trapezoidal" | "trapez" => {
                kinetic_trajectory::trapezoidal_per_joint(&path, vel, acc)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
            }
            "jerk_limited" | "s_curve" | "scurve" => {
                let jerk = jerk_limits.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "jerk_limits required for jerk_limited profile",
                    )
                })?;
                let jerk_slice = jerk.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
                })?;
                kinetic_trajectory::jerk_limited_per_joint(&path, vel, acc, jerk_slice)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
            }
            "totp" | "time_optimal" => {
                kinetic_trajectory::totp(&path, vel, acc, 0.001)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
            }
            "cubic_spline" | "spline" => {
                kinetic_trajectory::cubic_spline_time(&path, None, Some(vel))
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown profile '{}'. Use 'trapezoidal', 'jerk_limited', 'totp', or 'cubic_spline'",
                    profile
                )));
            }
        };

        Ok(PyTrajectory::from_timed(timed))
    }

    /// Blend this trajectory with another, creating a smooth transition.
    ///
    /// Args:
    ///     other: another Trajectory to blend into
    ///     blend_duration: duration of the blend zone in seconds
    ///
    /// Returns:
    ///     New blended Trajectory
    #[pyo3(text_signature = "(self, other, blend_duration)")]
    fn blend(&self, other: &PyTrajectory, blend_duration: f64) -> PyResult<Self> {
        let t1 = self.timed.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("First trajectory not time-parameterized")
        })?;
        let t2 = other.timed.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Second trajectory not time-parameterized")
        })?;

        let blended = kinetic_trajectory::blend(t1, t2, blend_duration)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(PyTrajectory::from_timed(blended))
    }

    /// Export trajectory as JSON string.
    #[pyo3(text_signature = "(self)")]
    fn to_json(&self) -> PyResult<String> {
        let timed = self.timed.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trajectory not time-parameterized")
        })?;
        Ok(kinetic_trajectory::export::trajectory_to_json(timed))
    }

    /// Export trajectory as CSV string.
    #[pyo3(text_signature = "(self)")]
    fn to_csv(&self) -> PyResult<String> {
        let timed = self.timed.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Trajectory not time-parameterized")
        })?;
        Ok(kinetic_trajectory::export::trajectory_to_csv(timed))
    }

    /// Load trajectory from JSON string.
    #[staticmethod]
    #[pyo3(text_signature = "(json_str)")]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let timed = kinetic_trajectory::export::trajectory_from_json(json_str)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyTrajectory::from_timed(timed))
    }

    /// Load trajectory from CSV string.
    #[staticmethod]
    #[pyo3(text_signature = "(csv_str)")]
    fn from_csv(csv_str: &str) -> PyResult<Self> {
        let timed = kinetic_trajectory::export::trajectory_from_csv(csv_str)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyTrajectory::from_timed(timed))
    }

    fn __repr__(&self) -> String {
        format!(
            "Trajectory(waypoints={}, duration={:.3}s, dof={})",
            self.num_waypoints(),
            self.duration(),
            self.dof()
        )
    }

    fn __len__(&self) -> usize {
        self.num_waypoints()
    }
}
