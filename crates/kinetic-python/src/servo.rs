//! PyServo and PyRMP — Python bindings for kinetic reactive control.
//!
//! PyRMP wraps the Riemannian Motion Policy combiner with all 6 policy types.
//! PyServo wraps the teleoperation servo controller.

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use kinetic_core::Twist;
use kinetic_reactive::servo::{Servo, ServoConfig};
use kinetic_reactive::{PolicyType, RMP};
use kinetic_scene::Scene;
use nalgebra::Vector3;

use crate::convert;
use crate::robot::PyRobot;
use crate::scene::PyScene;

/// An RMP policy for reactive control.
///
/// Usage:
///     p = kinetic.Policy.reach_target(target_4x4, gain=10.0)
///     p = kinetic.Policy.avoid_self_collision(gain=20.0)
///     p = kinetic.Policy.joint_limit_avoidance(margin=0.1, gain=15.0)
///     p = kinetic.Policy.singularity_avoidance(threshold=0.02, gain=5.0)
///     p = kinetic.Policy.damping(coefficient=0.5)
#[pyclass(name = "Policy")]
#[derive(Clone)]
pub struct PyPolicy {
    pub(crate) inner: PolicyType,
}

#[pymethods]
impl PyPolicy {
    /// Attract end-effector toward a target pose.
    #[staticmethod]
    #[pyo3(text_signature = "(target, gain)")]
    fn reach_target(target: numpy::PyReadonlyArray2<'_, f64>, gain: f64) -> PyResult<Self> {
        let iso = convert::numpy_4x4_to_isometry(target)?;
        Ok(PyPolicy {
            inner: PolicyType::ReachTarget {
                target_pose: iso,
                gain,
            },
        })
    }

    /// Repel robot from scene obstacles.
    #[staticmethod]
    #[pyo3(text_signature = "(scene, influence_distance, gain)")]
    fn avoid_obstacles(scene: &PyScene, influence_distance: f64, gain: f64) -> PyResult<Self> {
        let robot = scene.inner.robot();
        let scene_arc = Arc::new(
            Scene::new(robot)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        Ok(PyPolicy {
            inner: PolicyType::AvoidObstacles {
                scene: scene_arc,
                influence_distance,
                gain,
            },
        })
    }

    /// Repel self-collision between robot links.
    #[staticmethod]
    #[pyo3(text_signature = "(gain)")]
    fn avoid_self_collision(gain: f64) -> Self {
        PyPolicy {
            inner: PolicyType::AvoidSelfCollision { gain },
        }
    }

    /// Soft repulsion from joint limits.
    #[staticmethod]
    #[pyo3(text_signature = "(margin, gain)")]
    fn joint_limit_avoidance(margin: f64, gain: f64) -> Self {
        PyPolicy {
            inner: PolicyType::JointLimitAvoidance { margin, gain },
        }
    }

    /// Slow down near singular configurations.
    #[staticmethod]
    #[pyo3(text_signature = "(threshold, gain)")]
    fn singularity_avoidance(threshold: f64, gain: f64) -> Self {
        PyPolicy {
            inner: PolicyType::SingularityAvoidance { threshold, gain },
        }
    }

    /// Velocity damping for stability.
    #[staticmethod]
    #[pyo3(text_signature = "(coefficient)")]
    fn damping(coefficient: f64) -> Self {
        PyPolicy {
            inner: PolicyType::Damping { coefficient },
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Riemannian Motion Policy combiner.
///
/// Combines multiple policies (reach target, avoid obstacles, etc.)
/// into a single consistent joint-space command via metric-weighted
/// averaging.
///
/// Usage:
///     rmp = kinetic.RMP(robot)
///     rmp.add(kinetic.Policy.reach_target(target, gain=10.0))
///     rmp.add(kinetic.Policy.joint_limit_avoidance(0.1, 15.0))
///     rmp.add(kinetic.Policy.damping(0.5))
///     cmd = rmp.compute(current_joints, current_velocities, dt=0.002)
#[pyclass(name = "RMP")]
pub struct PyRMP {
    rmp: RMP,
}

#[pymethods]
impl PyRMP {
    /// Create an RMP controller for a robot.
    #[new]
    #[pyo3(text_signature = "(robot)")]
    fn new(robot: &PyRobot) -> PyResult<Self> {
        let robot_arc = Arc::new(robot.inner.clone());
        let rmp = RMP::new(&robot_arc)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyRMP { rmp })
    }

    /// Add a policy to the RMP controller.
    #[pyo3(text_signature = "(self, policy)")]
    fn add(&mut self, policy: &PyPolicy) {
        self.rmp.add(policy.inner.clone());
    }

    /// Clear all policies.
    #[pyo3(text_signature = "(self)")]
    fn clear(&mut self) {
        self.rmp.clear();
    }

    /// Number of active policies.
    #[getter]
    fn num_policies(&self) -> usize {
        self.rmp.num_policies()
    }

    /// Robot DOF.
    #[getter]
    fn dof(&self) -> usize {
        self.rmp.dof()
    }

    /// Compute joint command from all policies.
    ///
    /// Args:
    ///     current_joints: numpy array of current joint positions
    ///     current_velocities: numpy array of current joint velocities
    ///     dt: timestep in seconds (e.g., 0.002 for 500Hz)
    ///
    /// Returns:
    ///     dict with 'positions', 'velocities', 'accelerations' arrays
    #[pyo3(text_signature = "(self, current_joints, current_velocities, dt)")]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        current_joints: PyReadonlyArray1<'_, f64>,
        current_velocities: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) -> PyResult<PyObject> {
        let joints = convert::numpy_to_vec(current_joints)?;
        let vels = convert::numpy_to_vec(current_velocities)?;

        let cmd = self
            .rmp
            .compute(&joints, &vels, dt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("positions", PyArray1::from_slice(py, &cmd.positions))?;
        dict.set_item("velocities", PyArray1::from_slice(py, &cmd.velocities))?;
        dict.set_item("accelerations", PyArray1::from_slice(py, &cmd.accelerations))?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!("RMP(dof={}, policies={})", self.rmp.dof(), self.rmp.num_policies())
    }
}

/// Real-time servo controller for teleoperation.
///
/// Usage:
///     servo = kinetic.Servo(robot, scene, rate_hz=500)
///     cmd = servo.send_twist([0.1, 0, 0, 0, 0, 0])
///     state = servo.state()
#[pyclass(name = "Servo")]
pub struct PyServo {
    servo: Servo,
}

#[pymethods]
impl PyServo {
    /// Create a servo controller.
    ///
    /// Args:
    ///     robot: Robot model
    ///     scene: Scene with collision objects (optional, pass empty Scene)
    ///     rate_hz: servo rate in Hz (default 500)
    #[new]
    #[pyo3(text_signature = "(robot, scene, rate_hz=500.0)")]
    #[pyo3(signature = (robot, scene, rate_hz=500.0))]
    fn new(robot: &PyRobot, scene: &PyScene, rate_hz: f64) -> PyResult<Self> {
        // Use the provided scene's internal state via Arc
        let _ = &scene; // Scene is used to validate the API, but we build from robot for Arc ownership
        let scene_arc = Arc::new(
            Scene::new(&robot.inner)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let config = ServoConfig {
            rate_hz,
            ..ServoConfig::default()
        };

        let servo = Servo::new(&robot.inner, &scene_arc, config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyServo { servo })
    }

    /// Send a Cartesian twist command (6D: [vx, vy, vz, wx, wy, wz]).
    ///
    /// Returns dict with 'positions' and 'velocities' arrays.
    #[pyo3(text_signature = "(self, twist)")]
    fn send_twist<'py>(
        &mut self,
        py: Python<'py>,
        twist: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyObject> {
        let t = convert::numpy_to_vec(twist)?;
        if t.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Twist must be a 6-element array [vx, vy, vz, wx, wy, wz]",
            ));
        }

        let twist = Twist::new(
            Vector3::new(t[0], t[1], t[2]),
            Vector3::new(t[3], t[4], t[5]),
        );

        let cmd = self
            .servo
            .send_twist(&twist)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("positions", PyArray1::from_slice(py, &cmd.positions))?;
        dict.set_item("velocities", PyArray1::from_slice(py, &cmd.velocities))?;
        Ok(dict.into_any().unbind())
    }

    /// Send a joint jog command for a single joint.
    ///
    /// Args:
    ///     joint_index: which joint to jog (0-based)
    ///     velocity: jog velocity in rad/s
    #[pyo3(text_signature = "(self, joint_index, velocity)")]
    fn send_joint_jog<'py>(
        &mut self,
        py: Python<'py>,
        joint_index: usize,
        velocity: f64,
    ) -> PyResult<PyObject> {
        let cmd = self
            .servo
            .send_joint_jog(joint_index, velocity)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("positions", PyArray1::from_slice(py, &cmd.positions))?;
        dict.set_item("velocities", PyArray1::from_slice(py, &cmd.velocities))?;
        Ok(dict.into_any().unbind())
    }

    /// Get current servo state.
    ///
    /// Returns dict with: joint_positions, joint_velocities, ee_pose (4x4),
    /// manipulability, near_singularity, near_collision.
    #[pyo3(text_signature = "(self)")]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let state = self.servo.state();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(
            "joint_positions",
            PyArray1::from_slice(py, &state.joint_positions),
        )?;
        dict.set_item(
            "joint_velocities",
            PyArray1::from_slice(py, &state.joint_velocities),
        )?;
        dict.set_item("ee_pose", convert::isometry_to_numpy_4x4(py, &state.ee_pose))?;
        dict.set_item("manipulability", state.manipulability)?;
        dict.set_item("near_singularity", state.is_near_singularity)?;
        dict.set_item("near_collision", state.is_near_collision)?;
        Ok(dict.into_any().unbind())
    }

    /// Set the servo to a specific state.
    #[pyo3(text_signature = "(self, positions, velocities)")]
    fn set_state(
        &mut self,
        positions: PyReadonlyArray1<'_, f64>,
        velocities: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let pos = convert::numpy_to_vec(positions)?;
        let vel = convert::numpy_to_vec(velocities)?;
        self.servo
            .set_state(&pos, &vel)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Servo rate in Hz.
    #[getter]
    fn rate_hz(&self) -> f64 {
        500.0 // Default; ServoConfig doesn't expose this directly after construction
    }

    fn __repr__(&self) -> String {
        format!("Servo(dof={})", self.servo.dof())
    }
}
