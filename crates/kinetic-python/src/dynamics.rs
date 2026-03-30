//! PyDynamics — Python bindings for kinetic dynamics (featherstone bridge).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use kinetic_core::JointValues;
use kinetic_dynamics::{
    articulated_body_from_chain, forward_dynamics, gravity_compensation, inverse_dynamics,
    mass_matrix,
};

use crate::convert;
use crate::robot::PyRobot;

/// Dynamics queries for a robot (inverse dynamics, gravity compensation, etc.).
///
/// Usage:
///     dyn = kinetic.Dynamics(robot)
///     tau = dyn.gravity_compensation(joint_positions)
///     tau = dyn.inverse_dynamics(q, qd, qdd)
///     qdd = dyn.forward_dynamics(q, qd, tau)
///     M = dyn.mass_matrix(q)
#[pyclass(name = "Dynamics")]
pub struct PyDynamics {
    robot: std::sync::Arc<kinetic_robot::Robot>,
    chain: kinetic_kinematics::KinematicChain,
}

#[pymethods]
impl PyDynamics {
    #[new]
    #[pyo3(text_signature = "(robot)")]
    fn new(robot: &PyRobot) -> Self {
        PyDynamics {
            robot: robot.inner.clone(),
            chain: robot.chain.clone(),
        }
    }

    /// Compute gravity compensation torques at a given joint configuration.
    ///
    /// Returns numpy array of torques needed to hold the robot stationary.
    #[pyo3(text_signature = "(self, joint_positions)")]
    fn gravity_compensation<'py>(
        &self,
        py: Python<'py>,
        joint_positions: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = JointValues(convert::numpy_to_vec(joint_positions)?);
        let mut body = articulated_body_from_chain(&self.robot, &self.chain);
        let tau = gravity_compensation(&mut body, &q);
        Ok(convert::vec_to_numpy(py, &tau.0))
    }

    /// Compute inverse dynamics: torques required for (q, qd, qdd).
    ///
    /// Returns τ = M(q)q̈ + C(q,q̇)q̇ + g(q).
    #[pyo3(text_signature = "(self, q, qd, qdd)")]
    fn inverse_dynamics<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'_, f64>,
        qd: PyReadonlyArray1<'_, f64>,
        qdd: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = JointValues(convert::numpy_to_vec(q)?);
        let qd = JointValues(convert::numpy_to_vec(qd)?);
        let qdd = JointValues(convert::numpy_to_vec(qdd)?);
        let mut body = articulated_body_from_chain(&self.robot, &self.chain);
        let tau = inverse_dynamics(&mut body, &q, &qd, &qdd);
        Ok(convert::vec_to_numpy(py, &tau.0))
    }

    /// Compute forward dynamics: accelerations from applied torques.
    ///
    /// Returns q̈ = M⁻¹(q)(τ - C(q,q̇)q̇ - g(q)).
    #[pyo3(text_signature = "(self, q, qd, tau)")]
    fn forward_dynamics<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'_, f64>,
        qd: PyReadonlyArray1<'_, f64>,
        tau: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = JointValues(convert::numpy_to_vec(q)?);
        let qd = JointValues(convert::numpy_to_vec(qd)?);
        let tau = JointValues(convert::numpy_to_vec(tau)?);
        let mut body = articulated_body_from_chain(&self.robot, &self.chain);
        let qdd = forward_dynamics(&mut body, &q, &qd, &tau);
        Ok(convert::vec_to_numpy(py, &qdd.0))
    }

    /// Compute the joint-space mass matrix M(q).
    ///
    /// Returns (DOF, DOF) numpy array (symmetric positive-definite).
    #[pyo3(text_signature = "(self, joint_positions)")]
    fn mass_matrix<'py>(
        &self,
        py: Python<'py>,
        joint_positions: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = JointValues(convert::numpy_to_vec(joint_positions)?);
        let mut body = articulated_body_from_chain(&self.robot, &self.chain);
        let m = mass_matrix(&mut body, &q);

        let n = m.nrows();
        let mut data = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                data.push(m[(r, c)]);
            }
        }
        let flat = PyArray1::from_slice(py, &data);
        Ok(flat.reshape([n, n]).expect("reshape should succeed"))
    }

    fn __repr__(&self) -> String {
        format!("Dynamics('{}', dof={})", self.robot.name, self.chain.dof)
    }
}
