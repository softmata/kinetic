//! PyRobot — Python bindings for kinetic Robot.

use std::sync::Arc;

use numpy::{PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kinetic_kinematics::{
    fk, fk_batch, jacobian, manipulability, solve_ik, solve_ik_batch, IKConfig, KinematicChain,
};
use kinetic_robot::Robot;

use crate::convert;

/// A robot model loaded from URDF + config.
///
/// Usage:
///     robot = kinetic.Robot("ur5e")
///     robot = kinetic.Robot.from_urdf("/path/to/robot.urdf")
///     pose = robot.fk([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
#[pyclass(name = "Robot")]
pub struct PyRobot {
    pub(crate) inner: Arc<Robot>,
    pub(crate) chain: KinematicChain,
}

#[pymethods]
impl PyRobot {
    /// Load a robot from a built-in config name (e.g., "ur5e", "franka_panda").
    #[new]
    #[pyo3(text_signature = "(name)")]
    fn new(name: &str) -> PyResult<Self> {
        let robot = Robot::from_name(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let chain = auto_chain(&robot)?;
        Ok(PyRobot {
            inner: Arc::new(robot),
            chain,
        })
    }

    /// Load a robot from a URDF file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_urdf(path: &str) -> PyResult<Self> {
        let robot = Robot::from_urdf(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let chain = auto_chain(&robot)?;
        Ok(PyRobot {
            inner: Arc::new(robot),
            chain,
        })
    }

    /// Load from a built-in config name (alias for constructor).
    #[staticmethod]
    #[pyo3(text_signature = "(name)")]
    fn from_config(name: &str) -> PyResult<Self> {
        Self::new(name)
    }

    /// Load a robot from URDF + SRDF file paths.
    ///
    /// SRDF provides planning groups, disabled collision pairs, named
    /// poses, and end-effector definitions.
    #[staticmethod]
    #[pyo3(text_signature = "(urdf_path, srdf_path)")]
    fn from_urdf_srdf(urdf_path: &str, srdf_path: &str) -> PyResult<Self> {
        let robot = Robot::from_urdf_srdf(urdf_path, srdf_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let chain = auto_chain(&robot)?;
        Ok(PyRobot {
            inner: Arc::new(robot),
            chain,
        })
    }

    /// Get named pose joint values as a list (e.g., "home", "zero").
    #[pyo3(text_signature = "(self, name)")]
    fn named_pose<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, numpy::PyArray1<f64>>>> {
        Ok(self.inner.named_pose(name).map(|jv| {
            numpy::PyArray1::from_slice(py, &jv.0)
        }))
    }

    /// List all available named poses.
    #[getter]
    fn named_poses(&self) -> Vec<String> {
        self.inner.named_poses.keys().cloned().collect()
    }

    /// Per-joint velocity limits.
    #[getter]
    fn velocity_limits(&self) -> Vec<f64> {
        self.inner.velocity_limits()
    }

    /// Per-joint acceleration limits.
    #[getter]
    fn acceleration_limits(&self) -> Vec<f64> {
        self.inner.acceleration_limits()
    }

    /// Robot name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Number of degrees of freedom.
    #[getter]
    fn dof(&self) -> usize {
        self.chain.dof
    }

    /// Number of joints (including fixed).
    #[getter]
    fn num_joints(&self) -> usize {
        self.inner.joints.len()
    }

    /// Number of links.
    #[getter]
    fn num_links(&self) -> usize {
        self.inner.links.len()
    }

    /// Joint names (active revolute/prismatic only).
    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.chain
            .active_joints
            .iter()
            .map(|&idx| self.inner.joints[idx].name.clone())
            .collect()
    }

    /// Forward kinematics: joint_values → 4x4 pose matrix.
    #[pyo3(text_signature = "(self, joint_values)")]
    fn fk<'py>(
        &self,
        py: Python<'py>,
        joint_values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let joints = convert::numpy_to_vec(joint_values)?;
        let pose = fk(&self.inner, &self.chain, &joints)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(convert::pose_to_numpy_4x4(py, &pose))
    }

    /// Jacobian matrix: joint_values → (6, DOF) matrix.
    #[pyo3(text_signature = "(self, joint_values)")]
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        joint_values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let joints = convert::numpy_to_vec(joint_values)?;
        let jac = jacobian(&self.inner, &self.chain, &joints)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let rows = jac.nrows();
        let cols = jac.ncols();
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(jac[(r, c)]);
            }
        }
        let flat = numpy::PyArray1::from_slice(py, &data);
        Ok(flat
            .reshape([rows, cols])
            .expect("reshape should succeed"))
    }

    /// Inverse kinematics: target_pose (4x4) → joint solution.
    ///
    /// Args:
    ///     target_pose: (4,4) numpy array (SE3 homogeneous)
    ///     seed: optional starting configuration (1D array)
    ///
    /// Returns:
    ///     numpy array of joint values, or raises RuntimeError if IK fails.
    #[pyo3(text_signature = "(self, target_pose, seed=None)")]
    #[pyo3(signature = (target_pose, seed=None))]
    fn ik<'py>(
        &self,
        py: Python<'py>,
        target_pose: numpy::PyReadonlyArray2<'_, f64>,
        seed: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let iso = convert::numpy_4x4_to_isometry(target_pose)?;
        let pose = kinetic_core::Pose(iso);

        let config = IKConfig {
            seed: seed.map(|s| convert::numpy_to_vec(s)).transpose()?,
            ..IKConfig::default()
        };

        let sol = solve_ik(&self.inner, &self.chain, &pose, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(convert::vec_to_numpy(py, &sol.joints))
    }

    /// Inverse kinematics with full configuration options.
    ///
    /// Args:
    ///     target_pose: (4,4) numpy array
    ///     solver: "auto", "dls", "fabrik", "opw", "subproblem", "subproblem7dof" (default: "auto")
    ///     mode: "full6d", "position_only", "position_fallback" (default: "full6d")
    ///     seed: optional starting configuration
    ///     null_space: "manipulability", "minimal_displacement", "joint_centering", or None
    ///     max_iterations: max iterations for iterative solvers (default: 300)
    ///     num_restarts: number of random restarts (default: 10)
    ///
    /// Returns:
    ///     dict with 'joints' (numpy array), 'converged' (bool), 'position_error' (float),
    ///     'orientation_error' (float), 'iterations' (int)
    #[pyo3(text_signature = "(self, target_pose, solver='auto', mode='full6d', seed=None, null_space=None, max_iterations=300, num_restarts=10)")]
    #[pyo3(signature = (target_pose, solver="auto", mode="full6d", seed=None, null_space=None, max_iterations=300, num_restarts=10))]
    fn ik_config<'py>(
        &self,
        py: Python<'py>,
        target_pose: numpy::PyReadonlyArray2<'_, f64>,
        solver: &str,
        mode: &str,
        seed: Option<PyReadonlyArray1<'_, f64>>,
        null_space: Option<&str>,
        max_iterations: usize,
        num_restarts: usize,
    ) -> PyResult<PyObject> {
        let iso = convert::numpy_4x4_to_isometry(target_pose)?;
        let pose = kinetic_core::Pose(iso);

        let ik_solver = match solver.to_lowercase().as_str() {
            "auto" => kinetic_kinematics::IKSolver::Auto,
            "dls" => kinetic_kinematics::IKSolver::DLS { damping: 0.05 },
            "fabrik" => kinetic_kinematics::IKSolver::FABRIK,
            "opw" => kinetic_kinematics::IKSolver::OPW,
            "subproblem" => kinetic_kinematics::IKSolver::Subproblem,
            "subproblem7dof" => kinetic_kinematics::IKSolver::Subproblem7DOF { num_samples: 36 },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown solver '{}'. Use: auto, dls, fabrik, opw, subproblem, subproblem7dof",
                    solver
                )))
            }
        };

        let ik_mode = match mode.to_lowercase().as_str() {
            "full6d" | "full" => kinetic_kinematics::IKMode::Full6D,
            "position_only" | "position" => kinetic_kinematics::IKMode::PositionOnly,
            "position_fallback" | "fallback" => kinetic_kinematics::IKMode::PositionFallback,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown mode '{}'. Use: full6d, position_only, position_fallback",
                    mode
                )))
            }
        };

        let ns = match null_space {
            Some("manipulability") => Some(kinetic_kinematics::NullSpace::Manipulability),
            Some("minimal_displacement") => {
                Some(kinetic_kinematics::NullSpace::MinimalDisplacement)
            }
            Some("joint_centering") => Some(kinetic_kinematics::NullSpace::JointCentering),
            None => None,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown null_space '{}'. Use: manipulability, minimal_displacement, joint_centering",
                    other
                )))
            }
        };

        let config = IKConfig {
            solver: ik_solver,
            mode: ik_mode,
            max_iterations,
            seed: seed.map(|s| convert::numpy_to_vec(s)).transpose()?,
            null_space: ns,
            num_restarts,
            ..IKConfig::default()
        };

        let sol = solve_ik(&self.inner, &self.chain, &pose, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("joints", convert::vec_to_numpy(py, &sol.joints))?;
        dict.set_item("converged", sol.converged)?;
        dict.set_item("position_error", sol.position_error)?;
        dict.set_item("orientation_error", sol.orientation_error)?;
        dict.set_item("iterations", sol.iterations)?;
        Ok(dict.into_any().unbind())
    }

    /// Batch forward kinematics: multiple configs → list of 4x4 poses.
    ///
    /// Args:
    ///     configs: (N, DOF) numpy array of joint configurations
    ///
    /// Returns:
    ///     list of (4,4) numpy arrays (one per configuration)
    #[pyo3(text_signature = "(self, configs)")]
    fn batch_fk<'py>(
        &self,
        py: Python<'py>,
        configs: numpy::PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f64>>>> {
        let shape = configs.shape();
        let n = shape[0];
        let dof = shape[1];
        if dof != self.chain.dof {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} DOF per config, got {}",
                self.chain.dof, dof
            )));
        }
        let flat = configs.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let poses = fk_batch(&self.inner, &self.chain, flat, n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(poses.iter().map(|p| convert::pose_to_numpy_4x4(py, p)).collect())
    }

    /// Batch inverse kinematics: multiple target poses → list of solutions.
    ///
    /// Args:
    ///     target_poses: list of (4,4) numpy arrays
    ///     solver: solver name (default: "auto")
    ///
    /// Returns:
    ///     list of dicts (same format as ik_config), None for failed solves
    #[pyo3(text_signature = "(self, target_poses, solver='auto', num_restarts=10)")]
    #[pyo3(signature = (target_poses, solver="auto", num_restarts=10))]
    fn batch_ik<'py>(
        &self,
        py: Python<'py>,
        target_poses: Vec<numpy::PyReadonlyArray2<'_, f64>>,
        solver: &str,
        num_restarts: usize,
    ) -> PyResult<PyObject> {
        let poses: Vec<kinetic_core::Pose> = target_poses
            .into_iter()
            .map(|p| convert::numpy_4x4_to_isometry(p).map(kinetic_core::Pose))
            .collect::<PyResult<_>>()?;

        let ik_solver = match solver.to_lowercase().as_str() {
            "auto" => kinetic_kinematics::IKSolver::Auto,
            "dls" => kinetic_kinematics::IKSolver::DLS { damping: 0.05 },
            "fabrik" => kinetic_kinematics::IKSolver::FABRIK,
            "opw" => kinetic_kinematics::IKSolver::OPW,
            _ => kinetic_kinematics::IKSolver::Auto,
        };

        let config = IKConfig {
            solver: ik_solver,
            num_restarts,
            ..IKConfig::default()
        };

        let results = solve_ik_batch(&self.inner, &self.chain, &poses, &config);

        let list = pyo3::types::PyList::empty(py);
        for result in results {
            match result {
                Some(sol) => {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("joints", convert::vec_to_numpy(py, &sol.joints))?;
                    dict.set_item("converged", sol.converged)?;
                    dict.set_item("position_error", sol.position_error)?;
                    dict.set_item("orientation_error", sol.orientation_error)?;
                    dict.set_item("iterations", sol.iterations)?;
                    list.append(dict)?;
                }
                None => {
                    list.append(py.None())?;
                }
            }
        }
        Ok(list.into_any().unbind())
    }

    /// Manipulability index at given joint configuration.
    #[pyo3(text_signature = "(self, joint_values)")]
    fn manipulability(&self, joint_values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let joints = convert::numpy_to_vec(joint_values)?;
        manipulability(&self.inner, &self.chain, &joints)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Robot('{}', dof={}, joints={}, links={})",
            self.inner.name,
            self.chain.dof,
            self.inner.joints.len(),
            self.inner.links.len()
        )
    }
}

fn auto_chain(robot: &Robot) -> PyResult<KinematicChain> {
    if let Some(group) = robot.groups.values().next() {
        return KinematicChain::extract(robot, &group.base_link, &group.tip_link)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    // Fallback: root → deepest leaf
    let root = &robot.links[0].name;
    let mut farthest = root.clone();
    let mut max_depth = 0;

    fn walk(robot: &Robot, idx: usize, depth: usize, max: &mut usize, farthest: &mut String) {
        if depth > *max {
            *max = depth;
            *farthest = robot.links[idx].name.clone();
        }
        for j in &robot.joints {
            if j.parent_link == idx {
                walk(robot, j.child_link, depth + 1, max, farthest);
            }
        }
    }

    walk(robot, 0, 0, &mut max_depth, &mut farthest);
    KinematicChain::extract(robot, root, &farthest)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
