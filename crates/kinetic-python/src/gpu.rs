//! GPU-accelerated trajectory optimization and collision checking.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kinetic_collision::{RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic_gpu::{
    BatchCollisionResult, CpuCollisionChecker, CpuOptimizer, GpuCollisionChecker, GpuConfig,
    GpuOptimizer,
};

use crate::convert;
use crate::robot::PyRobot;
use crate::scene::PyScene;
use crate::trajectory::PyTrajectory;

/// GPU-accelerated trajectory optimizer (cuRobo-style parallel seeds).
///
/// Usage:
///     opt = kinetic.GpuOptimizer(robot)
///     opt = kinetic.GpuOptimizer(robot, preset="speed")  # fast
///     opt = kinetic.GpuOptimizer(robot, preset="quality")  # high quality
///     traj = opt.optimize(start, goal, scene=scene)
///
/// Falls back to CPU if no GPU is available.
#[pyclass(name = "GpuOptimizer")]
pub struct PyGpuOptimizer {
    gpu: Option<GpuOptimizer>,
    cpu: Option<CpuOptimizer>,
    robot: std::sync::Arc<kinetic_robot::Robot>,
}

#[pymethods]
impl PyGpuOptimizer {
    /// Create a GPU optimizer.
    ///
    /// Args:
    ///     robot: Robot model
    ///     preset: "balanced" (default), "speed", or "quality"
    ///     num_seeds: override number of parallel seeds (default from preset)
    ///     iterations: override optimization iterations (default from preset)
    #[new]
    #[pyo3(text_signature = "(robot, preset='balanced', num_seeds=None, iterations=None)")]
    #[pyo3(signature = (robot, preset="balanced", num_seeds=None, iterations=None))]
    fn new(
        robot: &PyRobot,
        preset: &str,
        num_seeds: Option<u32>,
        iterations: Option<u32>,
    ) -> PyResult<Self> {
        let mut config = match preset.to_lowercase().as_str() {
            "balanced" | "default" => GpuConfig::balanced(),
            "speed" | "fast" => GpuConfig::speed(),
            "quality" | "high" => GpuConfig::quality(),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown preset '{}'. Use: balanced, speed, quality",
                    preset
                )))
            }
        };

        if let Some(ns) = num_seeds {
            config.num_seeds = ns;
        }
        if let Some(it) = iterations {
            config.iterations = it;
        }

        // Try GPU, fall back to CPU
        match GpuOptimizer::new(config.clone()) {
            Ok(gpu) => Ok(PyGpuOptimizer {
                gpu: Some(gpu),
                cpu: None,
                robot: robot.inner.clone(),
            }),
            Err(_) => Ok(PyGpuOptimizer {
                gpu: None,
                cpu: Some(CpuOptimizer::new(config)),
                robot: robot.inner.clone(),
            }),
        }
    }

    /// Whether this optimizer is running on GPU (vs CPU fallback).
    #[getter]
    fn is_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Check if any GPU is available on this system.
    #[staticmethod]
    fn gpu_available() -> bool {
        GpuOptimizer::gpu_available()
    }

    /// Optimize a trajectory from start to goal.
    ///
    /// Args:
    ///     start: numpy array of start joint positions
    ///     goal: numpy array of goal joint positions
    ///     scene: optional Scene for collision avoidance
    ///     obstacle_spheres: optional list of [x, y, z, radius] obstacle spheres
    ///
    /// Returns:
    ///     Optimized Trajectory
    #[pyo3(text_signature = "(self, start, goal, scene=None, obstacle_spheres=None)")]
    #[pyo3(signature = (start, goal, scene=None, obstacle_spheres=None))]
    fn optimize(
        &self,
        start: PyReadonlyArray1<'_, f64>,
        goal: PyReadonlyArray1<'_, f64>,
        scene: Option<&PyScene>,
        obstacle_spheres: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<PyTrajectory> {
        let s = convert::numpy_to_vec(start)?;
        let g = convert::numpy_to_vec(goal)?;

        // Build obstacle spheres
        let spheres = if let Some(scene) = scene {
            scene.inner.build_environment_spheres()
        } else if let Some(obs) = obstacle_spheres {
            let shape = obs.shape();
            let n = shape[0];
            let flat = obs.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
            })?;
            let mut soa = SpheresSoA::new();
            for i in 0..n {
                let base = i * 4;
                soa.push(flat[base], flat[base + 1], flat[base + 2], flat[base + 3], i);
            }
            soa
        } else {
            SpheresSoA::new()
        };

        let traj = if let Some(gpu) = &self.gpu {
            gpu.optimize(&self.robot, &spheres, &s, &g)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        } else if let Some(cpu) = &self.cpu {
            cpu.optimize(&self.robot, &spheres, &s, &g)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No optimizer available",
            ));
        };

        // Convert Trajectory to waypoints for PyTrajectory
        let waypoints: Vec<Vec<f64>> = traj
            .waypoints()
            .into_iter()
            .map(|jv| jv.0)
            .collect();

        Ok(PyTrajectory::from_path(waypoints))
    }

    fn __repr__(&self) -> String {
        if self.is_gpu() {
            "GpuOptimizer(backend=GPU)".to_string()
        } else {
            "GpuOptimizer(backend=CPU)".to_string()
        }
    }
}

/// GPU-accelerated batch collision checker.
///
/// Usage:
///     checker = kinetic.GpuCollisionChecker(robot, scene)
///     results = checker.check_batch(configs)  # (N, DOF) numpy array
#[pyclass(name = "GpuCollisionChecker")]
pub struct PyGpuCollisionChecker {
    gpu: Option<GpuCollisionChecker>,
    cpu: Option<CpuCollisionChecker>,
    robot: std::sync::Arc<kinetic_robot::Robot>,
    sphere_model: RobotSphereModel,
}

#[pymethods]
impl PyGpuCollisionChecker {
    /// Create a batch collision checker from a scene.
    ///
    /// Args:
    ///     robot: Robot model
    ///     scene: Scene with obstacles
    ///     resolution: SDF voxel resolution in meters (default: 0.02)
    #[new]
    #[pyo3(text_signature = "(robot, scene, resolution=0.02)")]
    #[pyo3(signature = (robot, scene, resolution=0.02))]
    fn new(robot: &PyRobot, scene: &PyScene, resolution: f32) -> PyResult<Self> {
        let spheres = scene.inner.build_environment_spheres();
        let bounds: [f32; 6] = [-2.0, -2.0, -1.0, 2.0, 2.0, 2.0];
        let sphere_model = RobotSphereModel::from_robot(&robot.inner, &SphereGenConfig::coarse());

        match GpuCollisionChecker::from_spheres(&spheres, bounds, resolution) {
            Ok(gpu) => Ok(PyGpuCollisionChecker {
                gpu: Some(gpu),
                cpu: None,
                robot: robot.inner.clone(),
                sphere_model,
            }),
            Err(_) => {
                let cpu = CpuCollisionChecker::from_spheres(&spheres, bounds, resolution)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyGpuCollisionChecker {
                    gpu: None,
                    cpu: Some(cpu),
                    robot: robot.inner.clone(),
                    sphere_model,
                })
            }
        }
    }

    /// Whether this checker is running on GPU.
    #[getter]
    fn is_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Check multiple configurations for collision.
    ///
    /// Args:
    ///     configs: (N, DOF) numpy array of joint configurations
    ///
    /// Returns:
    ///     dict with 'in_collision' (list of bool) and 'min_distances' (list of float)
    #[pyo3(text_signature = "(self, configs)")]
    fn check_batch<'py>(
        &self,
        py: Python<'py>,
        configs: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyObject> {
        let shape = configs.shape();
        let n = shape[0];
        let flat = configs.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;
        let dof = shape[1];

        let config_vecs: Vec<Vec<f64>> = (0..n)
            .map(|i| flat[i * dof..(i + 1) * dof].to_vec())
            .collect();

        let result = if let Some(gpu) = &self.gpu {
            gpu.check_batch(&self.robot, &config_vecs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        } else if let Some(cpu) = &self.cpu {
            // CPU fallback: check each config individually
            let mut in_collision = Vec::with_capacity(n);
            let mut min_distances = Vec::with_capacity(n);
            for cfg in &config_vecs {
                let (colliding, dist) = cpu.check(&self.robot, cfg, &self.sphere_model);
                in_collision.push(colliding);
                min_distances.push(dist);
            }
            BatchCollisionResult {
                in_collision,
                min_distances,
            }
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No collision checker available",
            ));
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("in_collision", result.in_collision)?;
        let dists = PyArray1::from_slice(py, &result.min_distances);
        dict.set_item("min_distances", dists)?;
        Ok(dict.into_any().unbind())
    }

    /// Check a single configuration.
    ///
    /// Returns:
    ///     (in_collision: bool, min_distance: float)
    #[pyo3(text_signature = "(self, joint_values)")]
    fn check_single(
        &self,
        joint_values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(bool, f64)> {
        let cfg = convert::numpy_to_vec(joint_values)?;

        if let Some(gpu) = &self.gpu {
            gpu.check_single(&self.robot, &cfg)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        } else if let Some(cpu) = &self.cpu {
            Ok(cpu.check(&self.robot, &cfg, &self.sphere_model))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No collision checker available",
            ))
        }
    }

    fn __repr__(&self) -> String {
        if self.is_gpu() {
            "GpuCollisionChecker(backend=GPU)".to_string()
        } else {
            "GpuCollisionChecker(backend=CPU)".to_string()
        }
    }
}
