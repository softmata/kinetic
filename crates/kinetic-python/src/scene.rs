//! PyScene and PyShape — Python bindings for kinetic scene management.

use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kinetic_scene::{Scene, Shape};

use crate::convert;
use crate::robot::PyRobot;

/// Collision shape for scene objects.
///
/// Usage:
///     shape = kinetic.Shape.cuboid(1.0, 0.6, 0.02)
///     shape = kinetic.Shape.sphere(0.05)
///     shape = kinetic.Shape.cylinder(0.03, 0.10)
#[pyclass(name = "Shape")]
#[derive(Clone)]
pub struct PyShape {
    pub(crate) inner: Shape,
}

#[pymethods]
impl PyShape {
    /// Create a cuboid (box) shape.
    ///
    /// Args:
    ///     half_x, half_y, half_z: half-extents in meters.
    #[staticmethod]
    #[pyo3(text_signature = "(half_x, half_y, half_z)")]
    fn cuboid(half_x: f64, half_y: f64, half_z: f64) -> Self {
        PyShape {
            inner: Shape::Cuboid(half_x, half_y, half_z),
        }
    }

    /// Create a sphere shape.
    #[staticmethod]
    #[pyo3(text_signature = "(radius)")]
    fn sphere(radius: f64) -> Self {
        PyShape {
            inner: Shape::Sphere(radius),
        }
    }

    /// Create a cylinder shape.
    ///
    /// Args:
    ///     radius: cylinder radius in meters.
    ///     half_height: half-height in meters.
    #[staticmethod]
    #[pyo3(text_signature = "(radius, half_height)")]
    fn cylinder(radius: f64, half_height: f64) -> Self {
        PyShape {
            inner: Shape::Cylinder(radius, half_height),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Shape::Sphere(r) => format!("Shape.sphere({})", r),
            Shape::Cuboid(x, y, z) => format!("Shape.cuboid({}, {}, {})", x, y, z),
            Shape::Cylinder(r, h) => format!("Shape.cylinder({}, {})", r, h),
            Shape::HalfSpace(normal, offset) => {
                format!("Shape.half_space([{}, {}, {}], {})", normal.x, normal.y, normal.z, offset)
            }
        }
    }
}

/// Planning scene with collision objects.
///
/// Usage:
///     scene = kinetic.Scene(robot)
///     scene.add("table", kinetic.Shape.cuboid(1.0, 0.6, 0.02), pose_4x4)
///     colliding = scene.check_collision(joint_values)
#[pyclass(name = "Scene")]
pub struct PyScene {
    pub(crate) inner: Scene,
}

#[pymethods]
impl PyScene {
    /// Create a scene for the given robot.
    #[new]
    #[pyo3(text_signature = "(robot)")]
    fn new(robot: &PyRobot) -> PyResult<Self> {
        let scene = Scene::new(&robot.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyScene { inner: scene })
    }

    /// Add a collision object to the scene.
    ///
    /// Args:
    ///     name: unique identifier for the object
    ///     shape: Shape object (cuboid, sphere, or cylinder)
    ///     pose: (4,4) numpy array for object pose
    #[pyo3(text_signature = "(self, name, shape, pose)")]
    fn add(
        &mut self,
        name: &str,
        shape: &PyShape,
        pose: numpy::PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let iso = convert::numpy_4x4_to_isometry(pose)?;
        self.inner.add(name, shape.inner.clone(), iso);
        Ok(())
    }

    /// Remove a collision object from the scene.
    #[pyo3(text_signature = "(self, name)")]
    fn remove(&mut self, name: &str) -> bool {
        self.inner.remove(name).is_some()
    }

    /// Clear all objects from the scene.
    #[pyo3(text_signature = "(self)")]
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Attach an object to a robot link.
    #[pyo3(text_signature = "(self, name, shape, grasp_transform, link_name)")]
    fn attach(
        &mut self,
        name: &str,
        shape: &PyShape,
        grasp_transform: numpy::PyReadonlyArray2<'_, f64>,
        link_name: &str,
    ) -> PyResult<()> {
        let iso = convert::numpy_4x4_to_isometry(grasp_transform)?;
        self.inner
            .attach(name, shape.inner.clone(), iso, link_name);
        Ok(())
    }

    /// Detach an object from the robot, placing it at the given world pose.
    #[pyo3(text_signature = "(self, name, place_pose)")]
    fn detach(&mut self, name: &str, place_pose: numpy::PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
        let iso = convert::numpy_4x4_to_isometry(place_pose)?;
        Ok(self.inner.detach(name, iso))
    }

    /// Check if the robot is in collision at the given joint configuration.
    #[pyo3(text_signature = "(self, joint_values)")]
    fn check_collision(&self, joint_values: PyReadonlyArray1<'_, f64>) -> PyResult<bool> {
        let joints = convert::numpy_to_vec(joint_values)?;
        self.inner
            .check_collision(&joints)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Minimum distance from robot to nearest obstacle.
    #[pyo3(text_signature = "(self, joint_values)")]
    fn min_distance(&self, joint_values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let joints = convert::numpy_to_vec(joint_values)?;
        self.inner
            .min_distance_to_robot(&joints)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Number of collision objects in the scene.
    #[getter]
    fn num_objects(&self) -> usize {
        self.inner.num_objects()
    }

    /// Number of attached objects.
    #[getter]
    fn num_attached(&self) -> usize {
        self.inner.num_attached()
    }

    /// Number of octrees in the scene.
    #[getter]
    fn num_octrees(&self) -> usize {
        self.inner.num_octrees()
    }

    /// Update an octree with a new point cloud.
    ///
    /// Creates the octree if it doesn't exist. Uses ray-casting from
    /// sensor_origin to clear free space.
    ///
    /// Args:
    ///     name: octree identifier (e.g., "lidar", "depth_camera")
    ///     points: (N, 3) numpy array of 3D points
    ///     sensor_origin: [x, y, z] sensor position for ray-casting
    #[pyo3(text_signature = "(self, name, points, sensor_origin)")]
    fn update_octree(
        &mut self,
        name: &str,
        points: numpy::PyReadonlyArray2<'_, f64>,
        sensor_origin: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected (N, 3) array, got shape {:?}",
                shape
            )));
        }

        let origin_vec = convert::numpy_to_vec(sensor_origin)?;
        if origin_vec.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sensor_origin must be a 3-element array [x, y, z]",
            ));
        }
        let origin = [origin_vec[0], origin_vec[1], origin_vec[2]];

        let flat = points.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;

        let pts: Vec<[f64; 3]> = flat
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        self.inner.update_octree(name, &pts, &origin);
        Ok(())
    }

    /// Add a point cloud as collision obstacles.
    ///
    /// Args:
    ///     name: identifier for this point cloud source
    ///     points: (N, 3) numpy array of 3D points
    ///     sphere_radius: collision sphere radius per point (default: 0.01m)
    ///     max_points: downsample to this many points (default: 100000)
    #[pyo3(text_signature = "(self, name, points, sphere_radius=0.01, max_points=100000)")]
    #[pyo3(signature = (name, points, sphere_radius=0.01, max_points=100000))]
    fn add_pointcloud(
        &mut self,
        name: &str,
        points: numpy::PyReadonlyArray2<'_, f64>,
        sphere_radius: f64,
        max_points: usize,
    ) -> PyResult<()> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected (N, 3) array, got shape {:?}",
                shape
            )));
        }

        let flat = points.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
        })?;

        let pts: Vec<[f64; 3]> = flat
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let config = kinetic_scene::PointCloudConfig {
            sphere_radius,
            max_points,
            ..Default::default()
        };

        self.inner.add_pointcloud(name, &pts, config);
        Ok(())
    }

    /// Update scene from a depth image.
    ///
    /// Unprojects depth pixels to 3D points and adds as collision obstacles.
    ///
    /// Args:
    ///     name: identifier for this depth source
    ///     depth: (H, W) numpy array of depth values in meters (f32)
    ///     fx, fy, cx, cy: camera intrinsic parameters
    ///     camera_pose: (4, 4) numpy array for camera-to-world transform
    ///     min_depth: ignore pixels below this depth (default: 0.1m)
    ///     max_depth: ignore pixels above this depth (default: 5.0m)
    ///     sphere_radius: collision sphere radius per point (default: 0.01m)
    #[pyo3(text_signature = "(self, name, depth, fx, fy, cx, cy, camera_pose, min_depth=0.1, max_depth=5.0, sphere_radius=0.01)")]
    #[pyo3(signature = (name, depth, fx, fy, cx, cy, camera_pose, min_depth=0.1, max_depth=5.0, sphere_radius=0.01))]
    #[allow(clippy::too_many_arguments)]
    fn update_from_depth(
        &mut self,
        name: &str,
        depth: numpy::PyReadonlyArray2<'_, f32>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        camera_pose: numpy::PyReadonlyArray2<'_, f64>,
        min_depth: f64,
        max_depth: f64,
        sphere_radius: f64,
    ) -> PyResult<()> {
        let shape = depth.shape();
        let height = shape[0];
        let width = shape[1];

        let depth_flat = depth.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Depth array not contiguous: {}", e))
        })?;

        let cam_iso = convert::numpy_4x4_to_isometry(camera_pose)?;

        let intrinsics = kinetic_scene::CameraIntrinsics { fx, fy, cx, cy };

        let depth_config = kinetic_scene::DepthConfig {
            min_depth,
            max_depth,
            ..Default::default()
        };

        let pc_config = kinetic_scene::PointCloudConfig {
            sphere_radius,
            ..Default::default()
        };

        self.inner.update_from_depth(
            name,
            depth_flat,
            width,
            height,
            &intrinsics,
            &cam_iso,
            &depth_config,
            pc_config,
        );
        Ok(())
    }

    /// Allow collision between two named objects (skip collision checking).
    #[pyo3(text_signature = "(self, name_a, name_b)")]
    fn allow_collision(&mut self, name_a: &str, name_b: &str) {
        self.inner.allow_collision(name_a, name_b);
    }

    /// Disallow collision (re-enable collision checking between two objects).
    #[pyo3(text_signature = "(self, name_a, name_b)")]
    fn disallow_collision(&mut self, name_a: &str, name_b: &str) {
        self.inner.disallow_collision(name_a, name_b);
    }

    fn __repr__(&self) -> String {
        format!(
            "Scene(objects={}, attached={}, octrees={})",
            self.inner.num_objects(),
            self.inner.num_attached(),
            self.inner.num_octrees()
        )
    }
}
