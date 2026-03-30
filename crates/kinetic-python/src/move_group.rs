//! MoveGroup Python API — high-level orchestrator for planning + execution.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

use std::collections::HashMap;
use std::sync::Arc;

use kinetic_core::{Goal, JointValues, Pose};
use kinetic_planning::Planner;
use kinetic_robot::Robot;

use crate::convert::numpy_to_vec;
use crate::trajectory::PyTrajectory;

/// MoveGroup: high-level Python API for motion planning.
#[pyclass(name = "MoveGroup")]
pub struct PyMoveGroup {
    robot: Arc<Robot>,
    planner: Planner,
    current_joints: Vec<f64>,
    target_joints: Option<Vec<f64>>,
    target_pose: Option<Pose>,
    target_name: Option<String>,
    planning_time: f64,
    velocity_scale: f64,
    acceleration_scale: f64,
    planner_id: String,
    attached_objects: Vec<(String, String)>,
    named_poses: HashMap<String, Vec<f64>>,
}

#[pymethods]
impl PyMoveGroup {
    #[new]
    fn new(urdf: &str) -> PyResult<Self> {
        let robot = Robot::from_urdf_string(urdf)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let robot = Arc::new(robot);
        let planner = Planner::new(&robot)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let dof = robot.dof;

        // Auto-generate home pose
        let home: Vec<f64> = robot.joint_limits.iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();

        let mut named_poses = HashMap::new();
        named_poses.insert("home".into(), home);
        named_poses.insert("zeros".into(), vec![0.0; dof]);

        Ok(Self {
            robot,
            planner,
            current_joints: vec![0.0; dof],
            target_joints: None,
            target_pose: None,
            target_name: None,
            planning_time: 5.0,
            velocity_scale: 1.0,
            acceleration_scale: 1.0,
            planner_id: "rrt_connect".into(),
            attached_objects: Vec::new(),
            named_poses,
        })
    }

    #[getter]
    fn dof(&self) -> usize { self.robot.dof }

    #[getter]
    fn robot_name(&self) -> &str { &self.robot.name }

    fn set_joint_state(&mut self, joints: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.current_joints = numpy_to_vec(joints)?;
        Ok(())
    }

    fn get_joint_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.current_joints.clone())
    }

    fn set_joint_target(&mut self, target: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.target_joints = Some(numpy_to_vec(target)?);
        self.target_pose = None;
        self.target_name = None;
        Ok(())
    }

    fn set_pose_target(&mut self, position: [f64; 3], orientation: [f64; 4]) {
        let quat = nalgebra::UnitQuaternion::from_quaternion(
            nalgebra::Quaternion::new(orientation[3], orientation[0], orientation[1], orientation[2])
        );
        let iso = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(position[0], position[1], position[2]),
            quat,
        );
        self.target_pose = Some(Pose(iso));
        self.target_joints = None;
        self.target_name = None;
    }

    fn set_named_target(&mut self, name: &str) -> PyResult<()> {
        if self.named_poses.contains_key(name) {
            self.target_name = Some(name.to_string());
            self.target_joints = None;
            self.target_pose = None;
            Ok(())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!("Unknown pose: {}", name)))
        }
    }

    fn remember_pose(&mut self, name: &str, joints: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.named_poses.insert(name.to_string(), numpy_to_vec(joints)?);
        Ok(())
    }

    fn get_named_poses(&self) -> Vec<String> {
        self.named_poses.keys().cloned().collect()
    }

    fn set_planning_time(&mut self, seconds: f64) { self.planning_time = seconds; }
    fn set_max_velocity_scaling_factor(&mut self, f: f64) { self.velocity_scale = f.clamp(0.0, 1.0); }
    fn set_max_acceleration_scaling_factor(&mut self, f: f64) { self.acceleration_scale = f.clamp(0.0, 1.0); }
    fn set_planner_id(&mut self, id: &str) { self.planner_id = id.to_string(); }

    fn plan(&self) -> PyResult<Option<PyTrajectory>> {
        let goal = self.resolve_goal()?;
        match self.planner.plan(&self.current_joints, &goal) {
            Ok(result) => Ok(Some(PyTrajectory::from_path(result.waypoints))),
            Err(_) => Ok(None),
        }
    }

    fn go(&mut self) -> PyResult<bool> {
        let goal = self.resolve_goal()?;
        match self.planner.plan(&self.current_joints, &goal) {
            Ok(result) => {
                if let Some(last) = result.waypoints.last() {
                    self.current_joints = last.clone();
                }
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    fn stop(&self) {}

    fn attach_object(&mut self, object_name: &str, link_name: &str) {
        self.attached_objects.push((object_name.to_string(), link_name.to_string()));
    }

    fn detach_object(&mut self, object_name: &str) {
        self.attached_objects.retain(|(name, _)| name != object_name);
    }

    fn get_attached_objects(&self) -> Vec<(String, String)> {
        self.attached_objects.clone()
    }

    fn get_joint_names(&self) -> Vec<String> {
        self.robot.joints.iter()
            .filter(|j| j.joint_type != kinetic_robot::JointType::Fixed)
            .map(|j| j.name.clone())
            .collect()
    }

    fn get_joint_limits(&self) -> Vec<(f64, f64)> {
        self.robot.joint_limits.iter().map(|l| (l.lower, l.upper)).collect()
    }

    fn get_param(&self, key: &str) -> PyResult<f64> {
        match key {
            "planning_time" => Ok(self.planning_time),
            "velocity_scale" => Ok(self.velocity_scale),
            "acceleration_scale" => Ok(self.acceleration_scale),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(format!("Unknown param: {}", key))),
        }
    }

    fn set_param(&mut self, key: &str, value: f64) -> PyResult<()> {
        match key {
            "planning_time" => { self.planning_time = value; Ok(()) }
            "velocity_scale" => { self.velocity_scale = value.clamp(0.0, 1.0); Ok(()) }
            "acceleration_scale" => { self.acceleration_scale = value.clamp(0.0, 1.0); Ok(()) }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(format!("Unknown param: {}", key))),
        }
    }

    fn __repr__(&self) -> String {
        format!("MoveGroup(robot='{}', dof={}, planning_time={:.1}s)", self.robot.name, self.robot.dof, self.planning_time)
    }
}

impl PyMoveGroup {
    fn resolve_goal(&self) -> PyResult<Goal> {
        if let Some(ref joints) = self.target_joints {
            Ok(Goal::Joints(JointValues::new(joints.clone())))
        } else if let Some(ref name) = self.target_name {
            self.named_poses.get(name)
                .map(|j| Goal::Joints(JointValues::new(j.clone())))
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown pose: {}", name)))
        } else if let Some(ref pose) = self.target_pose {
            Ok(Goal::Pose(pose.clone()))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("No target set"))
        }
    }
}

