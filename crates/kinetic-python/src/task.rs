//! PyTask, PyGraspGenerator — Python bindings for task planning and grasp generation.

use pyo3::prelude::*;

use std::sync::Arc;

use kinetic_grasp::{GraspConfig, GraspGenerator, GraspMetric, GripperType};
use kinetic_scene::Scene;
use kinetic_task::{Approach, PickConfig, PlaceConfig, Task};
use nalgebra::Vector3;

use crate::convert;
use crate::robot::PyRobot;
use crate::scene::PyScene;

/// Gripper type for grasp generation.
///
/// Usage:
///     gripper = kinetic.GripperType.parallel(max_opening=0.08, finger_depth=0.03)
///     gripper = kinetic.GripperType.suction(cup_radius=0.02)
#[pyclass(name = "GripperType")]
#[derive(Clone)]
pub struct PyGripperType {
    pub(crate) inner: GripperType,
}

#[pymethods]
impl PyGripperType {
    #[staticmethod]
    #[pyo3(text_signature = "(max_opening, finger_depth)")]
    fn parallel(max_opening: f64, finger_depth: f64) -> Self {
        PyGripperType {
            inner: GripperType::Parallel {
                max_opening,
                finger_depth,
            },
        }
    }

    #[staticmethod]
    #[pyo3(text_signature = "(cup_radius)")]
    fn suction(cup_radius: f64) -> Self {
        PyGripperType {
            inner: GripperType::Suction { cup_radius },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            GripperType::Parallel { max_opening, finger_depth } => {
                format!("GripperType.parallel({:.3}, {:.3})", max_opening, finger_depth)
            }
            GripperType::Suction { cup_radius } => {
                format!("GripperType.suction({:.3})", cup_radius)
            }
        }
    }
}

/// A grasp candidate with pose and quality score.
#[pyclass(name = "Grasp")]
pub struct PyGrasp {
    #[pyo3(get)]
    pub quality_score: f64,
    #[pyo3(get)]
    pub approach_vector: [f64; 3],
    grasp_pose: nalgebra::Isometry3<f64>,
}

#[pymethods]
impl PyGrasp {
    #[pyo3(text_signature = "(self)")]
    fn pose<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        convert::isometry_to_numpy_4x4(py, &self.grasp_pose)
    }

    fn __repr__(&self) -> String {
        format!("Grasp(quality={:.3})", self.quality_score)
    }
}

/// Grasp candidate generator.
///
/// Usage:
///     gen = kinetic.GraspGenerator(kinetic.GripperType.parallel(0.08, 0.03))
///     grasps = gen.from_shape("cylinder", [0.04, 0.12], object_pose_4x4)
#[pyclass(name = "GraspGenerator")]
pub struct PyGraspGenerator {
    generator: GraspGenerator,
}

#[pymethods]
impl PyGraspGenerator {
    #[new]
    #[pyo3(text_signature = "(gripper)")]
    fn new(gripper: &PyGripperType) -> Self {
        PyGraspGenerator {
            generator: GraspGenerator::new(gripper.inner.clone()),
        }
    }

    /// Generate grasp candidates for a shape at a given pose.
    #[pyo3(text_signature = "(self, shape_type, dimensions, object_pose, num_candidates=100)")]
    #[pyo3(signature = (shape_type, dimensions, object_pose, num_candidates=100))]
    fn from_shape(
        &self,
        shape_type: &str,
        dimensions: Vec<f64>,
        object_pose: numpy::PyReadonlyArray2<'_, f64>,
        num_candidates: usize,
    ) -> PyResult<Vec<PyGrasp>> {
        let pose = convert::numpy_4x4_to_isometry(object_pose)?;

        let shape = match shape_type.to_lowercase().as_str() {
            "cuboid" | "box" => {
                if dimensions.len() != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Cuboid requires 3 dimensions [half_x, half_y, half_z]",
                    ));
                }
                kinetic_scene::Shape::Cuboid(dimensions[0], dimensions[1], dimensions[2])
            }
            "cylinder" => {
                if dimensions.len() != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Cylinder requires 2 dimensions [radius, half_height]",
                    ));
                }
                kinetic_scene::Shape::Cylinder(dimensions[0], dimensions[1])
            }
            "sphere" => {
                if dimensions.len() != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Sphere requires 1 dimension [radius]",
                    ));
                }
                kinetic_scene::Shape::Sphere(dimensions[0])
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown shape '{}'. Use 'cuboid', 'cylinder', or 'sphere'",
                    shape_type
                )))
            }
        };

        let config = GraspConfig {
            num_candidates,
            approach_axis: -Vector3::z(),
            rank_by: GraspMetric::ForceClosureQuality,
            check_collision: None,
            check_reachability: None,
        };

        let grasps = self
            .generator
            .from_shape(&shape, &pose, config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(grasps
            .into_iter()
            .map(|g| PyGrasp {
                quality_score: g.quality,
                approach_vector: [
                    g.approach_direction.x,
                    g.approach_direction.y,
                    g.approach_direction.z,
                ],
                grasp_pose: g.grasp_pose,
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "GraspGenerator(<gripper>)".to_string()
    }
}

/// Approach motion specification.
///
/// Usage:
///     approach = kinetic.Approach([0, 0, -1], 0.10)  # 10cm down
#[pyclass(name = "Approach")]
#[derive(Clone)]
pub struct PyApproach {
    pub(crate) inner: Approach,
}

#[pymethods]
impl PyApproach {
    #[new]
    #[pyo3(text_signature = "(direction, distance)")]
    fn new(direction: [f64; 3], distance: f64) -> Self {
        PyApproach {
            inner: Approach {
                direction: Vector3::new(direction[0], direction[1], direction[2]),
                distance,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Approach([{:.2}, {:.2}, {:.2}], {:.3}m)",
            self.inner.direction.x, self.inner.direction.y, self.inner.direction.z, self.inner.distance
        )
    }
}

/// High-level task solution containing planned stages.
#[pyclass(name = "TaskSolution")]
pub struct PyTaskSolution {
    #[pyo3(get)]
    pub num_stages: usize,
    #[pyo3(get)]
    pub total_duration: f64,
    #[pyo3(get)]
    pub total_planning_time: f64,
    #[pyo3(get)]
    pub stage_names: Vec<String>,
}

#[pymethods]
impl PyTaskSolution {
    fn __repr__(&self) -> String {
        format!(
            "TaskSolution(stages={}, duration={:.3}s)",
            self.num_stages, self.total_duration
        )
    }
}

/// Task planner for pick, place, move, and sequence operations.
///
/// Usage:
///     task = kinetic.Task.move_to(robot, goal)
///     task = kinetic.Task.gripper(0.08)
///     task = kinetic.Task.sequence([task1, task2, task3])
///     solution = task.plan(start_joints)
#[pyclass(name = "Task")]
pub struct PyTask {
    inner: Option<Task>,
}

#[pymethods]
impl PyTask {
    /// Create a move-to task.
    #[staticmethod]
    #[pyo3(text_signature = "(robot, goal)")]
    fn move_to(robot: &PyRobot, goal: &crate::planner::PyGoal) -> Self {
        PyTask {
            inner: Some(Task::MoveTo {
                robot: robot.inner.clone(),
                goal: goal.inner.clone(),
            }),
        }
    }

    /// Create a gripper command task.
    #[staticmethod]
    #[pyo3(text_signature = "(width)")]
    fn gripper(width: f64) -> Self {
        PyTask {
            inner: Some(Task::Gripper { width }),
        }
    }

    /// Create a pick task.
    ///
    /// Args:
    ///     robot: Robot model
    ///     scene: Scene with the object to pick
    ///     object_name: name of the object in the scene
    ///     grasp_poses: list of 4x4 numpy grasp pose candidates
    ///     approach: Approach motion before grasping
    ///     retreat: Approach motion after grasping
    ///     gripper_open: gripper width when open (default: 0.08)
    ///     gripper_close: gripper width when closed (default: 0.04)
    #[staticmethod]
    #[pyo3(text_signature = "(robot, scene, object_name, grasp_poses, approach, retreat, gripper_open=0.08, gripper_close=0.04)")]
    #[pyo3(signature = (robot, scene, object_name, grasp_poses, approach, retreat, gripper_open=0.08, gripper_close=0.04))]
    fn pick(
        robot: &PyRobot,
        scene: &PyScene,
        object_name: &str,
        grasp_poses: Vec<numpy::PyReadonlyArray2<'_, f64>>,
        approach: &PyApproach,
        retreat: &PyApproach,
        gripper_open: f64,
        gripper_close: f64,
    ) -> PyResult<Self> {
        let poses: Vec<nalgebra::Isometry3<f64>> = grasp_poses
            .into_iter()
            .map(|p| convert::numpy_4x4_to_isometry(p))
            .collect::<PyResult<_>>()?;

        // Build a fresh Scene from the same robot to get an Arc<Scene>
        let scene_arc = build_scene_arc(&robot.inner, &scene.inner)?;
        Ok(PyTask {
            inner: Some(Task::pick(
                &robot.inner,
                &scene_arc,
                PickConfig {
                    object: object_name.to_string(),
                    grasp_poses: poses,
                    approach: approach.inner.clone(),
                    retreat: retreat.inner.clone(),
                    gripper_open,
                    gripper_close,
                },
            )),
        })
    }

    /// Create a place task.
    ///
    /// Args:
    ///     robot: Robot model
    ///     scene: Scene with the target location
    ///     object_name: name of the object being placed
    ///     target_pose: 4x4 numpy target pose for the object
    ///     approach: Approach motion to place location
    ///     retreat: Approach motion after placing
    ///     gripper_open: gripper width to release (default: 0.08)
    #[staticmethod]
    #[pyo3(text_signature = "(robot, scene, object_name, target_pose, approach, retreat, gripper_open=0.08)")]
    #[pyo3(signature = (robot, scene, object_name, target_pose, approach, retreat, gripper_open=0.08))]
    fn place(
        robot: &PyRobot,
        scene: &PyScene,
        object_name: &str,
        target_pose: numpy::PyReadonlyArray2<'_, f64>,
        approach: &PyApproach,
        retreat: &PyApproach,
        gripper_open: f64,
    ) -> PyResult<Self> {
        let pose = convert::numpy_4x4_to_isometry(target_pose)?;
        let scene_arc = build_scene_arc(&robot.inner, &scene.inner)?;
        Ok(PyTask {
            inner: Some(Task::place(
                &robot.inner,
                &scene_arc,
                PlaceConfig {
                    object: object_name.to_string(),
                    target_pose: pose,
                    approach: approach.inner.clone(),
                    retreat: retreat.inner.clone(),
                    gripper_open,
                },
            )),
        })
    }

    /// Create a task sequence from a list of tasks.
    ///
    /// Note: consumes the tasks — they cannot be reused after sequencing.
    #[staticmethod]
    #[pyo3(text_signature = "(tasks)")]
    fn sequence(tasks: Vec<Bound<'_, PyTask>>) -> PyResult<Self> {
        let mut inner_tasks = Vec::with_capacity(tasks.len());
        for task in tasks {
            let mut task_mut = task.borrow_mut();
            let t = task_mut.inner.take().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Task already consumed (used in another sequence?)",
                )
            })?;
            inner_tasks.push(t);
        }
        Ok(PyTask {
            inner: Some(Task::sequence(inner_tasks)),
        })
    }

    /// Plan the task from a starting joint configuration.
    #[pyo3(text_signature = "(self, start_joints)")]
    fn plan(
        &self,
        start_joints: numpy::PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyTaskSolution> {
        let start = convert::numpy_to_vec(start_joints)?;
        let task = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Task already consumed")
        })?;

        let solution = task
            .plan(&start)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let stage_names: Vec<String> = solution.stages.iter().map(|s| s.name.clone()).collect();

        Ok(PyTaskSolution {
            num_stages: solution.stages.len(),
            total_duration: solution.total_duration.as_secs_f64(),
            total_planning_time: solution.total_planning_time.as_secs_f64(),
            stage_names,
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(Task::MoveTo { .. }) => "Task.move_to(...)".to_string(),
            Some(Task::CartesianMove { .. }) => "Task.cartesian_move(...)".to_string(),
            Some(Task::Pick { .. }) => "Task.pick(...)".to_string(),
            Some(Task::Place { .. }) => "Task.place(...)".to_string(),
            Some(Task::Sequence(tasks)) => format!("Task.sequence({} tasks)", tasks.len()),
            Some(Task::Gripper { width }) => format!("Task.gripper({:.3})", width),
            None => "Task(<consumed>)".to_string(),
        }
    }
}

/// Build an Arc<Scene> by copying objects from the Python scene.
fn build_scene_arc(
    robot: &Arc<kinetic_robot::Robot>,
    py_scene: &Scene,
) -> PyResult<Arc<Scene>> {
    let mut scene = Scene::new(robot)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    for obj in py_scene.objects_iter() {
        scene.add(&obj.name, obj.shape.clone(), obj.pose);
    }
    Ok(Arc::new(scene))
}
