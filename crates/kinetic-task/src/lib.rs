//! Task planning primitives for KINETIC.
//!
//! Composes motion primitives into multi-step manipulation sequences:
//! pick, place, move_to, Cartesian move, gripper commands, and
//! arbitrary task sequencing (MoveIt Task Constructor equivalent).
//!
//! # Example
//!
//! ```ignore
//! use kinetic_task::{Task, PickConfig, PlaceConfig, Approach};
//! use nalgebra::{Isometry3, Vector3};
//!
//! let pick = Task::pick(&robot, &scene, PickConfig {
//!     object: "cup".into(),
//!     grasp_poses: vec![Isometry3::identity()],
//!     approach: Approach::linear(-Vector3::z(), 0.10),
//!     retreat: Approach::linear(Vector3::z(), 0.05),
//!     gripper_open: 0.08,
//!     gripper_close: 0.04,
//! });
//!
//! let full_task = Task::sequence(vec![
//!     Task::move_to(&robot, Goal::Named("home".into())),
//!     pick,
//!     Task::move_to(&robot, Goal::Named("home".into())),
//! ]);
//!
//! let solution = full_task.plan(&[0.0; 6])?;
//! ```

mod planners;

use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::{Isometry3, Translation3, Vector3};

use kinetic_core::{Goal, KineticError};
use kinetic_grasp::{GraspConfig, GraspGenerator, GripperType};
use kinetic_planning::CartesianConfig;
use kinetic_robot::Robot;
use kinetic_scene::{Scene, Shape};
use kinetic_trajectory::{TimedTrajectory, TrajectoryValidator, TrajectoryViolation};

use crate::planners::{plan_cartesian_move, plan_move_to, plan_pick, plan_place};

/// Error type for task planning.
#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error("No valid grasp found for object '{0}'")]
    NoValidGrasp(String),
    #[error("Planning failed for stage '{stage}': {reason}")]
    PlanningFailed { stage: String, reason: String },
    #[error("Object '{0}' not found in scene")]
    ObjectNotFound(String),
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] KineticError),
    #[error("Trajectory error: {0}")]
    Trajectory(String),
}

/// Approach/retreat motion specification.
#[derive(Debug, Clone)]
pub struct Approach {
    /// Direction of motion (unit vector in world frame).
    pub direction: Vector3<f64>,
    /// Distance to travel in meters.
    pub distance: f64,
}

impl Approach {
    /// Create a linear approach/retreat motion.
    pub fn linear(direction: Vector3<f64>, distance: f64) -> Self {
        Self {
            direction: direction.normalize(),
            distance,
        }
    }

    /// Compute the offset pose relative to a target pose.
    fn offset_pose(&self, target: &Isometry3<f64>) -> Isometry3<f64> {
        let offset = self.direction * self.distance;
        Isometry3::from_parts(
            Translation3::from(target.translation.vector - offset),
            target.rotation,
        )
    }
}

/// Configuration for pick operations.
#[derive(Debug, Clone)]
pub struct PickConfig {
    /// Object name in the scene to pick.
    pub object: String,
    /// Candidate grasp poses for the gripper TCP.
    pub grasp_poses: Vec<Isometry3<f64>>,
    /// Approach motion before grasping (move toward object).
    pub approach: Approach,
    /// Retreat motion after grasping (move away with object).
    pub retreat: Approach,
    /// Gripper width before grasp (fully open).
    pub gripper_open: f64,
    /// Gripper width during grasp (closed on object).
    pub gripper_close: f64,
}

/// Configuration for place operations.
#[derive(Debug, Clone)]
pub struct PlaceConfig {
    /// Object name to place.
    pub object: String,
    /// Target pose for the object.
    pub target_pose: Isometry3<f64>,
    /// Approach motion to place location.
    pub approach: Approach,
    /// Retreat motion after placing.
    pub retreat: Approach,
    /// Gripper width to release object.
    pub gripper_open: f64,
}

/// A composable task planning primitive.
///
/// Tasks are composed into sequences and planned recursively.
/// Each task variant represents a single motion or action stage.
pub enum Task {
    /// Move to a goal configuration or pose via RRT.
    MoveTo { robot: Arc<Robot>, goal: Goal },
    /// Move in a straight Cartesian line.
    CartesianMove {
        robot: Arc<Robot>,
        target_pose: Isometry3<f64>,
        config: CartesianConfig,
    },
    /// Pick an object from the scene.
    Pick {
        robot: Arc<Robot>,
        scene: Arc<Scene>,
        config: PickConfig,
    },
    /// Place an object in the scene.
    Place {
        robot: Arc<Robot>,
        scene: Arc<Scene>,
        config: PlaceConfig,
    },
    /// Sequence of tasks executed in order.
    Sequence(Vec<Task>),
    /// Gripper command (open/close to specified width).
    Gripper { width: f64 },
}

// Scene doesn't implement Debug
impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::MoveTo { goal, .. } => f.debug_struct("MoveTo").field("goal", goal).finish(),
            Task::CartesianMove { target_pose, .. } => f
                .debug_struct("CartesianMove")
                .field("target_pose", target_pose)
                .finish(),
            Task::Pick { config, .. } => f
                .debug_struct("Pick")
                .field("object", &config.object)
                .finish(),
            Task::Place { config, .. } => f
                .debug_struct("Place")
                .field("object", &config.object)
                .finish(),
            Task::Sequence(tasks) => f
                .debug_struct("Sequence")
                .field("len", &tasks.len())
                .finish(),
            Task::Gripper { width } => f.debug_struct("Gripper").field("width", width).finish(),
        }
    }
}

/// Result of planning a complete task.
#[derive(Debug, Clone)]
pub struct TaskSolution {
    /// Ordered list of planned stages.
    pub stages: Vec<StageSolution>,
    /// Total expected execution duration.
    pub total_duration: Duration,
    /// Total planning computation time.
    pub total_planning_time: Duration,
}

impl TaskSolution {
    /// Get the final joint configuration from the last trajectory stage.
    pub fn final_joints(&self) -> Option<Vec<f64>> {
        for stage in self.stages.iter().rev() {
            if let Some(traj) = &stage.trajectory {
                if let Some(last_wp) = traj.waypoints.last() {
                    return Some(last_wp.positions.clone());
                }
            }
        }
        None
    }

    /// Validate all trajectory stages against joint limits.
    ///
    /// Returns a map of stage name → violations for any stage that fails.
    /// An empty map means all trajectories are safe.
    pub fn validate_trajectories(
        &self,
        validator: &TrajectoryValidator,
    ) -> Vec<(String, Vec<TrajectoryViolation>)> {
        let mut results = Vec::new();
        for stage in &self.stages {
            if let Some(traj) = &stage.trajectory {
                if let Err(violations) = validator.validate(traj) {
                    results.push((stage.name.clone(), violations));
                }
            }
        }
        results
    }
}

/// A single stage in a task solution.
#[derive(Debug, Clone)]
pub struct StageSolution {
    /// Human-readable stage name.
    pub name: String,
    /// Trajectory for this stage (None for gripper/scene commands).
    pub trajectory: Option<TimedTrajectory>,
    /// Gripper command (target width), if this is a gripper stage.
    pub gripper_command: Option<f64>,
    /// Scene modification for this stage.
    pub scene_diff: Option<SceneDiff>,
}

/// Scene modification that occurs during a task stage.
#[derive(Debug, Clone)]
pub enum SceneDiff {
    /// Attach an object to a robot link.
    Attach {
        object: String,
        link: String,
        shape: Shape,
        grasp_transform: Isometry3<f64>,
    },
    /// Detach an object from the robot to a world pose.
    Detach {
        object: String,
        place_pose: Isometry3<f64>,
    },
}

impl Task {
    /// Create a move-to task.
    pub fn move_to(robot: &Arc<Robot>, goal: Goal) -> Self {
        Task::MoveTo {
            robot: Arc::clone(robot),
            goal,
        }
    }

    /// Create a Cartesian move task.
    pub fn cartesian_move(
        robot: &Arc<Robot>,
        target_pose: Isometry3<f64>,
        config: CartesianConfig,
    ) -> Self {
        Task::CartesianMove {
            robot: Arc::clone(robot),
            target_pose,
            config,
        }
    }

    /// Create a pick task.
    pub fn pick(robot: &Arc<Robot>, scene: &Arc<Scene>, config: PickConfig) -> Self {
        Task::Pick {
            robot: Arc::clone(robot),
            scene: Arc::clone(scene),
            config,
        }
    }

    /// Create a pick task with auto-generated grasps from the scene object's geometry.
    ///
    /// Uses [`GraspGenerator`] to produce grasp candidates from the object's shape,
    /// then plans the pick using the best candidates.
    ///
    /// The `pick_config` should have an empty `grasp_poses` vec — it will be filled
    /// automatically from the generated grasps.
    #[allow(clippy::too_many_arguments)]
    pub fn pick_auto(
        robot: &Arc<Robot>,
        scene: &Arc<Scene>,
        gripper: GripperType,
        mut pick_config: PickConfig,
    ) -> std::result::Result<Self, TaskError> {
        let obj = scene
            .get_object(&pick_config.object)
            .ok_or_else(|| TaskError::ObjectNotFound(pick_config.object.clone()))?;

        let generator = GraspGenerator::new(gripper);
        let grasp_config = GraspConfig {
            num_candidates: 50,
            check_collision: Some(Arc::clone(scene)),
            ..Default::default()
        };

        let candidates = generator
            .from_shape(&obj.shape, &obj.pose, grasp_config)
            .map_err(|e| TaskError::NoValidGrasp(format!("{}: {}", pick_config.object, e)))?;

        pick_config.grasp_poses = candidates.into_iter().map(|g| g.grasp_pose).collect();

        Ok(Task::Pick {
            robot: Arc::clone(robot),
            scene: Arc::clone(scene),
            config: pick_config,
        })
    }

    /// Create a place task.
    pub fn place(robot: &Arc<Robot>, scene: &Arc<Scene>, config: PlaceConfig) -> Self {
        Task::Place {
            robot: Arc::clone(robot),
            scene: Arc::clone(scene),
            config,
        }
    }

    /// Create a task sequence.
    pub fn sequence(tasks: Vec<Task>) -> Self {
        Task::Sequence(tasks)
    }

    /// Create a gripper command.
    pub fn gripper(width: f64) -> Self {
        Task::Gripper { width }
    }

    /// Plan the complete task from a starting joint configuration.
    ///
    /// Returns a `TaskSolution` with all stages planned, including
    /// trajectories, gripper commands, and scene modifications.
    pub fn plan(&self, start_joints: &[f64]) -> std::result::Result<TaskSolution, TaskError> {
        let plan_start = Instant::now();
        let stages = self.plan_stages(start_joints)?;

        let total_duration: Duration = stages
            .iter()
            .filter_map(|s| s.trajectory.as_ref())
            .map(|t| t.duration)
            .sum();

        Ok(TaskSolution {
            stages,
            total_duration,
            total_planning_time: plan_start.elapsed(),
        })
    }

    /// Recursively plan stages for this task.
    fn plan_stages(
        &self,
        start_joints: &[f64],
    ) -> std::result::Result<Vec<StageSolution>, TaskError> {
        match self {
            Task::MoveTo { robot, goal } => plan_move_to(robot, start_joints, goal),

            Task::CartesianMove {
                robot,
                target_pose,
                config,
            } => plan_cartesian_move(robot, start_joints, target_pose, config),

            Task::Pick {
                robot,
                scene,
                config,
            } => plan_pick(robot, scene, start_joints, config),

            Task::Place {
                robot,
                scene,
                config,
            } => plan_place(robot, scene, start_joints, config),

            Task::Sequence(tasks) => {
                let mut all_stages = Vec::new();
                let mut current_joints = start_joints.to_vec();

                for task in tasks {
                    let stages = task.plan_stages(&current_joints)?;

                    // Update current joints to the end of the last trajectory
                    for stage in stages.iter().rev() {
                        if let Some(traj) = &stage.trajectory {
                            if let Some(last_wp) = traj.waypoints.last() {
                                current_joints = last_wp.positions.clone();
                                break;
                            }
                        }
                    }

                    all_stages.extend(stages);
                }

                Ok(all_stages)
            }

            Task::Gripper { width } => Ok(vec![StageSolution {
                name: format!("gripper_{}", if *width > 0.04 { "open" } else { "close" }),
                trajectory: None,
                gripper_command: Some(*width),
                scene_diff: None,
            }]),
        }
    }
}

// (Planning implementations + helpers moved to planners.rs)

/// Apply scene diffs from a task solution to a mutable scene.
pub fn apply_scene_diffs(scene: &mut Scene, solution: &TaskSolution) {
    for stage in &solution.stages {
        if let Some(diff) = &stage.scene_diff {
            match diff {
                SceneDiff::Attach {
                    object,
                    link,
                    shape,
                    grasp_transform,
                } => {
                    scene.attach(object, shape.clone(), *grasp_transform, link);
                }
                SceneDiff::Detach { object, place_pose } => {
                    scene.detach(object, *place_pose);
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::planners::{detect_chain, extract_joint_vel_accel};
    use kinetic_core::JointValues;
    use kinetic_kinematics::forward_kinematics;
    use kinetic_robot::Robot;

    fn ur5e() -> Arc<Robot> {
        Arc::new(Robot::from_name("ur5e").unwrap())
    }

    fn home_joints() -> Vec<f64> {
        vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn test_approach_offset() {
        let target = Isometry3::translation(1.0, 0.0, 0.5);
        let approach = Approach::linear(-Vector3::z(), 0.10);
        let pre = approach.offset_pose(&target);
        // Pre-approach should be 0.10 above the target (since direction is -Z, offset removes it)
        let expected_z = 0.5 + 0.10; // target.z - (direction * distance) = 0.5 - (-0.1) = 0.6
        assert!(
            (pre.translation.vector.z - expected_z).abs() < 1e-10,
            "Pre-approach z: {} expected {}",
            pre.translation.vector.z,
            expected_z
        );
    }

    #[test]
    fn test_gripper_task() {
        let task = Task::gripper(0.08);
        let solution = task.plan(&home_joints()).unwrap();
        assert_eq!(solution.stages.len(), 1);
        assert_eq!(solution.stages[0].gripper_command, Some(0.08));
        assert!(solution.stages[0].trajectory.is_none());
    }

    #[test]
    fn test_sequence_of_grippers() {
        let task = Task::sequence(vec![Task::gripper(0.08), Task::gripper(0.02)]);
        let solution = task.plan(&home_joints()).unwrap();
        assert_eq!(solution.stages.len(), 2);
        assert_eq!(solution.stages[0].gripper_command, Some(0.08));
        assert_eq!(solution.stages[1].gripper_command, Some(0.02));
    }

    #[test]
    fn test_move_to_joints() {
        let robot = ur5e();
        let start = home_joints();
        let goal_joints = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

        let task = Task::move_to(&robot, Goal::Joints(kinetic_core::JointValues(goal_joints)));
        let solution = task.plan(&start).unwrap();

        assert_eq!(solution.stages.len(), 1);
        assert!(solution.stages[0].trajectory.is_some());
        let traj = solution.stages[0].trajectory.as_ref().unwrap();
        assert!(traj.waypoints.len() >= 2);
    }

    #[test]
    fn test_sequence_move_to() {
        let robot = ur5e();
        let start = home_joints();

        let task = Task::sequence(vec![
            Task::move_to(
                &robot,
                Goal::Joints(kinetic_core::JointValues(vec![
                    0.5, -1.0, 0.5, 0.0, 0.5, 0.0,
                ])),
            ),
            Task::gripper(0.08),
            Task::move_to(
                &robot,
                Goal::Joints(kinetic_core::JointValues(vec![
                    0.0, -1.0, 0.8, 0.0, 0.0, 0.0,
                ])),
            ),
        ]);

        let solution = task.plan(&start).unwrap();
        assert_eq!(solution.stages.len(), 3);
        assert!(solution.stages[0].trajectory.is_some());
        assert!(solution.stages[1].trajectory.is_none()); // gripper
        assert!(solution.stages[2].trajectory.is_some());
    }

    #[test]
    fn test_task_solution_final_joints() {
        let robot = ur5e();
        let start = home_joints();
        let target = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

        let task = Task::move_to(
            &robot,
            Goal::Joints(kinetic_core::JointValues(target.clone())),
        );
        let solution = task.plan(&start).unwrap();

        let final_j = solution.final_joints().unwrap();
        // Should be close to target
        for (a, b) in final_j.iter().zip(target.iter()) {
            assert!((a - b).abs() < 0.1, "Final joint mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_scene_diff_attach() {
        let diff = SceneDiff::Attach {
            object: "cup".into(),
            link: "tool0".into(),
            shape: Shape::Cylinder(0.03, 0.06),
            grasp_transform: Isometry3::identity(),
        };
        // Just verify it constructs correctly
        match &diff {
            SceneDiff::Attach { object, link, .. } => {
                assert_eq!(object, "cup");
                assert_eq!(link, "tool0");
            }
            _ => panic!("Expected Attach"),
        }
    }

    #[test]
    fn test_scene_diff_detach() {
        let diff = SceneDiff::Detach {
            object: "cup".into(),
            place_pose: Isometry3::translation(0.5, 0.0, 0.3),
        };
        match &diff {
            SceneDiff::Detach {
                object, place_pose, ..
            } => {
                assert_eq!(object, "cup");
                assert!((place_pose.translation.vector.x - 0.5).abs() < 1e-10);
            }
            _ => panic!("Expected Detach"),
        }
    }

    #[test]
    fn test_pick_config_construction() {
        let config = PickConfig {
            object: "box1".into(),
            grasp_poses: vec![Isometry3::identity()],
            approach: Approach::linear(-Vector3::z(), 0.10),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.03,
        };
        assert_eq!(config.object, "box1");
        assert_eq!(config.grasp_poses.len(), 1);
        assert!((config.approach.distance - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_place_config_construction() {
        let config = PlaceConfig {
            object: "box1".into(),
            target_pose: Isometry3::translation(0.5, 0.0, 0.3),
            approach: Approach::linear(-Vector3::z(), 0.10),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
        };
        assert_eq!(config.object, "box1");
        assert!((config.target_pose.translation.vector.x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_scene_diffs() {
        let robot = ur5e();
        let mut scene = Scene::new(&robot).unwrap();

        // Add an object
        scene.add(
            "cup",
            Shape::Cylinder(0.03, 0.06),
            Isometry3::translation(0.5, 0.0, 0.3),
        );
        assert_eq!(scene.num_objects(), 1);
        assert_eq!(scene.num_attached(), 0);

        // Create a fake solution with attach diff
        let solution = TaskSolution {
            stages: vec![StageSolution {
                name: "pick_attach".into(),
                trajectory: None,
                gripper_command: None,
                scene_diff: Some(SceneDiff::Attach {
                    object: "cup".into(),
                    link: robot.links.last().unwrap().name.clone(),
                    shape: Shape::Cylinder(0.03, 0.06),
                    grasp_transform: Isometry3::identity(),
                }),
            }],
            total_duration: Duration::ZERO,
            total_planning_time: Duration::ZERO,
        };

        apply_scene_diffs(&mut scene, &solution);
        assert_eq!(
            scene.num_objects(),
            0,
            "Object should be removed from scene"
        );
        assert_eq!(scene.num_attached(), 1, "Object should be attached");
    }

    #[test]
    fn test_empty_sequence() {
        let task = Task::sequence(vec![]);
        let solution = task.plan(&home_joints()).unwrap();
        assert!(
            solution.stages.is_empty(),
            "Empty sequence should produce no stages"
        );
        assert_eq!(solution.total_duration, Duration::ZERO);
        assert!(solution.final_joints().is_none());
    }

    #[test]
    fn test_nested_sequence() {
        // Sequences inside sequences should flatten: all stages appear in order
        let inner = Task::sequence(vec![Task::gripper(0.08), Task::gripper(0.04)]);
        let outer = Task::sequence(vec![Task::gripper(0.10), inner, Task::gripper(0.02)]);

        let solution = outer.plan(&home_joints()).unwrap();
        assert_eq!(
            solution.stages.len(),
            4,
            "Nested sequence should produce 4 total stages"
        );
        assert_eq!(solution.stages[0].gripper_command, Some(0.10));
        assert_eq!(solution.stages[1].gripper_command, Some(0.08));
        assert_eq!(solution.stages[2].gripper_command, Some(0.04));
        assert_eq!(solution.stages[3].gripper_command, Some(0.02));
    }

    #[test]
    fn test_apply_attach_then_detach_same_object() {
        // Attach and then immediately detach the same object via scene diffs
        let robot = ur5e();
        let mut scene = Scene::new(&robot).unwrap();

        scene.add(
            "block",
            Shape::Cuboid(0.05, 0.05, 0.05),
            Isometry3::translation(0.5, 0.0, 0.3),
        );
        assert_eq!(scene.num_objects(), 1);
        assert_eq!(scene.num_attached(), 0);

        let solution = TaskSolution {
            stages: vec![
                StageSolution {
                    name: "attach".into(),
                    trajectory: None,
                    gripper_command: None,
                    scene_diff: Some(SceneDiff::Attach {
                        object: "block".into(),
                        link: robot.links.last().unwrap().name.clone(),
                        shape: Shape::Cuboid(0.05, 0.05, 0.05),
                        grasp_transform: Isometry3::identity(),
                    }),
                },
                StageSolution {
                    name: "detach".into(),
                    trajectory: None,
                    gripper_command: None,
                    scene_diff: Some(SceneDiff::Detach {
                        object: "block".into(),
                        place_pose: Isometry3::translation(0.8, 0.0, 0.3),
                    }),
                },
            ],
            total_duration: Duration::ZERO,
            total_planning_time: Duration::ZERO,
        };

        apply_scene_diffs(&mut scene, &solution);

        // After attach+detach cycle, object should be back in the world
        assert_eq!(scene.num_attached(), 0, "Object should be detached");
        assert_eq!(
            scene.num_objects(),
            1,
            "Object should be back in the scene at new pose"
        );
    }

    /// Gap 6: Task::Sequence with [MoveTo, Gripper(close), MoveTo, Gripper(open)]
    /// should produce a solution with 4 stages and correct names.
    #[test]
    fn test_sequence_move_gripper_move_gripper() {
        let robot = ur5e();
        let start = home_joints();

        let task = Task::sequence(vec![
            Task::move_to(
                &robot,
                Goal::Joints(kinetic_core::JointValues(vec![
                    0.5, -1.0, 0.5, 0.0, 0.5, 0.0,
                ])),
            ),
            Task::gripper(0.02), // close
            Task::move_to(
                &robot,
                Goal::Joints(kinetic_core::JointValues(vec![
                    0.0, -1.0, 0.8, 0.0, 0.0, 0.0,
                ])),
            ),
            Task::gripper(0.08), // open
        ]);

        let solution = task.plan(&start).unwrap();

        // Should have exactly 4 stages
        assert_eq!(
            solution.stages.len(),
            4,
            "Expected 4 stages, got {}",
            solution.stages.len()
        );

        // Stage 0: move_to (has trajectory)
        assert_eq!(solution.stages[0].name, "move_to");
        assert!(
            solution.stages[0].trajectory.is_some(),
            "First move_to should have trajectory"
        );

        // Stage 1: gripper close (no trajectory, gripper command)
        assert!(
            solution.stages[1].name.contains("gripper"),
            "Stage 1 should be gripper, got: {}",
            solution.stages[1].name
        );
        assert_eq!(solution.stages[1].gripper_command, Some(0.02));
        assert!(solution.stages[1].trajectory.is_none());

        // Stage 2: move_to (has trajectory)
        assert_eq!(solution.stages[2].name, "move_to");
        assert!(
            solution.stages[2].trajectory.is_some(),
            "Second move_to should have trajectory"
        );

        // Stage 3: gripper open (no trajectory, gripper command)
        assert!(
            solution.stages[3].name.contains("gripper"),
            "Stage 3 should be gripper, got: {}",
            solution.stages[3].name
        );
        assert_eq!(solution.stages[3].gripper_command, Some(0.08));
        assert!(solution.stages[3].trajectory.is_none());

        // Total duration should be positive (two motion stages)
        assert!(
            solution.total_duration.as_secs_f64() > 0.0,
            "Total duration should be positive"
        );
    }

    #[test]
    fn test_detect_chain_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = detect_chain(&robot).unwrap();
        assert_eq!(chain.dof, 6);
    }

    #[test]
    fn test_cartesian_move_task() {
        let robot = ur5e();
        // Use a non-singular start configuration
        let start = vec![0.0, -1.2, 1.0, -0.8, -std::f64::consts::FRAC_PI_2, 0.0];

        // Get current EE pose and move slightly
        let chain = detect_chain(&robot).unwrap();
        let current_pose = forward_kinematics(&robot, &chain, &start).unwrap();
        let offset = Vector3::new(0.0, 0.0, -0.01); // very small downward move
        let target = Isometry3::from_parts(
            Translation3::from(current_pose.translation() + offset),
            *current_pose.rotation(),
        );

        let task = Task::cartesian_move(&robot, target, CartesianConfig::default());
        let solution = task.plan(&start).unwrap();

        assert_eq!(solution.stages.len(), 1);
        assert!(solution.stages[0].trajectory.is_some());
    }

    // ─── Coverage: extract_joint_vel_accel ───────────────────────────

    #[test]
    fn test_extract_vel_accel_positive() {
        let robot = ur5e();
        let (vel, accel) = extract_joint_vel_accel(&robot);
        assert_eq!(vel.len(), robot.dof);
        assert_eq!(accel.len(), robot.dof);
        for &v in &vel {
            assert!(v > 0.0, "velocity limit should be positive: {v}");
        }
        for &a in &accel {
            assert!(a > 0.0, "acceleration limit should be positive: {a}");
        }
    }

    // ─── Coverage: Pick task ─────────────────────────────────────────

    #[test]
    fn test_pick_with_no_grasps_returns_error() {
        let robot = ur5e();
        let scene = Arc::new(Scene::new(&robot).unwrap());
        let config = PickConfig {
            object: "nonexistent_object".into(),
            grasp_poses: vec![],
            approach: Approach::linear(-Vector3::z(), 0.05),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.02,
        };
        let task = Task::pick(&robot, &scene, config);
        let result = task.plan(&home_joints());
        assert!(result.is_err(), "pick with no grasps should fail");
    }

    #[test]
    fn test_pick_with_missing_object_returns_error() {
        let robot = ur5e();
        let scene = Arc::new(Scene::new(&robot).unwrap());
        let config = PickConfig {
            object: "missing_box".into(),
            grasp_poses: vec![Isometry3::translation(0.4, 0.0, 0.3)],
            approach: Approach::linear(-Vector3::z(), 0.05),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.02,
        };
        let task = Task::pick(&robot, &scene, config);
        let result = task.plan(&home_joints());
        assert!(result.is_err(), "pick of missing object should fail");
    }

    // ─── Coverage: Place task ────────────────────────────────────────

    #[test]
    fn test_place_config_fields() {
        let config = PlaceConfig {
            object: "box".into(),
            target_pose: Isometry3::translation(0.5, 0.0, 0.3),
            approach: Approach::linear(-Vector3::z(), 0.05),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
        };
        assert_eq!(config.object, "box");
    }

    // ─── Coverage: apply_scene_diffs ─────────────────────────────────

    fn make_solution_with_diff(diff: SceneDiff) -> TaskSolution {
        TaskSolution {
            stages: vec![StageSolution {
                name: "test".into(),
                trajectory: None,
                gripper_command: None,
                scene_diff: Some(diff),
            }],
            total_duration: Duration::ZERO,
            total_planning_time: Duration::ZERO,
        }
    }

    #[test]
    fn test_apply_attach_diff() {
        let robot = ur5e();
        let mut scene = Scene::new(&robot).unwrap();
        scene.add("bolt", Shape::Cylinder(0.005, 0.03), Isometry3::identity());

        let solution = make_solution_with_diff(SceneDiff::Attach {
            object: "bolt".into(),
            link: "ee_link".into(),
            shape: Shape::Cylinder(0.005, 0.03),
            grasp_transform: Isometry3::identity(),
        });
        apply_scene_diffs(&mut scene, &solution);
        assert_eq!(scene.num_attached(), 1);
    }

    #[test]
    fn test_apply_detach_diff() {
        let robot = ur5e();
        let mut scene = Scene::new(&robot).unwrap();
        scene.attach("bolt", Shape::Cylinder(0.005, 0.03), Isometry3::identity(), "ee_link");

        let solution = make_solution_with_diff(SceneDiff::Detach {
            object: "bolt".into(),
            place_pose: Isometry3::translation(0.5, 0.0, 0.3),
        });
        apply_scene_diffs(&mut scene, &solution);
        assert_eq!(scene.num_attached(), 0);
        assert_eq!(scene.num_objects(), 1);
    }

    // ─── Coverage: TaskSolution methods ──────────────────────────────

    #[test]
    fn test_task_solution_duration_positive() {
        let robot = ur5e();
        let start = home_joints();
        let goal = Goal::Joints(JointValues(vec![0.3, -1.0, 0.5, -0.5, -1.0, 0.3]));
        let task = Task::move_to(&robot, goal);
        let solution = task.plan(&start).unwrap();
        assert!(solution.total_duration.as_secs_f64() > 0.0, "duration should be > 0");
        assert!(solution.total_planning_time.as_secs_f64() > 0.0, "planning time should be > 0");
    }

    // ─── Coverage: Sequence with mixed types ─────────────────────────

    #[test]
    fn test_sequence_four_stages_move_grip_move_grip() {
        let robot = ur5e();
        let start = home_joints();

        let task = Task::sequence(vec![
            Task::move_to(&robot, Goal::Joints(JointValues(vec![0.3, -1.0, 0.5, -0.5, -1.0, 0.3]))),
            Task::gripper(0.08), // open
            Task::move_to(&robot, Goal::Joints(JointValues(vec![0.1, -0.8, 0.3, -0.3, -0.8, 0.1]))),
            Task::gripper(0.02), // close
        ]);

        let solution = task.plan(&start).unwrap();
        assert_eq!(solution.stages.len(), 4);
        assert!(solution.stages[0].trajectory.is_some()); // move
        assert!(solution.stages[1].gripper_command.is_some()); // gripper
        assert!(solution.stages[2].trajectory.is_some()); // move
        assert!(solution.stages[3].gripper_command.is_some()); // gripper
    }

    // ─── Coverage: detect_chain error ────────────────────────────────

    #[test]
    fn test_task_error_display() {
        let e1 = TaskError::NoValidGrasp("box".into());
        let e2 = TaskError::ObjectNotFound("missing".into());
        let e3 = TaskError::PlanningFailed { stage: "move".into(), reason: "timeout".into() };
        let e4 = TaskError::Trajectory("bad timing".into());
        assert!(format!("{e1}").contains("box"));
        assert!(format!("{e2}").contains("missing"));
        assert!(format!("{e3}").contains("move"));
        assert!(format!("{e4}").contains("bad timing"));
    }
}
