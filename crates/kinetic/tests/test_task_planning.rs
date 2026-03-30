//! Integration test: task planning (pick, place, sequence).
//!
//! Tests the full task planning pipeline including move-to, Cartesian moves,
//! task sequencing, and pick/place decomposition.

use kinetic::prelude::*;
use kinetic::task::{SceneDiff, Task, TaskSolution};
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

#[test]
fn task_move_to_joints() {
    let robot = ur5e();
    let start = home_joints();
    let goal_joints = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

    let task = Task::move_to(&robot, Goal::Joints(JointValues(goal_joints.clone())));
    let solution = task.plan(&start).unwrap();

    assert_eq!(solution.stages.len(), 1);
    assert!(solution.stages[0].trajectory.is_some());

    let final_j = solution.final_joints().unwrap();
    for (a, b) in final_j.iter().zip(goal_joints.iter()) {
        assert!((a - b).abs() < 0.1, "Goal mismatch: {} vs {}", a, b);
    }
    assert!(solution.total_duration.as_secs_f64() > 0.0);
    assert!(solution.total_planning_time.as_secs_f64() > 0.0);
}

#[test]
fn task_sequence() {
    let robot = ur5e();
    let start = home_joints();

    let task = Task::sequence(vec![
        Task::move_to(
            &robot,
            Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        ),
        Task::gripper(0.08),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues(vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0])),
        ),
    ]);

    let solution = task.plan(&start).unwrap();
    assert_eq!(solution.stages.len(), 3);

    // First and third stages should be trajectories
    assert!(solution.stages[0].trajectory.is_some());
    assert!(solution.stages[2].trajectory.is_some());

    // Second stage is gripper
    assert!(solution.stages[1].trajectory.is_none());
    assert_eq!(solution.stages[1].gripper_command, Some(0.08));

    // Total duration should be sum of trajectory durations
    assert!(solution.total_duration.as_secs_f64() > 0.0);
}

#[test]
fn task_gripper_only() {
    let task = Task::gripper(0.04);
    let solution = task.plan(&home_joints()).unwrap();

    assert_eq!(solution.stages.len(), 1);
    assert_eq!(solution.stages[0].gripper_command, Some(0.04));
    assert!(solution.stages[0].trajectory.is_none());
}

#[test]
fn task_sequence_propagates_joints() {
    let robot = ur5e();
    let start = home_joints();

    // Sequence of three moves — each should start from where the previous ended
    let configs = [
        vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
        vec![-0.3, -1.2, 0.8, 0.0, -0.3, 0.0],
        vec![0.0, -0.8, 0.3, -0.5, 0.0, 0.2],
    ];

    let task = Task::sequence(
        configs
            .iter()
            .map(|c| Task::move_to(&robot, Goal::Joints(JointValues(c.clone()))))
            .collect(),
    );

    let solution = task.plan(&start).unwrap();
    assert_eq!(solution.stages.len(), 3);

    // Final joints should be close to the last target
    let final_j = solution.final_joints().unwrap();
    for (a, b) in final_j.iter().zip(configs[2].iter()) {
        assert!((a - b).abs() < 0.15, "Final joint mismatch: {} vs {}", a, b);
    }
}

#[test]
fn task_cartesian_move() {
    let robot = ur5e();
    // Non-singular start
    let start = vec![0.0, -1.2, 1.0, -0.8, -std::f64::consts::FRAC_PI_2, 0.0];

    let chain = kinetic::kinematics::KinematicChain::extract(
        &robot,
        &robot.links[0].name,
        &robot.links.last().unwrap().name,
    )
    .unwrap();

    let current_pose = kinetic::kinematics::forward_kinematics(&robot, &chain, &start).unwrap();
    let offset = Vector3::new(0.0, 0.0, -0.01);
    let target = Isometry3::from_parts(
        nalgebra::Translation3::from(current_pose.translation() + offset),
        *current_pose.rotation(),
    );

    let task = Task::cartesian_move(
        &robot,
        target,
        kinetic::planning::CartesianConfig::default(),
    );
    let solution = task.plan(&start).unwrap();

    assert_eq!(solution.stages.len(), 1);
    assert!(solution.stages[0].trajectory.is_some());
}

#[test]
fn task_scene_diff_application() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "cup",
        Shape::Cylinder(0.03, 0.06),
        Isometry3::translation(0.5, 0.0, 0.3),
    );
    assert_eq!(scene.num_objects(), 1);

    // Create a solution with attach diff
    let ee_link = robot.links.last().unwrap().name.clone();
    let solution = TaskSolution {
        stages: vec![kinetic::task::StageSolution {
            name: "pick_attach".into(),
            trajectory: None,
            gripper_command: None,
            scene_diff: Some(SceneDiff::Attach {
                object: "cup".into(),
                link: ee_link,
                shape: Shape::Cylinder(0.03, 0.06),
                grasp_transform: Isometry3::identity(),
            }),
        }],
        total_duration: std::time::Duration::ZERO,
        total_planning_time: std::time::Duration::ZERO,
    };

    kinetic::task::apply_scene_diffs(&mut scene, &solution);
    assert_eq!(scene.num_objects(), 0);
    assert_eq!(scene.num_attached(), 1);
}
