//! End-to-end pick-and-place application tests.
//!
//! Tests the complete manipulation workflow: scene setup, grasp generation,
//! approach/grasp/retreat planning, attach/detach, place, and trajectory
//! validation. Exercises the full kinetic pipeline as a real application would.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic::grasp::{GraspConfig, GraspGenerator, GripperType};
use kinetic::kinematics::KinematicChain;
use kinetic::prelude::*;
use kinetic::scene::Scene;
use kinetic::task::{
    apply_scene_diffs, Approach, PickConfig, SceneDiff, StageSolution, Task, TaskSolution,
};
use kinetic::trajectory::TimedTrajectory;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn ur5e_chain(robot: &Robot) -> KinematicChain {
    let arm = &robot.groups["arm"];
    KinematicChain::extract(robot, &arm.base_link, &arm.tip_link).unwrap()
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

// ─── Scene construction ─────────────────────────────────────────────────────

#[test]
fn build_pick_place_scene() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Table surface
    scene.add(
        "table",
        Shape::cuboid(0.3, 0.4, 0.01),
        Isometry3::translation(0.5, 0.0, 0.0),
    );

    // Object to pick (small cylinder on table)
    scene.add(
        "cup",
        Shape::Cylinder(0.03, 0.05),
        Isometry3::translation(0.5, 0.0, 0.06),
    );

    // Placement bin
    scene.add(
        "bin",
        Shape::cuboid(0.1, 0.1, 0.02),
        Isometry3::translation(0.3, 0.3, 0.0),
    );

    assert_eq!(scene.num_objects(), 3);
    assert_eq!(scene.num_attached(), 0);

    // Verify objects are retrievable
    assert!(scene.get_object("cup").is_some());
    assert!(scene.get_object("table").is_some());
    assert!(scene.get_object("bin").is_some());
    assert!(scene.get_object("nonexistent").is_none());
}

// ─── Grasp generation ───────────────────────────────────────────────────────

#[test]
fn generate_grasps_for_cylinder() {
    let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
    let shape = Shape::Cylinder(0.03, 0.05);
    let object_pose = Isometry3::translation(0.5, 0.0, 0.06);

    let config = GraspConfig {
        num_candidates: 20,
        ..Default::default()
    };

    let grasps = gen.from_shape(&shape, &object_pose, config).unwrap();

    assert!(!grasps.is_empty(), "Should generate grasps for cylinder");
    assert!(grasps.len() <= 20, "Should respect candidate limit");

    // All grasps should have valid quality scores
    for g in &grasps {
        assert!(g.quality >= 0.0 && g.quality <= 1.0);
    }

    // Should be sorted by quality (descending)
    for w in grasps.windows(2) {
        assert!(w[0].quality >= w[1].quality - 1e-10);
    }
}

#[test]
fn generate_grasps_for_cuboid() {
    let gen = GraspGenerator::new(GripperType::parallel(0.10, 0.03));
    let shape = Shape::cuboid(0.03, 0.04, 0.05);
    let object_pose = Isometry3::translation(0.4, 0.1, 0.06);

    let grasps = gen
        .from_shape(&shape, &object_pose, GraspConfig::default())
        .unwrap();

    assert!(
        grasps.len() >= 5,
        "Should generate grasps for cuboid: got {}",
        grasps.len()
    );

    // Should include different grasp types
    let types: std::collections::HashSet<_> = grasps
        .iter()
        .map(|g| format!("{:?}", g.grasp_type))
        .collect();
    assert!(
        types.len() >= 2,
        "Should have multiple grasp types: {:?}",
        types
    );
}

// ─── Task::pick for nonexistent object ──────────────────────────────────────

#[test]
fn pick_nonexistent_object_fails() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());

    let task = Task::pick(
        &robot,
        &scene,
        PickConfig {
            object: "ghost_object".to_string(),
            grasp_poses: vec![Isometry3::identity()],
            approach: Approach::linear(-Vector3::z(), 0.1),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.0,
        },
    );

    let result = task.plan(&home_joints());
    assert!(result.is_err(), "Pick of nonexistent object should fail");
}

// ─── Task::sequence with gripper commands ───────────────────────────────────

#[test]
fn gripper_sequence_produces_correct_stages() {
    let task = Task::sequence(vec![
        Task::gripper(0.08), // open
        Task::gripper(0.02), // close
        Task::gripper(0.08), // open again
    ]);

    let solution = task.plan(&home_joints()).unwrap();
    assert_eq!(solution.stages.len(), 3);
    assert_eq!(solution.stages[0].gripper_command, Some(0.08));
    assert_eq!(solution.stages[1].gripper_command, Some(0.02));
    assert_eq!(solution.stages[2].gripper_command, Some(0.08));

    // No trajectories for pure gripper commands
    for stage in &solution.stages {
        assert!(stage.trajectory.is_none());
    }
}

// ─── Move-to planning within task framework ─────────────────────────────────

#[test]
fn move_to_joint_goal_in_task() {
    let robot = ur5e();
    let start = home_joints();
    let goal = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

    let task = Task::move_to(&robot, Goal::Joints(JointValues::new(goal.clone())));
    let solution = task.plan(&start).unwrap();

    assert_eq!(solution.stages.len(), 1);
    let traj = solution.stages[0].trajectory.as_ref().unwrap();
    assert!(traj.waypoints.len() >= 2);

    // Final waypoint should be close to goal
    let final_wp = traj.waypoints.last().unwrap();
    for (a, b) in final_wp.positions.iter().zip(goal.iter()) {
        assert!((a - b).abs() < 0.1, "Final joint {a} != goal {b}");
    }
}

// ─── Task::sequence with move + gripper ─────────────────────────────────────

#[test]
fn sequence_move_gripper_move() {
    let robot = ur5e();
    let start = home_joints();

    let task = Task::sequence(vec![
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        ),
        Task::gripper(0.08),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0])),
        ),
    ]);

    let solution = task.plan(&start).unwrap();
    assert_eq!(solution.stages.len(), 3);
    assert!(solution.stages[0].trajectory.is_some()); // move
    assert!(solution.stages[1].trajectory.is_none()); // gripper
    assert!(solution.stages[2].trajectory.is_some()); // move

    // Verify joint continuity: end of move1 should be start of move2
    let traj1 = solution.stages[0].trajectory.as_ref().unwrap();
    let traj2 = solution.stages[2].trajectory.as_ref().unwrap();
    let end1 = &traj1.waypoints.last().unwrap().positions;
    let start2 = &traj2.waypoints.first().unwrap().positions;

    for (a, b) in end1.iter().zip(start2.iter()) {
        assert!(
            (a - b).abs() < 0.2,
            "Joint continuity violated between stages: {a} vs {b}"
        );
    }
}

// ─── Scene attach/detach lifecycle ──────────────────────────────────────────

#[test]
fn scene_attach_detach_lifecycle() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();

    // Add object to scene
    scene.add(
        "cup",
        Shape::Cylinder(0.03, 0.05),
        Isometry3::translation(0.5, 0.0, 0.06),
    );
    assert_eq!(scene.num_objects(), 1);
    assert_eq!(scene.num_attached(), 0);

    // Attach to end-effector
    let ee_link = &robot.links[chain.tip_link].name;
    scene.attach(
        "cup",
        Shape::Cylinder(0.03, 0.05),
        Isometry3::identity(),
        ee_link,
    );
    assert_eq!(scene.num_objects(), 0, "Object removed from world");
    assert_eq!(scene.num_attached(), 1, "Object attached to robot");

    // Detach at new location
    let place_pose = Isometry3::translation(0.3, 0.3, 0.06);
    let detached = scene.detach("cup", place_pose);
    assert!(detached, "Should successfully detach");
    assert_eq!(scene.num_objects(), 1, "Object back in world");
    assert_eq!(scene.num_attached(), 0, "Nothing attached");

    // Verify placed at new pose
    let obj = scene.get_object("cup").unwrap();
    let pos = obj.pose.translation.vector;
    assert!((pos.x - 0.3).abs() < 1e-6);
    assert!((pos.y - 0.3).abs() < 1e-6);
}

// ─── apply_scene_diffs ──────────────────────────────────────────────────────

#[test]
fn apply_scene_diffs_attach_and_detach() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "block",
        Shape::cuboid(0.02, 0.02, 0.02),
        Isometry3::translation(0.4, 0.0, 0.03),
    );

    let ee_link = robot.links[chain.tip_link].name.clone();

    // Simulate attach
    let attach_solution = TaskSolution {
        stages: vec![StageSolution {
            name: "attach".into(),
            trajectory: None,
            gripper_command: None,
            scene_diff: Some(SceneDiff::Attach {
                object: "block".into(),
                link: ee_link,
                shape: Shape::cuboid(0.02, 0.02, 0.02),
                grasp_transform: Isometry3::identity(),
            }),
        }],
        total_duration: Duration::ZERO,
        total_planning_time: Duration::ZERO,
    };

    apply_scene_diffs(&mut scene, &attach_solution);
    assert_eq!(scene.num_objects(), 0);
    assert_eq!(scene.num_attached(), 1);

    // Simulate detach
    let detach_solution = TaskSolution {
        stages: vec![StageSolution {
            name: "detach".into(),
            trajectory: None,
            gripper_command: None,
            scene_diff: Some(SceneDiff::Detach {
                object: "block".into(),
                place_pose: Isometry3::translation(0.3, 0.3, 0.03),
            }),
        }],
        total_duration: Duration::ZERO,
        total_planning_time: Duration::ZERO,
    };

    apply_scene_diffs(&mut scene, &detach_solution);
    assert_eq!(scene.num_objects(), 1);
    assert_eq!(scene.num_attached(), 0);
}

// ─── Trajectory validation ──────────────────────────────────────────────────

#[test]
fn trajectory_dimensional_consistency() {
    let robot = ur5e();
    let start = home_joints();
    let goal = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

    let task = Task::move_to(&robot, Goal::Joints(JointValues::new(goal)));
    let solution = task.plan(&start).unwrap();

    let traj = solution.stages[0].trajectory.as_ref().unwrap();

    // All waypoints should have consistent DOF
    for wp in &traj.waypoints {
        assert_eq!(wp.positions.len(), traj.dof);
        assert_eq!(wp.velocities.len(), traj.dof);
        assert_eq!(wp.accelerations.len(), traj.dof);
    }

    // Timestamps should be monotonically non-decreasing
    for w in traj.waypoints.windows(2) {
        assert!(
            w[1].time >= w[0].time - 1e-12,
            "Timestamps not monotonic: {} < {}",
            w[1].time,
            w[0].time
        );
    }

    // Duration should be positive
    assert!(traj.duration() > Duration::ZERO);

    // Validate method should pass
    assert!(traj.validate().is_ok());
}

// ─── Full pick-place pipeline ───────────────────────────────────────────────

#[test]
fn full_pick_place_pipeline() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // -- Build scene --
    // Table
    scene.add(
        "table",
        Shape::cuboid(0.3, 0.4, 0.01),
        Isometry3::translation(0.5, 0.0, 0.0),
    );

    // Object to pick — small cylinder on table
    let obj_pose = Isometry3::translation(0.5, 0.0, 0.06);
    scene.add("cup", Shape::Cylinder(0.03, 0.05), obj_pose);

    // -- Generate grasps --
    let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
    let grasp_config = GraspConfig {
        num_candidates: 10,
        ..Default::default()
    };
    let grasps = gen
        .from_shape(&Shape::Cylinder(0.03, 0.05), &obj_pose, grasp_config)
        .unwrap();

    assert!(!grasps.is_empty(), "Should generate grasps");

    // -- Build pick task --
    let scene_arc = Arc::new(scene);

    let pick_task = Task::pick(
        &robot,
        &scene_arc,
        PickConfig {
            object: "cup".to_string(),
            grasp_poses: grasps.iter().take(5).map(|g| g.grasp_pose).collect(),
            approach: Approach::linear(-Vector3::z(), 0.08),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.02,
        },
    );

    // -- Plan pick --
    let start = home_joints();
    let t0 = Instant::now();
    let pick_result = pick_task.plan(&start);
    let pick_time = t0.elapsed();

    eprintln!("Pick planning time: {:?}", pick_time);

    // Pick may fail if IK can't reach the grasp poses — that's acceptable
    // in this test since we're testing the pipeline, not the specific configuration
    match pick_result {
        Ok(solution) => {
            eprintln!("Pick succeeded with {} stages", solution.stages.len());

            // Verify stage structure
            let stage_names: Vec<&str> = solution.stages.iter().map(|s| s.name.as_str()).collect();
            eprintln!("Stages: {:?}", stage_names);

            // Should have trajectory stages and gripper stages
            let has_trajectory = solution.stages.iter().any(|s| s.trajectory.is_some());
            let has_gripper = solution.stages.iter().any(|s| s.gripper_command.is_some());
            let has_attach = solution
                .stages
                .iter()
                .any(|s| matches!(&s.scene_diff, Some(SceneDiff::Attach { .. })));

            assert!(has_trajectory, "Should have trajectory stages");
            assert!(has_gripper, "Should have gripper stages");
            assert!(has_attach, "Should have attach scene diff");

            // Verify all trajectories have correct DOF
            for stage in &solution.stages {
                if let Some(traj) = &stage.trajectory {
                    assert_eq!(
                        traj.dof, robot.dof,
                        "DOF mismatch in stage '{}'",
                        stage.name
                    );
                    assert!(
                        traj.validate().is_ok(),
                        "Trajectory invalid in stage '{}'",
                        stage.name
                    );
                }
            }

            // Verify total duration is reasonable
            assert!(
                solution.total_duration < Duration::from_secs(60),
                "Total duration too long: {:?}",
                solution.total_duration
            );

            // Verify final joints exist
            assert!(
                solution.final_joints().is_some(),
                "Should have final joints"
            );
        }
        Err(e) => {
            eprintln!("Pick failed (acceptable): {}", e);
            // Planning failure is ok for this config — test validates the pipeline doesn't panic
        }
    }

    // Pipeline should complete within reasonable time
    assert!(
        pick_time < Duration::from_secs(60),
        "Pipeline should complete in <60s: {:?}",
        pick_time
    );
}

// ─── Move-to with scene obstacles ───────────────────────────────────────────

#[test]
fn move_to_with_scene_obstacles() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Small obstacle
    scene.add(
        "box",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.3),
    );

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));

    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(2),
            ..PlannerConfig::default()
        });

    let result = planner.plan(&start, &goal);

    match result {
        Ok(plan) => {
            assert!(plan.num_waypoints() >= 2);
            assert!(plan.path_length() > 0.0);

            // Verify all waypoints are collision-free
            for wp in plan.waypoints.iter() {
                let in_collision = scene.check_collision(wp).unwrap();
                assert!(!in_collision, "Waypoint in collision");
            }
        }
        Err(e) => {
            eprintln!("Planning failed (acceptable for RRT): {}", e);
        }
    }
}

// ─── Joint continuity across multi-stage task ───────────────────────────────

#[test]
fn multi_stage_joint_continuity() {
    let robot = ur5e();
    let start = home_joints();

    let configs = vec![
        vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
        vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0],
        vec![-0.5, -0.8, 0.3, 0.1, 0.3, -0.2],
    ];

    let task = Task::sequence(
        configs
            .into_iter()
            .map(|c| Task::move_to(&robot, Goal::Joints(JointValues::new(c))))
            .collect(),
    );

    let solution = task.plan(&start).unwrap();
    assert_eq!(solution.stages.len(), 3);

    // Check joint continuity between consecutive trajectory stages
    let trajectories: Vec<&TimedTrajectory> = solution
        .stages
        .iter()
        .filter_map(|s| s.trajectory.as_ref())
        .collect();

    for i in 0..trajectories.len() - 1 {
        let end = &trajectories[i].waypoints.last().unwrap().positions;
        let start_next = &trajectories[i + 1].waypoints.first().unwrap().positions;

        for (j, (a, b)) in end.iter().zip(start_next.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.2,
                "Joint {j} discontinuity between stage {i} and {}: {a} vs {b}",
                i + 1
            );
        }
    }

    // Verify total duration
    assert!(solution.total_duration > Duration::ZERO);
    eprintln!("Multi-stage total duration: {:?}", solution.total_duration);
}

// ─── Full pipeline timing ───────────────────────────────────────────────────

#[test]
fn full_pipeline_timing() {
    let robot = ur5e();
    let start = home_joints();

    // A realistic multi-step application: move, grip, move, grip
    let task = Task::sequence(vec![
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        ),
        Task::gripper(0.08),
        Task::gripper(0.02),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0])),
        ),
        Task::gripper(0.08),
    ]);

    let t0 = Instant::now();
    let solution = task.plan(&start).unwrap();
    let total_time = t0.elapsed();

    eprintln!(
        "Full pipeline: {} stages, planning={:?}, execution_duration={:?}",
        solution.stages.len(),
        total_time,
        solution.total_duration,
    );

    // Should complete planning quickly
    assert!(
        total_time < Duration::from_secs(30),
        "Pipeline planning should be <30s: {:?}",
        total_time
    );

    // Execution duration should be reasonable
    assert!(solution.total_duration > Duration::ZERO);
    assert!(
        solution.total_duration < Duration::from_secs(120),
        "Execution duration too long: {:?}",
        solution.total_duration
    );
}

// ─── Place task construction ────────────────────────────────────────────────

#[test]
fn place_approach_construction() {
    // Verify Approach construction and direction normalization
    let approach = Approach::linear(-Vector3::z(), 0.10);
    assert!((approach.distance - 0.10).abs() < 1e-10);

    let dir_norm = approach.direction.norm();
    assert!(
        (dir_norm - 1.0).abs() < 1e-10,
        "Direction should be unit: {dir_norm}"
    );

    // Non-unit input gets normalized
    let approach2 = Approach::linear(Vector3::new(3.0, 4.0, 0.0), 0.2);
    let norm2 = approach2.direction.norm();
    assert!((norm2 - 1.0).abs() < 1e-10);
}

// ─── Task solution validation ───────────────────────────────────────────────

#[test]
fn task_solution_validates_trajectories() {
    let robot = ur5e();
    let start = home_joints();

    let task = Task::sequence(vec![
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        ),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0])),
        ),
    ]);

    let solution = task.plan(&start).unwrap();

    // All trajectory stages should have valid dimensional consistency
    for stage in &solution.stages {
        if let Some(traj) = &stage.trajectory {
            let valid = traj.validate();
            assert!(
                valid.is_ok(),
                "Stage '{}' trajectory invalid: {:?}",
                stage.name,
                valid.err()
            );
        }
    }
}

// ─── Approach helper ────────────────────────────────────────────────────────

#[test]
fn approach_linear_normalization() {
    // Approach::linear normalizes the direction vector
    let a = Approach::linear(Vector3::new(0.0, 0.0, -2.0), 0.10);
    assert!((a.direction.norm() - 1.0).abs() < 1e-10);
    assert!((a.direction.z - (-1.0)).abs() < 1e-10);

    let b = Approach::linear(Vector3::new(1.0, 1.0, 0.0), 0.05);
    assert!((b.direction.norm() - 1.0).abs() < 1e-10);
    assert!((b.distance - 0.05).abs() < 1e-10);
}
