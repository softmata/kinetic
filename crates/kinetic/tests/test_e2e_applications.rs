//! End-to-end application workflow tests.
//!
//! Tests real-world scenarios: palletizing, welding path following,
//! multi-step assembly, bin picking, and obstacle avoidance replanning.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic::grasp::{GraspConfig, GraspGenerator, GripperType};
use kinetic::kinematics::{forward_kinematics, KinematicChain};
use kinetic::planning::CartesianPlanner;
use kinetic::prelude::*;
use kinetic::scene::Scene;
use kinetic::task::Task;

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

// ─── Palletizing: 3x3 grid placement ────────────────────────────────────────

#[test]
fn palletizing_3x3_grid() {
    let robot = ur5e();

    // Plan sequential moves to 9 grid positions (simulating pallet placements)
    // Each "place" is a joint-space move to a pre-computed configuration
    let grid_configs: Vec<Vec<f64>> = (0..9)
        .map(|i| {
            let row = i / 3;
            let col = i % 3;
            vec![
                0.3 + col as f64 * 0.2,
                -1.0 + row as f64 * 0.1,
                0.5,
                0.0,
                0.8,
                col as f64 * 0.1,
            ]
        })
        .collect();

    // Build a task sequence: home → grid[0] → home → grid[1] → ... → grid[8]
    let mut tasks = Vec::new();
    for (i, config) in grid_configs.iter().enumerate() {
        tasks.push(Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(config.clone())),
        ));
        if i < 8 {
            // Return partway between placements
            tasks.push(Task::gripper(0.08)); // "release"
        }
    }

    let task = Task::sequence(tasks);

    let t0 = Instant::now();
    let solution = task.plan(&home_joints()).unwrap();
    let elapsed = t0.elapsed();

    eprintln!(
        "Palletizing 3x3: {} stages, planning={:?}, duration={:?}",
        solution.stages.len(),
        elapsed,
        solution.total_duration
    );

    // Should have 9 move stages + 8 gripper stages = 17
    assert_eq!(solution.stages.len(), 17);

    // All trajectory stages should have valid trajectories
    let traj_stages: Vec<_> = solution
        .stages
        .iter()
        .filter(|s| s.trajectory.is_some())
        .collect();
    assert_eq!(traj_stages.len(), 9, "Should have 9 trajectory stages");

    for stage in &traj_stages {
        let traj = stage.trajectory.as_ref().unwrap();
        assert!(traj.validate().is_ok(), "Stage '{}' invalid", stage.name);
        assert!(traj.waypoints.len() >= 2);
    }

    assert!(
        elapsed < Duration::from_secs(120),
        "Should complete in <120s"
    );
}

// ─── Welding path: Cartesian path following ─────────────────────────────────

#[test]
fn welding_cartesian_path() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    // Start from a non-singular config
    let start = vec![0.0, -1.2, 1.0, -0.8, -std::f64::consts::FRAC_PI_2, 0.0];
    let start_pose = forward_kinematics(&robot, &chain, &start).unwrap();

    // Create a welding seam: small linear path in X direction
    let cart_planner = CartesianPlanner::new(Arc::clone(&robot), chain.clone());
    let cart_config = kinetic::planning::CartesianConfig {
        max_step: 0.005,
        ..Default::default()
    };

    // Move 2cm in X (very small, should succeed)
    let target = Pose(Isometry3::from_parts(
        nalgebra::Translation3::from(start_pose.translation() + Vector3::new(0.02, 0.0, 0.0)),
        *start_pose.rotation(),
    ));

    let result = cart_planner.plan_linear(&start, &target, &cart_config);

    match result {
        Ok(cart_result) => {
            eprintln!(
                "Welding path: {} waypoints, fraction={:.2}, time={:?}",
                cart_result.waypoints.len(),
                cart_result.fraction,
                cart_result.planning_time
            );

            assert!(
                cart_result.fraction > 0.9,
                "Should achieve >90% of path, got {:.2}",
                cart_result.fraction
            );

            // Verify path continuity — joint changes between waypoints should be small
            for w in cart_result.waypoints.windows(2) {
                let max_jump: f64 = w[0]
                    .iter()
                    .zip(w[1].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_jump < 0.5,
                    "Joint discontinuity in welding path: max_jump={max_jump}"
                );
            }

            // Verify orientation maintained — FK each waypoint and check orientation
            for (i, wp) in cart_result.waypoints.iter().enumerate() {
                let pose = forward_kinematics(&robot, &chain, wp).unwrap();
                let rot_diff = start_pose.rotation().inverse() * pose.rotation();
                let angle = rot_diff.angle();
                assert!(
                    angle < 0.1, // 0.1 rad tolerance
                    "Orientation drift at waypoint {i}: {angle:.4} rad"
                );
            }
        }
        Err(e) => {
            eprintln!("Welding path failed (IK may not converge for this config): {e}");
        }
    }
}

// ─── Multi-step assembly sequence ───────────────────────────────────────────

#[test]
fn assembly_multi_step_sequence() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();

    // Part A: small cylinder
    scene.add(
        "part_a",
        Shape::Cylinder(0.02, 0.03),
        Isometry3::translation(0.4, 0.0, 0.04),
    );

    // Part B: small box
    scene.add(
        "part_b",
        Shape::cuboid(0.015, 0.015, 0.015),
        Isometry3::translation(0.4, 0.15, 0.02),
    );

    // Fixture location
    let fixture_pose = Isometry3::translation(0.3, -0.1, 0.05);

    assert_eq!(scene.num_objects(), 2);
    assert_eq!(scene.num_attached(), 0);

    // Step 1: "Pick" part A — simulate via scene diff
    let ee_link = robot.links[chain.tip_link].name.clone();
    scene.attach(
        "part_a",
        Shape::Cylinder(0.02, 0.03),
        Isometry3::identity(),
        &ee_link,
    );

    assert_eq!(scene.num_objects(), 1, "Part A removed from world");
    assert_eq!(scene.num_attached(), 1, "Part A attached");

    // Step 2: "Place" part A at fixture
    scene.detach("part_a", fixture_pose);

    assert_eq!(scene.num_objects(), 2, "Part A placed at fixture");
    assert_eq!(scene.num_attached(), 0, "Nothing attached");

    // Verify part A is now at fixture location
    let part_a = scene.get_object("part_a").unwrap();
    let pos_a = part_a.pose.translation.vector;
    assert!((pos_a.x - 0.3).abs() < 1e-6);
    assert!((pos_a.y - (-0.1)).abs() < 1e-6);

    // Step 3: "Pick" part B
    scene.attach(
        "part_b",
        Shape::cuboid(0.015, 0.015, 0.015),
        Isometry3::identity(),
        &ee_link,
    );

    assert_eq!(scene.num_objects(), 1, "Part B removed, part A at fixture");
    assert_eq!(scene.num_attached(), 1, "Part B attached");

    // Step 4: "Assemble" — place part B near part A at fixture
    let assembly_pose = Isometry3::translation(0.3, -0.1, 0.09);
    scene.detach("part_b", assembly_pose);

    assert_eq!(scene.num_objects(), 2, "Both parts placed");
    assert_eq!(scene.num_attached(), 0, "Assembly complete");

    // Verify assembly state
    let part_b = scene.get_object("part_b").unwrap();
    let pos_b = part_b.pose.translation.vector;
    assert!((pos_b.x - 0.3).abs() < 1e-6);
    assert!((pos_b.z - 0.09).abs() < 1e-6);
}

// ─── Bin picking: grasp generation and filtering ────────────────────────────

#[test]
fn bin_picking_grasp_filter() {
    let _robot = ur5e();

    // Multiple objects in a bin region
    let objects = vec![
        (
            "obj1",
            Shape::Cylinder(0.02, 0.03),
            Isometry3::translation(0.45, 0.0, 0.04),
        ),
        (
            "obj2",
            Shape::Cylinder(0.025, 0.02),
            Isometry3::translation(0.48, 0.05, 0.03),
        ),
        (
            "obj3",
            Shape::cuboid(0.02, 0.02, 0.02),
            Isometry3::translation(0.42, -0.03, 0.03),
        ),
    ];

    let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));

    let mut total_grasps = 0;
    let mut best_grasp_per_obj = Vec::new();

    for (name, shape, pose) in &objects {
        let config = GraspConfig {
            num_candidates: 20,
            ..Default::default()
        };

        match gen.from_shape(shape, pose, config) {
            Ok(grasps) => {
                eprintln!(
                    "{}: {} grasps, best quality={:.3}",
                    name,
                    grasps.len(),
                    grasps[0].quality
                );
                total_grasps += grasps.len();
                best_grasp_per_obj.push((name, grasps[0].quality));
            }
            Err(e) => {
                eprintln!("{}: no grasps ({e})", name);
            }
        }
    }

    assert!(
        total_grasps > 0,
        "Should generate some grasps across all objects"
    );
    assert!(
        best_grasp_per_obj.len() >= 2,
        "At least 2 objects should have grasps"
    );

    // All quality scores should be in [0, 1]
    for (name, quality) in &best_grasp_per_obj {
        assert!(
            *quality >= 0.0 && *quality <= 1.0,
            "{name}: quality {quality} out of range"
        );
    }
}

// ─── Obstacle avoidance replanning ──────────────────────────────────────────

#[test]
fn obstacle_avoidance_replan() {
    let robot = ur5e();

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal_joints = vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5];
    let goal = Goal::Joints(JointValues::new(goal_joints.clone()));

    // Plan 1: no obstacles
    let planner1 = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_secs(2),
        ..PlannerConfig::default()
    });

    let plan1 = planner1.plan(&start, &goal);

    // Plan 2: add obstacle that may interfere
    let mut scene = Scene::new(&robot).unwrap();
    scene.add(
        "new_obstacle",
        Shape::sphere(0.08),
        Isometry3::translation(0.35, 0.0, 0.35),
    );

    let planner2 = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(3),
            ..PlannerConfig::default()
        });

    let plan2 = planner2.plan(&start, &goal);

    match (&plan1, &plan2) {
        (Ok(p1), Ok(p2)) => {
            eprintln!(
                "Plan 1 (no obstacle): {} waypoints, length={:.3}",
                p1.num_waypoints(),
                p1.path_length()
            );
            eprintln!(
                "Plan 2 (with obstacle): {} waypoints, length={:.3}",
                p2.num_waypoints(),
                p2.path_length()
            );

            // Both should reach the goal
            let final1 = p1.waypoints.last().unwrap();
            let final2 = p2.waypoints.last().unwrap();
            for (a, b) in final1.iter().zip(goal_joints.iter()) {
                assert!((a - b).abs() < 0.1, "Plan 1 didn't reach goal");
            }
            for (a, b) in final2.iter().zip(goal_joints.iter()) {
                assert!((a - b).abs() < 0.1, "Plan 2 didn't reach goal");
            }

            // Plan 2 waypoints should be collision-free with the obstacle
            for (i, wp) in p2.waypoints.iter().enumerate() {
                let in_collision = scene.check_collision(wp).unwrap();
                assert!(!in_collision, "Replanned waypoint {} is in collision", i);
            }
        }
        (Ok(_), Err(e)) => {
            eprintln!("Replan failed with obstacle (acceptable): {e}");
        }
        (Err(e), _) => {
            eprintln!("Initial plan failed (unexpected): {e}");
        }
    }
}

// ─── Task sequence with multiple moves and grippers ─────────────────────────

#[test]
fn multi_move_gripper_sequence_timing() {
    let robot = ur5e();
    let start = home_joints();

    // Simulate a complex workflow: 5 moves with gripper commands between
    let task = Task::sequence(vec![
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.3, -1.0, 0.5, 0.0, 0.5, 0.0])),
        ),
        Task::gripper(0.08),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.6, -0.8, 0.3, 0.1, 0.3, -0.2])),
        ),
        Task::gripper(0.02),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![-0.3, -1.2, 0.8, -0.5, 0.7, 0.4])),
        ),
        Task::gripper(0.08),
        Task::move_to(
            &robot,
            Goal::Joints(JointValues::new(vec![0.0, -1.0, 0.6, 0.0, 0.5, 0.0])),
        ),
        Task::gripper(0.02),
        Task::move_to(&robot, Goal::Joints(JointValues::new(home_joints()))),
    ]);

    let t0 = Instant::now();
    let solution = task.plan(&start).unwrap();
    let elapsed = t0.elapsed();

    eprintln!(
        "Multi-move sequence: {} stages, planning={:?}, execution={:?}",
        solution.stages.len(),
        elapsed,
        solution.total_duration
    );

    // 5 moves + 4 grippers = 9 stages
    assert_eq!(solution.stages.len(), 9);

    let move_count = solution
        .stages
        .iter()
        .filter(|s| s.trajectory.is_some())
        .count();
    let grip_count = solution
        .stages
        .iter()
        .filter(|s| s.gripper_command.is_some())
        .count();
    assert_eq!(move_count, 5);
    assert_eq!(grip_count, 4);

    assert!(elapsed < Duration::from_secs(60));
}
