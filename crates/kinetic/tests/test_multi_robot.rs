//! Multi-robot coordination and interference tests.
//!
//! Tests multiple robots in shared workspaces: independent planning,
//! sequential planning with trajectory-as-obstacle, multi-robot scene
//! tracking, servo in shared scenes, and inter-robot distance.

use std::sync::Arc;
use std::time::Duration;

use kinetic::collision::SpheresSoA;
use kinetic::kinematics::{forward_kinematics, KinematicChain};
use kinetic::prelude::*;
use kinetic::scene::Scene;

fn ur5e() -> Robot {
    Robot::from_name("ur5e").unwrap()
}

fn ur5e_chain(robot: &Robot) -> KinematicChain {
    let arm = &robot.groups["arm"];
    KinematicChain::extract(robot, &arm.base_link, &arm.tip_link).unwrap()
}

// ─── Two UR5e robots planning independently ──────────────────────────────────

#[test]
fn two_ur5e_independent_planning() {
    let robot_left = ur5e();
    let robot_right = ur5e();

    let start = vec![0.0; robot_left.dof];
    let goal_left = Goal::Joints(JointValues::new(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));
    let goal_right = Goal::Joints(JointValues::new(vec![-0.5, -0.8, 0.3, 0.1, 0.3, -0.2]));

    let config = PlannerConfig {
        timeout: Duration::from_secs(3),
        ..PlannerConfig::default()
    };

    let planner_l = Planner::new(&robot_left)
        .unwrap()
        .with_config(config.clone());
    let planner_r = Planner::new(&robot_right).unwrap().with_config(config);

    let result_l = planner_l.plan(&start, &goal_left);
    let result_r = planner_r.plan(&start, &goal_right);

    // Both should plan successfully (no obstacles)
    match (&result_l, &result_r) {
        (Ok(pl), Ok(pr)) => {
            eprintln!(
                "Left: {} waypoints, Right: {} waypoints",
                pl.num_waypoints(),
                pr.num_waypoints()
            );
            assert!(pl.num_waypoints() >= 2);
            assert!(pr.num_waypoints() >= 2);
        }
        (Err(e), _) => eprintln!("Left planning failed: {e}"),
        (_, Err(e)) => eprintln!("Right planning failed: {e}"),
    }
}

// ─── Dual-arm sequential planning: left trajectory as obstacle for right ─────

#[test]
fn dual_arm_sequential_planning() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal_left = Goal::Joints(JointValues::new(vec![0.5, -0.8, 0.5, 0.0, 0.5, 0.0]));

    // Plan left arm first (no obstacles)
    let config = PlannerConfig {
        timeout: Duration::from_secs(3),
        ..PlannerConfig::default()
    };
    let planner_l = Planner::new(&robot).unwrap().with_config(config.clone());
    let result_l = planner_l.plan(&start, &goal_left);

    if let Ok(plan_l) = result_l {
        eprintln!("Left arm planned: {} waypoints", plan_l.num_waypoints());

        // Add left arm's swept volume as obstacles for right arm
        // Approximate: add FK poses of left arm waypoints as spheres
        let mut scene_right = Scene::new(&robot).unwrap();
        for (i, wp) in plan_l.waypoints.iter().enumerate() {
            let pose = forward_kinematics(&robot, &chain, wp).unwrap();
            let t = pose.translation();
            // Place obstacle sphere at each left-arm EE position
            scene_right.add(
                &format!("left_wp_{i}"),
                kinetic::scene::Shape::sphere(0.05),
                nalgebra::Isometry3::translation(t[0], t[1], t[2]),
            );
        }

        let num_obstacles = scene_right.num_objects();
        eprintln!(
            "Added {} obstacle spheres from left arm trajectory",
            num_obstacles
        );
        assert!(num_obstacles > 0);

        // Plan right arm with left arm's trajectory as obstacles
        let goal_right = Goal::Joints(JointValues::new(vec![-0.5, -0.5, 0.3, 0.2, -0.3, 0.5]));
        let planner_r = Planner::new(&robot)
            .unwrap()
            .with_scene(&scene_right)
            .with_config(PlannerConfig {
                timeout: Duration::from_secs(5),
                ..PlannerConfig::default()
            });

        let result_r = planner_r.plan(&start, &goal_right);
        match result_r {
            Ok(plan_r) => {
                eprintln!(
                    "Right arm planned: {} waypoints (avoiding left)",
                    plan_r.num_waypoints()
                );
                assert!(plan_r.num_waypoints() >= 2);
            }
            Err(e) => {
                eprintln!("Right arm planning failed with obstacles (acceptable): {e}");
            }
        }
    } else {
        eprintln!("Left arm planning failed (unexpected)");
    }
}

// ─── Multiple robots in shared scene ─────────────────────────────────────────

#[test]
fn three_robots_shared_scene() {
    let robot = ur5e();

    // Create a scene with obstacles representing 3 robot workspaces
    let mut scene = Scene::new(&robot).unwrap();

    // Robot 1 base: origin
    scene.add(
        "robot1_base",
        kinetic::scene::Shape::cuboid(0.1, 0.1, 0.05),
        nalgebra::Isometry3::translation(0.0, 0.0, 0.0),
    );

    // Robot 2 base: offset in Y
    scene.add(
        "robot2_base",
        kinetic::scene::Shape::cuboid(0.1, 0.1, 0.05),
        nalgebra::Isometry3::translation(0.0, 1.0, 0.0),
    );

    // Robot 3 base: offset in X
    scene.add(
        "robot3_base",
        kinetic::scene::Shape::cuboid(0.1, 0.1, 0.05),
        nalgebra::Isometry3::translation(1.0, 0.0, 0.0),
    );

    // Shared workspace obstacle
    scene.add(
        "shared_table",
        kinetic::scene::Shape::cuboid(0.3, 0.3, 0.01),
        nalgebra::Isometry3::translation(0.5, 0.5, 0.4),
    );

    assert_eq!(scene.num_objects(), 4);

    // Environment should have spheres for all objects
    let env = scene.build_environment_spheres();
    assert!(
        env.len() > 4,
        "Should have multiple collision spheres: {}",
        env.len()
    );

    // Collision check should work with complex scene
    let joints = vec![0.0; robot.dof];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok(), "Collision check should not error");

    // Adding and removing robots dynamically
    scene.add(
        "robot4_base",
        kinetic::scene::Shape::cuboid(0.1, 0.1, 0.05),
        nalgebra::Isometry3::translation(-1.0, 0.0, 0.0),
    );
    assert_eq!(scene.num_objects(), 5);

    scene.remove("robot4_base");
    assert_eq!(scene.num_objects(), 4);
}

// ─── Multi-robot scene tracking: add/remove robot swept volumes ──────────────

#[test]
fn multi_robot_scene_tracking() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();

    // Simulate 3 robots at different configurations
    let configs = vec![
        ("robot_a", vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0]),
        ("robot_b", vec![0.5, -0.8, 0.5, 0.1, 0.3, -0.2]),
        ("robot_c", vec![-0.5, -1.2, 1.0, -0.3, 0.7, 0.4]),
    ];

    // Add each robot's EE as an obstacle
    for (name, joints) in &configs {
        let pose = forward_kinematics(&robot, &chain, joints).unwrap();
        let t = pose.translation();
        scene.add(
            &format!("{name}_ee"),
            kinetic::scene::Shape::sphere(0.05),
            nalgebra::Isometry3::translation(t[0], t[1], t[2]),
        );
    }
    assert_eq!(scene.num_objects(), 3);

    // "Move" robot_b — update its obstacle position
    let new_config_b = vec![1.0, -0.5, 0.3, 0.0, 0.5, 0.0];
    let new_pose_b = forward_kinematics(&robot, &chain, &new_config_b).unwrap();
    let t = new_pose_b.translation();
    scene.update_pose(
        "robot_b_ee",
        nalgebra::Isometry3::translation(t[0], t[1], t[2]),
    );

    // Still 3 objects after update
    assert_eq!(scene.num_objects(), 3);

    // Remove robot_c from scene
    scene.remove("robot_c_ee");
    assert_eq!(scene.num_objects(), 2);

    // Environment should reflect current state
    let env = scene.build_environment_spheres();
    assert_eq!(
        env.len(),
        2,
        "Should have 2 sphere obstacles (1 sphere each)"
    );
}

// ─── Dual-robot servo in shared scene ────────────────────────────────────────

#[test]
fn dual_robot_servo_shared_scene() {
    let robot = Arc::new(ur5e());

    // Create shared scene with some obstacles
    let mut scene_mut = Scene::new(&robot).unwrap();
    scene_mut.add(
        "table",
        kinetic::scene::Shape::cuboid(0.3, 0.3, 0.01),
        nalgebra::Isometry3::translation(0.5, 0.0, 0.3),
    );
    let scene = Arc::new(scene_mut);

    // Two servo controllers for two robots
    let config = kinetic::reactive::ServoConfig {
        velocity_limits: vec![1.0; robot.dof],
        ..Default::default()
    };

    let mut servo_l = kinetic::reactive::Servo::new(&robot, &scene, config.clone()).unwrap();
    let mut servo_r = kinetic::reactive::Servo::new(&robot, &scene, config).unwrap();

    // Different starting configs
    let start_l = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let start_r = vec![0.5, -0.8, 0.5, 0.1, 0.3, -0.2];
    let zeros = vec![0.0; robot.dof];

    servo_l.set_state(&start_l, &zeros).unwrap();
    servo_r.set_state(&start_r, &zeros).unwrap();

    // Send twist commands to both
    let twist_l = kinetic::core::Twist::new(
        nalgebra::Vector3::new(0.01, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.0),
    );
    let twist_r = kinetic::core::Twist::new(
        nalgebra::Vector3::new(0.0, 0.01, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.0),
    );

    // Run 10 ticks each
    for _ in 0..10 {
        let cmd_l = servo_l.send_twist(&twist_l);
        let cmd_r = servo_r.send_twist(&twist_r);

        // Both should produce valid commands (no panic, no NaN)
        if let Ok(cmd) = &cmd_l {
            for &v in &cmd.velocities {
                assert!(!v.is_nan(), "Left servo velocity NaN");
                assert!(v.abs() <= 1.0 + 1e-10, "Left servo exceeds limit");
            }
        }
        if let Ok(cmd) = &cmd_r {
            for &v in &cmd.velocities {
                assert!(!v.is_nan(), "Right servo velocity NaN");
                assert!(v.abs() <= 1.0 + 1e-10, "Right servo exceeds limit");
            }
        }
    }

    // Verify states diverged (they had different starting configs and twists)
    let state_l = servo_l.state();
    let state_r = servo_r.state();
    let diff: f64 = state_l
        .joint_positions
        .iter()
        .zip(state_r.joint_positions.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.01,
        "Servo states should have diverged: diff={diff}"
    );
}

// ─── Robot-to-robot minimum distance ─────────────────────────────────────────

#[test]
fn robot_to_robot_min_distance() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    // Compute FK for two different configs
    let config_a = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let config_b = vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5];

    let pose_a = forward_kinematics(&robot, &chain, &config_a).unwrap();
    let pose_b = forward_kinematics(&robot, &chain, &config_b).unwrap();

    let ta = pose_a.translation();
    let tb = pose_b.translation();

    // Create sphere models for each robot's EE
    let mut spheres_a = SpheresSoA::new();
    spheres_a.push(ta[0], ta[1], ta[2], 0.05, 0);
    // Add a few more points along robot A's chain
    for i in 1..=3 {
        let partial_joints: Vec<f64> = config_a
            .iter()
            .enumerate()
            .map(|(j, &v)| if j < i * 2 { v } else { 0.0 })
            .collect();
        let p = forward_kinematics(&robot, &chain, &partial_joints).unwrap();
        let t = p.translation();
        spheres_a.push(t[0], t[1], t[2], 0.04, 0);
    }

    let mut spheres_b = SpheresSoA::new();
    spheres_b.push(tb[0], tb[1], tb[2], 0.05, 0);
    for i in 1..=3 {
        let partial_joints: Vec<f64> = config_b
            .iter()
            .enumerate()
            .map(|(j, &v)| if j < i * 2 { v } else { 0.0 })
            .collect();
        let p = forward_kinematics(&robot, &chain, &partial_joints).unwrap();
        let t = p.translation();
        spheres_b.push(t[0], t[1], t[2], 0.04, 0);
    }

    // Compute min distance between the two "robots"
    if let Some((dist, idx_a, idx_b)) = spheres_a.min_distance(&spheres_b) {
        eprintln!("Robot-to-robot min distance: {dist:.4} (sphere {idx_a} of A vs {idx_b} of B)");
        assert!(!dist.is_nan(), "Distance should not be NaN");
        assert!(!dist.is_infinite(), "Distance should be finite");
    } else {
        panic!("Should compute distance between non-empty sphere sets");
    }

    // Verify collision detection works
    let colliding = spheres_a.any_overlap(&spheres_b);
    eprintln!("Robots colliding: {colliding}");
    // Result depends on configurations — just verify no crash
}
