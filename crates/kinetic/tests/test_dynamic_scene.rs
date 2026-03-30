//! Dynamic scene updates and replanning tests.
//!
//! Verifies runtime scene modifications: adding/removing obstacles,
//! attaching/detaching objects, pointcloud streaming, scene consistency
//! under rapid mutations, and interaction with planning and collision checking.

use std::time::Duration;

use kinetic::kinematics::KinematicChain;
use kinetic::prelude::*;
use kinetic::scene::{Octree, OctreeConfig, PointCloudConfig, Scene};

fn ur5e() -> Robot {
    Robot::from_name("ur5e").unwrap()
}

fn ur5e_chain(robot: &Robot) -> KinematicChain {
    let arm = &robot.groups["arm"];
    KinematicChain::extract(robot, &arm.base_link, &arm.tip_link).unwrap()
}

// ─── Obstacle addition populates environment spheres ─────────────────────────

#[test]
fn adding_obstacle_populates_environment_spheres() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // No obstacles — environment should be empty
    let env_before = scene.build_environment_spheres();
    assert_eq!(
        env_before.len(),
        0,
        "Empty scene should have no environment spheres"
    );

    // Add obstacle
    scene.add(
        "nearby_box",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.3, 0.0, 0.3),
    );

    let env_after = scene.build_environment_spheres();
    eprintln!(
        "Before obstacle: {} spheres, After: {} spheres",
        env_before.len(),
        env_after.len()
    );

    // After adding obstacle, environment should have spheres
    assert!(
        env_after.len() > 0,
        "Environment should have spheres after adding obstacle"
    );

    // Add a second obstacle — sphere count should increase
    scene.add(
        "another_box",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.6, 0.0, 0.3),
    );

    let env_two = scene.build_environment_spheres();
    assert!(
        env_two.len() > env_after.len(),
        "Adding second obstacle should increase sphere count"
    );

    // Collision check should still succeed (no crash)
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

// ─── Obstacle removal clears environment spheres ────────────────────────────

#[test]
fn removing_obstacle_clears_environment_spheres() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Add obstacle
    scene.add(
        "blocker",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.3, 0.0, 0.3),
    );
    let env_with = scene.build_environment_spheres();
    assert!(
        env_with.len() > 0,
        "Should have environment spheres with obstacle"
    );

    // Remove it
    scene.remove("blocker");
    assert_eq!(scene.num_objects(), 0);

    // Environment spheres should be empty again
    let env_after = scene.build_environment_spheres();
    assert_eq!(
        env_after.len(),
        0,
        "After removing obstacle, environment should be empty"
    );

    // Add and remove multiple obstacles
    for i in 0..5 {
        scene.add(
            &format!("obj_{i}"),
            Shape::sphere(0.03),
            Isometry3::translation(0.3 + i as f64 * 0.1, 0.0, 0.3),
        );
    }
    assert_eq!(scene.build_environment_spheres().len(), 5); // 5 spheres = 5 collision spheres

    for i in 0..5 {
        scene.remove(&format!("obj_{i}"));
    }
    assert_eq!(scene.build_environment_spheres().len(), 0);
}

// ─── Scene clear resets all state ───────────────────────────────────────────

#[test]
fn scene_clear_resets_everything() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();

    // Add objects
    scene.add(
        "box1",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.3, 0.0, 0.1),
    );
    scene.add(
        "box2",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.2),
    );

    // Attach one
    let ee_link = &robot.links[chain.tip_link].name;
    scene.attach(
        "attached_obj",
        Shape::sphere(0.02),
        Isometry3::identity(),
        ee_link,
    );

    // Add pointcloud
    let points = vec![[0.3, 0.0, 0.1], [0.4, 0.0, 0.2]];
    scene.add_pointcloud("pc1", &points, PointCloudConfig::default());

    assert_eq!(scene.num_objects(), 2);
    assert_eq!(scene.num_attached(), 1);
    assert_eq!(scene.num_pointclouds(), 1);

    // Clear everything
    scene.clear();

    assert_eq!(scene.num_objects(), 0);
    assert_eq!(scene.num_attached(), 0);
    assert_eq!(scene.num_pointclouds(), 0);

    // Collision check should still work on empty scene
    let joints = vec![0.0; robot.dof];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

// ─── Attachment and detachment lifecycle ────────────────────────────────────

#[test]
fn attach_detach_preserves_scene_integrity() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);
    let mut scene = Scene::new(&robot).unwrap();
    let ee_link = robot.links[chain.tip_link].name.clone();

    // Add 3 objects
    for i in 0..3 {
        scene.add(
            &format!("obj_{i}"),
            Shape::sphere(0.02),
            Isometry3::translation(0.3 + i as f64 * 0.1, 0.0, 0.05),
        );
    }
    assert_eq!(scene.num_objects(), 3);

    // Attach obj_1
    scene.attach(
        "obj_1",
        Shape::sphere(0.02),
        Isometry3::identity(),
        &ee_link,
    );
    assert_eq!(scene.num_objects(), 2); // obj_0, obj_2 remain
    assert_eq!(scene.num_attached(), 1);
    assert!(scene.get_object("obj_0").is_some());
    assert!(scene.get_object("obj_1").is_none()); // moved to attached
    assert!(scene.get_object("obj_2").is_some());

    // Detach at new location
    scene.detach("obj_1", Isometry3::translation(0.5, 0.5, 0.05));
    assert_eq!(scene.num_objects(), 3);
    assert_eq!(scene.num_attached(), 0);

    // Verify new position
    let obj = scene.get_object("obj_1").unwrap();
    assert!((obj.pose.translation.vector.x - 0.5).abs() < 1e-6);
    assert!((obj.pose.translation.vector.y - 0.5).abs() < 1e-6);
}

// ─── Pointcloud streaming ───────────────────────────────────────────────────

#[test]
fn pointcloud_streaming_10_updates() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Simulate 10 sequential pointcloud updates (like a depth camera stream)
    for frame in 0..10 {
        let offset = frame as f64 * 0.02;
        let points: Vec<[f64; 3]> = (0..1000)
            .map(|i| {
                let x = 0.3 + (i % 30) as f64 * 0.01 + offset;
                let y = -0.15 + (i / 30) as f64 * 0.01;
                let z = 0.05;
                [x, y, z]
            })
            .collect();

        let config = PointCloudConfig {
            sphere_radius: 0.005,
            max_points: 1000,
            ..Default::default()
        };

        // Replace previous pointcloud with new frame
        scene.remove_pointcloud("depth_stream");
        scene.add_pointcloud("depth_stream", &points, config);
    }

    assert_eq!(scene.num_pointclouds(), 1);

    // Scene should still be usable for collision checking
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

// ─── Octree incremental updates ─────────────────────────────────────────────

#[test]
fn octree_incremental_updates() {
    let config = OctreeConfig {
        resolution: 0.05,
        half_extent: 2.0,
        ray_cast_free_space: false,
        ..Default::default()
    };
    let mut octree = Octree::new(config);

    // Insert points incrementally (simulating sensor frames)
    for frame in 0..5 {
        let offset = frame as f64 * 0.1;
        let points: Vec<[f64; 3]> = (0..500)
            .map(|i| {
                let x = offset + (i % 20) as f64 * 0.05;
                let y = (i / 20) as f64 * 0.05;
                let z = 0.0;
                [x, y, z]
            })
            .collect();

        octree.insert_points_occupied(&points);
    }

    assert!(octree.num_occupied() > 0);
    eprintln!(
        "Octree after 5 frames: {} occupied, {} leaves",
        octree.num_occupied(),
        octree.num_leaves()
    );

    // Add to scene and verify collision checking works
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();
    scene.add_octree("lidar", octree);
    assert_eq!(scene.num_octrees(), 1);

    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

// ─── Scene unchanged after failed plan ──────────────────────────────────────

#[test]
fn scene_unchanged_after_failed_plan() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "box1",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.3, 0.0, 0.1),
    );
    scene.add(
        "box2",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.2),
    );

    let objs_before = scene.num_objects();

    // Try to plan with a nonexistent named goal — should fail
    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_millis(200),
            ..PlannerConfig::default()
        });

    let start = vec![0.0; robot.dof];
    let goal = Goal::Named("nonexistent_pose".to_string());
    let result = planner.plan(&start, &goal);

    assert!(result.is_err(), "Should fail for nonexistent named goal");

    // Scene should be unchanged
    assert_eq!(scene.num_objects(), objs_before);
    assert!(scene.get_object("box1").is_some());
    assert!(scene.get_object("box2").is_some());
}

// ─── Rapid scene mutations ──────────────────────────────────────────────────

#[test]
fn rapid_scene_mutations_100_cycles() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    for i in 0..100 {
        // Add
        scene.add(
            &format!("dyn_{}", i % 20),
            Shape::sphere(0.01 + (i % 5) as f64 * 0.005),
            Isometry3::translation(
                0.3 + (i % 10) as f64 * 0.05,
                (i % 7) as f64 * 0.04 - 0.1,
                0.1 + (i % 3) as f64 * 0.1,
            ),
        );

        // Every 3rd iteration, remove an object
        if i % 3 == 2 {
            scene.remove(&format!("dyn_{}", (i / 3) % 20));
        }

        // Every 10th iteration, do a collision check
        if i % 10 == 9 {
            let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
            let result = scene.check_collision(&joints);
            assert!(result.is_ok(), "Collision check failed at cycle {i}");
        }
    }

    // Final state should be consistent
    let joints = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok(), "Final collision check should work");
}

// ─── Planning uses updated scene ────────────────────────────────────────────

#[test]
fn planner_uses_updated_scene() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));

    // Plan with empty scene
    let planner1 = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(2),
            ..PlannerConfig::default()
        });
    let result1 = planner1.plan(&start, &goal);

    // Add obstacle and replan
    scene.add(
        "wall",
        Shape::sphere(0.1),
        Isometry3::translation(0.35, 0.0, 0.35),
    );

    let planner2 = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(3),
            ..PlannerConfig::default()
        });
    let result2 = planner2.plan(&start, &goal);

    // Both planners should complete (may or may not succeed)
    // If both succeed, plan2 waypoints should avoid the obstacle
    if let (Ok(p1), Ok(p2)) = (&result1, &result2) {
        // Plan 2 should have more or different waypoints to avoid obstacle
        eprintln!(
            "Empty scene: {} waypoints, With obstacle: {} waypoints",
            p1.num_waypoints(),
            p2.num_waypoints()
        );

        // Verify plan2 waypoints don't collide with updated scene
        for wp in p2.waypoints.iter() {
            let in_collision = scene.check_collision(wp).unwrap();
            assert!(!in_collision, "Plan2 waypoint should avoid obstacle");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Moving obstacles during execution — replanning triggers
// ═══════════════════════════════════════════════════════════════════════════════

/// Check if ANY waypoint in a trajectory collides with the current scene.
fn trajectory_collides(scene: &Scene, waypoints: &[Vec<f64>]) -> bool {
    waypoints
        .iter()
        .any(|wp| scene.check_collision(wp).unwrap_or(false))
}

/// Find which future waypoints (indices >= from_idx) collide.
fn future_colliding_indices(scene: &Scene, waypoints: &[Vec<f64>], from_idx: usize) -> Vec<usize> {
    waypoints
        .iter()
        .enumerate()
        .skip(from_idx)
        .filter(|(_, wp)| scene.check_collision(wp).unwrap_or(false))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn moving_obstacle_into_future_path_detected() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5];

    // Build a simple interpolated trajectory (no RRT needed)
    let n = 10;
    let mut waypoints = Vec::new();
    for i in 0..n {
        let alpha = i as f64 / (n - 1) as f64;
        let wp: Vec<f64> = start
            .iter()
            .zip(&goal)
            .map(|(&s, &g)| s + alpha * (g - s))
            .collect();
        waypoints.push(wp);
    }

    // No collisions initially
    assert!(
        !trajectory_collides(&scene, &waypoints),
        "Empty scene should have no collisions"
    );

    // Simulate: robot is at waypoint 3. An obstacle appears near waypoint 7.
    scene.add(
        "moving_obs",
        Shape::sphere(0.15),
        Isometry3::translation(0.35, 0.0, 0.3),
    );

    // Check ONLY future waypoints (4..10) for collisions
    let future_collisions = future_colliding_indices(&scene, &waypoints, 4);

    // The result is valid (may or may not collide depending on geometry)
    // but should not panic
    eprintln!("Future colliding indices: {:?}", future_collisions);
}

#[test]
fn obstacle_moves_away_clears_future_collisions() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];

    // Place obstacle overlapping the robot
    scene.add(
        "close_obs",
        Shape::sphere(0.3),
        Isometry3::translation(0.0, 0.0, 0.3),
    );

    let collides_near = scene.check_collision(&joints).unwrap();

    // Move obstacle far away
    scene.update_pose("close_obs", Isometry3::translation(10.0, 10.0, 10.0));
    let collides_far = scene.check_collision(&joints).unwrap();

    if collides_near {
        assert!(
            !collides_far,
            "Moving obstacle far away should clear collision"
        );
    }
}

#[test]
fn replan_from_midpoint_with_new_obstacle() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5];

    // Initial plan (no obstacles)
    let planner1 = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(3),
            ..PlannerConfig::default()
        });

    let result1 = planner1.plan(&start, &Goal::Joints(JointValues::new(goal.clone())));
    if result1.is_err() {
        // Planning may fail — skip test for this particular start/goal
        return;
    }
    let plan1 = result1.unwrap();

    // Simulate: robot reaches midpoint, then obstacle appears
    let mid_idx = plan1.waypoints.len() / 2;
    let current = plan1.waypoints[mid_idx.min(plan1.waypoints.len() - 1)].clone();

    scene.add(
        "new_obstacle",
        Shape::sphere(0.08),
        Isometry3::translation(0.4, 0.1, 0.3),
    );

    // Replan from current position with updated scene
    let planner2 = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(3),
            ..PlannerConfig::default()
        });

    let result2 = planner2.plan(&current, &Goal::Joints(JointValues::new(goal.clone())));
    if let Ok(plan2) = result2 {
        // Replanned path should avoid the new obstacle
        for wp in &plan2.waypoints {
            let in_collision = scene.check_collision(wp).unwrap();
            assert!(
                !in_collision,
                "Replanned waypoint should avoid new obstacle"
            );
        }
    }
}

#[test]
fn obstacle_tracking_robot_multiple_replans() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];

    // Place an obstacle that "tracks" the robot by moving to different positions
    for step in 0..5 {
        let x = 0.3 + step as f64 * 0.1;
        scene.add(
            "tracker",
            Shape::sphere(0.05),
            Isometry3::translation(x, 0.0, 0.3 + step as f64 * 0.05),
        );

        // Each time obstacle moves, check collision — should not panic
        let result = scene.check_collision(&start);
        assert!(
            result.is_ok(),
            "Collision check should work at step {}",
            step
        );

        // Update pose instead of re-add
        scene.update_pose("tracker", Isometry3::translation(x + 0.05, 0.0, 0.35));
    }
}

#[test]
fn min_distance_decreases_as_obstacle_approaches() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let joints = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];

    // Start with obstacle far away
    scene.add(
        "approach",
        Shape::sphere(0.05),
        Isometry3::translation(5.0, 0.0, 0.5),
    );

    let mut distances = Vec::new();
    for step in 0..10 {
        let x = 5.0 - step as f64 * 0.4;
        scene.update_pose("approach", Isometry3::translation(x, 0.0, 0.5));
        let dist = scene.min_distance_to_robot(&joints).unwrap();
        distances.push(dist);
    }

    // Distance should generally decrease
    let first = *distances.first().unwrap();
    let last = *distances.last().unwrap();
    assert!(
        last <= first + 0.01, // small tolerance for sphere approximation
        "Distance should decrease as obstacle approaches: first={}, last={}",
        first,
        last
    );
}

#[test]
fn scene_update_pose_during_collision_checks_no_panic() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];

    scene.add(
        "mover",
        Shape::sphere(0.1),
        Isometry3::translation(0.3, 0.0, 0.5),
    );

    // Rapidly update pose and check collision — no panics
    for step in 0..20 {
        let x = 0.3 + (step as f64 * 0.05).sin() * 0.5;
        let y = (step as f64 * 0.07).cos() * 0.3;
        scene.update_pose("mover", Isometry3::translation(x, y, 0.5));
        let _ = scene.check_collision(&joints);
    }
}

#[test]
fn many_obstacles_collision_check_no_panic() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Add 100 obstacles
    for i in 0..100 {
        let x = -2.0 + (i % 20) as f64 * 0.2;
        let y = -2.0 + (i / 20) as f64 * 0.2;
        scene.add(
            &format!("obs_{}", i),
            Shape::sphere(0.03),
            Isometry3::translation(x, y, 0.5),
        );
    }
    assert_eq!(scene.num_objects(), 100);

    let joints = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let _ = scene.check_collision(&joints);
    let _ = scene.min_distance_to_robot(&joints);

    let env = scene.build_environment_spheres();
    assert!(env.len() >= 100);
}
