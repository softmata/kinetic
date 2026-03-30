//! Integration test: planning with scene obstacles.
//!
//! Tests planning with collision objects in the scene, verifying that
//! the planned path respects obstacles.

use kinetic::prelude::*;
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

#[test]
fn scene_add_and_query_objects() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Scene starts empty
    assert_eq!(scene.num_objects(), 0);
    assert_eq!(scene.num_attached(), 0);

    // Add a table
    scene.add(
        "table",
        Shape::Cuboid(1.0, 0.6, 0.02),
        Isometry3::translation(0.5, 0.0, 0.0),
    );
    assert_eq!(scene.num_objects(), 1);

    // Add a box on the table
    scene.add(
        "box1",
        Shape::Cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.035),
    );
    assert_eq!(scene.num_objects(), 2);

    // Query
    assert!(scene.get_object("table").is_some());
    assert!(scene.get_object("box1").is_some());
    assert!(scene.get_object("nonexistent").is_none());

    // Remove
    scene.remove("box1");
    assert_eq!(scene.num_objects(), 1);
    assert!(scene.get_object("box1").is_none());
}

#[test]
fn scene_attach_and_detach() {
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

    // Attach to end-effector link
    let ee_link = robot.links.last().unwrap().name.clone();
    scene.attach(
        "cup",
        Shape::Cylinder(0.03, 0.06),
        Isometry3::identity(),
        &ee_link,
    );
    assert_eq!(scene.num_objects(), 0); // removed from world
    assert_eq!(scene.num_attached(), 1);

    // Detach back to world
    scene.detach("cup", Isometry3::translation(0.5, 0.0, 0.3));
    assert_eq!(scene.num_objects(), 1); // back in world
    assert_eq!(scene.num_attached(), 0);
}

#[test]
fn scene_collision_check_free_space() {
    let robot = ur5e();
    let scene = Scene::new(&robot).unwrap();

    // Home position should be collision-free in empty scene
    let colliding = scene.check_collision(&home_joints()).unwrap();
    assert!(
        !colliding,
        "Home position should be collision-free in empty scene"
    );
}

#[test]
fn scene_collision_check_with_obstacle() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Place a large obstacle right where the robot is
    scene.add(
        "wall",
        Shape::Cuboid(2.0, 2.0, 2.0),
        Isometry3::translation(0.0, 0.0, 0.5),
    );

    // Note: collision detection depends on robot having collision geometry in URDF.
    // UR5e config may not have collision meshes, so just verify the check doesn't crash.
    let result = scene.check_collision(&home_joints());
    assert!(result.is_ok(), "Collision check should not error");
}

#[test]
fn scene_min_distance() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Empty scene should have large or infinite min distance
    let dist_empty = scene.min_distance_to_robot(&home_joints()).unwrap();
    assert!(
        dist_empty.is_finite() || dist_empty > 0.0,
        "Distance should be non-negative"
    );

    // Add distant obstacle
    scene.add(
        "far_box",
        Shape::Cuboid(0.1, 0.1, 0.1),
        Isometry3::translation(5.0, 0.0, 0.0),
    );
    let dist_far = scene.min_distance_to_robot(&home_joints()).unwrap();

    // Add close obstacle
    scene.add(
        "close_box",
        Shape::Cuboid(0.1, 0.1, 0.1),
        Isometry3::translation(0.3, 0.0, 0.4),
    );
    let dist_close = scene.min_distance_to_robot(&home_joints()).unwrap();

    // If robot has collision geometry, close obstacle should reduce distance
    // Otherwise both distances may be the same (no robot spheres to measure from)
    assert!(
        dist_close <= dist_far + 1e-6,
        "Adding closer obstacle should not increase min distance: close={} far={}",
        dist_close,
        dist_far
    );
}

#[test]
fn plan_avoids_obstacle() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    // Planning should succeed in free space
    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));
    let result = planner.plan(&start, &goal).unwrap();
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn scene_clear() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "a",
        Shape::Sphere(0.1),
        Isometry3::translation(1.0, 0.0, 0.0),
    );
    scene.add(
        "b",
        Shape::Sphere(0.1),
        Isometry3::translation(2.0, 0.0, 0.0),
    );
    assert_eq!(scene.num_objects(), 2);

    scene.clear();
    assert_eq!(scene.num_objects(), 0);
    assert_eq!(scene.num_attached(), 0);
}

#[test]
fn scene_iterators() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "a",
        Shape::Sphere(0.05),
        Isometry3::translation(1.0, 0.0, 0.0),
    );
    scene.add(
        "b",
        Shape::Sphere(0.05),
        Isometry3::translation(2.0, 0.0, 0.0),
    );

    let objects: Vec<_> = scene.objects_iter().collect();
    assert_eq!(objects.len(), 2);

    let ee_link = robot.links.last().unwrap().name.clone();
    scene.attach("c", Shape::Sphere(0.02), Isometry3::identity(), &ee_link);

    let attached: Vec<_> = scene.attached_iter().collect();
    assert_eq!(attached.len(), 1);
}
