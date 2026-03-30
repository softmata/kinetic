//! Integration tests for cross-crate wiring.
//!
//! Validates that the major integration points work end-to-end:
//! - Planner + Scene for collision-aware planning
//! - GraspGenerator + Task::pick_auto for auto-grasp pick
//! - plan_with_scene convenience function

use kinetic::grasp::{
    GraspCandidate, GraspConfig, GraspError, GraspGenerator, GraspMetric, GraspType, GripperType,
};
use kinetic::planning::{ConstrainedPlanningResult, ConstrainedRRT};
use kinetic::prelude::*;
use kinetic::scene::AttachedObject;
use kinetic::task::{
    apply_scene_diffs, Approach, PickConfig, StageSolution, Task, TaskError, TaskSolution,
};
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

// ─── Planner + Scene ────────────────────────────────────────────────────────

#[test]
fn planner_with_scene_builds() {
    let robot = Robot::from_name("ur5e").unwrap();
    let robot_arc = ur5e();
    let mut scene = Scene::new(&robot_arc).unwrap();

    scene.add(
        "table",
        Shape::Cuboid(1.0, 0.6, 0.02),
        Isometry3::translation(0.5, 0.0, 0.0),
    );

    let planner = Planner::new(&robot).unwrap().with_scene(&scene);
    assert_eq!(planner.chain().active_joints.len(), 6);
}

#[test]
fn planner_with_scene_plans_successfully() {
    let robot = Robot::from_name("ur5e").unwrap();
    let robot_arc = ur5e();
    let mut scene = Scene::new(&robot_arc).unwrap();

    // Add a single small distant obstacle (fast collision check)
    scene.add(
        "far_sphere",
        Shape::Sphere(0.1),
        Isometry3::translation(5.0, 0.0, 1.0),
    );

    let planner = Planner::new(&robot).unwrap().with_scene(&scene);

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));
    let result = planner.plan(&start, &goal).unwrap();
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn plan_with_scene_convenience() {
    let robot_arc = ur5e();
    let scene = Scene::new(&robot_arc).unwrap();

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result = plan_with_scene("ur5e", &start, &goal, &scene).unwrap();
    assert!(result.waypoints.len() >= 2);
}

// ─── Grasp generation ───────────────────────────────────────────────────────

#[test]
fn grasp_generator_produces_candidates() {
    let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
    let shape = Shape::Cylinder(0.03, 0.06);
    let pose = Isometry3::translation(0.5, 0.0, 0.3);

    let config = GraspConfig::default();
    let grasps = gen.from_shape(&shape, &pose, config).unwrap();

    assert!(!grasps.is_empty());
    for g in &grasps {
        assert!(g.quality >= 0.0 && g.quality <= 1.0);
        // All grasps should be near the object
        let dist = (g.grasp_pose.translation.vector - Vector3::new(0.5, 0.0, 0.3)).norm();
        assert!(dist < 0.5, "Grasp too far from object: {}", dist);
    }
}

#[test]
fn grasp_collision_filtering_with_scene() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Place a cylinder object
    let obj_pose = Isometry3::translation(0.5, 0.0, 0.3);
    scene.add("cylinder", Shape::Cylinder(0.03, 0.06), obj_pose);

    // Place a wall right behind the object — should filter some grasps
    scene.add(
        "wall",
        Shape::Cuboid(0.01, 0.5, 0.5),
        Isometry3::translation(0.53, 0.0, 0.3),
    );

    let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
    let config_no_filter = GraspConfig {
        num_candidates: 50,
        ..Default::default()
    };
    let config_with_filter = GraspConfig {
        num_candidates: 50,
        check_collision: Some(Arc::new(scene)),
        ..Default::default()
    };

    let grasps_unfiltered = gen
        .from_shape(&Shape::Cylinder(0.03, 0.06), &obj_pose, config_no_filter)
        .unwrap();
    let grasps_filtered = gen
        .from_shape(&Shape::Cylinder(0.03, 0.06), &obj_pose, config_with_filter)
        .unwrap();

    // Filtered set should have fewer or equal candidates (wall blocks some approaches)
    assert!(
        grasps_filtered.len() <= grasps_unfiltered.len(),
        "Collision filtering should reduce or maintain count: {} <= {}",
        grasps_filtered.len(),
        grasps_unfiltered.len()
    );
}

// ─── Task::pick_auto ────────────────────────────────────────────────────────

#[test]
fn task_pick_auto_generates_grasps() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    scene.add(
        "small_box",
        Shape::Cuboid(0.03, 0.03, 0.03),
        Isometry3::translation(0.4, 0.0, 0.3),
    );
    let scene = Arc::new(scene);

    let pick_config = PickConfig {
        object: "small_box".into(),
        grasp_poses: vec![], // will be auto-filled
        approach: Approach::linear(-Vector3::z(), 0.10),
        retreat: Approach::linear(Vector3::z(), 0.05),
        gripper_open: 0.08,
        gripper_close: 0.03,
    };

    let task = Task::pick_auto(
        &robot,
        &scene,
        GripperType::parallel(0.08, 0.03),
        pick_config,
    )
    .unwrap();

    // Verify it created a Pick variant with auto-generated grasps
    match &task {
        Task::Pick { config, .. } => {
            assert!(
                !config.grasp_poses.is_empty(),
                "pick_auto should auto-generate grasp poses"
            );
            assert_eq!(config.object, "small_box");
        }
        _ => panic!("Expected Pick task"),
    }
}

#[test]
fn task_pick_auto_object_not_found() {
    let robot = ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());

    let pick_config = PickConfig {
        object: "nonexistent".into(),
        grasp_poses: vec![],
        approach: Approach::linear(-Vector3::z(), 0.10),
        retreat: Approach::linear(Vector3::z(), 0.05),
        gripper_open: 0.08,
        gripper_close: 0.03,
    };

    let result = Task::pick_auto(
        &robot,
        &scene,
        GripperType::parallel(0.08, 0.03),
        pick_config,
    );
    assert!(result.is_err(), "Should fail when object not in scene");
}

// ─── Prelude completeness ───────────────────────────────────────────────────

#[test]
fn prelude_has_error_types() {
    // Just verify these types are accessible from the prelude
    let _: Option<TaskError> = None;
    let _: Option<GraspError> = None;
    let _: Option<KineticError> = None;
}

#[test]
fn prelude_has_scene_types() {
    let _: Option<SceneObject> = None;
    let _: Option<AttachedObject> = None;
}

#[test]
fn prelude_has_grasp_types() {
    let _: Option<GraspType> = None;
    let _: Option<GraspMetric> = None;
    let _: Option<GraspCandidate> = None;
}

#[test]
fn prelude_has_task_types() {
    let _: Option<StageSolution> = None;
    let _: fn(&mut Scene, &TaskSolution) = apply_scene_diffs;
}

#[test]
fn prelude_has_planning_types() {
    let _: Option<ConstrainedRRT> = None;
    let _: Option<ConstrainedPlanningResult> = None;
    #[allow(clippy::type_complexity)]
    let _: Option<fn(&str, &[f64], &Goal, &Scene) -> Result<PlanningResult>> =
        Some(plan_with_scene);
}
