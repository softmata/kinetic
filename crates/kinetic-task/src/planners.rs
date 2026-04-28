//! Per-stage planner helpers used by [`Task::plan`].
//!
//! All functions are `pub(crate)` and called from `impl Task` in lib.rs. They
//! return `Vec<StageSolution>` so the caller can flatten them into the final
//! [`TaskSolution`] without knowing how many sub-stages each primitive produced.

use std::sync::Arc;

use nalgebra::Isometry3;

use kinetic_core::{Goal, Pose};
use kinetic_kinematics::KinematicChain;
use kinetic_planning::{CartesianConfig, CartesianPlanner, Planner};
use kinetic_robot::Robot;
use kinetic_scene::{Scene, Shape};
use kinetic_trajectory::trapezoidal_per_joint;

use crate::{PickConfig, PlaceConfig, SceneDiff, StageSolution, TaskError};

/// Default joint velocity (rad/s) when the URDF doesn't provide a `velocity` limit.
pub(crate) const FALLBACK_MAX_VEL: f64 = 1.0;
/// Default joint acceleration (rad/s²) when no acceleration limit is provided.
pub(crate) const FALLBACK_MAX_ACCEL: f64 = 2.0;

/// Pull per-joint velocity / acceleration limits from the robot model.
///
/// Falls back to `FALLBACK_MAX_VEL` / `FALLBACK_MAX_ACCEL` when limits are absent.
pub(crate) fn extract_joint_vel_accel(robot: &Robot) -> (Vec<f64>, Vec<f64>) {
    let vel: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|lim| {
            if lim.velocity > 0.0 {
                lim.velocity
            } else {
                FALLBACK_MAX_VEL
            }
        })
        .collect();
    let accel: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|lim| lim.acceleration.unwrap_or(FALLBACK_MAX_ACCEL))
        .collect();
    (vel, accel)
}

/// Plan a move-to using the RRT-Connect planner.
pub(crate) fn plan_move_to(
    robot: &Robot,
    start: &[f64],
    goal: &Goal,
) -> std::result::Result<Vec<StageSolution>, TaskError> {
    let planner = Planner::new(robot)?;
    let result = planner
        .plan(start, goal)
        .map_err(|e| TaskError::PlanningFailed {
            stage: "move_to".into(),
            reason: e.to_string(),
        })?;

    let (vel_limits, accel_limits) = extract_joint_vel_accel(robot);
    let traj = trapezoidal_per_joint(&result.waypoints, &vel_limits, &accel_limits)
        .map_err(|e| TaskError::Trajectory(e.to_string()))?;

    Ok(vec![StageSolution {
        name: "move_to".into(),
        trajectory: Some(traj),
        gripper_command: None,
        scene_diff: None,
    }])
}

/// Plan a Cartesian linear move.
pub(crate) fn plan_cartesian_move(
    robot: &Robot,
    start: &[f64],
    target_pose: &Isometry3<f64>,
    cart_config: &CartesianConfig,
) -> std::result::Result<Vec<StageSolution>, TaskError> {
    let robot_arc = Arc::new(robot.clone());
    let chain = detect_chain(robot)?;
    let cart_planner = CartesianPlanner::new(robot_arc, chain);

    let goal_pose = Pose(*target_pose);
    let result = cart_planner
        .plan_linear(start, &goal_pose, cart_config)
        .map_err(|e| TaskError::PlanningFailed {
            stage: "cartesian_move".into(),
            reason: e.to_string(),
        })?;

    if result.fraction < 0.95 {
        return Err(TaskError::PlanningFailed {
            stage: "cartesian_move".into(),
            reason: format!(
                "Cartesian path only achieved {:.0}% of requested motion",
                result.fraction * 100.0
            ),
        });
    }

    // Cartesian moves use 50% of joint limits for smoother, safer motion
    let (vel_limits, accel_limits) = extract_joint_vel_accel(robot);
    let cart_vel: Vec<f64> = vel_limits.iter().map(|v| v * 0.5).collect();
    let cart_accel: Vec<f64> = accel_limits.iter().map(|a| a * 0.5).collect();
    let traj = trapezoidal_per_joint(&result.waypoints, &cart_vel, &cart_accel)
        .map_err(|e| TaskError::Trajectory(e.to_string()))?;

    Ok(vec![StageSolution {
        name: "cartesian_move".into(),
        trajectory: Some(traj),
        gripper_command: None,
        scene_diff: None,
    }])
}

/// Plan a complete pick sequence: move→approach→grasp→attach→retreat.
///
/// Tries each grasp candidate. Returns solution for first successful candidate.
pub(crate) fn plan_pick(
    robot: &Robot,
    scene: &Arc<Scene>,
    start_joints: &[f64],
    config: &PickConfig,
) -> std::result::Result<Vec<StageSolution>, TaskError> {
    let chain = detect_chain(robot)?;

    // Get object info from scene for attach
    let obj = scene
        .get_object(&config.object)
        .ok_or_else(|| TaskError::ObjectNotFound(config.object.clone()))?;
    let obj_shape = obj.shape.clone();

    // Try each grasp candidate
    let mut last_error = None;

    for (idx, grasp_pose) in config.grasp_poses.iter().enumerate() {
        match try_pick_with_grasp(
            robot,
            &chain,
            start_joints,
            grasp_pose,
            config,
            &obj_shape,
            idx,
        ) {
            Ok(stages) => return Ok(stages),
            Err(e) => {
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or(TaskError::NoValidGrasp(config.object.clone())))
}

/// Try to plan a pick with a specific grasp pose.
fn try_pick_with_grasp(
    robot: &Robot,
    chain: &KinematicChain,
    start_joints: &[f64],
    grasp_pose: &Isometry3<f64>,
    config: &PickConfig,
    obj_shape: &Shape,
    _grasp_idx: usize,
) -> std::result::Result<Vec<StageSolution>, TaskError> {
    let cart_config = CartesianConfig::default();
    let mut stages = Vec::new();
    let mut current_joints = start_joints.to_vec();

    // 1. Pre-grasp pose (offset by approach)
    let pre_grasp = config.approach.offset_pose(grasp_pose);

    // 2. Move to pre-grasp via RRT
    let move_stages = plan_move_to(robot, &current_joints, &Goal::Pose(Pose(pre_grasp)))?;
    if let Some(last_traj) = move_stages.last().and_then(|s| s.trajectory.as_ref()) {
        if let Some(wp) = last_traj.waypoints.last() {
            current_joints = wp.positions.clone();
        }
    }
    stages.extend(move_stages.into_iter().map(|mut s| {
        s.name = "pick_move_to_pregrasp".into();
        s
    }));

    // 3. Open gripper
    stages.push(StageSolution {
        name: "pick_open_gripper".into(),
        trajectory: None,
        gripper_command: Some(config.gripper_open),
        scene_diff: None,
    });

    // 4. Cartesian approach to grasp pose
    let approach_stages = plan_cartesian_move(robot, &current_joints, grasp_pose, &cart_config)?;
    if let Some(last_traj) = approach_stages.last().and_then(|s| s.trajectory.as_ref()) {
        if let Some(wp) = last_traj.waypoints.last() {
            current_joints = wp.positions.clone();
        }
    }
    stages.extend(approach_stages.into_iter().map(|mut s| {
        s.name = "pick_approach".into();
        s
    }));

    // 5. Close gripper
    stages.push(StageSolution {
        name: "pick_close_gripper".into(),
        trajectory: None,
        gripper_command: Some(config.gripper_close),
        scene_diff: None,
    });

    // 6. Attach object to end-effector link
    let ee_link_name = robot.links[chain.tip_link].name.clone();
    stages.push(StageSolution {
        name: "pick_attach".into(),
        trajectory: None,
        gripper_command: None,
        scene_diff: Some(SceneDiff::Attach {
            object: config.object.clone(),
            link: ee_link_name,
            shape: obj_shape.clone(),
            grasp_transform: *grasp_pose,
        }),
    });

    // 7. Cartesian retreat
    let post_grasp = config.retreat.offset_pose(grasp_pose);
    let retreat_stages = plan_cartesian_move(robot, &current_joints, &post_grasp, &cart_config)?;
    stages.extend(retreat_stages.into_iter().map(|mut s| {
        s.name = "pick_retreat".into();
        s
    }));

    Ok(stages)
}

/// Plan a complete place sequence: move→approach→release→detach→retreat.
pub(crate) fn plan_place(
    robot: &Robot,
    _scene: &Arc<Scene>,
    start_joints: &[f64],
    config: &PlaceConfig,
) -> std::result::Result<Vec<StageSolution>, TaskError> {
    let cart_config = CartesianConfig::default();
    let mut stages = Vec::new();
    let mut current_joints = start_joints.to_vec();

    // 1. Pre-place pose (offset by approach)
    let pre_place = config.approach.offset_pose(&config.target_pose);

    // 2. Move to pre-place via RRT
    let move_stages = plan_move_to(robot, &current_joints, &Goal::Pose(Pose(pre_place)))?;
    if let Some(last_traj) = move_stages.last().and_then(|s| s.trajectory.as_ref()) {
        if let Some(wp) = last_traj.waypoints.last() {
            current_joints = wp.positions.clone();
        }
    }
    stages.extend(move_stages.into_iter().map(|mut s| {
        s.name = "place_move_to_preplace".into();
        s
    }));

    // 3. Cartesian approach to place pose
    let approach_stages =
        plan_cartesian_move(robot, &current_joints, &config.target_pose, &cart_config)?;
    if let Some(last_traj) = approach_stages.last().and_then(|s| s.trajectory.as_ref()) {
        if let Some(wp) = last_traj.waypoints.last() {
            current_joints = wp.positions.clone();
        }
    }
    stages.extend(approach_stages.into_iter().map(|mut s| {
        s.name = "place_approach".into();
        s
    }));

    // 4. Open gripper to release
    stages.push(StageSolution {
        name: "place_open_gripper".into(),
        trajectory: None,
        gripper_command: Some(config.gripper_open),
        scene_diff: None,
    });

    // 5. Detach object
    stages.push(StageSolution {
        name: "place_detach".into(),
        trajectory: None,
        gripper_command: None,
        scene_diff: Some(SceneDiff::Detach {
            object: config.object.clone(),
            place_pose: config.target_pose,
        }),
    });

    // 6. Cartesian retreat
    let post_place = config.retreat.offset_pose(&config.target_pose);
    let retreat_stages = plan_cartesian_move(robot, &current_joints, &post_place, &cart_config)?;
    stages.extend(retreat_stages.into_iter().map(|mut s| {
        s.name = "place_retreat".into();
        s
    }));

    Ok(stages)
}

/// Auto-detect kinematic chain for a robot (same logic as Planner/Scene).
pub(crate) fn detect_chain(
    robot: &Robot,
) -> std::result::Result<KinematicChain, TaskError> {
    // Try planning groups first
    if let Some(group) = robot.groups.values().next() {
        return KinematicChain::extract(robot, &group.base_link, &group.tip_link)
            .map_err(TaskError::Kinetic);
    }

    // Fall back: root link -> farthest leaf
    let root = &robot.links[0].name;
    let mut farthest_link = root.clone();
    let mut max_depth = 0;

    fn walk(robot: &Robot, link_idx: usize, depth: usize, max: &mut usize, farthest: &mut String) {
        if depth > *max {
            *max = depth;
            *farthest = robot.links[link_idx].name.clone();
        }
        for j in robot.joints.iter() {
            if j.parent_link == link_idx {
                walk(robot, j.child_link, depth + 1, max, farthest);
            }
        }
    }

    walk(robot, 0, 0, &mut max_depth, &mut farthest_link);

    KinematicChain::extract(robot, root, &farthest_link).map_err(TaskError::Kinetic)
}
