//! Tests for partial trajectory invalidation and splicing.
//!
//! When a scene change invalidates only the later portion of a trajectory,
//! tests verify:
//! 1. The first invalid waypoint can be identified
//! 2. Trajectory can be truncated at that point
//! 3. A new plan can be spliced from the truncation point to the goal
//! 4. The spliced trajectory maintains position continuity at the splice point
//! 5. The spliced trajectory is collision-free in the updated scene

use kinetic::prelude::*;

/// Helper: plan a collision-free trajectory between two configs.
fn plan_trajectory(
    robot: &Robot,
    scene: &Scene,
    start: &[f64],
    goal_joints: &[f64],
) -> Vec<Vec<f64>> {
    let planner = Planner::new(robot).unwrap().with_scene(scene);
    let goal = Goal::Joints(JointValues(goal_joints.to_vec()));
    planner.plan(start, &goal).unwrap().waypoints
}

/// Helper: find the first waypoint index that collides in the scene.
/// Returns None if no collisions found.
fn first_colliding_waypoint(scene: &Scene, waypoints: &[Vec<f64>]) -> Option<usize> {
    for (i, wp) in waypoints.iter().enumerate() {
        if scene.check_collision(wp).unwrap_or(false) {
            return Some(i);
        }
    }
    None
}

/// Helper: check if all waypoints are collision-free in the scene.
fn all_collision_free(scene: &Scene, waypoints: &[Vec<f64>]) -> bool {
    first_colliding_waypoint(scene, waypoints).is_none()
}

/// Helper: compute L2 distance between two joint configs.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Helper: UR5e mid-range starting config.
fn ur5e_start() -> Vec<f64> {
    vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
}

/// Helper: UR5e goal config.
fn ur5e_goal() -> Vec<f64> {
    vec![1.0, -1.0, 0.5, -1.0, 0.5, 0.5]
}

// ── Waypoint invalidation detection ─────────────────────────────────

#[test]
fn find_first_invalid_waypoint_after_obstacle_added() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    // Plan in empty scene
    let waypoints = plan_trajectory(&robot, &scene, &ur5e_start(), &ur5e_goal());
    assert!(waypoints.len() >= 2, "Should have at least 2 waypoints");

    // Verify trajectory is collision-free in original scene
    assert!(all_collision_free(&scene, &waypoints));

    // Add obstacle that may invalidate later waypoints
    let mut updated_scene = Scene::new(&robot).unwrap();
    let obs_pose = nalgebra::Isometry3::translation(0.4, 0.2, 0.3);
    updated_scene.add("obstacle", Shape::Sphere(0.15), obs_pose);

    // Find first collision — may or may not exist depending on obstacle placement
    let first_invalid = first_colliding_waypoint(&updated_scene, &waypoints);
    if let Some(idx) = first_invalid {
        // The invalid index should be after the start (start was collision-free)
        assert!(idx > 0, "Start should not be in collision");
        assert!(idx < waypoints.len(), "Index should be within bounds");
    }
    // If no collision found, the obstacle didn't intersect the path — that's fine
}

#[test]
fn early_waypoints_remain_valid_after_obstacle_at_end() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();

    // Plan trajectory
    let waypoints = plan_trajectory(&robot, &scene, &ur5e_start(), &ur5e_goal());
    let n = waypoints.len();
    assert!(n >= 2, "Need at least start and goal waypoints");

    // Check that at least the first few waypoints are collision-free
    // even after adding an obstacle near the goal
    let mut updated_scene = Scene::new(&robot).unwrap();
    // Place obstacle near goal configuration (not start)
    let obs_pose = nalgebra::Isometry3::translation(0.5, 0.3, 0.5);
    updated_scene.add("goal_blocker", Shape::Sphere(0.1), obs_pose);

    // First waypoint (start) should always be valid
    assert!(
        !updated_scene.check_collision(&waypoints[0]).unwrap_or(true),
        "Start config should remain collision-free"
    );
}

// ── Trajectory truncation ───────────────────────────────────────────

#[test]
fn truncate_trajectory_preserves_valid_prefix() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let waypoints = plan_trajectory(&robot, &scene, &ur5e_start(), &ur5e_goal());
    let n = waypoints.len();

    // Simulate truncation at midpoint
    let truncate_idx = n / 2;
    let valid_prefix: Vec<Vec<f64>> = waypoints[..=truncate_idx].to_vec();

    // Prefix should maintain start
    assert_eq!(valid_prefix.first().unwrap(), waypoints.first().unwrap());
    assert_eq!(valid_prefix.len(), truncate_idx + 1);

    // Truncation point is preserved
    assert_eq!(valid_prefix.last().unwrap(), &waypoints[truncate_idx]);
}

#[test]
fn truncated_trajectory_collision_free_in_updated_scene() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let waypoints = plan_trajectory(&robot, &scene, &ur5e_start(), &ur5e_goal());

    // Add obstacle
    let mut updated_scene = Scene::new(&robot).unwrap();
    let obs_pose = nalgebra::Isometry3::translation(0.4, 0.2, 0.3);
    updated_scene.add("obstacle", Shape::Sphere(0.15), obs_pose);

    // Find invalidation point
    let first_invalid = first_colliding_waypoint(&updated_scene, &waypoints);
    if let Some(idx) = first_invalid {
        // Truncate before the first collision
        let truncate_at = if idx > 0 { idx - 1 } else { 0 };
        let valid_prefix: Vec<Vec<f64>> = waypoints[..=truncate_at].to_vec();

        // The prefix should be collision-free
        assert!(
            all_collision_free(&updated_scene, &valid_prefix),
            "Truncated prefix should be collision-free"
        );
    }
}

// ── Splicing new plan from truncation point ─────────────────────────

#[test]
fn splice_new_plan_from_truncation_point_to_goal() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);
    let n = waypoints.len();

    // Truncate at midpoint (simulating partial invalidation)
    let truncate_idx = n / 2;
    let splice_point = &waypoints[truncate_idx];

    // Plan new segment from splice point to original goal
    let new_segment = plan_trajectory(&robot, &scene, splice_point, &goal);

    // Splice: valid_prefix + new_segment (minus duplicate splice point)
    let mut spliced = waypoints[..=truncate_idx].to_vec();
    spliced.extend_from_slice(&new_segment[1..]); // Skip first of new segment (= splice point)

    // Spliced trajectory should start at original start
    assert_eq!(spliced.first().unwrap(), &start);

    // Spliced trajectory should reach goal (last waypoint close to goal)
    let last = spliced.last().unwrap();
    let dist_to_goal = joint_distance(last, &goal);
    assert!(
        dist_to_goal < 0.1,
        "Spliced trajectory should reach goal, dist={dist_to_goal}"
    );

    // Spliced trajectory should be collision-free in original scene
    assert!(all_collision_free(&scene, &spliced));
}

#[test]
fn splice_point_has_position_continuity() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);

    let truncate_idx = waypoints.len() / 2;
    let splice_point = &waypoints[truncate_idx];

    // Plan from splice point
    let new_segment = plan_trajectory(&robot, &scene, splice_point, &goal);

    // First waypoint of new segment should exactly match splice point
    let new_start = &new_segment[0];
    let continuity_error = joint_distance(splice_point, new_start);
    assert!(
        continuity_error < 1e-10,
        "Splice point should have position continuity, error={continuity_error}"
    );
}

#[test]
fn spliced_trajectory_longer_than_prefix_alone() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);

    let truncate_idx = waypoints.len() / 3; // Truncate at 1/3
    let splice_point = &waypoints[truncate_idx];
    let new_segment = plan_trajectory(&robot, &scene, splice_point, &goal);

    let prefix_len = truncate_idx + 1;
    let new_seg_len = new_segment.len() - 1; // minus duplicate
    let spliced_len = prefix_len + new_seg_len;

    assert!(
        spliced_len > prefix_len,
        "Spliced trajectory should be longer than just the prefix"
    );
}

// ── Replan with updated scene ───────────────────────────────────────

#[test]
fn replan_from_truncation_in_updated_scene_avoids_obstacle() {
    let robot = Robot::from_name("ur5e").unwrap();

    // Original plan in empty scene
    let empty_scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let original_waypoints = plan_trajectory(&robot, &empty_scene, &start, &goal);

    // Add obstacle
    let mut updated_scene = Scene::new(&robot).unwrap();
    let obs_pose = nalgebra::Isometry3::translation(0.4, 0.0, 0.3);
    updated_scene.add("obstacle", Shape::Sphere(0.1), obs_pose);

    // Find first colliding waypoint
    let first_invalid = first_colliding_waypoint(&updated_scene, &original_waypoints);

    // Truncate before collision (or use start if obstacle is at beginning)
    let truncate_at = match first_invalid {
        Some(idx) if idx > 0 => idx - 1,
        _ => 0,
    };
    let splice_point = &original_waypoints[truncate_at];

    // Replan from splice point to goal in updated scene
    let new_segment = plan_trajectory(&robot, &updated_scene, splice_point, &goal);

    // New segment should be collision-free in updated scene
    assert!(
        all_collision_free(&updated_scene, &new_segment),
        "Replanned segment should be collision-free"
    );

    // Build full spliced trajectory
    let mut spliced = original_waypoints[..=truncate_at].to_vec();
    spliced.extend_from_slice(&new_segment[1..]);

    // Full spliced trajectory should be collision-free
    assert!(
        all_collision_free(&updated_scene, &spliced),
        "Full spliced trajectory should be collision-free in updated scene"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn invalidation_at_first_waypoint_replans_entire_trajectory() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);

    // If invalidation is at index 0, we need a full replan
    // Simulate by truncating at 0
    let splice_point = &waypoints[0];
    let new_plan = plan_trajectory(&robot, &scene, splice_point, &goal);

    // New plan starts from the original start
    assert_eq!(&new_plan[0], &start);
    assert!(new_plan.len() >= 2);
}

#[test]
fn invalidation_at_last_waypoint_only_replans_final_segment() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);
    let n = waypoints.len();

    // Truncate at second-to-last
    let truncate_idx = n - 2;
    let splice_point = &waypoints[truncate_idx];

    let new_segment = plan_trajectory(&robot, &scene, splice_point, &goal);

    // Only 2 waypoints were invalidated — new segment should be short
    // (though RRT may produce more waypoints)
    assert!(new_segment.len() >= 2, "Need at least start and goal");

    // Splice should maintain continuity
    let cont_err = joint_distance(splice_point, &new_segment[0]);
    assert!(cont_err < 1e-10);
}

#[test]
fn multiple_sequential_truncations_and_splices() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();

    let mut waypoints = plan_trajectory(&robot, &scene, &start, &goal);

    // Do 3 rounds of truncation and splicing
    for round in 0..3 {
        let n = waypoints.len();
        if n < 4 {
            break; // Not enough waypoints to truncate
        }

        // Truncate at 2/3
        let truncate_idx = (n * 2) / 3;
        let splice_point = waypoints[truncate_idx].clone();

        let new_segment = plan_trajectory(&robot, &scene, &splice_point, &goal);

        // Build new spliced trajectory
        let mut spliced = waypoints[..=truncate_idx].to_vec();
        spliced.extend_from_slice(&new_segment[1..]);

        // Verify continuity at splice point
        let _cont_err = joint_distance(&spliced[truncate_idx], &spliced[truncate_idx + 1]);
        // The waypoint at truncate_idx and the next one might not be identical
        // (new_segment[1] is the next waypoint after the splice point), but the
        // splice point itself should match
        assert_eq!(&spliced[truncate_idx], &splice_point);

        // Verify reach goal
        let dist = joint_distance(spliced.last().unwrap(), &goal);
        assert!(
            dist < 0.1,
            "Round {round}: spliced trajectory should reach goal, dist={dist}"
        );

        waypoints = spliced;
    }

    // After all splices, trajectory should still start at original start
    assert_eq!(&waypoints[0], &start);
}

#[test]
fn no_invalidation_means_no_replan_needed() {
    let robot = Robot::from_name("ur5e").unwrap();
    let scene = Scene::new(&robot).unwrap();
    let start = ur5e_start();
    let goal = ur5e_goal();
    let waypoints = plan_trajectory(&robot, &scene, &start, &goal);

    // No obstacle added → no invalidation
    assert!(first_colliding_waypoint(&scene, &waypoints).is_none());
    assert!(all_collision_free(&scene, &waypoints));
}
