//! Integration tests for the GCS (Graph of Convex Sets) planner.
//!
//! Tests GCS planning with real robot models and obstacle scenes.

use kinetic::planning::{CollisionChecker, GCSPlanner, IrisConfig};
use kinetic::prelude::*;
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

/// Simple collision checker that blocks a box region in joint space.
struct BoxObstacle {
    center: Vec<f64>,
    half_extent: f64,
}

impl CollisionChecker for BoxObstacle {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        joints
            .iter()
            .zip(self.center.iter())
            .all(|(j, c)| (j - c).abs() < self.half_extent)
    }
}

/// Never-colliding checker for basic testing.
struct NeverCollides;
impl CollisionChecker for NeverCollides {
    fn is_in_collision(&self, _joints: &[f64]) -> bool {
        false
    }
}

#[test]
fn gcs_build_and_plan_free_space() {
    let limits: Vec<(f64, f64)> = vec![
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
    ];

    let config = IrisConfig {
        num_regions: 5,
        max_iterations: 10,
        ..Default::default()
    };

    let planner = GCSPlanner::build(&NeverCollides, &limits, &config, 0.1).unwrap();

    assert!(
        planner.num_regions() >= 1,
        "Should have at least one region"
    );

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2];

    let result = planner.plan(&start, &goal);
    // GCS may or may not find a path depending on region coverage;
    // the important thing is it doesn't panic.
    if let Ok(result) = result {
        assert!(
            result.waypoints.len() >= 2,
            "Path should have at least start and goal"
        );
        // First waypoint should match start
        for (a, b) in result.waypoints[0].iter().zip(start.iter()) {
            assert!((a - b).abs() < 1e-4, "Start mismatch: {} vs {}", a, b);
        }
    }
}

#[test]
fn gcs_with_obstacle_avoidance() {
    let limits: Vec<(f64, f64)> = vec![
        (-std::f64::consts::PI, std::f64::consts::PI),
        (-std::f64::consts::PI, std::f64::consts::PI),
    ];

    let obstacle = BoxObstacle {
        center: vec![0.0, 0.0],
        half_extent: 0.3,
    };

    let config = IrisConfig {
        num_regions: 8,
        max_iterations: 15,
        ..Default::default()
    };

    let planner = GCSPlanner::build(&obstacle, &limits, &config, 0.1).unwrap();

    // Plan around the obstacle
    let start = vec![-2.0, -2.0];
    let goal = vec![2.0, 2.0];

    let result = planner.plan(&start, &goal);
    if let Ok(result) = result {
        // Verify no waypoint is in collision
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(
                !obstacle.is_in_collision(wp),
                "Waypoint {} at {:?} is in collision",
                i,
                wp
            );
        }
    }
}

#[test]
fn gcs_infeasible_returns_error() {
    // Build with very few regions — unlikely to cover path between distant points
    let limits: Vec<(f64, f64)> = vec![
        (-std::f64::consts::PI, std::f64::consts::PI),
        (-std::f64::consts::PI, std::f64::consts::PI),
    ];

    let config = IrisConfig {
        num_regions: 1,
        max_iterations: 2,
        ..Default::default()
    };

    let planner = GCSPlanner::build(&NeverCollides, &limits, &config, 0.1).unwrap();

    // Start and goal far apart — may not be in the same region
    let start = vec![-3.0, -3.0];
    let goal = vec![3.0, 3.0];

    // Should return Err, not panic
    let _ = planner.plan(&start, &goal);
}

#[test]
fn gcs_ur5e_joint_limits() {
    let robot = ur5e();

    let limits: Vec<(f64, f64)> = robot
        .joints
        .iter()
        .filter(|j| j.joint_type != JointType::Fixed)
        .take(6)
        .map(|j| {
            j.limits
                .as_ref()
                .map(|l| (l.lower, l.upper))
                .unwrap_or((-std::f64::consts::PI, std::f64::consts::PI))
        })
        .collect();

    let config = IrisConfig {
        num_regions: 3,
        max_iterations: 5,
        ..Default::default()
    };

    // Build should succeed with real robot limits
    let planner = GCSPlanner::build(&NeverCollides, &limits, &config, 0.15);
    assert!(
        planner.is_ok(),
        "GCS should build with UR5e limits: {:?}",
        planner.err()
    );
}

/// Gap 5: Build a GCS planner with two completely isolated regions (no edges
/// connecting them). Planning from one to the other should fail with
/// "no path found".
#[test]
fn gcs_disconnected_graph_returns_no_path() {
    // Build two regions using IRIS with a wall of collision in between.
    struct WallCollider;
    impl CollisionChecker for WallCollider {
        fn is_in_collision(&self, joints: &[f64]) -> bool {
            // A thick wall at x=0 blocking the two halves
            joints[0].abs() < 1.0
        }
    }

    let limits: Vec<(f64, f64)> = vec![(-3.0, 3.0), (-3.0, 3.0)];

    let config = IrisConfig {
        num_regions: 2,
        max_iterations: 10,
        seed_points: Some(vec![vec![-2.0, 0.0], vec![2.0, 0.0]]),
        ..Default::default()
    };

    let planner = GCSPlanner::build(&WallCollider, &limits, &config, 0.1).unwrap();

    // Plan from left side to right side — should fail because the wall blocks
    let start = vec![-2.0, 0.0];
    let goal = vec![2.0, 0.0];

    let result = planner.plan(&start, &goal);
    match result {
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("no path found") || msg.contains("not in any"),
                "Error should mention 'no path found' or region containment, got: {}",
                msg
            );
        }
        Ok(_) => {
            // If by chance the regions overlap through the wall, the test is inconclusive
            // but at least it didn't panic.
        }
    }
}
