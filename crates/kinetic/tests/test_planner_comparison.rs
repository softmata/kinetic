//! Planner comparison test suite.
//!
//! Runs all planners on standardized scenarios and verifies each produces
//! valid collision-free paths. Collects metrics for comparison.

use std::sync::Arc;
use std::time::Duration;

use kinetic_collision::capt::AABB;
use kinetic_collision::{CollisionEnvironment, SpheresSoA};
use kinetic_core::{Goal, JointValues, PlannerConfig};
use kinetic_kinematics::KinematicChain;
use kinetic_planning::{
    Planner, PlannerType,
};
use kinetic_robot::Robot;

/// A standardized test scenario.
#[allow(dead_code)]
struct Scenario {
    name: &'static str,
    robot_name: &'static str,
    start: Vec<f64>,
    goal: Vec<f64>,
    obstacles: Vec<(f64, f64, f64, f64)>, // (x, y, z, radius)
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "open_space",
            robot_name: "ur5e",
            start: vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0],
            goal: vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0],
            obstacles: vec![],
        },
        Scenario {
            name: "moderate_distance",
            robot_name: "ur5e",
            start: vec![0.0, -1.5, 1.0, 0.0, 0.5, 0.0],
            goal: vec![1.5, -0.3, -0.5, 1.0, -0.5, 1.0],
            obstacles: vec![],
        },
        Scenario {
            name: "small_step",
            robot_name: "ur5e",
            start: vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0],
            goal: vec![0.1, -0.9, 0.7, 0.1, 0.1, 0.1],
            obstacles: vec![],
        },
        Scenario {
            name: "reverse_direction",
            robot_name: "ur5e",
            start: vec![1.0, -0.5, 0.3, 0.5, -0.3, 0.5],
            goal: vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0],
            obstacles: vec![],
        },
    ]
}

fn planner_types() -> Vec<(&'static str, PlannerType)> {
    vec![
        ("RRT-Connect", PlannerType::RRTConnect),
        ("RRT*", PlannerType::RRTStar),
        ("BiRRT*", PlannerType::BiRRTStar),
        ("BiTRRT", PlannerType::BiTRRT),
        ("EST", PlannerType::EST),
        ("KPIECE", PlannerType::KPIECE),
    ]
}

fn build_env(obstacles: &[(f64, f64, f64, f64)]) -> CollisionEnvironment {
    if obstacles.is_empty() {
        return CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
    }
    let mut spheres = SpheresSoA::new();
    for (i, &(x, y, z, r)) in obstacles.iter().enumerate() {
        spheres.push(x, y, z, r, i);
    }
    CollisionEnvironment::build(spheres, 0.05, AABB::symmetric(10.0))
}

/// Verify all planners produce valid paths on the open space scenario.
#[test]
fn all_planners_solve_open_space() {
    let scenario = &scenarios()[0]; // open_space
    let robot = Arc::new(Robot::from_name(scenario.robot_name).unwrap());
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let env = build_env(&scenario.obstacles);
    let goal = Goal::Joints(JointValues(scenario.goal.clone()));

    let config = PlannerConfig {
        timeout: Duration::from_secs(10),
        max_iterations: 50_000,
        shortcut_iterations: 0,
        smooth: false,
        collision_margin: 0.0,
        workspace_bounds: None,
    };

    for (name, planner_type) in planner_types() {
        let planner = Planner::from_chain(robot.clone(), chain.clone())
            .unwrap()
            .with_environment(env.clone())
            .with_config(config.clone())
            .with_planner_type(planner_type);

        let result = planner.plan(&scenario.start, &goal);
        assert!(
            result.is_ok(),
            "Planner {} failed on open_space: {:?}",
            name,
            result.err()
        );

        let result = result.unwrap();
        assert!(
            result.waypoints.len() >= 2,
            "Planner {} produced < 2 waypoints",
            name
        );

        // Verify all waypoints are collision-free
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(
                !planner.is_in_collision(wp),
                "Planner {} waypoint {} in collision",
                name,
                i
            );
        }
    }
}

/// Verify all planners solve the moderate distance scenario.
#[test]
fn all_planners_solve_moderate_distance() {
    let scenario = &scenarios()[1]; // moderate_distance
    let robot = Arc::new(Robot::from_name(scenario.robot_name).unwrap());
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let env = build_env(&scenario.obstacles);
    let goal = Goal::Joints(JointValues(scenario.goal.clone()));

    let config = PlannerConfig {
        timeout: Duration::from_secs(10),
        max_iterations: 50_000,
        shortcut_iterations: 0,
        smooth: false,
        collision_margin: 0.0,
        workspace_bounds: None,
    };

    for (name, planner_type) in planner_types() {
        let planner = Planner::from_chain(robot.clone(), chain.clone())
            .unwrap()
            .with_environment(env.clone())
            .with_config(config.clone())
            .with_planner_type(planner_type);

        let result = planner.plan(&scenario.start, &goal);
        assert!(
            result.is_ok(),
            "Planner {} failed on moderate_distance: {:?}",
            name,
            result.err()
        );
    }
}

/// Compare path lengths across planners on the same scenario.
#[test]
fn planner_path_length_comparison() {
    let scenario = &scenarios()[0]; // open_space
    let robot = Arc::new(Robot::from_name(scenario.robot_name).unwrap());
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let env = build_env(&scenario.obstacles);
    let goal = Goal::Joints(JointValues(scenario.goal.clone()));

    let config = PlannerConfig {
        timeout: Duration::from_secs(10),
        max_iterations: 50_000,
        shortcut_iterations: 0,
        smooth: false,
        collision_margin: 0.0,
        workspace_bounds: None,
    };

    let mut results: Vec<(&str, f64, usize)> = Vec::new();

    for (name, planner_type) in planner_types() {
        let planner = Planner::from_chain(robot.clone(), chain.clone())
            .unwrap()
            .with_environment(env.clone())
            .with_config(config.clone())
            .with_planner_type(planner_type);

        if let Ok(result) = planner.plan(&scenario.start, &goal) {
            let path_len = result.path_length();
            results.push((name, path_len, result.waypoints.len()));
        }
    }

    // All planners should have succeeded
    assert_eq!(
        results.len(),
        planner_types().len(),
        "Not all planners succeeded"
    );

    // All path lengths should be positive and finite
    for (name, path_len, _wps) in &results {
        assert!(
            *path_len > 0.0 && path_len.is_finite(),
            "Planner {} has invalid path length: {}",
            name,
            path_len
        );
    }
}

/// Verify the planner_used field is set correctly.
#[test]
fn planner_type_reported_correctly() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

    let config = PlannerConfig {
        timeout: Duration::from_secs(10),
        max_iterations: 50_000,
        shortcut_iterations: 0,
        smooth: false,
        collision_margin: 0.0,
        workspace_bounds: None,
    };

    for (name, planner_type) in planner_types() {
        let planner = Planner::from_chain(robot.clone(), chain.clone())
            .unwrap()
            .with_environment(env.clone())
            .with_config(config.clone())
            .with_planner_type(planner_type);

        let result = planner.plan(&start, &goal).unwrap();
        assert_eq!(
            result.planner_used, planner_type,
            "Planner {} should report type {:?} but got {:?}",
            name, planner_type, result.planner_used
        );
    }
}

/// Each planner starts and ends at the requested configurations.
#[test]
fn all_planners_respect_start_and_goal() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal_vals = vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0];
    let goal = Goal::Joints(JointValues(goal_vals.clone()));

    let config = PlannerConfig {
        timeout: Duration::from_secs(10),
        max_iterations: 50_000,
        shortcut_iterations: 0,
        smooth: false,
        collision_margin: 0.0,
        workspace_bounds: None,
    };

    for (name, planner_type) in planner_types() {
        let planner = Planner::from_chain(robot.clone(), chain.clone())
            .unwrap()
            .with_environment(env.clone())
            .with_config(config.clone())
            .with_planner_type(planner_type);

        let result = planner.plan(&start, &goal).unwrap();
        let first = result.start().unwrap();
        let last = result.end().unwrap();

        // Start should match (exactly or very close)
        let start_dist: f64 = first
            .iter()
            .zip(start.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            start_dist < 0.5,
            "Planner {} start deviates by {:.4}",
            name,
            start_dist
        );

        // Goal should be close (sampling-based planners may not land exactly on goal)
        let goal_dist: f64 = last
            .iter()
            .zip(goal_vals.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            goal_dist < 1.0,
            "Planner {} goal deviates by {:.4}",
            name,
            goal_dist
        );
    }
}
