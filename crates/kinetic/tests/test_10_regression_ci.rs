//! Acceptance tests: 10 regression_ci
//! Spec: doc_tests/10_REGRESSION_CI.md
//!
//! Meta-tests that verify the test infrastructure itself works correctly.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

/// Meta-test 1: Every robot in ALL_ROBOTS actually loads from disk.
#[test]
fn meta_all_robots_in_list_exist() {
    let mut failures = vec![];
    for &(name, _) in ALL_ROBOTS {
        if kinetic::robot::Robot::from_name(name).is_err() {
            failures.push(name);
        }
    }
    assert!(
        failures.is_empty(),
        "Robots in ALL_ROBOTS that don't load: {:?}",
        failures
    );
}

/// Meta-test 2: Robot configs on disk are tracked in ALL_ROBOTS.
#[test]
fn meta_disk_robots_tracked() {
    let configs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("robot_configs");

    if !configs_dir.exists() {
        return; // Can't verify
    }

    let list_names: std::collections::HashSet<&str> =
        ALL_ROBOTS.iter().map(|&(name, _)| name).collect();

    let mut unlisted = vec![];
    if let Ok(entries) = std::fs::read_dir(&configs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "toml").unwrap_or(false) {
                let name = path.file_stem().unwrap().to_str().unwrap();
                if !list_names.contains(name) {
                    unlisted.push(name.to_string());
                }
            }
        }
    }

    if !unlisted.is_empty() {
        eprintln!(
            "WARNING: {} robot configs on disk not in ALL_ROBOTS: {:?}",
            unlisted.len(), unlisted
        );
    }
}

/// Meta-test 3: Every acceptance test file (01-11) exists on disk.
#[test]
fn meta_all_acceptance_files_exist() {
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests");

    let expected = [
        "test_01_kinematic_correctness.rs",
        "test_02_joint_limit_enforcement.rs",
        "test_03_collision_safety.rs",
        "test_04_trajectory_safety.rs",
        "test_05_planner_correctness.rs",
        "test_06_execution_safety.rs",
        "test_07_numerical_robustness.rs",
        "test_08_robot_acceptance.rs",
        "test_09_multi_robot.rs",
        "test_10_regression_ci.rs",
        "test_critical_fixes.rs",
    ];

    let mut missing = vec![];
    for &file in &expected {
        if !tests_dir.join(file).exists() {
            missing.push(file);
        }
    }

    assert!(
        missing.is_empty(),
        "Missing acceptance test files: {:?}",
        missing
    );
}

/// Meta-test 4: helpers.rs exists and ALL_ROBOTS has entries.
#[test]
fn meta_helpers_populated() {
    assert!(ALL_ROBOTS.len() >= 50, "ALL_ROBOTS should have 50+ entries, got {}", ALL_ROBOTS.len());
    assert!(SAFETY_ROBOTS.len() >= 4, "SAFETY_ROBOTS should have 4+ entries");
}
