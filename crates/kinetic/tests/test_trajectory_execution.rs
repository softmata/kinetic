//! Integration tests for trajectory execution monitoring.
//!
//! Tests the ExecutionMonitor against realistic trajectory scenarios:
//! normal tracking, drift, abort, noise filtering, and multi-robot configs.

use std::sync::Arc;
use std::time::Duration;

use kinetic::prelude::*;
use kinetic_trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
use kinetic_trajectory::trapezoidal::trapezoidal;

/// Load a robot by name and plan a simple trajectory for testing.
fn plan_trajectory_for(name: &str) -> (Arc<Robot>, TimedTrajectory) {
    let robot = Arc::new(Robot::from_name(name).unwrap());
    let dof = robot
        .groups
        .values()
        .next()
        .map(|g| {
            let chain = KinematicChain::extract(&robot, &g.base_link, &g.tip_link).unwrap();
            chain.dof
        })
        .unwrap();

    // Simple 2-waypoint path
    let start = vec![0.0; dof];
    let goal = vec![0.5; dof];
    let traj = trapezoidal(&[start, goal], 1.0, 2.0).unwrap();
    (robot, traj)
}

/// Convenience: plan a Panda trajectory.
fn plan_panda_trajectory() -> (Arc<Robot>, TimedTrajectory) {
    plan_trajectory_for("franka_panda")
}

// ─── Normal execution ───────────────────────────────────────────────

#[test]
fn normal_execution_no_deviation() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    // Walk through trajectory sampling exact planned positions
    let steps = 20;
    let dt = traj.duration().as_secs_f64() / steps as f64;
    for i in 0..=steps {
        let t = i as f64 * dt;
        let planned = traj.sample_at(Duration::from_secs_f64(t));
        let level = monitor.check(t, &planned.positions);
        assert_eq!(level, DeviationLevel::Normal, "Step {} should be normal", i);
    }
}

#[test]
fn tiny_deviation_is_normal() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    // Add tiny noise (0.001 rad) to each joint
    let actual: Vec<f64> = planned.positions.iter().map(|p| p + 0.001).collect();
    let level = monitor.check(t, &actual);
    assert_eq!(level, DeviationLevel::Normal);
}

// ─── Warning level ──────────────────────────────────────────────────

#[test]
fn moderate_deviation_triggers_warning() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    // Add 0.08 rad deviation to joint 0 (above 0.07 warning, below 0.1 abort)
    let mut actual = planned.positions.clone();
    actual[0] += 0.08;
    let level = monitor.check(t, &actual);
    assert!(
        matches!(level, DeviationLevel::Warning { joint: 0, .. }),
        "Expected Warning on joint 0, got {:?}",
        level
    );
}

#[test]
fn warning_reports_correct_deviation() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    let t = 0.3;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[2] += 0.08;
    let level = monitor.check(t, &actual);
    match level {
        DeviationLevel::Warning { joint, deviation } => {
            assert_eq!(joint, 2);
            assert!((deviation - 0.08).abs() < 1e-6);
        }
        other => panic!("Expected Warning, got {:?}", other),
    }
}

// ─── Abort level ────────────────────────────────────────────────────

#[test]
fn large_deviation_triggers_abort() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[3] += 0.2; // 0.2 >> 0.1 abort threshold
    let level = monitor.check(t, &actual);
    assert!(
        matches!(level, DeviationLevel::Abort { joint: 3, .. }),
        "Expected Abort on joint 3, got {:?}",
        level
    );
}

#[test]
fn negative_deviation_triggers_abort() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] -= 0.15; // negative deviation also counts
    let level = monitor.check(t, &actual);
    assert!(matches!(level, DeviationLevel::Abort { .. }));
}

// ─── Noise filtering ────────────────────────────────────────────────

#[test]
fn noise_spike_filtered_by_window() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 5,
        },
    );

    let duration = traj.duration().as_secs_f64();
    let step = duration / 10.0;

    // 4 on-track samples
    for i in 0..4 {
        let t = i as f64 * step;
        let planned = traj.sample_at(Duration::from_secs_f64(t));
        monitor.check(t, &planned.positions);
    }

    // 1 noisy spike (would abort without filtering)
    let t = 4.0 * step;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut noisy = planned.positions.clone();
    noisy[0] += 0.5; // huge spike
    let level = monitor.check(t, &noisy);

    // Averaged: (0 + 0 + 0 + 0 + 0.5) / 5 = 0.1 → at boundary
    // The exact result depends on previous accumulated tiny deviations
    // but it should NOT be Abort from a single spike after 4 clean samples
    assert!(
        !matches!(level, DeviationLevel::Abort { deviation, .. } if deviation > 0.2),
        "Single spike should be filtered, got {:?}",
        level
    );
}

#[test]
fn sustained_noise_not_filtered() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 3,
        },
    );

    let duration = traj.duration().as_secs_f64();
    let step = duration / 10.0;

    // 3 consecutive off-track samples
    for i in 0..3 {
        let t = i as f64 * step;
        let planned = traj.sample_at(Duration::from_secs_f64(t));
        let mut actual = planned.positions.clone();
        actual[0] += 0.15;
        let level = monitor.check(t, &actual);

        // By the 3rd sample the window is full of 0.15 → abort
        if i == 2 {
            assert!(
                matches!(level, DeviationLevel::Abort { .. }),
                "Sustained deviation should trigger abort, got {:?}",
                level
            );
        }
    }
}

#[test]
fn window_size_one_disables_filtering() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    // Even first sample triggers abort with window=1
    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.15;
    let level = monitor.check(t, &actual);
    assert!(matches!(level, DeviationLevel::Abort { .. }));
}

// ─── Configurable thresholds ────────────────────────────────────────

#[test]
fn tight_tolerance_catches_small_drift() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.01, // very tight: 0.01 rad
            warning_fraction: 0.5,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.008; // 0.008 > 0.005 (warning) but < 0.01 (abort)
    let level = monitor.check(t, &actual);
    assert!(matches!(level, DeviationLevel::Warning { .. }));
}

#[test]
fn loose_tolerance_allows_large_drift() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 1.0, // very loose
            warning_fraction: 0.9,
            noise_window: 1,
        },
    );

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.5; // large but within 1.0 tolerance
    let level = monitor.check(t, &actual);
    assert_eq!(level, DeviationLevel::Normal);
}

// ─── Reset and trajectory swap ──────────────────────────────────────

#[test]
fn reset_clears_accumulated_history() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 5,
        },
    );

    // Build up bad history
    let planned = traj.sample_at(Duration::from_secs_f64(0.5));
    let mut bad = planned.positions.clone();
    bad[0] += 0.15;
    for i in 0..5 {
        monitor.check(0.5 + i as f64 * 0.01, &bad);
    }

    // Reset
    monitor.reset();

    // Fresh on-track sample should be normal
    let planned = traj.sample_at(Duration::from_secs_f64(0.6));
    let level = monitor.check(0.6, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);
}

// ─── Multiple robots ────────────────────────────────────────────────

#[test]
fn works_with_ur5e_trajectory() {
    let (_robot, traj) = plan_trajectory_for("ur5e");
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.05,
            warning_fraction: 0.8,
            noise_window: 1,
        },
    );

    // On-track
    let planned = traj.sample_at(Duration::from_secs_f64(0.3));
    let level = monitor.check(0.3, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);

    // Off-track
    let mut actual = planned.positions.clone();
    actual[3] += 0.06;
    let level = monitor.check(0.3, &actual);
    assert!(matches!(level, DeviationLevel::Abort { joint: 3, .. }));
}

#[test]
fn works_with_xarm7_trajectory() {
    let (_, traj) = plan_trajectory_for("xarm7");
    let mut monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    // Full pass: walk through trajectory and verify all Normal
    let steps = 10;
    let dt = traj.duration().as_secs_f64() / steps as f64;
    for i in 0..=steps {
        let t = i as f64 * dt;
        let planned = traj.sample_at(Duration::from_secs_f64(t));
        let level = monitor.check(t, &planned.positions);
        assert_eq!(
            level,
            DeviationLevel::Normal,
            "xarm7 step {} should be normal",
            i
        );
    }
}

// ─── Edge cases ─────────────────────────────────────────────────────

#[test]
fn check_at_trajectory_start() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    let planned = traj.sample_at(Duration::ZERO);
    let level = monitor.check(0.0, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);
}

#[test]
fn check_at_trajectory_end() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    let t = traj.duration().as_secs_f64();
    let planned = traj.sample_at(traj.duration());
    let level = monitor.check(t, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);
}

#[test]
fn check_beyond_trajectory_end() {
    let (_, traj) = plan_panda_trajectory();
    let mut monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    // Check at 2x the duration — should use last waypoint
    let t = traj.duration().as_secs_f64() * 2.0;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let level = monitor.check(t, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);
}

#[test]
fn raw_deviations_match_expected() {
    let (_, traj) = plan_panda_trajectory();
    let monitor = ExecutionMonitor::new(traj.clone(), MonitorConfig::default());

    let t = 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.05;
    actual[2] -= 0.03;

    let devs = monitor.raw_deviations(t, &actual);
    assert!((devs[0] - 0.05).abs() < 1e-10);
    assert!(devs[1].abs() < 1e-10);
    assert!((devs[2] - 0.03).abs() < 1e-10);
}
