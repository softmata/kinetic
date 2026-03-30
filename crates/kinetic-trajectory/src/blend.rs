//! Trajectory blending.
//!
//! Smooth transition from one trajectory to another without stopping.
//! Uses parabolic blending at the connection point to create a smooth
//! velocity profile through the blend region.
//!
//! Important for continuous motion: pick -> move -> place without pauses.

use std::time::Duration;

use crate::trapezoidal::{TimedTrajectory, TimedWaypoint};

/// Blend two trajectories together with a smooth transition.
///
/// Creates a parabolic blend at the connection point so there's no velocity
/// or acceleration discontinuity. The blend region spans `blend_duration`
/// seconds centered on the connection point.
///
/// `traj1`: first trajectory (leading up to the blend).
/// `traj2`: second trajectory (continuing after the blend).
/// `blend_duration`: duration of the blend region in seconds.
///
/// Requirements:
/// - Both trajectories must have the same DOF.
/// - `blend_duration` must be positive and not exceed either trajectory's duration.
pub fn blend(
    traj1: &TimedTrajectory,
    traj2: &TimedTrajectory,
    blend_duration: f64,
) -> Result<TimedTrajectory, String> {
    if traj1.waypoints.is_empty() && traj2.waypoints.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    if traj1.waypoints.is_empty() {
        return Ok(traj2.clone());
    }
    if traj2.waypoints.is_empty() {
        return Ok(traj1.clone());
    }

    if traj1.dof != traj2.dof {
        return Err(format!(
            "DOF mismatch: traj1={}, traj2={}",
            traj1.dof, traj2.dof
        ));
    }

    if blend_duration <= 0.0 {
        return Err("Blend duration must be positive".to_string());
    }

    let dof = traj1.dof;
    let t1_dur = traj1.duration.as_secs_f64();
    let t2_dur = traj2.duration.as_secs_f64();

    let half_blend = blend_duration / 2.0;

    if half_blend > t1_dur {
        return Err(format!(
            "Blend half-duration ({}) exceeds traj1 duration ({})",
            half_blend, t1_dur
        ));
    }
    if half_blend > t2_dur {
        return Err(format!(
            "Blend half-duration ({}) exceeds traj2 duration ({})",
            half_blend, t2_dur
        ));
    }

    let mut waypoints = Vec::new();

    // Phase 1: Copy traj1 up to the blend start
    let blend_start = t1_dur - half_blend;
    for wp in &traj1.waypoints {
        if wp.time <= blend_start {
            waypoints.push(wp.clone());
        }
    }

    // Get the state at the start and end of the blend region
    // These reference points are used conceptually in the blend algorithm
    // but the actual sampling happens per-point inside the blend loop.

    // Phase 2: Generate blend region using parabolic interpolation
    let blend_samples = 10;
    let time_offset = blend_start;

    for k in 1..blend_samples {
        let alpha = k as f64 / blend_samples as f64;
        let t_local = alpha * blend_duration;
        let t_global = time_offset + t_local;

        let mut positions = vec![0.0; dof];
        let mut velocities = vec![0.0; dof];
        let mut accelerations = vec![0.0; dof];

        for j in 0..dof {
            // Parabolic blend between traj1 end-state and traj2 start-state
            // We interpolate between the two trajectory samples at this point

            // Sample from traj1 at (blend_start + t_local) clamped to t1_dur
            let t1_sample_time = (blend_start + t_local).min(t1_dur);
            let t1_sample = traj1.sample_at(Duration::from_secs_f64(t1_sample_time));

            // Sample from traj2 at t_local clamped to t2_dur
            let t2_sample_time = t_local.min(t2_dur);
            let t2_sample = traj2.sample_at(Duration::from_secs_f64(t2_sample_time));

            // Smooth blending weight: use quintic (5th order) Hermite interpolant
            // s(alpha) = 6*alpha^5 - 15*alpha^4 + 10*alpha^3
            // This gives zero velocity and acceleration at alpha=0 and alpha=1
            let s = 6.0 * alpha.powi(5) - 15.0 * alpha.powi(4) + 10.0 * alpha.powi(3);
            let s_dot = (30.0 * alpha.powi(4) - 60.0 * alpha.powi(3) + 30.0 * alpha.powi(2))
                / blend_duration;
            let s_ddot = (120.0 * alpha.powi(3) - 180.0 * alpha.powi(2) + 60.0 * alpha)
                / (blend_duration * blend_duration);

            // Blend positions
            let p1 = t1_sample.positions[j];
            let p2 = t2_sample.positions[j];
            positions[j] = (1.0 - s) * p1 + s * p2;

            // Blend velocities
            let v1 = t1_sample.velocities[j];
            let v2 = t2_sample.velocities[j];
            velocities[j] = (1.0 - s) * v1 + s * v2 + s_dot * (p2 - p1);

            // Blend accelerations
            let a1 = t1_sample.accelerations[j];
            let a2 = t2_sample.accelerations[j];
            accelerations[j] =
                (1.0 - s) * a1 + s * a2 + 2.0 * s_dot * (v2 - v1) + s_ddot * (p2 - p1);
        }

        waypoints.push(TimedWaypoint {
            time: t_global,
            positions,
            velocities,
            accelerations,
        });
    }

    // Phase 3: Copy traj2 from blend end onwards, with time offset
    let time_offset_t2 = blend_start + blend_duration;
    for wp in &traj2.waypoints {
        if wp.time >= half_blend {
            waypoints.push(TimedWaypoint {
                time: wp.time - half_blend + time_offset_t2,
                positions: wp.positions.clone(),
                velocities: wp.velocities.clone(),
                accelerations: wp.accelerations.clone(),
            });
        }
    }

    let total_duration = t1_dur - half_blend + blend_duration + t2_dur - half_blend;

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(total_duration),
        dof,
        waypoints,
    })
}

/// Blend multiple trajectories in sequence.
///
/// Given a list of trajectories, blends each consecutive pair with the
/// specified blend duration, producing one continuous trajectory.
pub fn blend_sequence(
    trajectories: &[TimedTrajectory],
    blend_duration: f64,
) -> Result<TimedTrajectory, String> {
    if trajectories.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    if trajectories.len() == 1 {
        return Ok(trajectories[0].clone());
    }

    let mut result = trajectories[0].clone();
    for traj in trajectories.iter().skip(1) {
        result = blend(&result, traj, blend_duration)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple linear trajectory.
    fn linear_trajectory(start: Vec<f64>, end: Vec<f64>, duration: f64) -> TimedTrajectory {
        let dof = start.len();
        let n = 10;

        let waypoints: Vec<TimedWaypoint> = (0..=n)
            .map(|k| {
                let alpha = k as f64 / n as f64;
                let t = alpha * duration;
                let positions: Vec<f64> = start
                    .iter()
                    .zip(&end)
                    .map(|(&s, &e)| s + alpha * (e - s))
                    .collect();
                let velocities: Vec<f64> = start
                    .iter()
                    .zip(&end)
                    .map(|(&s, &e)| (e - s) / duration)
                    .collect();

                TimedWaypoint {
                    time: t,
                    positions,
                    velocities,
                    accelerations: vec![0.0; dof],
                }
            })
            .collect();

        TimedTrajectory {
            duration: Duration::from_secs_f64(duration),
            dof,
            waypoints,
        }
    }

    #[test]
    fn blend_basic() {
        let t1 = linear_trajectory(vec![0.0, 0.0], vec![1.0, 1.0], 2.0);
        let t2 = linear_trajectory(vec![1.0, 1.0], vec![2.0, 0.0], 2.0);

        let result = blend(&t1, &t2, 0.5).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(!result.waypoints.is_empty());
    }

    #[test]
    fn blend_empty_traj1() {
        let empty = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        };
        let t2 = linear_trajectory(vec![0.0], vec![1.0], 1.0);
        let result = blend(&empty, &t2, 0.1).unwrap();
        assert_eq!(result.dof, t2.dof);
    }

    #[test]
    fn blend_empty_traj2() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 1.0);
        let empty = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        };
        let result = blend(&t1, &empty, 0.1).unwrap();
        assert_eq!(result.dof, t1.dof);
    }

    #[test]
    fn blend_dof_mismatch() {
        let t1 = linear_trajectory(vec![0.0, 0.0], vec![1.0, 1.0], 1.0);
        let t2 = linear_trajectory(vec![0.0], vec![1.0], 1.0);
        assert!(blend(&t1, &t2, 0.1).is_err());
    }

    #[test]
    fn blend_invalid_duration() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 1.0);
        let t2 = linear_trajectory(vec![1.0], vec![2.0], 1.0);
        assert!(blend(&t1, &t2, 0.0).is_err());
        assert!(blend(&t1, &t2, -0.5).is_err());
    }

    #[test]
    fn blend_too_long() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 0.5);
        let t2 = linear_trajectory(vec![1.0], vec![2.0], 0.5);
        // Blend duration 2.0 > trajectory durations
        assert!(blend(&t1, &t2, 2.0).is_err());
    }

    #[test]
    fn blend_monotonic_time() {
        let t1 = linear_trajectory(vec![0.0, 0.0], vec![1.0, 1.0], 2.0);
        let t2 = linear_trajectory(vec![1.0, 1.0], vec![2.0, 0.0], 2.0);

        let result = blend(&t1, &t2, 0.5).unwrap();

        for i in 1..result.waypoints.len() {
            assert!(
                result.waypoints[i].time >= result.waypoints[i - 1].time - 1e-10,
                "Time not monotonic at {}: {} < {}",
                i,
                result.waypoints[i].time,
                result.waypoints[i - 1].time
            );
        }
    }

    #[test]
    fn blend_smooth_transition() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 2.0);
        let t2 = linear_trajectory(vec![1.0], vec![2.0], 2.0);

        let result = blend(&t1, &t2, 0.5).unwrap();

        // Check that velocities in the blend region don't jump too much
        for i in 1..result.waypoints.len() {
            let vel_diff =
                (result.waypoints[i].velocities[0] - result.waypoints[i - 1].velocities[0]).abs();
            let dt = result.waypoints[i].time - result.waypoints[i - 1].time;
            if dt > 1e-10 {
                let jerk = vel_diff / dt;
                // Allow some jerk but it should be bounded
                assert!(
                    jerk < 100.0,
                    "Excessive jerk at {}: {} (vel_diff={}, dt={})",
                    i,
                    jerk,
                    vel_diff,
                    dt
                );
            }
        }
    }

    #[test]
    fn blend_sequence_basic() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 2.0);
        let t2 = linear_trajectory(vec![1.0], vec![2.0], 2.0);
        let t3 = linear_trajectory(vec![2.0], vec![3.0], 2.0);

        let result = blend_sequence(&[t1, t2, t3], 0.3).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(!result.waypoints.is_empty());
    }

    #[test]
    fn blend_sequence_single() {
        let t1 = linear_trajectory(vec![0.0], vec![1.0], 1.0);
        let result = blend_sequence(std::slice::from_ref(&t1), 0.1).unwrap();
        assert_eq!(result.waypoints.len(), t1.waypoints.len());
    }

    #[test]
    fn blend_accelerations_nonempty() {
        let t1 = linear_trajectory(vec![0.0, 0.0], vec![1.0, 1.0], 2.0);
        let t2 = linear_trajectory(vec![1.0, 1.0], vec![2.0, 0.0], 2.0);
        let result = blend(&t1, &t2, 0.5).unwrap();
        result.validate().unwrap();

        // Blend region should produce non-zero accelerations (quintic blend)
        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Blend should produce non-zero accelerations in blend region"
        );
    }

    #[test]
    fn blend_sequence_empty() {
        let result = blend_sequence(&[], 0.1).unwrap();
        assert!(result.is_empty());
    }
}
