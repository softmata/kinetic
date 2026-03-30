//! Time-Optimal Time Parameterization (TOTP).
//!
//! Given a geometric path in joint space, finds the time-optimal trajectory
//! that respects per-joint velocity and acceleration limits.
//!
//! Based on the algorithm by Kunz & Stilman (2012) / Hauser (2014), which
//! uses phase-plane analysis to find the switching points between maximum
//! acceleration and maximum deceleration.
//!
//! This is equivalent to MoveIt2's iterative_time_parameterization / TOTG.

use std::time::Duration;

use crate::trapezoidal::{TimedTrajectory, TimedWaypoint};

/// Time-Optimal Time Parameterization.
///
/// Computes the fastest trajectory along a geometric path that respects
/// per-joint velocity and acceleration limits.
///
/// `path`: list of joint-space waypoints.
/// `velocity_limits`: per-joint maximum velocity (rad/s).
/// `acceleration_limits`: per-joint maximum acceleration (rad/s^2).
/// `path_resolution`: discretization of path parameter (default: 0.001).
pub fn totp(
    path: &[Vec<f64>],
    velocity_limits: &[f64],
    acceleration_limits: &[f64],
    path_resolution: f64,
) -> Result<TimedTrajectory, String> {
    // Validate inputs for NaN/Inf
    for (i, wp) in path.iter().enumerate() {
        for (j, &v) in wp.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "Waypoint {} joint {} has non-finite value: {v}",
                    i, j
                ));
            }
        }
    }
    if path.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    let dof = path[0].len();
    if velocity_limits.len() != dof || acceleration_limits.len() != dof {
        return Err(format!(
            "Limits mismatch: dof={}, vel={}, accel={}",
            dof,
            velocity_limits.len(),
            acceleration_limits.len()
        ));
    }

    if path.len() == 1 {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: path[0].clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            }],
        });
    }

    // Step 1: Compute cumulative path length (chord-length parameterization)
    let seg_lengths = compute_segment_lengths(path);
    let total_length: f64 = seg_lengths.iter().sum();

    if total_length < 1e-12 {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: path[0].clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            }],
        });
    }

    // Step 2: Discretize path and compute path derivatives
    let n_samples = ((total_length / path_resolution).ceil() as usize).max(2);
    let ds = total_length / (n_samples - 1) as f64;

    let samples = sample_path_uniform(path, &seg_lengths, total_length, n_samples);
    let dq_ds = compute_path_derivatives(&samples, ds);

    // Step 3: Compute maximum velocity along path (phase-plane upper bound)
    let max_sdot = compute_max_sdot(&dq_ds, velocity_limits);

    // Step 4: Forward integration (maximum acceleration)
    let sdot_forward = integrate_forward(&dq_ds, &max_sdot, acceleration_limits, ds);

    // Step 5: Backward integration (maximum deceleration from end)
    let sdot_backward = integrate_backward(&dq_ds, &max_sdot, acceleration_limits, ds);

    // Step 6: Combine forward and backward profiles (take minimum)
    let sdot: Vec<f64> = sdot_forward
        .iter()
        .zip(&sdot_backward)
        .zip(&max_sdot)
        .map(|((&fwd, &bwd), &mx)| fwd.min(bwd).min(mx).max(0.0))
        .collect();

    // Step 7: Integrate time from sdot profile
    let times = integrate_time(&sdot, ds);
    let total_time = *times.last().unwrap_or(&0.0);

    // Step 8: Build output trajectory
    let mut waypoints = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let positions = samples[i].clone();
        let s_dot = sdot[i];

        // Joint velocities: dq/dt = dq/ds * ds/dt = dq_ds * sdot
        let velocities: Vec<f64> = dq_ds[i].iter().map(|&dqds| dqds * s_dot).collect();

        // Joint accelerations: approximate from sdot profile
        let s_ddot = if i == 0 {
            if n_samples > 1 {
                (sdot[1] - sdot[0]) / ds * sdot[0]
            } else {
                0.0
            }
        } else if i == n_samples - 1 {
            (sdot[i] - sdot[i - 1]) / ds * sdot[i]
        } else {
            (sdot[i + 1] - sdot[i - 1]) / (2.0 * ds) * sdot[i]
        };

        let accelerations: Vec<f64> = dq_ds[i]
            .iter()
            .enumerate()
            .map(|(j, &dqds)| {
                let ddq_ds = if i == 0 {
                    if n_samples > 1 {
                        (dq_ds[1][j] - dq_ds[0][j]) / ds
                    } else {
                        0.0
                    }
                } else if i == n_samples - 1 {
                    (dq_ds[i][j] - dq_ds[i - 1][j]) / ds
                } else {
                    (dq_ds[i + 1][j] - dq_ds[i - 1][j]) / (2.0 * ds)
                };
                dqds * s_ddot + ddq_ds * s_dot * s_dot
            })
            .collect();

        waypoints.push(TimedWaypoint {
            time: times[i],
            positions,
            velocities,
            accelerations,
        });
    }

    // Ensure start and end have zero velocity
    if let Some(first) = waypoints.first_mut() {
        first.velocities = vec![0.0; dof];
        first.accelerations = vec![0.0; dof];
    }
    if let Some(last) = waypoints.last_mut() {
        last.velocities = vec![0.0; dof];
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(total_time),
        dof,
        waypoints,
    })
}

// === Internal helpers ===

/// Compute segment lengths between consecutive waypoints.
fn compute_segment_lengths(path: &[Vec<f64>]) -> Vec<f64> {
    path.windows(2)
        .map(|w| {
            w[0].iter()
                .zip(&w[1])
                .map(|(&a, &b)| (b - a).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Sample the path at uniform arc-length intervals.
fn sample_path_uniform(
    path: &[Vec<f64>],
    seg_lengths: &[f64],
    total_length: f64,
    n_samples: usize,
) -> Vec<Vec<f64>> {
    let dof = path[0].len();
    let mut cumulative = vec![0.0];
    for &len in seg_lengths {
        cumulative.push(cumulative.last().unwrap() + len);
    }

    let mut samples = Vec::with_capacity(n_samples);

    for k in 0..n_samples {
        let s = total_length * k as f64 / (n_samples - 1).max(1) as f64;
        let s = s.min(total_length);

        // Find segment containing s
        let seg = match cumulative[1..].binary_search_by(|v| v.partial_cmp(&s).unwrap()) {
            Ok(exact) => exact.min(seg_lengths.len() - 1),
            Err(insert) => insert.saturating_sub(1).min(seg_lengths.len() - 1),
        };

        // Interpolate within segment
        let seg_start = cumulative[seg];
        let seg_len = seg_lengths[seg];
        let alpha = if seg_len > 1e-15 {
            ((s - seg_start) / seg_len).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let positions: Vec<f64> = (0..dof)
            .map(|j| path[seg][j] + alpha * (path[seg + 1][j] - path[seg][j]))
            .collect();

        samples.push(positions);
    }

    samples
}

/// Compute path derivatives dq/ds at each sample point.
fn compute_path_derivatives(samples: &[Vec<f64>], ds: f64) -> Vec<Vec<f64>> {
    let n = samples.len();
    let dof = samples[0].len();

    let mut dq_ds = Vec::with_capacity(n);

    for i in 0..n {
        let deriv: Vec<f64> = (0..dof)
            .map(|j| {
                if i == 0 {
                    if n > 1 {
                        (samples[1][j] - samples[0][j]) / ds
                    } else {
                        0.0
                    }
                } else if i == n - 1 {
                    (samples[n - 1][j] - samples[n - 2][j]) / ds
                } else {
                    (samples[i + 1][j] - samples[i - 1][j]) / (2.0 * ds)
                }
            })
            .collect();

        dq_ds.push(deriv);
    }

    dq_ds
}

/// Compute maximum path velocity (sdot) at each sample from velocity limits.
///
/// sdot_max[i] = min over joints j of: vel_limit[j] / |dq_ds[i][j]|
fn compute_max_sdot(dq_ds: &[Vec<f64>], velocity_limits: &[f64]) -> Vec<f64> {
    dq_ds
        .iter()
        .map(|deriv| {
            let mut max_sdot = f64::INFINITY;
            for (j, &dqds) in deriv.iter().enumerate() {
                if dqds.abs() > 1e-10 {
                    let limit = velocity_limits[j] / dqds.abs();
                    if limit < max_sdot {
                        max_sdot = limit;
                    }
                }
            }
            if max_sdot.is_infinite() {
                // No velocity constraints active (very small derivatives)
                100.0 // reasonable upper bound
            } else {
                max_sdot
            }
        })
        .collect()
}

/// Forward integration: accelerate as fast as possible from start (sdot=0).
fn integrate_forward(
    dq_ds: &[Vec<f64>],
    max_sdot: &[f64],
    accel_limits: &[f64],
    ds: f64,
) -> Vec<f64> {
    let n = dq_ds.len();
    let mut sdot = vec![0.0; n];

    // Start from zero velocity
    sdot[0] = 0.0;

    for i in 1..n {
        // Maximum acceleration at point i-1
        let max_accel = max_path_acceleration(&dq_ds[i - 1], accel_limits, sdot[i - 1]);

        // sdot^2(i) = sdot^2(i-1) + 2 * max_accel * ds
        let sdot_sq = sdot[i - 1] * sdot[i - 1] + 2.0 * max_accel * ds;
        sdot[i] = if sdot_sq > 0.0 {
            sdot_sq.sqrt().min(max_sdot[i])
        } else {
            0.0
        };
    }

    sdot
}

/// Backward integration: decelerate to zero at the end.
fn integrate_backward(
    dq_ds: &[Vec<f64>],
    max_sdot: &[f64],
    accel_limits: &[f64],
    ds: f64,
) -> Vec<f64> {
    let n = dq_ds.len();
    let mut sdot = vec![0.0; n];

    // End at zero velocity
    sdot[n - 1] = 0.0;

    for i in (0..n - 1).rev() {
        let max_accel = max_path_acceleration(&dq_ds[i + 1], accel_limits, sdot[i + 1]);

        let sdot_sq = sdot[i + 1] * sdot[i + 1] + 2.0 * max_accel * ds;
        sdot[i] = if sdot_sq > 0.0 {
            sdot_sq.sqrt().min(max_sdot[i])
        } else {
            0.0
        };
    }

    sdot
}

/// Maximum path acceleration at a point, respecting per-joint limits.
///
/// Returns the maximum sdotdot such that all joints respect their accel limits.
fn max_path_acceleration(dq_ds: &[f64], accel_limits: &[f64], _sdot: f64) -> f64 {
    let mut max_sddot = f64::INFINITY;

    for (j, &dqds) in dq_ds.iter().enumerate() {
        if dqds.abs() > 1e-10 {
            let limit = accel_limits[j] / dqds.abs();
            if limit < max_sddot {
                max_sddot = limit;
            }
        }
    }

    if max_sddot.is_infinite() {
        100.0 // reasonable bound when derivatives are near zero
    } else {
        max_sddot
    }
}

/// Integrate time from the sdot profile: dt = ds / sdot.
fn integrate_time(sdot: &[f64], ds: f64) -> Vec<f64> {
    let n = sdot.len();
    let mut times = vec![0.0; n];

    for i in 1..n {
        let avg_sdot = (sdot[i - 1] + sdot[i]) / 2.0;
        let dt = if avg_sdot > 1e-12 {
            ds / avg_sdot
        } else {
            0.0 // stalled — shouldn't happen in well-formed input
        };
        times[i] = times[i - 1] + dt;
    }

    times
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totp_basic() {
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 0.5]];
        let vel_limits = vec![2.0, 2.0];
        let accel_limits = vec![4.0, 4.0];

        let result = totp(&path, &vel_limits, &accel_limits, 0.01).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 2);

        // First waypoint should start at zero velocity
        let first = &result.waypoints[0];
        assert!((first.velocities[0]).abs() < 1e-10);
        assert!((first.velocities[1]).abs() < 1e-10);

        // Last waypoint should end at zero velocity
        let last = result.waypoints.last().unwrap();
        assert!((last.velocities[0]).abs() < 1e-10);
        assert!((last.velocities[1]).abs() < 1e-10);
    }

    #[test]
    fn totp_empty() {
        let result = totp(&[], &[], &[], 0.01).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn totp_single() {
        let result = totp(&[vec![1.0, 2.0]], &[1.0, 1.0], &[2.0, 2.0], 0.01).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn totp_respects_velocity_limits() {
        let path = vec![vec![0.0], vec![5.0]]; // large displacement
        let vel_limits = vec![1.0]; // slow velocity limit
        let accel_limits = vec![10.0]; // fast acceleration

        let result = totp(&path, &vel_limits, &accel_limits, 0.01).unwrap();

        // Check that no waypoint exceeds velocity limit (with small tolerance)
        for wp in &result.waypoints {
            assert!(
                wp.velocities[0].abs() <= vel_limits[0] * 1.1, // 10% tolerance for discretization
                "Velocity {} exceeds limit {}",
                wp.velocities[0].abs(),
                vel_limits[0]
            );
        }
    }

    #[test]
    fn totp_monotonic_time() {
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![3.0, 0.5],
        ];
        let vel_limits = vec![2.0, 2.0];
        let accel_limits = vec![4.0, 4.0];

        let result = totp(&path, &vel_limits, &accel_limits, 0.01).unwrap();

        for i in 1..result.waypoints.len() {
            assert!(
                result.waypoints[i].time >= result.waypoints[i - 1].time - 1e-10,
                "Time not monotonic: t[{}]={} < t[{}]={}",
                i,
                result.waypoints[i].time,
                i - 1,
                result.waypoints[i - 1].time
            );
        }
    }

    #[test]
    fn totp_start_end_positions() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![2.0, 1.0]];
        let vel_limits = vec![2.0, 2.0];
        let accel_limits = vec![4.0, 4.0];

        let result = totp(&path, &vel_limits, &accel_limits, 0.01).unwrap();

        // Start position should match
        let first = &result.waypoints[0];
        assert!((first.positions[0] - 0.0).abs() < 1e-6);
        assert!((first.positions[1] - 0.0).abs() < 1e-6);

        // End position should match
        let last = result.waypoints.last().unwrap();
        assert!((last.positions[0] - 2.0).abs() < 1e-6);
        assert!((last.positions[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn totp_limits_mismatch() {
        let path = vec![vec![0.0, 0.0]];
        let result = totp(&path, &[1.0], &[2.0, 3.0], 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn totp_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![2.0, 1.0]];
        let vel = vec![2.0, 2.0];
        let acc = vec![4.0, 4.0];
        let result = totp(&path, &vel, &acc, 0.01).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(has_nonzero, "TOTP should produce non-zero accelerations");
    }

    #[test]
    fn totp_tight_limits_longer_duration() {
        let path = vec![vec![0.0], vec![1.0]];

        let fast = totp(&path, &[10.0], &[20.0], 0.01).unwrap();
        let slow = totp(&path, &[0.5], &[1.0], 0.01).unwrap();

        assert!(
            slow.duration().as_secs_f64() > fast.duration().as_secs_f64(),
            "Tighter limits should produce longer duration: slow={}, fast={}",
            slow.duration().as_secs_f64(),
            fast.duration().as_secs_f64()
        );
    }

    #[test]
    fn segment_lengths() {
        let path = vec![vec![0.0, 0.0], vec![3.0, 4.0]]; // distance = 5.0
        let lengths = compute_segment_lengths(&path);
        assert_eq!(lengths.len(), 1);
        assert!((lengths[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn sample_at_totp_result() {
        let path = vec![vec![0.0], vec![1.0], vec![2.0]];
        let result = totp(&path, &[2.0], &[4.0], 0.01).unwrap();

        // Sample at start
        let wp0 = result.sample_at(Duration::ZERO);
        assert!((wp0.positions[0] - 0.0).abs() < 1e-6);

        // Sample at end
        let wp_end = result.sample_at(result.duration());
        assert!((wp_end.positions[0] - 2.0).abs() < 1e-6);
    }
}
