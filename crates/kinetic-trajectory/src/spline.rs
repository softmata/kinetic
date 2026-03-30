//! Cubic spline time parameterization.
//!
//! Fits cubic splines through waypoints to produce a C2-continuous trajectory.
//! Time is distributed proportionally to segment arc lengths, then cubic
//! polynomials are fitted per joint using natural or clamped boundary conditions.

use std::time::Duration;

use crate::trapezoidal::{TimedTrajectory, TimedWaypoint};

/// Cubic spline time parameterization.
///
/// Fits a natural cubic spline through the waypoints per joint, with time
/// distributed proportional to segment arc-length.
///
/// `path`: list of joint-space waypoints.
/// `total_duration`: optional total trajectory duration. If `None`, auto-computed
///   from velocity limits (defaults to a reasonable duration based on path length).
/// `max_velocity`: optional per-joint velocity limits for auto-duration computation.
pub fn cubic_spline_time(
    path: &[Vec<f64>],
    total_duration: Option<f64>,
    max_velocity: Option<&[f64]>,
) -> Result<TimedTrajectory, String> {
    if path.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    let dof = path[0].len();

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

    // Compute segment arc-lengths
    let seg_lengths: Vec<f64> = path
        .windows(2)
        .map(|w| {
            w[0].iter()
                .zip(&w[1])
                .map(|(&a, &b)| (b - a).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let total_arc = seg_lengths.iter().sum::<f64>();

    // Compute total duration
    let total_t = if let Some(t) = total_duration {
        if t <= 0.0 {
            return Err("Total duration must be positive".to_string());
        }
        t
    } else if let Some(vel_limits) = max_velocity {
        // Auto-compute: estimate duration from path length and velocity limits
        if vel_limits.len() != dof {
            return Err(format!(
                "Velocity limits length mismatch: dof={}, limits={}",
                dof,
                vel_limits.len()
            ));
        }
        // Use max displacement per joint / velocity limit, take the maximum
        let mut max_t = 0.0_f64;
        for seg in 0..path.len() - 1 {
            for j in 0..dof {
                let d = (path[seg + 1][j] - path[seg][j]).abs();
                if vel_limits[j] > 1e-12 {
                    max_t = max_t.max(d / vel_limits[j]);
                }
            }
        }
        // Add factor for acceleration/deceleration phases
        max_t * 1.5
    } else {
        // Heuristic: 1 second per unit of arc length
        total_arc.max(0.5)
    };

    // Compute knot times proportional to arc-length
    let mut knot_times = vec![0.0];
    for &seg_len in &seg_lengths {
        let dt = if total_arc > 1e-12 {
            total_t * seg_len / total_arc
        } else {
            total_t / (path.len() - 1) as f64
        };
        knot_times.push(knot_times.last().unwrap() + dt);
    }

    let n = path.len(); // number of knots

    // Fit natural cubic spline per joint
    // For each joint, we solve for the second derivatives (moments) M[i]
    // at each knot, then compute cubic coefficients.
    let mut all_moments: Vec<Vec<f64>> = Vec::with_capacity(dof);

    for j in 0..dof {
        let values: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
        let moments = natural_spline_moments(&knot_times, &values);
        all_moments.push(moments);
    }

    // Generate output waypoints by sampling the spline
    // Include all original knot points plus some intermediate samples
    let samples_per_segment = 4;
    let total_samples = (n - 1) * samples_per_segment + 1;

    let mut waypoints = Vec::with_capacity(total_samples);

    for seg in 0..n - 1 {
        let t0 = knot_times[seg];
        let t1 = knot_times[seg + 1];
        let dt = t1 - t0;

        let num_samples = if seg == n - 2 {
            samples_per_segment + 1 // include last point
        } else {
            samples_per_segment
        };

        for k in 0..num_samples {
            let alpha = k as f64 / samples_per_segment as f64;
            let t = t0 + alpha * dt;

            let mut positions = vec![0.0; dof];
            let mut velocities = vec![0.0; dof];
            let mut accelerations = vec![0.0; dof];

            for jj in 0..dof {
                let (p, v, a) = evaluate_cubic_segment(
                    t,
                    knot_times[seg],
                    knot_times[seg + 1],
                    path[seg][jj],
                    path[seg + 1][jj],
                    all_moments[jj][seg],
                    all_moments[jj][seg + 1],
                );
                positions[jj] = p;
                velocities[jj] = v;
                accelerations[jj] = a;
            }

            waypoints.push(TimedWaypoint {
                time: t,
                positions,
                velocities,
                accelerations,
            });
        }
    }

    // Enforce zero velocity at start and end for stop-to-stop motion
    if let Some(first) = waypoints.first_mut() {
        first.velocities = vec![0.0; dof];
    }
    if let Some(last) = waypoints.last_mut() {
        last.velocities = vec![0.0; dof];
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(total_t),
        dof,
        waypoints,
    })
}

/// Cubic spline with clamped boundary conditions (specified start/end velocities).
pub fn cubic_spline_time_clamped(
    path: &[Vec<f64>],
    total_duration: f64,
    start_velocities: &[f64],
    end_velocities: &[f64],
) -> Result<TimedTrajectory, String> {
    if path.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    let dof = path[0].len();
    if start_velocities.len() != dof || end_velocities.len() != dof {
        return Err(format!(
            "Velocity length mismatch: dof={}, start={}, end={}",
            dof,
            start_velocities.len(),
            end_velocities.len()
        ));
    }

    if total_duration <= 0.0 {
        return Err("Total duration must be positive".to_string());
    }

    if path.len() == 1 {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: path[0].clone(),
                velocities: start_velocities.to_vec(),
                accelerations: vec![0.0; dof],
            }],
        });
    }

    // Compute knot times (uniform for simplicity in clamped case)
    let n = path.len();
    let dt = total_duration / (n - 1) as f64;
    let knot_times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();

    // Fit clamped cubic spline per joint
    let mut all_moments: Vec<Vec<f64>> = Vec::with_capacity(dof);

    for j in 0..dof {
        let values: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
        let moments =
            clamped_spline_moments(&knot_times, &values, start_velocities[j], end_velocities[j]);
        all_moments.push(moments);
    }

    // Generate output waypoints
    let samples_per_segment = 4;
    let total_samples = (n - 1) * samples_per_segment + 1;
    let mut waypoints = Vec::with_capacity(total_samples);

    for seg in 0..n - 1 {
        let t0 = knot_times[seg];
        let t1 = knot_times[seg + 1];
        let seg_dt = t1 - t0;

        let num_samples = if seg == n - 2 {
            samples_per_segment + 1
        } else {
            samples_per_segment
        };

        for k in 0..num_samples {
            let alpha = k as f64 / samples_per_segment as f64;
            let t = t0 + alpha * seg_dt;

            let mut positions = vec![0.0; dof];
            let mut velocities = vec![0.0; dof];
            let mut accelerations = vec![0.0; dof];

            for jj in 0..dof {
                let (p, v, a) = evaluate_cubic_segment(
                    t,
                    knot_times[seg],
                    knot_times[seg + 1],
                    path[seg][jj],
                    path[seg + 1][jj],
                    all_moments[jj][seg],
                    all_moments[jj][seg + 1],
                );
                positions[jj] = p;
                velocities[jj] = v;
                accelerations[jj] = a;
            }

            waypoints.push(TimedWaypoint {
                time: t,
                positions,
                velocities,
                accelerations,
            });
        }
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(total_duration),
        dof,
        waypoints,
    })
}

// === Internal helpers ===

/// Solve for natural cubic spline second derivatives (moments) using Thomas algorithm.
///
/// Natural boundary: M[0] = M[n-1] = 0.
fn natural_spline_moments(times: &[f64], values: &[f64]) -> Vec<f64> {
    let n = times.len();
    if n <= 2 {
        return vec![0.0; n];
    }

    // h[i] = t[i+1] - t[i]
    let h: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();

    // Set up tridiagonal system for M[1..n-2]
    let m = n - 2;
    let mut a = vec![0.0; m]; // sub-diagonal
    let mut b = vec![0.0; m]; // main diagonal
    let mut c = vec![0.0; m]; // super-diagonal
    let mut d = vec![0.0; m]; // RHS

    for i in 0..m {
        let idx = i + 1; // knot index (1-based in interior)
        if i > 0 {
            a[i] = h[idx - 1];
        }
        b[i] = 2.0 * (h[idx - 1] + h[idx]);
        if i < m - 1 {
            c[i] = h[idx];
        }
        d[i] = 6.0
            * ((values[idx + 1] - values[idx]) / h[idx]
                - (values[idx] - values[idx - 1]) / h[idx - 1]);
    }

    // Thomas algorithm forward sweep
    for i in 1..m {
        if b[i - 1].abs() < 1e-15 {
            continue;
        }
        let factor = a[i] / b[i - 1];
        b[i] -= factor * c[i - 1];
        d[i] -= factor * d[i - 1];
    }

    // Back substitution
    let mut moments_inner = vec![0.0; m];
    if b[m - 1].abs() > 1e-15 {
        moments_inner[m - 1] = d[m - 1] / b[m - 1];
    }
    for i in (0..m - 1).rev() {
        if b[i].abs() > 1e-15 {
            moments_inner[i] = (d[i] - c[i] * moments_inner[i + 1]) / b[i];
        }
    }

    // Full moments: M[0] = 0, M[1..n-2] = inner, M[n-1] = 0
    let mut moments = vec![0.0; n];
    moments[1..(m + 1)].copy_from_slice(&moments_inner[..m]);

    moments
}

/// Solve for clamped cubic spline second derivatives.
///
/// Clamped boundary: specified first derivatives at endpoints.
fn clamped_spline_moments(times: &[f64], values: &[f64], start_vel: f64, end_vel: f64) -> Vec<f64> {
    let n = times.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    if n == 2 {
        // With only 2 points and clamped BCs, we can compute directly
        let h = times[1] - times[0];
        if h.abs() < 1e-15 {
            return vec![0.0; 2];
        }
        let slope = (values[1] - values[0]) / h;
        let m0 = (6.0 * (slope - start_vel)) / h;
        let m1 = (6.0 * (end_vel - slope)) / h;
        return vec![m0, m1];
    }

    let h: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();

    // Full tridiagonal system (n equations, including boundary rows)
    let mut a_diag = vec![0.0; n]; // sub-diagonal
    let mut b_diag = vec![0.0; n]; // main diagonal
    let mut c_diag = vec![0.0; n]; // super-diagonal
    let mut d_rhs = vec![0.0; n]; // RHS

    // Row 0 (clamped boundary): 2*h[0]*M[0] + h[0]*M[1] = 6*((y1-y0)/h[0] - start_vel)
    b_diag[0] = 2.0 * h[0];
    c_diag[0] = h[0];
    d_rhs[0] = 6.0 * ((values[1] - values[0]) / h[0] - start_vel);

    // Interior rows
    for i in 1..n - 1 {
        a_diag[i] = h[i - 1];
        b_diag[i] = 2.0 * (h[i - 1] + h[i]);
        c_diag[i] = h[i];
        d_rhs[i] =
            6.0 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]);
    }

    // Row n-1 (clamped boundary): h[n-2]*M[n-2] + 2*h[n-2]*M[n-1] = 6*(end_vel - (yn-yn-1)/h[n-2])
    let last_h = h[n - 2];
    a_diag[n - 1] = last_h;
    b_diag[n - 1] = 2.0 * last_h;
    d_rhs[n - 1] = 6.0 * (end_vel - (values[n - 1] - values[n - 2]) / last_h);

    // Thomas algorithm
    for i in 1..n {
        if b_diag[i - 1].abs() < 1e-15 {
            continue;
        }
        let factor = a_diag[i] / b_diag[i - 1];
        b_diag[i] -= factor * c_diag[i - 1];
        d_rhs[i] -= factor * d_rhs[i - 1];
    }

    let mut moments = vec![0.0; n];
    if b_diag[n - 1].abs() > 1e-15 {
        moments[n - 1] = d_rhs[n - 1] / b_diag[n - 1];
    }
    for i in (0..n - 1).rev() {
        if b_diag[i].abs() > 1e-15 {
            moments[i] = (d_rhs[i] - c_diag[i] * moments[i + 1]) / b_diag[i];
        }
    }

    moments
}

/// Evaluate cubic spline segment at time t, returning (position, velocity, acceleration).
fn evaluate_cubic_segment(
    t: f64,
    t0: f64,
    t1: f64,
    y0: f64,
    y1: f64,
    m0: f64,
    m1: f64,
) -> (f64, f64, f64) {
    let h = t1 - t0;
    if h.abs() < 1e-15 {
        return (y0, 0.0, 0.0);
    }

    let a = (t1 - t) / h;
    let b = (t - t0) / h;

    // Position: S(t) = a*y0 + b*y1 + (a^3 - a)*h^2/6*M0 + (b^3 - b)*h^2/6*M1
    let h2_6 = h * h / 6.0;
    let position = a * y0 + b * y1 + (a * a * a - a) * h2_6 * m0 + (b * b * b - b) * h2_6 * m1;

    // Velocity: dS/dt = (y1-y0)/h - (3a^2-1)*h/6*M0 + (3b^2-1)*h/6*M1
    let h_6 = h / 6.0;
    let velocity = (y1 - y0) / h - (3.0 * a * a - 1.0) * h_6 * m0 + (3.0 * b * b - 1.0) * h_6 * m1;

    // Acceleration: d^2S/dt^2 = a*M0 + b*M1
    let acceleration = a * m0 + b * m1;

    (position, velocity, acceleration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_basic() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![2.0, 1.0]];
        let result = cubic_spline_time(&path, Some(2.0), None).unwrap();

        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 3);

        // Start position
        assert!((result.waypoints[0].positions[0] - 0.0).abs() < 1e-6);
        assert!((result.waypoints[0].positions[1] - 0.0).abs() < 1e-6);

        // End position
        let last = result.waypoints.last().unwrap();
        assert!((last.positions[0] - 2.0).abs() < 1e-6);
        assert!((last.positions[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spline_empty() {
        let result = cubic_spline_time(&[], None, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn spline_single() {
        let result = cubic_spline_time(&[vec![1.0]], None, None).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn spline_auto_duration() {
        let path = vec![vec![0.0], vec![5.0]];
        let vel_limits = vec![1.0]; // should take ~5s at v=1

        let result = cubic_spline_time(&path, None, Some(&vel_limits)).unwrap();
        assert!(result.duration().as_secs_f64() > 1.0);
    }

    #[test]
    fn spline_invalid_duration() {
        let path = vec![vec![0.0], vec![1.0]];
        assert!(cubic_spline_time(&path, Some(-1.0), None).is_err());
        assert!(cubic_spline_time(&path, Some(0.0), None).is_err());
    }

    #[test]
    fn spline_monotonic_time() {
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![3.0, 0.5],
        ];
        let result = cubic_spline_time(&path, Some(3.0), None).unwrap();

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
    fn spline_c2_continuous() {
        // Verify acceleration continuity at knot points
        let path = vec![vec![0.0], vec![1.0], vec![0.5], vec![2.0]];
        let result = cubic_spline_time(&path, Some(3.0), None).unwrap();

        // Check that accelerations don't jump drastically between consecutive samples
        for i in 1..result.waypoints.len() {
            let accel_diff = (result.waypoints[i].accelerations[0]
                - result.waypoints[i - 1].accelerations[0])
                .abs();
            let dt = result.waypoints[i].time - result.waypoints[i - 1].time;
            if dt > 1e-10 {
                // Acceleration rate of change should be bounded
                let jerk_estimate = accel_diff / dt;
                assert!(
                    jerk_estimate < 1000.0,
                    "Excessive jerk at waypoint {}: {}",
                    i,
                    jerk_estimate
                );
            }
        }
    }

    #[test]
    fn spline_clamped() {
        let path = vec![vec![0.0], vec![1.0], vec![2.0]];
        let start_vel = vec![0.5];
        let end_vel = vec![0.0];

        let result = cubic_spline_time_clamped(&path, 2.0, &start_vel, &end_vel).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 3);
    }

    #[test]
    fn spline_clamped_mismatch() {
        let path = vec![vec![0.0, 0.0]];
        assert!(cubic_spline_time_clamped(&path, 1.0, &[0.0], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn spline_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![2.0, 1.0]];
        let result = cubic_spline_time(&path, Some(2.0), None).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Cubic spline should produce non-zero accelerations"
        );
    }

    #[test]
    fn spline_clamped_accelerations_nonempty() {
        let path = vec![vec![0.0], vec![1.0], vec![2.0]];
        let result = cubic_spline_time_clamped(&path, 2.0, &[0.5], &[0.0]).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Clamped spline should produce non-zero accelerations"
        );
    }

    #[test]
    fn spline_natural_moments() {
        // Test that natural spline moments are zero at boundaries
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 1.0, 0.5, 2.0];
        let moments = natural_spline_moments(&times, &values);

        assert!((moments[0]).abs() < 1e-10, "Natural BC: M[0] should be 0");
        assert!(
            (moments[moments.len() - 1]).abs() < 1e-10,
            "Natural BC: M[n-1] should be 0"
        );
    }
}
