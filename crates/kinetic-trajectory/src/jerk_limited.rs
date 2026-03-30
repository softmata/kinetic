//! Jerk-limited (S-curve) time parameterization.
//!
//! Produces smoother motion than trapezoidal by constraining jerk (rate of
//! change of acceleration). The profile has 7 phases:
//!
//! 1. Jerk+ (acceleration ramp-up)
//! 2. Constant acceleration
//! 3. Jerk- (acceleration ramp-down to zero)
//! 4. Cruise (constant velocity)
//! 5. Jerk- (deceleration ramp-up)
//! 6. Constant deceleration
//! 7. Jerk+ (deceleration ramp-down to zero)
//!
//! Important for delicate tasks, CNC, painting, polishing where smooth
//! acceleration profiles prevent mechanical vibration.

use std::time::Duration;

use crate::trapezoidal::{TimedTrajectory, TimedWaypoint};

/// Compute a jerk-limited (S-curve) time parameterization for a joint-space path.
///
/// Each segment between consecutive waypoints gets an S-curve profile.
/// The slowest joint determines the segment duration.
///
/// `path`: list of joint-space waypoints (each `Vec<f64>` of length DOF).
/// `max_velocity`: maximum joint velocity (rad/s), applied uniformly.
/// `max_acceleration`: maximum joint acceleration (rad/s^2), applied uniformly.
/// `max_jerk`: maximum joint jerk (rad/s^3), applied uniformly.
pub fn jerk_limited(
    path: &[Vec<f64>],
    max_velocity: f64,
    max_acceleration: f64,
    max_jerk: f64,
) -> Result<TimedTrajectory, String> {
    if max_velocity <= 0.0 || max_acceleration <= 0.0 || max_jerk <= 0.0 {
        return Err("Limits must be positive".to_string());
    }

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

    let mut waypoints = Vec::new();
    let mut current_time = 0.0;

    // Add start waypoint
    waypoints.push(TimedWaypoint {
        time: 0.0,
        positions: path[0].clone(),
        velocities: vec![0.0; dof],
        accelerations: vec![0.0; dof],
    });

    for seg in 0..path.len() - 1 {
        let from = &path[seg];
        let to = &path[seg + 1];

        // Find the joint with the largest displacement
        let max_displacement = from
            .iter()
            .zip(to.iter())
            .map(|(&a, &b)| (b - a).abs())
            .fold(0.0_f64, f64::max);

        if max_displacement < 1e-12 {
            waypoints.push(TimedWaypoint {
                time: current_time,
                positions: to.clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            });
            continue;
        }

        let profile =
            compute_scurve_profile(max_displacement, max_velocity, max_acceleration, max_jerk);

        let displacements: Vec<f64> = from.iter().zip(to.iter()).map(|(&a, &b)| b - a).collect();
        let total_seg_time = profile.total_time();

        // Sample S-curve at key phase boundaries for intermediate waypoints
        let phase_times = profile.phase_times();
        let mut cumulative_phase_time = 0.0;

        for (phase_idx, &phase_dt) in phase_times.iter().enumerate() {
            if phase_dt < 1e-12 {
                continue;
            }

            cumulative_phase_time += phase_dt;
            let t = cumulative_phase_time;

            // Skip the last phase boundary — we'll add the final waypoint separately
            if (t - total_seg_time).abs() < 1e-12 {
                continue;
            }

            let (frac, vel_frac, accel_frac) = evaluate_scurve_at(&profile, t, max_displacement);

            let positions: Vec<f64> = from
                .iter()
                .zip(&displacements)
                .map(|(&f, &d)| f + frac * d)
                .collect();

            let velocities: Vec<f64> = displacements
                .iter()
                .map(|&d| {
                    let sign = d.signum();
                    sign * vel_frac * max_velocity.min(profile.v_reach)
                })
                .collect();

            let accelerations: Vec<f64> = displacements
                .iter()
                .map(|&d| {
                    let sign = d.signum();
                    sign * accel_frac * max_acceleration.min(profile.a_reach)
                })
                .collect();

            waypoints.push(TimedWaypoint {
                time: current_time + t,
                positions,
                velocities,
                accelerations,
            });

            let _ = phase_idx; // used for clarity in the loop
        }

        // End of segment
        current_time += total_seg_time;
        waypoints.push(TimedWaypoint {
            time: current_time,
            positions: to.clone(),
            velocities: vec![0.0; dof],
            accelerations: vec![0.0; dof],
        });
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(current_time),
        dof,
        waypoints,
    })
}

/// Compute a jerk-limited (S-curve) time parameterization with per-joint limits.
///
/// Similar to `jerk_limited` but accepts per-joint velocity, acceleration, and jerk limits.
pub fn jerk_limited_per_joint(
    path: &[Vec<f64>],
    velocity_limits: &[f64],
    acceleration_limits: &[f64],
    jerk_limits: &[f64],
) -> Result<TimedTrajectory, String> {
    if path.is_empty() {
        return Ok(TimedTrajectory {
            duration: Duration::ZERO,
            dof: 0,
            waypoints: vec![],
        });
    }

    let dof = path[0].len();
    if velocity_limits.len() != dof || acceleration_limits.len() != dof || jerk_limits.len() != dof
    {
        return Err(format!(
            "Limits length mismatch: dof={}, vel={}, accel={}, jerk={}",
            dof,
            velocity_limits.len(),
            acceleration_limits.len(),
            jerk_limits.len()
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

    let mut waypoints = Vec::new();
    let mut current_time = 0.0;

    waypoints.push(TimedWaypoint {
        time: 0.0,
        positions: path[0].clone(),
        velocities: vec![0.0; dof],
        accelerations: vec![0.0; dof],
    });

    for seg in 0..path.len() - 1 {
        let from = &path[seg];
        let to = &path[seg + 1];

        // Compute per-joint S-curve timing, take the slowest
        let mut max_seg_time = 0.0_f64;

        for j in 0..dof {
            let d = (to[j] - from[j]).abs();
            if d < 1e-12 {
                continue;
            }
            let profile = compute_scurve_profile(
                d,
                velocity_limits[j],
                acceleration_limits[j],
                jerk_limits[j],
            );
            max_seg_time = max_seg_time.max(profile.total_time());
        }

        // Compute per-joint velocity and acceleration at start of segment
        let velocities: Vec<f64> = (0..dof)
            .map(|j| {
                if max_seg_time > 1e-12 {
                    (to[j] - from[j]) / max_seg_time
                } else {
                    0.0
                }
            })
            .collect();

        // Acceleration at start: approximate from per-joint S-curve jerk phase
        let accel_start: Vec<f64> = (0..dof)
            .map(|j| {
                let d = (to[j] - from[j]).abs();
                if d < 1e-12 || max_seg_time < 1e-12 {
                    return 0.0;
                }
                let profile = compute_scurve_profile(
                    d,
                    velocity_limits[j],
                    acceleration_limits[j],
                    jerk_limits[j],
                );
                let sign = (to[j] - from[j]).signum();
                sign * profile.a_reach
            })
            .collect();

        // Deceleration at end
        let accel_end: Vec<f64> = (0..dof)
            .map(|j| {
                let d = (to[j] - from[j]).abs();
                if d < 1e-12 || max_seg_time < 1e-12 {
                    return 0.0;
                }
                let profile = compute_scurve_profile(
                    d,
                    velocity_limits[j],
                    acceleration_limits[j],
                    jerk_limits[j],
                );
                let sign = (to[j] - from[j]).signum();
                -sign * profile.a_reach
            })
            .collect();

        current_time += max_seg_time;

        // Update previous waypoint with start-of-segment velocity and acceleration
        let prev_idx = waypoints.len() - 1;
        waypoints[prev_idx].velocities = velocities;
        waypoints[prev_idx].accelerations = accel_start;

        waypoints.push(TimedWaypoint {
            time: current_time,
            positions: to.clone(),
            velocities: vec![0.0; dof],
            accelerations: accel_end,
        });
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(current_time),
        dof,
        waypoints,
    })
}

// === Internal S-curve computation ===

/// S-curve profile parameters for a single 1D displacement.
///
/// 7 phases: T_j1, T_a, T_j1, T_v, T_j2, T_d, T_j2
/// where T_j = jerk phase, T_a = constant accel, T_v = cruise, T_d = constant decel
#[derive(Debug, Clone)]
struct ScurveProfile {
    /// Jerk time during acceleration (phase 1 and 3)
    t_j1: f64,
    /// Constant acceleration time (phase 2)
    t_a: f64,
    /// Jerk time during deceleration (phase 5 and 7)
    t_j2: f64,
    /// Constant deceleration time (phase 6)
    t_d: f64,
    /// Cruise time (phase 4)
    t_v: f64,
    /// Actually reached peak velocity
    v_reach: f64,
    /// Actually reached peak acceleration
    a_reach: f64,
}

impl ScurveProfile {
    fn total_time(&self) -> f64 {
        // Total = 2*T_j1 + T_a + T_v + 2*T_j2 + T_d
        2.0 * self.t_j1 + self.t_a + self.t_v + 2.0 * self.t_j2 + self.t_d
    }

    /// Return the 7 phase durations in order.
    fn phase_times(&self) -> [f64; 7] {
        [
            self.t_j1, // Phase 1: jerk+
            self.t_a,  // Phase 2: const accel
            self.t_j1, // Phase 3: jerk-
            self.t_v,  // Phase 4: cruise
            self.t_j2, // Phase 5: jerk-
            self.t_d,  // Phase 6: const decel
            self.t_j2, // Phase 7: jerk+
        ]
    }
}

/// Compute S-curve profile for a 1D point-to-point motion.
///
/// Based on Biagiotti & Melchiorri, "Trajectory Planning for Automatic Machines and Robots".
fn compute_scurve_profile(
    displacement: f64,
    max_vel: f64,
    max_accel: f64,
    max_jerk: f64,
) -> ScurveProfile {
    let d = displacement.abs();

    // Time to reach max acceleration: T_j = a_max / j_max
    let t_j = max_accel / max_jerk;

    // Check if max acceleration can be reached
    // Velocity gained during jerk-up + jerk-down (no constant accel phase):
    // v_j = j_max * t_j^2 = a_max^2 / j_max
    let v_j = max_accel * max_accel / max_jerk;

    // Case 1: max velocity is reachable
    // The acceleration phase (jerk-up + const accel + jerk-down) needs to produce v_max
    // v_max = a_max * (T_j + T_a) where T_a is the constant acceleration time
    // If v_max <= v_j, we can't even reach max_accel, so T_a = 0

    let (t_j1, t_a, v_reach, a_reach) = if max_vel >= v_j {
        // Can reach max acceleration
        let t_a = (max_vel - v_j) / max_accel;
        (t_j, t_a, max_vel, max_accel)
    } else {
        // Cannot reach max acceleration — reduce jerk phase time
        // v_max = j_max * t_j'^2, so t_j' = sqrt(v_max / j_max)
        let t_j_reduced = (max_vel / max_jerk).sqrt();
        let a_actual = max_jerk * t_j_reduced;
        (t_j_reduced, 0.0, max_vel, a_actual)
    };

    // Assume symmetric profile: t_j2 = t_j1, t_d = t_a
    let t_j2 = t_j1;
    let t_d = t_a;
    // Distance covered during accel phase: d_accel = 0.5 * v_reach * (2*t_j1 + t_a)
    let t_accel_total = 2.0 * t_j1 + t_a;
    let d_accel = 0.5 * v_reach * t_accel_total;

    // Distance during decel phase (symmetric): same as d_accel
    let t_decel_total = 2.0 * t_j2 + t_d;
    let d_decel = 0.5 * v_reach * t_decel_total;

    let d_accel_decel = d_accel + d_decel;

    if d >= d_accel_decel {
        // Full S-curve: there is a cruise phase
        let d_cruise = d - d_accel_decel;
        let t_v = d_cruise / v_reach;

        ScurveProfile {
            t_j1,
            t_a,
            t_j2,
            t_d,
            t_v,
            v_reach,
            a_reach,
        }
    } else {
        // No cruise phase — need to reduce peak velocity
        // Binary search for the achievable peak velocity
        let mut v_lo = 0.0;
        let mut v_hi = v_reach;

        for _ in 0..64 {
            let v_mid = (v_lo + v_hi) / 2.0;
            let dist = distance_for_velocity(v_mid, max_accel, max_jerk);
            if dist < d {
                v_lo = v_mid;
            } else {
                v_hi = v_mid;
            }
        }

        let v_actual = (v_lo + v_hi) / 2.0;

        // Recompute jerk and accel times for this reduced velocity
        let v_j_check = max_accel * max_accel / max_jerk;
        let (tj, ta, a_actual) = if v_actual >= v_j_check {
            let ta = (v_actual - v_j_check) / max_accel;
            (t_j, ta, max_accel)
        } else {
            let tj_r = (v_actual / max_jerk).sqrt();
            let a_act = max_jerk * tj_r;
            (tj_r, 0.0, a_act)
        };

        ScurveProfile {
            t_j1: tj,
            t_a: ta,
            t_j2: tj,
            t_d: ta,
            t_v: 0.0,
            v_reach: v_actual,
            a_reach: a_actual,
        }
    }
}

/// Compute total distance for a symmetric accel+decel S-curve at a given peak velocity.
fn distance_for_velocity(v: f64, max_accel: f64, max_jerk: f64) -> f64 {
    let v_j = max_accel * max_accel / max_jerk;
    let t_accel_total = if v >= v_j {
        let t_j = max_accel / max_jerk;
        let t_a = (v - v_j) / max_accel;
        2.0 * t_j + t_a
    } else {
        let t_j = (v / max_jerk).sqrt();
        2.0 * t_j
    };

    // Distance = v * t_accel_total (for both accel and decel)
    v * t_accel_total
}

/// Evaluate the S-curve position, velocity fraction, and acceleration fraction at time t.
///
/// Returns (position_fraction, velocity_fraction, acceleration_fraction) where:
/// - position_fraction: 0..1 fraction of total displacement
/// - velocity_fraction: fraction of peak velocity (0..1)
/// - acceleration_fraction: fraction of peak acceleration (-1..1)
fn evaluate_scurve_at(profile: &ScurveProfile, t: f64, displacement: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, profile.total_time());
    let j = if displacement >= 0.0 {
        profile.a_reach / profile.t_j1.max(1e-15)
    } else {
        -(profile.a_reach / profile.t_j1.max(1e-15))
    };
    let phases = profile.phase_times();
    let mut t_remaining = t;
    let mut pos = 0.0;
    let mut vel = 0.0;
    let mut acc = 0.0;

    // Phase 1: jerk = +j (acceleration increasing from 0)
    // a(t) = j*t, v(t) = 0.5*j*t^2, p(t) = j*t^3/6
    let dt = t_remaining.min(phases[0]);
    if dt > 0.0 {
        pos = j * dt * dt * dt / 6.0; // starting from zero vel and accel
        vel = 0.5 * j * dt * dt;
        acc = j * dt;
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 2: jerk = 0, constant acceleration = a_reach
    let dt = t_remaining.min(phases[1]);
    if dt > 0.0 {
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
        // acc stays the same
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 3: jerk = -j (acceleration decreasing to 0)
    let dt = t_remaining.min(phases[2]);
    if dt > 0.0 {
        pos += vel * dt + 0.5 * acc * dt * dt - j * dt * dt * dt / 6.0;
        vel += acc * dt - 0.5 * j * dt * dt;
        acc -= j * dt;
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 4: cruise (jerk = 0, accel = 0, vel = v_reach)
    let dt = t_remaining.min(phases[3]);
    if dt > 0.0 {
        pos += vel * dt;
        // vel stays the same, acc = 0
        acc = 0.0;
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 5: jerk = -j (deceleration starting, acceleration becomes negative)
    let dt = t_remaining.min(phases[4]);
    if dt > 0.0 {
        pos += vel * dt - j * dt * dt * dt / 6.0;
        vel -= 0.5 * j * dt * dt;
        acc = -j * dt;
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 6: constant deceleration (jerk = 0, accel = -a_reach)
    let dt = t_remaining.min(phases[5]);
    if dt > 0.0 {
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;
        // acc stays the same
    }
    t_remaining -= dt;
    if t_remaining <= 0.0 {
        return (
            pos / displacement.abs().max(1e-15),
            vel / profile.v_reach.max(1e-15),
            acc / profile.a_reach.max(1e-15),
        );
    }

    // Phase 7: jerk = +j (deceleration decreasing to 0)
    let dt = t_remaining.min(phases[6]);
    if dt > 0.0 {
        pos += vel * dt + 0.5 * acc * dt * dt + j * dt * dt * dt / 6.0;
        vel += acc * dt + 0.5 * j * dt * dt;
        acc += j * dt;
    }

    (
        pos / displacement.abs().max(1e-15),
        vel / profile.v_reach.max(1e-15),
        acc / profile.a_reach.max(1e-15),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scurve_basic() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 2);

        // Start at zero velocity
        assert!((result.waypoints[0].velocities[0]).abs() < 1e-10);
        // End at zero velocity
        let last = result.waypoints.last().unwrap();
        assert!((last.velocities[0]).abs() < 1e-10);
    }

    #[test]
    fn scurve_empty() {
        let result = jerk_limited(&[], 1.0, 2.0, 10.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn scurve_single_waypoint() {
        let result = jerk_limited(&[vec![1.0, 2.0]], 1.0, 2.0, 10.0).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.duration(), Duration::ZERO);
    }

    #[test]
    fn scurve_invalid_limits() {
        let path = vec![vec![0.0], vec![1.0]];
        assert!(jerk_limited(&path, 0.0, 2.0, 10.0).is_err());
        assert!(jerk_limited(&path, 1.0, 0.0, 10.0).is_err());
        assert!(jerk_limited(&path, 1.0, 2.0, 0.0).is_err());
        assert!(jerk_limited(&path, -1.0, 2.0, 10.0).is_err());
    }

    #[test]
    fn scurve_monotonic_time() {
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 0.0]];
        let result = jerk_limited(&path, 2.0, 4.0, 20.0).unwrap();

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
    fn scurve_longer_than_trapezoidal() {
        // S-curve should be slower than trapezoidal with same v_max and a_max
        // because jerk constraint limits how quickly acceleration can change
        let path = vec![vec![0.0], vec![2.0]];

        let trap = crate::trapezoidal::trapezoidal(&path, 1.0, 2.0).unwrap();
        let scurve = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

        assert!(
            scurve.duration().as_secs_f64() >= trap.duration().as_secs_f64() - 1e-6,
            "S-curve should be >= trapezoidal: scurve={}, trap={}",
            scurve.duration().as_secs_f64(),
            trap.duration().as_secs_f64()
        );
    }

    #[test]
    fn scurve_profile_computation() {
        // Test the internal profile computation
        let profile = compute_scurve_profile(2.0, 1.0, 2.0, 10.0);
        assert!(profile.total_time() > 0.0);
        assert!(
            profile.v_reach <= 1.0 + 1e-10,
            "Peak velocity should not exceed max"
        );
        assert!(
            profile.a_reach <= 2.0 + 1e-10,
            "Peak accel should not exceed max"
        );
    }

    #[test]
    fn scurve_short_distance() {
        // Very short distance — should get reduced velocity profile
        let profile = compute_scurve_profile(0.001, 10.0, 20.0, 100.0);
        assert!(profile.total_time() > 0.0);
        assert!(profile.v_reach <= 10.0);
        // Should have no cruise phase
        assert!(
            profile.t_v < 1e-10,
            "Short distance should have no cruise phase"
        );
    }

    #[test]
    fn scurve_per_joint() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let vel_limits = vec![1.0, 0.5];
        let accel_limits = vec![2.0, 1.0];
        let jerk_limits = vec![10.0, 5.0];

        let result =
            jerk_limited_per_joint(&path, &vel_limits, &accel_limits, &jerk_limits).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
    }

    #[test]
    fn scurve_per_joint_mismatch() {
        let path = vec![vec![0.0, 0.0]];
        let result = jerk_limited_per_joint(&path, &[1.0], &[2.0, 3.0], &[10.0]);
        assert!(result.is_err());
    }

    #[test]
    fn scurve_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Jerk-limited should produce non-zero accelerations"
        );
    }

    #[test]
    fn scurve_per_joint_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let vel = vec![1.0, 0.5];
        let acc = vec![2.0, 1.0];
        let jerk = vec![10.0, 5.0];
        let result = jerk_limited_per_joint(&path, &vel, &acc, &jerk).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Per-joint jerk-limited should produce non-zero accelerations"
        );
    }

    #[test]
    fn scurve_high_jerk_approaches_trapezoidal() {
        // With very high jerk, S-curve should approach trapezoidal duration
        let path = vec![vec![0.0], vec![2.0]];

        let trap = crate::trapezoidal::trapezoidal(&path, 1.0, 2.0).unwrap();
        let scurve = jerk_limited(&path, 1.0, 2.0, 1000.0).unwrap();

        let diff = (scurve.duration().as_secs_f64() - trap.duration().as_secs_f64()).abs();
        assert!(
            diff < 0.1,
            "High jerk S-curve should be close to trapezoidal: diff={}",
            diff
        );
    }

    // ─── Jerk-limited edge case tests ───

    /// Very small jerk limit forces extended jerk phases (long t_j).
    #[test]
    fn scurve_small_jerk_limit() {
        let path = vec![vec![0.0], vec![1.0]];
        // Jerk = 0.5 with accel=2.0 → t_j = 2.0/0.5 = 4.0s per jerk phase
        let result = jerk_limited(&path, 1.0, 2.0, 0.5).unwrap();

        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 2);

        // With small jerk, profile should be significantly longer than trapezoidal
        let trap = crate::trapezoidal::trapezoidal(&path, 1.0, 2.0).unwrap();
        assert!(
            result.duration().as_secs_f64() > trap.duration().as_secs_f64(),
            "Small jerk should make S-curve slower than trapezoidal: scurve={}, trap={}",
            result.duration().as_secs_f64(),
            trap.duration().as_secs_f64()
        );

        // Start and end positions correct
        assert!((result.waypoints[0].positions[0] - 0.0).abs() < 1e-10);
        let last = result.waypoints.last().unwrap();
        assert!((last.positions[0] - 1.0).abs() < 1e-6);
    }

    /// Reverse motion (negative displacement): trajectory should handle sign correctly.
    #[test]
    fn scurve_reverse_motion() {
        let path = vec![vec![1.0, 2.0], vec![0.0, 0.0]]; // moving backward
        let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

        assert!(result.duration().as_secs_f64() > 0.0);

        // Start positions
        assert!((result.waypoints[0].positions[0] - 1.0).abs() < 1e-10);
        assert!((result.waypoints[0].positions[1] - 2.0).abs() < 1e-10);

        // End positions
        let last = result.waypoints.last().unwrap();
        assert!((last.positions[0] - 0.0).abs() < 1e-6);
        assert!((last.positions[1] - 0.0).abs() < 1e-6);

        // Start and end velocities should be zero
        assert!((result.waypoints[0].velocities[0]).abs() < 1e-10);
        assert!((last.velocities[0]).abs() < 1e-10);

        // Duration should match the forward case (symmetric profile)
        let forward_path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let forward = jerk_limited(&forward_path, 1.0, 2.0, 10.0).unwrap();
        let dur_diff = (result.duration().as_secs_f64() - forward.duration().as_secs_f64()).abs();
        assert!(
            dur_diff < 1e-10,
            "Reverse should have same duration as forward: diff={}",
            dur_diff
        );
    }

    /// Phase boundary continuity: positions should be continuous (no jumps).
    #[test]
    fn scurve_phase_boundary_continuity() {
        let path = vec![vec![0.0], vec![3.0]];
        let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

        // Check that consecutive waypoints have monotonically increasing positions
        // (for positive displacement) and no jumps
        for i in 1..result.waypoints.len() {
            let prev = &result.waypoints[i - 1];
            let curr = &result.waypoints[i];

            // Position should increase monotonically (positive displacement)
            assert!(
                curr.positions[0] >= prev.positions[0] - 1e-10,
                "Position decreased at waypoint {}: {} → {}",
                i,
                prev.positions[0],
                curr.positions[0]
            );

            // Time should increase
            assert!(
                curr.time >= prev.time - 1e-10,
                "Time decreased at waypoint {}: {} → {}",
                i,
                prev.time,
                curr.time
            );
        }

        // First position = 0, last position = 3
        assert!((result.waypoints[0].positions[0]).abs() < 1e-10);
        let last = result.waypoints.last().unwrap();
        assert!(
            (last.positions[0] - 3.0).abs() < 1e-6,
            "Final position should be 3.0, got {}",
            last.positions[0]
        );
    }

    /// Profile with no cruise phase (short distance): velocity should not exceed limits.
    #[test]
    fn scurve_no_cruise_phase_velocity_bounded() {
        // Very short distance relative to limits → no cruise phase
        let profile = compute_scurve_profile(0.01, 5.0, 10.0, 50.0);

        assert!(
            profile.t_v < 1e-10,
            "Short distance should have no cruise phase, got t_v={}",
            profile.t_v
        );
        assert!(
            profile.v_reach <= 5.0 + 1e-10,
            "Peak velocity {} exceeds max 5.0",
            profile.v_reach
        );
        assert!(
            profile.a_reach <= 10.0 + 1e-10,
            "Peak accel {} exceeds max 10.0",
            profile.a_reach
        );
        assert!(profile.total_time() > 0.0);
    }

    /// Identical waypoints (zero displacement): should produce zero-duration segment.
    #[test]
    fn scurve_zero_displacement_segment() {
        let path = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0], // identical to previous
            vec![2.0, 3.0],
        ];
        let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

        // Should produce a valid trajectory despite zero-displacement segment
        assert!(result.waypoints.len() >= 3);

        // Final positions correct
        let last = result.waypoints.last().unwrap();
        assert!((last.positions[0] - 2.0).abs() < 1e-6);
        assert!((last.positions[1] - 3.0).abs() < 1e-6);
    }

    /// Per-joint limits with one joint having very small jerk: slowest joint dominates.
    #[test]
    fn scurve_per_joint_slowest_dominates() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        // Joint 0: fast limits, Joint 1: slow (small jerk)
        let vel = vec![2.0, 2.0];
        let acc = vec![4.0, 4.0];
        let jerk = vec![100.0, 1.0]; // joint 1 has very small jerk

        let result = jerk_limited_per_joint(&path, &vel, &acc, &jerk).unwrap();

        // Compare with uniform fast jerk
        let fast_result = jerk_limited_per_joint(&path, &vel, &acc, &[100.0, 100.0]).unwrap();

        assert!(
            result.duration().as_secs_f64() > fast_result.duration().as_secs_f64(),
            "Slow joint should make trajectory longer: slow={}, fast={}",
            result.duration().as_secs_f64(),
            fast_result.duration().as_secs_f64()
        );
    }
}
