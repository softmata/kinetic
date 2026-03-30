//! Trapezoidal velocity profile for time parameterization.
//!
//! Classic industrial profile: accelerate → cruise at max velocity → decelerate.
//! Applied per-segment (between consecutive waypoints) with uniform limits.
//!
//! For short segments where max velocity can't be reached, produces a
//! triangular profile (accelerate → decelerate, no cruise phase).

use std::time::Duration;

/// A single timed waypoint with position, velocity, and acceleration.
#[derive(Debug, Clone)]
pub struct TimedWaypoint {
    /// Time from trajectory start (seconds).
    pub time: f64,
    /// Joint positions at this time.
    pub positions: Vec<f64>,
    /// Joint velocities at this time (rad/s).
    pub velocities: Vec<f64>,
    /// Joint accelerations at this time (rad/s^2).
    pub accelerations: Vec<f64>,
}

/// Result of time parameterization.
#[derive(Debug, Clone)]
pub struct TimedTrajectory {
    /// Total trajectory duration.
    pub duration: Duration,
    /// DOF (number of joints).
    pub dof: usize,
    /// Timed waypoints (sorted by time).
    pub waypoints: Vec<TimedWaypoint>,
}

impl TimedTrajectory {
    /// Total duration of the trajectory.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Number of waypoints.
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    /// Whether the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Validate dimensional consistency of the trajectory.
    ///
    /// Checks that every waypoint has `positions.len() == velocities.len()
    /// == accelerations.len() == dof`, that accelerations are non-empty,
    /// and that timestamps are monotonically non-decreasing.
    ///
    /// Returns `Ok(())` or a description of the first inconsistency found.
    pub fn validate(&self) -> Result<(), String> {
        for (i, wp) in self.waypoints.iter().enumerate() {
            if wp.positions.len() != self.dof {
                return Err(format!(
                    "Waypoint {}: positions.len()={} != dof={}",
                    i,
                    wp.positions.len(),
                    self.dof
                ));
            }
            if wp.velocities.len() != self.dof {
                return Err(format!(
                    "Waypoint {}: velocities.len()={} != dof={}",
                    i,
                    wp.velocities.len(),
                    self.dof
                ));
            }
            if wp.accelerations.len() != self.dof {
                return Err(format!(
                    "Waypoint {}: accelerations.len()={} != dof={}",
                    i,
                    wp.accelerations.len(),
                    self.dof
                ));
            }
            if i > 0 && wp.time < self.waypoints[i - 1].time - 1e-12 {
                return Err(format!(
                    "Waypoint {}: time {} < previous time {}",
                    i,
                    wp.time,
                    self.waypoints[i - 1].time
                ));
            }
        }
        Ok(())
    }

    /// Sample the trajectory at an arbitrary time (linear interpolation between waypoints).
    pub fn sample_at(&self, t: Duration) -> TimedWaypoint {
        let t_sec = t.as_secs_f64();

        if self.waypoints.is_empty() {
            return TimedWaypoint {
                time: t_sec,
                positions: vec![],
                velocities: vec![],
                accelerations: vec![],
            };
        }

        if self.waypoints.len() == 1 || t_sec <= self.waypoints[0].time {
            return self.waypoints[0].clone();
        }

        let last = self.waypoints.last().unwrap();
        if t_sec >= last.time {
            return last.clone();
        }

        // Binary search for the surrounding waypoints
        let idx = match self
            .waypoints
            .binary_search_by(|wp| wp.time.partial_cmp(&t_sec).unwrap())
        {
            Ok(exact) => return self.waypoints[exact].clone(),
            Err(insert) => insert.saturating_sub(1),
        };

        let wp_lo = &self.waypoints[idx];
        let wp_hi = &self.waypoints[idx + 1];
        let dt = wp_hi.time - wp_lo.time;

        if dt.abs() < 1e-15 {
            return wp_lo.clone();
        }

        let alpha = (t_sec - wp_lo.time) / dt;

        let positions: Vec<f64> = wp_lo
            .positions
            .iter()
            .zip(&wp_hi.positions)
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect();

        let velocities: Vec<f64> = wp_lo
            .velocities
            .iter()
            .zip(&wp_hi.velocities)
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect();

        let accelerations: Vec<f64> = wp_lo
            .accelerations
            .iter()
            .zip(&wp_hi.accelerations)
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect();

        TimedWaypoint {
            time: t_sec,
            positions,
            velocities,
            accelerations,
        }
    }
}

/// Compute trapezoidal velocity profile for a joint-space path.
///
/// Each segment between consecutive waypoints gets a trapezoidal (or triangular)
/// profile. The slowest joint determines the segment duration.
///
/// `path`: list of joint-space waypoints (each `Vec<f64>` of length DOF).
/// `max_velocity`: maximum joint velocity (rad/s), applied uniformly.
/// `max_acceleration`: maximum joint acceleration (rad/s^2), applied uniformly.
pub fn trapezoidal(
    path: &[Vec<f64>],
    max_velocity: f64,
    max_acceleration: f64,
) -> Result<TimedTrajectory, String> {
    if max_velocity <= 0.0 || !max_velocity.is_finite() {
        return Err(format!(
            "max_velocity must be positive and finite, got {max_velocity}"
        ));
    }
    if max_acceleration <= 0.0 || !max_acceleration.is_finite() {
        return Err(format!(
            "max_acceleration must be positive and finite, got {max_acceleration}"
        ));
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

        // Find the joint with the largest displacement (determines timing)
        let max_displacement = from
            .iter()
            .zip(to.iter())
            .map(|(&a, &b)| (b - a).abs())
            .fold(0.0_f64, f64::max);

        if max_displacement < 1e-12 {
            // No motion — add zero-duration waypoint
            waypoints.push(TimedWaypoint {
                time: current_time,
                positions: to.clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            });
            continue;
        }

        // Compute trapezoidal profile timing for the max-displacement joint
        let seg_profile =
            compute_trapezoidal_timing(max_displacement, max_velocity, max_acceleration);

        // Generate intermediate waypoints: start of cruise, end of cruise, end of decel
        let t_accel = seg_profile.t_accel;
        let t_cruise = seg_profile.t_cruise;
        let t_decel = seg_profile.t_decel;
        let total = t_accel + t_cruise + t_decel;

        // Compute per-joint velocities and accelerations
        // Scale each joint proportionally to its displacement
        let displacements: Vec<f64> = from.iter().zip(to.iter()).map(|(&a, &b)| b - a).collect();

        // At end of accel phase
        if t_accel > 1e-12 {
            let t = current_time + t_accel;
            let frac = trapezoidal_position_fraction(t_accel, &seg_profile);
            let positions: Vec<f64> = from
                .iter()
                .zip(&displacements)
                .map(|(&f, &d)| f + frac * d)
                .collect();
            let velocities: Vec<f64> = displacements
                .iter()
                .map(|&d| d / max_displacement * seg_profile.cruise_vel)
                .collect();
            let accelerations: Vec<f64> = displacements
                .iter()
                .map(|&d| {
                    if t_accel > 1e-12 {
                        d / max_displacement * seg_profile.cruise_vel / t_accel
                    } else {
                        0.0
                    }
                })
                .collect();

            waypoints.push(TimedWaypoint {
                time: t,
                positions,
                velocities,
                accelerations,
            });
        }

        // At end of cruise phase (start of decel)
        if t_cruise > 1e-12 {
            let t = current_time + t_accel + t_cruise;
            let frac = trapezoidal_position_fraction(t_accel + t_cruise, &seg_profile);
            let positions: Vec<f64> = from
                .iter()
                .zip(&displacements)
                .map(|(&f, &d)| f + frac * d)
                .collect();
            let velocities: Vec<f64> = displacements
                .iter()
                .map(|&d| d / max_displacement * seg_profile.cruise_vel)
                .collect();

            waypoints.push(TimedWaypoint {
                time: t,
                positions,
                velocities,
                accelerations: vec![0.0; dof],
            });
        }

        // End of segment
        current_time += total;
        let decel_acc: Vec<f64> = displacements
            .iter()
            .map(|&d| {
                if t_decel > 1e-12 {
                    -d / max_displacement * seg_profile.cruise_vel / t_decel
                } else {
                    0.0
                }
            })
            .collect();

        waypoints.push(TimedWaypoint {
            time: current_time,
            positions: to.clone(),
            velocities: vec![0.0; dof], // stopped at end of segment
            accelerations: decel_acc,
        });
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(current_time),
        dof,
        waypoints,
    })
}

/// Internal: trapezoidal profile timing for a single 1D displacement.
struct TrapezoidalTiming {
    t_accel: f64,
    t_cruise: f64,
    t_decel: f64,
    cruise_vel: f64,
}

fn compute_trapezoidal_timing(
    displacement: f64,
    max_vel: f64,
    max_accel: f64,
) -> TrapezoidalTiming {
    let d = displacement.abs();

    // Time to accelerate to max velocity
    let t_ramp = max_vel / max_accel;
    // Distance covered during accel and decel
    let d_ramp = max_vel * t_ramp; // = max_vel^2 / max_accel

    if d >= d_ramp {
        // Full trapezoidal: accel → cruise → decel
        let t_accel = t_ramp;
        let t_decel = t_ramp;
        let d_cruise = d - d_ramp;
        let t_cruise = d_cruise / max_vel;
        TrapezoidalTiming {
            t_accel,
            t_cruise,
            t_decel,
            cruise_vel: max_vel,
        }
    } else {
        // Triangular: accel → decel (can't reach max velocity)
        let v_peak = (d * max_accel).sqrt();
        let t_ramp = v_peak / max_accel;
        TrapezoidalTiming {
            t_accel: t_ramp,
            t_cruise: 0.0,
            t_decel: t_ramp,
            cruise_vel: v_peak,
        }
    }
}

/// Compute the position fraction (0..1) at local time t within a trapezoidal segment.
fn trapezoidal_position_fraction(local_t: f64, timing: &TrapezoidalTiming) -> f64 {
    let t_a = timing.t_accel;
    let t_c = timing.t_cruise;
    let t_d = timing.t_decel;
    let v = timing.cruise_vel;
    let total_time = t_a + t_c + t_d;

    if total_time < 1e-15 {
        return 1.0;
    }

    // Total displacement (for normalization)
    let total_dist = 0.5 * v * t_a + v * t_c + 0.5 * v * t_d;
    if total_dist < 1e-15 {
        return 1.0;
    }

    let t = local_t.clamp(0.0, total_time);

    let dist = if t <= t_a {
        // Acceleration phase
        let accel = v / t_a;
        0.5 * accel * t * t
    } else if t <= t_a + t_c {
        // Cruise phase
        let d_accel = 0.5 * v * t_a;
        let dt = t - t_a;
        d_accel + v * dt
    } else {
        // Deceleration phase
        let d_accel = 0.5 * v * t_a;
        let d_cruise = v * t_c;
        let dt = t - t_a - t_c;
        let decel = v / t_d;
        d_accel + d_cruise + v * dt - 0.5 * decel * dt * dt
    };

    dist / total_dist
}

/// Compute trapezoidal parameterization with per-joint velocity and acceleration limits.
///
/// Like `trapezoidal` but respects individual joint limits.
pub fn trapezoidal_per_joint(
    path: &[Vec<f64>],
    velocity_limits: &[f64],
    acceleration_limits: &[f64],
) -> Result<TimedTrajectory, String> {
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
            "Limits length mismatch: dof={}, vel_limits={}, accel_limits={}",
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

        // For each joint, compute the minimum time to traverse this segment
        let mut max_seg_time = 0.0_f64;

        for j in 0..dof {
            let d = (to[j] - from[j]).abs();
            if d < 1e-12 {
                continue;
            }
            let timing = compute_trapezoidal_timing(d, velocity_limits[j], acceleration_limits[j]);
            let seg_time = timing.t_accel + timing.t_cruise + timing.t_decel;
            max_seg_time = max_seg_time.max(seg_time);
        }

        // Compute per-joint cruise velocities and accelerations for this segment
        let velocities: Vec<f64> = from
            .iter()
            .zip(to.iter())
            .map(|(&a, &b)| {
                if max_seg_time > 1e-12 {
                    (b - a) / max_seg_time
                } else {
                    0.0
                }
            })
            .collect();

        // Acceleration at start of segment: v / t_accel for each joint
        // Approximate using the segment time as a trapezoidal profile
        let accel_start: Vec<f64> = (0..dof)
            .map(|j| {
                let d = (to[j] - from[j]).abs();
                if d < 1e-12 || max_seg_time < 1e-12 {
                    return 0.0;
                }
                let timing =
                    compute_trapezoidal_timing(d, velocity_limits[j], acceleration_limits[j]);
                let seg_vel = (to[j] - from[j]) / max_seg_time;
                if timing.t_accel > 1e-12 {
                    seg_vel / timing.t_accel
                } else {
                    0.0
                }
            })
            .collect();

        // Deceleration at end of segment
        let accel_end: Vec<f64> = (0..dof)
            .map(|j| {
                let d = (to[j] - from[j]).abs();
                if d < 1e-12 || max_seg_time < 1e-12 {
                    return 0.0;
                }
                let timing =
                    compute_trapezoidal_timing(d, velocity_limits[j], acceleration_limits[j]);
                let seg_vel = (to[j] - from[j]) / max_seg_time;
                if timing.t_decel > 1e-12 {
                    -seg_vel / timing.t_decel
                } else {
                    0.0
                }
            })
            .collect();

        current_time += max_seg_time;

        waypoints.push(TimedWaypoint {
            time: current_time,
            positions: to.clone(),
            velocities: vec![0.0; dof], // stopped at waypoints
            accelerations: accel_end,
        });

        // Update previous waypoint's velocity and acceleration
        let prev_idx = waypoints.len() - 2;
        waypoints[prev_idx].velocities = velocities;
        waypoints[prev_idx].accelerations = accel_start;
    }

    Ok(TimedTrajectory {
        duration: Duration::from_secs_f64(current_time),
        dof,
        waypoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoidal_basic() {
        // Simple 2-waypoint path
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];

        let result = trapezoidal(&path, 1.0, 2.0).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);
        assert!(result.waypoints.len() >= 2);

        // Start at zero velocity
        assert!((result.waypoints[0].velocities[0]).abs() < 1e-10);
        // End at zero velocity
        let last = result.waypoints.last().unwrap();
        assert!((last.velocities[0]).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_empty() {
        let empty: Vec<Vec<f64>> = vec![];
        let result = trapezoidal(&empty, 1.0, 2.0).unwrap();
        assert!(result.is_empty());
        assert_eq!(result.duration(), Duration::ZERO);
    }

    #[test]
    fn trapezoidal_single_waypoint() {
        let path = vec![vec![1.0, 2.0, 3.0]];
        let result = trapezoidal(&path, 1.0, 2.0).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.duration(), Duration::ZERO);
    }

    #[test]
    fn trapezoidal_multi_segment() {
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 0.0]];

        let result = trapezoidal(&path, 2.0, 4.0).unwrap();
        assert!(result.waypoints.len() >= 3);
        assert!(result.duration().as_secs_f64() > 0.0);

        // Timestamps should be monotonically increasing
        for i in 1..result.waypoints.len() {
            assert!(
                result.waypoints[i].time >= result.waypoints[i - 1].time,
                "Timestamps not monotonic at index {}",
                i
            );
        }
    }

    #[test]
    fn trapezoidal_timing_correctness() {
        // Compute timing for a known case
        let timing = compute_trapezoidal_timing(1.0, 1.0, 2.0);
        // v_max=1.0, a=2.0, t_ramp=0.5, d_ramp=0.5*1.0=0.5 < 1.0
        // Full trapezoidal: t_accel=0.5, d_cruise=0.5, t_cruise=0.5, t_decel=0.5
        assert!((timing.t_accel - 0.5).abs() < 1e-10);
        assert!((timing.t_cruise - 0.5).abs() < 1e-10);
        assert!((timing.t_decel - 0.5).abs() < 1e-10);
        assert!((timing.cruise_vel - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_triangular_profile() {
        // Short displacement: can't reach max velocity
        let timing = compute_trapezoidal_timing(0.1, 10.0, 2.0);
        // d=0.1, d_ramp = v_max^2/a = 100/2 = 50 >> 0.1, so triangular
        assert!(
            timing.t_cruise.abs() < 1e-10,
            "Should be triangular (no cruise)"
        );
        assert!(
            timing.cruise_vel < 10.0,
            "Peak velocity should be less than max"
        );
        // v_peak = sqrt(d * a) = sqrt(0.1 * 2) = sqrt(0.2) ≈ 0.447
        assert!((timing.cruise_vel - (0.2_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn sample_at_endpoints() {
        let path = vec![vec![0.0], vec![1.0]];
        let result = trapezoidal(&path, 1.0, 2.0).unwrap();

        // Sample at start
        let wp0 = result.sample_at(Duration::ZERO);
        assert!((wp0.positions[0] - 0.0).abs() < 1e-10);

        // Sample at end
        let wp_end = result.sample_at(result.duration());
        assert!((wp_end.positions[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sample_at_midpoint() {
        let path = vec![vec![0.0], vec![2.0]];
        let result = trapezoidal(&path, 2.0, 4.0).unwrap();

        // Sample at middle of trajectory
        let mid_t = Duration::from_secs_f64(result.duration().as_secs_f64() / 2.0);
        let wp = result.sample_at(mid_t);

        // Position should be approximately 1.0 (midway)
        assert!(
            wp.positions[0] > 0.3 && wp.positions[0] < 1.7,
            "Midpoint position should be roughly halfway, got {}",
            wp.positions[0]
        );
    }

    #[test]
    fn per_joint_limits() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let vel_limits = vec![1.0, 0.5]; // joint 1 is slower
        let accel_limits = vec![2.0, 1.0];

        let result = trapezoidal_per_joint(&path, &vel_limits, &accel_limits).unwrap();
        assert!(result.duration().as_secs_f64() > 0.0);

        // Duration should be determined by the slower joint (joint 1)
        // Joint 1 needs to move 2.0 rad with v_max=0.5, a_max=1.0
        let j1_timing = compute_trapezoidal_timing(2.0, 0.5, 1.0);
        let j1_time = j1_timing.t_accel + j1_timing.t_cruise + j1_timing.t_decel;
        assert!(
            (result.duration().as_secs_f64() - j1_time).abs() < 0.1,
            "Duration should be close to slowest joint time: expected ~{}, got {}",
            j1_time,
            result.duration().as_secs_f64()
        );
    }

    #[test]
    fn per_joint_limits_mismatch() {
        let path = vec![vec![0.0, 0.0]];
        let result = trapezoidal_per_joint(&path, &[1.0], &[2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn trapezoidal_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let result = trapezoidal(&path, 1.0, 2.0).unwrap();
        result.validate().unwrap();

        // At least one waypoint should have non-zero acceleration
        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Trapezoidal should produce non-zero accelerations"
        );
    }

    #[test]
    fn per_joint_accelerations_nonempty() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let vel = vec![1.0, 0.5];
        let acc = vec![2.0, 1.0];
        let result = trapezoidal_per_joint(&path, &vel, &acc).unwrap();
        result.validate().unwrap();

        let has_nonzero = result
            .waypoints
            .iter()
            .any(|wp| wp.accelerations.iter().any(|&a| a.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Per-joint trapezoidal should produce non-zero accelerations"
        );
    }

    #[test]
    fn validate_ok() {
        let path = vec![vec![0.0], vec![1.0]];
        let result = trapezoidal(&path, 1.0, 2.0).unwrap();
        assert!(result.validate().is_ok());
    }

    #[test]
    fn validate_bad_dimensions() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 2,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: vec![0.0, 0.0],
                velocities: vec![0.0], // wrong length
                accelerations: vec![0.0, 0.0],
            }],
        };
        assert!(traj.validate().is_err());
    }

    #[test]
    fn validate_non_monotonic_time() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 1,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.5,
                    positions: vec![0.0],
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
                TimedWaypoint {
                    time: 0.1, // out of order
                    positions: vec![1.0],
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
            ],
        };
        assert!(traj.validate().is_err());
    }
}
