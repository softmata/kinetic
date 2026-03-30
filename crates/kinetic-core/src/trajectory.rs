//! Trajectory type — the central data structure flowing through the planning pipeline.
//!
//! Stores waypoints in a joint-major SoA layout for SIMD-friendly access.

use crate::JointValues;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A waypoint with position data and its index.
#[derive(Debug, Clone, PartialEq)]
pub struct Waypoint {
    pub positions: JointValues,
    pub index: usize,
}

/// A timed waypoint with position, velocity, acceleration, and time.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWaypoint {
    pub positions: JointValues,
    pub velocities: JointValues,
    pub accelerations: JointValues,
    pub time: f64,
}

/// A trajectory: sequence of joint-space waypoints.
///
/// Data layout is joint-major SoA:
/// `positions[joint * num_waypoints + waypoint_idx]`
///
/// This means all waypoints for joint 0 are contiguous, then joint 1, etc.
/// Enables SIMD-friendly sequential access when interpolating along one joint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    /// Number of joints (DOF).
    pub dof: usize,
    /// Joint names (for validation and display).
    pub joint_names: Vec<String>,
    /// Positions in joint-major SoA layout.
    /// Length = dof * num_waypoints.
    /// Access: positions[joint * num_waypoints + wp_idx]
    positions: Vec<f64>,
    /// Number of waypoints.
    num_waypoints: usize,
    /// Timestamps (seconds from start). Length = num_waypoints if present.
    timestamps: Option<Vec<f64>>,
    /// Velocities in same SoA layout as positions.
    velocities: Option<Vec<f64>>,
    /// Accelerations in same SoA layout as positions.
    accelerations: Option<Vec<f64>>,
    /// Planning time (how long the planner took).
    pub planning_time: Option<Duration>,
}

impl Trajectory {
    /// Create an empty trajectory for a robot with given DOF.
    pub fn new(dof: usize, joint_names: Vec<String>) -> Self {
        assert_eq!(
            dof,
            joint_names.len(),
            "DOF ({}) must match joint_names length ({})",
            dof,
            joint_names.len()
        );
        Self {
            dof,
            joint_names,
            positions: Vec::new(),
            num_waypoints: 0,
            timestamps: None,
            velocities: None,
            accelerations: None,
            planning_time: None,
        }
    }

    /// Create an empty trajectory with just DOF (no joint names).
    pub fn with_dof(dof: usize) -> Self {
        Self {
            dof,
            joint_names: (0..dof).map(|i| format!("joint_{}", i)).collect(),
            positions: Vec::new(),
            num_waypoints: 0,
            timestamps: None,
            velocities: None,
            accelerations: None,
            planning_time: None,
        }
    }

    /// Number of waypoints.
    pub fn len(&self) -> usize {
        self.num_waypoints
    }

    /// Whether trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.num_waypoints == 0
    }

    /// Whether this trajectory has timing information.
    pub fn is_timed(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Total duration (if timed).
    pub fn duration(&self) -> Option<Duration> {
        self.timestamps
            .as_ref()
            .and_then(|ts| ts.last().map(|&t| Duration::from_secs_f64(t)))
    }

    /// Add a waypoint (positions for all joints).
    ///
    /// The slice length must equal DOF.
    pub fn push_waypoint(&mut self, positions: &[f64]) {
        assert_eq!(
            positions.len(),
            self.dof,
            "Waypoint has {} values, expected {} (DOF)",
            positions.len(),
            self.dof
        );

        if self.num_waypoints == 0 {
            // First waypoint: initialize with space for each joint
            self.positions = vec![0.0; self.dof];
            for (j, &val) in positions.iter().enumerate() {
                self.positions[j] = val;
            }
        } else {
            // Subsequent waypoints: insert into each joint's slice
            // We need to expand each joint's contiguous block by one
            let old_n = self.num_waypoints;
            let new_n = old_n + 1;
            let mut new_positions = vec![0.0; self.dof * new_n];
            #[allow(clippy::needless_range_loop)]
            for j in 0..self.dof {
                // Copy existing waypoints for this joint
                let old_start = j * old_n;
                let new_start = j * new_n;
                new_positions[new_start..new_start + old_n]
                    .copy_from_slice(&self.positions[old_start..old_start + old_n]);
                // Add the new waypoint
                new_positions[new_start + old_n] = positions[j];
            }
            self.positions = new_positions;
        }
        self.num_waypoints += 1;
    }

    /// Get the waypoint at the given index.
    pub fn waypoint(&self, index: usize) -> Waypoint {
        assert!(index < self.num_waypoints, "Waypoint index out of bounds");
        let mut values = vec![0.0; self.dof];
        #[allow(clippy::needless_range_loop)]
        for j in 0..self.dof {
            values[j] = self.positions[j * self.num_waypoints + index];
        }
        Waypoint {
            positions: JointValues::new(values),
            index,
        }
    }

    /// Get all waypoints as a vector of JointValues.
    pub fn waypoints(&self) -> Vec<JointValues> {
        (0..self.num_waypoints)
            .map(|i| self.waypoint(i).positions)
            .collect()
    }

    /// Get the first waypoint's positions.
    pub fn first_waypoint(&self) -> Option<JointValues> {
        if self.num_waypoints > 0 {
            Some(self.waypoint(0).positions)
        } else {
            None
        }
    }

    /// Get the last waypoint's positions.
    pub fn last_waypoint(&self) -> Option<JointValues> {
        if self.num_waypoints > 0 {
            Some(self.waypoint(self.num_waypoints - 1).positions)
        } else {
            None
        }
    }

    /// Interpolate at normalized parameter t ∈ [0, 1] using linear interpolation.
    ///
    /// t=0.0 returns the first waypoint, t=1.0 returns the last.
    /// Intermediate values interpolate linearly between adjacent waypoints.
    ///
    /// See also [`sample_normalized()`](Self::sample_normalized) (explicit alias)
    /// and [`sample_at()`](Self::sample_at) (absolute time).
    pub fn sample(&self, t: f64) -> JointValues {
        assert!(!self.is_empty(), "Cannot sample empty trajectory");
        let t = t.clamp(0.0, 1.0);

        if self.num_waypoints == 1 {
            return self.waypoint(0).positions;
        }

        // Map t to waypoint index space
        let segment_count = (self.num_waypoints - 1) as f64;
        let continuous_idx = t * segment_count;
        let idx_low = (continuous_idx.floor() as usize).min(self.num_waypoints - 2);
        let idx_high = idx_low + 1;
        let alpha = continuous_idx - idx_low as f64;

        // Interpolate per joint
        let mut values = vec![0.0; self.dof];
        #[allow(clippy::needless_range_loop)]
        for j in 0..self.dof {
            let base = j * self.num_waypoints;
            let low = self.positions[base + idx_low];
            let high = self.positions[base + idx_high];
            values[j] = low + alpha * (high - low);
        }
        JointValues::new(values)
    }

    /// Interpolate at normalized parameter t ∈ [0, 1].
    ///
    /// Equivalent to [`sample()`](Self::sample), but with an explicit name
    /// that distinguishes from [`sample_at()`](Self::sample_at) which takes
    /// absolute time.
    ///
    /// - `t = 0.0` → first waypoint
    /// - `t = 0.5` → midpoint
    /// - `t = 1.0` → last waypoint
    pub fn sample_normalized(&self, t: f64) -> JointValues {
        self.sample(t)
    }

    /// Interpolate at absolute time (requires timestamps).
    ///
    /// Returns `None` if the trajectory is not timed.
    pub fn sample_at(&self, time: Duration) -> Option<TimedWaypoint> {
        let timestamps = self.timestamps.as_ref()?;
        let velocities = self.velocities.as_ref();
        let accelerations = self.accelerations.as_ref();

        if self.is_empty() {
            return None;
        }

        let t = time.as_secs_f64();

        // Clamp to trajectory range
        let t = t.clamp(timestamps[0], *timestamps.last().unwrap());

        // Find surrounding waypoints via binary search
        let idx = match timestamps.binary_search_by(|ts| ts.partial_cmp(&t).unwrap()) {
            Ok(exact) => {
                // Exact match — return waypoint directly
                let pos = self.waypoint(exact).positions;
                let vel = velocities
                    .map(|v| {
                        let mut vals = vec![0.0; self.dof];
                        for j in 0..self.dof {
                            vals[j] = v[j * self.num_waypoints + exact];
                        }
                        JointValues::new(vals)
                    })
                    .unwrap_or_else(|| JointValues::zeros(self.dof));
                let acc = accelerations
                    .map(|a| {
                        let mut vals = vec![0.0; self.dof];
                        for j in 0..self.dof {
                            vals[j] = a[j * self.num_waypoints + exact];
                        }
                        JointValues::new(vals)
                    })
                    .unwrap_or_else(|| JointValues::zeros(self.dof));

                return Some(TimedWaypoint {
                    positions: pos,
                    velocities: vel,
                    accelerations: acc,
                    time: t,
                });
            }
            Err(insert) => insert.saturating_sub(1).min(self.num_waypoints - 2),
        };

        let t_low = timestamps[idx];
        let t_high = timestamps[idx + 1];
        let alpha = if (t_high - t_low).abs() < 1e-15 {
            0.0
        } else {
            (t - t_low) / (t_high - t_low)
        };

        let mut pos = vec![0.0; self.dof];
        let mut vel = vec![0.0; self.dof];
        let mut acc = vec![0.0; self.dof];

        for j in 0..self.dof {
            let base = j * self.num_waypoints;
            pos[j] = self.positions[base + idx]
                + alpha * (self.positions[base + idx + 1] - self.positions[base + idx]);
            if let Some(v) = velocities {
                vel[j] = v[base + idx] + alpha * (v[base + idx + 1] - v[base + idx]);
            }
            if let Some(a) = accelerations {
                acc[j] = a[base + idx] + alpha * (a[base + idx + 1] - a[base + idx]);
            }
        }

        Some(TimedWaypoint {
            positions: JointValues::new(pos),
            velocities: JointValues::new(vel),
            accelerations: JointValues::new(acc),
            time: t,
        })
    }

    /// Set timing data for this trajectory (converting geometric path to timed trajectory).
    pub fn set_timing(
        &mut self,
        timestamps: Vec<f64>,
        velocities: Vec<f64>,
        accelerations: Vec<f64>,
    ) {
        assert_eq!(timestamps.len(), self.num_waypoints);
        assert_eq!(velocities.len(), self.dof * self.num_waypoints);
        assert_eq!(accelerations.len(), self.dof * self.num_waypoints);
        self.timestamps = Some(timestamps);
        self.velocities = Some(velocities);
        self.accelerations = Some(accelerations);
    }

    /// Reverse the trajectory (waypoint order).
    pub fn reverse(&self) -> Trajectory {
        let mut reversed = self.clone();
        let n = self.num_waypoints;
        for j in 0..self.dof {
            let base = j * n;
            reversed.positions[base..base + n].reverse();
        }
        if let Some(ref mut ts) = reversed.timestamps {
            let max_t = *ts.last().unwrap_or(&0.0);
            for t in ts.iter_mut() {
                *t = max_t - *t;
            }
            ts.reverse();
        }
        if let Some(ref mut v) = reversed.velocities {
            for j in 0..self.dof {
                let base = j * n;
                v[base..base + n].reverse();
                // Negate velocities (reversed direction)
                for val in &mut v[base..base + n] {
                    *val = -*val;
                }
            }
        }
        if let Some(ref mut a) = reversed.accelerations {
            for j in 0..self.dof {
                let base = j * n;
                a[base..base + n].reverse();
            }
        }
        reversed
    }

    /// Append another trajectory to this one.
    ///
    /// The other trajectory must have the same DOF.
    pub fn append(&mut self, other: &Trajectory) {
        assert_eq!(
            self.dof, other.dof,
            "Cannot append trajectories with different DOF"
        );

        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other.clone();
            return;
        }

        let old_n = self.num_waypoints;
        let add_n = other.num_waypoints;
        let new_n = old_n + add_n;

        let mut new_positions = vec![0.0; self.dof * new_n];
        for j in 0..self.dof {
            // Copy existing
            new_positions[j * new_n..j * new_n + old_n]
                .copy_from_slice(&self.positions[j * old_n..j * old_n + old_n]);
            // Copy appended
            new_positions[j * new_n + old_n..j * new_n + new_n]
                .copy_from_slice(&other.positions[j * add_n..j * add_n + add_n]);
        }
        self.positions = new_positions;
        self.num_waypoints = new_n;

        // Clear timing data on append (would need re-parameterization)
        self.timestamps = None;
        self.velocities = None;
        self.accelerations = None;
    }

    /// Access raw positions data (SoA layout).
    pub fn raw_positions(&self) -> &[f64] {
        &self.positions
    }

    /// Get position of joint `j` at waypoint `wp`.
    pub fn position(&self, joint: usize, waypoint: usize) -> f64 {
        self.positions[joint * self.num_waypoints + waypoint]
    }

    /// Compute total path length in joint space.
    pub fn path_length(&self) -> f64 {
        if self.num_waypoints < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..self.num_waypoints - 1 {
            let wp_a = self.waypoint(i).positions;
            let wp_b = self.waypoint(i + 1).positions;
            total += wp_a.distance_to(&wp_b);
        }
        total
    }
}

/// Build a trajectory from a list of waypoints (convenience).
impl FromIterator<Vec<f64>> for Trajectory {
    fn from_iter<I: IntoIterator<Item = Vec<f64>>>(iter: I) -> Self {
        let waypoints: Vec<Vec<f64>> = iter.into_iter().collect();
        if waypoints.is_empty() {
            return Trajectory::with_dof(0);
        }
        let dof = waypoints[0].len();
        let mut traj = Trajectory::with_dof(dof);
        for wp in &waypoints {
            traj.push_waypoint(wp);
        }
        traj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory() -> Trajectory {
        let mut traj = Trajectory::with_dof(3);
        traj.push_waypoint(&[0.0, 0.0, 0.0]);
        traj.push_waypoint(&[1.0, 2.0, 3.0]);
        traj.push_waypoint(&[2.0, 4.0, 6.0]);
        traj
    }

    #[test]
    fn basic_push_and_len() {
        let traj = make_trajectory();
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.dof, 3);
    }

    #[test]
    fn waypoint_access() {
        let traj = make_trajectory();
        let wp0 = traj.waypoint(0);
        assert!((wp0.positions[0] - 0.0).abs() < 1e-15);
        assert!((wp0.positions[1] - 0.0).abs() < 1e-15);

        let wp2 = traj.waypoint(2);
        assert!((wp2.positions[0] - 2.0).abs() < 1e-15);
        assert!((wp2.positions[2] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn sample_endpoints() {
        let traj = make_trajectory();
        let first = traj.sample(0.0);
        assert!((first[0] - 0.0).abs() < 1e-10);

        let last = traj.sample(1.0);
        assert!((last[0] - 2.0).abs() < 1e-10);
        assert!((last[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn sample_midpoint() {
        let traj = make_trajectory();
        let mid = traj.sample(0.5);
        // Midpoint between wp0=(0,0,0) and wp2=(2,4,6) is at wp1=(1,2,3)
        assert!((mid[0] - 1.0).abs() < 1e-10);
        assert!((mid[1] - 2.0).abs() < 1e-10);
        assert!((mid[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn sample_quarter() {
        let traj = make_trajectory();
        let q = traj.sample(0.25);
        // t=0.25 maps to continuous_idx=0.5, so between wp0 and wp1
        // alpha=0.5, so midpoint of wp0=(0,0,0) and wp1=(1,2,3) = (0.5,1.0,1.5)
        assert!((q[0] - 0.5).abs() < 1e-10);
        assert!((q[1] - 1.0).abs() < 1e-10);
        assert!((q[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn sample_clamp() {
        let traj = make_trajectory();
        let below = traj.sample(-0.5);
        assert!((below[0] - 0.0).abs() < 1e-10);

        let above = traj.sample(1.5);
        assert!((above[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn reverse_trajectory() {
        let traj = make_trajectory();
        let rev = traj.reverse();
        assert_eq!(rev.len(), 3);

        let first = rev.waypoint(0);
        assert!((first.positions[0] - 2.0).abs() < 1e-10);

        let last = rev.waypoint(2);
        assert!((last.positions[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn append_trajectory() {
        let mut traj1 = Trajectory::with_dof(2);
        traj1.push_waypoint(&[0.0, 0.0]);
        traj1.push_waypoint(&[1.0, 1.0]);

        let mut traj2 = Trajectory::with_dof(2);
        traj2.push_waypoint(&[2.0, 2.0]);
        traj2.push_waypoint(&[3.0, 3.0]);

        traj1.append(&traj2);
        assert_eq!(traj1.len(), 4);
        assert!((traj1.waypoint(2).positions[0] - 2.0).abs() < 1e-10);
        assert!((traj1.waypoint(3).positions[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn path_length() {
        let traj = make_trajectory();
        let expected = (1.0_f64.powi(2) + 2.0_f64.powi(2) + 3.0_f64.powi(2)).sqrt() * 2.0;
        assert!((traj.path_length() - expected).abs() < 1e-10);
    }

    #[test]
    fn timed_trajectory() {
        let mut traj = make_trajectory();
        let timestamps = vec![0.0, 0.5, 1.0];
        let velocities = vec![
            1.0, 1.0, 0.0, // joint 0 velocities at 3 waypoints
            2.0, 2.0, 0.0, // joint 1
            3.0, 3.0, 0.0, // joint 2
        ];
        let accelerations = vec![0.0; 9];
        traj.set_timing(timestamps, velocities, accelerations);

        assert!(traj.is_timed());
        assert!((traj.duration().unwrap().as_secs_f64() - 1.0).abs() < 1e-10);

        let wp = traj.sample_at(Duration::from_secs_f64(0.25)).unwrap();
        // t=0.25 is between wp0 (t=0) and wp1 (t=0.5), alpha=0.5
        assert!((wp.positions[0] - 0.5).abs() < 1e-10);
        assert!((wp.time - 0.25).abs() < 1e-10);
    }

    #[test]
    fn from_iterator() {
        let traj: Trajectory = vec![vec![0.0, 0.0], vec![1.0, 2.0]].into_iter().collect();
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.dof, 2);
    }

    #[test]
    fn single_waypoint_sample() {
        let mut traj = Trajectory::with_dof(2);
        traj.push_waypoint(&[1.5, 2.5]);
        let s = traj.sample(0.5);
        assert!((s[0] - 1.5).abs() < 1e-10);
        assert!((s[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn position_access() {
        let traj = make_trajectory();
        assert!((traj.position(0, 1) - 1.0).abs() < 1e-10);
        assert!((traj.position(2, 2) - 6.0).abs() < 1e-10);
    }
}
