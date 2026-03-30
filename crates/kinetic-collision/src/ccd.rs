//! Continuous Collision Detection (CCD) via conservative advancement.
//!
//! Given two robot configurations (q_start, q_end), finds the earliest
//! collision time `t` in [0, 1]. Uses conservative advancement: computes
//! an upper bound on sphere motion per time step, then advances by the
//! safe distance `min_clearance / max_velocity`.
//!
//! # Algorithm (Conservative Advancement)
//!
//! 1. Start at t = 0.
//! 2. Interpolate configuration at t: `q(t) = q_start + t * (q_end - q_start)`.
//! 3. Compute FK, update sphere positions.
//! 4. Compute minimum clearance `d` to obstacles (and self if enabled).
//! 5. If `d <= 0`: collision at time `t`. Return `Some(t)`.
//! 6. Compute maximum sphere velocity `v_max` (upper bound on sphere center speed).
//! 7. Advance: `t += d / v_max`.
//! 8. If `t > 1`: no collision. Return `None`.
//! 9. Repeat from step 2.
//!
//! The conservative guarantee: we never skip over a collision because
//! the advancement step is bounded by the clearance-to-velocity ratio.

use kinetic_core::Pose;

use crate::check::CollisionEnvironment;
use crate::soa::SpheresSoA;
use crate::sphere_model::RobotSphereModel;

/// CCD configuration.
#[derive(Debug, Clone)]
pub struct CCDConfig {
    /// Minimum advancement step to prevent infinite loops at tangent contacts (default: 1e-6).
    pub min_step: f64,
    /// Maximum number of advancement iterations (default: 1000).
    pub max_iterations: usize,
    /// Distance threshold: treat clearance < this as collision (default: 1e-4 meters).
    pub collision_threshold: f64,
    /// Whether to check self-collision during CCD (default: false — expensive).
    pub check_self_collision: bool,
}

impl Default for CCDConfig {
    fn default() -> Self {
        Self {
            min_step: 1e-6,
            max_iterations: 1000,
            collision_threshold: 1e-4,
            check_self_collision: false,
        }
    }
}

/// Result of a CCD query.
#[derive(Debug, Clone)]
pub struct CCDResult {
    /// Time of first collision in [0, 1], or None if no collision.
    pub time_of_impact: Option<f64>,
    /// Minimum clearance observed during the sweep.
    pub min_clearance: f64,
    /// Number of advancement iterations used.
    pub iterations: usize,
}

impl CCDResult {
    /// Whether a collision was detected.
    pub fn collides(&self) -> bool {
        self.time_of_impact.is_some()
    }
}

/// Result of CCD checking across an entire trajectory (multiple segments).
#[derive(Debug, Clone)]
pub struct TrajectoryCollisionResult {
    /// Whether any collision was detected.
    pub collision: bool,
    /// Index of the first colliding segment (waypoint pair), if any.
    pub segment_index: Option<usize>,
    /// Time of impact within the colliding segment [0, 1], if any.
    pub time_in_segment: Option<f64>,
    /// Minimum clearance observed across all segments.
    pub min_clearance: f64,
    /// Total CCD iterations across all segments.
    pub total_iterations: usize,
}

/// Continuous collision detector using conservative advancement.
pub struct ContinuousCollisionDetector<'a> {
    sphere_model: &'a RobotSphereModel,
    environment: &'a CollisionEnvironment,
    skip_pairs: Vec<(usize, usize)>,
    config: CCDConfig,
}

impl<'a> ContinuousCollisionDetector<'a> {
    pub fn new(
        sphere_model: &'a RobotSphereModel,
        environment: &'a CollisionEnvironment,
        config: CCDConfig,
    ) -> Self {
        Self {
            sphere_model,
            environment,
            skip_pairs: Vec::new(),
            config,
        }
    }

    /// Set self-collision skip pairs (e.g., adjacent links).
    pub fn with_skip_pairs(mut self, skip_pairs: Vec<(usize, usize)>) -> Self {
        self.skip_pairs = skip_pairs;
        self
    }

    /// Check for collision along the motion from `poses_start` to `poses_end`.
    ///
    /// `poses_start` and `poses_end` are FK link poses at the start and end
    /// configurations respectively. The motion is linearly interpolated in
    /// workspace (link pose space).
    ///
    /// Returns `CCDResult` with `time_of_impact` if a collision is found.
    pub fn check_motion(
        &self,
        poses_start: &[Pose],
        poses_end: &[Pose],
    ) -> CCDResult {
        let mut t = 0.0;
        let mut min_clearance = f64::INFINITY;
        let mut iterations = 0;

        // Pre-compute sphere positions at start and end
        let spheres_start = self.compute_world_spheres(poses_start);
        let spheres_end = self.compute_world_spheres(poses_end);

        // Pre-compute maximum velocity for each sphere
        let max_velocity = self.compute_max_velocity(&spheres_start, &spheres_end);

        if max_velocity < 1e-12 {
            // No motion — just check start configuration
            let d = self.min_distance(&spheres_start);
            return CCDResult {
                time_of_impact: if d <= self.config.collision_threshold {
                    Some(0.0)
                } else {
                    None
                },
                min_clearance: d,
                iterations: 1,
            };
        }

        while t <= 1.0 && iterations < self.config.max_iterations {
            iterations += 1;

            // Interpolate sphere positions at time t
            let spheres_t = self.interpolate_spheres(&spheres_start, &spheres_end, t);

            // Compute minimum clearance
            let d = self.min_distance(&spheres_t);
            min_clearance = min_clearance.min(d);

            if d <= self.config.collision_threshold {
                return CCDResult {
                    time_of_impact: Some(t),
                    min_clearance,
                    iterations,
                };
            }

            // Conservative advancement: safe step = clearance / max_velocity
            let dt = (d / max_velocity).max(self.config.min_step);
            t += dt;
        }

        CCDResult {
            time_of_impact: None,
            min_clearance,
            iterations,
        }
    }

    /// Convenience: check motion given joint configurations instead of poses.
    /// Requires an FK function to convert joints → poses.
    pub fn check_joint_motion<F>(
        &self,
        q_start: &[f64],
        q_end: &[f64],
        fk_fn: F,
    ) -> CCDResult
    where
        F: Fn(&[f64]) -> Vec<Pose>,
    {
        let poses_start = fk_fn(q_start);
        let poses_end = fk_fn(q_end);
        self.check_motion(&poses_start, &poses_end)
    }

    /// Check an entire trajectory (sequence of joint configurations) for collisions.
    ///
    /// Each consecutive pair `(waypoints[i], waypoints[i+1])` is checked with CCD.
    /// Returns the first collision found, with `segment_index` indicating which segment.
    ///
    /// `fk_fn` converts joint values → link poses (e.g., `forward_kinematics_all`).
    pub fn check_trajectory<F>(
        &self,
        waypoints: &[Vec<f64>],
        fk_fn: F,
    ) -> TrajectoryCollisionResult
    where
        F: Fn(&[f64]) -> Vec<Pose>,
    {
        let mut min_clearance = f64::INFINITY;
        let mut total_iterations = 0;

        for i in 0..waypoints.len().saturating_sub(1) {
            let poses_start = fk_fn(&waypoints[i]);
            let poses_end = fk_fn(&waypoints[i + 1]);

            let result = self.check_motion(&poses_start, &poses_end);
            total_iterations += result.iterations;
            min_clearance = min_clearance.min(result.min_clearance);

            if let Some(toi) = result.time_of_impact {
                return TrajectoryCollisionResult {
                    collision: true,
                    segment_index: Some(i),
                    time_in_segment: Some(toi),
                    min_clearance,
                    total_iterations,
                };
            }
        }

        TrajectoryCollisionResult {
            collision: false,
            segment_index: None,
            time_in_segment: None,
            min_clearance,
            total_iterations,
        }
    }

    /// Check motion using swept volume (capsule) collision.
    ///
    /// Each robot sphere sweeps from its start to end position, forming a capsule.
    /// The capsule is checked against each obstacle sphere using capsule-sphere
    /// distance. This is more accurate than conservative advancement for fast motions
    /// and requires only a single pass (no iterative advancement).
    ///
    /// Returns true if any swept capsule intersects an obstacle.
    pub fn check_swept_volume(
        &self,
        poses_start: &[Pose],
        poses_end: &[Pose],
    ) -> bool {
        let spheres_start = self.compute_world_spheres(poses_start);
        let spheres_end = self.compute_world_spheres(poses_end);
        let obs = &self.environment.obstacle_spheres;

        for i in 0..spheres_start.len().min(spheres_end.len()) {
            let ax = spheres_start.x[i];
            let ay = spheres_start.y[i];
            let az = spheres_start.z[i];
            let bx = spheres_end.x[i];
            let by = spheres_end.y[i];
            let bz = spheres_end.z[i];
            let r_robot = spheres_start.radius[i];

            for j in 0..obs.len() {
                let dist = point_segment_distance(
                    obs.x[j], obs.y[j], obs.z[j],
                    ax, ay, az,
                    bx, by, bz,
                );
                if dist < r_robot + obs.radius[j] + self.config.collision_threshold {
                    return true;
                }
            }
        }

        false
    }

    /// Check motion using swept volume and return the minimum clearance.
    ///
    /// Like `check_swept_volume` but returns the signed distance (negative = penetration).
    pub fn swept_volume_clearance(
        &self,
        poses_start: &[Pose],
        poses_end: &[Pose],
    ) -> f64 {
        let spheres_start = self.compute_world_spheres(poses_start);
        let spheres_end = self.compute_world_spheres(poses_end);
        let obs = &self.environment.obstacle_spheres;
        let mut min_clearance = f64::INFINITY;

        for i in 0..spheres_start.len().min(spheres_end.len()) {
            let ax = spheres_start.x[i];
            let ay = spheres_start.y[i];
            let az = spheres_start.z[i];
            let bx = spheres_end.x[i];
            let by = spheres_end.y[i];
            let bz = spheres_end.z[i];
            let r_robot = spheres_start.radius[i];

            for j in 0..obs.len() {
                let dist = point_segment_distance(
                    obs.x[j], obs.y[j], obs.z[j],
                    ax, ay, az,
                    bx, by, bz,
                );
                let clearance = dist - r_robot - obs.radius[j];
                min_clearance = min_clearance.min(clearance);
            }
        }

        min_clearance
    }

    /// Compute world-frame spheres from FK link poses.
    fn compute_world_spheres(&self, link_poses: &[Pose]) -> SpheresSoA {
        let local = &self.sphere_model.local;
        let mut world = SpheresSoA::with_capacity(local.len());

        for i in 0..local.len() {
            let link_idx = local.link_id[i];
            if link_idx >= link_poses.len() {
                continue;
            }
            let iso = &link_poses[link_idx].0;
            let local_pt = nalgebra::Point3::new(local.x[i], local.y[i], local.z[i]);
            let world_pt = iso.transform_point(&local_pt);
            world.push(world_pt.x, world_pt.y, world_pt.z, local.radius[i], link_idx);
        }

        world
    }

    /// Compute maximum velocity (max displacement) across all spheres.
    fn compute_max_velocity(&self, start: &SpheresSoA, end: &SpheresSoA) -> f64 {
        let mut max_v = 0.0_f64;
        for i in 0..start.len().min(end.len()) {
            let dx = end.x[i] - start.x[i];
            let dy = end.y[i] - start.y[i];
            let dz = end.z[i] - start.z[i];
            let v = (dx * dx + dy * dy + dz * dz).sqrt();
            max_v = max_v.max(v);
        }
        max_v
    }

    /// Linearly interpolate sphere positions between start and end at time t.
    fn interpolate_spheres(
        &self,
        start: &SpheresSoA,
        end: &SpheresSoA,
        t: f64,
    ) -> SpheresSoA {
        let n = start.len().min(end.len());
        let mut result = SpheresSoA::with_capacity(n);
        let t1 = 1.0 - t;

        for i in 0..n {
            result.push(
                t1 * start.x[i] + t * end.x[i],
                t1 * start.y[i] + t * end.y[i],
                t1 * start.z[i] + t * end.z[i],
                start.radius[i], // radius doesn't change
                start.link_id[i],
            );
        }

        result
    }

    /// Compute minimum distance from robot spheres to environment (and optionally self).
    fn min_distance(&self, spheres: &SpheresSoA) -> f64 {
        let mut min_d = f64::INFINITY;

        // Environment distance
        let obs = &self.environment.obstacle_spheres;
        for i in 0..spheres.len() {
            for j in 0..obs.len() {
                let dx = spheres.x[i] - obs.x[j];
                let dy = spheres.y[i] - obs.y[j];
                let dz = spheres.z[i] - obs.z[j];
                let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let clearance = center_dist - spheres.radius[i] - obs.radius[j];
                min_d = min_d.min(clearance);
            }
        }

        // Self-collision distance
        if self.config.check_self_collision {
            for i in 0..spheres.len() {
                for j in (i + 1)..spheres.len() {
                    let link_a = spheres.link_id[i];
                    let link_b = spheres.link_id[j];
                    if link_a == link_b {
                        continue;
                    }
                    if self.skip_pairs.iter().any(|&(a, b)| {
                        (a == link_a && b == link_b) || (a == link_b && b == link_a)
                    }) {
                        continue;
                    }

                    let dx = spheres.x[i] - spheres.x[j];
                    let dy = spheres.y[i] - spheres.y[j];
                    let dz = spheres.z[i] - spheres.z[j];
                    let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let clearance = center_dist - spheres.radius[i] - spheres.radius[j];
                    min_d = min_d.min(clearance);
                }
            }
        }

        min_d
    }
}

/// Distance from point P to line segment AB.
///
/// Projects P onto the line through A→B, clamps to [0,1], returns distance.
fn point_segment_distance(
    px: f64, py: f64, pz: f64,
    ax: f64, ay: f64, az: f64,
    bx: f64, by: f64, bz: f64,
) -> f64 {
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;

    let apx = px - ax;
    let apy = py - ay;
    let apz = pz - az;

    let ab_sq = abx * abx + aby * aby + abz * abz;

    if ab_sq < 1e-20 {
        // Degenerate segment (A == B)
        return (apx * apx + apy * apy + apz * apz).sqrt();
    }

    // Project P onto AB, clamp to [0, 1]
    let t = ((apx * abx + apy * aby + apz * abz) / ab_sq).clamp(0.0, 1.0);

    let cx = ax + t * abx - px;
    let cy = ay + t * aby - py;
    let cz = az + t * abz - pz;

    (cx * cx + cy * cy + cz * cz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capt::AABB;

    fn make_sphere(x: f64, y: f64, z: f64, r: f64) -> SpheresSoA {
        let mut s = SpheresSoA::new();
        s.push(x, y, z, r, 0);
        s
    }

    #[test]
    fn ccd_no_collision_no_obstacles() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let result = detector.check_motion(&pose_start, &pose_end);
        assert!(!result.collides());
        assert!(result.min_clearance.is_infinite()); // No obstacles
    }

    #[test]
    fn ccd_detects_collision_with_obstacle() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle at (0.5, 0, 0) with radius 0.1
        let mut obs = SpheresSoA::new();
        obs.push(0.5, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // Move sphere from origin to (1, 0, 0) — passes through obstacle at 0.5
        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let result = detector.check_motion(&pose_start, &pose_end);
        assert!(result.collides(), "Should detect collision with obstacle");

        let toi = result.time_of_impact.unwrap();
        // Sphere (r=0.05) meets obstacle (r=0.1) at x=0.5
        // Contact at sphere_x + 0.05 = 0.5 - 0.1 → sphere_x = 0.35 → t ≈ 0.35
        assert!(toi > 0.0 && toi < 1.0, "TOI should be in (0,1): {}", toi);
        assert!(toi < 0.5, "TOI should be before midpoint: {}", toi);
    }

    #[test]
    fn ccd_no_collision_miss() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle far away at (0, 2, 0)
        let mut obs = SpheresSoA::new();
        obs.push(0.0, 2.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(5.0));

        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // Move along X axis — doesn't intersect obstacle on Y axis
        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let result = detector.check_motion(&pose_start, &pose_end);
        assert!(!result.collides());
    }

    #[test]
    fn ccd_collision_at_start() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.2),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle overlapping with start position
        let mut obs = SpheresSoA::new();
        obs.push(0.1, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let result = detector.check_motion(&pose_start, &pose_end);
        assert!(result.collides());
        assert!(result.time_of_impact.unwrap() < 0.01, "Should collide at t≈0");
    }

    #[test]
    fn ccd_stationary_no_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(1.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // No motion
        let pose = vec![Pose::identity()];
        let result = detector.check_motion(&pose, &pose);
        assert!(!result.collides());
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn ccd_config_defaults() {
        let config = CCDConfig::default();
        assert_eq!(config.min_step, 1e-6);
        assert_eq!(config.max_iterations, 1000);
        assert!(!config.check_self_collision);
    }

    #[test]
    fn ccd_result_api() {
        let r = CCDResult {
            time_of_impact: Some(0.5),
            min_clearance: -0.01,
            iterations: 42,
        };
        assert!(r.collides());

        let r2 = CCDResult {
            time_of_impact: None,
            min_clearance: 0.1,
            iterations: 10,
        };
        assert!(!r2.collides());
    }

    // --- Trajectory CCD tests ---

    #[test]
    fn trajectory_ccd_no_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // 3 waypoints, no obstacles
        let waypoints = vec![
            vec![0.0],
            vec![0.5],
            vec![1.0],
        ];

        let fk = |q: &[f64]| -> Vec<Pose> {
            vec![Pose(nalgebra::Isometry3::translation(q[0], 0.0, 0.0))]
        };

        let result = detector.check_trajectory(&waypoints, fk);
        assert!(!result.collision);
        assert!(result.segment_index.is_none());
    }

    #[test]
    fn trajectory_ccd_collision_second_segment() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle at x=0.75
        let mut obs = SpheresSoA::new();
        obs.push(0.75, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // 3 waypoints: [0] → [0.4] → [1.0]
        // Segment 0 (0→0.4): no collision
        // Segment 1 (0.4→1.0): passes through obstacle at 0.75
        let waypoints = vec![
            vec![0.0],
            vec![0.4],
            vec![1.0],
        ];

        let fk = |q: &[f64]| -> Vec<Pose> {
            vec![Pose(nalgebra::Isometry3::translation(q[0], 0.0, 0.0))]
        };

        let result = detector.check_trajectory(&waypoints, fk);
        assert!(result.collision);
        assert_eq!(result.segment_index, Some(1), "Collision should be in segment 1");
        assert!(result.time_in_segment.unwrap() > 0.0);
    }

    #[test]
    fn trajectory_ccd_empty_and_single() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let fk = |q: &[f64]| -> Vec<Pose> {
            vec![Pose(nalgebra::Isometry3::translation(q[0], 0.0, 0.0))]
        };

        // Empty trajectory
        let result = detector.check_trajectory(&[], fk);
        assert!(!result.collision);

        // Single waypoint (no segments)
        let result = detector.check_trajectory(&[vec![0.0]], fk);
        assert!(!result.collision);
    }

    // --- CCD accuracy vs discrete sampling ---

    #[test]
    fn ccd_catches_tunneling_discrete_misses() {
        // Fast motion: sphere moves from x=0 to x=2.
        // Thin obstacle at x=1.0 with radius 0.01.
        // Discrete sampling at 5 steps (0, 0.4, 0.8, 1.2, 1.6, 2.0) misses it
        // because no sample lands within 0.01 + 0.05 = 0.06 of x=1.0.
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(1.0, 0.0, 0.0, 0.01, 0); // Thin obstacle
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(3.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(2.0, 0.0, 0.0))];

        // CCD should catch it
        let ccd_result = detector.check_motion(&pose_start, &pose_end);
        assert!(ccd_result.collides(), "CCD should detect tunneling collision");

        // Discrete sampling at 5 steps would miss (step size 0.4, obstacle width 0.02+0.1=0.12)
        let discrete_steps = 5;
        let mut discrete_found = false;
        for step in 0..=discrete_steps {
            let t = step as f64 / discrete_steps as f64;
            let x = t * 2.0;
            // Check if sphere at (x,0,0) with r=0.05 overlaps obstacle at (1,0,0) with r=0.01
            let dist = (x - 1.0).abs();
            if dist < 0.05 + 0.01 {
                discrete_found = true;
            }
        }
        // With 5 steps, samples at x = 0, 0.4, 0.8, 1.2, 1.6, 2.0
        // x=0.8: dist=0.2, x=1.2: dist=0.2, both > 0.06
        assert!(!discrete_found, "Discrete 5-step should miss the thin obstacle");
    }

    #[test]
    fn ccd_conservative_no_false_negatives() {
        // Run CCD on a variety of motions, verify it detects collisions
        // whenever discrete fine sampling (100 steps) does.
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(0.5, 0.1, 0.0, 0.15, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let ccd_result = detector.check_motion(&pose_start, &pose_end);

        // Fine discrete check
        let mut discrete_collision = false;
        for step in 0..=100 {
            let t = step as f64 / 100.0;
            let x = t * 1.0;
            let dist = ((x - 0.5) * (x - 0.5) + 0.1 * 0.1).sqrt();
            if dist < 0.05 + 0.15 {
                discrete_collision = true;
                break;
            }
        }

        // CCD must find collision if discrete does (conservative guarantee)
        if discrete_collision {
            assert!(
                ccd_result.collides(),
                "CCD must not miss a collision that discrete sampling finds"
            );
        }
    }

    // --- Swept volume tests ---

    #[test]
    fn point_segment_distance_basic() {
        // Point at origin, segment from (1,0,0) to (3,0,0)
        let d = point_segment_distance(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10, "Distance should be 1.0, got {}", d);

        // Point perpendicular to segment midpoint
        let d = point_segment_distance(2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10, "Distance should be 1.0, got {}", d);

        // Degenerate segment (point)
        let d = point_segment_distance(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10, "Distance should be 1.0, got {}", d);
    }

    #[test]
    fn swept_volume_no_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle far away at y=2
        let mut obs = SpheresSoA::new();
        obs.push(0.5, 2.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(5.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        assert!(!detector.check_swept_volume(&pose_start, &pose_end));
        let clearance = detector.swept_volume_clearance(&pose_start, &pose_end);
        assert!(clearance > 0.0, "Should have positive clearance: {}", clearance);
    }

    #[test]
    fn swept_volume_detects_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle on the sweep path at x=0.5
        let mut obs = SpheresSoA::new();
        obs.push(0.5, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        assert!(detector.check_swept_volume(&pose_start, &pose_end));
        let clearance = detector.swept_volume_clearance(&pose_start, &pose_end);
        assert!(clearance < 0.0, "Should have negative clearance (penetration): {}", clearance);
    }

    #[test]
    fn swept_volume_near_miss() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        // Obstacle just outside sweep path (radius 0.05 sphere sweeping along x,
        // obstacle at y=0.2 with r=0.1 → clearance = 0.2 - 0.05 - 0.1 = 0.05)
        let mut obs = SpheresSoA::new();
        obs.push(0.5, 0.2, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let pose_start = vec![Pose::identity()];
        let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        // Should NOT collide (clearance ~0.05 > threshold 1e-4)
        assert!(!detector.check_swept_volume(&pose_start, &pose_end));
        let clearance = detector.swept_volume_clearance(&pose_start, &pose_end);
        assert!(clearance > 0.0 && clearance < 0.1, "Near miss clearance: {}", clearance);
    }

    // ─── check_motion coverage ──────────────────────────────────────

    #[test]
    fn check_motion_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(0.5, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let start = vec![Pose::identity()];
        let end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

        let result = detector.check_motion(&start, &end);
        assert!(result.collides(), "motion through obstacle should collide");
        assert!(result.time_of_impact.is_some());
        let toi = result.time_of_impact.unwrap();
        assert!(toi >= 0.0 && toi <= 1.0, "TOI should be in [0,1]: {toi}");
    }

    #[test]
    fn check_motion_no_collision() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(5.0, 5.0, 5.0, 0.1, 0); // far away
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        let start = vec![Pose::identity()];
        let end = vec![Pose(nalgebra::Isometry3::translation(0.1, 0.0, 0.0))];

        let result = detector.check_motion(&start, &end);
        assert!(!result.collides(), "motion away from obstacle should not collide");
        assert!(result.time_of_impact.is_none());
        assert!(result.min_clearance > 0.0);
    }

    #[test]
    fn check_motion_zero_motion() {
        let model = RobotSphereModel {
            local: make_sphere(0.0, 0.0, 0.0, 0.05),
            link_ranges: vec![(0, 1)],
            num_links: 1,
        };

        let mut obs = SpheresSoA::new();
        obs.push(5.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

        // Same start and end — zero motion
        let start = vec![Pose::identity()];
        let end = vec![Pose::identity()];

        let result = detector.check_motion(&start, &end);
        assert!(!result.collides(), "zero motion, far from obstacle");
    }

    #[test]
    fn check_motion_with_skip_pairs() {
        let mut local = SpheresSoA::new();
        local.push(0.0, 0.0, 0.0, 0.05, 0);
        local.push(0.0, 0.0, 0.0, 0.05, 1);
        let model = RobotSphereModel {
            local,
            link_ranges: vec![(0, 1), (1, 2)],
            num_links: 2,
        };

        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig {
            check_self_collision: true,
            ..CCDConfig::default()
        }).with_skip_pairs(vec![(0, 1)]); // skip adjacent

        let start = vec![Pose::identity(), Pose::identity()];
        let end = vec![Pose::identity(), Pose::identity()];

        let result = detector.check_motion(&start, &end);
        // With skip pairs, overlapping adjacent links should not count
        assert!(!result.collides());
    }

    // ─── point_segment_distance coverage ────────────────────────────

    #[test]
    fn point_segment_distance_degenerate() {
        // A == B (zero-length segment)
        let d = point_segment_distance(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10, "distance to degenerate segment: {d}");
    }

    #[test]
    fn point_segment_distance_perpendicular() {
        // Point at (0, 1, 0), segment from (0,0,0) to (1,0,0)
        let d = point_segment_distance(0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10, "perpendicular distance: {d}");
    }
}
