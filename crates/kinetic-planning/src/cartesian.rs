//! Cartesian planner — straight-line and circular arc end-effector motion.
//!
//! Essential for approach/retreat motions, welding, painting, deburring,
//! and any task requiring controlled Cartesian trajectories.
//!
//! # Algorithm (linear)
//!
//! 1. Compute start EE pose via FK.
//! 2. Interpolate Cartesian poses from start to goal (LERP position, SLERP rotation).
//! 3. At each interpolated pose, solve IK seeded from previous waypoint.
//! 4. Check for joint jumps (IK discontinuities) via `jump_threshold`.
//! 5. Check collision at each waypoint.
//! 6. Return achieved fraction if path is only partially feasible.

use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::{Isometry3, Vector3};

use kinetic_collision::{CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig};
use kinetic_core::{Pose, Result};
use kinetic_kinematics::{forward_kinematics_all, forward_kinematics, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;

/// Configuration for Cartesian path planning.
#[derive(Debug, Clone)]
pub struct CartesianConfig {
    /// Maximum Cartesian step between waypoints (meters). Default: 0.005 (5mm).
    pub max_step: f64,
    /// Joint-space jump detection threshold. Default: 1.4.
    ///
    /// If any joint changes by more than `jump_threshold * avg_step_size`,
    /// the path is truncated at that point (IK branch switch detected).
    pub jump_threshold: f64,
    /// Whether to check collisions along the path. Default: true.
    pub avoid_collisions: bool,
    /// Collision margin in meters. Default: 0.02.
    pub collision_margin: f64,
}

impl Default for CartesianConfig {
    fn default() -> Self {
        Self {
            max_step: 0.005,
            jump_threshold: 1.4,
            avoid_collisions: true,
            collision_margin: 0.02,
        }
    }
}

/// Result of Cartesian planning.
#[derive(Debug, Clone)]
pub struct CartesianResult {
    /// Joint-space waypoints along the path.
    pub waypoints: Vec<Vec<f64>>,
    /// Fraction of the requested path that was achieved (0.0 to 1.0).
    /// 1.0 means the full path was planned successfully.
    pub fraction: f64,
    /// Total planning time.
    pub planning_time: Duration,
}

/// Cartesian planner for linear and circular arc motions.
pub struct CartesianPlanner {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
}

impl CartesianPlanner {
    /// Create a Cartesian planner for a robot.
    pub fn new(robot: Arc<Robot>, chain: KinematicChain) -> Self {
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let acm = ResolvedACM::from_robot(&robot);
        let environment =
            CollisionEnvironment::empty(0.05, kinetic_collision::capt::AABB::symmetric(10.0));

        Self {
            robot,
            chain,
            sphere_model,
            acm,
            environment,
        }
    }

    /// Set the collision environment.
    pub fn with_environment(mut self, environment: CollisionEnvironment) -> Self {
        self.environment = environment;
        self
    }

    /// Plan a straight line in Cartesian space from start joints to goal pose.
    ///
    /// Interpolates the end-effector pose linearly (LERP position, SLERP rotation)
    /// and solves IK at each interpolated pose, seeded from the previous solution.
    pub fn plan_linear(
        &self,
        start_joints: &[f64],
        goal_pose: &Pose,
        config: &CartesianConfig,
    ) -> Result<CartesianResult> {
        let start_time = Instant::now();

        // Compute start EE pose
        let start_pose = forward_kinematics(&self.robot, &self.chain, start_joints)?;

        // Compute number of interpolation steps based on Cartesian distance
        let pos_dist = (goal_pose.translation() - start_pose.translation()).norm();
        let num_steps = ((pos_dist / config.max_step).ceil() as usize).max(1);

        // Generate interpolated Cartesian poses
        let poses = interpolate_poses(&start_pose, goal_pose, num_steps);

        // Solve IK at each pose, seeded from previous solution
        self.trace_cartesian_path(start_joints, &poses, config, start_time)
    }

    /// Plan relative motion in end-effector frame.
    ///
    /// Moves the end-effector by `relative_motion` (translation in the EE frame).
    pub fn plan_relative(
        &self,
        start_joints: &[f64],
        relative_motion: &Vector3<f64>,
        config: &CartesianConfig,
    ) -> Result<CartesianResult> {
        let start_pose = forward_kinematics(&self.robot, &self.chain, start_joints)?;

        // Apply relative motion in EE frame
        let offset_in_world = start_pose.rotation() * relative_motion;
        let goal_translation = start_pose.0.translation.vector + offset_in_world;
        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(goal_translation),
            *start_pose.rotation(),
        );

        self.plan_linear(start_joints, &Pose(goal_iso), config)
    }

    /// Plan a circular arc in Cartesian space through three points.
    ///
    /// The arc passes through the start EE pose, the `via_pose`, and the `goal_pose`.
    /// Positions trace a circular arc; rotations are interpolated with SLERP.
    pub fn plan_circular(
        &self,
        start_joints: &[f64],
        via_pose: &Pose,
        goal_pose: &Pose,
        config: &CartesianConfig,
    ) -> Result<CartesianResult> {
        let start_time = Instant::now();

        let start_pose = forward_kinematics(&self.robot, &self.chain, start_joints)?;

        // Fit circle through three 3D points
        let p1 = start_pose.translation();
        let p2 = via_pose.translation();
        let p3 = goal_pose.translation();

        let poses = match fit_circular_arc(&p1, &p2, &p3, config.max_step) {
            Some(arc_positions) => {
                // Apply SLERP rotation along the arc
                let start_rot = *start_pose.rotation();
                let goal_rot = *goal_pose.rotation();
                let n = arc_positions.len();

                arc_positions
                    .into_iter()
                    .enumerate()
                    .map(|(i, pos)| {
                        let t = if n <= 1 {
                            1.0
                        } else {
                            i as f64 / (n - 1) as f64
                        };
                        let rot = start_rot.slerp(&goal_rot, t);
                        Pose(Isometry3::from_parts(
                            nalgebra::Translation3::from(pos),
                            rot,
                        ))
                    })
                    .collect::<Vec<_>>()
            }
            None => {
                // Points are collinear — fall back to linear interpolation
                let pos_dist = (goal_pose.translation() - start_pose.translation()).norm();
                let num_steps = ((pos_dist / config.max_step).ceil() as usize).max(1);
                interpolate_poses(&start_pose, goal_pose, num_steps)
            }
        };

        self.trace_cartesian_path(start_joints, &poses, config, start_time)
    }

    /// Trace a sequence of Cartesian poses through IK, collision checking, and jump detection.
    fn trace_cartesian_path(
        &self,
        start_joints: &[f64],
        poses: &[Pose],
        config: &CartesianConfig,
        start_time: Instant,
    ) -> Result<CartesianResult> {
        if poses.is_empty() {
            return Ok(CartesianResult {
                waypoints: vec![start_joints.to_vec()],
                fraction: 1.0,
                planning_time: start_time.elapsed(),
            });
        }

        let mut waypoints = Vec::with_capacity(poses.len() + 1);
        waypoints.push(start_joints.to_vec());

        let mut prev_joints = start_joints.to_vec();
        let total_poses = poses.len();

        for (i, target_pose) in poses.iter().enumerate() {
            // Solve IK seeded from previous solution
            let ik_config = IKConfig {
                seed: Some(prev_joints.clone()),
                num_restarts: 1, // single restart since we're seeded
                ..Default::default()
            };

            let ik_result = match solve_ik(&self.robot, &self.chain, target_pose, &ik_config) {
                Ok(sol) => sol.joints,
                Err(_) => {
                    // IK failed — return partial path
                    let fraction = if total_poses > 0 {
                        i as f64 / total_poses as f64
                    } else {
                        0.0
                    };
                    return Ok(CartesianResult {
                        waypoints,
                        fraction,
                        planning_time: start_time.elapsed(),
                    });
                }
            };

            // Jump detection: check if any joint changed too much
            if detect_jump(&prev_joints, &ik_result, config.jump_threshold) {
                let fraction = if total_poses > 0 {
                    i as f64 / total_poses as f64
                } else {
                    0.0
                };
                return Ok(CartesianResult {
                    waypoints,
                    fraction,
                    planning_time: start_time.elapsed(),
                });
            }

            // Collision check
            if config.avoid_collisions && self.is_in_collision(&ik_result, config.collision_margin)
            {
                let fraction = if total_poses > 0 {
                    i as f64 / total_poses as f64
                } else {
                    0.0
                };
                return Ok(CartesianResult {
                    waypoints,
                    fraction,
                    planning_time: start_time.elapsed(),
                });
            }

            waypoints.push(ik_result.clone());
            prev_joints = ik_result;
        }

        Ok(CartesianResult {
            waypoints,
            fraction: 1.0,
            planning_time: start_time.elapsed(),
        })
    }

    /// Check if a joint configuration is in collision.
    fn is_in_collision(&self, joints: &[f64], margin: f64) -> bool {
        let link_poses = match forward_kinematics_all(&self.robot, &self.chain, joints) {
            Ok(poses) => poses,
            Err(_) => return true,
        };

        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);

        if self
            .environment
            .check_collision_with_margin(&runtime.world, margin)
        {
            return true;
        }

        let skip_pairs = self.acm.to_skip_pairs();
        runtime.self_collision_with_margin(&skip_pairs, margin)
    }
}

// === Helper functions ===

/// Interpolate between two poses with LERP (position) and SLERP (rotation).
///
/// Returns `num_steps + 1` poses from start (inclusive) to goal (inclusive).
fn interpolate_poses(start: &Pose, goal: &Pose, num_steps: usize) -> Vec<Pose> {
    let mut poses = Vec::with_capacity(num_steps + 1);

    let start_pos = start.0.translation.vector;
    let goal_pos = goal.0.translation.vector;
    let start_rot = *start.rotation();
    let goal_rot = *goal.rotation();

    for i in 0..=num_steps {
        let t = if num_steps == 0 {
            1.0
        } else {
            i as f64 / num_steps as f64
        };

        let pos = start_pos + t * (goal_pos - start_pos);
        let rot = start_rot.slerp(&goal_rot, t);
        let iso = Isometry3::from_parts(nalgebra::Translation3::from(pos), rot);
        poses.push(Pose(iso));
    }

    poses
}

/// Detect joint-space jump (IK branch switch).
///
/// Returns true if any joint changed by more than `threshold` radians.
fn detect_jump(prev: &[f64], next: &[f64], threshold: f64) -> bool {
    prev.iter()
        .zip(next.iter())
        .any(|(&a, &b)| (a - b).abs() > threshold)
}

/// Fit a circular arc through three 3D points.
///
/// Returns evenly-spaced positions along the arc, or None if points are collinear.
fn fit_circular_arc(
    p1: &Vector3<f64>,
    p2: &Vector3<f64>,
    p3: &Vector3<f64>,
    max_step: f64,
) -> Option<Vec<Vector3<f64>>> {
    // Vectors from p1 to p2 and p1 to p3
    let v12 = p2 - p1;
    let v13 = p3 - p1;

    // Normal to the plane containing the three points
    let normal = v12.cross(&v13);
    let normal_norm = normal.norm();

    if normal_norm < 1e-10 {
        return None; // collinear points
    }

    let normal = normal / normal_norm;

    // Find circumcenter in the plane of the three points
    // Using the formula for circumcenter of a triangle
    let a = v12;
    let b = v13;

    let a_sq = a.dot(&a);
    let b_sq = b.dot(&b);
    let a_cross_b = a.cross(&b);
    let denom = 2.0 * a_cross_b.dot(&a_cross_b);

    if denom.abs() < 1e-15 {
        return None; // degenerate
    }

    let center_offset = (b_sq * a_cross_b.cross(&a) + a_sq * b.cross(&a_cross_b)) / denom;
    let center = p1 + center_offset;

    let radius = center_offset.norm();

    // Compute angles for each point relative to center
    let r1 = p1 - center;
    let r3 = p3 - center;

    // Create local coordinate frame in the circle plane
    let u = r1 / r1.norm();
    let v = normal.cross(&u);

    // Compute angles
    let angle1 = 0.0; // p1 is at angle 0 by definition
    let r2_local = p2 - center;
    let angle2 = f64::atan2(r2_local.dot(&v), r2_local.dot(&u));
    let angle3 = f64::atan2(r3.dot(&v), r3.dot(&u));

    // Ensure we go through p2 on the way to p3
    // Determine arc direction
    let mut sweep = angle3 - angle1;

    // Check if p2 is on the shorter arc from p1 to p3
    let mid_check = angle2 - angle1;
    let mid_normalized = normalize_angle(mid_check);
    let sweep_normalized = normalize_angle(sweep);

    // If p2 is not between p1 and p3 on the short arc, go the long way
    if (sweep_normalized > 0.0 && (mid_normalized < 0.0 || mid_normalized > sweep_normalized))
        || (sweep_normalized < 0.0 && (mid_normalized > 0.0 || mid_normalized < sweep_normalized))
    {
        // Go the other way
        if sweep > 0.0 {
            sweep -= 2.0 * std::f64::consts::PI;
        } else {
            sweep += 2.0 * std::f64::consts::PI;
        }
    }

    // Compute arc length and number of steps
    let arc_length = (radius * sweep.abs()).max(1e-10);
    let num_steps = ((arc_length / max_step).ceil() as usize).max(1);

    let mut positions = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = if num_steps == 0 {
            1.0
        } else {
            i as f64 / num_steps as f64
        };
        let angle = angle1 + t * sweep;
        let pos = center + radius * (angle.cos() * u + angle.sin() * v);
        positions.push(pos);
    }

    Some(positions)
}

/// Normalize angle to [-pi, pi].
fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * std::f64::consts::PI);
    if a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    } else if a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_ur5e() -> (Arc<Robot>, KinematicChain) {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        (robot, chain)
    }

    #[test]
    fn linear_path_free_space() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        // Move 5cm in Z
        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(
                start_pose.0.translation.vector + Vector3::new(0.0, 0.0, 0.05),
            ),
            *start_pose.rotation(),
        );

        let config = CartesianConfig::default();
        let result = planner
            .plan_linear(&start_joints, &Pose(goal_iso), &config)
            .unwrap();

        assert!(
            result.fraction > 0.9,
            "Expected near-complete path, got fraction={}",
            result.fraction
        );
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn relative_motion() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot, chain);

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];

        // Move 3cm forward in EE frame
        let config = CartesianConfig {
            max_step: 0.01, // coarser for speed
            ..Default::default()
        };

        let result = planner
            .plan_relative(&start_joints, &Vector3::new(0.0, 0.0, 0.03), &config)
            .unwrap();

        assert!(result.fraction > 0.5, "Expected partial or full path");
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn interpolate_poses_basic() {
        let start = Pose::identity();
        let goal = Pose(Isometry3::translation(1.0, 0.0, 0.0));

        let poses = interpolate_poses(&start, &goal, 10);
        assert_eq!(poses.len(), 11); // 0..=10

        // First should be start
        let first_pos = poses[0].translation();
        assert!((first_pos.x).abs() < 1e-10);

        // Last should be goal
        let last_pos = poses[10].translation();
        assert!((last_pos.x - 1.0).abs() < 1e-10);

        // Middle should be 0.5
        let mid_pos = poses[5].translation();
        assert!((mid_pos.x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jump_detection() {
        let prev = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let next_small = vec![0.1, -0.1, 0.05, 0.0, 0.0, 0.0];
        let next_big = vec![0.1, -0.1, 2.0, 0.0, 0.0, 0.0]; // joint 2 jumps

        assert!(!detect_jump(&prev, &next_small, 1.4));
        assert!(detect_jump(&prev, &next_big, 1.4));
    }

    #[test]
    fn circular_arc_basic() {
        // Three points forming a quarter circle in XY plane
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        );
        let p3 = Vector3::new(0.0, 1.0, 0.0);

        let positions = fit_circular_arc(&p1, &p2, &p3, 0.01).unwrap();

        // All points should be approximately at distance 1 from origin
        for pos in &positions {
            let dist = pos.norm();
            assert!(
                (dist - 1.0).abs() < 0.01,
                "Point should be on unit circle, got dist={}",
                dist
            );
        }

        // First point should be p1
        let first = &positions[0];
        assert!((first.x - 1.0).abs() < 1e-6);
        assert!(first.y.abs() < 1e-6);

        // Last point should be p3
        let last = positions.last().unwrap();
        assert!(last.x.abs() < 1e-6);
        assert!((last.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn circular_arc_collinear_returns_none() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(1.0, 0.0, 0.0);
        let p3 = Vector3::new(2.0, 0.0, 0.0);

        assert!(fit_circular_arc(&p1, &p2, &p3, 0.01).is_none());
    }

    #[test]
    fn cartesian_result_fraction() {
        let result = CartesianResult {
            waypoints: vec![vec![0.0; 6], vec![0.1; 6]],
            fraction: 0.5,
            planning_time: Duration::from_millis(1),
        };

        assert!((result.fraction - 0.5).abs() < 1e-10);
        assert_eq!(result.waypoints.len(), 2);
    }

    #[test]
    fn empty_path() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        // Goal is same as start — should succeed immediately
        let config = CartesianConfig::default();
        let result = planner
            .plan_linear(&start_joints, &start_pose, &config)
            .unwrap();

        assert!((result.fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_angle_test() {
        assert!((normalize_angle(0.0) - 0.0).abs() < 1e-10);
        assert!((normalize_angle(std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        assert!(
            (normalize_angle(-3.0 * std::f64::consts::PI) - (-std::f64::consts::PI)).abs() < 1e-10
        );
    }

    // ─── New edge case tests ───

    /// fit_circular_arc() with nearly collinear points (precision limit).
    #[test]
    fn circular_arc_nearly_collinear() {
        // Three points that are almost collinear (offset of 1e-13)
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(1.0, 1e-13, 0.0);
        let p3 = Vector3::new(2.0, 0.0, 0.0);

        // Should return None (collinear within tolerance)
        let result = fit_circular_arc(&p1, &p2, &p3, 0.01);
        assert!(
            result.is_none(),
            "nearly collinear points should return None"
        );
    }

    /// fit_circular_arc() with coincident points.
    #[test]
    fn circular_arc_coincident_points() {
        let p = Vector3::new(1.0, 0.0, 0.0);
        let result = fit_circular_arc(&p, &p, &p, 0.01);
        assert!(result.is_none(), "coincident points should return None");
    }

    /// plan_circular() with collinear points falls back to linear.
    #[test]
    fn plan_circular_collinear_fallback() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        // Via and goal along a line from start
        let start_pos = start_pose.translation();
        let direction = Vector3::new(0.0, 0.0, 0.01);
        let via_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(start_pos + direction),
            *start_pose.rotation(),
        );
        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(start_pos + direction * 2.0),
            *start_pose.rotation(),
        );

        let config = CartesianConfig {
            max_step: 0.005,
            avoid_collisions: false, // skip collision for speed
            ..Default::default()
        };

        // Should not panic — falls back to linear
        let result = planner
            .plan_circular(&start_joints, &Pose(via_iso), &Pose(goal_iso), &config)
            .unwrap();

        assert!(result.waypoints.len() >= 2, "should produce a path");
        assert!(result.fraction > 0.0, "should achieve some fraction");
    }

    /// Cartesian path with 180-degree orientation change (SLERP q/-q singularity).
    #[test]
    fn interpolate_poses_180_degree_rotation() {
        use nalgebra::UnitQuaternion;

        let start = Pose(Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        ));
        // 180 degrees around Z axis
        let goal = Pose(Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI),
        ));

        let poses = interpolate_poses(&start, &goal, 20);
        assert_eq!(poses.len(), 21);

        // Verify no NaN in any pose
        for (i, pose) in poses.iter().enumerate() {
            let t = pose.translation();
            assert!(t.x.is_finite(), "NaN in position at step {i}");
            let q = pose.rotation();
            assert!(q.w.is_finite(), "NaN in rotation at step {i}");
            assert!(q.i.is_finite(), "NaN in rotation at step {i}");
        }

        // Position should interpolate linearly
        let mid = &poses[10];
        assert!(
            (mid.translation().x - 0.5).abs() < 1e-6,
            "midpoint x should be 0.5, got {}",
            mid.translation().x
        );
    }

    /// plan_relative() with zero displacement — should succeed trivially.
    #[test]
    fn plan_relative_zero_displacement() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot, chain);

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let config = CartesianConfig {
            avoid_collisions: false,
            ..Default::default()
        };

        let result = planner
            .plan_relative(&start_joints, &Vector3::zeros(), &config)
            .unwrap();

        // Zero displacement → goal equals start, should achieve fraction 1.0
        assert!(
            (result.fraction - 1.0).abs() < 1e-10,
            "zero displacement should achieve fraction 1.0, got {}",
            result.fraction
        );
    }

    /// detect_jump() edge cases: identical joints, threshold of 0.
    #[test]
    fn detect_jump_edge_cases() {
        let a = vec![1.0, 2.0, 3.0];

        // Identical joints — no jump
        assert!(!detect_jump(&a, &a, 1.4));

        // Tiny threshold — any change triggers jump
        let b = vec![1.001, 2.0, 3.0];
        assert!(detect_jump(&a, &b, 0.0001));

        // Large threshold — big change doesn't trigger
        let c = vec![5.0, 2.0, 3.0];
        assert!(!detect_jump(&a, &c, 10.0));
    }

    /// Interpolate with num_steps=0 — should produce single pose at goal.
    #[test]
    fn interpolate_poses_zero_steps() {
        let start = Pose::identity();
        let goal = Pose(Isometry3::translation(1.0, 0.0, 0.0));

        let poses = interpolate_poses(&start, &goal, 0);
        assert_eq!(poses.len(), 1);
        assert!((poses[0].translation().x - 1.0).abs() < 1e-10);
    }

    /// Circular arc in 3D (not just XY plane).
    #[test]
    fn circular_arc_3d() {
        // Quarter circle in XZ plane
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let p3 = Vector3::new(0.0, 0.0, 1.0);

        let positions = fit_circular_arc(&p1, &p2, &p3, 0.01).unwrap();

        // All points should be on the unit circle (in XZ plane)
        for pos in &positions {
            let dist = pos.norm();
            assert!(
                (dist - 1.0).abs() < 0.01,
                "3D arc point should be at radius 1, got dist={}",
                dist
            );
            assert!(
                pos.y.abs() < 1e-6,
                "3D arc should stay in XZ plane, y={}",
                pos.y
            );
        }
    }

    // ─── Additional coverage tests ───

    #[test]
    fn cartesian_config_default_values() {
        let cfg = CartesianConfig::default();
        assert!((cfg.max_step - 0.005).abs() < 1e-10);
        assert!((cfg.jump_threshold - 1.4).abs() < 1e-10);
        assert!(cfg.avoid_collisions);
        assert!((cfg.collision_margin - 0.02).abs() < 1e-10);
    }

    #[test]
    fn cartesian_config_debug_and_clone() {
        let cfg = CartesianConfig {
            max_step: 0.01,
            jump_threshold: 2.0,
            avoid_collisions: false,
            collision_margin: 0.05,
        };
        let cloned = cfg.clone();
        assert!((cloned.max_step - 0.01).abs() < 1e-10);
        assert!(!cloned.avoid_collisions);
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("max_step"));
    }

    #[test]
    fn cartesian_result_debug_and_clone() {
        let result = CartesianResult {
            waypoints: vec![vec![0.0; 6], vec![1.0; 6]],
            fraction: 0.75,
            planning_time: std::time::Duration::from_millis(5),
        };
        let cloned = result.clone();
        assert_eq!(cloned.waypoints.len(), 2);
        assert!((cloned.fraction - 0.75).abs() < 1e-10);
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("fraction"));
    }

    #[test]
    fn detect_jump_empty_joints() {
        // Empty joint vectors: no elements to compare, so no jump
        let empty: Vec<f64> = vec![];
        assert!(!detect_jump(&empty, &empty, 1.0));
    }

    #[test]
    fn detect_jump_single_joint() {
        let a = vec![0.0];
        let b = vec![0.5];
        assert!(!detect_jump(&a, &b, 1.0));
        assert!(detect_jump(&a, &b, 0.4));
    }

    #[test]
    fn detect_jump_exactly_at_threshold() {
        let a = vec![0.0, 0.0];
        let b = vec![1.4, 0.0];
        // Exactly at threshold: 1.4 > 1.4 is false, so no jump
        assert!(!detect_jump(&a, &b, 1.4));
        // Slightly over threshold: should detect jump
        let c = vec![1.4001, 0.0];
        assert!(detect_jump(&a, &c, 1.4));
    }

    #[test]
    fn detect_jump_negative_values() {
        let a = vec![-1.0, 0.0];
        let b = vec![1.0, 0.0];
        // diff is 2.0, threshold 1.5 -> jump
        assert!(detect_jump(&a, &b, 1.5));
        // threshold 2.5 -> no jump
        assert!(!detect_jump(&a, &b, 2.5));
    }

    #[test]
    fn normalize_angle_various_values() {
        let pi = std::f64::consts::PI;
        // Exact values
        assert!((normalize_angle(0.0)).abs() < 1e-10);
        assert!((normalize_angle(pi) - pi).abs() < 1e-10);
        assert!((normalize_angle(-pi) - (-pi)).abs() < 1e-10);
        // Wrapping
        assert!((normalize_angle(2.0 * pi)).abs() < 1e-10);
        assert!((normalize_angle(-2.0 * pi)).abs() < 1e-10);
        // Large values
        assert!((normalize_angle(5.0 * pi) - pi).abs() < 1e-10);
        assert!((normalize_angle(-5.0 * pi) - (-pi)).abs() < 1e-10);
        // Small positive
        assert!((normalize_angle(0.5) - 0.5).abs() < 1e-10);
        // Small negative
        assert!((normalize_angle(-0.5) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn interpolate_poses_single_step() {
        let start = Pose::identity();
        let goal = Pose(Isometry3::translation(2.0, 0.0, 0.0));
        let poses = interpolate_poses(&start, &goal, 1);
        assert_eq!(poses.len(), 2);
        assert!((poses[0].translation().x).abs() < 1e-10);
        assert!((poses[1].translation().x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_poses_large_number_of_steps() {
        let start = Pose::identity();
        let goal = Pose(Isometry3::translation(1.0, 1.0, 1.0));
        let poses = interpolate_poses(&start, &goal, 1000);
        assert_eq!(poses.len(), 1001);

        // Check monotonicity in x
        for i in 1..poses.len() {
            assert!(
                poses[i].translation().x >= poses[i - 1].translation().x - 1e-10,
                "x should be monotonically increasing"
            );
        }
    }

    #[test]
    fn linear_path_with_collisions_disabled() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(
                start_pose.0.translation.vector + Vector3::new(0.0, 0.0, 0.03),
            ),
            *start_pose.rotation(),
        );

        let config = CartesianConfig {
            avoid_collisions: false,
            ..Default::default()
        };

        let result = planner
            .plan_linear(&start_joints, &Pose(goal_iso), &config)
            .unwrap();
        assert!(result.fraction > 0.0);
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn circular_arc_half_circle() {
        // Half circle: (1,0,0) -> (0,1,0) -> (-1,0,0)
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 1.0, 0.0);
        let p3 = Vector3::new(-1.0, 0.0, 0.0);

        let positions = fit_circular_arc(&p1, &p2, &p3, 0.01).unwrap();

        // All points should be on the unit circle
        for pos in &positions {
            let dist = pos.norm();
            assert!(
                (dist - 1.0).abs() < 0.02,
                "Half-circle point off unit circle, dist={}",
                dist
            );
        }

        // First and last should match p1 and p3
        assert!((positions[0].x - 1.0).abs() < 1e-4);
        let last = positions.last().unwrap();
        assert!((last.x - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn circular_arc_large_step_size() {
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 1.0, 0.0);
        let p3 = Vector3::new(-1.0, 0.0, 0.0);

        // Large step size -> fewer points
        let positions = fit_circular_arc(&p1, &p2, &p3, 10.0).unwrap();
        assert!(positions.len() >= 2, "Should have at least 2 points");
        assert!(positions.len() <= 5, "Large step size should produce few points, got {}", positions.len());
    }

    #[test]
    fn circular_arc_very_small_step_size() {
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        );
        let p3 = Vector3::new(0.0, 1.0, 0.0);

        // Very small step -> many points
        let positions = fit_circular_arc(&p1, &p2, &p3, 0.001).unwrap();
        assert!(positions.len() > 100, "Small step should produce many points, got {}", positions.len());
    }

    #[test]
    fn plan_relative_negative_direction() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot, chain);

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let config = CartesianConfig {
            max_step: 0.01,
            avoid_collisions: false,
            ..Default::default()
        };

        // Move -3cm in x
        let result = planner
            .plan_relative(&start_joints, &Vector3::new(-0.03, 0.0, 0.0), &config)
            .unwrap();
        assert!(result.fraction > 0.0);
    }

    /// Gap 9: Plan Cartesian with 2 waypoints very close together.
    /// Verify fraction is 1.0 and no jump detected by the planner's jump_threshold.
    #[test]
    fn cartesian_short_path_no_jump() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        // Move just 1mm — extremely close, should produce no jump
        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(
                start_pose.0.translation.vector + Vector3::new(0.0, 0.0, 0.001),
            ),
            *start_pose.rotation(),
        );

        let config = CartesianConfig {
            max_step: 0.005,
            jump_threshold: 1.4,
            avoid_collisions: false,
            ..Default::default()
        };

        let result = planner
            .plan_linear(&start_joints, &Pose(goal_iso), &config)
            .unwrap();

        // Fraction should be 1.0 (full path achieved — no jump was detected
        // by the planner's jump_threshold, meaning the path was not truncated)
        assert!(
            (result.fraction - 1.0).abs() < 1e-10,
            "Short path should achieve fraction 1.0, got {}",
            result.fraction
        );

        // Should have at least 2 waypoints (start + goal)
        assert!(
            result.waypoints.len() >= 2,
            "Should have at least 2 waypoints, got {}",
            result.waypoints.len()
        );

        // The detect_jump function uses the planner's jump_threshold (1.4 radians).
        // For a 1mm Cartesian displacement, no joint should exceed that threshold
        // between consecutive waypoints (which is what fraction == 1.0 already confirms).
        // Verify this explicitly:
        for pair in result.waypoints.windows(2) {
            let has_jump = detect_jump(&pair[0], &pair[1], config.jump_threshold);
            assert!(
                !has_jump,
                "No joint-space jump should be detected for a 1mm Cartesian path"
            );
        }
    }

    #[test]
    fn with_environment_builder() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot, chain);
        let env = CollisionEnvironment::empty(
            0.05,
            kinetic_collision::capt::AABB::symmetric(5.0),
        );
        // Just verify the builder chain compiles and returns Self
        let _planner_with_env = planner.with_environment(env);
    }

    #[test]
    fn cartesian_planner_plan_linear_small_displacement() {
        let (robot, chain) = setup_ur5e();
        let planner = CartesianPlanner::new(robot.clone(), chain.clone());

        let start_joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let start_pose = forward_kinematics(&robot, &chain, &start_joints).unwrap();

        // Very tiny displacement (0.1mm)
        let goal_iso = Isometry3::from_parts(
            nalgebra::Translation3::from(
                start_pose.0.translation.vector + Vector3::new(0.0, 0.0, 0.0001),
            ),
            *start_pose.rotation(),
        );

        let config = CartesianConfig {
            avoid_collisions: false,
            ..Default::default()
        };

        let result = planner
            .plan_linear(&start_joints, &Pose(goal_iso), &config)
            .unwrap();

        // Tiny displacement -> should need only a couple of steps
        assert!(result.fraction > 0.9);
    }
}
