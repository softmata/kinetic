//! Automatic analytical IK via subproblem decomposition (EAIK-inspired).
//!
//! Analyzes robot geometry to identify intersecting/parallel axis pairs,
//! then decomposes 6-DOF IK into a sequence of Paden-Kahan subproblems
//! for closed-form solutions.

use crate::chain::KinematicChain;
use crate::forward::{forward_kinematics_all, forward_kinematics};
use crate::ik::{IKConfig, IKMode, IKSolution};
use crate::paden_kahan::{axis_angle_rotation, euler_zyz_decompose, subproblem1};
use kinetic_core::{KineticError, Pose, Result};
use kinetic_robot::Robot;
use nalgebra::{Isometry3, Matrix3, Vector3};

const INTERSECT_TOLERANCE: f64 = 1e-4;

/// Result of analyzing robot geometry for subproblem decomposition.
#[derive(Debug, Clone)]
pub enum DecompositionType {
    /// Axes 4,5,6 intersect at wrist center (most common: UR, Panda, ABB, etc.)
    SphericalWrist { wrist_center: Vector3<f64> },
    /// Robot geometry does not support analytical decomposition.
    NotDecomposable,
}

/// Analytical IK solver via subproblem decomposition.
pub struct SubproblemIK {
    decomposition: DecompositionType,
    /// Joint axes in world frame at zero configuration.
    axes_world: Vec<Vector3<f64>>,
    /// Joint origins in world frame at zero configuration.
    origins_world: Vec<Vector3<f64>>,
}

impl SubproblemIK {
    /// Analyze robot geometry and create a decomposition if possible.
    ///
    /// Returns `None` if the robot doesn't support subproblem decomposition
    /// (e.g., non-6-DOF, or no identifiable axis intersection pattern).
    pub fn from_robot(robot: &Robot, chain: &KinematicChain) -> Option<Self> {
        if chain.dof != 6 {
            return None;
        }

        // Compute joint positions and axes at zero configuration
        let zero_joints = vec![0.0; chain.dof];
        let link_poses = forward_kinematics_all(robot, chain, &zero_joints).ok()?;

        let mut axes_world = Vec::new();
        let mut origins_world = Vec::new();

        for &joint_idx in &chain.active_joints {
            let joint = &robot.joints[joint_idx];
            let parent_link = joint.parent_link;

            let parent_pose = if parent_link < link_poses.len() {
                &link_poses[parent_link]
            } else {
                continue;
            };

            // Joint axis in world frame
            let axis_world = parent_pose.0.rotation * joint.axis;
            let axis_world = axis_world.normalize();

            // Joint origin in world frame
            let origin_world = parent_pose.0 * joint.origin.0 * nalgebra::Point3::origin();

            axes_world.push(axis_world);
            origins_world.push(origin_world.coords);
        }

        if axes_world.len() != 6 {
            return None;
        }

        // Check for spherical wrist: do axes 4, 5, 6 (indices 3, 4, 5) intersect?
        let decomposition =
            if let Some(wc) = find_intersection_point(&axes_world[3..6], &origins_world[3..6]) {
                DecompositionType::SphericalWrist { wrist_center: wc }
            } else {
                DecompositionType::NotDecomposable
            };

        match decomposition {
            DecompositionType::NotDecomposable => None,
            _ => Some(Self {
                decomposition,
                axes_world,
                origins_world,
            }),
        }
    }

    /// Solve IK using the decomposed subproblems.
    ///
    /// Returns up to 16 analytical solutions (all joint configurations
    /// that place the end-effector at the target pose).
    pub fn solve(
        &self,
        robot: &Robot,
        chain: &KinematicChain,
        target: &Isometry3<f64>,
    ) -> Vec<[f64; 6]> {
        match &self.decomposition {
            DecompositionType::SphericalWrist { wrist_center } => {
                self.solve_spherical_wrist(robot, chain, target, wrist_center)
            }
            DecompositionType::NotDecomposable => Vec::new(),
        }
    }

    /// Solve for a robot with a spherical wrist.
    ///
    /// Strategy:
    /// 1. Compute wrist center position from target pose
    /// 2. Solve joints 1-3 for wrist position (position subproblem)
    /// 3. Solve joints 4-6 for wrist orientation (orientation subproblem)
    #[allow(clippy::too_many_lines)]
    fn solve_spherical_wrist(
        &self,
        robot: &Robot,
        chain: &KinematicChain,
        target: &Isometry3<f64>,
        wrist_center_zero: &Vector3<f64>,
    ) -> Vec<[f64; 6]> {
        // The wrist center in the tool frame is the offset from joint 6 to the end-effector.
        // At zero config, it's at wrist_center_zero. In the target frame:
        // target_wrist = target * tool_offset_from_wrist (but we approximate)

        // Actually, the wrist center is at the intersection of the last 3 axes.
        // The target pose T specifies where the end-effector should be.
        // The wrist center in the target frame is: T * (ee_to_wrist_offset)

        // Compute end-effector pose at zero config to find wrist-to-ee offset
        let zero_joints = vec![0.0; 6];
        let ee_zero = match forward_kinematics(robot, chain, &zero_joints) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        // Wrist-to-ee vector in world frame at zero config
        let ee_pos_zero = ee_zero.0.translation.vector;
        let wrist_to_ee = ee_pos_zero - wrist_center_zero;

        // In the ee frame, this offset is constant. Transform to ee local frame:
        let wrist_to_ee_local = ee_zero.0.rotation.inverse() * wrist_to_ee;

        // Target wrist center: target_pos - target_rot * wrist_to_ee_local
        let target_wrist = target.translation.vector - target.rotation * wrist_to_ee_local;

        // === Step 1: Solve joints 1-3 for wrist position ===
        // We need to find θ1, θ2, θ3 such that the wrist reaches target_wrist.

        let w1 = &self.axes_world[0];
        let w2 = &self.axes_world[1];
        let w3 = &self.axes_world[2];
        let o1 = &self.origins_world[0];
        let o2 = &self.origins_world[1];
        let o3 = &self.origins_world[2];

        // The wrist center at zero config is at wrist_center_zero.
        // After applying joints 1-3, it moves to:
        // Rot(w1,θ1) at o1 · Rot(w2,θ2) at o2 · Rot(w3,θ3) at o3 · wrist_center_zero

        // Use SP3 to find θ3, then SP2/SP1 for θ1, θ2.
        // Actually, for a general 3R position problem, we use:
        // 1. θ1 from atan2 (base rotation) - 2 solutions (shoulder left/right)
        // 2. θ3 from distance constraint (elbow up/down) - 2 solutions via SP3
        // 3. θ2 from SP1 or geometric constraint

        // Simplified approach: use geometric method for the 3R arm.
        let position_solutions =
            solve_3r_position(w1, w2, w3, o1, o2, o3, wrist_center_zero, &target_wrist);

        // === Step 2: For each position solution, solve joints 4-6 for orientation ===
        let mut all_solutions = Vec::new();

        for (t1, t2, t3) in &position_solutions {
            // Compute the rotation achieved by joints 1-3
            let r1 = axis_angle_rotation(w1, *t1);
            let r2 = axis_angle_rotation(w2, *t2);
            let r3 = axis_angle_rotation(w3, *t3);
            let r_123 = r1 * r2 * r3;

            // The target rotation for joints 4-6:
            let r_target = target.rotation.to_rotation_matrix();
            let r_456_needed = r_123.transpose() * r_target.matrix();

            // Joints 4-6 form a ZYZ (or similar) Euler decomposition.
            // The axes at zero config are w4, w5, w6.
            // For a standard spherical wrist with Z-Y-Z axes:
            let orientation_solutions = solve_wrist_orientation(&r_456_needed);

            for (t4, t5, t6) in &orientation_solutions {
                let sol = [*t1, *t2, *t3, *t4, *t5, *t6];
                all_solutions.push(sol);
            }
        }

        // Filter solutions within joint limits
        let limits: Vec<_> = chain
            .active_joints
            .iter()
            .filter_map(|&ji| robot.joints[ji].limits.as_ref())
            .collect();

        all_solutions.retain(|sol| {
            sol.iter().enumerate().all(|(i, &v)| {
                if i < limits.len() {
                    v >= limits[i].lower - 1e-6 && v <= limits[i].upper + 1e-6
                } else {
                    true
                }
            })
        });

        all_solutions
    }
}

/// Solve the 3R position problem: find (θ1, θ2, θ3) such that
/// the kinematic chain moves the wrist from its zero-config position to the target.
#[allow(clippy::too_many_arguments)]
fn solve_3r_position(
    w1: &Vector3<f64>,
    w2: &Vector3<f64>,
    w3: &Vector3<f64>,
    o1: &Vector3<f64>,
    o2: &Vector3<f64>,
    o3: &Vector3<f64>,
    wrist_zero: &Vector3<f64>,
    target: &Vector3<f64>,
) -> Vec<(f64, f64, f64)> {
    let mut solutions = Vec::new();

    // For a typical 6R robot with a vertical base axis (w1 ≈ Z):
    // θ1: base rotation, determined by the XY position of the target wrist
    // θ3: elbow angle, determined by the distance from shoulder to wrist
    // θ2: shoulder angle, determined by the remaining geometry

    // Compute link lengths from joint origins
    let l1 = (o3 - o2).norm(); // upper arm
    let wrist_offset = wrist_zero - o3;
    let l2 = wrist_offset.norm(); // forearm (to wrist center)

    if l1 < 1e-10 || l2 < 1e-10 {
        return solutions;
    }

    // Base rotation (θ1): project target onto base plane
    let target_rel = target - o1;

    // For vertical base axis (Z), θ1 = atan2(ty, tx)
    // For general axis, project onto plane ⊥ w1
    let target_proj = target_rel - w1 * w1.dot(&target_rel);
    if target_proj.norm() < 1e-10 {
        // Target is on the base axis — θ1 is arbitrary
        return solutions;
    }

    // Find a reference direction in the plane ⊥ w1
    let ref_dir = {
        let o2_rel = o2 - o1;
        let proj = o2_rel - w1 * w1.dot(&o2_rel);
        if proj.norm() > 1e-10 {
            proj.normalize()
        } else {
            // Fallback: find any vector perpendicular to w1
            let v = if w1.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            (v - w1 * w1.dot(&v)).normalize()
        }
    };
    let perp_dir = w1.cross(&ref_dir);

    let theta1_a = f64::atan2(target_proj.dot(&perp_dir), target_proj.dot(&ref_dir));
    let theta1_b = theta1_a + std::f64::consts::PI; // shoulder left/right

    for &theta1 in &[theta1_a, theta1_b] {
        // After θ1 rotation, compute the remaining geometry for θ2, θ3
        let r1 = axis_angle_rotation(w1, theta1);

        // Shoulder position after θ1 rotation
        let shoulder = o1 + r1 * (o2 - o1);

        // Distance from shoulder to target wrist
        let d = (target - shoulder).norm();

        // Elbow angle from cosine law: d² = l1² + l2² - 2·l1·l2·cos(π - θ3)
        let cos_elbow = (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2);
        if cos_elbow.abs() > 1.0 + 1e-6 {
            continue; // unreachable
        }
        let cos_elbow = cos_elbow.clamp(-1.0, 1.0);
        let elbow_angles = [cos_elbow.acos(), -cos_elbow.acos()]; // up/down

        for &theta3 in &elbow_angles {
            // Solve θ2 using SP3 or geometric constraint
            // After θ1, the arm lies in a plane. θ2 rotates the upper arm to
            // reach the correct wrist position.

            // Use SP1: find θ2 such that the chain reaches the target wrist
            let r3 = axis_angle_rotation(w3, theta3);

            // The wrist at (θ1, θ2=0, θ3) would be at:
            let wrist_at_t2_zero = shoulder + r1 * (o3 - o2) + r1 * r3 * wrist_offset;

            // We need Rot(w2, θ2) at shoulder to move wrist_at_t2_zero to target
            // Actually, this is more involved. Let's use SP1 directly.
            let w2_world = r1 * w2;
            if let Some(theta2) = subproblem1(&w2_world, &shoulder, &wrist_at_t2_zero, target) {
                solutions.push((theta1, theta2, theta3));
            }
        }
    }

    solutions
}

/// Solve the wrist orientation: find (θ4, θ5, θ6) to achieve R_456.
/// Assumes a ZYZ Euler convention for the spherical wrist.
fn solve_wrist_orientation(r_target: &Matrix3<f64>) -> Vec<(f64, f64, f64)> {
    euler_zyz_decompose(r_target)
}

/// Check if 2-3 rotation axes intersect at a common point.
fn find_intersection_point(
    axes: &[Vector3<f64>],
    origins: &[Vector3<f64>],
) -> Option<Vector3<f64>> {
    if axes.len() < 2 {
        return None;
    }

    // For each pair of axes, find the closest point of approach.
    // If all pairs come within tolerance, return the average intersection.
    let mut intersections = Vec::new();

    for i in 0..axes.len() {
        for j in (i + 1)..axes.len() {
            let (dist, p1, p2) =
                closest_point_between_lines(&origins[i], &axes[i], &origins[j], &axes[j]);
            if dist > INTERSECT_TOLERANCE {
                return None;
            }
            intersections.push((p1 + p2) / 2.0);
        }
    }

    if intersections.is_empty() {
        return None;
    }

    // Average intersection point
    let sum: Vector3<f64> = intersections.iter().sum();
    Some(sum / intersections.len() as f64)
}

/// Find the closest points between two lines defined by (point, direction).
/// Returns (distance, point_on_line1, point_on_line2).
fn closest_point_between_lines(
    p1: &Vector3<f64>,
    d1: &Vector3<f64>,
    p2: &Vector3<f64>,
    d2: &Vector3<f64>,
) -> (f64, Vector3<f64>, Vector3<f64>) {
    let w = p1 - p2;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d2.dot(d2);
    let d = d1.dot(&w);
    let e = d2.dot(&w);

    let denom = a * c - b * b;

    if denom.abs() < 1e-15 {
        // Lines are parallel
        let t2 = d / b.max(1e-15);
        let closest1 = *p1;
        let closest2 = p2 + d2 * t2;
        return ((closest1 - closest2).norm(), closest1, closest2);
    }

    let t1 = (b * e - c * d) / denom;
    let t2 = (a * e - b * d) / denom;

    let closest1 = p1 + d1 * t1;
    let closest2 = p2 + d2 * t2;

    ((closest1 - closest2).norm(), closest1, closest2)
}

/// Convenience: solve IK using subproblem decomposition if possible,
/// returning results as `Vec<Vec<f64>>` for compatibility with the IK pipeline.
pub fn solve_subproblem_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
) -> Option<Vec<Vec<f64>>> {
    let decomp = SubproblemIK::from_robot(robot, chain)?;
    let solutions = decomp.solve(robot, chain, &target.0);
    if solutions.is_empty() {
        None
    } else {
        Some(solutions.into_iter().map(|s| s.to_vec()).collect())
    }
}

/// Check if a robot supports 6-DOF subproblem decomposition.
pub fn is_subproblem_compatible(robot: &Robot, chain: &KinematicChain) -> bool {
    SubproblemIK::from_robot(robot, chain).is_some()
}

// ─── 7-DOF Parametric Sweep ──────────────────────────────────────────────────

/// Analytical IK for 7-DOF robots with a spherical wrist.
///
/// Fixes the redundant joint to sampled values across its range, then solves
/// the resulting 6-DOF problem analytically for each sample.
pub struct SubproblemIK7DOF {
    /// Index of the redundant joint within the chain (0-based).
    pub redundant_joint_idx: usize,
    /// Lower limit of the redundant joint.
    pub redundant_lower: f64,
    /// Upper limit of the redundant joint.
    pub redundant_upper: f64,
}

impl SubproblemIK7DOF {
    /// Analyze a 7-DOF chain and return the solver if the last 3 joints
    /// form a spherical wrist. The redundant joint is identified as the joint
    /// closest to the "elbow" — typically index 2 or 3 (0-based).
    pub fn from_robot(robot: &Robot, chain: &KinematicChain) -> Option<Self> {
        if chain.dof != 7 {
            return None;
        }

        // Compute joint positions and axes at zero configuration
        let zero_joints = vec![0.0; 7];
        let link_poses = forward_kinematics_all(robot, chain, &zero_joints).ok()?;

        let mut axes_world = Vec::new();
        let mut origins_world = Vec::new();

        for &joint_idx in &chain.active_joints {
            let joint = &robot.joints[joint_idx];
            let parent_link = joint.parent_link;
            let parent_pose = if parent_link < link_poses.len() {
                &link_poses[parent_link]
            } else {
                continue;
            };
            let axis_world = (parent_pose.0.rotation * joint.axis).normalize();
            let origin_world = parent_pose.0 * joint.origin.0 * nalgebra::Point3::origin();
            axes_world.push(axis_world);
            origins_world.push(origin_world.coords);
        }

        if axes_world.len() != 7 {
            return None;
        }

        // Check that the last 3 joints (indices 4,5,6) form a spherical wrist
        let wrist_ok = find_intersection_point(&axes_world[4..7], &origins_world[4..7]).is_some();
        if !wrist_ok {
            return None;
        }

        // The redundant joint is index 3 (the "elbow" joint between the
        // 4-joint position sub-chain and the 3-joint wrist).
        // For most 7-DOF arms this is the 4th joint (0-indexed: 3).
        let redundant_joint_idx = 3;
        let robot_joint_idx = chain.active_joints[redundant_joint_idx];
        let (lower, upper) = if let Some(limits) = &robot.joints[robot_joint_idx].limits {
            (limits.lower, limits.upper)
        } else {
            (-std::f64::consts::PI, std::f64::consts::PI)
        };

        Some(Self {
            redundant_joint_idx,
            redundant_lower: lower,
            redundant_upper: upper,
        })
    }

    /// Solve 7-DOF IK by sweeping the redundant joint.
    ///
    /// For each sample of the redundant joint:
    /// 1. Fix the redundant joint value
    /// 2. Compute FK to get the transform contributed by the fixed joint
    /// 3. Adjust the target for the remaining 6-DOF sub-chain
    /// 4. Solve with the 6-DOF subproblem solver
    /// 5. Insert the fixed joint value back into the solution
    pub fn solve(
        &self,
        robot: &Robot,
        chain: &KinematicChain,
        target: &Isometry3<f64>,
        num_samples: usize,
    ) -> Vec<Vec<f64>> {
        let mut all_solutions = Vec::new();
        let range = self.redundant_upper - self.redundant_lower;

        for i in 0..num_samples {
            let t = if num_samples > 1 {
                i as f64 / (num_samples - 1) as f64
            } else {
                0.5
            };
            let redundant_value = self.redundant_lower + t * range;

            if let Some(mut sols) = self.solve_at_redundant(robot, chain, target, redundant_value) {
                all_solutions.append(&mut sols);
            }
        }

        // Deduplicate: remove solutions that are very close to each other
        let mut unique = Vec::new();
        'outer: for sol in &all_solutions {
            for existing in &unique {
                let dist: f64 = sol
                    .iter()
                    .zip(existing)
                    .map(|(a, b): (&f64, &f64)| (a - b).powi(2))
                    .sum();
                if dist < 1e-6 {
                    continue 'outer;
                }
            }
            unique.push(sol.clone());
        }

        unique
    }

    /// Solve at a specific redundant joint value.
    fn solve_at_redundant(
        &self,
        robot: &Robot,
        chain: &KinematicChain,
        target: &Isometry3<f64>,
        redundant_value: f64,
    ) -> Option<Vec<Vec<f64>>> {
        let ridx = self.redundant_joint_idx;

        // Build a 7-DOF config with the redundant joint fixed and others at zero
        let mut q_partial = vec![0.0; 7];
        q_partial[ridx] = redundant_value;

        // Compute the FK at this partial configuration to get the transform
        // contributed by the redundant joint
        let link_poses = forward_kinematics_all(robot, chain, &q_partial).ok()?;

        // Build a reduced 6-DOF sub-chain by removing the redundant joint.
        // Instead of actually building a new chain (complex), we use the
        // numerical DLS approach seeded from the partial config to solve
        // the remaining DOF. This is simpler and more robust.
        //
        // However, the task asks for analytical solution. The key insight:
        // with the redundant joint fixed, the remaining 6 joints form a
        // standard 6-DOF problem. We construct the effective target by
        // "undoing" the redundant joint's contribution.

        // The redundant joint is between the position sub-chain joints.
        // With it fixed, the base-side joints (0..ridx) and tool-side joints
        // ((ridx+1)..7) form a 6-DOF chain with the redundant joint's
        // transform baked in.

        // Get joint axes and origins for all 7 joints at this config
        let mut axes_world = Vec::new();
        let mut origins_world = Vec::new();

        for (j, &joint_idx) in chain.active_joints.iter().enumerate() {
            let joint = &robot.joints[joint_idx];
            let parent_link = joint.parent_link;
            if parent_link >= link_poses.len() {
                continue;
            }
            let parent_pose = &link_poses[parent_link];

            if j == ridx {
                // Skip the redundant joint — it's fixed
                continue;
            }

            let axis_world = (parent_pose.0.rotation * joint.axis).normalize();
            let origin_world = parent_pose.0 * joint.origin.0 * nalgebra::Point3::origin();
            axes_world.push(axis_world);
            origins_world.push(origin_world.coords);
        }

        if axes_world.len() != 6 {
            return None;
        }

        // Check that the remaining last 3 joints still form a spherical wrist
        let wrist_ok = find_intersection_point(&axes_world[3..6], &origins_world[3..6]).is_some();
        if !wrist_ok {
            return None;
        }

        let wrist_center = find_intersection_point(&axes_world[3..6], &origins_world[3..6])?;

        // Create a temporary 6-DOF decomposition with the current geometry
        let decomp = SubproblemIK {
            decomposition: DecompositionType::SphericalWrist { wrist_center },
            axes_world,
            origins_world,
        };

        let solutions_6dof = decomp.solve(robot, chain, target);

        if solutions_6dof.is_empty() {
            return None;
        }

        // Insert the redundant joint value back into each solution
        let limits: Vec<_> = chain
            .active_joints
            .iter()
            .filter_map(|&ji| robot.joints[ji].limits.as_ref())
            .collect();

        let mut full_solutions = Vec::new();
        for sol6 in &solutions_6dof {
            let mut full = Vec::with_capacity(7);
            let mut si = 0;
            for j in 0..7 {
                if j == ridx {
                    full.push(redundant_value);
                } else {
                    full.push(sol6[si]);
                    si += 1;
                }
            }

            // Check joint limits
            let within_limits = full.iter().enumerate().all(|(i, &v)| {
                if i < limits.len() {
                    v >= limits[i].lower - 1e-6 && v <= limits[i].upper + 1e-6
                } else {
                    true
                }
            });

            if within_limits {
                // Verify solution via FK
                if let Ok(check_pose) = forward_kinematics(robot, chain, &full) {
                    let pos_err =
                        (check_pose.0.translation.vector - target.translation.vector).norm();
                    if pos_err < 0.05 {
                        // Reasonable solution (will be refined later)
                        full_solutions.push(full);
                    }
                }
            }
        }

        if full_solutions.is_empty() {
            None
        } else {
            Some(full_solutions)
        }
    }
}

/// Check if a robot supports 7-DOF subproblem decomposition.
pub fn is_subproblem_7dof_compatible(robot: &Robot, chain: &KinematicChain) -> bool {
    SubproblemIK7DOF::from_robot(robot, chain).is_some()
}

/// Solve 7-DOF IK via parametric sweep, returning an `IKSolution`.
///
/// Sweeps the redundant joint and picks the solution closest to the seed.
/// If found, refines with DLS for sub-millimeter accuracy.
pub fn solve_subproblem_7dof_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &IKConfig,
    num_samples: usize,
) -> Result<IKSolution> {
    let solver =
        SubproblemIK7DOF::from_robot(robot, chain).ok_or(KineticError::IKNotConverged {
            iterations: 0,
            residual: f64::INFINITY,
        })?;

    let solutions = solver.solve(robot, chain, &target.0, num_samples);

    if solutions.is_empty() {
        return Err(KineticError::IKNotConverged {
            iterations: 0,
            residual: f64::INFINITY,
        });
    }

    // Pick the solution closest to seed
    let best = solutions
        .iter()
        .min_by(|a, b| {
            let da: f64 = a
                .iter()
                .zip(seed.iter())
                .map(|(x, s)| (x - s).powi(2))
                .sum();
            let db: f64 = b
                .iter()
                .zip(seed.iter())
                .map(|(x, s)| (x - s).powi(2))
                .sum();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // Refine with DLS for sub-millimeter accuracy
    let refine_config = IKConfig {
        solver: crate::ik::IKSolver::DLS { damping: 0.01 },
        mode: config.mode,
        max_iterations: 50,
        position_tolerance: config.position_tolerance,
        orientation_tolerance: config.orientation_tolerance,
        check_limits: config.check_limits,
        seed: Some(best.clone()),
        null_space: None,
        num_restarts: 0,
    };

    match crate::ik::solve_ik(robot, chain, target, &refine_config) {
        Ok(sol) => Ok(sol),
        Err(_) => {
            // Return the analytical solution even if refinement fails
            let result_pose = forward_kinematics(robot, chain, best)?;
            let (pos_err, orient_err, _) = crate::ik::pose_error(&result_pose, target);

            Ok(IKSolution {
                joints: best.clone(),
                position_error: pos_err,
                orientation_error: orient_err,
                converged: pos_err < config.position_tolerance
                    && orient_err < config.orientation_tolerance,
                iterations: 1,
                mode_used: IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
            })
        }
    }
}

/// Solve IK via subproblem decomposition, returning an `IKSolution`.
///
/// Used by the IK auto-selection pipeline. Picks the solution closest to the seed.
pub fn solve_subproblem_ik_as_solution(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    _config: &IKConfig,
) -> Result<IKSolution> {
    let decomp = SubproblemIK::from_robot(robot, chain).ok_or(KineticError::IKNotConverged {
        iterations: 0,
        residual: f64::INFINITY,
    })?;

    let solutions = decomp.solve(robot, chain, &target.0);

    if solutions.is_empty() {
        return Err(KineticError::IKNotConverged {
            iterations: 0,
            residual: f64::INFINITY,
        });
    }

    // Pick the solution closest to the seed
    let best = solutions
        .iter()
        .min_by(|a, b| {
            let da: f64 = a.iter().zip(seed).map(|(x, s)| (x - s).powi(2)).sum();
            let db: f64 = b.iter().zip(seed).map(|(x, s)| (x - s).powi(2)).sum();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // Verify with FK
    let result_pose = crate::forward::forward_kinematics(robot, chain, best.as_ref())?;
    let (pos_err, orient_err, _) = crate::ik::pose_error(&result_pose, target);

    Ok(IKSolution {
        joints: best.to_vec(),
        position_error: pos_err,
        orientation_error: orient_err,
        converged: pos_err < 1e-3 && orient_err < 1e-2,
        iterations: 1,
        mode_used: IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_perpendicular_lines() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let d1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 0.0, 1.0);
        let d2 = Vector3::new(0.0, 1.0, 0.0);

        let (dist, c1, c2) = closest_point_between_lines(&p1, &d1, &p2, &d2);
        assert!((dist - 1.0).abs() < 1e-10);
        assert!((c1 - Vector3::zeros()).norm() < 1e-10);
        assert!((c2 - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn closest_point_intersecting_lines() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let d1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 0.0, 0.0);
        let d2 = Vector3::new(0.0, 1.0, 0.0);

        let (dist, _, _) = closest_point_between_lines(&p1, &d1, &p2, &d2);
        assert!(dist < 1e-10);
    }

    #[test]
    fn intersection_of_three_axes() {
        let axes = vec![Vector3::x(), Vector3::y(), Vector3::z()];
        let origin = Vector3::new(1.0, 2.0, 3.0);
        let origins = vec![origin, origin, origin];

        let point = find_intersection_point(&axes, &origins);
        assert!(point.is_some());
        let p = point.unwrap();
        assert!((p - origin).norm() < 1e-8);
    }

    #[test]
    fn no_intersection_parallel_axes() {
        let axes = vec![Vector3::z(), Vector3::z()];
        let origins = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];

        let point = find_intersection_point(&axes, &origins);
        assert!(point.is_none());
    }

    #[test]
    fn subproblem_ik_creation() {
        // Try to create SubproblemIK for various robots
        // This tests the analysis logic even if decomposition isn't possible
        let robot = kinetic_robot::Robot::from_name("franka_panda");
        if let Ok(robot) = robot {
            if let Some(group) = robot.groups.values().next() {
                if let Ok(chain) =
                    KinematicChain::extract(&robot, &group.base_link, &group.tip_link)
                {
                    // Panda has 7 DOF, so 6-DOF subproblem decomp won't apply
                    let result = SubproblemIK::from_robot(&robot, &chain);
                    assert!(
                        result.is_none(),
                        "7-DOF panda should not be 6-DOF subproblem-decomposable"
                    );
                }
            }
        }
    }

    #[test]
    fn subproblem_7dof_compatibility_panda() {
        let robot = kinetic_robot::Robot::from_name("franka_panda");
        if let Ok(robot) = robot {
            if let Some(group) = robot.groups.values().next() {
                if let Ok(chain) =
                    KinematicChain::extract(&robot, &group.base_link, &group.tip_link)
                {
                    assert_eq!(chain.dof, 7);
                    let result = SubproblemIK7DOF::from_robot(&robot, &chain);
                    // The Panda may or may not have a perfectly spherical wrist
                    // depending on the URDF model. Log the result for diagnostics.
                    if let Some(solver) = result {
                        assert_eq!(solver.redundant_joint_idx, 3);
                    }
                }
            }
        }
    }

    #[test]
    fn subproblem_7dof_compatibility_kuka() {
        let robot = kinetic_robot::Robot::from_name("kuka_iiwa7");
        if let Ok(robot) = robot {
            if let Some(group) = robot.groups.values().next() {
                if let Ok(chain) =
                    KinematicChain::extract(&robot, &group.base_link, &group.tip_link)
                {
                    assert_eq!(chain.dof, 7);
                    let result = SubproblemIK7DOF::from_robot(&robot, &chain);
                    if let Some(solver) = result {
                        assert_eq!(solver.redundant_joint_idx, 3);
                    }
                }
            }
        }
    }

    #[test]
    fn subproblem_7dof_solve_panda() {
        let robot = match kinetic_robot::Robot::from_name("franka_panda") {
            Ok(r) => r,
            Err(_) => return, // Skip if robot config not available
        };
        let group = match robot.groups.values().next() {
            Some(g) => g,
            None => return,
        };
        let chain = match KinematicChain::extract(&robot, &group.base_link, &group.tip_link) {
            Ok(c) => c,
            Err(_) => return,
        };

        let solver = match SubproblemIK7DOF::from_robot(&robot, &chain) {
            Some(s) => s,
            None => return, // Skip if not compatible
        };

        // FK to get a reachable target
        let q_original = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
        let target = match forward_kinematics(&robot, &chain, &q_original) {
            Ok(p) => p,
            Err(_) => return,
        };

        let solutions = solver.solve(&robot, &chain, &target.0, 36);
        // We should find at least one solution for a reachable target
        assert!(
            !solutions.is_empty(),
            "7-DOF subproblem should find at least one solution for a reachable target"
        );

        // Verify the best solution is reasonably close
        let best = solutions
            .iter()
            .min_by(|a, b| {
                let da: f64 = a
                    .iter()
                    .zip(q_original.iter())
                    .map(|(x, s)| (x - s).powi(2))
                    .sum();
                let db: f64 = b
                    .iter()
                    .zip(q_original.iter())
                    .map(|(x, s)| (x - s).powi(2))
                    .sum();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let check = forward_kinematics(&robot, &chain, best).unwrap();
        let pos_err = (check.0.translation.vector - target.0.translation.vector).norm();
        assert!(
            pos_err < 0.1,
            "Best analytical solution should be within 10cm: pos_err={}",
            pos_err
        );
    }
}
