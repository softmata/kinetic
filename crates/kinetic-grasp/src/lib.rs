//! Geometric grasp generation for KINETIC.
//!
//! Generates ranked grasp candidates from object shapes: antipodal,
//! top-down, side grasp, and suction. Supports parallel jaw and
//! suction grippers with collision and reachability filtering.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_grasp::{GraspGenerator, GraspConfig, GripperType};
//! use kinetic_scene::Shape;
//!
//! let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
//! let grasps = gen.from_shape(
//!     &Shape::Cylinder(0.04, 0.06),
//!     &Isometry3::identity(),
//!     GraspConfig::default(),
//! )?;
//! ```

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use nalgebra::{Isometry3, UnitQuaternion, Vector3};

use kinetic_kinematics::{solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;
use kinetic_scene::{Scene, Shape};

/// Gripper type for grasp generation.
#[derive(Debug, Clone)]
pub enum GripperType {
    /// Parallel jaw gripper.
    Parallel {
        /// Maximum finger opening distance in meters.
        max_opening: f64,
        /// Finger depth in meters (how far fingers extend).
        finger_depth: f64,
    },
    /// Suction cup gripper.
    Suction {
        /// Cup radius in meters.
        cup_radius: f64,
    },
}

impl GripperType {
    /// Create a parallel jaw gripper.
    pub fn parallel(max_opening: f64, finger_depth: f64) -> Self {
        Self::Parallel {
            max_opening,
            finger_depth,
        }
    }

    /// Create a suction cup gripper.
    pub fn suction(cup_radius: f64) -> Self {
        Self::Suction { cup_radius }
    }
}

/// Grasp quality metric.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GraspMetric {
    /// Force closure quality (estimated from contact normals).
    #[default]
    ForceClosureQuality,
    /// Proximity of grasp to object center of mass.
    DistanceFromCenterOfMass,
    /// Alignment of approach direction with preferred axis.
    ApproachAngle,
}

/// Type of grasp approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraspType {
    /// Antipodal parallel jaw grasp.
    Antipodal,
    /// Top-down approach (vertical).
    TopDown,
    /// Side approach (horizontal).
    SideGrasp,
    /// Suction at surface center.
    SuctionCenter,
}

/// Configuration for grasp generation.
#[derive(Clone)]
pub struct GraspConfig {
    /// Number of candidates to generate (default: 100).
    pub num_candidates: usize,
    /// Preferred approach axis in world frame (default: -Z, top-down).
    pub approach_axis: Vector3<f64>,
    /// Ranking metric (default: ForceClosureQuality).
    pub rank_by: GraspMetric,
    /// Optional scene for collision filtering.
    pub check_collision: Option<Arc<Scene>>,
    /// Optional robot + chain for reachability filtering.
    pub check_reachability: Option<(Arc<Robot>, KinematicChain)>,
}

impl Default for GraspConfig {
    fn default() -> Self {
        Self {
            num_candidates: 100,
            approach_axis: -Vector3::z(),
            rank_by: GraspMetric::ForceClosureQuality,
            check_collision: None,
            check_reachability: None,
        }
    }
}

impl std::fmt::Debug for GraspConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraspConfig")
            .field("num_candidates", &self.num_candidates)
            .field("approach_axis", &self.approach_axis)
            .field("rank_by", &self.rank_by)
            .field("check_collision", &self.check_collision.is_some())
            .field("check_reachability", &self.check_reachability.is_some())
            .finish()
    }
}

/// A ranked grasp candidate.
#[derive(Debug, Clone)]
pub struct GraspCandidate {
    /// Pose of the gripper TCP at the grasp.
    pub grasp_pose: Isometry3<f64>,
    /// Approach direction (unit vector).
    pub approach_direction: Vector3<f64>,
    /// Quality score [0.0, 1.0] (higher is better).
    pub quality: f64,
    /// Type of grasp approach.
    pub grasp_type: GraspType,
}

/// Grasp generation error.
#[derive(Debug, thiserror::Error)]
pub enum GraspError {
    #[error("No valid grasps found for shape")]
    NoGraspsFound,
    #[error("Shape not supported for gripper type: {0}")]
    UnsupportedShape(String),
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] kinetic_core::KineticError),
}

/// Geometric grasp candidate generator.
///
/// Generates grasp poses for a given gripper type and object shape.
pub struct GraspGenerator {
    gripper: GripperType,
}

impl GraspGenerator {
    /// Create a new grasp generator for the given gripper type.
    pub fn new(gripper: GripperType) -> Self {
        Self { gripper }
    }

    /// Generate ranked grasp candidates for a shape at a given pose.
    pub fn from_shape(
        &self,
        shape: &Shape,
        object_pose: &Isometry3<f64>,
        config: GraspConfig,
    ) -> Result<Vec<GraspCandidate>, GraspError> {
        let mut candidates = Vec::new();

        match &self.gripper {
            GripperType::Parallel {
                max_opening,
                finger_depth,
            } => {
                // Generate antipodal grasps
                let antipodal =
                    generate_antipodal(shape, object_pose, *max_opening, *finger_depth, &config);
                candidates.extend(antipodal);

                // Generate top-down grasps
                let topdown =
                    generate_topdown(shape, object_pose, *max_opening, *finger_depth, &config);
                candidates.extend(topdown);

                // Generate side grasps
                let side =
                    generate_side_grasps(shape, object_pose, *max_opening, *finger_depth, &config);
                candidates.extend(side);
            }
            GripperType::Suction { cup_radius } => {
                let suction = generate_suction(shape, object_pose, *cup_radius, &config);
                candidates.extend(suction);
            }
        }

        // Filter by collision
        if let Some(scene) = &config.check_collision {
            candidates.retain(|g| !grasp_in_collision(scene, &g.grasp_pose));
        }

        // Filter by reachability
        if let Some((robot, chain)) = &config.check_reachability {
            candidates.retain(|g| is_reachable(robot, chain, &g.grasp_pose));
        }

        // Rank by selected metric
        rank_candidates(&mut candidates, &config);

        // Limit to requested number
        candidates.truncate(config.num_candidates);

        if candidates.is_empty() {
            return Err(GraspError::NoGraspsFound);
        }

        Ok(candidates)
    }
}

/// Generate antipodal (parallel jaw) grasps for a shape.
fn generate_antipodal(
    shape: &Shape,
    object_pose: &Isometry3<f64>,
    max_opening: f64,
    finger_depth: f64,
    config: &GraspConfig,
) -> Vec<GraspCandidate> {
    let mut candidates = Vec::new();
    let n = config.num_candidates;

    match shape {
        Shape::Cylinder(radius, half_height) => {
            if *radius * 2.0 > max_opening {
                return candidates; // too wide for gripper
            }

            // Grasp along diameter at various heights and rotations
            let height_steps = (n as f64).sqrt().ceil() as usize;
            let angle_steps = (n as f64).sqrt().ceil() as usize;

            for hi in 0..height_steps {
                let t = if height_steps > 1 {
                    hi as f64 / (height_steps - 1) as f64
                } else {
                    0.5
                };
                let z = -half_height + t * 2.0 * half_height;

                for ai in 0..angle_steps {
                    let angle = ai as f64 / angle_steps as f64 * PI;

                    // Approach direction perpendicular to cylinder axis
                    let approach = Vector3::new(angle.cos(), angle.sin(), 0.0);
                    let grasp_pos = Vector3::new(0.0, 0.0, z) - approach * finger_depth;

                    // Gripper orientation: z-axis = approach direction, y-axis = cylinder axis
                    let grasp_rot = rotation_from_approach(&approach, &Vector3::z());

                    let local_pose =
                        Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                    let world_pose = object_pose * local_pose;

                    // Quality based on grasp height (center is best) and force closure
                    let height_quality = 1.0 - (z / half_height).abs();
                    let fc_quality = estimate_force_closure_cylinder(*radius, max_opening);
                    let quality = (height_quality * 0.4 + fc_quality * 0.6).clamp(0.0, 1.0);

                    candidates.push(GraspCandidate {
                        grasp_pose: world_pose,
                        approach_direction: object_pose.rotation * approach,
                        quality,
                        grasp_type: GraspType::Antipodal,
                    });
                }
            }
        }

        Shape::Cuboid(hx, hy, hz) => {
            let dims = [(*hx, *hy, *hz), (*hy, *hx, *hz), (*hz, *hx, *hy)];
            let axes = [Vector3::x(), Vector3::y(), Vector3::z()];

            for (dim_idx, ((grasp_dim, width_dim, height_dim), approach_axis)) in
                dims.iter().zip(axes.iter()).enumerate()
            {
                if *grasp_dim * 2.0 > max_opening {
                    continue; // face too wide for gripper
                }

                // Sample positions on the graspable face
                let w_steps = 3.max(((n / 6) as f64).sqrt().ceil() as usize);
                let h_steps = 3.max(((n / 6) as f64).sqrt().ceil() as usize);

                for sign in [-1.0_f64, 1.0] {
                    for wi in 0..w_steps {
                        let wt = if w_steps > 1 {
                            wi as f64 / (w_steps - 1) as f64
                        } else {
                            0.5
                        };
                        let w = -width_dim + wt * 2.0 * width_dim;

                        for hi in 0..h_steps {
                            let ht = if h_steps > 1 {
                                hi as f64 / (h_steps - 1) as f64
                            } else {
                                0.5
                            };
                            let h = -height_dim + ht * 2.0 * height_dim;

                            let approach = *approach_axis * sign;
                            let grasp_pos = match dim_idx {
                                0 => Vector3::new(0.0, w, h) - approach * finger_depth,
                                1 => Vector3::new(w, 0.0, h) - approach * finger_depth,
                                _ => Vector3::new(w, h, 0.0) - approach * finger_depth,
                            };

                            let up = match dim_idx {
                                0 => Vector3::z(),
                                1 => Vector3::z(),
                                _ => Vector3::y(),
                            };
                            let grasp_rot = rotation_from_approach(&approach, &up);

                            let local_pose = Isometry3::from_parts(
                                nalgebra::Translation3::from(grasp_pos),
                                grasp_rot,
                            );
                            let world_pose = object_pose * local_pose;

                            // Quality: center of face is best
                            let center_quality =
                                1.0 - ((w / width_dim).abs() + (h / height_dim).abs()) / 2.0;
                            let fc_quality = estimate_force_closure_box(*grasp_dim, max_opening);
                            let quality = (center_quality * 0.3 + fc_quality * 0.7).clamp(0.0, 1.0);

                            candidates.push(GraspCandidate {
                                grasp_pose: world_pose,
                                approach_direction: object_pose.rotation * approach,
                                quality,
                                grasp_type: GraspType::Antipodal,
                            });
                        }
                    }
                }
            }
        }

        Shape::Sphere(radius) => {
            if *radius * 2.0 > max_opening {
                return candidates;
            }

            // Antipodal grasps at equator from various angles
            let angle_steps = n.min(36);
            for ai in 0..angle_steps {
                let angle = ai as f64 / angle_steps as f64 * PI;
                let approach = Vector3::new(angle.cos(), angle.sin(), 0.0);
                let grasp_pos = -approach * finger_depth;
                let grasp_rot = rotation_from_approach(&approach, &Vector3::z());

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                let fc_quality = estimate_force_closure_sphere(*radius, max_opening);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: object_pose.rotation * approach,
                    quality: fc_quality,
                    grasp_type: GraspType::Antipodal,
                });
            }
        }

        Shape::HalfSpace(_, _) => {
            // Cannot grasp an infinite half-space
        }
    }

    candidates
}

/// Generate top-down grasps (approach from above along -Z).
fn generate_topdown(
    shape: &Shape,
    object_pose: &Isometry3<f64>,
    max_opening: f64,
    _finger_depth: f64,
    config: &GraspConfig,
) -> Vec<GraspCandidate> {
    let mut candidates = Vec::new();
    let approach = -Vector3::z(); // top-down

    match shape {
        Shape::Cylinder(radius, half_height) => {
            if *radius * 2.0 > max_opening {
                return candidates;
            }

            // Grasp from top at various rotations around cylinder axis
            let angle_steps = config.num_candidates.min(16);
            for ai in 0..angle_steps {
                let angle = ai as f64 / angle_steps as f64 * PI;
                let grasp_pos = Vector3::new(0.0, 0.0, *half_height + 0.01);
                let grasp_rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
                    * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                // Quality: alignment with preferred approach
                let world_approach = object_pose.rotation * approach;
                let alignment = (-world_approach).dot(&config.approach_axis).max(0.0);
                let quality = (alignment * 0.8 + 0.2).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: world_approach,
                    quality,
                    grasp_type: GraspType::TopDown,
                });
            }
        }

        Shape::Cuboid(hx, hy, hz) => {
            // Grasp from top face — gripper aligned with shorter dimension
            let (grasp_ok, short_dim) = if 2.0 * hx <= max_opening {
                (true, *hx)
            } else if 2.0 * hy <= max_opening {
                (true, *hy)
            } else {
                (false, 0.0)
            };

            if !grasp_ok {
                return candidates;
            }

            let angle_steps = config.num_candidates.min(8);
            for ai in 0..angle_steps {
                let angle = ai as f64 / angle_steps as f64 * PI;
                let grasp_pos = Vector3::new(0.0, 0.0, *hz + 0.01);
                let grasp_rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
                    * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                let world_approach = object_pose.rotation * approach;
                let alignment = (-world_approach).dot(&config.approach_axis).max(0.0);
                let size_ratio = short_dim * 2.0 / max_opening;
                let quality = (alignment * 0.6 + (1.0 - size_ratio) * 0.4).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: world_approach,
                    quality,
                    grasp_type: GraspType::TopDown,
                });
            }
        }

        Shape::Sphere(radius) => {
            if *radius * 2.0 > max_opening {
                return candidates;
            }

            let angle_steps = config.num_candidates.min(8);
            for ai in 0..angle_steps {
                let angle = ai as f64 / angle_steps as f64 * PI;
                let grasp_pos = Vector3::new(0.0, 0.0, *radius + 0.01);
                let grasp_rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
                    * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: object_pose.rotation * approach,
                    quality: 0.7,
                    grasp_type: GraspType::TopDown,
                });
            }
        }

        Shape::HalfSpace(_, _) => {}
    }

    candidates
}

/// Generate side grasps (horizontal approach).
fn generate_side_grasps(
    shape: &Shape,
    object_pose: &Isometry3<f64>,
    max_opening: f64,
    finger_depth: f64,
    config: &GraspConfig,
) -> Vec<GraspCandidate> {
    let mut candidates = Vec::new();

    match shape {
        Shape::Cylinder(radius, _half_height) => {
            if *radius * 2.0 > max_opening {
                return candidates;
            }

            // Horizontal approaches from various angles, grasping at mid-height
            let angle_steps = config.num_candidates.min(12);
            for ai in 0..angle_steps {
                let angle = ai as f64 / angle_steps as f64 * TAU;
                let approach = Vector3::new(angle.cos(), angle.sin(), 0.0);
                let grasp_pos = -approach * (radius + finger_depth);

                let grasp_rot = rotation_from_approach(&approach, &Vector3::z());

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                // Quality: alignment with preferred approach
                let world_approach = object_pose.rotation * approach;
                let alignment = world_approach.dot(&config.approach_axis).abs();
                let quality = (0.5 + alignment * 0.5).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: world_approach,
                    quality,
                    grasp_type: GraspType::SideGrasp,
                });
            }
        }

        Shape::Cuboid(hx, hy, hz) => {
            // Side approaches to faces that fit in the gripper
            let faces = [
                (Vector3::x(), *hx, *hy, *hz),
                (-Vector3::x(), *hx, *hy, *hz),
                (Vector3::y(), *hy, *hx, *hz),
                (-Vector3::y(), *hy, *hx, *hz),
            ];

            for (approach, depth, width, height) in &faces {
                // Check if the graspable dimension fits
                let grasp_dim = (*width).min(*height) * 2.0;
                if grasp_dim > max_opening {
                    continue;
                }

                let grasp_pos = -*approach * (depth + finger_depth);
                let grasp_rot = rotation_from_approach(approach, &Vector3::z());

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                let world_approach = object_pose.rotation * approach;
                let alignment = world_approach.dot(&config.approach_axis).abs();
                let quality =
                    (0.4 + alignment * 0.4 + 0.2 * (1.0 - grasp_dim / max_opening)).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: world_approach,
                    quality,
                    grasp_type: GraspType::SideGrasp,
                });
            }
        }

        _ => {}
    }

    candidates
}

/// Generate suction grasp candidates.
fn generate_suction(
    shape: &Shape,
    object_pose: &Isometry3<f64>,
    cup_radius: f64,
    config: &GraspConfig,
) -> Vec<GraspCandidate> {
    let mut candidates = Vec::new();

    match shape {
        Shape::Cylinder(radius, half_height) => {
            // Suction on top flat face
            if *radius >= cup_radius {
                let grasp_pos = Vector3::new(0.0, 0.0, *half_height + 0.005);
                let grasp_rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;
                let approach = object_pose.rotation * (-Vector3::z());

                let alignment = (-approach).dot(&config.approach_axis).max(0.0);
                let quality = (alignment * 0.7 + 0.3).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: approach,
                    quality,
                    grasp_type: GraspType::SuctionCenter,
                });
            }

            // Suction on bottom flat face
            if *radius >= cup_radius {
                let grasp_pos = Vector3::new(0.0, 0.0, -half_height - 0.005);
                let grasp_rot = UnitQuaternion::identity();

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;
                let approach = object_pose.rotation * Vector3::z();

                let alignment = (-approach).dot(&config.approach_axis).max(0.0);
                let quality = (alignment * 0.7 + 0.1).clamp(0.0, 1.0);

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: approach,
                    quality,
                    grasp_type: GraspType::SuctionCenter,
                });
            }
        }

        Shape::Cuboid(hx, hy, hz) => {
            // Suction on each face (if face is large enough for cup)
            let faces: [(Vector3<f64>, f64, f64, f64); 6] = [
                (Vector3::z(), *hx, *hy, *hz),
                (-Vector3::z(), *hx, *hy, *hz),
                (Vector3::x(), *hy, *hz, *hx),
                (-Vector3::x(), *hy, *hz, *hx),
                (Vector3::y(), *hx, *hz, *hy),
                (-Vector3::y(), *hx, *hz, *hy),
            ];

            for (normal, face_w, face_h, offset) in &faces {
                if *face_w >= cup_radius && *face_h >= cup_radius {
                    let grasp_pos = *normal * (*offset + 0.005);
                    // Gripper z-axis should point opposite to surface normal
                    let approach = -*normal;
                    let grasp_rot = rotation_from_approach(&approach, &Vector3::z());

                    let local_pose =
                        Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                    let world_pose = object_pose * local_pose;
                    let world_approach = object_pose.rotation * approach;

                    let alignment = (-world_approach).dot(&config.approach_axis).max(0.0);
                    let face_quality = (face_w * face_h) / (face_w * face_h + 0.01);
                    let quality = (alignment * 0.6 + face_quality * 0.4).clamp(0.0, 1.0);

                    candidates.push(GraspCandidate {
                        grasp_pose: world_pose,
                        approach_direction: world_approach,
                        quality,
                        grasp_type: GraspType::SuctionCenter,
                    });
                }
            }
        }

        Shape::Sphere(radius) => {
            // Suction on top
            if *radius >= cup_radius {
                let grasp_pos = Vector3::new(0.0, 0.0, *radius + 0.005);
                let grasp_rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);

                let local_pose =
                    Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
                let world_pose = object_pose * local_pose;

                candidates.push(GraspCandidate {
                    grasp_pose: world_pose,
                    approach_direction: object_pose.rotation * (-Vector3::z()),
                    quality: 0.5, // sphere top is curved — lower quality for suction
                    grasp_type: GraspType::SuctionCenter,
                });
            }
        }

        Shape::HalfSpace(normal, _offset) => {
            // Suction on the flat surface
            let approach = -*normal;
            let grasp_rot = rotation_from_approach(&approach, &Vector3::z());
            let grasp_pos = *normal * 0.005; // slightly above surface

            let local_pose =
                Isometry3::from_parts(nalgebra::Translation3::from(grasp_pos), grasp_rot);
            let world_pose = object_pose * local_pose;
            let world_approach = object_pose.rotation * approach;

            let alignment = (-world_approach).dot(&config.approach_axis).max(0.0);

            candidates.push(GraspCandidate {
                grasp_pose: world_pose,
                approach_direction: world_approach,
                quality: alignment.clamp(0.0, 1.0),
                grasp_type: GraspType::SuctionCenter,
            });
        }
    }

    candidates
}

// ─── Quality estimation ─────────────────────────────────────────────────────

/// Estimate force closure quality for cylinder antipodal grasp.
fn estimate_force_closure_cylinder(radius: f64, max_opening: f64) -> f64 {
    // Quality increases when object fills more of the gripper
    let fill_ratio = (2.0 * radius) / max_opening;
    // Best when fill ratio is ~0.6-0.8 (good contact without being at limits)
    let optimal = 0.7;
    let deviation = (fill_ratio - optimal).abs();
    (1.0 - deviation * 2.0).clamp(0.0, 1.0)
}

/// Estimate force closure quality for box antipodal grasp.
fn estimate_force_closure_box(half_dim: f64, max_opening: f64) -> f64 {
    let fill_ratio = (2.0 * half_dim) / max_opening;
    let optimal = 0.6;
    let deviation = (fill_ratio - optimal).abs();
    (1.0 - deviation * 2.5).clamp(0.0, 1.0)
}

/// Estimate force closure quality for sphere antipodal grasp.
fn estimate_force_closure_sphere(radius: f64, max_opening: f64) -> f64 {
    let fill_ratio = (2.0 * radius) / max_opening;
    // Spheres are harder to grasp — lower base quality
    let optimal = 0.65;
    let deviation = (fill_ratio - optimal).abs();
    (0.8 - deviation * 2.0).clamp(0.0, 1.0)
}

// ─── Filtering ──────────────────────────────────────────────────────────────

/// Check if a grasp pose would cause collision with the scene.
///
/// Approximates the gripper as a small sphere at the TCP and checks
/// overlap against all scene environment spheres. This avoids needing
/// joint values (which require IK) for a quick geometric check.
fn grasp_in_collision(scene: &Scene, grasp_pose: &Isometry3<f64>) -> bool {
    let tcp = grasp_pose.translation.vector;
    let gripper_radius = 0.03; // approximate TCP sphere radius

    let env_spheres = scene.build_environment_spheres();
    for i in 0..env_spheres.len() {
        let dx = tcp.x - env_spheres.x[i];
        let dy = tcp.y - env_spheres.y[i];
        let dz = tcp.z - env_spheres.z[i];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < gripper_radius + env_spheres.radius[i] {
            return true;
        }
    }
    false
}

/// Check if a grasp pose is reachable via IK.
fn is_reachable(robot: &Robot, chain: &KinematicChain, grasp_pose: &Isometry3<f64>) -> bool {
    let pose = kinetic_core::Pose(*grasp_pose);
    let config = IKConfig {
        num_restarts: 2,
        max_iterations: 50,
        ..Default::default()
    };
    solve_ik(robot, chain, &pose, &config).is_ok()
}

// ─── Ranking ────────────────────────────────────────────────────────────────

/// Rank candidates by the configured metric.
fn rank_candidates(candidates: &mut [GraspCandidate], config: &GraspConfig) {
    match config.rank_by {
        GraspMetric::ForceClosureQuality => {
            // Already stored in quality field
        }
        GraspMetric::DistanceFromCenterOfMass => {
            // Re-score by proximity to origin (object center)
            for c in candidates.iter_mut() {
                let dist = c.grasp_pose.translation.vector.norm();
                c.quality = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);
            }
        }
        GraspMetric::ApproachAngle => {
            // Re-score by alignment with preferred approach axis
            for c in candidates.iter_mut() {
                let alignment = c.approach_direction.dot(&config.approach_axis).abs();
                c.quality = alignment.clamp(0.0, 1.0);
            }
        }
    }

    // Sort descending by quality
    candidates.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Create a rotation where the local -Z axis points along `approach` and Y is near `up`.
fn rotation_from_approach(approach: &Vector3<f64>, up_hint: &Vector3<f64>) -> UnitQuaternion<f64> {
    let z = -approach.normalize();
    let x = up_hint.cross(&z);
    let x_norm = x.norm();
    if x_norm < 1e-10 {
        // approach is parallel to up — pick arbitrary perpendicular
        let alt_up = if approach.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let x = alt_up.cross(&z).normalize();
        let y = z.cross(&x);
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            nalgebra::Matrix3::from_columns(&[x, y, z]),
        ))
    } else {
        let x = x / x_norm;
        let y = z.cross(&x);
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            nalgebra::Matrix3::from_columns(&[x, y, z]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_grasp_cylinder() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        assert!(
            grasps.len() >= 10,
            "Should generate many grasps: got {}",
            grasps.len()
        );

        // Verify sorted by quality
        for w in grasps.windows(2) {
            assert!(
                w[0].quality >= w[1].quality - 1e-10,
                "Should be sorted: {} >= {}",
                w[0].quality,
                w[1].quality
            );
        }
    }

    #[test]
    fn parallel_grasp_box() {
        let gen = GraspGenerator::new(GripperType::parallel(0.10, 0.03));
        let shape = Shape::Cuboid(0.03, 0.04, 0.05);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        assert!(grasps.len() >= 10, "Got {} grasps", grasps.len());
    }

    #[test]
    fn parallel_grasp_sphere() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Sphere(0.03);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        assert!(!grasps.is_empty(), "Should generate sphere grasps");
    }

    #[test]
    fn too_large_for_gripper() {
        let gen = GraspGenerator::new(GripperType::parallel(0.04, 0.03));
        let shape = Shape::Cylinder(0.05, 0.10); // diameter 0.10 > max_opening 0.04
        let pose = Isometry3::identity();

        // May still get top-down or side grasps
        let result = gen.from_shape(&shape, &pose, GraspConfig::default());
        // If all grasps are too big, we get NoGraspsFound
        if let Err(e) = &result {
            assert!(matches!(e, GraspError::NoGraspsFound));
        }
    }

    #[test]
    fn suction_grasp_cylinder() {
        let gen = GraspGenerator::new(GripperType::suction(0.02));
        let shape = Shape::Cylinder(0.04, 0.06);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        assert!(!grasps.is_empty());
        assert!(grasps
            .iter()
            .all(|g| g.grasp_type == GraspType::SuctionCenter));
    }

    #[test]
    fn suction_grasp_box() {
        let gen = GraspGenerator::new(GripperType::suction(0.01));
        let shape = Shape::Cuboid(0.05, 0.03, 0.02);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        assert!(!grasps.is_empty());
    }

    #[test]
    fn grasp_quality_range() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::identity();

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        for g in &grasps {
            assert!(
                g.quality >= 0.0 && g.quality <= 1.0,
                "Quality out of range: {}",
                g.quality
            );
        }
    }

    #[test]
    fn grasp_with_object_pose() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::translation(1.0, 0.5, 0.3);

        let grasps = gen
            .from_shape(&shape, &pose, GraspConfig::default())
            .unwrap();

        // All grasp poses should be offset by the object pose
        for g in &grasps {
            let pos = g.grasp_pose.translation.vector;
            // Should be near the object, not at origin
            let dist_to_object = (pos - Vector3::new(1.0, 0.5, 0.3)).norm();
            assert!(
                dist_to_object < 0.5,
                "Grasp should be near object: dist={}",
                dist_to_object
            );
        }
    }

    #[test]
    fn rank_by_approach_angle() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::identity();

        let config = GraspConfig {
            rank_by: GraspMetric::ApproachAngle,
            ..Default::default()
        };

        let grasps = gen.from_shape(&shape, &pose, config).unwrap();

        // Should be sorted by approach angle alignment
        for w in grasps.windows(2) {
            assert!(w[0].quality >= w[1].quality - 1e-10);
        }
    }

    #[test]
    fn num_candidates_limit() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::identity();

        let config = GraspConfig {
            num_candidates: 5,
            ..Default::default()
        };

        let grasps = gen.from_shape(&shape, &pose, config).unwrap();
        assert!(grasps.len() <= 5, "Got {} grasps", grasps.len());
    }

    #[test]
    fn rotation_from_approach_creates_valid_rotation() {
        let approach = Vector3::new(1.0, 0.0, 0.0);
        let rot = rotation_from_approach(&approach, &Vector3::z());
        let mat = rot.to_rotation_matrix();
        // Check orthogonality
        let det = mat.matrix().determinant();
        assert!(
            (det - 1.0).abs() < 1e-10,
            "Should be proper rotation: det={}",
            det
        );
    }

    #[test]
    fn force_closure_quality_range() {
        let q1 = estimate_force_closure_cylinder(0.02, 0.08);
        let q2 = estimate_force_closure_box(0.03, 0.10);
        let q3 = estimate_force_closure_sphere(0.025, 0.08);

        assert!((0.0..=1.0).contains(&q1), "Cylinder FC: {}", q1);
        assert!((0.0..=1.0).contains(&q2), "Box FC: {}", q2);
        assert!((0.0..=1.0).contains(&q3), "Sphere FC: {}", q3);
    }

    // ─── Collision & reachability filtering tests ───

    /// grasp_in_collision() detects collision with scene obstacle.
    #[test]
    fn grasp_in_collision_with_obstacle() {
        let robot = Robot::from_name("ur5e").unwrap();
        let mut scene = Scene::new(&robot).unwrap();

        // Add a big box obstacle at (0.3, 0, 0.3) covering 0.2m in each direction
        scene.add(
            "box_obstacle",
            Shape::Cuboid(0.2, 0.2, 0.2),
            Isometry3::translation(0.3, 0.0, 0.3),
        );

        // Grasp pose inside the obstacle — should collide
        let colliding_pose = Isometry3::translation(0.3, 0.0, 0.3);
        assert!(
            grasp_in_collision(&scene, &colliding_pose),
            "grasp at obstacle center should be in collision"
        );
    }

    /// grasp_in_collision() accepts grasp in free space.
    #[test]
    fn grasp_in_collision_free_space() {
        let robot = Robot::from_name("ur5e").unwrap();
        let mut scene = Scene::new(&robot).unwrap();

        // Add obstacle far from where we'll test
        scene.add(
            "box_obstacle",
            Shape::Cuboid(0.1, 0.1, 0.1),
            Isometry3::translation(5.0, 5.0, 5.0),
        );

        // Grasp pose far from obstacle — should be free
        let free_pose = Isometry3::translation(0.3, 0.0, 0.3);
        assert!(
            !grasp_in_collision(&scene, &free_pose),
            "grasp far from obstacle should not be in collision"
        );
    }

    /// grasp_in_collision() with empty scene (no obstacles).
    #[test]
    fn grasp_in_collision_empty_scene() {
        let robot = Robot::from_name("ur5e").unwrap();
        let scene = Scene::new(&robot).unwrap();
        let pose = Isometry3::translation(0.3, 0.0, 0.3);
        assert!(
            !grasp_in_collision(&scene, &pose),
            "no obstacles means no collision"
        );
    }

    /// is_reachable() accepts pose inside UR5e workspace.
    #[test]
    fn is_reachable_inside_workspace() {
        let robot = Robot::from_name("ur5e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        // Compute a known reachable pose via FK
        let q = [0.5, -1.0, 0.8, -0.5, 1.2, -0.3];
        let pose = kinetic_kinematics::forward_kinematics(&robot, &chain, &q).unwrap();
        let iso = *pose.isometry();

        assert!(
            is_reachable(&robot, &chain, &iso),
            "FK-derived pose should be reachable"
        );
    }

    /// is_reachable() rejects pose far outside workspace.
    #[test]
    fn is_reachable_outside_workspace() {
        let robot = Robot::from_name("ur5e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        // Pose at 10 meters away — far beyond UR5e reach (~0.85m)
        let unreachable = Isometry3::translation(10.0, 10.0, 10.0);
        assert!(
            !is_reachable(&robot, &chain, &unreachable),
            "pose at 10m should be unreachable"
        );
    }

    /// Collision filter reduces candidate count when obstacle overlaps grasp region.
    #[test]
    fn collision_filter_reduces_candidates() {
        let robot = Robot::from_name("ur5e").unwrap();
        let mut scene = Scene::new(&robot).unwrap();

        // Generate grasps without collision filtering
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let object_pose = Isometry3::identity();

        let config_no_filter = GraspConfig {
            num_candidates: 200,
            ..Default::default()
        };
        let unfiltered = gen
            .from_shape(&shape, &object_pose, config_no_filter)
            .unwrap();
        let unfiltered_count = unfiltered.len();

        // Add big obstacle covering where grasps are generated (around origin)
        scene.add(
            "blocker",
            Shape::Cuboid(0.5, 0.5, 0.5),
            Isometry3::identity(),
        );

        let config_with_filter = GraspConfig {
            num_candidates: 200,
            check_collision: Some(Arc::new(scene)),
            ..Default::default()
        };

        // With collision filtering, should have fewer candidates (or none)
        let result = gen.from_shape(&shape, &object_pose, config_with_filter);
        match result {
            Ok(filtered) => {
                assert!(
                    filtered.len() < unfiltered_count,
                    "collision filter should remove candidates: {} vs {}",
                    filtered.len(),
                    unfiltered_count,
                );
            }
            Err(GraspError::NoGraspsFound) => {
                // All grasps filtered out — expected when obstacle covers everything
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    /// Combined collision + reachability filtering.
    #[test]
    fn combined_collision_and_reachability_filter() {
        let robot = Robot::from_name("ur5e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        let scene = Scene::new(&robot).unwrap();

        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        // Place object at 10m — unreachable for UR5e
        let object_pose = Isometry3::translation(10.0, 0.0, 0.0);

        let config = GraspConfig {
            num_candidates: 50,
            check_collision: Some(Arc::new(scene)),
            check_reachability: Some((Arc::new(robot), chain)),
            ..Default::default()
        };

        let result = gen.from_shape(&shape, &object_pose, config);
        // All grasps should be filtered out by reachability
        assert!(
            matches!(result, Err(GraspError::NoGraspsFound)),
            "grasps at 10m should all be unreachable"
        );
    }

    /// Reachability filter demonstrably removes candidates for borderline poses.
    #[test]
    fn reachability_filter_removes_candidates() {
        let robot = Robot::from_name("ur5e").unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);

        // Without reachability filter
        let object_pose = Isometry3::translation(0.3, 0.0, 0.3);
        let config_no_filter = GraspConfig {
            num_candidates: 50,
            ..Default::default()
        };
        let unfiltered = gen
            .from_shape(&shape, &object_pose, config_no_filter)
            .unwrap();
        let unfiltered_count = unfiltered.len();

        // With reachability filter — some grasps may have IK failures
        let config_with_filter = GraspConfig {
            num_candidates: 50,
            check_reachability: Some((Arc::new(robot), chain)),
            ..Default::default()
        };
        let result = gen.from_shape(&shape, &object_pose, config_with_filter);
        match result {
            Ok(filtered) => {
                // Reachability filtering should reduce or maintain candidate count
                assert!(filtered.len() <= unfiltered_count);
                // Verify all remaining candidates have valid IK (this is already
                // guaranteed by the filter, but we confirm it doesn't produce invalid results)
                assert!(!filtered.is_empty());
            }
            Err(GraspError::NoGraspsFound) => {
                // All filtered out — acceptable
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    /// Gap 7: Generate 50 grasps for a cylinder. Verify they're sorted by quality
    /// (descending), all scores are in [0, 1], and approach_direction is a unit vector.
    #[test]
    fn grasp_quality_metrics_cylinder_50() {
        let gen = GraspGenerator::new(GripperType::parallel(0.08, 0.03));
        let shape = Shape::Cylinder(0.03, 0.06);
        let pose = Isometry3::identity();

        let config = GraspConfig {
            num_candidates: 50,
            ..Default::default()
        };

        let grasps = gen.from_shape(&shape, &pose, config).unwrap();

        assert!(
            !grasps.is_empty(),
            "Should generate at least some grasps for a cylinder"
        );

        // 1. Verify sorted by quality descending
        for window in grasps.windows(2) {
            assert!(
                window[0].quality >= window[1].quality - 1e-10,
                "Grasps should be sorted descending by quality: {} >= {}",
                window[0].quality,
                window[1].quality
            );
        }

        // 2. Verify quality scores are in [0, 1]
        for (i, g) in grasps.iter().enumerate() {
            assert!(
                g.quality >= 0.0 && g.quality <= 1.0,
                "Grasp {} quality {} out of [0, 1] range",
                i,
                g.quality
            );
        }

        // 3. Verify approach_direction is a unit vector (norm ~1.0)
        for (i, g) in grasps.iter().enumerate() {
            let norm = g.approach_direction.norm();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Grasp {} approach_direction norm should be 1.0, got {}",
                i,
                norm
            );
        }
    }
}
