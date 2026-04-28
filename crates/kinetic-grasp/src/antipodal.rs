//! Antipodal (parallel-jaw) grasp generation.
//!
//! Approaches the object perpendicular to two opposing faces or surfaces
//! such that the fingers close along antiparallel surface normals. Best
//! for cylinders (grasp across diameter), boxes (grasp across faces), and
//! spheres (grasp at equator).

use std::f64::consts::PI;

use nalgebra::{Isometry3, Vector3};

use kinetic_scene::Shape;

use crate::helpers::rotation_from_approach;
use crate::quality::{
    estimate_force_closure_box, estimate_force_closure_cylinder, estimate_force_closure_sphere,
};
use crate::{GraspCandidate, GraspConfig, GraspType};

/// Generate antipodal (parallel jaw) grasps for a shape.
pub(crate) fn generate_antipodal(
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
