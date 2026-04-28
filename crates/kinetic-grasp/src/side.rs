//! Side grasp generation (horizontal approach perpendicular to gravity).

use std::f64::consts::TAU;

use nalgebra::{Isometry3, Vector3};

use kinetic_scene::Shape;

use crate::helpers::rotation_from_approach;
use crate::{GraspCandidate, GraspConfig, GraspType};

/// Generate side grasps (horizontal approach).
pub(crate) fn generate_side_grasps(
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
