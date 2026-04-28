//! Top-down grasp generation (vertical approach along world -Z).

use std::f64::consts::PI;

use nalgebra::{Isometry3, UnitQuaternion, Vector3};

use kinetic_scene::Shape;

use crate::{GraspCandidate, GraspConfig, GraspType};

/// Generate top-down grasps (approach from above along -Z).
pub(crate) fn generate_topdown(
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
