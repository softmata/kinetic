//! Suction grasp generation: cup placement on flat or near-flat surfaces.

use std::f64::consts::PI;

use nalgebra::{Isometry3, UnitQuaternion, Vector3};

use kinetic_scene::Shape;

use crate::helpers::rotation_from_approach;
use crate::{GraspCandidate, GraspConfig, GraspType};

/// Generate suction grasp candidates.
pub(crate) fn generate_suction(
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
