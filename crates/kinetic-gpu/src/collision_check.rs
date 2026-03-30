//! GPU-accelerated batch collision checking.
//!
//! Checks many joint configurations against a pre-built SDF in parallel.
//! Upload the robot sphere model and scene SDF once, then batch-check
//! hundreds of configurations on the GPU.

use crate::batch_fk;
use crate::sdf::SignedDistanceField;
use crate::{GpuError, Result};
use kinetic_collision::SpheresSoA;
use kinetic_robot::Robot;

/// Result of a batch collision check.
pub struct BatchCollisionResult {
    /// Whether each configuration is in collision.
    pub in_collision: Vec<bool>,
    /// Minimum SDF distance for each configuration (negative = penetrating).
    pub min_distances: Vec<f64>,
}

/// GPU-accelerated batch collision checker.
///
/// Pre-computes the SDF from scene obstacles, then checks many configurations
/// in parallel. Much faster than CPU for batches >64 configurations.
pub struct GpuCollisionChecker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sdf: SignedDistanceField,
}

impl GpuCollisionChecker {
    /// Create a collision checker with a pre-built SDF.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, sdf: SignedDistanceField) -> Self {
        Self { device, queue, sdf }
    }

    /// Create a collision checker from obstacle spheres.
    pub fn from_spheres(
        obstacle_spheres: &SpheresSoA,
        workspace_bounds: [f32; 6],
        resolution: f32,
    ) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or(GpuError::NoAdapter)?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kinetic-collision-gpu"),
                ..Default::default()
            },
            None,
        ))?;

        let sdf = SignedDistanceField::from_spheres(
            &device,
            &queue,
            obstacle_spheres,
            [
                workspace_bounds[0],
                workspace_bounds[1],
                workspace_bounds[2],
            ],
            [
                workspace_bounds[3],
                workspace_bounds[4],
                workspace_bounds[5],
            ],
            resolution,
        )?;

        Ok(Self { device, queue, sdf })
    }

    /// Create from a planning scene.
    #[cfg(feature = "scene")]
    pub fn from_scene(
        scene: &kinetic_scene::Scene,
        workspace_bounds: [f32; 6],
        resolution: f32,
    ) -> Result<Self> {
        let spheres = scene.build_environment_spheres();
        Self::from_spheres(&spheres, workspace_bounds, resolution)
    }

    /// Check a batch of configurations for collision.
    ///
    /// Returns a `BatchCollisionResult` with per-configuration collision status
    /// and minimum signed distance. Uses GPU FK to compute sphere positions,
    /// then CPU SDF queries (GPU SDF query kernel for future optimization).
    pub fn check_batch(&self, robot: &Robot, configs: &[Vec<f64>]) -> Result<BatchCollisionResult> {
        if configs.is_empty() {
            return Ok(BatchCollisionResult {
                in_collision: Vec::new(),
                min_distances: Vec::new(),
            });
        }

        // Use GPU batch FK to get world-frame sphere positions
        let fk_result = batch_fk::batch_fk_gpu(&self.device, &self.queue, robot, configs)?;

        // For each config, check all spheres against the SDF
        let mut in_collision = Vec::with_capacity(configs.len());
        let mut min_distances = Vec::with_capacity(configs.len());

        for spheres in &fk_result.world_spheres {
            let mut min_dist = f64::INFINITY;
            let mut colliding = false;

            for i in 0..spheres.len() {
                let sdf_dist = self.sdf.query(spheres.x[i], spheres.y[i], spheres.z[i]);
                let signed_dist = sdf_dist - spheres.radius[i];

                if signed_dist < min_dist {
                    min_dist = signed_dist;
                }
                if signed_dist < 0.0 {
                    colliding = true;
                }
            }

            // If no spheres, not in collision
            if spheres.is_empty() {
                min_dist = f64::INFINITY;
            }

            in_collision.push(colliding);
            min_distances.push(min_dist);
        }

        Ok(BatchCollisionResult {
            in_collision,
            min_distances,
        })
    }

    /// Check a single configuration (convenience wrapper).
    pub fn check_single(&self, robot: &Robot, config: &[f64]) -> Result<(bool, f64)> {
        let result = self.check_batch(robot, &[config.to_vec()])?;
        Ok((result.in_collision[0], result.min_distances[0]))
    }

    /// Access the underlying SDF.
    pub fn sdf(&self) -> &SignedDistanceField {
        &self.sdf
    }
}

/// CPU-only batch collision checker (no GPU required).
///
/// Uses CPU SDF for collision checking. Suitable for small batches
/// or when no GPU is available.
pub struct CpuCollisionChecker {
    sdf: SignedDistanceField,
}

impl CpuCollisionChecker {
    /// Create from obstacle spheres using CPU SDF.
    pub fn from_spheres(
        obstacle_spheres: &SpheresSoA,
        workspace_bounds: [f32; 6],
        resolution: f32,
    ) -> Result<Self> {
        let sdf = SignedDistanceField::from_spheres_cpu(
            obstacle_spheres,
            [
                workspace_bounds[0],
                workspace_bounds[1],
                workspace_bounds[2],
            ],
            [
                workspace_bounds[3],
                workspace_bounds[4],
                workspace_bounds[5],
            ],
            resolution,
        )?;
        Ok(Self { sdf })
    }

    /// Check a single configuration using CPU FK + CPU SDF.
    pub fn check(
        &self,
        robot: &Robot,
        config: &[f64],
        sphere_model: &kinetic_collision::RobotSphereModel,
    ) -> (bool, f64) {
        // Compute world spheres using CPU FK
        let world_spheres = cpu_world_spheres(robot, sphere_model, config);
        let mut min_dist = f64::INFINITY;
        let mut colliding = false;

        let num_spheres = world_spheres.len() / 4;
        for i in 0..num_spheres {
            let x = world_spheres[i * 4];
            let y = world_spheres[i * 4 + 1];
            let z = world_spheres[i * 4 + 2];
            let r = world_spheres[i * 4 + 3];

            let sdf_dist = self.sdf.query(x, y, z);
            let signed_dist = sdf_dist - r;

            if signed_dist < min_dist {
                min_dist = signed_dist;
            }
            if signed_dist < 0.0 {
                colliding = true;
            }
        }

        (colliding, min_dist)
    }

    /// Access the underlying SDF.
    pub fn sdf(&self) -> &SignedDistanceField {
        &self.sdf
    }
}

/// Compute world-frame sphere positions via CPU FK.
fn cpu_world_spheres(
    robot: &Robot,
    sphere_model: &kinetic_collision::RobotSphereModel,
    joints: &[f64],
) -> Vec<f64> {
    use nalgebra::{Isometry3, UnitQuaternion, Vector3};

    let num_joints = robot.joints.len();
    let mut transforms = Vec::with_capacity(num_joints);

    for joint in &robot.joints {
        let parent_transform = if joint.parent_link == 0 {
            Isometry3::identity()
        } else {
            robot
                .joints
                .iter()
                .position(|j| j.child_link == joint.parent_link)
                .and_then(|pi| transforms.get(pi).copied())
                .unwrap_or(Isometry3::identity())
        };

        let ji = transforms.len();
        let joint_val = if ji < joints.len() { joints[ji] } else { 0.0 };

        let local_transform = match joint.joint_type {
            kinetic_robot::JointType::Revolute | kinetic_robot::JointType::Continuous => {
                let axis = Vector3::new(joint.axis.x, joint.axis.y, joint.axis.z);
                let rot = UnitQuaternion::new(axis * joint_val);
                joint.origin.0 * Isometry3::from_parts(Default::default(), rot)
            }
            kinetic_robot::JointType::Prismatic => {
                let axis = Vector3::new(joint.axis.x, joint.axis.y, joint.axis.z);
                let trans = axis * joint_val;
                joint.origin.0
                    * Isometry3::from_parts(
                        nalgebra::Translation3::from(trans),
                        UnitQuaternion::identity(),
                    )
            }
            kinetic_robot::JointType::Fixed => joint.origin.0,
        };

        transforms.push(parent_transform * local_transform);
    }

    let mut result = Vec::new();
    for link_idx in 0..sphere_model.num_links {
        let (start, end) = sphere_model.link_ranges[link_idx];
        let joint_idx = robot
            .joints
            .iter()
            .position(|j| j.child_link == link_idx)
            .unwrap_or(0);

        let tf = transforms
            .get(joint_idx)
            .copied()
            .unwrap_or(Isometry3::identity());

        for i in start..end {
            let local_pt = nalgebra::Point3::new(
                sphere_model.local.x[i],
                sphere_model.local.y[i],
                sphere_model.local.z[i],
            );
            let world_pt = tf * local_pt;
            result.push(world_pt.x);
            result.push(world_pt.y);
            result.push(world_pt.z);
            result.push(sphere_model.local.radius[i]);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_collision_checker_no_obstacles() {
        let empty_obs = SpheresSoA::default();
        let checker =
            CpuCollisionChecker::from_spheres(&empty_obs, [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], 0.1)
                .unwrap();

        let robot = Robot::from_name("franka_panda").unwrap();
        let sphere_model = kinetic_collision::RobotSphereModel::from_robot_default(&robot);
        let mid: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();

        let (colliding, min_dist) = checker.check(&robot, &mid, &sphere_model);
        // No obstacles → should not be in collision
        assert!(!colliding || min_dist > -0.001);
    }

    #[test]
    fn batch_result_empty() {
        let result = BatchCollisionResult {
            in_collision: vec![],
            min_distances: vec![],
        };
        assert_eq!(result.in_collision.len(), 0);
    }
}
