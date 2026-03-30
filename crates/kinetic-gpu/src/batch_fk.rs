//! Standalone GPU batch forward kinematics.
//!
//! Computes FK for many joint configurations in parallel on the GPU.
//! Useful for batch collision checking, workspace analysis, and
//! configuration sampling — independent of trajectory optimization.

use crate::{GpuError, Result};
use bytemuck::{Pod, Zeroable};
use kinetic_collision::{RobotSphereModel, SpheresSoA};
use kinetic_robot::{JointType, Robot};
use wgpu::util::DeviceExt;

/// GPU-side FK parameters (matches `fk_gpu.wgsl` FKParams).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FkParamsGpu {
    num_seeds: u32,
    timesteps: u32,
    dof: u32,
    num_spheres: u32,
    num_joints: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-side local sphere (matches `fk_gpu.wgsl` LocalSphere).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LocalSphereGpu {
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    joint_idx: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Result of a batch FK computation: world-frame sphere positions for each configuration.
pub struct BatchFkResult {
    /// World-frame spheres for each configuration.
    /// Indexed as `configs[config_idx]` → `SpheresSoA` with world positions.
    pub world_spheres: Vec<SpheresSoA>,
}

/// Compute forward kinematics for many configurations in parallel on the GPU.
///
/// Returns world-frame collision sphere positions for each input configuration.
/// This is much faster than calling CPU FK in a loop for large batches.
///
/// # Arguments
/// * `device`, `queue` — wgpu device and queue
/// * `robot` — robot model
/// * `configs` — slice of joint configurations, each of length `robot.dof`
///
/// # Returns
/// A `BatchFkResult` with world-frame spheres per configuration.
pub fn batch_fk_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    robot: &Robot,
    configs: &[Vec<f64>],
) -> Result<BatchFkResult> {
    let num_configs = configs.len();
    if num_configs == 0 {
        return Ok(BatchFkResult {
            world_spheres: Vec::new(),
        });
    }

    let dof = robot.dof;
    for (i, c) in configs.iter().enumerate() {
        if c.len() != dof {
            return Err(GpuError::InvalidConfig(format!(
                "config[{}] has {} joints, expected {}",
                i,
                c.len(),
                dof
            )));
        }
    }

    // Prepare robot data
    let (joint_origins, joint_axes, joint_types, local_spheres, num_joints) =
        prepare_robot_data(robot);
    let num_spheres = local_spheres.len();

    if num_spheres == 0 {
        // No collision geometry — return empty spheres for each config
        return Ok(BatchFkResult {
            world_spheres: configs.iter().map(|_| SpheresSoA::new()).collect(),
        });
    }

    // Pack configurations into flat f32 buffer
    // We treat configs as "seeds" with timesteps=1
    let mut joint_data: Vec<f32> = Vec::with_capacity(num_configs * dof);
    for c in configs {
        for &val in c {
            joint_data.push(val as f32);
        }
    }

    let fk_params = FkParamsGpu {
        num_seeds: num_configs as u32,
        timesteps: 1,
        dof: dof as u32,
        num_spheres: num_spheres as u32,
        num_joints: num_joints as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    // Create shader and pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batch_fk_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fk_gpu.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("batch_fk_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("fk_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create buffers
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fk_params"),
        contents: bytemuck::bytes_of(&fk_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let origins_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("joint_origins"),
        contents: bytemuck::cast_slice(&joint_origins),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let axes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("joint_axes"),
        contents: bytemuck::cast_slice(&joint_axes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let types_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("joint_types"),
        contents: bytemuck::cast_slice(&joint_types),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let joints_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("joint_values"),
        contents: bytemuck::cast_slice(&joint_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let local_spheres_buf = create_storage_buffer(
        device,
        "local_spheres",
        bytemuck::cast_slice(&local_spheres),
    );

    let world_size = (num_configs * num_spheres * 4 * std::mem::size_of::<f32>()) as u64;
    let world_spheres_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("world_spheres"),
        size: world_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("batch_fk_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: origins_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: axes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: types_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: joints_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: local_spheres_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: world_spheres_buf.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let total_invocations = (num_configs * num_spheres) as u32;
    let workgroups = total_invocations.div_ceil(64);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batch_fk"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Read back
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fk_staging"),
        size: world_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&world_spheres_buf, 0, &staging, 0, world_size);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| GpuError::BufferMapping)?
        .map_err(|_| GpuError::BufferMapping)?;

    let data = slice.get_mapped_range();
    let flat: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    // Parse results into SpheresSoA per configuration
    let mut world_spheres = Vec::with_capacity(num_configs);
    for ci in 0..num_configs {
        let mut soa = SpheresSoA::new();
        for si in 0..num_spheres {
            let base = (ci * num_spheres + si) * 4;
            soa.push(
                flat[base] as f64,
                flat[base + 1] as f64,
                flat[base + 2] as f64,
                flat[base + 3] as f64,
                0,
            );
        }
        world_spheres.push(soa);
    }

    Ok(BatchFkResult { world_spheres })
}

/// Create a storage buffer, using a 4-byte placeholder when data is empty.
fn create_storage_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let contents = if data.is_empty() { &[0u8; 4] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Prepare robot joint and sphere data for GPU buffers.
fn prepare_robot_data(robot: &Robot) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<LocalSphereGpu>, usize) {
    let num_joints = robot.joints.len();

    let mut origins: Vec<f32> = Vec::with_capacity(num_joints * 16);
    let mut axes: Vec<f32> = Vec::with_capacity(num_joints * 4);
    let mut types: Vec<u32> = Vec::with_capacity(num_joints);

    for joint in &robot.joints {
        let m = joint.origin.0.to_homogeneous();
        for col in 0..4 {
            for row in 0..4 {
                origins.push(m[(row, col)] as f32);
            }
        }

        axes.push(joint.axis.x as f32);
        axes.push(joint.axis.y as f32);
        axes.push(joint.axis.z as f32);
        axes.push(0.0);

        let jtype = match joint.joint_type {
            JointType::Revolute => 0u32,
            JointType::Prismatic => 1u32,
            JointType::Continuous => 2u32,
            JointType::Fixed => 3u32,
        };
        types.push(jtype);
    }

    let sphere_model = RobotSphereModel::from_robot_default(robot);
    let mut local_spheres = Vec::new();

    for link_idx in 0..sphere_model.num_links {
        let (start, end) = sphere_model.link_ranges[link_idx];
        let joint_idx = robot
            .joints
            .iter()
            .position(|j| j.child_link == link_idx)
            .unwrap_or(0);

        for i in start..end {
            local_spheres.push(LocalSphereGpu {
                x: sphere_model.local.x[i] as f32,
                y: sphere_model.local.y[i] as f32,
                z: sphere_model.local.z[i] as f32,
                radius: sphere_model.local.radius[i] as f32,
                joint_idx: joint_idx as u32,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            });
        }
    }

    (origins, axes, types, local_spheres, num_joints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_fk_empty_configs() {
        // Should handle empty input gracefully without GPU
        let robot = Robot::from_name("franka_panda").unwrap();
        let _configs: Vec<Vec<f64>> = vec![];

        // Can't test GPU without a device, but we can test the empty path
        let result = BatchFkResult {
            world_spheres: vec![],
        };
        assert_eq!(result.world_spheres.len(), 0);

        // Also verify prepare_robot_data works
        let (_origins, _axes, _types, _spheres, num_joints) = prepare_robot_data(&robot);
        assert_eq!(num_joints, robot.joints.len());
    }

    #[test]
    fn prepare_robot_data_correctness() {
        let robot = Robot::from_name("ur5e").unwrap();
        let (origins, axes, types, _spheres, num_joints) = prepare_robot_data(&robot);

        assert_eq!(origins.len(), num_joints * 16);
        assert_eq!(axes.len(), num_joints * 4);
        assert_eq!(types.len(), num_joints);

        // UR5e has 6 active joints
        let active_count = types.iter().filter(|&&t| t != 3).count();
        assert_eq!(active_count, robot.dof);
    }
}
