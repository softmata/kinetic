//! GPU trajectory optimizer using parallel seeds.
//!
//! Implements cuRobo-style optimization:
//! 1. Generate N trajectory seeds (random perturbations of linear interpolation)
//! 2. Build SDF from environment obstacle spheres
//! 3. For each iteration: run FK on GPU → run optimization kernel (cost + gradient step)
//! 4. Pick the lowest-cost trajectory

use crate::sdf::SignedDistanceField;
use crate::{GpuError, Result};
use bytemuck::{Pod, Zeroable};
use kinetic_collision::{RobotSphereModel, SpheresSoA};
use kinetic_core::Trajectory;
use kinetic_robot::{JointType, Robot};
use rand::Rng;
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

/// GPU-side optimization parameters (matches `trajectory_optimize.wgsl` OptParams).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OptParamsGpu {
    num_seeds: u32,
    timesteps: u32,
    dof: u32,
    num_spheres: u32,
    collision_weight: f32,
    smoothness_weight: f32,
    goal_weight: f32,
    step_size: f32,
    sdf_min_x: f32,
    sdf_min_y: f32,
    sdf_min_z: f32,
    sdf_resolution: f32,
    sdf_nx: u32,
    sdf_ny: u32,
    sdf_nz: u32,
    _pad: u32,
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

/// Configuration for GPU trajectory optimization.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of parallel trajectory seeds to optimize.
    pub num_seeds: u32,
    /// Number of timesteps per trajectory.
    pub timesteps: u32,
    /// Number of gradient descent iterations.
    pub iterations: u32,
    /// Weight for collision cost.
    pub collision_weight: f32,
    /// Weight for smoothness cost (jerk minimization).
    pub smoothness_weight: f32,
    /// Weight for goal-reaching cost.
    pub goal_weight: f32,
    /// Gradient descent step size.
    pub step_size: f32,
    /// SDF voxel resolution in meters.
    pub sdf_resolution: f32,
    /// Workspace bounds `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub workspace_bounds: [f32; 6],
    /// Random seed perturbation magnitude (radians for revolute joints).
    pub seed_perturbation: f32,
    /// Optional warm-start trajectory (e.g. from RRT). Used as seed 0
    /// instead of unperturbed linear interpolation.
    pub warm_start: Option<Vec<Vec<f64>>>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            num_seeds: 128,
            timesteps: 32,
            iterations: 100,
            collision_weight: 100.0,
            smoothness_weight: 1.0,
            goal_weight: 50.0,
            step_size: 0.01,
            sdf_resolution: 0.02,
            workspace_bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
            seed_perturbation: 0.3,
            warm_start: None,
        }
    }
}

impl GpuConfig {
    /// Balanced trade-off between speed and trajectory quality.
    ///
    /// Good default for most planning scenarios.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Fast optimization with fewer seeds and iterations.
    ///
    /// Use for real-time replanning or when latency matters more than optimality.
    pub fn speed() -> Self {
        Self {
            num_seeds: 32,
            timesteps: 24,
            iterations: 30,
            sdf_resolution: 0.05,
            ..Self::default()
        }
    }

    /// High-quality optimization with more seeds and finer resolution.
    ///
    /// Use for offline planning where trajectory quality matters most.
    pub fn quality() -> Self {
        Self {
            num_seeds: 512,
            timesteps: 48,
            iterations: 200,
            collision_weight: 200.0,
            sdf_resolution: 0.01,
            ..Self::default()
        }
    }
}

/// GPU-accelerated parallel trajectory optimizer.
///
/// Uses wgpu compute shaders to run FK, SDF construction, and trajectory
/// optimization across many parallel seeds simultaneously.
pub struct GpuOptimizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: GpuConfig,
}

impl GpuOptimizer {
    /// Create a new GPU optimizer, requesting a high-performance GPU device.
    pub fn new(config: GpuConfig) -> Result<Self> {
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
                label: Some("kinetic-gpu"),
                ..Default::default()
            },
            None,
        ))?;

        Ok(Self {
            device,
            queue,
            config,
        })
    }

    /// Create an optimizer with an existing wgpu device and queue.
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, config: GpuConfig) -> Self {
        Self {
            device,
            queue,
            config,
        }
    }

    /// Optimize a trajectory from start to goal joint configuration.
    ///
    /// `obstacle_spheres` are the environment collision spheres (from `Scene::build_environment_spheres()`
    /// or constructed manually). Returns the best trajectory found across all seeds.
    pub fn optimize(
        &self,
        robot: &Robot,
        obstacle_spheres: &SpheresSoA,
        start: &[f64],
        goal: &[f64],
    ) -> Result<Trajectory> {
        let dof = robot.dof;
        let num_seeds = self.config.num_seeds as usize;
        let timesteps = self.config.timesteps as usize;

        if start.len() != dof || goal.len() != dof {
            return Err(GpuError::InvalidConfig(format!(
                "start/goal length ({}/{}) doesn't match robot DOF ({})",
                start.len(),
                goal.len(),
                dof
            )));
        }

        // --- Prepare robot data for GPU ---
        let (joint_origins, joint_axes, joint_types, local_spheres, num_gpu_joints) =
            prepare_robot_data(robot);
        let num_spheres = local_spheres.len();

        // num_spheres == 0 is valid: optimizer still uses smoothness + goal costs.
        // FK pass becomes a no-op and collision cost is zero everywhere.

        // --- Build SDF from obstacle spheres ---
        let wb = &self.config.workspace_bounds;
        let sdf = SignedDistanceField::from_spheres(
            &self.device,
            &self.queue,
            obstacle_spheres,
            [wb[0], wb[1], wb[2]],
            [wb[3], wb[4], wb[5]],
            self.config.sdf_resolution,
        )?;

        // --- Generate trajectory seeds ---
        let trajectories = generate_seeds(
            start,
            goal,
            robot,
            num_seeds,
            timesteps,
            self.config.seed_perturbation,
            self.config.warm_start.as_ref(),
        );

        // --- Create GPU pipelines ---
        let fk_pipeline = self.create_fk_pipeline();
        let opt_pipeline = self.create_opt_pipeline();

        // --- Create GPU buffers ---
        let fk_params = FkParamsGpu {
            num_seeds: self.config.num_seeds,
            timesteps: self.config.timesteps,
            dof: dof as u32,
            num_spheres: num_spheres as u32,
            num_joints: num_gpu_joints as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let fk_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fk_params"),
                contents: bytemuck::bytes_of(&fk_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let origins_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("joint_origins"),
                contents: bytemuck::cast_slice(&joint_origins),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let axes_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("joint_axes"),
                contents: bytemuck::cast_slice(&joint_axes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let types_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("joint_types"),
                contents: bytemuck::cast_slice(&joint_types),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let traj_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trajectories"),
                contents: bytemuck::cast_slice(&trajectories),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let local_spheres_buf = create_storage_buffer(
            &self.device,
            "local_spheres",
            bytemuck::cast_slice(&local_spheres),
        );

        // At least 4 bytes for wgpu buffer minimum size
        let world_spheres_size =
            (num_seeds * timesteps * num_spheres.max(1) * 4 * std::mem::size_of::<f32>()) as u64;
        let world_spheres_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world_spheres"),
            size: world_spheres_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sdf_grid_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sdf_grid"),
                contents: bytemuck::cast_slice(&sdf.grid),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let goal_f32: Vec<f32> = goal.iter().map(|&v| v as f32).collect();
        let goal_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goal_joints"),
                contents: bytemuck::cast_slice(&goal_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let joint_lower: Vec<f32> = robot.joint_limits.iter().map(|l| l.lower as f32).collect();
        let joint_upper: Vec<f32> = robot.joint_limits.iter().map(|l| l.upper as f32).collect();
        let lower_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("joint_lower"),
                contents: bytemuck::cast_slice(&joint_lower),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let upper_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("joint_upper"),
                contents: bytemuck::cast_slice(&joint_upper),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let costs_size = (num_seeds * std::mem::size_of::<f32>()) as u64;
        let costs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("costs"),
            size: costs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let opt_params = OptParamsGpu {
            num_seeds: self.config.num_seeds,
            timesteps: self.config.timesteps,
            dof: dof as u32,
            num_spheres: num_spheres as u32,
            collision_weight: self.config.collision_weight,
            smoothness_weight: self.config.smoothness_weight,
            goal_weight: self.config.goal_weight,
            step_size: self.config.step_size,
            sdf_min_x: sdf.min_x,
            sdf_min_y: sdf.min_y,
            sdf_min_z: sdf.min_z,
            sdf_resolution: sdf.resolution,
            sdf_nx: sdf.nx,
            sdf_ny: sdf.ny,
            sdf_nz: sdf.nz,
            _pad: 0,
        };
        let opt_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opt_params"),
                contents: bytemuck::bytes_of(&opt_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // --- Create bind groups ---
        let fk_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk_bind_group"),
            layout: &fk_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fk_params_buf.as_entire_binding(),
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
                    resource: traj_buf.as_entire_binding(),
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

        let opt_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opt_bind_group"),
            layout: &opt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: opt_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: traj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: world_spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sdf_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: goal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: lower_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: upper_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: costs_buf.as_entire_binding(),
                },
            ],
        });

        // --- Optimization loop ---
        let fk_workgroups = ((num_seeds * timesteps * num_spheres) as u32).div_ceil(64);
        let opt_workgroups = self.config.num_seeds; // workgroup_size(1) in optimizer

        for _ in 0..self.config.iterations {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            // FK pass: joint values → world-frame sphere positions
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fk_pass"),
                    ..Default::default()
                });
                pass.set_pipeline(&fk_pipeline);
                pass.set_bind_group(0, &fk_bind_group, &[]);
                pass.dispatch_workgroups(fk_workgroups, 1, 1);
            }

            // Optimization pass: compute costs + gradient descent step
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("opt_pass"),
                    ..Default::default()
                });
                pass.set_pipeline(&opt_pipeline);
                pass.set_bind_group(0, &opt_bind_group, &[]);
                pass.dispatch_workgroups(opt_workgroups, 1, 1);
            }

            self.queue.submit([encoder.finish()]);
        }

        // --- Read back results ---
        let traj_size = (num_seeds * timesteps * dof * std::mem::size_of::<f32>()) as u64;
        let traj_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traj_staging"),
            size: traj_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let costs_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("costs_staging"),
            size: costs_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&traj_buf, 0, &traj_staging, 0, traj_size);
        encoder.copy_buffer_to_buffer(&costs_buf, 0, &costs_staging, 0, costs_size);
        self.queue.submit([encoder.finish()]);

        let costs = read_buffer_f32(&self.device, &costs_staging, num_seeds)?;
        let trajs = read_buffer_f32(&self.device, &traj_staging, num_seeds * timesteps * dof)?;

        // Find best seed (lowest cost)
        let best_seed = costs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Extract best trajectory and convert to kinetic Trajectory
        let mut traj = Trajectory::with_dof(dof);
        for t in 0..timesteps {
            let base = best_seed * timesteps * dof + t * dof;
            let wp: Vec<f64> = (0..dof).map(|j| trajs[base + j] as f64).collect();
            traj.push_waypoint(&wp);
        }

        Ok(traj)
    }

    /// Optimize a trajectory using obstacles from a planning scene.
    ///
    /// Convenience wrapper that extracts obstacle spheres from the scene.
    #[cfg(feature = "scene")]
    pub fn optimize_with_scene(
        &self,
        robot: &Robot,
        scene: &kinetic_scene::Scene,
        start: &[f64],
        goal: &[f64],
    ) -> Result<Trajectory> {
        let obstacles = scene.build_environment_spheres();
        self.optimize(robot, &obstacles, start, goal)
    }

    /// Check if GPU is available without creating a full optimizer.
    pub fn gpu_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .is_some()
    }

    /// Access the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the GPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    fn create_fk_pipeline(&self) -> wgpu::ComputePipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fk_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fk_gpu.wgsl").into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fk_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("fk_main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn create_opt_pipeline(&self) -> wgpu::ComputePipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("opt_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/trajectory_optimize.wgsl").into(),
                ),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("opt_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("optimize_main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}

/// Create a storage buffer, using a 4-byte placeholder when data is empty
/// (wgpu requires non-zero buffer sizes).
fn create_storage_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let contents = if data.is_empty() { &[0u8; 4] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Read f32 values from a mapped staging buffer.
fn read_buffer_f32(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| GpuError::BufferMapping)?
        .map_err(|_| GpuError::BufferMapping)?;
    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
    drop(data);
    buffer.unmap();
    Ok(result)
}

/// Generate trajectory seeds as random perturbations of a linear interpolation
/// between start and goal.
///
/// Layout: `data[seed * timesteps * dof + t * dof + j]` (timestep-major per seed).
/// Seed 0 is the warm-start trajectory (if provided) or unperturbed linear interpolation.
/// First and last timesteps are always fixed to start and goal respectively.
fn generate_seeds(
    start: &[f64],
    goal: &[f64],
    robot: &Robot,
    num_seeds: usize,
    timesteps: usize,
    perturbation: f32,
    warm_start: Option<&Vec<Vec<f64>>>,
) -> Vec<f32> {
    let dof = start.len();
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0f32; num_seeds * timesteps * dof];

    for seed in 0..num_seeds {
        for t in 0..timesteps {
            let alpha = t as f64 / (timesteps - 1).max(1) as f64;
            for j in 0..dof {
                let base_val = if seed == 0 {
                    if let Some(ws) = warm_start {
                        // Resample warm-start trajectory to match timesteps
                        let ws_alpha = alpha * (ws.len() - 1).max(1) as f64;
                        let ws_idx = ws_alpha.floor() as usize;
                        let ws_frac = ws_alpha - ws_idx as f64;
                        if ws_idx + 1 < ws.len() {
                            ws[ws_idx][j] * (1.0 - ws_frac) + ws[ws_idx + 1][j] * ws_frac
                        } else {
                            ws[ws.len() - 1][j]
                        }
                    } else {
                        start[j] + alpha * (goal[j] - start[j])
                    }
                } else {
                    start[j] + alpha * (goal[j] - start[j])
                };

                let noise = if seed == 0 || t == 0 || t == timesteps - 1 {
                    // Seed 0 is unperturbed (warm-start or linear); endpoints are fixed
                    0.0
                } else {
                    rng.gen_range(-(perturbation as f64)..(perturbation as f64))
                };
                let val = (base_val + noise)
                    .clamp(robot.joint_limits[j].lower, robot.joint_limits[j].upper)
                    as f32;
                data[seed * timesteps * dof + t * dof + j] = val;
            }
        }
    }

    data
}

/// Prepare robot joint and sphere data for GPU buffers.
///
/// Returns (joint_origins as flat f32, joint_axes as flat f32, joint_types, local_spheres, num_joints).
fn prepare_robot_data(robot: &Robot) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<LocalSphereGpu>, usize) {
    let num_joints = robot.joints.len();

    let mut origins: Vec<f32> = Vec::with_capacity(num_joints * 16);
    let mut axes: Vec<f32> = Vec::with_capacity(num_joints * 4);
    let mut types: Vec<u32> = Vec::with_capacity(num_joints);

    for joint in &robot.joints {
        // Convert joint origin (Isometry3<f64>) to 4x4 column-major f32
        let m = joint.origin.0.to_homogeneous();
        for col in 0..4 {
            for row in 0..4 {
                origins.push(m[(row, col)] as f32);
            }
        }

        // Joint axis as vec4 (xyz + 0 padding)
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

    // Generate collision spheres from the robot model
    let sphere_model = RobotSphereModel::from_robot_default(robot);
    let mut local_spheres = Vec::new();

    for link_idx in 0..sphere_model.num_links {
        let (start, end) = sphere_model.link_ranges[link_idx];
        // Find the joint that has this link as its child — that joint's
        // accumulated transform positions this link in world space.
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
    fn default_config() {
        let config = GpuConfig::default();
        assert_eq!(config.num_seeds, 128);
        assert_eq!(config.timesteps, 32);
        assert_eq!(config.iterations, 100);
        assert!(config.collision_weight > 0.0);
        assert!(config.step_size > 0.0);
    }

    #[test]
    fn generate_seeds_unperturbed_is_linear() {
        let robot = kinetic_robot::Robot::from_name("franka_panda").unwrap();
        let dof = robot.dof;
        // Use mid-range values that are valid for all joints
        let start: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        let goal: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| l.lower + (l.upper - l.lower) * 0.75)
            .collect();
        let seeds = generate_seeds(&start, &goal, &robot, 4, 10, 0.3, None);

        // Seed 0 should be unperturbed linear interpolation
        for t in 0..10 {
            let alpha = t as f64 / 9.0;
            for j in 0..dof {
                let expected = (start[j] + alpha * (goal[j] - start[j])) as f32;
                let actual = seeds[t * dof + j];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "seed 0, t={}, j={}: expected {}, got {}",
                    t,
                    j,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn generate_seeds_endpoints_are_fixed() {
        let robot = kinetic_robot::Robot::from_name("franka_panda").unwrap();
        let dof = robot.dof;
        let start: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        let goal: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| l.lower + (l.upper - l.lower) * 0.75)
            .collect();
        let seeds = generate_seeds(&start, &goal, &robot, 16, 10, 0.5, None);

        for seed in 0..16 {
            for j in 0..dof {
                // First timestep = start
                let first = seeds[seed * 10 * dof + j];
                let expected_start = start[j] as f32;
                assert!(
                    (first - expected_start).abs() < 1e-4,
                    "seed {}, first timestep, joint {}: expected {}, got {}",
                    seed,
                    j,
                    expected_start,
                    first
                );
                // Last timestep = goal
                let last = seeds[seed * 10 * dof + 9 * dof + j];
                let expected_goal = goal[j] as f32;
                assert!(
                    (last - expected_goal).abs() < 1e-4,
                    "seed {}, last timestep, joint {}: expected {}, got {}",
                    seed,
                    j,
                    expected_goal,
                    last
                );
            }
        }
    }

    #[test]
    fn generate_seeds_within_joint_limits() {
        let robot = kinetic_robot::Robot::from_name("franka_panda").unwrap();
        let dof = robot.dof;
        let start: Vec<f64> = robot.joint_limits.iter().map(|l| l.lower).collect();
        let goal: Vec<f64> = robot.joint_limits.iter().map(|l| l.upper).collect();
        let seeds = generate_seeds(&start, &goal, &robot, 32, 20, 1.0, None);

        for seed in 0..32 {
            for t in 0..20 {
                for j in 0..dof {
                    let val = seeds[seed * 20 * dof + t * dof + j] as f64;
                    assert!(
                        val >= robot.joint_limits[j].lower - 1e-5
                            && val <= robot.joint_limits[j].upper + 1e-5,
                        "seed {}, t={}, j={}: {} not in [{}, {}]",
                        seed,
                        t,
                        j,
                        val,
                        robot.joint_limits[j].lower,
                        robot.joint_limits[j].upper
                    );
                }
            }
        }
    }

    #[test]
    fn prepare_robot_data_panda() {
        let robot = kinetic_robot::Robot::from_name("franka_panda").unwrap();
        let (origins, axes, types, _spheres, num_joints) = prepare_robot_data(&robot);

        assert_eq!(num_joints, robot.joints.len());
        assert_eq!(origins.len(), num_joints * 16);
        assert_eq!(axes.len(), num_joints * 4);
        assert_eq!(types.len(), num_joints);
        // Note: franka_panda URDF has no <collision> elements, so sphere count may be 0.
        // This is fine — the optimizer handles zero spheres gracefully.

        // Verify joint types make sense
        let num_fixed = types.iter().filter(|&&t| t == 3).count();
        let num_active = types.iter().filter(|&&t| t != 3).count();
        assert_eq!(num_active, robot.dof);
        assert_eq!(num_fixed, num_joints - robot.dof);
    }
}
