//! GPU-accelerated Signed Distance Field construction.
//!
//! Builds a 3D voxel grid where each cell stores the minimum signed distance
//! to the nearest obstacle. Supports construction from:
//! - Obstacle spheres (fast, used during planning)
//! - Scene objects (convenience wrapper)
//! - Depth images (perception pipeline)

use crate::{GpuError, Result};
use bytemuck::{Pod, Zeroable};
use kinetic_collision::SpheresSoA;
use wgpu::util::DeviceExt;

/// GPU-side SDF parameters (matches `sdf_build.wgsl` SDFParams).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SdfParamsGpu {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    resolution: f32,
    nx: u32,
    ny: u32,
    nz: u32,
    num_spheres: u32,
}

/// A 3D signed distance field built on the GPU.
///
/// Each voxel stores the minimum signed distance to the nearest obstacle sphere.
/// Negative values indicate penetration into an obstacle. Supports trilinear
/// interpolation for smooth queries and finite-difference gradients.
pub struct SignedDistanceField {
    pub(crate) grid: Vec<f32>,
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) min_x: f32,
    pub(crate) min_y: f32,
    pub(crate) min_z: f32,
    pub(crate) resolution: f32,
}

impl SignedDistanceField {
    /// Build an SDF from obstacle spheres on the GPU.
    ///
    /// `min` and `max` define the workspace AABB. `resolution` is the voxel edge length.
    pub fn from_spheres(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &SpheresSoA,
        min: [f32; 3],
        max: [f32; 3],
        resolution: f32,
    ) -> Result<Self> {
        let nx = ((max[0] - min[0]) / resolution).ceil() as u32;
        let ny = ((max[1] - min[1]) / resolution).ceil() as u32;
        let nz = ((max[2] - min[2]) / resolution).ceil() as u32;
        let total_voxels = (nx as usize) * (ny as usize) * (nz as usize);

        if total_voxels == 0 {
            return Err(GpuError::InvalidConfig("SDF grid has zero voxels".into()));
        }

        // No obstacles → large positive distances everywhere
        if spheres.is_empty() {
            return Ok(Self {
                grid: vec![1e10_f32; total_voxels],
                nx,
                ny,
                nz,
                min_x: min[0],
                min_y: min[1],
                min_z: min[2],
                resolution,
            });
        }

        // Pack spheres as vec4<f32> (x, y, z, radius)
        let mut sphere_data: Vec<f32> = Vec::with_capacity(spheres.len() * 4);
        for i in 0..spheres.len() {
            sphere_data.push(spheres.x[i] as f32);
            sphere_data.push(spheres.y[i] as f32);
            sphere_data.push(spheres.z[i] as f32);
            sphere_data.push(spheres.radius[i] as f32);
        }

        let params = SdfParamsGpu {
            min_x: min[0],
            min_y: min[1],
            min_z: min[2],
            resolution,
            nx,
            ny,
            nz,
            num_spheres: spheres.len() as u32,
        };

        // Shader + pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_build_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_build.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_build_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("sdf_build_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Buffers
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let spheres_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_spheres"),
            contents: bytemuck::cast_slice(&sphere_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let grid_size = (total_voxels * std::mem::size_of::<f32>()) as u64;
        let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_grid"),
            size: grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups = (total_voxels as u32).div_ceil(64);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_build"),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy result to staging buffer for readback
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_staging"),
            size: grid_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&grid_buf, 0, &staging, 0, grid_size);
        queue.submit([encoder.finish()]);

        // Read back
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
        let grid: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(Self {
            grid,
            nx,
            ny,
            nz,
            min_x: min[0],
            min_y: min[1],
            min_z: min[2],
            resolution,
        })
    }

    /// Build an SDF from a planning scene's environment objects.
    ///
    /// Extracts obstacle spheres from the scene and builds the SDF on GPU.
    /// This is a convenience wrapper around `from_spheres()`.
    #[cfg(feature = "scene")]
    pub fn from_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &kinetic_scene::Scene,
        resolution: f32,
        workspace_bounds: [f32; 6],
    ) -> Result<Self> {
        let spheres = scene.build_environment_spheres();
        Self::from_spheres(
            device,
            queue,
            &spheres,
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
        )
    }

    /// Build an SDF from a depth image.
    ///
    /// Unprojects the depth image into 3D points using camera intrinsics,
    /// converts each point to a small obstacle sphere, then builds the SDF.
    ///
    /// # Arguments
    /// * `device`, `queue` — wgpu device and queue
    /// * `depth` — depth values in meters, row-major `[height * width]`
    /// * `width`, `height` — image dimensions
    /// * `fx`, `fy`, `cx`, `cy` — camera intrinsic parameters
    /// * `camera_pose` — 4x4 column-major camera-to-world transform
    /// * `point_radius` — radius of each point sphere (e.g. 0.01 for 1cm)
    /// * `max_depth` — ignore points beyond this depth
    /// * `resolution` — SDF voxel resolution
    /// * `workspace_bounds` — `[min_x, min_y, min_z, max_x, max_y, max_z]`
    #[allow(clippy::too_many_arguments)]
    pub fn from_depth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth: &[f32],
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        camera_pose: &[f32; 16],
        point_radius: f32,
        max_depth: f32,
        resolution: f32,
        workspace_bounds: [f32; 6],
    ) -> Result<Self> {
        if depth.len() != (width * height) as usize {
            return Err(GpuError::InvalidConfig(format!(
                "depth buffer length ({}) doesn't match {}x{}",
                depth.len(),
                width,
                height
            )));
        }

        // Unproject depth to 3D points and transform to world frame
        let mut spheres = SpheresSoA::new();
        let wb = &workspace_bounds;

        for v in 0..height {
            for u in 0..width {
                let d = depth[(v * width + u) as usize];
                if d <= 0.0 || d > max_depth || d.is_nan() {
                    continue;
                }

                // Unproject to camera frame
                let cam_x = (u as f32 - cx) * d / fx;
                let cam_y = (v as f32 - cy) * d / fy;
                let cam_z = d;

                // Transform to world frame (camera_pose is column-major 4x4)
                let m = camera_pose;
                let wx = m[0] * cam_x + m[4] * cam_y + m[8] * cam_z + m[12];
                let wy = m[1] * cam_x + m[5] * cam_y + m[9] * cam_z + m[13];
                let wz = m[2] * cam_x + m[6] * cam_y + m[10] * cam_z + m[14];

                // Skip if outside workspace bounds
                if wx < wb[0] || wx > wb[3] || wy < wb[1] || wy > wb[4] || wz < wb[2] || wz > wb[5]
                {
                    continue;
                }

                spheres.push(wx as f64, wy as f64, wz as f64, point_radius as f64, 0);
            }
        }

        Self::from_spheres(
            device,
            queue,
            &spheres,
            [wb[0], wb[1], wb[2]],
            [wb[3], wb[4], wb[5]],
            resolution,
        )
    }

    /// Build an SDF on CPU (no GPU required).
    ///
    /// Same algorithm as the GPU shader but runs on CPU. Useful for testing
    /// and when no GPU is available.
    pub fn from_spheres_cpu(
        spheres: &SpheresSoA,
        min: [f32; 3],
        max: [f32; 3],
        resolution: f32,
    ) -> Result<Self> {
        let nx = ((max[0] - min[0]) / resolution).ceil() as u32;
        let ny = ((max[1] - min[1]) / resolution).ceil() as u32;
        let nz = ((max[2] - min[2]) / resolution).ceil() as u32;
        let total_voxels = (nx as usize) * (ny as usize) * (nz as usize);

        if total_voxels == 0 {
            return Err(GpuError::InvalidConfig("SDF grid has zero voxels".into()));
        }

        if spheres.is_empty() {
            return Ok(Self {
                grid: vec![1e10_f32; total_voxels],
                nx,
                ny,
                nz,
                min_x: min[0],
                min_y: min[1],
                min_z: min[2],
                resolution,
            });
        }

        let mut grid = vec![1e10_f32; total_voxels];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let vx = min[0] + (ix as f32 + 0.5) * resolution;
                    let vy = min[1] + (iy as f32 + 0.5) * resolution;
                    let vz = min[2] + (iz as f32 + 0.5) * resolution;

                    let mut min_dist = 1e10_f32;
                    for i in 0..spheres.len() {
                        let dx = vx - spheres.x[i] as f32;
                        let dy = vy - spheres.y[i] as f32;
                        let dz = vz - spheres.z[i] as f32;
                        let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        let signed_dist = center_dist - spheres.radius[i] as f32;
                        min_dist = min_dist.min(signed_dist);
                    }

                    let idx = (iz * ny * nx + iy * nx + ix) as usize;
                    grid[idx] = min_dist;
                }
            }
        }

        Ok(Self {
            grid,
            nx,
            ny,
            nz,
            min_x: min[0],
            min_y: min[1],
            min_z: min[2],
            resolution,
        })
    }

    /// Query SDF at a world position using trilinear interpolation.
    ///
    /// Returns a smooth distance estimate between voxel centers.
    /// Points outside the grid return a large positive value (1e10).
    pub fn query(&self, x: f64, y: f64, z: f64) -> f64 {
        let inv_res = 1.0 / self.resolution as f64;
        // Continuous voxel coordinates (offset by 0.5 since voxel centers are at +0.5)
        let fx = (x - self.min_x as f64) * inv_res - 0.5;
        let fy = (y - self.min_y as f64) * inv_res - 0.5;
        let fz = (z - self.min_z as f64) * inv_res - 0.5;

        let ix = fx.floor() as i32;
        let iy = fy.floor() as i32;
        let iz = fz.floor() as i32;

        // If all 8 corners would be out of bounds, return large distance
        if ix < -1
            || iy < -1
            || iz < -1
            || ix + 1 >= self.nx as i32
            || iy + 1 >= self.ny as i32
            || iz + 1 >= self.nz as i32
        {
            return 1e10;
        }

        // Fractional part for interpolation
        let tx = fx - ix as f64;
        let ty = fy - iy as f64;
        let tz = fz - iz as f64;

        // Sample 8 corners with boundary clamping
        let v000 = self.sample_clamped(ix, iy, iz);
        let v100 = self.sample_clamped(ix + 1, iy, iz);
        let v010 = self.sample_clamped(ix, iy + 1, iz);
        let v110 = self.sample_clamped(ix + 1, iy + 1, iz);
        let v001 = self.sample_clamped(ix, iy, iz + 1);
        let v101 = self.sample_clamped(ix + 1, iy, iz + 1);
        let v011 = self.sample_clamped(ix, iy + 1, iz + 1);
        let v111 = self.sample_clamped(ix + 1, iy + 1, iz + 1);

        // Trilinear interpolation
        let c00 = v000 * (1.0 - tx) + v100 * tx;
        let c10 = v010 * (1.0 - tx) + v110 * tx;
        let c01 = v001 * (1.0 - tx) + v101 * tx;
        let c11 = v011 * (1.0 - tx) + v111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Query SDF at a world position using nearest-neighbor (no interpolation).
    ///
    /// Faster than `query()` but produces step-wise discontinuities.
    pub fn query_nearest(&self, x: f64, y: f64, z: f64) -> f64 {
        let inv_res = 1.0 / self.resolution as f64;
        let ix = ((x - self.min_x as f64) * inv_res) as i32;
        let iy = ((y - self.min_y as f64) * inv_res) as i32;
        let iz = ((z - self.min_z as f64) * inv_res) as i32;

        if ix < 0
            || iy < 0
            || iz < 0
            || (ix as u32) >= self.nx
            || (iy as u32) >= self.ny
            || (iz as u32) >= self.nz
        {
            return 1e10;
        }

        let idx = (iz as u32) * self.nx * self.ny + (iy as u32) * self.nx + (ix as u32);
        self.grid[idx as usize] as f64
    }

    /// Compute the SDF gradient at a world position via finite differences.
    ///
    /// Returns `[dSDF/dx, dSDF/dy, dSDF/dz]`. The gradient points away from
    /// obstacles (toward increasing distance). Useful for trajectory optimization.
    pub fn gradient(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let h = self.resolution as f64 * 0.5;

        let dx = self.query(x + h, y, z) - self.query(x - h, y, z);
        let dy = self.query(x, y + h, z) - self.query(x, y - h, z);
        let dz = self.query(x, y, z + h) - self.query(x, y, z - h);

        let inv_2h = 1.0 / (2.0 * h);
        [dx * inv_2h, dy * inv_2h, dz * inv_2h]
    }

    /// Sample a voxel with clamped indices (returns grid value or 1e10 for out-of-bounds).
    fn sample_clamped(&self, ix: i32, iy: i32, iz: i32) -> f64 {
        let cx = ix.clamp(0, self.nx as i32 - 1) as u32;
        let cy = iy.clamp(0, self.ny as i32 - 1) as u32;
        let cz = iz.clamp(0, self.nz as i32 - 1) as u32;
        let idx = cz * self.nx * self.ny + cy * self.nx + cx;
        self.grid[idx as usize] as f64
    }

    /// Total number of voxels in the grid.
    pub fn num_voxels(&self) -> usize {
        (self.nx as usize) * (self.ny as usize) * (self.nz as usize)
    }

    /// Grid dimensions (nx, ny, nz).
    pub fn dimensions(&self) -> (u32, u32, u32) {
        (self.nx, self.ny, self.nz)
    }

    /// Voxel resolution in meters.
    pub fn resolution(&self) -> f32 {
        self.resolution
    }

    /// Workspace minimum corner.
    pub fn min_corner(&self) -> [f32; 3] {
        [self.min_x, self.min_y, self.min_z]
    }

    /// Workspace maximum corner.
    pub fn max_corner(&self) -> [f32; 3] {
        [
            self.min_x + self.nx as f32 * self.resolution,
            self.min_y + self.ny as f32 * self.resolution,
            self.min_z + self.nz as f32 * self.resolution,
        ]
    }

    /// Raw grid data (for GPU upload or debugging).
    pub fn grid_data(&self) -> &[f32] {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_spheres_returns_large_distances() {
        let sdf = SignedDistanceField {
            grid: vec![1e10; 8],
            nx: 2,
            ny: 2,
            nz: 2,
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            resolution: 1.0,
        };
        assert!(sdf.query(0.5, 0.5, 0.5) > 1e9);
        assert_eq!(sdf.num_voxels(), 8);
        assert_eq!(sdf.dimensions(), (2, 2, 2));
    }

    #[test]
    fn query_out_of_bounds() {
        let sdf = SignedDistanceField {
            grid: vec![0.5; 8],
            nx: 2,
            ny: 2,
            nz: 2,
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            resolution: 1.0,
        };
        assert!(sdf.query(-2.0, 0.0, 0.0) > 1e9);
        assert!(sdf.query(0.0, 4.0, 0.0) > 1e9);
        assert!(sdf.query(0.0, 0.0, 6.0) > 1e9);
    }

    #[test]
    fn query_nearest_within_grid() {
        let mut grid = vec![1.0_f32; 27]; // 3x3x3
        grid[13] = -0.5; // center voxel (1,1,1)
        let sdf = SignedDistanceField {
            grid,
            nx: 3,
            ny: 3,
            nz: 3,
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            resolution: 1.0,
        };
        assert!((sdf.query_nearest(1.0, 1.0, 1.0) - (-0.5)).abs() < 1e-6);
        assert!((sdf.query_nearest(0.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trilinear_interpolation_smooths() {
        // Create a 3x3x3 grid with a gradient: value = x_index
        let mut grid = vec![0.0_f32; 27];
        for iz in 0..3u32 {
            for iy in 0..3u32 {
                for ix in 0..3u32 {
                    grid[(iz * 9 + iy * 3 + ix) as usize] = ix as f32;
                }
            }
        }
        let sdf = SignedDistanceField {
            grid,
            nx: 3,
            ny: 3,
            nz: 3,
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            resolution: 1.0,
        };
        // Query at midpoint between voxel 0 and 1 (x = 1.0 is between centers 0.5 and 1.5)
        let val = sdf.query(1.0, 1.5, 1.5);
        // Should be interpolated between 0.0 and 1.0 → ~0.5
        assert!(
            (val - 0.5).abs() < 0.1,
            "trilinear at midpoint should be ~0.5, got {}",
            val
        );
    }

    #[test]
    fn gradient_points_away_from_obstacle() {
        // Sphere at origin with radius 0.2
        let mut spheres = SpheresSoA::new();
        spheres.push(0.0, 0.0, 0.0, 0.2, 0);

        let sdf = SignedDistanceField::from_spheres_cpu(
            &spheres,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            0.05,
        )
        .unwrap();

        // Gradient at (0.5, 0, 0) should point in +x direction
        let grad = sdf.gradient(0.5, 0.0, 0.0);
        assert!(
            grad[0] > 0.0,
            "gradient x should be positive, got {}",
            grad[0]
        );
        assert!(
            grad[0].abs() > grad[1].abs(),
            "gradient should be primarily in x"
        );

        // Gradient at (-0.5, 0, 0) should point in -x direction
        let grad_neg = sdf.gradient(-0.5, 0.0, 0.0);
        assert!(
            grad_neg[0] < 0.0,
            "gradient x should be negative, got {}",
            grad_neg[0]
        );
    }

    #[test]
    fn cpu_sdf_matches_analytical() {
        // Single sphere at (0.5, 0.5, 0.5) with radius 0.1
        let mut spheres = SpheresSoA::new();
        spheres.push(0.5, 0.5, 0.5, 0.1, 0);

        let sdf =
            SignedDistanceField::from_spheres_cpu(&spheres, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.02)
                .unwrap();

        // Query at center of sphere — should be negative (inside)
        let at_center = sdf.query(0.5, 0.5, 0.5);
        assert!(
            at_center < 0.0,
            "inside sphere should be negative, got {}",
            at_center
        );

        // Query far away — should be positive
        let far_away = sdf.query(0.9, 0.9, 0.9);
        assert!(
            far_away > 0.0,
            "far away should be positive, got {}",
            far_away
        );

        // Query on surface — should be near zero
        let on_surface = sdf.query(0.6, 0.5, 0.5);
        assert!(
            on_surface.abs() < 0.05,
            "on surface should be near zero, got {}",
            on_surface
        );
    }

    #[test]
    fn from_depth_basic() {
        // Create a simple 4x4 depth image with uniform depth
        let width = 4u32;
        let height = 4u32;
        let depth: Vec<f32> = vec![1.0; (width * height) as usize];

        // Camera at origin looking along +z
        let camera_pose: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ];

        // Simple intrinsics
        let fx = 2.0;
        let fy = 2.0;
        let cx = 2.0;
        let cy = 2.0;

        let sdf = SignedDistanceField::from_depth_cpu(
            &depth,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            &camera_pose,
            0.05,
            5.0,
            0.1,
            [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
        )
        .unwrap();

        // Should have created some voxels
        assert!(sdf.num_voxels() > 0);
        // Query near where the depth points should be (z=1.0)
        let val = sdf.query(0.0, 0.0, 1.0);
        assert!(
            val < 0.2,
            "near depth points should have small distance, got {}",
            val
        );
    }

    #[test]
    fn min_max_corner() {
        let sdf = SignedDistanceField {
            grid: vec![0.0; 1000],
            nx: 10,
            ny: 10,
            nz: 10,
            min_x: -0.5,
            min_y: -0.5,
            min_z: 0.0,
            resolution: 0.1,
        };
        let min = sdf.min_corner();
        let max = sdf.max_corner();
        assert_eq!(min, [-0.5, -0.5, 0.0]);
        assert!((max[0] - 0.5).abs() < 1e-5);
        assert!((max[1] - 0.5).abs() < 1e-5);
        assert!((max[2] - 1.0).abs() < 1e-5);
    }
}

impl SignedDistanceField {
    /// CPU-only depth-to-SDF conversion (no GPU required).
    #[allow(clippy::too_many_arguments)]
    pub fn from_depth_cpu(
        depth: &[f32],
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        camera_pose: &[f32; 16],
        point_radius: f32,
        max_depth: f32,
        resolution: f32,
        workspace_bounds: [f32; 6],
    ) -> Result<Self> {
        if depth.len() != (width * height) as usize {
            return Err(GpuError::InvalidConfig(format!(
                "depth buffer length ({}) doesn't match {}x{}",
                depth.len(),
                width,
                height
            )));
        }

        let mut spheres = SpheresSoA::new();
        let wb = &workspace_bounds;

        for v in 0..height {
            for u in 0..width {
                let d = depth[(v * width + u) as usize];
                if d <= 0.0 || d > max_depth || d.is_nan() {
                    continue;
                }

                let cam_x = (u as f32 - cx) * d / fx;
                let cam_y = (v as f32 - cy) * d / fy;
                let cam_z = d;

                let m = camera_pose;
                let wx = m[0] * cam_x + m[4] * cam_y + m[8] * cam_z + m[12];
                let wy = m[1] * cam_x + m[5] * cam_y + m[9] * cam_z + m[13];
                let wz = m[2] * cam_x + m[6] * cam_y + m[10] * cam_z + m[14];

                if wx < wb[0] || wx > wb[3] || wy < wb[1] || wy > wb[4] || wz < wb[2] || wz > wb[5]
                {
                    continue;
                }

                spheres.push(wx as f64, wy as f64, wz as f64, point_radius as f64, 0);
            }
        }

        Self::from_spheres_cpu(
            &spheres,
            [wb[0], wb[1], wb[2]],
            [wb[3], wb[4], wb[5]],
            resolution,
        )
    }
}
