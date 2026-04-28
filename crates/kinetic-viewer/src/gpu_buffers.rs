//! GPU buffer management: mesh upload, uniform buffers, instance batching.
//!
//! [`GpuScene`] owns all GPU-side resources needed for rendering: uploaded meshes,
//! uniform buffers, instance buffers, and the bind group that ties them together.

use wgpu::util::DeviceExt;

use crate::pipeline::{LightUniforms, LineVertex, MeshVertex, Pipelines};
use crate::dryrun_renderer::{InstanceData, ViewUniforms};
use crate::{Camera, MeshData, MeshHandle, MeshRegistry, Projection};

/// A mesh uploaded to the GPU with vertex and index buffers.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

/// All GPU resources for the viewer scene.
pub struct GpuScene {
    /// Uploaded meshes, indexed by MeshHandle.
    meshes: Vec<GpuMesh>,

    /// Uniform buffer for view/projection matrices.
    pub view_uniform_buffer: wgpu::Buffer,
    /// Uniform buffer for lighting.
    pub light_uniform_buffer: wgpu::Buffer,

    /// Bind group for mesh pipeline (view + light uniforms).
    pub mesh_bind_group: wgpu::BindGroup,
    /// Bind group for line pipeline (view uniforms only).
    pub line_bind_group: wgpu::BindGroup,

    /// Shared instance buffer, reallocated as needed.
    pub instance_buffer: wgpu::Buffer,
    /// Current capacity (number of InstanceData entries).
    instance_capacity: usize,

    /// Line vertex buffer, reallocated as needed.
    pub line_buffer: wgpu::Buffer,
    /// Current line vertex capacity.
    line_capacity: usize,
}

/// Default initial capacity for instance and line buffers.
const INITIAL_INSTANCE_CAPACITY: usize = 256;
const INITIAL_LINE_CAPACITY: usize = 1024;

impl GpuScene {
    /// Create GPU scene with uniform buffers and bind groups.
    pub fn new(device: &wgpu::Device, pipelines: &Pipelines) -> Self {
        let view_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view_uniforms"),
            contents: bytemuck::bytes_of(&ViewUniforms {
                view_proj: [[0.0; 4]; 4],
                view: [[0.0; 4]; 4],
                camera_pos: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light_uniforms"),
            contents: bytemuck::bytes_of(&LightUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_bind_group"),
            layout: &pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Line bind group uses only the view uniforms.
        // We need a separate layout — but the line pipeline's layout only has binding 0.
        // Create it from the line pipeline's bind group layout (group 0).
        let line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line_bind_group"),
            layout: &pipelines.line_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_uniform_buffer.as_entire_binding(),
            }],
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: (INITIAL_INSTANCE_CAPACITY * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line_buffer"),
            size: (INITIAL_LINE_CAPACITY * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            meshes: Vec::new(),
            view_uniform_buffer,
            light_uniform_buffer,
            mesh_bind_group,
            line_bind_group,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            line_buffer,
            line_capacity: INITIAL_LINE_CAPACITY,
        }
    }

    /// Upload a single mesh to the GPU. Returns the index matching MeshHandle.
    pub fn upload_mesh(&mut self, device: &wgpu::Device, mesh: &MeshData) -> usize {
        // Interleave position + normal into MeshVertex array
        let vertices: Vec<MeshVertex> = mesh
            .vertices
            .iter()
            .zip(mesh.normals.iter())
            .map(|(p, n)| MeshVertex {
                position: *p,
                normal: *n,
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let id = self.meshes.len();
        self.meshes.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        });
        id
    }

    /// Upload all meshes from a registry.
    pub fn upload_registry(&mut self, device: &wgpu::Device, registry: &MeshRegistry) {
        for i in 0..registry.count() {
            if let Some(mesh) = registry.get(MeshHandle(i)) {
                self.upload_mesh(device, mesh);
            }
        }
    }

    /// Get a GPU mesh by handle.
    pub fn get_mesh(&self, handle: MeshHandle) -> Option<&GpuMesh> {
        self.meshes.get(handle.0)
    }

    /// Number of uploaded meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Compute ViewUniforms from a Camera and aspect ratio.
    pub fn update_view_uniforms_from_camera(
        &self,
        camera: &Camera,
        aspect: f32,
    ) -> ViewUniforms {
        let view = camera.view_matrix();
        let proj = match camera.projection {
            Projection::Perspective { fov_y, near, far } => {
                nalgebra::Matrix4::new_perspective(aspect, fov_y.to_radians(), near, far)
            }
            Projection::Orthographic { scale, near, far } => {
                nalgebra::Matrix4::new_orthographic(
                    -scale * aspect,
                    scale * aspect,
                    -scale,
                    scale,
                    near,
                    far,
                )
            }
        };
        let view_proj = proj * view;

        fn m2a(m: &nalgebra::Matrix4<f32>) -> [[f32; 4]; 4] {
            let s = m.as_slice();
            [
                [s[0], s[1], s[2], s[3]],
                [s[4], s[5], s[6], s[7]],
                [s[8], s[9], s[10], s[11]],
                [s[12], s[13], s[14], s[15]],
            ]
        }

        ViewUniforms {
            view_proj: m2a(&view_proj),
            view: m2a(&view),
            camera_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
        }
    }

    /// Update view uniforms on the GPU.
    pub fn update_view_uniforms(&self, queue: &wgpu::Queue, uniforms: &ViewUniforms) {
        queue.write_buffer(&self.view_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Update light uniforms on the GPU.
    pub fn update_light_uniforms(&self, queue: &wgpu::Queue, uniforms: &LightUniforms) {
        queue.write_buffer(&self.light_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Write instance data to the instance buffer, growing it if needed.
    /// Returns the number of instances written.
    pub fn write_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    ) -> u32 {
        if instances.is_empty() {
            return 0;
        }
        if instances.len() > self.instance_capacity {
            // Grow to next power of 2
            self.instance_capacity = instances.len().next_power_of_two();
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: (self.instance_capacity * std::mem::size_of::<InstanceData>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        instances.len() as u32
    }

    /// Write line vertices to the line buffer, growing it if needed.
    /// Returns the number of vertices written.
    pub fn write_lines(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[LineVertex],
    ) -> u32 {
        if vertices.is_empty() {
            return 0;
        }
        if vertices.len() > self.line_capacity {
            self.line_capacity = vertices.len().next_power_of_two();
            self.line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("line_buffer"),
                size: (self.line_capacity * std::mem::size_of::<LineVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.line_buffer, 0, bytemuck::cast_slice(vertices));
        vertices.len() as u32
    }
}
