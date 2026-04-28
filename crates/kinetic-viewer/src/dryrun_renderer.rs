//! Data model and dry-run renderer for testing without a GPU.
//!
//! This module provides the stats-only renderer used when the `visual` feature
//! is disabled. For actual GPU rendering, see [`gpu_buffers`] and [`app`].
//! The types [`ViewUniforms`] and [`InstanceData`] are shared with the real
//! GPU pipeline.
//!
//! # Architecture
//!
//! ```text
//! RenderCommand → WgpuRenderer → FrameStats (no GPU)
//! ```

use super::{Camera, MeshData, MeshHandle, MeshRegistry, RenderCommand};
#[cfg(test)]
use super::Material;
use nalgebra::Matrix4;

/// GPU-side mesh: uploaded vertex/index buffers.
#[derive(Debug)]
pub struct GpuMesh {
    pub vertex_count: usize,
    pub index_count: usize,
    /// Opaque handle (would be wgpu::Buffer in real impl).
    pub id: usize,
}

/// Render statistics for one frame.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub draw_calls: usize,
    pub triangles: usize,
    pub points: usize,
    pub lines: usize,
    pub upload_bytes: usize,
    pub frame_time_us: u64,
}

/// Configuration for the wgpu renderer.
#[derive(Debug, Clone)]
pub struct RendererConfig {
    /// Window width.
    pub width: u32,
    /// Window height.
    pub height: u32,
    /// MSAA sample count (1 = no AA, 4 = 4x MSAA).
    pub msaa_samples: u32,
    /// Enable shadow mapping.
    pub shadows: bool,
    /// Max instanced draw count.
    pub max_instances: usize,
    /// VSync mode.
    pub vsync: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            msaa_samples: 4,
            shadows: true,
            max_instances: 10000,
            vsync: true,
        }
    }
}

/// Light source for the scene.
#[derive(Debug, Clone)]
pub struct Light {
    pub light_type: LightType,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub enum LightType {
    Ambient,
    Directional { direction: [f32; 3] },
    Point { position: [f32; 3], range: f32 },
}

impl Light {
    pub fn ambient(intensity: f32) -> Self {
        Self { light_type: LightType::Ambient, color: [1.0; 3], intensity }
    }
    pub fn directional(direction: [f32; 3], intensity: f32) -> Self {
        Self { light_type: LightType::Directional { direction }, color: [1.0; 3], intensity }
    }
}

/// Uniform data for the vertex shader.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ViewUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

/// Per-instance data for instanced rendering.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

/// The wgpu renderer (data model — actual wgpu calls are behind feature gate).
///
/// This struct holds the render state and processes RenderCommands into
/// draw calls. Without the `visual` feature, it operates in "dry run" mode
/// collecting stats without touching the GPU.
pub struct WgpuRenderer {
    config: RendererConfig,
    gpu_meshes: Vec<GpuMesh>,
    lights: Vec<Light>,
    frame_stats: FrameStats,
    /// Instanced draw batches: (mesh_id, instances).
    instance_batches: Vec<(usize, Vec<InstanceData>)>,
}

impl WgpuRenderer {
    /// Create a new renderer.
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            gpu_meshes: Vec::new(),
            lights: vec![
                Light::ambient(0.3),
                Light::directional([0.5, 1.0, 0.3], 1.0),
            ],
            frame_stats: FrameStats::default(),
            instance_batches: Vec::new(),
        }
    }

    /// Upload a mesh to the GPU (simulated — creates GpuMesh handle).
    pub fn upload_mesh(&mut self, mesh: &MeshData) -> usize {
        let id = self.gpu_meshes.len();
        self.gpu_meshes.push(GpuMesh {
            vertex_count: mesh.num_vertices(),
            index_count: mesh.indices.len(),
            id,
        });
        id
    }

    /// Upload all meshes from a registry.
    pub fn upload_registry(&mut self, registry: &MeshRegistry) {
        for i in 0..registry.count() {
            if let Some(mesh) = registry.get(MeshHandle(i)) {
                self.upload_mesh(mesh);
            }
        }
    }

    /// Set lights.
    pub fn set_lights(&mut self, lights: Vec<Light>) {
        self.lights = lights;
    }

    /// Compute view uniforms from camera.
    pub fn compute_view_uniforms(&self, camera: &Camera, aspect: f32) -> ViewUniforms {
        let view = camera.view_matrix();
        let proj = match camera.projection {
            super::Projection::Perspective { fov_y, near, far } => {
                nalgebra::Matrix4::new_perspective(aspect, fov_y.to_radians(), near, far)
            }
            super::Projection::Orthographic { scale, near, far } => {
                nalgebra::Matrix4::new_orthographic(-scale * aspect, scale * aspect, -scale, scale, near, far)
            }
        };
        let view_proj = proj * view;

        ViewUniforms {
            view_proj: matrix4_to_array(&view_proj),
            view: matrix4_to_array(&view),
            camera_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
        }
    }

    /// Process render commands into draw calls (frame rendering).
    ///
    /// In dry-run mode (no GPU), just collects statistics.
    pub fn render_frame(&mut self, commands: &[RenderCommand], _camera: &Camera) {
        let mut stats = FrameStats::default();
        let start = std::time::Instant::now();

        self.instance_batches.clear();

        for cmd in commands {
            match cmd {
                RenderCommand::DrawMesh { handle, transform, material } => {
                    let gpu_mesh = self.gpu_meshes.get(handle.0);
                    if let Some(gm) = gpu_mesh {
                        stats.draw_calls += 1;
                        stats.triangles += gm.index_count / 3;

                        // Collect for instanced rendering
                        let instance = InstanceData {
                            model: matrix4_to_array(transform),
                            color: material.color,
                        };

                        if let Some(batch) = self.instance_batches.iter_mut().find(|(id, _)| *id == handle.0) {
                            batch.1.push(instance);
                        } else {
                            self.instance_batches.push((handle.0, vec![instance]));
                        }
                    }
                }
                RenderCommand::DrawLine { .. } => {
                    stats.draw_calls += 1;
                    stats.lines += 1;
                }
                RenderCommand::DrawPoint { .. } => {
                    stats.points += 1;
                }
                RenderCommand::DrawGrid { divisions, .. } => {
                    stats.draw_calls += 1;
                    stats.lines += divisions * 4;
                }
                RenderCommand::DrawAxes { .. } => {
                    stats.draw_calls += 1;
                    stats.lines += 3;
                }
            }
        }

        stats.frame_time_us = start.elapsed().as_micros() as u64;
        self.frame_stats = stats;
    }

    /// Get last frame statistics.
    pub fn frame_stats(&self) -> &FrameStats {
        &self.frame_stats
    }

    /// Number of uploaded GPU meshes.
    pub fn gpu_mesh_count(&self) -> usize {
        self.gpu_meshes.len()
    }

    /// Number of instance batches in last frame.
    pub fn instance_batch_count(&self) -> usize {
        self.instance_batches.len()
    }

    /// Total instances across all batches.
    pub fn total_instances(&self) -> usize {
        self.instance_batches.iter().map(|(_, insts)| insts.len()).sum()
    }

    /// Renderer config.
    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    /// Resize the render surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
    }

    /// Take a screenshot (returns raw RGBA pixel data).
    ///
    /// In dry-run mode, returns a solid-color image.
    pub fn screenshot(&self) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let mut pixels = vec![0u8; w * h * 4];
        // Fill with background color
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                pixels[idx] = 38;     // R
                pixels[idx + 1] = 38; // G
                pixels[idx + 2] = 46; // B
                pixels[idx + 3] = 255; // A
            }
        }
        pixels
    }

    /// Render text label at a 3D position.
    ///
    /// Stub in dry-run mode — increments draw call counter only.
    /// Real text rendering requires a font atlas (e.g., glyph-brush).
    pub fn draw_text(&mut self, _text: &str, _world_pos: [f32; 3], _color: [f32; 4]) {
        self.frame_stats.draw_calls += 1;
    }
}

fn matrix4_to_array(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_creation() {
        let renderer = WgpuRenderer::new(RendererConfig::default());
        assert_eq!(renderer.gpu_mesh_count(), 0);
        assert_eq!(renderer.config().width, 1280);
    }

    #[test]
    fn upload_mesh() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        let mesh = MeshData::cube();
        let id = renderer.upload_mesh(&mesh);
        assert_eq!(id, 0);
        assert_eq!(renderer.gpu_mesh_count(), 1);
    }

    #[test]
    fn render_frame_stats() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        let mesh = MeshData::cube();
        renderer.upload_mesh(&mesh);

        let commands = vec![
            RenderCommand::DrawMesh {
                handle: MeshHandle(0),
                transform: Matrix4::identity(),
                material: Material::default(),
            },
            RenderCommand::DrawGrid { size: 2.0, divisions: 10 },
            RenderCommand::DrawAxes { length: 0.5 },
        ];

        let camera = Camera::perspective([2.0, 1.0, 2.0], [0.0, 0.0, 0.0], 45.0);
        renderer.render_frame(&commands, &camera);

        let stats = renderer.frame_stats();
        assert_eq!(stats.draw_calls, 3);
        assert!(stats.triangles > 0);
    }

    #[test]
    fn instanced_batching() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        renderer.upload_mesh(&MeshData::cube());

        // Multiple draws of the same mesh → should batch
        let commands: Vec<_> = (0..5).map(|i| RenderCommand::DrawMesh {
            handle: MeshHandle(0),
            transform: Matrix4::new_translation(&nalgebra::Vector3::new(i as f32, 0.0, 0.0)),
            material: Material::solid(1.0, 0.0, 0.0),
        }).collect();

        let camera = Camera::perspective([5.0, 3.0, 5.0], [0.0, 0.0, 0.0], 45.0);
        renderer.render_frame(&commands, &camera);

        assert_eq!(renderer.instance_batch_count(), 1, "Same mesh should batch");
        assert_eq!(renderer.total_instances(), 5);
    }

    #[test]
    fn screenshot_returns_pixels() {
        let renderer = WgpuRenderer::new(RendererConfig { width: 64, height: 48, ..Default::default() });
        let pixels = renderer.screenshot();
        assert_eq!(pixels.len(), 64 * 48 * 4);
    }

    #[test]
    fn view_uniforms() {
        let renderer = WgpuRenderer::new(RendererConfig::default());
        let camera = Camera::perspective([2.0, 1.0, 2.0], [0.0, 0.0, 0.0], 45.0);
        let uniforms = renderer.compute_view_uniforms(&camera, 16.0 / 9.0);
        // view_proj should be non-zero
        assert!(uniforms.view_proj[0][0] != 0.0);
    }

    #[test]
    fn resize() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        renderer.resize(1920, 1080);
        assert_eq!(renderer.config().width, 1920);
        assert_eq!(renderer.config().height, 1080);
    }

    #[test]
    fn lights() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        assert_eq!(renderer.lights.len(), 2); // ambient + directional default

        renderer.set_lights(vec![
            Light::ambient(0.5),
            Light::directional([1.0, -1.0, 0.5], 2.0),
        ]);
        assert_eq!(renderer.lights.len(), 2);
    }

    #[test]
    fn text_rendering_stub() {
        let mut renderer = WgpuRenderer::new(RendererConfig::default());
        renderer.draw_text("Hello", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(renderer.frame_stats().draw_calls, 1);
    }
}
