//! Test utilities: headless rendering and screenshot comparison.
//!
//! [`HeadlessRenderer`] renders scenes to an offscreen texture without a window.
//! [`compare_screenshots`] computes per-pixel RMSE between two images for
//! visual regression testing.

use crate::gpu_buffers::GpuScene;
use crate::pipeline::{LightUniforms, Pipelines};
use crate::dryrun_renderer::ViewUniforms;
use crate::{
    collect_render_commands, Camera, MeshHandle, MeshRegistry, RenderCommand, SceneNode,
    ViewerSettings,
};
use nalgebra::Matrix4;

/// Headless renderer: renders to an offscreen texture, no window needed.
pub struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    scene: GpuScene,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    depth_texture: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl HeadlessRenderer {
    /// Create a headless renderer. Works on CI with software rasterization (lavapipe).
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or("No GPU adapter found for headless rendering")?;

        // Request POLYGON_MODE_LINE if the adapter supports it (needed for
        // the wireframe pipeline). Falls back gracefully on software rasterizers
        // that don't advertise the feature.
        let mut required = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            required |= wgpu::Features::POLYGON_MODE_LINE;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("headless_renderer"),
                required_features: required,
                ..Default::default()
            },
            None,
        ))
        .map_err(|e| format!("Device request failed: {e}"))?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // No MSAA for headless (simpler, deterministic)
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let depth_texture = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("headless_depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pipelines = Pipelines::new(&device, format, 1); // 1x sample for headless
        let scene = GpuScene::new(&device, &pipelines);

        Ok(Self {
            device,
            queue,
            pipelines,
            scene,
            width,
            height,
            texture,
            depth_texture,
            format,
        })
    }

    /// Upload meshes from a registry.
    pub fn upload_meshes(&mut self, registry: &MeshRegistry) {
        self.scene.upload_registry(&self.device, registry);
    }

    /// Render a scene and return RGBA pixel data.
    pub fn render(
        &mut self,
        scene_root: &SceneNode,
        camera: &Camera,
        settings: &ViewerSettings,
    ) -> Vec<u8> {
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update uniforms
        let aspect = self.width as f32 / self.height as f32;
        let view_uniforms = self.scene.update_view_uniforms_from_camera(camera, aspect);
        self.scene.update_view_uniforms(&self.queue, &view_uniforms);
        self.scene
            .update_light_uniforms(&self.queue, &LightUniforms::default());

        // Collect commands
        let mut commands = Vec::new();
        collect_render_commands(scene_root, &Matrix4::identity(), &mut commands);
        if settings.show_grid {
            commands.push(RenderCommand::DrawGrid {
                size: settings.grid_size,
                divisions: settings.grid_divisions,
            });
        }
        if settings.show_axes {
            commands.push(RenderCommand::DrawAxes { length: 0.5 });
        }

        // Build draw data
        let (batches, line_vertices) = build_draw_data(&commands);
        let line_count =
            self.scene
                .write_lines(&self.device, &self.queue, &line_vertices);

        // Render pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: settings.background_color[0] as f64,
                            g: settings.background_color[1] as f64,
                            b: settings.background_color[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            for batch in &batches {
                if self.scene.get_mesh(MeshHandle(batch.mesh_id)).is_none() {
                    continue;
                }
                let instance_count = self.scene.write_instances(
                    &self.device,
                    &self.queue,
                    &batch.instances,
                );
                let gpu_mesh = self.scene.get_mesh(MeshHandle(batch.mesh_id)).unwrap();

                if batch.wireframe {
                    pass.set_pipeline(&self.pipelines.wireframe_pipeline);
                } else {
                    pass.set_pipeline(&self.pipelines.mesh_pipeline);
                }
                pass.set_bind_group(0, &self.scene.mesh_bind_group, &[]);
                pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, self.scene.instance_buffer.slice(..));
                pass.set_index_buffer(
                    gpu_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..instance_count);
            }

            if line_count > 0 {
                pass.set_pipeline(&self.pipelines.line_pipeline);
                pass.set_bind_group(0, &self.scene.line_bind_group, &[]);
                pass.set_vertex_buffer(0, self.scene.line_buffer.slice(..));
                pass.draw(0..line_count, 0..1);
            }
        }

        // Copy texture to buffer for readback
        let bytes_per_pixel = 4u32;
        let unpadded_row = self.width * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255; // wgpu requires 256-byte row alignment

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: (padded_row * self.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back pixels
        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Remove row padding
        let mut pixels = Vec::with_capacity((self.width * self.height * bytes_per_pixel) as usize);
        for row in 0..self.height {
            let start = (row * padded_row) as usize;
            let end = start + unpadded_row as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        pixels
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Result of comparing two screenshots.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Root mean squared error per channel (0-255 scale).
    pub rmse: f64,
    /// Maximum per-pixel difference.
    pub max_diff: u8,
    /// Number of pixels that differ by more than the threshold.
    pub changed_pixels: usize,
    /// Total pixels.
    pub total_pixels: usize,
}

/// Compare two RGBA pixel buffers of the same dimensions.
pub fn compare_screenshots(
    actual: &[u8],
    expected: &[u8],
    width: u32,
    height: u32,
    threshold: u8,
) -> ComparisonResult {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), (width * height * 4) as usize);

    let total_pixels = (width * height) as usize;
    let mut sum_sq = 0.0f64;
    let mut max_diff = 0u8;
    let mut changed = 0usize;

    for i in 0..actual.len() {
        let diff = (actual[i] as i16 - expected[i] as i16).unsigned_abs() as u8;
        sum_sq += (diff as f64) * (diff as f64);
        if diff > max_diff {
            max_diff = diff;
        }
        // Count pixel as changed if any channel exceeds threshold
        if i % 4 == 0 {
            let r = (actual[i] as i16 - expected[i] as i16).unsigned_abs() as u8;
            let g = (actual[i + 1] as i16 - expected[i + 1] as i16).unsigned_abs() as u8;
            let b = (actual[i + 2] as i16 - expected[i + 2] as i16).unsigned_abs() as u8;
            if r > threshold || g > threshold || b > threshold {
                changed += 1;
            }
        }
    }

    let rmse = (sum_sq / actual.len() as f64).sqrt();

    ComparisonResult {
        rmse,
        max_diff,
        changed_pixels: changed,
        total_pixels,
    }
}

/// Generate a diff image (green=match, red=mismatch).
pub fn diff_image(actual: &[u8], expected: &[u8], width: u32, height: u32) -> Vec<u8> {
    assert_eq!(actual.len(), expected.len());
    let mut diff = vec![0u8; actual.len()];

    for i in (0..actual.len()).step_by(4) {
        let r = (actual[i] as i16 - expected[i] as i16).unsigned_abs() as u8;
        let g = (actual[i + 1] as i16 - expected[i + 1] as i16).unsigned_abs() as u8;
        let b = (actual[i + 2] as i16 - expected[i + 2] as i16).unsigned_abs() as u8;
        let max_ch = r.max(g).max(b);

        if max_ch < 3 {
            // Match: dark green
            diff[i] = 0;
            diff[i + 1] = 40;
            diff[i + 2] = 0;
        } else {
            // Mismatch: red, intensity = magnitude
            diff[i] = max_ch.saturating_mul(2);
            diff[i + 1] = 0;
            diff[i + 2] = 0;
        }
        diff[i + 3] = 255;
    }

    diff
}

// --- Internal: reuse draw data building from app.rs ---

use crate::pipeline::LineVertex;
use crate::dryrun_renderer::InstanceData;

struct DrawBatch {
    mesh_id: usize,
    wireframe: bool,
    instances: Vec<InstanceData>,
}

fn build_draw_data(commands: &[RenderCommand]) -> (Vec<DrawBatch>, Vec<LineVertex>) {
    let mut batches: Vec<DrawBatch> = Vec::new();
    let mut line_vertices: Vec<LineVertex> = Vec::new();

    for cmd in commands {
        match cmd {
            RenderCommand::DrawMesh { handle, transform, material } => {
                let instance = InstanceData {
                    model: m2a(transform),
                    color: material.color,
                };
                let wireframe = material.wireframe;
                let mesh_id = handle.0;
                if let Some(batch) = batches.iter_mut().find(|b| b.mesh_id == mesh_id && b.wireframe == wireframe) {
                    batch.instances.push(instance);
                } else {
                    batches.push(DrawBatch { mesh_id, wireframe, instances: vec![instance] });
                }
            }
            RenderCommand::DrawLine { start, end, color } => {
                line_vertices.push(LineVertex { position: *start, color: *color });
                line_vertices.push(LineVertex { position: *end, color: *color });
            }
            RenderCommand::DrawGrid { size, divisions } => {
                let half = *size;
                let step = (2.0 * half) / *divisions as f32;
                let color = [0.3, 0.3, 0.3, 0.5];
                for i in 0..=*divisions {
                    let pos = -half + step * i as f32;
                    line_vertices.push(LineVertex { position: [pos, 0.0, -half], color });
                    line_vertices.push(LineVertex { position: [pos, 0.0, half], color });
                    line_vertices.push(LineVertex { position: [-half, 0.0, pos], color });
                    line_vertices.push(LineVertex { position: [half, 0.0, pos], color });
                }
            }
            RenderCommand::DrawAxes { length } => {
                let l = *length;
                line_vertices.push(LineVertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0, 1.0] });
                line_vertices.push(LineVertex { position: [l, 0.0, 0.0], color: [1.0, 0.0, 0.0, 1.0] });
                line_vertices.push(LineVertex { position: [0.0, 0.0, 0.0], color: [0.0, 1.0, 0.0, 1.0] });
                line_vertices.push(LineVertex { position: [0.0, l, 0.0], color: [0.0, 1.0, 0.0, 1.0] });
                line_vertices.push(LineVertex { position: [0.0, 0.0, 0.0], color: [0.0, 0.0, 1.0, 1.0] });
                line_vertices.push(LineVertex { position: [0.0, 0.0, l], color: [0.0, 0.0, 1.0, 1.0] });
            }
            RenderCommand::DrawPoint { .. } => {}
        }
    }
    (batches, line_vertices)
}

fn m2a(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [[s[0],s[1],s[2],s[3]], [s[4],s[5],s[6],s[7]], [s[8],s[9],s[10],s[11]], [s[12],s[13],s[14],s[15]]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_identical_images() {
        let pixels = vec![128u8; 4 * 4 * 4]; // 4x4 RGBA
        let result = compare_screenshots(&pixels, &pixels, 4, 4, 2);
        assert_eq!(result.rmse, 0.0);
        assert_eq!(result.max_diff, 0);
        assert_eq!(result.changed_pixels, 0);
    }

    #[test]
    fn compare_different_images() {
        let actual = vec![255u8; 4 * 4 * 4];
        let expected = vec![0u8; 4 * 4 * 4];
        let result = compare_screenshots(&actual, &expected, 4, 4, 2);
        assert!(result.rmse > 100.0);
        assert_eq!(result.max_diff, 255);
        assert_eq!(result.changed_pixels, 16); // all pixels differ
    }

    #[test]
    fn diff_image_generates_correct_size() {
        let a = vec![100u8; 4 * 4 * 4];
        let b = vec![100u8; 4 * 4 * 4];
        let diff = diff_image(&a, &b, 4, 4);
        assert_eq!(diff.len(), a.len());
        // All match → green tint
        assert!(diff[1] > 0); // green channel
        assert_eq!(diff[0], 0); // red channel = 0 (match)
    }

    #[test]
    fn diff_image_shows_mismatch_in_red() {
        let a = vec![255u8; 4 * 4 * 4];
        let mut b = vec![255u8; 4 * 4 * 4];
        b[0] = 0; // first pixel R channel differs
        let diff = diff_image(&a, &b, 4, 4);
        // First pixel should be red (mismatch)
        assert!(diff[0] > 0); // red channel > 0
    }
}
