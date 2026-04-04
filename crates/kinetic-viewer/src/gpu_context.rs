//! wgpu device, surface, and depth/MSAA texture management.
//!
//! [`GpuContext`] owns the wgpu state needed for rendering: instance, adapter,
//! device, queue, surface, and the depth/MSAA resolve textures. It handles
//! surface (re)configuration on window resize and DPI changes.

use std::sync::Arc;

/// Errors from GPU initialization.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found — ensure a Vulkan/Metal/DX12 driver is installed")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("failed to create surface: {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
}

/// Holds all wgpu state needed for rendering.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface_format: wgpu::TextureFormat,
    pub depth_texture: wgpu::TextureView,
    pub msaa_texture: wgpu::TextureView,
    pub msaa_samples: u32,
    pub window: Arc<winit::window::Window>,
}

impl GpuContext {
    /// Create a new GPU context for the given window.
    ///
    /// Requests a high-performance adapter, creates the device, and configures
    /// the surface with the window's current size. Also creates depth and MSAA
    /// textures matching the surface dimensions.
    pub fn new(window: Arc<winit::window::Window>, msaa_samples: u32) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .ok_or(GpuError::NoAdapter)?;

        // Request POLYGON_MODE_LINE if the adapter supports it (needed for
        // the wireframe pipeline).
        let mut required = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            required |= wgpu::Features::POLYGON_MODE_LINE;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kinetic-viewer"),
                required_features: required,
                ..Default::default()
            },
            None,
        ))?;

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let depth_texture = Self::create_depth_texture(&device, width, height, msaa_samples);
        let msaa_texture = Self::create_msaa_texture(&device, width, height, msaa_samples, surface_format);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            surface_format,
            depth_texture,
            msaa_texture,
            msaa_samples,
            window,
        })
    }

    /// Reconfigure the surface and recreate depth/MSAA textures for a new size.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.surface_config.width && height == self.surface_config.height {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.depth_texture = Self::create_depth_texture(&self.device, width, height, self.msaa_samples);
        self.msaa_texture = Self::create_msaa_texture(
            &self.device,
            width,
            height,
            self.msaa_samples,
            self.surface_format,
        );
    }

    /// Current surface width.
    pub fn width(&self) -> u32 {
        self.surface_config.width
    }

    /// Current surface height.
    pub fn height(&self) -> u32 {
        self.surface_config.height
    }

    /// Aspect ratio (width / height).
    pub fn aspect(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height as f32
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}
