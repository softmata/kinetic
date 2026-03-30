//! GPU trajectory optimization for KINETIC.
//!
//! cuRobo-style parallel-seed trajectory optimization via wgpu compute
//! shaders. Runs on NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan),
//! and Apple Silicon (Metal). Includes GPU SDF construction.

pub mod batch_fk;
pub mod collision_check;
pub mod cpu_fallback;
mod optimizer;
mod sdf;

pub use batch_fk::{batch_fk_gpu, BatchFkResult};
pub use collision_check::{BatchCollisionResult, CpuCollisionChecker, GpuCollisionChecker};
pub use cpu_fallback::CpuOptimizer;
pub use optimizer::{GpuConfig, GpuOptimizer};
pub use sdf::SignedDistanceField;

use thiserror::Error;

/// Errors from GPU trajectory optimization.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("GPU buffer mapping failed")]
    BufferMapping,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;
