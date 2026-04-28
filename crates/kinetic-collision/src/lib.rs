//! SIMD-vectorized collision detection for KINETIC.
//!
//! Two-tier collision checking:
//! 1. CAPT (Collision-Affording Point Tree) for <100ns broadphase queries
//! 2. Sphere-sphere distance with SIMD acceleration (AVX2/NEON/scalar)
//!
//! # Architecture
//!
//! Robot link geometry is approximated by bounding spheres stored in
//! Structure-of-Arrays (SoA) layout for efficient SIMD vectorization.
//!
//! - [`SpheresSoA`]: Core SoA sphere storage
//! - [`RobotSphereModel`]: Pre-computed spheres in local link frames
//! - [`RobotSpheres`]: World-frame spheres updated from FK poses
//! - [`CollisionPointTree`]: CAPT spatial grid for environment queries
//! - [`CollisionEnvironment`]: High-level environment wrapper
//! - [`simd`]: SIMD dispatch with AVX2/NEON/scalar kernels

pub mod acm;
pub mod capt;
pub mod ccd;
pub mod check;
mod convex_backend;
pub mod lod;
pub mod mesh;
pub mod sdf;
pub mod self_collision;
pub mod simd;
pub mod soa;
pub mod sphere_model;
pub mod two_tier;
pub mod viz;

pub use acm::{AllowedCollisionMatrix, ResolvedACM};
pub use capt::{CollisionPointTree, GridParams, AABB};
pub use ccd::{CCDConfig, ContinuousCollisionDetector};
pub use check::{CollisionEnvironment, CollisionResult};
pub use mesh::{
    convex_decomposition, pointcloud_to_spheres, poses_to_isometries, shape_from_box,
    shape_from_cylinder, shape_from_sphere, ContactPoint, ConvexDecompConfig,
    MeshCollisionBackend,
};
pub use sdf::{MultiResolutionSDF, SDFConfig, SignedDistanceField};
pub use self_collision::SelfCollisionPairs;
pub use simd::SimdTier;
pub use soa::SpheresSoA;
pub use sphere_model::{
    adjacent_link_pairs, LinkCollisionConfig, RobotSphereModel, RobotSpheres, SphereGenConfig,
};
pub use two_tier::TwoTierCollisionChecker;
