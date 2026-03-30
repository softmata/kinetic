//! Voxel occupancy engine and perception pipeline for KINETIC.
//!
//! OctoMap-equivalent octree voxel occupancy for real-world manipulation:
//! point cloud insertion, raycasting, occupancy decay, and memory-bounded updates.
//!
//! # Architecture
//!
//! - [`Octree`]: Core octree structure with configurable depth and resolution.
//! - [`OccupancyModel`]: Log-odds occupancy with clamping and thresholds.
//! - Point cloud insertion with Bresenham raycasting for free-space clearing.

pub mod colored;
pub mod depth;
pub mod objects;
pub mod octree;
pub mod pipeline;

pub use depth::{
    deproject_f32, deproject_u16, project_point, transform_point_cloud, voxel_downsample,
    CameraIntrinsics, DepthConfig, DepthFormat, DistortionModel, OrganizedPointCloud, PointValid,
};
pub use objects::{
    convex_hull_points, deserialize_objects, euclidean_clustering, extract_cluster, is_occluded,
    marching_cubes, serialize_objects, simplify_mesh, KnownObject, KnownObjectDatabase,
    ObjectDescriptor, ObjectLifecycleManager, ObjectState, TrackedObject, TriangleMesh,
};
pub use pipeline::{
    estimate_normals, statistical_outlier_removal, CollisionFormat, CollisionObjects, Normal,
    PerceptionPipeline, PipelineConfig, SensorRegistration,
};
pub use octree::{
    Octree, OctreeConfig, OctreeNode, OccupancyGrid2D, OccupancyState, PointFilter,
};
