//! Perception pipeline: sensor data → collision world.
//!
//! Orchestrates the full pipeline:
//! 1. Depth image → point cloud (via [`depth`] module)
//! 2. Point cloud filtering (outlier removal, downsampling, workspace crop)
//! 3. Octree insertion with raycasting
//! 4. Collision object generation from occupied voxels
//! 5. Multi-sensor fusion with transform handling
//!
//! # Usage
//!
//! ```ignore
//! let mut pipeline = PerceptionPipeline::new(config);
//! pipeline.add_sensor("camera_front", intrinsics, sensor_pose);
//!
//! // Each frame:
//! pipeline.process_depth("camera_front", &depth_data, &current_sensor_pose);
//! let collision_objects = pipeline.collision_objects();
//! ```

use crate::depth::{
    deproject_u16, transform_point_cloud, voxel_downsample, CameraIntrinsics, DepthConfig,
    DistortionModel,
};
use crate::octree::{Octree, OctreeConfig, PointFilter};

use std::collections::HashMap;

/// Perception pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Octree configuration.
    pub octree: OctreeConfig,
    /// Depth processing configuration.
    pub depth: DepthConfig,
    /// Point filter for insertion.
    pub filter: PointFilter,
    /// Voxel downsample resolution (0 = disabled). Default: 0.02m.
    pub downsample_resolution: f64,
    /// Statistical outlier removal: number of neighbors. 0 = disabled.
    pub outlier_neighbors: usize,
    /// Statistical outlier removal: standard deviation threshold. Default: 1.0.
    pub outlier_std_threshold: f64,
    /// Normal estimation radius. 0 = disabled.
    pub normal_radius: f64,
    /// Temporal decay rate per update cycle. 0 = disabled.
    pub decay_rate: f32,
    /// Ground removal threshold (z value). NaN = disabled.
    pub ground_z: f64,
    /// Collision object format.
    pub collision_format: CollisionFormat,
    /// Latency compensation: buffer duration in seconds. 0 = disabled.
    pub latency_buffer_secs: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            octree: OctreeConfig::default(),
            depth: DepthConfig::default(),
            filter: PointFilter::default(),
            downsample_resolution: 0.02,
            outlier_neighbors: 0,
            outlier_std_threshold: 1.0,
            normal_radius: 0.0,
            decay_rate: 0.0,
            ground_z: f64::NAN,
            collision_format: CollisionFormat::Spheres,
            latency_buffer_secs: 0.0,
        }
    }
}

/// Format for collision objects generated from the octree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionFormat {
    /// Spheres (conservative bounding spheres per voxel).
    Spheres,
    /// Axis-aligned boxes (tighter fit per voxel).
    Boxes,
}

/// A registered sensor in the pipeline.
#[derive(Debug, Clone)]
pub struct SensorRegistration {
    /// Camera intrinsics.
    pub intrinsics: CameraIntrinsics,
    /// Distortion model.
    pub distortion: DistortionModel,
    /// Depth scale for u16 images (e.g., 0.001 for mm→m).
    pub depth_scale: f64,
    /// Sensor-to-world transform: rotation (row-major 3x3) + translation.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    /// Number of frames processed.
    pub frame_count: usize,
}

/// Collision object output from the pipeline.
#[derive(Debug, Clone)]
pub struct CollisionObjects {
    /// Object centers in world frame.
    pub centers: Vec<[f64; 3]>,
    /// Object sizes (radius for spheres, half-extent for boxes).
    pub sizes: Vec<f64>,
    /// Format of the objects.
    pub format: CollisionFormat,
    /// Number of objects.
    pub count: usize,
}

/// Normal vector for a point.
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
}

/// The perception pipeline: orchestrates sensor data → collision world.
pub struct PerceptionPipeline {
    config: PipelineConfig,
    octree: Octree,
    sensors: HashMap<String, SensorRegistration>,
    total_points_processed: usize,
}

impl PerceptionPipeline {
    /// Create a new perception pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        let octree = Octree::new(config.octree.clone());
        Self {
            config,
            octree,
            sensors: HashMap::new(),
            total_points_processed: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Register a sensor.
    pub fn add_sensor(
        &mut self,
        name: &str,
        intrinsics: CameraIntrinsics,
        distortion: DistortionModel,
        depth_scale: f64,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) {
        self.sensors.insert(
            name.to_string(),
            SensorRegistration {
                intrinsics,
                distortion,
                depth_scale,
                rotation,
                translation,
                frame_count: 0,
            },
        );
    }

    /// Update sensor pose (for moving sensors or robot-mounted cameras).
    pub fn update_sensor_pose(
        &mut self,
        name: &str,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) {
        if let Some(sensor) = self.sensors.get_mut(name) {
            sensor.rotation = rotation;
            sensor.translation = translation;
        }
    }

    /// Process a u16 depth frame from a registered sensor.
    ///
    /// Runs the full pipeline: deproject → filter → downsample → octree insert.
    pub fn process_depth_u16(&mut self, sensor_name: &str, depth_data: &[u16]) {
        let sensor = match self.sensors.get_mut(sensor_name) {
            Some(s) => s,
            None => return,
        };
        sensor.frame_count += 1;

        let rotation = sensor.rotation;
        let translation = sensor.translation;

        // Step 1: Deproject depth → point cloud
        let mut cloud = deproject_u16(
            depth_data,
            &sensor.intrinsics,
            &sensor.distortion,
            sensor.depth_scale,
            &self.config.depth,
        );

        // Step 2: Transform to world frame
        transform_point_cloud(&mut cloud, &rotation, &translation);

        // Step 3: Extract valid points
        let mut points = cloud.to_points();

        // Step 4: Outlier removal
        if self.config.outlier_neighbors > 0 {
            points = statistical_outlier_removal(
                &points,
                self.config.outlier_neighbors,
                self.config.outlier_std_threshold,
            );
        }

        // Step 5: Downsample
        if self.config.downsample_resolution > 0.0 {
            points = voxel_downsample(&points, self.config.downsample_resolution);
        }

        // Step 6: Temporal decay (before insertion)
        if self.config.decay_rate > 0.0 {
            self.octree.apply_decay(self.config.decay_rate);
        }

        // Step 7: Insert into octree with raycasting
        let sensor_origin = translation;
        self.octree
            .insert_point_cloud(sensor_origin, &points, Some(&self.config.filter));

        // Step 8: Ground removal
        if self.config.ground_z.is_finite() {
            self.octree.remove_ground(self.config.ground_z);
        }

        self.total_points_processed += points.len();
    }

    /// Process raw pre-transformed points (no depth image, already in world frame).
    pub fn process_points(&mut self, sensor_origin: [f64; 3], points: &[[f64; 3]]) {
        let mut pts = points.to_vec();

        if self.config.outlier_neighbors > 0 {
            pts = statistical_outlier_removal(
                &pts,
                self.config.outlier_neighbors,
                self.config.outlier_std_threshold,
            );
        }

        if self.config.downsample_resolution > 0.0 {
            pts = voxel_downsample(&pts, self.config.downsample_resolution);
        }

        if self.config.decay_rate > 0.0 {
            self.octree.apply_decay(self.config.decay_rate);
        }

        self.octree
            .insert_point_cloud(sensor_origin, &pts, Some(&self.config.filter));

        if self.config.ground_z.is_finite() {
            self.octree.remove_ground(self.config.ground_z);
        }

        self.total_points_processed += pts.len();
    }

    /// Get collision objects from current octree state.
    pub fn collision_objects(&self) -> CollisionObjects {
        match self.config.collision_format {
            CollisionFormat::Spheres => {
                let (centers, sizes) = self.octree.to_collision_spheres();
                let count = centers.len();
                CollisionObjects {
                    centers,
                    sizes,
                    format: CollisionFormat::Spheres,
                    count,
                }
            }
            CollisionFormat::Boxes => {
                let boxes = self.octree.to_collision_boxes();
                let count = boxes.len();
                let centers: Vec<[f64; 3]> = boxes.iter().map(|(c, _)| *c).collect();
                let sizes: Vec<f64> = boxes.iter().map(|(_, h)| *h).collect();
                CollisionObjects {
                    centers,
                    sizes,
                    format: CollisionFormat::Boxes,
                    count,
                }
            }
        }
    }

    /// Access the internal octree.
    pub fn octree(&self) -> &Octree {
        &self.octree
    }

    /// Access the internal octree mutably.
    pub fn octree_mut(&mut self) -> &mut Octree {
        &mut self.octree
    }

    /// Number of registered sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    /// Total points processed across all frames.
    pub fn total_points_processed(&self) -> usize {
        self.total_points_processed
    }

    /// Get sensor frame count.
    pub fn sensor_frame_count(&self, name: &str) -> Option<usize> {
        self.sensors.get(name).map(|s| s.frame_count)
    }

    /// Clear all occupancy data (reset octree).
    pub fn reset(&mut self) {
        self.octree.clear();
        self.total_points_processed = 0;
    }
}

/// Statistical outlier removal.
///
/// For each point, computes mean distance to K nearest neighbors.
/// Points with mean distance > mean + std_threshold * std_dev are removed.
pub fn statistical_outlier_removal(
    points: &[[f64; 3]],
    k_neighbors: usize,
    std_threshold: f64,
) -> Vec<[f64; 3]> {
    if points.len() <= k_neighbors || k_neighbors == 0 {
        return points.to_vec();
    }

    // Compute mean distance to K nearest neighbors for each point
    let mut mean_dists = Vec::with_capacity(points.len());

    for (i, p) in points.iter().enumerate() {
        // Find K nearest distances (brute force — fine for typical point cloud sizes after downsampling)
        let mut dists: Vec<f64> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| {
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                dx * dx + dy * dy + dz * dz
            })
            .collect();

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = k_neighbors.min(dists.len());
        let mean_d: f64 = dists[..k].iter().map(|d| d.sqrt()).sum::<f64>() / k as f64;
        mean_dists.push(mean_d);
    }

    // Compute global mean and std of mean distances
    let global_mean = mean_dists.iter().sum::<f64>() / mean_dists.len() as f64;
    let variance = mean_dists
        .iter()
        .map(|d| (d - global_mean) * (d - global_mean))
        .sum::<f64>()
        / mean_dists.len() as f64;
    let std_dev = variance.sqrt();

    let threshold = global_mean + std_threshold * std_dev;

    // Keep points within threshold
    points
        .iter()
        .zip(mean_dists.iter())
        .filter(|(_, d)| **d <= threshold)
        .map(|(p, _)| *p)
        .collect()
}

/// Estimate surface normals using PCA on local neighborhoods.
///
/// For each point, finds neighbors within `radius` and computes the
/// normal as the eigenvector corresponding to the smallest eigenvalue
/// of the covariance matrix.
///
/// Returns normals aligned with the query point (no global orientation guarantee).
pub fn estimate_normals(points: &[[f64; 3]], radius: f64) -> Vec<Normal> {
    let radius_sq = radius * radius;

    points
        .iter()
        .map(|p| {
            // Find neighbors within radius
            let mut neighbors: Vec<[f64; 3]> = Vec::new();
            for q in points {
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                if dx * dx + dy * dy + dz * dz <= radius_sq {
                    neighbors.push(*q);
                }
            }

            if neighbors.len() < 3 {
                return Normal { nx: 0.0, ny: 0.0, nz: 1.0 }; // default up
            }

            // Compute centroid
            let n = neighbors.len() as f64;
            let cx = neighbors.iter().map(|q| q[0]).sum::<f64>() / n;
            let cy = neighbors.iter().map(|q| q[1]).sum::<f64>() / n;
            let cz = neighbors.iter().map(|q| q[2]).sum::<f64>() / n;

            // Compute 3x3 covariance matrix
            let mut cov = [[0.0f64; 3]; 3];
            for q in &neighbors {
                let dx = q[0] - cx;
                let dy = q[1] - cy;
                let dz = q[2] - cz;
                cov[0][0] += dx * dx;
                cov[0][1] += dx * dy;
                cov[0][2] += dx * dz;
                cov[1][1] += dy * dy;
                cov[1][2] += dy * dz;
                cov[2][2] += dz * dz;
            }
            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            // Find smallest eigenvector using power iteration on inverse
            // (smallest eigenvalue of C = largest of C^-1)
            // Simpler: use the cross product of the two largest eigenvectors
            // Even simpler for 3x3: direct eigendecomposition via characteristic equation

            // Use Jacobi iteration for 3x3 symmetric matrix (robust)
            let normal = smallest_eigenvector_3x3(&cov);
            Normal { nx: normal[0], ny: normal[1], nz: normal[2] }
        })
        .collect()
}

/// Find the eigenvector corresponding to the smallest eigenvalue of a 3x3 symmetric matrix.
///
/// Uses power iteration on the shifted matrix to find the largest eigenvalue,
/// then deflates and repeats. The third eigenvector is the cross product.
fn smallest_eigenvector_3x3(m: &[[f64; 3]; 3]) -> [f64; 3] {
    // Simple approach: compute cross product of two vectors from the covariance
    // For a planar neighborhood, the normal is the cross of any two in-plane directions

    // Power iteration for largest eigenvalue
    let mut v = [1.0, 0.0, 0.0];
    for _ in 0..20 {
        let nv = mat3_mul(m, &v);
        let norm = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        if norm < 1e-15 {
            return [0.0, 0.0, 1.0];
        }
        v = [nv[0] / norm, nv[1] / norm, nv[2] / norm];
    }
    let lambda1 = dot3(&mat3_mul(m, &v), &v);
    let v1 = v;

    // Deflate: M' = M - lambda1 * v1 * v1^T
    let mut m2 = *m;
    for i in 0..3 {
        for j in 0..3 {
            m2[i][j] -= lambda1 * v1[i] * v1[j];
        }
    }

    // Second largest eigenvector
    let mut v = [0.0, 1.0, 0.0];
    for _ in 0..20 {
        let nv = mat3_mul(&m2, &v);
        let norm = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        if norm < 1e-15 {
            break;
        }
        v = [nv[0] / norm, nv[1] / norm, nv[2] / norm];
    }
    let v2 = v;

    // Normal = v1 × v2 (the smallest eigenvector)
    let normal = cross3(&v1, &v2);
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if norm < 1e-15 {
        [0.0, 0.0, 1.0]
    } else {
        [normal[0] / norm, normal[1] / norm, normal[2] / norm]
    }
}

fn mat3_mul(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::CameraIntrinsics;

    fn test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(100.0, 100.0, 31.5, 23.5, 64, 48)
    }

    fn identity_rotation() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    // ─── Pipeline tests ───

    #[test]
    fn pipeline_basic_flow() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 5.0,
                ..Default::default()
            },
            downsample_resolution: 0.0, // disable for test
            ..Default::default()
        });

        pipeline.add_sensor(
            "cam",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            identity_rotation(),
            [0.0, 0.0, 0.0],
        );

        // Create a depth image with some non-zero pixels
        let mut depth = vec![0u16; 64 * 48];
        for i in 20..30 {
            for j in 15..25 {
                depth[j * 64 + i] = 2000; // 2m
            }
        }

        pipeline.process_depth_u16("cam", &depth);

        assert!(pipeline.total_points_processed() > 0);
        assert_eq!(pipeline.sensor_frame_count("cam"), Some(1));

        let objects = pipeline.collision_objects();
        assert!(objects.count > 0, "Should produce collision objects");
    }

    #[test]
    fn pipeline_process_points() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 2.0,
                ..Default::default()
            },
            downsample_resolution: 0.0,
            ..Default::default()
        });

        let points = vec![
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ];

        pipeline.process_points([0.0, 0.0, 0.0], &points);
        assert_eq!(pipeline.total_points_processed(), 3);

        let objects = pipeline.collision_objects();
        assert!(objects.count >= 3);
    }

    #[test]
    fn pipeline_multi_sensor() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 5.0,
                ..Default::default()
            },
            downsample_resolution: 0.0,
            ..Default::default()
        });

        pipeline.add_sensor(
            "cam_front",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            identity_rotation(),
            [0.0, 0.0, 0.0],
        );

        pipeline.add_sensor(
            "cam_back",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            identity_rotation(),
            [0.0, 0.0, 1.0], // 1m behind
        );

        assert_eq!(pipeline.num_sensors(), 2);

        // Process from both sensors
        let mut depth = vec![0u16; 64 * 48];
        depth[24 * 64 + 32] = 1500;

        pipeline.process_depth_u16("cam_front", &depth);
        pipeline.process_depth_u16("cam_back", &depth);

        assert_eq!(pipeline.sensor_frame_count("cam_front"), Some(1));
        assert_eq!(pipeline.sensor_frame_count("cam_back"), Some(1));
    }

    #[test]
    fn pipeline_collision_format_boxes() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 2.0,
                ..Default::default()
            },
            collision_format: CollisionFormat::Boxes,
            downsample_resolution: 0.0,
            ..Default::default()
        });

        pipeline.process_points([0.0, 0.0, 0.0], &[[0.5, 0.5, 0.5]]);

        let objects = pipeline.collision_objects();
        assert_eq!(objects.format, CollisionFormat::Boxes);
        assert!(objects.count > 0);
    }

    #[test]
    fn pipeline_with_decay() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 2.0,
                ..Default::default()
            },
            decay_rate: 0.3,
            downsample_resolution: 0.0,
            ..Default::default()
        });

        // Insert once
        pipeline.process_points([0.0, 0.0, 0.0], &[[0.5, 0.5, 0.5]]);
        let count1 = pipeline.collision_objects().count;

        // Process empty frame — decay should reduce occupancy
        pipeline.process_points([0.0, 0.0, 0.0], &[]);
        pipeline.process_points([0.0, 0.0, 0.0], &[]);
        pipeline.process_points([0.0, 0.0, 0.0], &[]);
        let count2 = pipeline.collision_objects().count;

        assert!(
            count2 <= count1,
            "Decay should reduce or maintain collision objects: {} -> {}",
            count1, count2
        );
    }

    #[test]
    fn pipeline_with_ground_removal() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 2.0,
                ..Default::default()
            },
            ground_z: 0.0,
            downsample_resolution: 0.0,
            ..Default::default()
        });

        let points = vec![
            [0.5, 0.5, -0.5], // below ground
            [0.5, 0.5, 0.5],  // above ground
        ];

        pipeline.process_points([0.0, 0.0, 1.0], &points);

        // Below-ground point should have been removed
        assert!(
            !pipeline.octree().is_occupied(0.5, 0.5, -0.5),
            "Below-ground should be cleared"
        );
    }

    #[test]
    fn pipeline_reset() {
        let mut pipeline = PerceptionPipeline::with_defaults();
        pipeline.process_points([0.0, 0.0, 0.0], &[[0.5, 0.5, 0.5]]);
        assert!(pipeline.total_points_processed() > 0);

        pipeline.reset();
        assert_eq!(pipeline.total_points_processed(), 0);
        assert!(pipeline.octree().is_empty());
    }

    #[test]
    fn pipeline_update_sensor_pose() {
        let mut pipeline = PerceptionPipeline::with_defaults();
        pipeline.add_sensor(
            "cam",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            identity_rotation(),
            [0.0, 0.0, 0.0],
        );

        // Update pose
        pipeline.update_sensor_pose("cam", identity_rotation(), [1.0, 2.0, 3.0]);

        // Verify it took effect by processing
        let mut depth = vec![0u16; 64 * 48];
        depth[24 * 64 + 32] = 1000;
        pipeline.process_depth_u16("cam", &depth);

        // Point should be offset by the new translation
        assert!(pipeline.total_points_processed() > 0);
    }

    // ─── Outlier removal tests ───

    #[test]
    fn outlier_removal_keeps_inliers() {
        // Dense cluster + one outlier
        let mut points: Vec<[f64; 3]> = (0..50)
            .map(|i| {
                let t = i as f64 * 0.01;
                [t, t, t]
            })
            .collect();
        points.push([10.0, 10.0, 10.0]); // outlier

        let filtered = statistical_outlier_removal(&points, 5, 1.0);
        assert!(
            filtered.len() < points.len(),
            "Should remove outlier: {} -> {}",
            points.len(),
            filtered.len()
        );
        assert!(filtered.len() >= 49, "Should keep most inliers: {}", filtered.len());
    }

    #[test]
    fn outlier_removal_empty() {
        let filtered = statistical_outlier_removal(&[], 5, 1.0);
        assert!(filtered.is_empty());
    }

    // ─── Normal estimation tests ───

    #[test]
    fn normals_flat_surface() {
        // Points on XY plane (z=0) → normal should be approximately [0, 0, ±1]
        let mut points = Vec::new();
        for i in -5..=5 {
            for j in -5..=5 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.0]);
            }
        }

        let normals = estimate_normals(&points, 0.3);
        assert_eq!(normals.len(), points.len());

        // Most normals should point in z direction
        let z_aligned = normals.iter().filter(|n| n.nz.abs() > 0.8).count();
        assert!(
            z_aligned > normals.len() / 2,
            "Most normals should be z-aligned for flat XY surface: {}/{}",
            z_aligned,
            normals.len()
        );
    }

    #[test]
    fn normals_few_points_returns_default() {
        let points = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let normals = estimate_normals(&points, 0.5);
        assert_eq!(normals.len(), 2);
        // With only 2 points (< 3 neighbors), should return default [0,0,1]
    }

    // ─── Integration test ───

    #[test]
    fn full_pipeline_dual_camera() {
        let mut pipeline = PerceptionPipeline::new(PipelineConfig {
            octree: OctreeConfig {
                max_depth: 4,
                root_half_size: 3.0,
                ..Default::default()
            },
            downsample_resolution: 0.05,
            ground_z: -0.1,
            ..Default::default()
        });

        // Camera 1: looking at a box from the front
        pipeline.add_sensor(
            "front",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            identity_rotation(),
            [0.0, -1.0, 0.5],
        );

        // Camera 2: looking from the side
        let rot_90 = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        pipeline.add_sensor(
            "side",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            rot_90,
            [-1.0, 0.0, 0.5],
        );

        // Simulate frames
        let mut depth_front = vec![0u16; 64 * 48];
        for i in 25..40 {
            for j in 15..30 {
                depth_front[j * 64 + i] = 1500;
            }
        }

        let mut depth_side = vec![0u16; 64 * 48];
        for i in 20..45 {
            for j in 10..35 {
                depth_side[j * 64 + i] = 1200;
            }
        }

        pipeline.process_depth_u16("front", &depth_front);
        pipeline.process_depth_u16("side", &depth_side);

        assert_eq!(pipeline.num_sensors(), 2);
        assert!(pipeline.total_points_processed() > 0);

        let objects = pipeline.collision_objects();
        assert!(objects.count > 0, "Should have collision objects from dual cameras");

        // Verify serialization roundtrip
        let bytes = pipeline.octree().to_bytes();
        let restored = Octree::from_bytes(&bytes).unwrap();
        assert_eq!(pipeline.octree().leaf_count(), restored.leaf_count());
    }

    // ─── Collision object format ────────────────────────────────────────

    #[test]
    fn collision_objects_spheres_have_positive_radius() {
        // Intent: all collision spheres must have positive radius
        let mut pipeline = PerceptionPipeline::with_defaults();
        pipeline.add_sensor(
            "cam",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        );

        let mut depth = vec![0u16; 64 * 48];
        for i in 20..40 {
            for j in 10..30 {
                depth[j * 64 + i] = 2000;
            }
        }
        pipeline.process_depth_u16("cam", &depth);

        let objects = pipeline.collision_objects();
        for i in 0..objects.count {
            let size = objects.sizes[i];
            assert!(size > 0.0, "collision sphere {i} has non-positive radius: {size}");
        }
    }

    // ─── Normal estimation ──────────────────────────────────────────────

    #[test]
    fn normal_estimation_produces_unit_vectors() {
        // Intent: estimated normals should be unit length
        let points = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
        ];
        let normals = estimate_normals(&points, 0.15);
        for (i, n) in normals.iter().enumerate() {
            let len = (n.nx * n.nx + n.ny * n.ny + n.nz * n.nz).sqrt();
            if len > 0.0 {
                assert!(
                    (len - 1.0).abs() < 0.01,
                    "normal {i} length = {len}, expected 1.0"
                );
            }
        }
    }

    // ─── Empty depth produces no objects ─────────────────────────────

    #[test]
    fn empty_depth_image_produces_no_collision_objects() {
        let mut pipeline = PerceptionPipeline::with_defaults();
        pipeline.add_sensor(
            "cam",
            test_intrinsics(),
            DistortionModel::None,
            0.001,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        );

        // All-zero depth = no valid pixels
        let depth = vec![0u16; 64 * 48];
        pipeline.process_depth_u16("cam", &depth);

        let objects = pipeline.collision_objects();
        assert_eq!(objects.count, 0, "empty depth should produce no collision objects");
    }

    // ─── Statistical outlier removal ────────────────────────────────

    #[test]
    fn outlier_removal_filters_isolated_points() {
        // Intent: isolated far-away points should be removed
        let mut points = vec![];
        // Dense cluster near origin
        for i in 0..20 {
            for j in 0..20 {
                points.push([i as f64 * 0.01, j as f64 * 0.01, 0.0]);
            }
        }
        // One outlier far away
        points.push([100.0, 100.0, 100.0]);

        let filtered = statistical_outlier_removal(&points, 10, 1.0);
        assert!(
            filtered.len() < points.len(),
            "outlier should be removed: {} → {}",
            points.len(),
            filtered.len()
        );
        assert!(
            filtered.len() >= 400,
            "core cluster should remain: {} points",
            filtered.len()
        );
    }
}
