//! Point cloud ingestion into the collision scene.
//!
//! Converts raw 3D point arrays into collision spheres that integrate
//! with KINETIC's sphere-based collision pipeline.

use kinetic_collision::{SpheresSoA, AABB};

use crate::processing;

/// Configuration for point cloud ingestion.
#[derive(Debug, Clone)]
pub struct PointCloudConfig {
    /// Collision sphere radius per point (meters). Default: 0.01 (1 cm).
    pub sphere_radius: f64,
    /// Maximum number of points. Clouds larger than this are downsampled. Default: 100,000.
    pub max_points: usize,
    /// Apply RANSAC floor/plane removal. Default: false.
    pub remove_floor: bool,
    /// RANSAC distance threshold for inlier classification (meters). Default: 0.02.
    pub floor_distance_threshold: f64,
    /// Workspace bounds crop. Points outside this AABB are removed.
    pub crop_box: Option<AABB>,
    /// Voxel grid filter resolution (meters). `None` disables voxel filtering.
    pub voxel_downsample: Option<f64>,
    /// Statistical outlier removal parameters. `None` disables.
    pub outlier_removal: Option<OutlierConfig>,
    /// Radius-based outlier removal parameters. `None` disables.
    /// Applied after statistical outlier removal if both are set.
    pub radius_outlier_removal: Option<RadiusOutlierConfig>,
}

/// Statistical outlier removal configuration.
#[derive(Debug, Clone, Copy)]
pub struct OutlierConfig {
    /// Number of nearest neighbors to consider.
    pub k: usize,
    /// Standard deviation multiplier threshold.
    pub std_dev_multiplier: f64,
}

/// Radius-based outlier removal configuration.
#[derive(Debug, Clone, Copy)]
pub struct RadiusOutlierConfig {
    /// Radius to search for neighbors (meters).
    pub radius: f64,
    /// Minimum number of neighbors within radius to keep a point.
    pub min_neighbors: usize,
}

impl Default for PointCloudConfig {
    fn default() -> Self {
        Self {
            sphere_radius: 0.01,
            max_points: 100_000,
            remove_floor: false,
            floor_distance_threshold: 0.02,
            crop_box: None,
            voxel_downsample: None,
            outlier_removal: None,
            radius_outlier_removal: None,
        }
    }
}

/// A named point cloud source tracked in the scene.
///
/// Stores the processed collision spheres and metadata to enable
/// incremental updates (replace old data without rebuilding the entire scene).
#[derive(Debug, Clone)]
pub struct PointCloudSource {
    /// Source identifier.
    pub name: String,
    /// Number of raw points ingested (before processing).
    pub raw_count: usize,
    /// Number of points after processing.
    pub processed_count: usize,
    /// Configuration used for this source.
    pub config: PointCloudConfig,
    /// Collision spheres generated from the points.
    pub spheres: SpheresSoA,
}

/// Process raw points through the configured pipeline and produce collision spheres.
///
/// Pipeline order: crop → outlier removal → floor removal → voxel downsample → max points limit.
pub fn process_pointcloud(
    points: &[[f64; 3]],
    config: &PointCloudConfig,
) -> (Vec<[f64; 3]>, SpheresSoA) {
    let mut processed: Vec<[f64; 3]> = points.to_vec();

    // 1. Crop to workspace bounds
    if let Some(ref bounds) = config.crop_box {
        processed = processing::crop_to_aabb(&processed, bounds);
    }

    // 2. Statistical outlier removal
    if let Some(ref outlier_cfg) = config.outlier_removal {
        if processed.len() > outlier_cfg.k + 1 {
            processed = processing::statistical_outlier_removal(
                &processed,
                outlier_cfg.k,
                outlier_cfg.std_dev_multiplier,
            );
        }
    }

    // 2b. Radius-based outlier removal
    if let Some(ref radius_cfg) = config.radius_outlier_removal {
        if !processed.is_empty() {
            processed = processing::radius_outlier_removal(
                &processed,
                radius_cfg.radius,
                radius_cfg.min_neighbors,
            );
        }
    }

    // 3. RANSAC floor removal
    if config.remove_floor && processed.len() >= 3 {
        let (remaining, _model) =
            processing::ransac_remove_plane(&processed, config.floor_distance_threshold, 200);
        processed = remaining;
    }

    // 4. Voxel grid downsampling
    if let Some(voxel_size) = config.voxel_downsample {
        processed = processing::voxel_downsample(&processed, voxel_size);
    }

    // 5. Uniform downsample to max_points
    if processed.len() > config.max_points {
        processed = processing::uniform_downsample(&processed, config.max_points);
    }

    // Convert to collision spheres
    let mut spheres = SpheresSoA::with_capacity(processed.len());
    for p in &processed {
        // link_id 0 is used for environment objects
        spheres.push(p[0], p[1], p[2], config.sphere_radius, 0);
    }

    (processed, spheres)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = PointCloudConfig::default();
        assert!((config.sphere_radius - 0.01).abs() < 1e-10);
        assert_eq!(config.max_points, 100_000);
        assert!(!config.remove_floor);
        assert!(config.crop_box.is_none());
        assert!(config.voxel_downsample.is_none());
    }

    #[test]
    fn process_empty_cloud() {
        let config = PointCloudConfig::default();
        let (processed, spheres) = process_pointcloud(&[], &config);
        assert!(processed.is_empty());
        assert!(spheres.is_empty());
    }

    #[test]
    fn process_simple_cloud() {
        let points: Vec<[f64; 3]> = (0..50).map(|i| [i as f64 * 0.01, 0.0, 0.5]).collect();

        let config = PointCloudConfig {
            sphere_radius: 0.02,
            ..Default::default()
        };

        let (processed, spheres) = process_pointcloud(&points, &config);
        assert_eq!(processed.len(), 50);
        assert_eq!(spheres.len(), 50);
        assert!((spheres.radius[0] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn process_with_crop() {
        let points = [
            [0.0, 0.0, 0.5],  // inside
            [10.0, 0.0, 0.5], // outside
            [0.5, 0.5, 0.5],  // inside
        ];

        let config = PointCloudConfig {
            crop_box: Some(AABB::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)),
            ..Default::default()
        };

        let (processed, spheres) = process_pointcloud(&points, &config);
        assert_eq!(processed.len(), 2);
        assert_eq!(spheres.len(), 2);
    }

    #[test]
    fn process_with_max_points() {
        let points: Vec<[f64; 3]> = (0..1000).map(|i| [i as f64 * 0.001, 0.0, 0.5]).collect();

        let config = PointCloudConfig {
            max_points: 100,
            ..Default::default()
        };

        let (processed, spheres) = process_pointcloud(&points, &config);
        assert_eq!(processed.len(), 100);
        assert_eq!(spheres.len(), 100);
    }

    #[test]
    fn process_with_voxel_downsample() {
        let mut points = Vec::new();
        // Dense cluster of points
        for i in 0..10 {
            for j in 0..10 {
                points.push([i as f64 * 0.001, j as f64 * 0.001, 0.5]);
            }
        }

        let config = PointCloudConfig {
            voxel_downsample: Some(0.005),
            ..Default::default()
        };

        let (processed, spheres) = process_pointcloud(&points, &config);
        assert!(processed.len() < 100);
        assert_eq!(spheres.len(), processed.len());
    }

    #[test]
    fn process_with_floor_removal() {
        let mut points = Vec::new();
        // Floor points at z=0
        for i in 0..10 {
            for j in 0..10 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.0]);
            }
        }
        // Object points at z=0.5
        for i in 0..5 {
            points.push([i as f64 * 0.1, 0.0, 0.5]);
        }

        let config = PointCloudConfig {
            remove_floor: true,
            floor_distance_threshold: 0.02,
            ..Default::default()
        };

        let (processed, _spheres) = process_pointcloud(&points, &config);
        // Floor should be removed, leaving mostly the object points
        assert!(processed.len() < points.len());
    }

    #[test]
    fn spheres_have_correct_link_id() {
        let points = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]];
        let config = PointCloudConfig::default();
        let (_processed, spheres) = process_pointcloud(&points, &config);

        // All point cloud spheres should have link_id 0 (environment)
        for &id in &spheres.link_id {
            assert_eq!(id, 0);
        }
    }
}
