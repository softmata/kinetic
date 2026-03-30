//! Depth image to 3D point cloud conversion.
//!
//! Back-projects depth images using camera intrinsics and transforms
//! the resulting points to the world frame via the camera pose.

use nalgebra::Isometry3;

/// Camera intrinsic parameters (pinhole model).
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create new camera intrinsics.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }
}

/// Configuration for depth image processing.
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// Minimum valid depth in meters. Points closer are discarded.
    pub min_depth: f64,
    /// Maximum valid depth in meters. Points farther are discarded.
    pub max_depth: f64,
    /// Skip every N pixels to reduce point count (1 = use all, 2 = every other, etc.).
    pub stride: usize,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.1,
            max_depth: 5.0,
            stride: 1,
        }
    }
}

/// Back-project a depth image to 3D points in the camera frame.
///
/// Each pixel `(u, v)` with depth `d` produces:
/// ```text
/// x = (u - cx) * d / fx
/// y = (v - cy) * d / fy
/// z = d
/// ```
///
/// Invalid depth values (NaN, Inf, zero, or outside `[min_depth, max_depth]`)
/// are skipped.
pub fn depth_to_points_camera_frame(
    depth_image: &[f32],
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    config: &DepthConfig,
) -> Vec<[f64; 3]> {
    let expected = width * height;
    if depth_image.len() < expected {
        return Vec::new();
    }

    let stride = config.stride.max(1);
    let capacity = (width / stride) * (height / stride);
    let mut points = Vec::with_capacity(capacity);

    let inv_fx = 1.0 / intrinsics.fx;
    let inv_fy = 1.0 / intrinsics.fy;

    let mut v = 0;
    while v < height {
        let row_offset = v * width;
        let mut u = 0;
        while u < width {
            let d = depth_image[row_offset + u] as f64;

            if d.is_finite() && d >= config.min_depth && d <= config.max_depth {
                let x = (u as f64 - intrinsics.cx) * d * inv_fx;
                let y = (v as f64 - intrinsics.cy) * d * inv_fy;
                points.push([x, y, d]);
            }

            u += stride;
        }
        v += stride;
    }

    points
}

/// Back-project depth image and transform to world frame.
///
/// Combines `depth_to_points_camera_frame` with a rigid transform
/// from camera frame to world frame.
pub fn depth_to_points_world(
    depth_image: &[f32],
    width: usize,
    height: usize,
    intrinsics: &CameraIntrinsics,
    camera_pose: &Isometry3<f64>,
    config: &DepthConfig,
) -> Vec<[f64; 3]> {
    let camera_points =
        depth_to_points_camera_frame(depth_image, width, height, intrinsics, config);

    camera_points
        .iter()
        .map(|p| {
            let world = camera_pose * nalgebra::Point3::new(p[0], p[1], p[2]);
            [world.x, world.y, world.z]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0)
    }

    #[test]
    fn center_pixel_projects_along_z() {
        // A pixel at the principal point should project to (0, 0, d)
        let intrinsics = test_intrinsics();
        let width = 640;
        let height = 480;

        let mut depth = vec![0.0f32; width * height];
        // Place a depth value at the center pixel
        let center_v = 240;
        let center_u = 320;
        depth[center_v * width + center_u] = 1.0; // 1 meter

        let config = DepthConfig::default();
        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);

        assert_eq!(points.len(), 1);
        assert!((points[0][0]).abs() < 1e-10); // x ≈ 0
        assert!((points[0][1]).abs() < 1e-10); // y ≈ 0
        assert!((points[0][2] - 1.0).abs() < 1e-10); // z = 1.0
    }

    #[test]
    fn off_center_pixel_projects_correctly() {
        let intrinsics = test_intrinsics();
        let width = 640;
        let height = 480;

        let mut depth = vec![0.0f32; width * height];
        // Pixel at (420, 240) = 100px right of center, at 2m depth
        let u = 420;
        let v = 240;
        depth[v * width + u] = 2.0;

        let config = DepthConfig::default();
        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);

        assert_eq!(points.len(), 1);
        // x = (420 - 320) * 2.0 / 500.0 = 0.4
        assert!((points[0][0] - 0.4).abs() < 1e-10);
        // y = (240 - 240) * 2.0 / 500.0 = 0.0
        assert!((points[0][1]).abs() < 1e-10);
        assert!((points[0][2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn invalid_depth_values_skipped() {
        let intrinsics = test_intrinsics();
        let width = 4;
        let height = 1;
        let depth = [0.0f32, f32::NAN, f32::INFINITY, -1.0];

        let config = DepthConfig::default();
        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);

        assert!(points.is_empty());
    }

    #[test]
    fn depth_range_filter() {
        let intrinsics = test_intrinsics();
        let width = 5;
        let height = 1;
        let depth = [0.05f32, 0.1, 1.0, 5.0, 10.0]; // too close, min, mid, max, too far

        let config = DepthConfig {
            min_depth: 0.1,
            max_depth: 5.0,
            stride: 1,
        };
        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);

        // 0.05 is < min_depth, 10.0 is > max_depth → 3 valid
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn stride_reduces_points() {
        let intrinsics = test_intrinsics();
        let width = 640;
        let height = 480;
        let depth = vec![1.0f32; width * height];

        let config_full = DepthConfig {
            stride: 1,
            ..Default::default()
        };
        let config_stride4 = DepthConfig {
            stride: 4,
            ..Default::default()
        };

        let full = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config_full);
        let strided =
            depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config_stride4);

        assert!(strided.len() < full.len());
        // stride=4 gives roughly width/4 * height/4 = 160*120 = 19200 points
        let expected = (width / 4) * (height / 4);
        assert_eq!(strided.len(), expected);
    }

    #[test]
    fn world_transform_applied() {
        let intrinsics = test_intrinsics();
        let width = 640;
        let height = 480;

        let mut depth = vec![0.0f32; width * height];
        depth[240 * width + 320] = 1.0; // center pixel at 1m

        // Camera is at (1, 2, 3) looking along +Z (identity rotation)
        let camera_pose = Isometry3::translation(1.0, 2.0, 3.0);
        let config = DepthConfig::default();

        let points =
            depth_to_points_world(&depth, width, height, &intrinsics, &camera_pose, &config);

        assert_eq!(points.len(), 1);
        assert!((points[0][0] - 1.0).abs() < 1e-10); // x = 0 + 1
        assert!((points[0][1] - 2.0).abs() < 1e-10); // y = 0 + 2
        assert!((points[0][2] - 4.0).abs() < 1e-10); // z = 1 + 3
    }

    #[test]
    fn empty_or_short_image() {
        let intrinsics = test_intrinsics();
        let config = DepthConfig::default();

        let empty: Vec<f32> = Vec::new();
        assert!(depth_to_points_camera_frame(&empty, 640, 480, &intrinsics, &config).is_empty());

        // Image smaller than width*height
        let short = vec![1.0f32; 10];
        assert!(depth_to_points_camera_frame(&short, 640, 480, &intrinsics, &config).is_empty());
    }

    // ─── Depth edge case tests ───

    /// All-zero depth values: all filtered out by min_depth.
    #[test]
    fn all_zero_depth() {
        let intrinsics = test_intrinsics();
        let width = 10;
        let height = 10;
        let depth = vec![0.0f32; width * height];
        let config = DepthConfig::default(); // min_depth = 0.1

        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
        assert!(
            points.is_empty(),
            "All-zero depth should produce no points, got {}",
            points.len()
        );
    }

    /// 1x1 image: single pixel produces a single point.
    #[test]
    fn single_pixel_image() {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 0.0, 0.0);
        let depth = [1.0f32]; // 1 meter at pixel (0,0)
        let config = DepthConfig::default();

        let points = depth_to_points_camera_frame(&depth, 1, 1, &intrinsics, &config);
        assert_eq!(points.len(), 1);
        // x = (0 - 0) * 1.0 / 100.0 = 0.0
        // y = (0 - 0) * 1.0 / 100.0 = 0.0
        assert!((points[0][0]).abs() < 1e-10);
        assert!((points[0][1]).abs() < 1e-10);
        assert!((points[0][2] - 1.0).abs() < 1e-10);
    }

    /// Zero focal length: produces Inf coordinates (documents behavior).
    #[test]
    fn zero_focal_length_behavior() {
        let intrinsics = CameraIntrinsics::new(0.0, 0.0, 0.0, 0.0);
        let width = 3;
        let height = 1;
        let depth = [1.0f32, 1.0, 1.0];
        let config = DepthConfig::default();

        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);

        // With fx=0, inv_fx = Inf. Pixel at u=0: x = (0-0)*1.0*Inf = NaN (0*Inf)
        // Pixel at u=1: x = (1-0)*1.0*Inf = Inf
        // This documents that zero focal length produces non-finite coordinates.
        // The function doesn't crash — it produces points with Inf/NaN coordinates.
        // Real code should validate intrinsics before calling this function.
        assert!(
            !points.is_empty(),
            "Should produce points (even if coordinates are non-finite)"
        );
    }

    /// Large stride that exceeds image dimensions: produces at most 1 point.
    #[test]
    fn stride_exceeds_dimensions() {
        let intrinsics = test_intrinsics();
        let width = 10;
        let height = 10;
        let depth = vec![1.0f32; width * height];

        let config = DepthConfig {
            min_depth: 0.1,
            max_depth: 5.0,
            stride: 100, // much larger than width/height
        };

        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
        // stride=100 means only pixel (0,0) is sampled
        assert_eq!(
            points.len(),
            1,
            "Stride > dimensions should sample only (0,0), got {} points",
            points.len()
        );
    }

    /// All-NaN depth image: all values filtered out.
    #[test]
    fn all_nan_depth() {
        let intrinsics = test_intrinsics();
        let width = 5;
        let height = 5;
        let depth = vec![f32::NAN; width * height];
        let config = DepthConfig::default();

        let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
        assert!(points.is_empty(), "All NaN should produce no points");
    }
}
