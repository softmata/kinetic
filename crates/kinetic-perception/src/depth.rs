//! Depth image to point cloud conversion.
//!
//! Supports pinhole and fisheye (equidistant) camera models. Converts
//! depth images (u16 with scale factor or f32 in meters) to organized
//! 3D point clouds with invalid pixel handling.
//!
//! # Camera Models
//!
//! - **Pinhole**: Standard perspective projection. `z * [u; v; 1] = K * [X; Y; Z]`
//! - **Fisheye**: Equidistant model with radial distortion coefficients.

/// Camera intrinsic parameters.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Focal length X (pixels).
    pub fx: f64,
    /// Focal length Y (pixels).
    pub fy: f64,
    /// Principal point X (pixels).
    pub cx: f64,
    /// Principal point Y (pixels).
    pub cy: f64,
    /// Image width (pixels).
    pub width: usize,
    /// Image height (pixels).
    pub height: usize,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: usize, height: usize) -> Self {
        Self { fx, fy, cx, cy, width, height }
    }
}

/// Camera distortion model.
#[derive(Debug, Clone)]
pub enum DistortionModel {
    /// No distortion (ideal pinhole).
    None,
    /// Fisheye/equidistant distortion with up to 4 coefficients [k1, k2, k3, k4].
    Fisheye { coeffs: [f64; 4] },
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self::None
    }
}

/// Depth image format.
#[derive(Debug, Clone, Copy)]
pub enum DepthFormat {
    /// 16-bit unsigned integer with a scale factor (e.g., 0.001 = millimeters).
    U16 { scale: f64 },
    /// 32-bit float in meters.
    F32,
}

/// A 3D point with validity flag.
#[derive(Debug, Clone, Copy)]
pub struct PointValid {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub valid: bool,
}

/// Organized point cloud: preserves the 2D grid structure from the depth image.
///
/// Points are stored row-major: `points[v * width + u]`.
/// Invalid pixels (zero depth, NaN, out-of-range) have `valid = false`.
#[derive(Debug, Clone)]
pub struct OrganizedPointCloud {
    pub points: Vec<PointValid>,
    pub width: usize,
    pub height: usize,
}

impl OrganizedPointCloud {
    /// Get a point at pixel coordinates.
    pub fn at(&self, u: usize, v: usize) -> &PointValid {
        &self.points[v * self.width + u]
    }

    /// Number of valid points.
    pub fn valid_count(&self) -> usize {
        self.points.iter().filter(|p| p.valid).count()
    }

    /// Total number of points (width * height).
    pub fn total(&self) -> usize {
        self.points.len()
    }

    /// Extract valid points as flat array of [x, y, z] — suitable for octree insertion.
    pub fn to_points(&self) -> Vec<[f64; 3]> {
        self.points
            .iter()
            .filter(|p| p.valid)
            .map(|p| [p.x, p.y, p.z])
            .collect()
    }
}

/// Configuration for depth processing.
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// Minimum valid depth in meters (default: 0.1).
    pub min_depth: f64,
    /// Maximum valid depth in meters (default: 10.0).
    pub max_depth: f64,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.1,
            max_depth: 10.0,
        }
    }
}

/// Convert a u16 depth image to an organized point cloud using a pinhole model.
///
/// `depth_data`: row-major u16 depth values (width * height).
/// `intrinsics`: camera intrinsic parameters.
/// `scale`: depth scale factor (e.g., 0.001 for millimeters → meters).
/// `config`: depth processing configuration.
pub fn deproject_u16(
    depth_data: &[u16],
    intrinsics: &CameraIntrinsics,
    distortion: &DistortionModel,
    scale: f64,
    config: &DepthConfig,
) -> OrganizedPointCloud {
    let w = intrinsics.width;
    let h = intrinsics.height;
    assert_eq!(depth_data.len(), w * h, "Depth data size mismatch");

    let mut points = Vec::with_capacity(w * h);

    let fx_inv = 1.0 / intrinsics.fx;
    let fy_inv = 1.0 / intrinsics.fy;

    for v in 0..h {
        for u in 0..w {
            let raw = depth_data[v * w + u];
            if raw == 0 {
                points.push(PointValid { x: 0.0, y: 0.0, z: 0.0, valid: false });
                continue;
            }

            let z = raw as f64 * scale;
            if z < config.min_depth || z > config.max_depth || !z.is_finite() {
                points.push(PointValid { x: 0.0, y: 0.0, z: 0.0, valid: false });
                continue;
            }

            let (x, y) = deproject_pixel(
                u as f64, v as f64, z,
                intrinsics, distortion, fx_inv, fy_inv,
            );

            points.push(PointValid { x, y, z, valid: true });
        }
    }

    OrganizedPointCloud { points, width: w, height: h }
}

/// Convert an f32 depth image to an organized point cloud.
///
/// `depth_data`: row-major f32 depth values in meters (width * height).
pub fn deproject_f32(
    depth_data: &[f32],
    intrinsics: &CameraIntrinsics,
    distortion: &DistortionModel,
    config: &DepthConfig,
) -> OrganizedPointCloud {
    let w = intrinsics.width;
    let h = intrinsics.height;
    assert_eq!(depth_data.len(), w * h, "Depth data size mismatch");

    let mut points = Vec::with_capacity(w * h);

    let fx_inv = 1.0 / intrinsics.fx;
    let fy_inv = 1.0 / intrinsics.fy;

    for v in 0..h {
        for u in 0..w {
            let z = depth_data[v * w + u] as f64;

            if z <= 0.0 || z < config.min_depth || z > config.max_depth
                || !z.is_finite()
            {
                points.push(PointValid { x: 0.0, y: 0.0, z: 0.0, valid: false });
                continue;
            }

            let (x, y) = deproject_pixel(
                u as f64, v as f64, z,
                intrinsics, distortion, fx_inv, fy_inv,
            );

            points.push(PointValid { x, y, z, valid: true });
        }
    }

    OrganizedPointCloud { points, width: w, height: h }
}

/// Deproject a single pixel to 3D.
fn deproject_pixel(
    u: f64, v: f64, z: f64,
    intrinsics: &CameraIntrinsics,
    distortion: &DistortionModel,
    fx_inv: f64, fy_inv: f64,
) -> (f64, f64) {
    match distortion {
        DistortionModel::None => {
            // Standard pinhole: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
            let x = (u - intrinsics.cx) * z * fx_inv;
            let y = (v - intrinsics.cy) * z * fy_inv;
            (x, y)
        }
        DistortionModel::Fisheye { coeffs } => {
            // Equidistant model: theta_d = theta + k1*theta^3 + k2*theta^5 + ...
            // Normalized image coords
            let mx = (u - intrinsics.cx) * fx_inv;
            let my = (v - intrinsics.cy) * fy_inv;
            let r_d = (mx * mx + my * my).sqrt();

            if r_d < 1e-10 {
                return (0.0, 0.0);
            }

            // Iterative undistortion (Newton's method, 10 iterations)
            let mut theta = r_d;
            for _ in 0..10 {
                let theta2 = theta * theta;
                let theta3 = theta2 * theta;
                let theta5 = theta3 * theta2;
                let theta7 = theta5 * theta2;
                let theta9 = theta7 * theta2;

                let f = theta
                    + coeffs[0] * theta3
                    + coeffs[1] * theta5
                    + coeffs[2] * theta7
                    + coeffs[3] * theta9
                    - r_d;

                let df = 1.0
                    + 3.0 * coeffs[0] * theta2
                    + 5.0 * coeffs[1] * theta2 * theta2
                    + 7.0 * coeffs[2] * theta2 * theta2 * theta2
                    + 9.0 * coeffs[3] * theta2 * theta2 * theta2 * theta2;

                if df.abs() < 1e-15 {
                    break;
                }
                theta -= f / df;
            }

            let scale = if theta.abs() < 1e-10 {
                1.0
            } else {
                theta.tan() / r_d
            };

            let x = mx * scale * z;
            let y = my * scale * z;
            (x, y)
        }
    }
}

/// Project a 3D point to pixel coordinates (inverse of deprojection).
///
/// Returns `(u, v)` pixel coordinates. Used for round-trip testing.
pub fn project_point(
    x: f64, y: f64, z: f64,
    intrinsics: &CameraIntrinsics,
) -> (f64, f64) {
    if z.abs() < 1e-10 {
        return (0.0, 0.0);
    }
    let u = intrinsics.fx * x / z + intrinsics.cx;
    let v = intrinsics.fy * y / z + intrinsics.cy;
    (u, v)
}

/// Transform a point cloud by a rigid body transform [R|t].
///
/// `rotation`: 3x3 rotation matrix (row-major).
/// `translation`: 3D translation vector.
pub fn transform_point_cloud(
    cloud: &mut OrganizedPointCloud,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
) {
    for p in &mut cloud.points {
        if !p.valid {
            continue;
        }
        let x = rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z + translation[0];
        let y = rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z + translation[1];
        let z = rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z + translation[2];
        p.x = x;
        p.y = y;
        p.z = z;
    }
}

/// Voxel-grid downsample a point cloud.
///
/// Groups points into voxels of given size and keeps the centroid of each occupied voxel.
pub fn voxel_downsample(points: &[[f64; 3]], voxel_size: f64) -> Vec<[f64; 3]> {
    use std::collections::HashMap;

    let mut voxels: HashMap<(i64, i64, i64), (f64, f64, f64, usize)> = HashMap::new();

    for p in points {
        let key = (
            (p[0] / voxel_size).floor() as i64,
            (p[1] / voxel_size).floor() as i64,
            (p[2] / voxel_size).floor() as i64,
        );
        let entry = voxels.entry(key).or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += p[0];
        entry.1 += p[1];
        entry.2 += p[2];
        entry.3 += 1;
    }

    voxels
        .values()
        .map(|(sx, sy, sz, count)| {
            let n = *count as f64;
            [sx / n, sy / n, sz / n]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intrinsics() -> CameraIntrinsics {
        // Typical 640x480 camera
        CameraIntrinsics::new(525.0, 525.0, 319.5, 239.5, 640, 480)
    }

    #[test]
    fn deproject_u16_center_pixel() {
        let intrinsics = test_intrinsics();
        let mut depth = vec![0u16; 640 * 480];
        // Set center pixel to 1000 (1m at 0.001 scale)
        depth[239 * 640 + 319] = 1000;

        let cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        assert_eq!(cloud.width, 640);
        assert_eq!(cloud.height, 480);

        // Center pixel should be approximately at (0, 0, 1.0)
        let p = cloud.at(319, 239);
        assert!(p.valid);
        assert!((p.z - 1.0).abs() < 0.01);
        assert!(p.x.abs() < 0.01, "Center X should be ~0: {}", p.x);
        assert!(p.y.abs() < 0.01, "Center Y should be ~0: {}", p.y);
    }

    #[test]
    fn deproject_u16_corner_pixel() {
        let intrinsics = test_intrinsics();
        let mut depth = vec![0u16; 640 * 480];
        depth[0] = 2000; // top-left pixel at 2m

        let cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        let p = cloud.at(0, 0);
        assert!(p.valid);
        assert!((p.z - 2.0).abs() < 0.01);
        // u=0, v=0 → X = (0 - 319.5) * 2.0 / 525.0 = -1.217
        assert!(p.x < -1.0, "Top-left X should be negative: {}", p.x);
        assert!(p.y < -0.5, "Top-left Y should be negative: {}", p.y);
    }

    #[test]
    fn deproject_u16_zero_depth_invalid() {
        let intrinsics = test_intrinsics();
        let depth = vec![0u16; 640 * 480]; // all zero

        let cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        assert_eq!(cloud.valid_count(), 0, "All zero depth should be invalid");
    }

    #[test]
    fn deproject_f32_basic() {
        let intrinsics = test_intrinsics();
        let mut depth = vec![0.0f32; 640 * 480];
        depth[240 * 640 + 320] = 1.5; // center-ish pixel at 1.5m

        let cloud = deproject_f32(
            &depth, &intrinsics, &DistortionModel::None,
            &DepthConfig::default(),
        );

        let p = cloud.at(320, 240);
        assert!(p.valid);
        assert!((p.z - 1.5).abs() < 0.01);
    }

    #[test]
    fn deproject_f32_nan_invalid() {
        let intrinsics = test_intrinsics();
        let mut depth = vec![0.0f32; 640 * 480];
        depth[0] = f32::NAN;
        depth[1] = f32::INFINITY;
        depth[2] = -1.0; // negative

        let cloud = deproject_f32(
            &depth, &intrinsics, &DistortionModel::None,
            &DepthConfig::default(),
        );

        assert!(!cloud.at(0, 0).valid, "NaN should be invalid");
        assert!(!cloud.at(1, 0).valid, "Inf should be invalid");
        assert!(!cloud.at(2, 0).valid, "Negative should be invalid");
    }

    #[test]
    fn deproject_range_filter() {
        let intrinsics = test_intrinsics();
        let mut depth = vec![0u16; 640 * 480];
        depth[240 * 640 + 320] = 50;   // 0.05m — below min_depth 0.1
        depth[240 * 640 + 321] = 500;  // 0.5m — valid
        depth[240 * 640 + 322] = 15000; // 15m — above max_depth 10

        let cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig { min_depth: 0.1, max_depth: 10.0 },
        );

        assert!(!cloud.at(320, 240).valid, "Below min should be invalid");
        assert!(cloud.at(321, 240).valid, "Within range should be valid");
        assert!(!cloud.at(322, 240).valid, "Above max should be invalid");
    }

    #[test]
    fn round_trip_project_deproject() {
        let intrinsics = test_intrinsics();

        // Start with known 3D points, project to pixels, create depth image, deproject
        let test_points = [
            (0.5, 0.3, 2.0),
            (-0.2, 0.1, 1.5),
            (0.0, 0.0, 3.0),
        ];

        for (x, y, z) in test_points {
            let (u, v) = project_point(x, y, z, &intrinsics);

            // Deproject back
            let fx_inv = 1.0 / intrinsics.fx;
            let fy_inv = 1.0 / intrinsics.fy;
            let (rx, ry) = deproject_pixel(u, v, z, &intrinsics, &DistortionModel::None, fx_inv, fy_inv);

            assert!(
                (rx - x).abs() < 1e-6,
                "X round-trip: {} -> {} (error {})",
                x, rx, (rx - x).abs()
            );
            assert!(
                (ry - y).abs() < 1e-6,
                "Y round-trip: {} -> {} (error {})",
                y, ry, (ry - y).abs()
            );
        }
    }

    #[test]
    fn to_points_extracts_valid() {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 1.5, 1.5, 4, 4);
        let mut depth = vec![0u16; 16];
        depth[0] = 1000;
        depth[5] = 2000;
        depth[15] = 3000;

        let cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        let pts = cloud.to_points();
        assert_eq!(pts.len(), 3, "Should extract 3 valid points");
    }

    #[test]
    fn fisheye_no_distortion_matches_pinhole() {
        let intrinsics = test_intrinsics();
        let zero_coeffs = DistortionModel::Fisheye { coeffs: [0.0, 0.0, 0.0, 0.0] };

        let mut depth = vec![0u16; 640 * 480];
        depth[200 * 640 + 300] = 2000;

        let cloud_pinhole = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );
        let cloud_fisheye = deproject_u16(
            &depth, &intrinsics, &zero_coeffs,
            0.001, &DepthConfig::default(),
        );

        let p_p = cloud_pinhole.at(300, 200);
        let p_f = cloud_fisheye.at(300, 200);
        assert!(p_p.valid && p_f.valid);
        assert!(
            (p_p.x - p_f.x).abs() < 0.01,
            "Zero-distortion fisheye should match pinhole: {} vs {}",
            p_p.x, p_f.x
        );
    }

    #[test]
    fn voxel_downsample_reduces_points() {
        let points: Vec<[f64; 3]> = (0..100)
            .map(|i| [i as f64 * 0.01, 0.0, 0.0])
            .collect();

        let downsampled = voxel_downsample(&points, 0.1);

        assert!(
            downsampled.len() < points.len(),
            "Downsampled ({}) should be fewer than original ({})",
            downsampled.len(),
            points.len()
        );
        assert!(downsampled.len() >= 9, "Should have ~10 voxels for 1m range / 0.1m voxels");
    }

    #[test]
    fn voxel_downsample_centroids() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
        ];

        let downsampled = voxel_downsample(&points, 0.1);
        assert_eq!(downsampled.len(), 1, "All points in same voxel");

        // Centroid should be mean
        let c = downsampled[0];
        assert!((c[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn transform_point_cloud_identity() {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 0.5, 0.5, 2, 2);
        let depth = vec![1000u16; 4];

        let mut cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        let original: Vec<_> = cloud.points.iter().map(|p| (p.x, p.y, p.z)).collect();

        // Identity transform
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];
        transform_point_cloud(&mut cloud, &rotation, &translation);

        for (i, p) in cloud.points.iter().enumerate() {
            assert!((p.x - original[i].0).abs() < 1e-10);
        }
    }

    #[test]
    fn transform_point_cloud_translation() {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 0.5, 0.5, 2, 2);
        let depth = vec![1000u16; 4];

        let mut cloud = deproject_u16(
            &depth, &intrinsics, &DistortionModel::None,
            0.001, &DepthConfig::default(),
        );

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [1.0, 2.0, 3.0];
        transform_point_cloud(&mut cloud, &rotation, &translation);

        for p in &cloud.points {
            if p.valid {
                assert!((p.z - 4.0).abs() < 0.01, "Z should be shifted by 3: {}", p.z);
            }
        }
    }
}
