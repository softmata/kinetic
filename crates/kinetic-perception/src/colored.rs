//! Color and semantic label extensions for the octree.
//!
//! Adds per-voxel RGB color and semantic labels without changing the core
//! octree memory layout. Uses a parallel HashMap keyed by voxel position.

use std::collections::HashMap;

use crate::octree::Octree;

/// RGB color as [r, g, b] in 0..255.
pub type Color = [u8; 3];

/// Semantic label (e.g., "table", "cup", "floor").
pub type Label = String;

/// Per-voxel data: color + semantic label.
#[derive(Debug, Clone)]
pub struct VoxelData {
    pub color: Color,
    pub label: Option<Label>,
    pub confidence: f32,
}

impl VoxelData {
    pub fn new(color: Color) -> Self {
        Self { color, label: None, confidence: 1.0 }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// Spatial key for voxel data lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VoxelKey {
    x: i32,
    y: i32,
    z: i32,
}

/// Colored octree: wraps an Octree with per-voxel color and label data.
///
/// Color data is stored in a parallel HashMap, keyed by discretized position.
/// This avoids changing the core Octree memory layout while adding rich metadata.
pub struct ColoredOctree {
    /// Core occupancy octree.
    pub octree: Octree,
    /// Per-voxel color and label data.
    data: HashMap<VoxelKey, VoxelData>,
    /// Resolution for color data spatial hashing.
    color_resolution: f64,
}

impl ColoredOctree {
    /// Create from an existing octree.
    pub fn new(octree: Octree) -> Self {
        let color_resolution = octree.resolution();
        Self {
            octree,
            data: HashMap::new(),
            color_resolution,
        }
    }

    /// Create with default octree config.
    pub fn with_defaults() -> Self {
        Self::new(Octree::with_defaults())
    }

    /// Insert a colored point.
    pub fn insert_colored(&mut self, x: f64, y: f64, z: f64, color: Color) {
        self.octree.insert_point(x, y, z);
        let key = self.key(x, y, z);
        self.data.entry(key)
            .and_modify(|d| {
                // Blend colors (running average)
                for i in 0..3 {
                    d.color[i] = ((d.color[i] as u16 + color[i] as u16) / 2) as u8;
                }
            })
            .or_insert_with(|| VoxelData::new(color));
    }

    /// Insert a labeled point.
    pub fn insert_labeled(&mut self, x: f64, y: f64, z: f64, color: Color, label: &str) {
        self.octree.insert_point(x, y, z);
        let key = self.key(x, y, z);
        self.data.insert(key, VoxelData::new(color).with_label(label));
    }

    /// Insert a colored point cloud.
    pub fn insert_colored_cloud(&mut self, points: &[[f64; 3]], colors: &[Color], sensor_origin: [f64; 3]) {
        for (point, color) in points.iter().zip(colors.iter()) {
            self.octree.insert_ray(sensor_origin, *point);
            let key = self.key(point[0], point[1], point[2]);
            self.data.entry(key)
                .and_modify(|d| {
                    for i in 0..3 {
                        d.color[i] = ((d.color[i] as u16 + color[i] as u16) / 2) as u8;
                    }
                })
                .or_insert_with(|| VoxelData::new(*color));
        }
    }

    /// Query color at a position.
    pub fn color_at(&self, x: f64, y: f64, z: f64) -> Option<Color> {
        self.data.get(&self.key(x, y, z)).map(|d| d.color)
    }

    /// Query label at a position.
    pub fn label_at(&self, x: f64, y: f64, z: f64) -> Option<&str> {
        self.data.get(&self.key(x, y, z)).and_then(|d| d.label.as_deref())
    }

    /// Query full voxel data at a position.
    pub fn data_at(&self, x: f64, y: f64, z: f64) -> Option<&VoxelData> {
        self.data.get(&self.key(x, y, z))
    }

    /// Get all colored occupied voxels.
    pub fn colored_voxels(&self) -> Vec<([f64; 3], f32, Color)> {
        let occupied = self.octree.occupied_voxels();
        occupied.into_iter().map(|(pos, lo)| {
            let color = self.color_at(pos[0], pos[1], pos[2]).unwrap_or([128, 128, 128]);
            (pos, lo, color)
        }).collect()
    }

    /// Get all labeled voxels.
    pub fn labeled_voxels(&self) -> Vec<([f64; 3], &str)> {
        let mut result = Vec::new();
        for (key, data) in &self.data {
            if let Some(ref label) = data.label {
                let x = key.x as f64 * self.color_resolution;
                let y = key.y as f64 * self.color_resolution;
                let z = key.z as f64 * self.color_resolution;
                if self.octree.is_occupied(x, y, z) {
                    result.push(([x, y, z], label.as_str()));
                }
            }
        }
        result
    }

    /// Count of voxels with color data.
    pub fn colored_count(&self) -> usize {
        self.data.len()
    }

    /// Get unique labels.
    pub fn unique_labels(&self) -> Vec<&str> {
        let mut labels: Vec<&str> = self.data.values()
            .filter_map(|d| d.label.as_deref())
            .collect();
        labels.sort();
        labels.dedup();
        labels
    }

    /// Count voxels per label.
    pub fn label_counts(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for data in self.data.values() {
            if let Some(ref label) = data.label {
                *counts.entry(label.as_str()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Clear color/label data (keeps octree).
    pub fn clear_data(&mut self) {
        self.data.clear();
    }

    fn key(&self, x: f64, y: f64, z: f64) -> VoxelKey {
        VoxelKey {
            x: (x / self.color_resolution).floor() as i32,
            y: (y / self.color_resolution).floor() as i32,
            z: (z / self.color_resolution).floor() as i32,
        }
    }
}

/// Poisson surface reconstruction from oriented point cloud.
///
/// Simplified implementation: creates a smooth implicit surface from points
/// with normals by solving the Poisson equation on a voxel grid using
/// Gauss-Seidel iteration.
///
/// `points`: 3D positions.
/// `normals`: surface normals (one per point, must be unit vectors).
/// `resolution`: grid cell size.
/// `iterations`: number of Gauss-Seidel smoothing iterations.
///
/// Returns a signed distance field grid and its dimensions.
pub fn poisson_reconstruct(
    points: &[[f64; 3]],
    normals: &[[f64; 3]],
    resolution: f64,
    iterations: usize,
) -> PoissonResult {
    if points.is_empty() || normals.is_empty() {
        return PoissonResult { grid: vec![], dims: [0, 0, 0], origin: [0.0; 3], resolution };
    }

    // Compute bounding box with padding
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 { min[i] = min[i].min(p[i]); max[i] = max[i].max(p[i]); }
    }
    let pad = resolution * 3.0;
    for i in 0..3 { min[i] -= pad; max[i] += pad; }

    let nx = ((max[0] - min[0]) / resolution).ceil() as usize + 1;
    let ny = ((max[1] - min[1]) / resolution).ceil() as usize + 1;
    let nz = ((max[2] - min[2]) / resolution).ceil() as usize + 1;
    let total = nx * ny * nz;

    // Initialize divergence field from normals
    let mut div = vec![0.0f64; total];
    for (p, n) in points.iter().zip(normals.iter()) {
        let ix = ((p[0] - min[0]) / resolution) as usize;
        let iy = ((p[1] - min[1]) / resolution) as usize;
        let iz = ((p[2] - min[2]) / resolution) as usize;
        if ix >= nx || iy >= ny || iz >= nz { continue; }

        let idx = iz * ny * nx + iy * nx + ix;
        // Splat normal divergence to grid
        if ix > 0 { div[idx - 1] -= n[0] / resolution; }
        if ix + 1 < nx { div[idx + 1] += n[0] / resolution; }
        if iy > 0 { div[idx - nx] -= n[1] / resolution; }
        if iy + 1 < ny { div[idx + nx] += n[1] / resolution; }
        if iz > 0 { div[idx - ny * nx] -= n[2] / resolution; }
        if iz + 1 < nz { div[idx + ny * nx] += n[2] / resolution; }
    }

    // Solve Laplacian(phi) = div using Gauss-Seidel
    let mut phi = vec![0.0f64; total];
    for _ in 0..iterations {
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let neighbors = phi[idx - 1] + phi[idx + 1]
                        + phi[idx - nx] + phi[idx + nx]
                        + phi[idx - ny * nx] + phi[idx + ny * nx];
                    phi[idx] = (neighbors - div[idx] * resolution * resolution) / 6.0;
                }
            }
        }
    }

    PoissonResult { grid: phi, dims: [nx, ny, nz], origin: min, resolution }
}

/// Result of Poisson reconstruction.
#[derive(Debug, Clone)]
pub struct PoissonResult {
    /// Implicit function values (grid[z * ny * nx + y * nx + x]).
    pub grid: Vec<f64>,
    /// Grid dimensions [nx, ny, nz].
    pub dims: [usize; 3],
    /// Grid origin (minimum corner).
    pub origin: [f64; 3],
    /// Grid cell size.
    pub resolution: f64,
}

impl PoissonResult {
    /// Query the implicit function value at a world-frame point.
    pub fn value_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let ix = ((x - self.origin[0]) / self.resolution) as usize;
        let iy = ((y - self.origin[1]) / self.resolution) as usize;
        let iz = ((z - self.origin[2]) / self.resolution) as usize;
        if ix >= self.dims[0] || iy >= self.dims[1] || iz >= self.dims[2] {
            return 0.0;
        }
        self.grid[iz * self.dims[1] * self.dims[0] + iy * self.dims[0] + ix]
    }

    /// Total grid cells.
    pub fn num_cells(&self) -> usize { self.grid.len() }

    /// Whether the result is empty.
    pub fn is_empty(&self) -> bool { self.grid.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::OctreeConfig;

    #[test]
    fn colored_octree_insert_and_query() {
        let mut co = ColoredOctree::new(crate::octree::Octree::new(OctreeConfig {
            max_depth: 4, root_half_size: 1.0, ..Default::default()
        }));

        co.insert_colored(0.3, 0.3, 0.3, [255, 0, 0]);
        assert!(co.octree.is_occupied(0.3, 0.3, 0.3));
        let color = co.color_at(0.3, 0.3, 0.3);
        assert!(color.is_some());
        assert_eq!(color.unwrap(), [255, 0, 0]);
    }

    #[test]
    fn colored_octree_blending() {
        let mut co = ColoredOctree::new(crate::octree::Octree::new(OctreeConfig {
            max_depth: 4, root_half_size: 1.0, ..Default::default()
        }));

        co.insert_colored(0.3, 0.3, 0.3, [200, 0, 0]);
        co.insert_colored(0.3, 0.3, 0.3, [100, 0, 0]);
        let color = co.color_at(0.3, 0.3, 0.3).unwrap();
        assert!(color[0] > 100 && color[0] < 200, "Should blend: {}", color[0]);
    }

    #[test]
    fn colored_octree_labels() {
        let mut co = ColoredOctree::new(crate::octree::Octree::new(OctreeConfig {
            max_depth: 4, root_half_size: 1.0, ..Default::default()
        }));

        co.insert_labeled(0.3, 0.3, 0.3, [200, 100, 50], "table");
        co.insert_labeled(-0.3, -0.3, -0.3, [50, 200, 50], "cup");

        assert_eq!(co.label_at(0.3, 0.3, 0.3), Some("table"));
        assert_eq!(co.unique_labels().len(), 2);
        assert_eq!(*co.label_counts().get("table").unwrap(), 1);
    }

    #[test]
    fn colored_voxels_output() {
        let mut co = ColoredOctree::new(crate::octree::Octree::new(OctreeConfig {
            max_depth: 4, root_half_size: 1.0, ..Default::default()
        }));

        co.insert_colored(0.3, 0.3, 0.3, [255, 0, 0]);
        co.insert_colored(-0.3, -0.3, -0.3, [0, 255, 0]);

        let voxels = co.colored_voxels();
        assert_eq!(voxels.len(), 2);
    }

    #[test]
    fn poisson_reconstruct_basic() {
        // Flat plane at z=0: points with normals pointing up
        let mut points = Vec::new();
        let mut normals = Vec::new();
        for i in -5..=5 {
            for j in -5..=5 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.0]);
                normals.push([0.0, 0.0, 1.0]);
            }
        }

        let result = poisson_reconstruct(&points, &normals, 0.1, 20);
        assert!(!result.is_empty());
        assert!(result.num_cells() > 0);

        // Values should be different above and below the plane
        let above = result.value_at(0.0, 0.0, 0.2);
        let below = result.value_at(0.0, 0.0, -0.2);
        assert_ne!(above, below, "Values should differ above/below plane");
    }

    #[test]
    fn poisson_reconstruct_empty() {
        let result = poisson_reconstruct(&[], &[], 0.1, 10);
        assert!(result.is_empty());
    }
}
