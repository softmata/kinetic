//! Core octree data structure for voxel occupancy.
//!
//! A sparse octree that recursively subdivides 3D space into octants.
//! Leaf nodes store log-odds occupancy values. Interior nodes have 8 children.
//! Unused regions consume no memory (sparse representation).
//!
//! # Log-odds Occupancy
//!
//! Occupancy is stored as log-odds: `L = log(p / (1 - p))`.
//! Updates are additive: `L_new = L_old + L_observation`.
//! This avoids repeated Bayes rule multiplications and stays numerically stable.
//!
//! - L = 0.0 → unknown (p = 0.5)
//! - L > 0.0 → occupied (p > 0.5)
//! - L < 0.0 → free (p < 0.5)

/// Octree configuration.
#[derive(Debug, Clone)]
pub struct OctreeConfig {
    /// Maximum tree depth (default: 16). Resolution = size / 2^depth.
    pub max_depth: u8,
    /// Root node half-size in meters (default: 10.0 → 20m cube).
    pub root_half_size: f64,
    /// Root node center (default: origin).
    pub center: [f64; 3],
    /// Log-odds increment for occupied observation (default: 0.85).
    pub hit_log_odds: f32,
    /// Log-odds decrement for free observation (default: -0.4).
    pub miss_log_odds: f32,
    /// Log-odds clamping bounds (default: [-2.0, 3.5]).
    pub clamp_min: f32,
    pub clamp_max: f32,
    /// Occupancy threshold in log-odds (default: 0.0 → p=0.5).
    pub occupied_threshold: f32,
    /// Free threshold in log-odds (default: -0.4).
    pub free_threshold: f32,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 16,
            root_half_size: 10.0,
            center: [0.0, 0.0, 0.0],
            hit_log_odds: 0.85,
            miss_log_odds: -0.4,
            clamp_min: -2.0,
            clamp_max: 3.5,
            occupied_threshold: 0.0,
            free_threshold: -0.4,
        }
    }
}

impl OctreeConfig {
    /// Leaf voxel resolution at maximum depth.
    pub fn resolution(&self) -> f64 {
        self.root_half_size * 2.0 / (1u64 << self.max_depth as u64) as f64
    }
}

/// Occupancy state of a voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyState {
    /// Unknown (never observed or log-odds near zero).
    Unknown,
    /// Occupied (log-odds above threshold).
    Occupied,
    /// Free (log-odds below free threshold).
    Free,
}

/// A node in the octree.
///
/// Interior nodes have 8 children (boxed array). Leaf nodes store log-odds.
/// Unvisited space is represented by `None` children (sparse).
#[derive(Debug, Clone)]
pub enum OctreeNode {
    /// Interior node with 8 children (one per octant).
    Interior {
        children: Box<[Option<OctreeNode>; 8]>,
    },
    /// Leaf node with log-odds occupancy value.
    Leaf {
        log_odds: f32,
    },
}

impl OctreeNode {
    fn new_leaf(log_odds: f32) -> Self {
        Self::Leaf { log_odds }
    }

    fn new_interior() -> Self {
        Self::Interior {
            children: Box::new([None, None, None, None, None, None, None, None]),
        }
    }

    /// Get log-odds value. For interior nodes, returns average of children.
    pub fn log_odds(&self) -> f32 {
        match self {
            OctreeNode::Leaf { log_odds } => *log_odds,
            OctreeNode::Interior { children } => {
                let mut sum = 0.0f32;
                let mut count = 0;
                for child in children.iter() {
                    if let Some(c) = child {
                        sum += c.log_odds();
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f32 } else { 0.0 }
            }
        }
    }

    /// Count total nodes (including this one).
    pub fn node_count(&self) -> usize {
        match self {
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Interior { children } => {
                1 + children.iter().filter_map(|c| c.as_ref()).map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Count leaf nodes.
    pub fn leaf_count(&self) -> usize {
        match self {
            OctreeNode::Leaf { .. } => 1,
            OctreeNode::Interior { children } => {
                children.iter().filter_map(|c| c.as_ref()).map(|c| c.leaf_count()).sum()
            }
        }
    }

    /// Maximum depth of this subtree.
    pub fn depth(&self) -> usize {
        match self {
            OctreeNode::Leaf { .. } => 0,
            OctreeNode::Interior { children } => {
                1 + children.iter().filter_map(|c| c.as_ref()).map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }
}

/// Sparse octree for voxel occupancy mapping.
///
/// Recursively subdivides a cubic volume into 8 octants. Only regions that
/// have been observed consume memory. Supports:
///
/// - Insert point (mark occupied)
/// - Insert ray (mark free along ray, occupied at endpoint)
/// - Query occupancy at any point
/// - Configurable resolution via max depth
pub struct Octree {
    root: Option<OctreeNode>,
    config: OctreeConfig,
    /// Number of leaf nodes with non-zero log-odds.
    num_updated: usize,
}

impl Octree {
    /// Create an empty octree with the given configuration.
    pub fn new(config: OctreeConfig) -> Self {
        Self {
            root: None,
            config,
            num_updated: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(OctreeConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &OctreeConfig {
        &self.config
    }

    /// Leaf voxel resolution in meters.
    pub fn resolution(&self) -> f64 {
        self.config.resolution()
    }

    /// Whether the octree is empty (no observations).
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Total node count (interior + leaf).
    pub fn node_count(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.node_count())
    }

    /// Leaf node count.
    pub fn leaf_count(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.leaf_count())
    }

    /// Number of updated voxels.
    pub fn num_updated(&self) -> usize {
        self.num_updated
    }

    /// Insert a point as occupied.
    ///
    /// Subdivides the octree to maximum depth at the point location and
    /// updates the leaf's log-odds by `hit_log_odds`.
    pub fn insert_point(&mut self, x: f64, y: f64, z: f64) {
        if !self.contains(x, y, z) {
            return;
        }

        if self.root.is_none() {
            self.root = Some(OctreeNode::new_interior());
        }

        let cx = self.config.center[0];
        let cy = self.config.center[1];
        let cz = self.config.center[2];
        let half = self.config.root_half_size;

        self.num_updated += 1;
        let hit = self.config.hit_log_odds;
        let clamp_min = self.config.clamp_min;
        let clamp_max = self.config.clamp_max;
        let max_depth = self.config.max_depth;

        insert_recursive(
            self.root.as_mut().unwrap(),
            x, y, z,
            cx, cy, cz, half,
            0, max_depth,
            hit, clamp_min, clamp_max,
        );
    }

    /// Mark a voxel as free (e.g., along a ray path).
    pub fn insert_free(&mut self, x: f64, y: f64, z: f64) {
        if !self.contains(x, y, z) {
            return;
        }

        if self.root.is_none() {
            self.root = Some(OctreeNode::new_interior());
        }

        let cx = self.config.center[0];
        let cy = self.config.center[1];
        let cz = self.config.center[2];
        let half = self.config.root_half_size;
        let miss = self.config.miss_log_odds;
        let clamp_min = self.config.clamp_min;
        let clamp_max = self.config.clamp_max;
        let max_depth = self.config.max_depth;

        insert_recursive(
            self.root.as_mut().unwrap(),
            x, y, z,
            cx, cy, cz, half,
            0, max_depth,
            miss, clamp_min, clamp_max,
        );
    }

    /// Insert a ray from sensor origin to endpoint.
    ///
    /// Marks voxels along the ray as free (miss), and the endpoint as occupied (hit).
    /// Uses 3D Bresenham-like stepping at leaf resolution.
    pub fn insert_ray(&mut self, origin: [f64; 3], endpoint: [f64; 3]) {
        let res = self.resolution();
        let dx = endpoint[0] - origin[0];
        let dy = endpoint[1] - origin[1];
        let dz = endpoint[2] - origin[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();

        if len < 1e-10 {
            self.insert_point(endpoint[0], endpoint[1], endpoint[2]);
            return;
        }

        let step = res * 0.5; // half-resolution stepping for accuracy
        let steps = (len / step).ceil() as usize;
        let inv_steps = 1.0 / steps as f64;

        // Mark free along ray (skip last step — that's the endpoint)
        for i in 0..steps.saturating_sub(1) {
            let t = i as f64 * inv_steps;
            let rx = origin[0] + t * dx;
            let ry = origin[1] + t * dy;
            let rz = origin[2] + t * dz;
            self.insert_free(rx, ry, rz);
        }

        // Mark endpoint as occupied
        self.insert_point(endpoint[0], endpoint[1], endpoint[2]);
    }

    /// Query occupancy state at a point.
    pub fn query(&self, x: f64, y: f64, z: f64) -> OccupancyState {
        let log_odds = self.query_log_odds(x, y, z);
        if log_odds > self.config.occupied_threshold {
            OccupancyState::Occupied
        } else if log_odds < self.config.free_threshold {
            OccupancyState::Free
        } else {
            OccupancyState::Unknown
        }
    }

    /// Query raw log-odds at a point. Returns 0.0 (unknown) for unvisited regions.
    pub fn query_log_odds(&self, x: f64, y: f64, z: f64) -> f32 {
        let root = match &self.root {
            Some(r) => r,
            None => return 0.0,
        };

        if !self.contains(x, y, z) {
            return 0.0;
        }

        let cx = self.config.center[0];
        let cy = self.config.center[1];
        let cz = self.config.center[2];
        let half = self.config.root_half_size;

        query_recursive(root, x, y, z, cx, cy, cz, half)
    }

    /// Check if a point is occupied.
    pub fn is_occupied(&self, x: f64, y: f64, z: f64) -> bool {
        self.query(x, y, z) == OccupancyState::Occupied
    }

    /// Check if a point is free.
    pub fn is_free(&self, x: f64, y: f64, z: f64) -> bool {
        self.query(x, y, z) == OccupancyState::Free
    }

    /// Check if a point is within the octree bounds.
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        let h = self.config.root_half_size;
        let c = &self.config.center;
        (x - c[0]).abs() <= h && (y - c[1]).abs() <= h && (z - c[2]).abs() <= h
    }

    /// Get all occupied leaf centers and their log-odds.
    pub fn occupied_voxels(&self) -> Vec<([f64; 3], f32)> {
        let mut voxels = Vec::new();
        if let Some(root) = &self.root {
            let cx = self.config.center[0];
            let cy = self.config.center[1];
            let cz = self.config.center[2];
            let half = self.config.root_half_size;
            collect_occupied(root, cx, cy, cz, half, self.config.occupied_threshold, &mut voxels);
        }
        voxels
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.root = None;
        self.num_updated = 0;
    }

    // ─── Pruning ─────────────────────────────────────────────────────────

    /// Prune the octree: merge children that all share the same occupancy state.
    ///
    /// If all 8 children of an interior node are leaves with similar log-odds
    /// (within `tolerance`), they are merged into a single leaf. This reduces
    /// memory usage without losing significant information.
    ///
    /// Returns the number of nodes removed.
    pub fn prune(&mut self, tolerance: f32) -> usize {
        match self.root.take() {
            Some(mut root) => {
                let removed = prune_recursive(&mut root, tolerance);
                self.root = Some(root);
                removed
            }
            None => 0,
        }
    }

    /// Memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Each Interior: 8 * Option<OctreeNode> + enum discriminant
        // Each Leaf: f32 + enum discriminant
        let nodes = self.node_count();
        let leaves = self.leaf_count();
        let interiors = nodes - leaves;
        // Rough estimate: Interior ~136 bytes (Box<[Option<Node>;8]>), Leaf ~8 bytes
        interiors * 136 + leaves * 8
    }

    // ─── Spatial Queries ─────────────────────────────────────────────────

    /// Find all occupied voxels within a sphere of given radius.
    pub fn query_radius(&self, x: f64, y: f64, z: f64, radius: f64) -> Vec<([f64; 3], f32)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            let cx = self.config.center[0];
            let cy = self.config.center[1];
            let cz = self.config.center[2];
            let half = self.config.root_half_size;
            query_radius_recursive(
                root, cx, cy, cz, half,
                x, y, z, radius * radius,
                self.config.occupied_threshold,
                &mut results,
            );
        }
        results
    }

    /// Find the nearest occupied voxel to a query point.
    ///
    /// Returns `Some((position, log_odds, distance))` or `None` if no occupied voxels.
    pub fn nearest_occupied(&self, x: f64, y: f64, z: f64) -> Option<([f64; 3], f32, f64)> {
        let occupied = self.occupied_voxels();
        occupied
            .into_iter()
            .map(|(pos, lo)| {
                let dx = pos[0] - x;
                let dy = pos[1] - y;
                let dz = pos[2] - z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                (pos, lo, dist)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    }

    /// Query all occupied voxels within an axis-aligned bounding box.
    pub fn query_aabb(
        &self,
        min: [f64; 3],
        max: [f64; 3],
    ) -> Vec<([f64; 3], f32)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            let cx = self.config.center[0];
            let cy = self.config.center[1];
            let cz = self.config.center[2];
            let half = self.config.root_half_size;
            query_aabb_recursive(
                root, cx, cy, cz, half,
                &min, &max,
                self.config.occupied_threshold,
                &mut results,
            );
        }
        results
    }

    // ─── Iterator / Visitor ──────────────────────────────────────────────

    /// Visit all leaf nodes with their centers and half-sizes.
    ///
    /// The callback receives `(center, half_size, log_odds)` for each leaf.
    pub fn visit_leaves<F>(&self, mut callback: F)
    where
        F: FnMut([f64; 3], f64, f32),
    {
        if let Some(root) = &self.root {
            let cx = self.config.center[0];
            let cy = self.config.center[1];
            let cz = self.config.center[2];
            let half = self.config.root_half_size;
            visit_leaves_recursive(root, cx, cy, cz, half, &mut callback);
        }
    }

    /// Collect all leaf nodes as `(center, half_size, log_odds)`.
    pub fn all_leaves(&self) -> Vec<([f64; 3], f64, f32)> {
        let mut leaves = Vec::new();
        self.visit_leaves(|center, half_size, log_odds| {
            leaves.push((center, half_size, log_odds));
        });
        leaves
    }

    // ─── Collision Bridge ────────────────────────────────────────────────

    /// Convert occupied voxels to collision spheres.
    ///
    /// Each occupied voxel becomes a sphere centered at the voxel center with
    /// radius = half_diagonal of the voxel (conservative bounding sphere).
    /// Returns positions and radii suitable for `kinetic_collision::SpheresSoA`.
    pub fn to_collision_spheres(&self) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut positions = Vec::new();
        let mut radii = Vec::new();

        self.visit_leaves(|center, half_size, log_odds| {
            if log_odds > self.config.occupied_threshold {
                positions.push(center);
                // Conservative: sphere radius = voxel half-diagonal
                let radius = half_size * 1.732; // sqrt(3)
                radii.push(radius);
            }
        });

        (positions, radii)
    }

    /// Convert occupied voxels to axis-aligned boxes for collision.
    ///
    /// Returns `(center, half_extents)` for each occupied voxel.
    pub fn to_collision_boxes(&self) -> Vec<([f64; 3], f64)> {
        let mut boxes = Vec::new();
        self.visit_leaves(|center, half_size, log_odds| {
            if log_odds > self.config.occupied_threshold {
                boxes.push((center, half_size));
            }
        });
        boxes
    }

    // ─── Batch Point Cloud Insertion ─────────────────────────────────────

    /// Insert a point cloud with raycasting from a sensor origin.
    ///
    /// For each point in the cloud:
    /// 1. Marks voxels along the ray from `sensor_origin` to the point as free.
    /// 2. Marks the point itself as occupied.
    ///
    /// `filter`: optional filter to skip points (range, height, ROI).
    pub fn insert_point_cloud(
        &mut self,
        sensor_origin: [f64; 3],
        points: &[[f64; 3]],
        filter: Option<&PointFilter>,
    ) {
        for point in points {
            if let Some(f) = filter {
                if !f.accept(sensor_origin, *point) {
                    continue;
                }
            }
            self.insert_ray(sensor_origin, *point);
        }
    }

    /// Insert points as occupied only (no raycasting / free-space clearing).
    ///
    /// Faster than `insert_point_cloud` when free-space information isn't needed.
    pub fn insert_points_occupied(&mut self, points: &[[f64; 3]]) {
        for point in points {
            self.insert_point(point[0], point[1], point[2]);
        }
    }

    // ─── Temporal Decay ──────────────────────────────────────────────────

    /// Apply temporal decay to all occupied voxels.
    ///
    /// Reduces log-odds toward zero by `decay_rate` per call. Used to fade
    /// out old observations in dynamic environments.
    ///
    /// `decay_rate`: amount to subtract from positive log-odds (and add to negative).
    /// Voxels that cross zero are set to zero (unknown).
    pub fn apply_decay(&mut self, decay_rate: f32) {
        if let Some(root) = &mut self.root {
            apply_decay_recursive(root, decay_rate);
        }
    }

    // ─── 2D Projection ──────────────────────────────────────────────────

    /// Project occupied voxels onto a 2D grid (top-down view).
    ///
    /// Returns a grid of boolean values (true = occupied column) at the
    /// octree's leaf resolution. The grid covers the XY extent of the octree.
    ///
    /// `z_min`/`z_max`: only consider voxels within this height range.
    pub fn project_2d(&self, z_min: f64, z_max: f64) -> OccupancyGrid2D {
        let res = self.resolution();
        let half = self.config.root_half_size;
        let cx = self.config.center[0];
        let cy = self.config.center[1];

        let nx = ((2.0 * half) / res).ceil() as usize;
        let ny = ((2.0 * half) / res).ceil() as usize;

        let mut grid = vec![false; nx * ny];

        self.visit_leaves(|center, _half_size, log_odds| {
            if log_odds > self.config.occupied_threshold
                && center[2] >= z_min
                && center[2] <= z_max
            {
                let ix = ((center[0] - (cx - half)) / res) as usize;
                let iy = ((center[1] - (cy - half)) / res) as usize;
                if ix < nx && iy < ny {
                    grid[iy * nx + ix] = true;
                }
            }
        });

        OccupancyGrid2D {
            grid,
            width: nx,
            height: ny,
            resolution: res,
            origin: [cx - half, cy - half],
        }
    }

    // ─── Serialization ───────────────────────────────────────────────────

    /// Serialize the octree to a compact binary format.
    ///
    /// Format: [config (fixed)] + [node stream (recursive pre-order)].
    /// Each node: 1 byte type (0=leaf, 1=interior, 2=none) + payload.
    /// Leaf payload: 4 bytes f32 log-odds.
    /// Interior payload: 8 child nodes (recursive).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Config header
        buf.extend_from_slice(&self.config.max_depth.to_le_bytes());
        buf.extend_from_slice(&self.config.root_half_size.to_le_bytes());
        buf.extend_from_slice(&self.config.center[0].to_le_bytes());
        buf.extend_from_slice(&self.config.center[1].to_le_bytes());
        buf.extend_from_slice(&self.config.center[2].to_le_bytes());
        buf.extend_from_slice(&self.config.hit_log_odds.to_le_bytes());
        buf.extend_from_slice(&self.config.miss_log_odds.to_le_bytes());
        buf.extend_from_slice(&self.config.clamp_min.to_le_bytes());
        buf.extend_from_slice(&self.config.clamp_max.to_le_bytes());
        buf.extend_from_slice(&self.config.occupied_threshold.to_le_bytes());
        buf.extend_from_slice(&self.config.free_threshold.to_le_bytes());

        // Node tree
        match &self.root {
            None => buf.push(2), // none
            Some(root) => serialize_node(root, &mut buf),
        }

        buf
    }

    /// Deserialize an octree from bytes produced by `to_bytes()`.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 53 {
            return None; // min header size
        }

        let mut pos = 0;

        let max_depth = data[pos];
        pos += 1;
        let root_half_size = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let cx = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let cy = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let cz = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let hit_log_odds = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let miss_log_odds = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let clamp_min = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let clamp_max = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let occupied_threshold = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let free_threshold = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;

        let config = OctreeConfig {
            max_depth,
            root_half_size,
            center: [cx, cy, cz],
            hit_log_odds,
            miss_log_odds,
            clamp_min,
            clamp_max,
            occupied_threshold,
            free_threshold,
        };

        let (root, _) = deserialize_node(data, pos)?;
        let num_updated = root.as_ref().map_or(0, |r| r.leaf_count());

        Some(Self {
            root,
            config,
            num_updated,
        })
    }

    // ─── Ground Plane Removal ────────────────────────────────────────────

    /// Remove ground plane: set all voxels below `z_threshold` to free.
    ///
    /// Simple height-based ground removal. For more sophisticated methods
    /// (RANSAC), filter the point cloud before insertion.
    pub fn remove_ground(&mut self, z_threshold: f64) {
        if let Some(root) = &mut self.root {
            let cx = self.config.center[0];
            let cy = self.config.center[1];
            let cz = self.config.center[2];
            let half = self.config.root_half_size;
            remove_ground_recursive(root, cx, cy, cz, half, z_threshold, self.config.clamp_min);
        }
    }
}

/// Point filter for point cloud insertion.
#[derive(Debug, Clone)]
pub struct PointFilter {
    /// Maximum range from sensor origin (default: f64::INFINITY).
    pub max_range: f64,
    /// Minimum range (skip points too close to sensor).
    pub min_range: f64,
    /// Height bounds: only insert points within [z_min, z_max].
    pub z_min: f64,
    pub z_max: f64,
    /// Optional ROI (region of interest) AABB: [x_min, y_min, z_min, x_max, y_max, z_max].
    pub roi: Option<[f64; 6]>,
}

impl Default for PointFilter {
    fn default() -> Self {
        Self {
            max_range: f64::INFINITY,
            min_range: 0.0,
            z_min: f64::NEG_INFINITY,
            z_max: f64::INFINITY,
            roi: None,
        }
    }
}

impl PointFilter {
    /// Check if a point passes the filter.
    pub fn accept(&self, sensor_origin: [f64; 3], point: [f64; 3]) -> bool {
        // Range check
        let dx = point[0] - sensor_origin[0];
        let dy = point[1] - sensor_origin[1];
        let dz = point[2] - sensor_origin[2];
        let range_sq = dx * dx + dy * dy + dz * dz;

        if range_sq > self.max_range * self.max_range {
            return false;
        }
        if range_sq < self.min_range * self.min_range {
            return false;
        }

        // Height check
        if point[2] < self.z_min || point[2] > self.z_max {
            return false;
        }

        // ROI check
        if let Some(roi) = &self.roi {
            if point[0] < roi[0] || point[0] > roi[3] { return false; }
            if point[1] < roi[1] || point[1] > roi[4] { return false; }
            if point[2] < roi[2] || point[2] > roi[5] { return false; }
        }

        true
    }
}

/// 2D occupancy grid (top-down projection).
#[derive(Debug, Clone)]
pub struct OccupancyGrid2D {
    /// Row-major grid: grid[y * width + x].
    pub grid: Vec<bool>,
    /// Grid width (X cells).
    pub width: usize,
    /// Grid height (Y cells).
    pub height: usize,
    /// Cell resolution in meters.
    pub resolution: f64,
    /// World-frame origin of the grid [x_min, y_min].
    pub origin: [f64; 2],
}

impl OccupancyGrid2D {
    /// Number of occupied cells.
    pub fn occupied_count(&self) -> usize {
        self.grid.iter().filter(|&&v| v).count()
    }

    /// Check occupancy at grid coordinates.
    pub fn is_occupied(&self, ix: usize, iy: usize) -> bool {
        if ix < self.width && iy < self.height {
            self.grid[iy * self.width + ix]
        } else {
            false
        }
    }
}

/// Determine which octant a point falls in relative to center.
/// Returns index 0-7.
fn octant_index(x: f64, y: f64, z: f64, cx: f64, cy: f64, cz: f64) -> usize {
    let mut idx = 0;
    if x >= cx { idx |= 1; }
    if y >= cy { idx |= 2; }
    if z >= cz { idx |= 4; }
    idx
}

/// Compute child center for a given octant.
fn child_center(
    cx: f64, cy: f64, cz: f64, half: f64, octant: usize,
) -> (f64, f64, f64) {
    let quarter = half * 0.5;
    let nx = if octant & 1 != 0 { cx + quarter } else { cx - quarter };
    let ny = if octant & 2 != 0 { cy + quarter } else { cy - quarter };
    let nz = if octant & 4 != 0 { cz + quarter } else { cz - quarter };
    (nx, ny, nz)
}

/// Recursive insertion into the octree.
fn insert_recursive(
    node: &mut OctreeNode,
    x: f64, y: f64, z: f64,
    cx: f64, cy: f64, cz: f64, half: f64,
    depth: u8, max_depth: u8,
    log_odds_update: f32,
    clamp_min: f32, clamp_max: f32,
) {
    match node {
        OctreeNode::Leaf { log_odds } => {
            if depth >= max_depth {
                // At max depth: update log-odds
                *log_odds = (*log_odds + log_odds_update).clamp(clamp_min, clamp_max);
            } else {
                // Need to subdivide: convert leaf to interior
                let old_log_odds = *log_odds;
                *node = OctreeNode::new_interior();

                // Re-insert into the new interior node
                insert_recursive(
                    node, x, y, z, cx, cy, cz, half,
                    depth, max_depth, log_odds_update, clamp_min, clamp_max,
                );

                // Propagate old log-odds to the child that was just created
                // (The other 7 octants inherit "unknown" = 0.0, which is correct)
                let _ = old_log_odds; // old value is lost on subdivision (acceptable)
            }
        }
        OctreeNode::Interior { children } => {
            if depth >= max_depth {
                return;
            }

            let octant = octant_index(x, y, z, cx, cy, cz);
            let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, octant);
            let new_half = half * 0.5;

            if children[octant].is_none() {
                if depth + 1 >= max_depth {
                    children[octant] = Some(OctreeNode::new_leaf(0.0));
                } else {
                    children[octant] = Some(OctreeNode::new_interior());
                }
            }

            insert_recursive(
                children[octant].as_mut().unwrap(),
                x, y, z,
                ncx, ncy, ncz, new_half,
                depth + 1, max_depth,
                log_odds_update, clamp_min, clamp_max,
            );
        }
    }
}

/// Recursive query of log-odds at a point.
fn query_recursive(
    node: &OctreeNode,
    x: f64, y: f64, z: f64,
    cx: f64, cy: f64, cz: f64, half: f64,
) -> f32 {
    match node {
        OctreeNode::Leaf { log_odds } => *log_odds,
        OctreeNode::Interior { children } => {
            let octant = octant_index(x, y, z, cx, cy, cz);
            let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, octant);
            let new_half = half * 0.5;

            match &children[octant] {
                Some(child) => query_recursive(child, x, y, z, ncx, ncy, ncz, new_half),
                None => 0.0, // unknown
            }
        }
    }
}

/// Collect all occupied leaf voxel centers.
fn collect_occupied(
    node: &OctreeNode,
    cx: f64, cy: f64, cz: f64, half: f64,
    threshold: f32,
    out: &mut Vec<([f64; 3], f32)>,
) {
    match node {
        OctreeNode::Leaf { log_odds } => {
            if *log_odds > threshold {
                out.push(([cx, cy, cz], *log_odds));
            }
        }
        OctreeNode::Interior { children } => {
            for (i, child) in children.iter().enumerate() {
                if let Some(c) = child {
                    let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, i);
                    collect_occupied(c, ncx, ncy, ncz, half * 0.5, threshold, out);
                }
            }
        }
    }
}

/// Recursive pruning: merge children into leaf when all similar.
fn prune_recursive(node: &mut OctreeNode, tolerance: f32) -> usize {
    match node {
        OctreeNode::Leaf { .. } => 0,
        OctreeNode::Interior { children } => {
            let mut removed = 0;

            // First, recursively prune children
            for child in children.iter_mut() {
                if let Some(c) = child {
                    removed += prune_recursive(c, tolerance);
                }
            }

            // Check if all children are leaves with similar values
            let mut all_leaves = true;
            let mut sum = 0.0f32;
            let mut count = 0u8;
            let mut min_lo = f32::INFINITY;
            let mut max_lo = f32::NEG_INFINITY;

            for child in children.iter() {
                match child {
                    Some(OctreeNode::Leaf { log_odds }) => {
                        sum += log_odds;
                        count += 1;
                        min_lo = min_lo.min(*log_odds);
                        max_lo = max_lo.max(*log_odds);
                    }
                    Some(OctreeNode::Interior { .. }) => {
                        all_leaves = false;
                        break;
                    }
                    None => {
                        // Treat missing children as unknown (0.0)
                        count += 1;
                        min_lo = min_lo.min(0.0);
                        max_lo = max_lo.max(0.0);
                    }
                }
            }

            if all_leaves && count > 0 && (max_lo - min_lo) <= tolerance {
                let avg = sum / count as f32;
                removed += count as usize; // removing N children, replacing with 1 leaf
                *node = OctreeNode::new_leaf(avg);
            }

            removed
        }
    }
}

/// Recursive radius query.
fn query_radius_recursive(
    node: &OctreeNode,
    cx: f64, cy: f64, cz: f64, half: f64,
    qx: f64, qy: f64, qz: f64, radius_sq: f64,
    threshold: f32,
    out: &mut Vec<([f64; 3], f32)>,
) {
    // Check if this node's bounding box intersects the query sphere
    let closest_x = qx.clamp(cx - half, cx + half);
    let closest_y = qy.clamp(cy - half, cy + half);
    let closest_z = qz.clamp(cz - half, cz + half);
    let dx = closest_x - qx;
    let dy = closest_y - qy;
    let dz = closest_z - qz;
    if dx * dx + dy * dy + dz * dz > radius_sq {
        return; // bounding box doesn't intersect sphere
    }

    match node {
        OctreeNode::Leaf { log_odds } => {
            if *log_odds > threshold {
                let dx = cx - qx;
                let dy = cy - qy;
                let dz = cz - qz;
                if dx * dx + dy * dy + dz * dz <= radius_sq {
                    out.push(([cx, cy, cz], *log_odds));
                }
            }
        }
        OctreeNode::Interior { children } => {
            for (i, child) in children.iter().enumerate() {
                if let Some(c) = child {
                    let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, i);
                    query_radius_recursive(
                        c, ncx, ncy, ncz, half * 0.5,
                        qx, qy, qz, radius_sq, threshold, out,
                    );
                }
            }
        }
    }
}

/// Recursive AABB query.
fn query_aabb_recursive(
    node: &OctreeNode,
    cx: f64, cy: f64, cz: f64, half: f64,
    min: &[f64; 3], max: &[f64; 3],
    threshold: f32,
    out: &mut Vec<([f64; 3], f32)>,
) {
    // Check AABB intersection
    if cx + half < min[0] || cx - half > max[0] { return; }
    if cy + half < min[1] || cy - half > max[1] { return; }
    if cz + half < min[2] || cz - half > max[2] { return; }

    match node {
        OctreeNode::Leaf { log_odds } => {
            if *log_odds > threshold {
                out.push(([cx, cy, cz], *log_odds));
            }
        }
        OctreeNode::Interior { children } => {
            for (i, child) in children.iter().enumerate() {
                if let Some(c) = child {
                    let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, i);
                    query_aabb_recursive(
                        c, ncx, ncy, ncz, half * 0.5, min, max, threshold, out,
                    );
                }
            }
        }
    }
}

/// Recursive leaf visitor.
fn visit_leaves_recursive<F>(
    node: &OctreeNode,
    cx: f64, cy: f64, cz: f64, half: f64,
    callback: &mut F,
)
where
    F: FnMut([f64; 3], f64, f32),
{
    match node {
        OctreeNode::Leaf { log_odds } => {
            callback([cx, cy, cz], half, *log_odds);
        }
        OctreeNode::Interior { children } => {
            for (i, child) in children.iter().enumerate() {
                if let Some(c) = child {
                    let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, i);
                    visit_leaves_recursive(c, ncx, ncy, ncz, half * 0.5, callback);
                }
            }
        }
    }
}

/// Apply temporal decay recursively.
fn apply_decay_recursive(node: &mut OctreeNode, decay_rate: f32) {
    match node {
        OctreeNode::Leaf { log_odds } => {
            if *log_odds > 0.0 {
                *log_odds = (*log_odds - decay_rate).max(0.0);
            } else if *log_odds < 0.0 {
                *log_odds = (*log_odds + decay_rate).min(0.0);
            }
        }
        OctreeNode::Interior { children } => {
            for child in children.iter_mut() {
                if let Some(c) = child {
                    apply_decay_recursive(c, decay_rate);
                }
            }
        }
    }
}

/// Serialize a node to bytes.
fn serialize_node(node: &OctreeNode, buf: &mut Vec<u8>) {
    match node {
        OctreeNode::Leaf { log_odds } => {
            buf.push(0); // leaf tag
            buf.extend_from_slice(&log_odds.to_le_bytes());
        }
        OctreeNode::Interior { children } => {
            buf.push(1); // interior tag
            for child in children.iter() {
                match child {
                    None => buf.push(2), // none tag
                    Some(c) => serialize_node(c, buf),
                }
            }
        }
    }
}

/// Deserialize a node from bytes. Returns (node_option, bytes_consumed).
fn deserialize_node(data: &[u8], pos: usize) -> Option<(Option<OctreeNode>, usize)> {
    if pos >= data.len() {
        return None;
    }

    match data[pos] {
        0 => {
            // Leaf
            if pos + 5 > data.len() { return None; }
            let lo = f32::from_le_bytes(data[pos + 1..pos + 5].try_into().ok()?);
            Some((Some(OctreeNode::Leaf { log_odds: lo }), pos + 5))
        }
        1 => {
            // Interior
            let mut children: [Option<OctreeNode>; 8] = Default::default();
            let mut p = pos + 1;
            for child in children.iter_mut() {
                let (node, next_pos) = deserialize_node(data, p)?;
                *child = node;
                p = next_pos;
            }
            Some((Some(OctreeNode::Interior { children: Box::new(children) }), p))
        }
        2 => {
            // None
            Some((None, pos + 1))
        }
        _ => None,
    }
}

/// Remove ground voxels below z threshold.
fn remove_ground_recursive(
    node: &mut OctreeNode,
    cx: f64, cy: f64, cz: f64, half: f64,
    z_threshold: f64,
    clamp_min: f32,
) {
    match node {
        OctreeNode::Leaf { log_odds } => {
            if cz + half <= z_threshold {
                // Entire voxel below ground → set to strongly free
                *log_odds = clamp_min;
            }
        }
        OctreeNode::Interior { children } => {
            for (i, child) in children.iter_mut().enumerate() {
                if let Some(c) = child {
                    let (ncx, ncy, ncz) = child_center(cx, cy, cz, half, i);
                    remove_ground_recursive(c, ncx, ncy, ncz, half * 0.5, z_threshold, clamp_min);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> OctreeConfig {
        OctreeConfig {
            max_depth: 4, // 16 subdivisions → resolution = 20/16 = 1.25m... too coarse
            root_half_size: 1.0,
            center: [0.0, 0.0, 0.0],
            ..Default::default()
        }
        // resolution = 2.0 / 16 = 0.125m
    }

    #[test]
    fn empty_octree() {
        let octree = Octree::new(small_config());
        assert!(octree.is_empty());
        assert_eq!(octree.node_count(), 0);
        assert_eq!(octree.leaf_count(), 0);
        assert_eq!(octree.query(0.0, 0.0, 0.0), OccupancyState::Unknown);
    }

    #[test]
    fn insert_single_point() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.0, 0.0, 0.0);

        assert!(!octree.is_empty());
        assert!(octree.node_count() > 0);
        assert!(octree.leaf_count() > 0);

        // Querying at the inserted point should return occupied
        let lo = octree.query_log_odds(0.0, 0.0, 0.0);
        assert!(
            lo > 0.0,
            "Inserted point should have positive log-odds: {}",
            lo
        );
        assert!(octree.is_occupied(0.0, 0.0, 0.0));
    }

    #[test]
    fn insert_multiple_points() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.5, 0.5, 0.5);
        octree.insert_point(-0.5, -0.5, -0.5);
        octree.insert_point(0.5, -0.5, 0.5);

        assert!(octree.is_occupied(0.5, 0.5, 0.5));
        assert!(octree.is_occupied(-0.5, -0.5, -0.5));
        assert!(octree.is_occupied(0.5, -0.5, 0.5));

        // Unvisited point should be unknown
        assert_eq!(octree.query(0.9, 0.9, -0.9), OccupancyState::Unknown);
    }

    #[test]
    fn insert_free_marks_free() {
        let mut octree = Octree::new(small_config());
        // Insert free several times to push below free threshold
        for _ in 0..5 {
            octree.insert_free(0.3, 0.3, 0.3);
        }

        let lo = octree.query_log_odds(0.3, 0.3, 0.3);
        assert!(lo < 0.0, "Free point should have negative log-odds: {}", lo);
        assert!(octree.is_free(0.3, 0.3, 0.3));
    }

    #[test]
    fn log_odds_accumulate() {
        let mut octree = Octree::new(small_config());

        octree.insert_point(0.0, 0.0, 0.0);
        let lo1 = octree.query_log_odds(0.0, 0.0, 0.0);

        octree.insert_point(0.0, 0.0, 0.0);
        let lo2 = octree.query_log_odds(0.0, 0.0, 0.0);

        assert!(
            lo2 > lo1,
            "Second hit should increase log-odds: {} -> {}",
            lo1, lo2
        );
    }

    #[test]
    fn log_odds_clamped() {
        let mut octree = Octree::new(small_config());

        // Insert many times to try to exceed clamp
        for _ in 0..100 {
            octree.insert_point(0.0, 0.0, 0.0);
        }

        let lo = octree.query_log_odds(0.0, 0.0, 0.0);
        assert!(
            lo <= octree.config().clamp_max,
            "Log-odds {} should be <= clamp_max {}",
            lo,
            octree.config().clamp_max
        );
    }

    #[test]
    fn out_of_bounds_ignored() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(100.0, 100.0, 100.0); // way outside bounds
        assert!(octree.is_empty()); // should not have created any nodes
    }

    #[test]
    fn contains_check() {
        let octree = Octree::new(small_config());
        assert!(octree.contains(0.0, 0.0, 0.0));
        assert!(octree.contains(0.9, 0.9, 0.9));
        assert!(!octree.contains(1.1, 0.0, 0.0));
    }

    #[test]
    fn resolution_correct() {
        let config = OctreeConfig {
            max_depth: 4,
            root_half_size: 1.0,
            ..Default::default()
        };
        // resolution = 2.0 / 2^4 = 2.0 / 16 = 0.125
        assert!((config.resolution() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn insert_ray_marks_free_and_occupied() {
        let mut octree = Octree::new(OctreeConfig {
            max_depth: 4,
            root_half_size: 2.0,
            ..Default::default()
        });

        // Ray from (-1, 0, 0) to (1, 0, 0)
        octree.insert_ray([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);

        // Endpoint should be occupied
        assert!(
            octree.is_occupied(1.0, 0.0, 0.0),
            "Ray endpoint should be occupied"
        );

        // Points along the ray (before endpoint) should be free
        let lo_mid = octree.query_log_odds(0.0, 0.0, 0.0);
        assert!(
            lo_mid < 0.0,
            "Ray interior should have negative log-odds (free): {}",
            lo_mid
        );
    }

    #[test]
    fn occupied_voxels_returns_correct() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);
        octree.insert_point(-0.3, -0.3, -0.3);

        let occupied = octree.occupied_voxels();
        assert_eq!(occupied.len(), 2, "Should have 2 occupied voxels");

        for (pos, lo) in &occupied {
            assert!(*lo > 0.0, "Occupied voxel should have positive log-odds");
            // Center should be near the inserted point (within resolution)
            let near_first = (pos[0] - 0.3).abs() < 0.2
                && (pos[1] - 0.3).abs() < 0.2
                && (pos[2] - 0.3).abs() < 0.2;
            let near_second = (pos[0] + 0.3).abs() < 0.2
                && (pos[1] + 0.3).abs() < 0.2
                && (pos[2] + 0.3).abs() < 0.2;
            assert!(
                near_first || near_second,
                "Occupied voxel at {:?} not near either inserted point",
                pos
            );
        }
    }

    #[test]
    fn clear_empties_octree() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.0, 0.0, 0.0);
        assert!(!octree.is_empty());

        octree.clear();
        assert!(octree.is_empty());
        assert_eq!(octree.node_count(), 0);
    }

    #[test]
    fn different_octants_independent() {
        let mut octree = Octree::new(small_config());

        // Insert in octant 0 (negative x, y, z)
        octree.insert_point(-0.5, -0.5, -0.5);
        // Insert in octant 7 (positive x, y, z)
        octree.insert_point(0.5, 0.5, 0.5);

        assert!(octree.is_occupied(-0.5, -0.5, -0.5));
        assert!(octree.is_occupied(0.5, 0.5, 0.5));

        // Other octants should be unknown
        assert_eq!(octree.query(0.5, -0.5, 0.5), OccupancyState::Unknown);
        assert_eq!(octree.query(-0.5, 0.5, -0.5), OccupancyState::Unknown);
    }

    #[test]
    fn depth_increases_with_insertions() {
        let mut octree = Octree::new(OctreeConfig {
            max_depth: 8,
            root_half_size: 1.0,
            ..Default::default()
        });

        octree.insert_point(0.0, 0.0, 0.0);
        let depth = octree.root.as_ref().unwrap().depth();
        assert!(
            depth >= 7,
            "Single point should create deep tree: depth={}",
            depth
        );
    }

    #[test]
    fn node_count_grows_with_spread() {
        let mut octree = Octree::new(small_config());

        octree.insert_point(0.5, 0.5, 0.5);
        let count1 = octree.node_count();

        octree.insert_point(-0.5, -0.5, -0.5);
        let count2 = octree.node_count();

        assert!(
            count2 > count1,
            "Spread points should increase node count: {} -> {}",
            count1, count2
        );
    }

    // ─── Pruning tests ───

    #[test]
    fn prune_merges_uniform_children() {
        let mut octree = Octree::new(OctreeConfig {
            max_depth: 2,
            root_half_size: 1.0,
            ..Default::default()
        });

        // Fill an entire octant uniformly
        for x in [0.25, 0.75] {
            for y in [0.25, 0.75] {
                for z in [0.25, 0.75] {
                    octree.insert_point(x, y, z);
                }
            }
        }

        let before = octree.node_count();
        let removed = octree.prune(1.0); // generous tolerance
        let after = octree.node_count();

        assert!(
            after <= before,
            "Pruning should reduce or maintain node count: {} -> {} (removed {})",
            before, after, removed
        );
    }

    #[test]
    fn prune_preserves_distinct_children() {
        let mut octree = Octree::new(OctreeConfig {
            max_depth: 2,
            root_half_size: 1.0,
            ..Default::default()
        });

        // Insert occupied and free in different octants
        octree.insert_point(0.5, 0.5, 0.5);
        for _ in 0..10 {
            octree.insert_free(-0.5, -0.5, -0.5);
        }

        let before = octree.node_count();
        octree.prune(0.001); // very tight tolerance
        let after = octree.node_count();

        // Distinct values shouldn't merge
        assert!(
            after >= before - 1,
            "Tight pruning shouldn't significantly reduce: {} -> {}",
            before, after
        );
    }

    // ─── Spatial query tests ───

    #[test]
    fn query_radius_finds_nearby() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.1, 0.0, 0.0);
        octree.insert_point(0.5, 0.5, 0.5);

        let results = octree.query_radius(0.0, 0.0, 0.0, 0.3);
        assert!(
            results.len() >= 1,
            "Should find at least the nearby point"
        );

        // The point at 0.5,0.5,0.5 is ~0.87 away, should NOT be in radius 0.3
        for (pos, _) in &results {
            let dist = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
            assert!(dist < 0.4, "Result at {:?} should be within radius", pos);
        }
    }

    #[test]
    fn query_radius_empty_for_far() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.5, 0.5, 0.5);

        let results = octree.query_radius(-0.9, -0.9, -0.9, 0.1);
        assert!(results.is_empty(), "No points near query location");
    }

    #[test]
    fn nearest_occupied_finds_closest() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.5, 0.0, 0.0);
        octree.insert_point(-0.5, 0.0, 0.0);

        let result = octree.nearest_occupied(0.3, 0.0, 0.0);
        assert!(result.is_some());
        let (pos, _, dist) = result.unwrap();
        // Should be the point at 0.5 (closer to 0.3 than -0.5)
        assert!(pos[0] > 0.0, "Nearest should be the positive-x point");
        assert!(dist < 0.4, "Distance should be small: {}", dist);
    }

    #[test]
    fn query_aabb_finds_contained() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);
        octree.insert_point(-0.3, -0.3, -0.3);

        // AABB covering only the positive octant
        let results = octree.query_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(
            results.len() >= 1,
            "Should find point in positive octant"
        );
        for (pos, _) in &results {
            assert!(pos[0] >= -0.1, "Should only find positive-octant points");
        }
    }

    // ─── Iterator/Visitor tests ───

    #[test]
    fn visit_leaves_counts_match() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);
        octree.insert_point(-0.3, -0.3, -0.3);

        let mut count = 0;
        octree.visit_leaves(|_, _, _| count += 1);
        assert_eq!(count, octree.leaf_count());
    }

    #[test]
    fn all_leaves_returns_vec() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.0, 0.0, 0.0);

        let leaves = octree.all_leaves();
        assert!(!leaves.is_empty());
        assert_eq!(leaves.len(), octree.leaf_count());
    }

    // ─── Collision bridge tests ───

    #[test]
    fn to_collision_spheres_returns_occupied() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);
        octree.insert_point(-0.3, -0.3, -0.3);

        let (positions, radii) = octree.to_collision_spheres();
        assert_eq!(positions.len(), 2);
        assert_eq!(radii.len(), 2);
        for r in &radii {
            assert!(*r > 0.0, "Radius should be positive: {}", r);
        }
    }

    #[test]
    fn to_collision_boxes_returns_occupied() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.5, 0.5, 0.5);

        let boxes = octree.to_collision_boxes();
        assert_eq!(boxes.len(), 1);
        let (center, half) = boxes[0];
        assert!(half > 0.0);
        assert!(center[0] > 0.0, "Box center should be in positive octant");
    }

    #[test]
    fn memory_bytes_nonzero() {
        let mut octree = Octree::new(small_config());
        assert_eq!(octree.memory_bytes(), 0);
        octree.insert_point(0.0, 0.0, 0.0);
        assert!(octree.memory_bytes() > 0);
    }

    // ─── Batch insertion tests ───

    #[test]
    fn insert_point_cloud_batch() {
        let mut octree = Octree::new(small_config());
        let sensor = [0.0, 0.0, 0.0];
        let points = vec![
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ];

        octree.insert_point_cloud(sensor, &points, None);

        assert!(octree.is_occupied(0.5, 0.0, 0.0));
        assert!(octree.is_occupied(0.0, 0.5, 0.0));
        assert!(octree.is_occupied(0.0, 0.0, 0.5));
    }

    #[test]
    fn insert_point_cloud_with_filter() {
        let mut octree = Octree::new(small_config());
        let sensor = [0.0, 0.0, 0.0];
        let points = vec![
            [0.3, 0.0, 0.0],  // within range
            [0.9, 0.0, 0.0],  // beyond max_range 0.5
        ];

        let filter = PointFilter {
            max_range: 0.5,
            ..Default::default()
        };

        octree.insert_point_cloud(sensor, &points, Some(&filter));

        assert!(octree.is_occupied(0.3, 0.0, 0.0));
        // 0.9 should NOT be inserted (beyond filter range)
        assert!(!octree.is_occupied(0.9, 0.0, 0.0));
    }

    #[test]
    fn point_filter_height_bounds() {
        let filter = PointFilter {
            z_min: 0.0,
            z_max: 1.0,
            ..Default::default()
        };

        assert!(filter.accept([0.0; 3], [0.5, 0.5, 0.5]));
        assert!(!filter.accept([0.0; 3], [0.5, 0.5, 1.5]));
        assert!(!filter.accept([0.0; 3], [0.5, 0.5, -0.5]));
    }

    #[test]
    fn point_filter_roi() {
        let filter = PointFilter {
            roi: Some([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            ..Default::default()
        };

        assert!(filter.accept([0.0; 3], [0.5, 0.5, 0.5]));
        assert!(!filter.accept([0.0; 3], [1.5, 0.5, 0.5]));
    }

    // ─── Temporal decay tests ───

    #[test]
    fn temporal_decay_reduces_occupancy() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);

        let lo_before = octree.query_log_odds(0.3, 0.3, 0.3);
        assert!(lo_before > 0.0);

        octree.apply_decay(0.1);
        let lo_after = octree.query_log_odds(0.3, 0.3, 0.3);
        assert!(
            lo_after < lo_before,
            "Decay should reduce log-odds: {} -> {}",
            lo_before, lo_after
        );
    }

    #[test]
    fn temporal_decay_clamps_at_zero() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);

        // Decay more than the log-odds value
        for _ in 0..100 {
            octree.apply_decay(0.5);
        }

        let lo = octree.query_log_odds(0.3, 0.3, 0.3);
        assert!(
            lo.abs() < 1e-6,
            "Decay should clamp at zero: {}",
            lo
        );
    }

    // ─── Serialization tests ───

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);
        octree.insert_point(-0.3, -0.3, -0.3);
        for _ in 0..5 {
            octree.insert_free(0.5, 0.5, 0.5);
        }

        let bytes = octree.to_bytes();
        let restored = Octree::from_bytes(&bytes).expect("Deserialization should succeed");

        // Check config matches
        assert_eq!(restored.config().max_depth, octree.config().max_depth);
        assert!((restored.config().root_half_size - octree.config().root_half_size).abs() < 1e-10);

        // Check occupancy matches
        assert_eq!(
            octree.query(0.3, 0.3, 0.3),
            restored.query(0.3, 0.3, 0.3),
        );
        assert_eq!(
            octree.query(-0.3, -0.3, -0.3),
            restored.query(-0.3, -0.3, -0.3),
        );

        // Log-odds should match
        let lo_orig = octree.query_log_odds(0.3, 0.3, 0.3);
        let lo_rest = restored.query_log_odds(0.3, 0.3, 0.3);
        assert!(
            (lo_orig - lo_rest).abs() < 1e-6,
            "Log-odds mismatch: {} vs {}",
            lo_orig, lo_rest
        );
    }

    #[test]
    fn serialize_empty_roundtrip() {
        let octree = Octree::new(small_config());
        let bytes = octree.to_bytes();
        let restored = Octree::from_bytes(&bytes).expect("Should deserialize empty");
        assert!(restored.is_empty());
    }

    // ─── 2D projection tests ───

    #[test]
    fn project_2d_shows_occupied_columns() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.3);

        let grid = octree.project_2d(-1.0, 1.0);
        assert!(grid.width > 0);
        assert!(grid.height > 0);
        assert!(
            grid.occupied_count() > 0,
            "Should have at least one occupied cell"
        );
    }

    #[test]
    fn project_2d_respects_height_filter() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, 0.8); // high point
        octree.insert_point(0.3, 0.3, -0.8); // low point

        // Only look at positive z
        let grid_high = octree.project_2d(0.5, 1.0);
        let grid_low = octree.project_2d(-1.0, -0.5);

        assert!(grid_high.occupied_count() > 0, "Should see high point");
        assert!(grid_low.occupied_count() > 0, "Should see low point");
    }

    // ─── Ground removal tests ───

    #[test]
    fn remove_ground_clears_low_voxels() {
        let mut octree = Octree::new(small_config());
        octree.insert_point(0.3, 0.3, -0.8); // below ground
        octree.insert_point(0.3, 0.3, 0.5);  // above ground

        assert!(octree.is_occupied(0.3, 0.3, -0.8));
        assert!(octree.is_occupied(0.3, 0.3, 0.5));

        octree.remove_ground(-0.5); // ground at z=-0.5

        // Below ground should be free now
        assert!(
            !octree.is_occupied(0.3, 0.3, -0.8),
            "Below-ground voxel should be cleared"
        );
        // Above ground should still be occupied
        assert!(
            octree.is_occupied(0.3, 0.3, 0.5),
            "Above-ground voxel should remain"
        );
    }

    // ─── Integration test ───

    #[test]
    fn full_perception_pipeline() {
        // Simulate: sensor at origin, point cloud of obstacles, query + collision bridge
        let mut octree = Octree::new(OctreeConfig {
            max_depth: 4,
            root_half_size: 2.0,
            ..Default::default()
        });

        let sensor = [0.0, 0.0, 0.5]; // sensor at 0.5m height

        // Simulate a table surface at z=0 in front of the sensor
        let mut points = Vec::new();
        for ix in -5..=5 {
            for iy in 0..5 {
                points.push([
                    ix as f64 * 0.1,
                    0.5 + iy as f64 * 0.1,
                    0.0,
                ]);
            }
        }

        // Simulate an object on the table at (0, 0.7, 0.15)
        for iz in 0..3 {
            points.push([0.0, 0.7, 0.05 + iz as f64 * 0.05]);
        }

        let filter = PointFilter {
            max_range: 3.0,
            z_min: -0.5,
            z_max: 2.0,
            ..Default::default()
        };

        octree.insert_point_cloud(sensor, &points, Some(&filter));

        // Remove ground
        octree.remove_ground(-0.1);

        // Query: object should still be occupied
        let occupied = octree.occupied_voxels();
        assert!(
            !occupied.is_empty(),
            "Should have occupied voxels after insertion"
        );

        // Collision bridge
        let (spheres, radii) = octree.to_collision_spheres();
        assert!(
            !spheres.is_empty(),
            "Should produce collision spheres"
        );
        assert_eq!(spheres.len(), radii.len());

        // 2D projection
        let grid = octree.project_2d(-0.1, 2.0);
        assert!(grid.occupied_count() > 0, "2D grid should show obstacles");

        // Serialization roundtrip
        let bytes = octree.to_bytes();
        let restored = Octree::from_bytes(&bytes).unwrap();
        assert_eq!(octree.leaf_count(), restored.leaf_count());
    }
}
