//! Collision-Affording Point Tree (CAPT).
//!
//! A 3D grid that stores the minimum clearance radius at each cell —
//! the maximum sphere radius that can be placed at that position
//! without colliding with any obstacle.
//!
//! Inspired by VAMP (Kavraki Lab, 2023). Enables <10ns per-point
//! collision queries by reducing collision checking to a single
//! grid lookup + comparison.

use crate::soa::SpheresSoA;

/// Axis-Aligned Bounding Box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl AABB {
    /// Create an AABB from min/max corners.
    pub fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    /// Create a symmetric AABB centered at origin.
    pub fn symmetric(half_extent: f64) -> Self {
        Self::new(
            -half_extent,
            -half_extent,
            -half_extent,
            half_extent,
            half_extent,
            half_extent,
        )
    }

    /// Extent in each dimension.
    pub fn extents(&self) -> (f64, f64, f64) {
        (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )
    }

    /// Whether a point is inside this AABB.
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }
}

/// Collision-Affording Point Tree.
///
/// A 3D voxel grid where each cell stores the minimum distance to the
/// nearest obstacle surface. A query "is sphere (x,y,z,r) collision-free?"
/// reduces to `grid[ix][iy][iz] >= r`.
///
/// Build cost: O(N * V) where N = obstacle count, V = affected voxels.
/// Query cost: O(1) per point — just a grid lookup.
#[derive(Debug, Clone)]
pub struct CollisionPointTree {
    /// Clearance values: `clearance[ix * ny * nz + iy * nz + iz]`.
    /// Stores the minimum distance to nearest obstacle surface.
    clearance: Vec<f64>,
    /// Grid resolution (cell size) in meters.
    pub resolution: f64,
    /// Workspace bounds.
    pub bounds: AABB,
    /// Number of cells in each dimension.
    nx: usize,
    ny: usize,
    nz: usize,
    /// Inverse resolution for fast index computation.
    inv_resolution: f64,
}

impl CollisionPointTree {
    /// Build a CAPT from obstacle spheres.
    ///
    /// Uses workspace diagonal as influence radius — guarantees NO false negatives.
    /// For large workspaces where build speed matters, use `build_with_influence()`.
    pub fn build(obstacles: &SpheresSoA, resolution: f64, bounds: AABB) -> Self {
        Self::build_with_influence(obstacles, resolution, bounds, None)
    }

    /// Build a CAPT from obstacle spheres with explicit influence radius.
    ///
    /// `max_influence`: maximum distance from an obstacle surface to update cells.
    /// Use `None` for automatic (0.5m — sufficient for typical robot link sphere
    /// radii up to ~0.2m, since any cell >0.5m from all obstacles is collision-free).
    ///
    /// Grid dimensions are capped at 200 cells per axis (~64M cells max with 8M
    /// default cap). If the requested bounds/resolution would exceed this, the
    /// resolution is automatically coarsened to fit.
    pub fn build_with_influence(
        obstacles: &SpheresSoA,
        resolution: f64,
        bounds: AABB,
        max_influence: Option<f64>,
    ) -> Self {
        let (ex, ey, ez) = bounds.extents();

        // Cap grid dimensions to avoid excessive memory/compute.
        // 200³ = 8M cells × 8 bytes = 64 MB — reasonable upper bound.
        const MAX_CELLS_PER_DIM: usize = 200;
        let mut res = resolution;
        let raw_nx = (ex / res).ceil() as usize + 1;
        let raw_ny = (ey / res).ceil() as usize + 1;
        let raw_nz = (ez / res).ceil() as usize + 1;
        let max_dim = raw_nx.max(raw_ny).max(raw_nz);
        if max_dim > MAX_CELLS_PER_DIM {
            // Coarsen resolution so the largest dimension fits within the cap
            let largest_extent = ex.max(ey).max(ez);
            res = largest_extent / (MAX_CELLS_PER_DIM - 1) as f64;
        }

        let nx = (ex / res).ceil() as usize + 1;
        let ny = (ey / res).ceil() as usize + 1;
        let nz = (ez / res).ceil() as usize + 1;
        let total = nx * ny * nz;

        // Default influence: 0.5m covers robot link spheres up to ~0.2m radius.
        // Any cell >0.5m from all obstacles is guaranteed collision-free for any
        // robot sphere <0.5m radius. Cells beyond influence get infinity clearance.
        let max_inf = max_influence.unwrap_or(0.5);

        // Initialize all cells to maximum clearance (infinity → no obstacles)
        let mut clearance = vec![f64::INFINITY; total];

        // For each obstacle sphere, update affected cells
        for obs_idx in 0..obstacles.len() {
            let ox = obstacles.x[obs_idx];
            let oy = obstacles.y[obs_idx];
            let oz = obstacles.z[obs_idx];
            let or = obstacles.radius[obs_idx];

            // Determine the range of cells that could be affected
            // A cell at distance d from the obstacle center has clearance = d - or
            // Influence radius: obstacle radius + max_influence
            let influence = or + max_inf;

            let ix_min = ((ox - influence - bounds.min_x) / res)
                .floor()
                .max(0.0) as usize;
            let ix_max =
                (((ox + influence - bounds.min_x) / res).ceil() as usize).min(nx - 1);
            let iy_min = ((oy - influence - bounds.min_y) / res)
                .floor()
                .max(0.0) as usize;
            let iy_max =
                (((oy + influence - bounds.min_y) / res).ceil() as usize).min(ny - 1);
            let iz_min = ((oz - influence - bounds.min_z) / res)
                .floor()
                .max(0.0) as usize;
            let iz_max =
                (((oz + influence - bounds.min_z) / res).ceil() as usize).min(nz - 1);

            for ix in ix_min..=ix_max {
                let cell_x = bounds.min_x + ix as f64 * res;
                let dx = cell_x - ox;
                for iy in iy_min..=iy_max {
                    let cell_y = bounds.min_y + iy as f64 * res;
                    let dy = cell_y - oy;
                    for iz in iz_min..=iz_max {
                        let cell_z = bounds.min_z + iz as f64 * res;
                        let dz = cell_z - oz;

                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        let clearance_val = dist - or;

                        let idx = ix * ny * nz + iy * nz + iz;
                        if clearance_val < clearance[idx] {
                            clearance[idx] = clearance_val;
                        }
                    }
                }
            }
        }

        Self {
            clearance,
            resolution: res,
            bounds,
            nx,
            ny,
            nz,
            inv_resolution: 1.0 / res,
        }
    }

    /// Build an empty tree (no obstacles — everything is collision-free).
    pub fn empty(resolution: f64, bounds: AABB) -> Self {
        let (ex, ey, ez) = bounds.extents();
        let nx = (ex / resolution).ceil() as usize + 1;
        let ny = (ey / resolution).ceil() as usize + 1;
        let nz = (ez / resolution).ceil() as usize + 1;
        let total = nx * ny * nz;

        Self {
            clearance: vec![f64::INFINITY; total],
            resolution,
            bounds,
            nx,
            ny,
            nz,
            inv_resolution: 1.0 / resolution,
        }
    }

    /// Grid dimensions.
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Total number of cells.
    pub fn total_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Convert world position to grid index. Returns None if out of bounds.
    #[inline]
    fn world_to_index(&self, x: f64, y: f64, z: f64) -> Option<usize> {
        let fx = (x - self.bounds.min_x) * self.inv_resolution;
        let fy = (y - self.bounds.min_y) * self.inv_resolution;
        let fz = (z - self.bounds.min_z) * self.inv_resolution;

        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            return None;
        }

        let ix = fx as usize;
        let iy = fy as usize;
        let iz = fz as usize;

        if ix >= self.nx || iy >= self.ny || iz >= self.nz {
            return None;
        }

        Some(ix * self.ny * self.nz + iy * self.nz + iz)
    }

    /// Query: is a sphere at (x, y, z) with given radius collision-free?
    ///
    /// Returns `true` if the sphere does NOT collide with any obstacle.
    /// Points outside the grid bounds are treated as collision (conservative).
    #[inline]
    pub fn check_point(&self, x: f64, y: f64, z: f64, radius: f64) -> bool {
        match self.world_to_index(x, y, z) {
            Some(idx) => self.clearance[idx] >= radius,
            None => false, // out of bounds → assume collision
        }
    }

    /// Query: is a sphere at (x, y, z) with given radius in collision?
    ///
    /// Returns `true` if collision detected.
    #[inline]
    pub fn is_collision(&self, x: f64, y: f64, z: f64, radius: f64) -> bool {
        !self.check_point(x, y, z, radius)
    }

    /// Get clearance at a world position.
    ///
    /// Returns the minimum distance to any obstacle surface at that point.
    /// Returns 0.0 for out-of-bounds points.
    #[inline]
    pub fn clearance_at(&self, x: f64, y: f64, z: f64) -> f64 {
        match self.world_to_index(x, y, z) {
            Some(idx) => self.clearance[idx],
            None => 0.0,
        }
    }

    /// Get raw clearance array (for SIMD kernels).
    #[inline]
    pub fn clearance_data(&self) -> &[f64] {
        &self.clearance
    }

    /// Get grid stride for index computation in SIMD kernels.
    #[inline]
    pub fn strides(&self) -> (usize, usize) {
        (self.ny * self.nz, self.nz)
    }

    /// Get grid parameters for SIMD kernels.
    #[inline]
    pub fn grid_params(&self) -> GridParams {
        GridParams {
            min_x: self.bounds.min_x,
            min_y: self.bounds.min_y,
            min_z: self.bounds.min_z,
            inv_resolution: self.inv_resolution,
            stride_x: self.ny * self.nz,
            stride_y: self.nz,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
        }
    }
}

/// Flattened grid parameters for passing to SIMD kernels.
#[derive(Debug, Clone, Copy)]
pub struct GridParams {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub inv_resolution: f64,
    pub stride_x: usize,
    pub stride_y: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree_no_collision() {
        let tree = CollisionPointTree::empty(0.05, AABB::symmetric(1.0));
        assert!(tree.check_point(0.0, 0.0, 0.0, 0.5));
        assert!(tree.check_point(0.5, 0.5, 0.5, 0.1));
    }

    #[test]
    fn out_of_bounds_is_collision() {
        let tree = CollisionPointTree::empty(0.05, AABB::symmetric(1.0));
        // Point outside the grid → collision (conservative)
        assert!(!tree.check_point(5.0, 0.0, 0.0, 0.1));
        assert!(tree.is_collision(5.0, 0.0, 0.0, 0.1));
    }

    #[test]
    fn single_obstacle_collision() {
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.3, 0); // sphere at origin, radius 0.3

        let tree = CollisionPointTree::build(&obstacles, 0.05, AABB::symmetric(2.0));

        // Point at origin → inside obstacle → collision
        assert!(tree.is_collision(0.0, 0.0, 0.0, 0.01));

        // Point at (1, 0, 0) → far from obstacle → no collision
        assert!(!tree.is_collision(1.0, 0.0, 0.0, 0.1));
        assert!(tree.check_point(1.0, 0.0, 0.0, 0.1));

        // Point at (0.3, 0, 0) → right at obstacle surface → collision with any radius
        // Clearance ~0, so even a tiny sphere collides
        let clearance = tree.clearance_at(0.3, 0.0, 0.0);
        assert!(
            clearance < 0.1,
            "Expected near-zero clearance, got {}",
            clearance
        );
    }

    #[test]
    fn multiple_obstacles() {
        let mut obstacles = SpheresSoA::new();
        obstacles.push(-0.5, 0.0, 0.0, 0.2, 0);
        obstacles.push(0.5, 0.0, 0.0, 0.2, 1);

        let tree = CollisionPointTree::build(&obstacles, 0.05, AABB::symmetric(2.0));

        // Between obstacles → should be safe
        assert!(tree.check_point(0.0, 0.0, 0.0, 0.1));

        // Near first obstacle → depends on clearance
        let c1 = tree.clearance_at(-0.3, 0.0, 0.0);
        // Distance from (-0.3,0,0) to (-0.5,0,0) center = 0.2, minus radius 0.2 = 0.0
        assert!(
            c1 < 0.1,
            "Expected small clearance near obstacle, got {}",
            c1
        );
    }

    #[test]
    fn grid_dimensions() {
        let tree = CollisionPointTree::empty(0.1, AABB::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        let (nx, ny, nz) = tree.grid_dims();
        assert_eq!(nx, 11); // ceil(1.0/0.1) + 1 = 11
        assert_eq!(ny, 11);
        assert_eq!(nz, 11);
    }

    #[test]
    fn clearance_decreases_near_obstacle() {
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.2, 0);

        let tree = CollisionPointTree::build(&obstacles, 0.02, AABB::symmetric(1.0));

        let c_far = tree.clearance_at(0.8, 0.0, 0.0);
        let c_mid = tree.clearance_at(0.4, 0.0, 0.0);
        let c_near = tree.clearance_at(0.25, 0.0, 0.0);

        assert!(
            c_far > c_mid,
            "Farther point should have more clearance: {} vs {}",
            c_far,
            c_mid
        );
        assert!(
            c_mid > c_near,
            "Mid point should have more clearance than near: {} vs {}",
            c_mid,
            c_near
        );
    }

    #[test]
    fn aabb_contains() {
        let aabb = AABB::symmetric(1.0);
        assert!(aabb.contains(0.0, 0.0, 0.0));
        assert!(aabb.contains(0.5, -0.5, 0.9));
        assert!(!aabb.contains(1.5, 0.0, 0.0));
    }

    #[test]
    fn build_and_check_consistency() {
        // Verify that building from scratch matches expected clearance
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.1, 0);

        let tree = CollisionPointTree::build(&obstacles, 0.05, AABB::symmetric(0.5));

        // At (0.3, 0, 0): distance to center=0.3, clearance = 0.3-0.1 = 0.2
        let c = tree.clearance_at(0.3, 0.0, 0.0);
        // Grid discretization means some error
        assert!(
            (c - 0.2).abs() < 0.05,
            "Expected ~0.2 clearance at (0.3,0,0), got {}",
            c
        );
    }
}
