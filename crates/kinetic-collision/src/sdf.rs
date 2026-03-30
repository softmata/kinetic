//! Signed Distance Field (SDF) — voxel grid with cached signed distances.
//!
//! Provides O(1) distance queries for collision checking and trajectory optimization.
//! Supports incremental updates: add/remove obstacle spheres without full rebuild.
//!
//! # Architecture
//!
//! The SDF is a 3D voxel grid where each voxel stores the signed distance to
//! the nearest obstacle surface. Negative values indicate interior (collision),
//! positive values indicate free space.
//!
//! Incremental updates re-compute only voxels within the affected radius
//! when obstacles are added or removed.

use crate::soa::SpheresSoA;

/// SDF configuration.
#[derive(Debug, Clone)]
pub struct SDFConfig {
    /// Voxel size in meters (default: 0.02 = 2cm).
    pub resolution: f64,
    /// Workspace bounds: [x_min, y_min, z_min, x_max, y_max, z_max].
    pub bounds: [f64; 6],
    /// Maximum distance to compute (truncation distance). Voxels farther
    /// than this from any obstacle are clamped to this value. Default: 0.5m.
    pub truncation: f64,
}

impl Default for SDFConfig {
    fn default() -> Self {
        Self {
            resolution: 0.02,
            bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
            truncation: 0.5,
        }
    }
}

/// A persistent signed distance field on a regular voxel grid.
#[derive(Debug)]
pub struct SignedDistanceField {
    /// Flat voxel array storing signed distances.
    voxels: Vec<f64>,
    /// Grid dimensions (nx, ny, nz).
    dims: [usize; 3],
    /// Grid origin (x_min, y_min, z_min).
    origin: [f64; 3],
    /// Voxel size.
    resolution: f64,
    /// Truncation distance.
    truncation: f64,
    /// Tracked obstacle spheres for incremental updates.
    obstacles: Vec<Sphere>,
}

/// A tracked obstacle sphere.
#[derive(Debug, Clone)]
struct Sphere {
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    id: usize,
}

impl SignedDistanceField {
    /// Create an empty SDF from configuration.
    pub fn new(config: &SDFConfig) -> Self {
        let nx = ((config.bounds[3] - config.bounds[0]) / config.resolution).ceil() as usize;
        let ny = ((config.bounds[4] - config.bounds[1]) / config.resolution).ceil() as usize;
        let nz = ((config.bounds[5] - config.bounds[2]) / config.resolution).ceil() as usize;

        let total = nx.max(1) * ny.max(1) * nz.max(1);

        Self {
            voxels: vec![config.truncation; total],
            dims: [nx.max(1), ny.max(1), nz.max(1)],
            origin: [config.bounds[0], config.bounds[1], config.bounds[2]],
            resolution: config.resolution,
            truncation: config.truncation,
            obstacles: Vec::new(),
        }
    }

    /// Grid dimensions (nx, ny, nz).
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Grid resolution (meters per voxel).
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Grid origin (minimum corner).
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        self.voxels.len()
    }

    /// Number of tracked obstacles.
    pub fn num_obstacles(&self) -> usize {
        self.obstacles.len()
    }

    /// Build SDF from obstacle spheres (full rebuild).
    pub fn from_spheres(spheres: &SpheresSoA, config: &SDFConfig) -> Self {
        let mut sdf = Self::new(config);
        for i in 0..spheres.len() {
            sdf.obstacles.push(Sphere {
                x: spheres.x[i],
                y: spheres.y[i],
                z: spheres.z[i],
                r: spheres.radius[i],
                id: spheres.link_id[i],
            });
        }
        sdf.rebuild();
        sdf
    }

    /// Full rebuild: recompute all voxels from tracked obstacles.
    pub fn rebuild(&mut self) {
        let [nx, ny, nz] = self.dims;

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let (wx, wy, wz) = self.voxel_center(ix, iy, iz);
                    let mut min_dist = self.truncation;

                    for obs in &self.obstacles {
                        let dx = wx - obs.x;
                        let dy = wy - obs.y;
                        let dz = wz - obs.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt() - obs.r;
                        min_dist = min_dist.min(dist);
                    }

                    self.voxels[ix + iy * nx + iz * nx * ny] = min_dist.max(-self.truncation);
                }
            }
        }
    }

    /// Add an obstacle sphere incrementally.
    ///
    /// Only updates voxels within `radius + truncation` of the sphere center.
    pub fn add_sphere(&mut self, x: f64, y: f64, z: f64, r: f64, id: usize) {
        let obs = Sphere { x, y, z, r, id };
        self.obstacles.push(obs.clone());
        self.update_region(&obs);
    }

    /// Remove an obstacle by id. Triggers partial rebuild of affected region.
    pub fn remove_obstacle(&mut self, id: usize) {
        if let Some(pos) = self.obstacles.iter().position(|o| o.id == id) {
            let removed = self.obstacles.remove(pos);
            // Need to rebuild the region around the removed obstacle
            self.rebuild_region(removed.x, removed.y, removed.z, removed.r + self.truncation);
        }
    }

    /// Query signed distance at a world-space point.
    ///
    /// Uses trilinear interpolation between neighboring voxels.
    /// Returns `truncation` if outside grid bounds.
    pub fn distance_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let fx = (x - self.origin[0]) / self.resolution;
        let fy = (y - self.origin[1]) / self.resolution;
        let fz = (z - self.origin[2]) / self.resolution;

        let ix = fx.floor() as isize;
        let iy = fy.floor() as isize;
        let iz = fz.floor() as isize;

        // Bounds check
        let [nx, ny, nz] = self.dims;
        if ix < 0 || iy < 0 || iz < 0
            || ix >= (nx as isize - 1)
            || iy >= (ny as isize - 1)
            || iz >= (nz as isize - 1)
        {
            return self.truncation;
        }

        let ix = ix as usize;
        let iy = iy as usize;
        let iz = iz as usize;

        // Trilinear interpolation weights
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let tz = fz - fz.floor();

        let v000 = self.voxels[ix + iy * nx + iz * nx * ny];
        let v100 = self.voxels[(ix + 1) + iy * nx + iz * nx * ny];
        let v010 = self.voxels[ix + (iy + 1) * nx + iz * nx * ny];
        let v110 = self.voxels[(ix + 1) + (iy + 1) * nx + iz * nx * ny];
        let v001 = self.voxels[ix + iy * nx + (iz + 1) * nx * ny];
        let v101 = self.voxels[(ix + 1) + iy * nx + (iz + 1) * nx * ny];
        let v011 = self.voxels[ix + (iy + 1) * nx + (iz + 1) * nx * ny];
        let v111 = self.voxels[(ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny];

        let c00 = v000 * (1.0 - tx) + v100 * tx;
        let c10 = v010 * (1.0 - tx) + v110 * tx;
        let c01 = v001 * (1.0 - tx) + v101 * tx;
        let c11 = v011 * (1.0 - tx) + v111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Query whether a point is in collision (distance < 0).
    pub fn is_occupied(&self, x: f64, y: f64, z: f64) -> bool {
        self.distance_at(x, y, z) < 0.0
    }

    /// Check if a sphere at (x, y, z) with radius r is in collision.
    pub fn sphere_collision(&self, x: f64, y: f64, z: f64, r: f64) -> bool {
        self.distance_at(x, y, z) < r
    }

    /// Compute the gradient of the distance field at a point.
    ///
    /// Returns `[∂d/∂x, ∂d/∂y, ∂d/∂z]` via central finite differences.
    /// The gradient points away from the nearest obstacle (direction of steepest ascent).
    /// Useful for CHOMP/STOMP trajectory optimization cost gradients.
    pub fn gradient_at(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let h = self.resolution;
        let dx = (self.distance_at(x + h, y, z) - self.distance_at(x - h, y, z)) / (2.0 * h);
        let dy = (self.distance_at(x, y + h, z) - self.distance_at(x, y - h, z)) / (2.0 * h);
        let dz = (self.distance_at(x, y, z + h) - self.distance_at(x, y, z - h)) / (2.0 * h);
        [dx, dy, dz]
    }

    /// Query distance and gradient simultaneously (avoids redundant lookups).
    ///
    /// Returns `(distance, [∂d/∂x, ∂d/∂y, ∂d/∂z])`.
    pub fn distance_and_gradient(&self, x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
        let d = self.distance_at(x, y, z);
        let grad = self.gradient_at(x, y, z);
        (d, grad)
    }

    /// Compute collision cost and gradient for a sphere.
    ///
    /// Cost = max(0, `margin` - distance). Gradient is the negated SDF gradient
    /// scaled by the cost (pushes sphere away from obstacles).
    /// Used directly by trajectory optimizers (CHOMP/STOMP).
    pub fn sphere_cost_and_gradient(
        &self,
        x: f64, y: f64, z: f64,
        radius: f64,
        margin: f64,
    ) -> (f64, [f64; 3]) {
        let (d, grad) = self.distance_and_gradient(x, y, z);
        let effective_dist = d - radius;

        if effective_dist >= margin {
            return (0.0, [0.0, 0.0, 0.0]);
        }

        let cost = margin - effective_dist;
        // Gradient of cost = -gradient of distance (push away from obstacle)
        let cost_grad = [-grad[0] * cost, -grad[1] * cost, -grad[2] * cost];
        (cost, cost_grad)
    }

    /// Get voxel center in world coordinates.
    fn voxel_center(&self, ix: usize, iy: usize, iz: usize) -> (f64, f64, f64) {
        (
            self.origin[0] + (ix as f64 + 0.5) * self.resolution,
            self.origin[1] + (iy as f64 + 0.5) * self.resolution,
            self.origin[2] + (iz as f64 + 0.5) * self.resolution,
        )
    }

    /// Update voxels in the region affected by a newly added sphere.
    fn update_region(&mut self, obs: &Sphere) {
        let affect_radius = obs.r + self.truncation;
        let [nx, ny, nz] = self.dims;

        let ix_min = ((obs.x - affect_radius - self.origin[0]) / self.resolution).floor().max(0.0) as usize;
        let ix_max = ((obs.x + affect_radius - self.origin[0]) / self.resolution).ceil().min(nx as f64) as usize;
        let iy_min = ((obs.y - affect_radius - self.origin[1]) / self.resolution).floor().max(0.0) as usize;
        let iy_max = ((obs.y + affect_radius - self.origin[1]) / self.resolution).ceil().min(ny as f64) as usize;
        let iz_min = ((obs.z - affect_radius - self.origin[2]) / self.resolution).floor().max(0.0) as usize;
        let iz_max = ((obs.z + affect_radius - self.origin[2]) / self.resolution).ceil().min(nz as f64) as usize;

        for iz in iz_min..iz_max {
            for iy in iy_min..iy_max {
                for ix in ix_min..ix_max {
                    let (wx, wy, wz) = self.voxel_center(ix, iy, iz);
                    let dx = wx - obs.x;
                    let dy = wy - obs.y;
                    let dz = wz - obs.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt() - obs.r;

                    let idx = ix + iy * nx + iz * nx * ny;
                    self.voxels[idx] = self.voxels[idx].min(dist.max(-self.truncation));
                }
            }
        }
    }

    /// Rebuild voxels in a region (used after obstacle removal).
    fn rebuild_region(&mut self, cx: f64, cy: f64, cz: f64, affect_radius: f64) {
        let [nx, ny, nz] = self.dims;

        let ix_min = ((cx - affect_radius - self.origin[0]) / self.resolution).floor().max(0.0) as usize;
        let ix_max = ((cx + affect_radius - self.origin[0]) / self.resolution).ceil().min(nx as f64) as usize;
        let iy_min = ((cy - affect_radius - self.origin[1]) / self.resolution).floor().max(0.0) as usize;
        let iy_max = ((cy + affect_radius - self.origin[1]) / self.resolution).ceil().min(ny as f64) as usize;
        let iz_min = ((cz - affect_radius - self.origin[2]) / self.resolution).floor().max(0.0) as usize;
        let iz_max = ((cz + affect_radius - self.origin[2]) / self.resolution).ceil().min(nz as f64) as usize;

        for iz in iz_min..iz_max {
            for iy in iy_min..iy_max {
                for ix in ix_min..ix_max {
                    let (wx, wy, wz) = self.voxel_center(ix, iy, iz);
                    let mut min_dist = self.truncation;

                    for obs in &self.obstacles {
                        let dx = wx - obs.x;
                        let dy = wy - obs.y;
                        let dz = wz - obs.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt() - obs.r;
                        min_dist = min_dist.min(dist);
                    }

                    let idx = ix + iy * nx + iz * nx * ny;
                    self.voxels[idx] = min_dist.max(-self.truncation);
                }
            }
        }
    }
}

/// Multi-resolution SDF: coarse grid for fast broadphase + fine grid near obstacles.
///
/// Queries automatically select resolution: if the coarse SDF distance is large
/// (point is far from obstacles), the coarse value is returned directly.
/// If the point is near obstacles, the fine grid is consulted for accuracy.
#[derive(Debug)]
pub struct MultiResolutionSDF {
    /// Coarse grid covering the full workspace.
    coarse: SignedDistanceField,
    /// Fine grid covering the same workspace at higher resolution.
    fine: SignedDistanceField,
    /// Threshold: if coarse distance > this, skip fine query. Default: 2 * fine_resolution.
    pub refinement_threshold: f64,
}

impl MultiResolutionSDF {
    /// Create a multi-resolution SDF.
    ///
    /// `coarse_resolution`: voxel size for broadphase (e.g., 0.1m).
    /// `fine_resolution`: voxel size for detailed queries (e.g., 0.02m).
    pub fn new(bounds: [f64; 6], coarse_resolution: f64, fine_resolution: f64, truncation: f64) -> Self {
        let coarse = SignedDistanceField::new(&SDFConfig {
            resolution: coarse_resolution,
            bounds,
            truncation,
        });
        let fine = SignedDistanceField::new(&SDFConfig {
            resolution: fine_resolution,
            bounds,
            truncation,
        });

        Self {
            coarse,
            fine,
            refinement_threshold: fine_resolution * 2.0,
        }
    }

    /// Build from obstacle spheres.
    pub fn from_spheres(spheres: &SpheresSoA, bounds: [f64; 6], coarse_res: f64, fine_res: f64, truncation: f64) -> Self {
        let coarse = SignedDistanceField::from_spheres(spheres, &SDFConfig {
            resolution: coarse_res,
            bounds,
            truncation,
        });
        let fine = SignedDistanceField::from_spheres(spheres, &SDFConfig {
            resolution: fine_res,
            bounds,
            truncation,
        });

        Self {
            coarse,
            fine,
            refinement_threshold: fine_res * 2.0,
        }
    }

    /// Add an obstacle sphere to both grids.
    pub fn add_sphere(&mut self, x: f64, y: f64, z: f64, r: f64, id: usize) {
        self.coarse.add_sphere(x, y, z, r, id);
        self.fine.add_sphere(x, y, z, r, id);
    }

    /// Remove an obstacle from both grids.
    pub fn remove_obstacle(&mut self, id: usize) {
        self.coarse.remove_obstacle(id);
        self.fine.remove_obstacle(id);
    }

    /// Query distance with automatic resolution selection.
    ///
    /// If the coarse grid indicates the point is far from obstacles,
    /// returns the coarse value (fast). Otherwise queries the fine grid (accurate).
    pub fn distance_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let coarse_d = self.coarse.distance_at(x, y, z);
        if coarse_d > self.refinement_threshold {
            return coarse_d;
        }
        self.fine.distance_at(x, y, z)
    }

    /// Query gradient with automatic resolution selection.
    pub fn gradient_at(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let coarse_d = self.coarse.distance_at(x, y, z);
        if coarse_d > self.refinement_threshold {
            return self.coarse.gradient_at(x, y, z);
        }
        self.fine.gradient_at(x, y, z)
    }

    /// Distance and gradient with auto resolution.
    pub fn distance_and_gradient(&self, x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
        let coarse_d = self.coarse.distance_at(x, y, z);
        if coarse_d > self.refinement_threshold {
            return (coarse_d, self.coarse.gradient_at(x, y, z));
        }
        self.fine.distance_and_gradient(x, y, z)
    }

    /// Check collision.
    pub fn is_occupied(&self, x: f64, y: f64, z: f64) -> bool {
        self.distance_at(x, y, z) < 0.0
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        (self.coarse.num_voxels() + self.fine.num_voxels()) * std::mem::size_of::<f64>()
    }

    /// Coarse grid voxel count.
    pub fn coarse_voxels(&self) -> usize {
        self.coarse.num_voxels()
    }

    /// Fine grid voxel count.
    pub fn fine_voxels(&self) -> usize {
        self.fine.num_voxels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SDFConfig {
        SDFConfig {
            resolution: 0.1,
            bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            truncation: 0.5,
        }
    }

    #[test]
    fn sdf_empty() {
        let sdf = SignedDistanceField::new(&small_config());
        assert!(sdf.num_voxels() > 0);
        assert_eq!(sdf.num_obstacles(), 0);
        // All distances should be truncation
        assert!((sdf.distance_at(0.0, 0.0, 0.0) - 0.5).abs() < 0.01);
        assert!(!sdf.is_occupied(0.0, 0.0, 0.0));
    }

    #[test]
    fn sdf_from_spheres() {
        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 0.2, 0);
        let sdf = SignedDistanceField::from_spheres(&obs, &small_config());

        assert_eq!(sdf.num_obstacles(), 1);
        // Center of sphere should be negative (inside)
        assert!(sdf.distance_at(0.0, 0.0, 0.0) < 0.0, "Center should be inside");
        // Far away should be positive
        assert!(sdf.distance_at(0.9, 0.9, 0.9) > 0.0, "Far point should be outside");
    }

    #[test]
    fn sdf_add_sphere_incremental() {
        let mut sdf = SignedDistanceField::new(&small_config());

        // Initially empty — center is free
        assert!(!sdf.is_occupied(0.0, 0.0, 0.0));

        // Add obstacle at origin
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        // Now center should be occupied
        assert!(sdf.is_occupied(0.0, 0.0, 0.0));
        assert_eq!(sdf.num_obstacles(), 1);
    }

    #[test]
    fn sdf_remove_obstacle() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 42);

        assert!(sdf.is_occupied(0.0, 0.0, 0.0));

        sdf.remove_obstacle(42);
        assert_eq!(sdf.num_obstacles(), 0);
        assert!(!sdf.is_occupied(0.0, 0.0, 0.0));
    }

    #[test]
    fn sdf_sphere_collision_query() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.5, 0.0, 0.0, 0.2, 1);

        // Robot sphere at origin with r=0.1 — should not collide (distance ~0.2)
        assert!(!sdf.sphere_collision(0.0, 0.0, 0.0, 0.1));

        // Robot sphere closer (x=0.25) with r=0.1 — distance ~0.05, still no collision
        // sphere_collision checks distance_at < r, so distance_at(0.25) ≈ 0.05, r=0.1 → collides
        assert!(sdf.sphere_collision(0.25, 0.0, 0.0, 0.1));
    }

    #[test]
    fn sdf_trilinear_interpolation() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.3, 1);

        // Points at different distances should give smooth gradient
        let d1 = sdf.distance_at(0.0, 0.0, 0.0);
        let d2 = sdf.distance_at(0.2, 0.0, 0.0);
        let d3 = sdf.distance_at(0.4, 0.0, 0.0);

        assert!(d1 < d2, "Distance should increase away from obstacle");
        assert!(d2 < d3, "Distance should increase away from obstacle");
    }

    #[test]
    fn sdf_out_of_bounds() {
        let sdf = SignedDistanceField::new(&small_config());
        // Outside grid bounds should return truncation
        let d = sdf.distance_at(5.0, 5.0, 5.0);
        assert!((d - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sdf_multiple_obstacles() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(-0.5, 0.0, 0.0, 0.1, 1);
        sdf.add_sphere(0.5, 0.0, 0.0, 0.1, 2);

        assert_eq!(sdf.num_obstacles(), 2);
        assert!(sdf.is_occupied(-0.5, 0.0, 0.0));
        assert!(sdf.is_occupied(0.5, 0.0, 0.0));
        assert!(!sdf.is_occupied(0.0, 0.0, 0.0)); // Between obstacles
    }

    #[test]
    fn sdf_config_defaults() {
        let config = SDFConfig::default();
        assert_eq!(config.resolution, 0.02);
        assert_eq!(config.truncation, 0.5);
    }

    #[test]
    fn sdf_dims() {
        let config = SDFConfig {
            resolution: 0.5,
            bounds: [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            truncation: 0.5,
        };
        let sdf = SignedDistanceField::new(&config);
        assert_eq!(sdf.dims(), [4, 4, 4]);
        assert_eq!(sdf.num_voxels(), 64);
    }

    // --- Gradient tests ---

    #[test]
    fn sdf_gradient_points_away_from_obstacle() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        // Gradient at a point on +X side should point in +X direction
        let grad = sdf.gradient_at(0.4, 0.0, 0.0);
        assert!(grad[0] > 0.0, "Gradient X should be positive (away from obstacle): {}", grad[0]);
        // Y and Z components should be near zero (symmetric)
        assert!(grad[1].abs() < 0.5, "Gradient Y should be small: {}", grad[1]);
        assert!(grad[2].abs() < 0.5, "Gradient Z should be small: {}", grad[2]);
    }

    #[test]
    fn sdf_gradient_magnitude() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        // For a sphere SDF, gradient magnitude should be approximately 1.0
        // (SDF of a sphere has unit gradient everywhere except at the center)
        let grad = sdf.gradient_at(0.5, 0.0, 0.0);
        let mag = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
        assert!(
            mag > 0.5 && mag < 1.5,
            "Gradient magnitude should be ~1.0 for sphere SDF, got {}",
            mag
        );
    }

    #[test]
    fn sdf_distance_and_gradient() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        let (d, grad) = sdf.distance_and_gradient(0.4, 0.0, 0.0);
        assert!(d > 0.0, "Should be outside obstacle");
        assert!(grad[0] > 0.0, "Gradient should point away");
    }

    #[test]
    fn sdf_sphere_cost_outside_margin() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        // Point far from obstacle — cost should be 0
        let (cost, grad) = sdf.sphere_cost_and_gradient(0.8, 0.0, 0.0, 0.05, 0.1);
        assert_eq!(cost, 0.0, "Cost should be 0 far from obstacle");
        assert_eq!(grad, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn sdf_sphere_cost_inside_margin() {
        let mut sdf = SignedDistanceField::new(&small_config());
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        // Point near obstacle — cost should be positive
        let (cost, grad) = sdf.sphere_cost_and_gradient(0.3, 0.0, 0.0, 0.05, 0.2);
        assert!(cost > 0.0, "Cost should be positive near obstacle: {}", cost);
        // Gradient should push away from obstacle (negative X since cost gradient = -SDF gradient * cost)
        assert!(grad[0] < 0.0, "Cost gradient should push away: {:?}", grad);
    }

    #[test]
    fn sdf_gradient_empty_field() {
        let sdf = SignedDistanceField::new(&small_config());
        let grad = sdf.gradient_at(0.0, 0.0, 0.0);
        assert!(grad[0].abs() < 0.01, "Empty SDF gradient should be ~0: {:?}", grad);
        assert!(grad[1].abs() < 0.01);
        assert!(grad[2].abs() < 0.01);
    }

    // --- Multi-resolution SDF tests ---

    #[test]
    fn multi_res_basic() {
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut sdf = MultiResolutionSDF::new(bounds, 0.2, 0.05, 0.5);
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        assert!(sdf.is_occupied(0.0, 0.0, 0.0));
        assert!(!sdf.is_occupied(0.8, 0.8, 0.8));
    }

    #[test]
    fn multi_res_fine_more_voxels() {
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let sdf = MultiResolutionSDF::new(bounds, 0.2, 0.05, 0.5);
        assert!(
            sdf.fine_voxels() > sdf.coarse_voxels() * 10,
            "Fine grid should have many more voxels: fine={}, coarse={}",
            sdf.fine_voxels(),
            sdf.coarse_voxels()
        );
    }

    #[test]
    fn multi_res_from_spheres() {
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 0.2, 1);
        let sdf = MultiResolutionSDF::from_spheres(&obs, bounds, 0.2, 0.05, 0.5);

        assert!(sdf.is_occupied(0.0, 0.0, 0.0));
        assert!(!sdf.is_occupied(0.8, 0.8, 0.8));
    }

    #[test]
    fn multi_res_gradient() {
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut sdf = MultiResolutionSDF::new(bounds, 0.2, 0.05, 0.5);
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 1);

        let grad = sdf.gradient_at(0.4, 0.0, 0.0);
        assert!(grad[0] > 0.0, "Gradient should point away from obstacle");
    }

    #[test]
    fn multi_res_remove() {
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut sdf = MultiResolutionSDF::new(bounds, 0.2, 0.05, 0.5);
        sdf.add_sphere(0.0, 0.0, 0.0, 0.2, 42);
        assert!(sdf.is_occupied(0.0, 0.0, 0.0));

        sdf.remove_obstacle(42);
        assert!(!sdf.is_occupied(0.0, 0.0, 0.0));
    }
}
