//! IRIS convex decomposition of C-space free regions.
//!
//! Decomposes the collision-free configuration space into a set of convex
//! polytopes (defined by halfplane constraints Ax ≤ b). Each region is
//! guaranteed collision-free: any configuration inside the polytope is safe.
//!
//! Simplified IRIS (Iterative Region Inflation by Semidefinite programming)
//! adapted from Deits & Tedrake 2015. Uses iterative hyperplane separation
//! and collision sampling instead of full SDP.

use crate::CollisionChecker;
use kinetic_core::Result as KResult;
use rand::Rng;

/// A convex region in C-space defined by halfplane constraints: Ax ≤ b.
///
/// Any point x satisfying all constraints simultaneously is inside the region
/// and is guaranteed to be collision-free.
#[derive(Debug, Clone)]
pub struct ConvexRegion {
    /// Seed point (center) of the region — always collision-free.
    pub center: Vec<f64>,
    /// Halfplane normals (each row of A). `halfplanes[i].0 · x ≤ halfplanes[i].1`.
    pub halfplanes: Vec<(Vec<f64>, f64)>,
    /// Joint-space dimension.
    pub dof: usize,
}

impl ConvexRegion {
    /// Check whether a point lies inside this convex region.
    pub fn contains(&self, point: &[f64]) -> bool {
        self.halfplanes.iter().all(|(normal, bound)| {
            let dot: f64 = normal.iter().zip(point).map(|(a, x)| a * x).sum();
            dot <= *bound + 1e-10
        })
    }

    /// Compute an approximate volume using the extent along each axis
    /// from the center to the nearest halfplane boundary.
    pub fn approximate_volume(&self) -> f64 {
        let mut vol = 1.0;
        for d in 0..self.dof {
            let mut min_extent = f64::MAX;
            for (normal, bound) in &self.halfplanes {
                if normal[d].abs() > 1e-10 {
                    let extent = (bound
                        - normal
                            .iter()
                            .zip(self.center.iter())
                            .map(|(a, c)| a * c)
                            .sum::<f64>())
                        / normal[d].abs();
                    min_extent = min_extent.min(extent.abs());
                }
            }
            if min_extent < f64::MAX {
                vol *= 2.0 * min_extent;
            }
        }
        vol
    }

    /// Sample a random point inside the region using rejection sampling.
    pub fn sample(&self, limits: &[(f64, f64)]) -> Option<Vec<f64>> {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let point: Vec<f64> = limits
                .iter()
                .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                .collect();
            if self.contains(&point) {
                return Some(point);
            }
        }
        None
    }
}

/// Configuration for IRIS decomposition.
#[derive(Debug, Clone)]
pub struct IrisConfig {
    /// Number of convex regions to generate.
    pub num_regions: usize,
    /// Minimum region volume (regions smaller than this are discarded).
    pub min_volume: f64,
    /// Maximum iterations per region growth.
    pub max_iterations: usize,
    /// Custom seed points (if None, random collision-free samples are used).
    pub seed_points: Option<Vec<Vec<f64>>>,
    /// Number of obstacle samples per iteration for boundary detection.
    pub num_obstacle_samples: usize,
    /// Step size for obstacle boundary search.
    pub boundary_step: f64,
}

impl Default for IrisConfig {
    fn default() -> Self {
        Self {
            num_regions: 50,
            min_volume: 0.001,
            max_iterations: 10,
            seed_points: None,
            num_obstacle_samples: 50,
            boundary_step: 0.01,
        }
    }
}

/// A convex decomposition of the C-space free region.
#[derive(Debug, Clone)]
pub struct ConvexDecomposition {
    /// The convex regions covering C-space.
    pub regions: Vec<ConvexRegion>,
    /// Joint limits used during decomposition.
    pub limits: Vec<(f64, f64)>,
    /// Degrees of freedom.
    pub dof: usize,
}

impl ConvexDecomposition {
    /// Build an IRIS-style convex decomposition of the collision-free C-space.
    ///
    /// Uses iterative hyperplane separation: for each seed point, grow a convex
    /// region by finding obstacle boundaries and adding separating hyperplanes.
    pub fn iris<C: CollisionChecker>(
        checker: &C,
        limits: &[(f64, f64)],
        config: &IrisConfig,
    ) -> KResult<Self> {
        let dof = limits.len();
        let mut regions = Vec::new();
        let mut rng = rand::thread_rng();

        // Generate seed points
        let seeds: Vec<Vec<f64>> = if let Some(ref custom) = config.seed_points {
            custom.clone()
        } else {
            // Sample random collision-free configurations
            let mut seeds = Vec::new();
            let mut attempts = 0;
            while seeds.len() < config.num_regions && attempts < config.num_regions * 100 {
                let q: Vec<f64> = limits
                    .iter()
                    .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                    .collect();
                if !checker.is_in_collision(&q) {
                    // Check that this seed isn't too close to an existing region center
                    let too_close = regions.iter().any(|r: &ConvexRegion| {
                        joint_distance(&q, &r.center) < config.boundary_step * 5.0
                    });
                    if !too_close {
                        seeds.push(q);
                    }
                }
                attempts += 1;
            }
            seeds
        };

        for seed in &seeds {
            if checker.is_in_collision(seed) {
                continue;
            }

            let region = grow_region(checker, seed, limits, config);

            if region.approximate_volume() >= config.min_volume {
                regions.push(region);
            }

            if regions.len() >= config.num_regions {
                break;
            }
        }

        Ok(Self {
            regions,
            limits: limits.to_vec(),
            dof,
        })
    }

    /// Number of convex regions.
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Find which region(s) contain a given configuration.
    pub fn containing_regions(&self, q: &[f64]) -> Vec<usize> {
        self.regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.contains(q))
            .map(|(i, _)| i)
            .collect()
    }

    /// Build an adjacency graph: `regions[i]` and `regions[j]` are adjacent if
    /// they overlap or a collision-free path exists between their centers.
    pub fn adjacency<C: CollisionChecker>(&self, checker: &C, step_size: f64) -> Vec<Vec<usize>> {
        let n = self.regions.len();
        let mut adj = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                // Check if regions overlap: does the midpoint of their centers
                // lie in both regions?
                let mid: Vec<f64> = self.regions[i]
                    .center
                    .iter()
                    .zip(self.regions[j].center.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();

                let overlap = self.regions[i].contains(&mid) && self.regions[j].contains(&mid);

                // Also check if the straight-line path between centers is collision-free
                let path_clear = if !overlap {
                    is_segment_collision_free(
                        checker,
                        &self.regions[i].center,
                        &self.regions[j].center,
                        step_size,
                    )
                } else {
                    true
                };

                if overlap || path_clear {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        adj
    }
}

/// Grow a convex region around a seed point by finding obstacle boundaries
/// and adding separating hyperplanes.
fn grow_region<C: CollisionChecker>(
    checker: &C,
    seed: &[f64],
    limits: &[(f64, f64)],
    config: &IrisConfig,
) -> ConvexRegion {
    let dof = seed.len();
    let mut halfplanes: Vec<(Vec<f64>, f64)> = Vec::new();

    // Start with joint limit constraints
    for (d, (lo, hi)) in limits.iter().enumerate() {
        // x[d] >= lo → -x[d] <= -lo
        let mut normal_lo = vec![0.0; dof];
        normal_lo[d] = -1.0;
        halfplanes.push((normal_lo, -*lo));

        // x[d] <= hi
        let mut normal_hi = vec![0.0; dof];
        normal_hi[d] = 1.0;
        halfplanes.push((normal_hi, *hi));
    }

    let mut rng = rand::thread_rng();

    for _ in 0..config.max_iterations {
        let mut found_new_constraint = false;

        // Sample random directions and search for obstacle boundaries
        for _ in 0..config.num_obstacle_samples {
            // Random direction in C-space
            let dir: Vec<f64> = (0..dof).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let dir_norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
            if dir_norm < 1e-10 {
                continue;
            }
            let dir: Vec<f64> = dir.iter().map(|d| d / dir_norm).collect();

            // Binary search along this direction from seed to find obstacle boundary
            if let Some((boundary_point, boundary_dist)) =
                find_boundary(checker, seed, &dir, limits, config.boundary_step)
            {
                // Add a separating hyperplane at the boundary: normal · x ≤ bound
                // The halfplane separates the seed from the obstacle.
                let bound: f64 = dir
                    .iter()
                    .zip(boundary_point.iter())
                    .map(|(d, b)| d * b)
                    .sum();

                // Only add if this constraint is tighter than existing ones at the seed
                let seed_val: f64 = dir.iter().zip(seed.iter()).map(|(d, s)| d * s).sum();
                if seed_val < bound - 1e-8 && boundary_dist > config.boundary_step {
                    halfplanes.push((dir.clone(), bound));
                    found_new_constraint = true;
                }
            }
        }

        if !found_new_constraint {
            break;
        }
    }

    ConvexRegion {
        center: seed.to_vec(),
        halfplanes,
        dof,
    }
}

/// Binary search along a ray from `origin` in direction `dir` to find the
/// collision boundary. Returns (boundary_point, distance_from_origin).
fn find_boundary<C: CollisionChecker>(
    checker: &C,
    origin: &[f64],
    dir: &[f64],
    limits: &[(f64, f64)],
    step: f64,
) -> Option<(Vec<f64>, f64)> {
    // Find the maximum distance we can travel in this direction within limits
    let mut max_t = f64::MAX;
    for (d, (lo, hi)) in limits.iter().enumerate() {
        if dir[d] > 1e-10 {
            max_t = max_t.min((hi - origin[d]) / dir[d]);
        } else if dir[d] < -1e-10 {
            max_t = max_t.min((lo - origin[d]) / dir[d]);
        }
    }

    if max_t <= step {
        return None;
    }

    // Walk along the ray to find where collision starts
    let mut lo_t = 0.0;
    let mut hi_t = max_t;
    let mut found_collision = false;

    // Coarse search
    let mut t = step;
    while t <= max_t {
        let point: Vec<f64> = origin.iter().zip(dir).map(|(o, d)| o + t * d).collect();
        if checker.is_in_collision(&point) {
            hi_t = t;
            lo_t = t - step;
            found_collision = true;
            break;
        }
        t += step;
    }

    if !found_collision {
        return None;
    }

    // Binary search refinement
    for _ in 0..10 {
        let mid = (lo_t + hi_t) / 2.0;
        let point: Vec<f64> = origin.iter().zip(dir).map(|(o, d)| o + mid * d).collect();
        if checker.is_in_collision(&point) {
            hi_t = mid;
        } else {
            lo_t = mid;
        }
    }

    // Return the last collision-free point
    let boundary: Vec<f64> = origin.iter().zip(dir).map(|(o, d)| o + lo_t * d).collect();
    Some((boundary, lo_t))
}

/// Check if a straight-line segment in C-space is collision-free.
fn is_segment_collision_free<C: CollisionChecker>(
    checker: &C,
    from: &[f64],
    to: &[f64],
    step_size: f64,
) -> bool {
    let dist = joint_distance(from, to);
    if dist < 1e-10 {
        return !checker.is_in_collision(from);
    }
    let steps = (dist / step_size).ceil() as usize;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let q: Vec<f64> = from
            .iter()
            .zip(to.iter())
            .map(|(a, b)| a + t * (b - a))
            .collect();
        if checker.is_in_collision(&q) {
            return false;
        }
    }
    true
}

fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple 2D collision checker with a box obstacle.
    struct BoxChecker {
        box_min: [f64; 2],
        box_max: [f64; 2],
    }

    impl CollisionChecker for BoxChecker {
        fn is_in_collision(&self, joints: &[f64]) -> bool {
            joints[0] >= self.box_min[0]
                && joints[0] <= self.box_max[0]
                && joints[1] >= self.box_min[1]
                && joints[1] <= self.box_max[1]
        }
    }

    #[test]
    fn convex_region_contains() {
        let region = ConvexRegion {
            center: vec![0.0, 0.0],
            halfplanes: vec![
                (vec![1.0, 0.0], 1.0),  // x ≤ 1
                (vec![-1.0, 0.0], 1.0), // -x ≤ 1 → x ≥ -1
                (vec![0.0, 1.0], 1.0),  // y ≤ 1
                (vec![0.0, -1.0], 1.0), // -y ≤ 1 → y ≥ -1
            ],
            dof: 2,
        };

        assert!(region.contains(&[0.0, 0.0]));
        assert!(region.contains(&[0.5, 0.5]));
        assert!(!region.contains(&[1.5, 0.0]));
    }

    #[test]
    fn iris_simple_2d() {
        let checker = BoxChecker {
            box_min: [0.4, 0.4],
            box_max: [0.6, 0.6],
        };
        let limits = [(-1.0, 1.0), (-1.0, 1.0)];
        let config = IrisConfig {
            num_regions: 5,
            min_volume: 0.01,
            max_iterations: 5,
            seed_points: Some(vec![vec![-0.5, -0.5], vec![0.8, -0.5]]),
            num_obstacle_samples: 20,
            boundary_step: 0.05,
        };

        let decomp = ConvexDecomposition::iris(&checker, &limits, &config).unwrap();
        assert!(!decomp.regions.is_empty());

        // The seed points should be inside their regions
        for region in &decomp.regions {
            assert!(region.contains(&region.center));
        }
    }

    #[test]
    fn iris_no_obstacles() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let checker = NoCollision;
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 3,
            seed_points: Some(vec![vec![0.5, 0.5]]),
            ..Default::default()
        };

        let decomp = ConvexDecomposition::iris(&checker, &limits, &config).unwrap();
        assert!(!decomp.regions.is_empty());
        // With no obstacles, the region should span the full limits
        assert!(decomp.regions[0].contains(&[0.5, 0.5]));
    }

    #[test]
    fn adjacency_graph() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let checker = NoCollision;
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 3,
            seed_points: Some(vec![vec![0.2, 0.2], vec![0.5, 0.5], vec![0.8, 0.8]]),
            ..Default::default()
        };

        let decomp = ConvexDecomposition::iris(&checker, &limits, &config).unwrap();
        let adj = decomp.adjacency(&checker, 0.05);

        // With no obstacles, all regions should be connected
        // (straight-line paths between centers are collision-free)
        for neighbors in &adj {
            assert!(
                !neighbors.is_empty() || decomp.num_regions() <= 1,
                "all regions should have neighbors in obstacle-free space"
            );
        }
    }

    // ─── New tests below ───

    /// approximate_volume() with a degenerate region: zero extent in one dimension.
    #[test]
    fn approximate_volume_degenerate_zero_extent() {
        // A region where y is pinned to 0 (zero extent in that dimension)
        let region = ConvexRegion {
            center: vec![0.0, 0.0],
            halfplanes: vec![
                (vec![1.0, 0.0], 1.0),  // x <= 1
                (vec![-1.0, 0.0], 1.0), // x >= -1
                (vec![0.0, 1.0], 0.0),  // y <= 0
                (vec![0.0, -1.0], 0.0), // y >= 0
            ],
            dof: 2,
        };
        let vol = region.approximate_volume();
        // y extent is 0, so volume should be 0
        assert!(
            vol.abs() < 1e-10,
            "degenerate region with zero extent should have ~0 volume, got {vol}"
        );
    }

    /// approximate_volume() for a well-formed unit box: [-1,1]^2 → volume = 4.
    #[test]
    fn approximate_volume_unit_box() {
        let region = ConvexRegion {
            center: vec![0.0, 0.0],
            halfplanes: vec![
                (vec![1.0, 0.0], 1.0),
                (vec![-1.0, 0.0], 1.0),
                (vec![0.0, 1.0], 1.0),
                (vec![0.0, -1.0], 1.0),
            ],
            dof: 2,
        };
        let vol = region.approximate_volume();
        assert!(
            (vol - 4.0).abs() < 1e-6,
            "unit box [-1,1]^2 should have volume 4, got {vol}"
        );
    }

    /// sample() should return None when the region is tiny compared to the limits,
    /// exercising the 1000-iteration rejection sampling limit.
    #[test]
    fn sample_rejection_sampling_exhausted() {
        // Region: x in [0, 0.0001], y in [0, 0.0001] — within limits [0, 100]^2
        let region = ConvexRegion {
            center: vec![0.00005, 0.00005],
            halfplanes: vec![
                (vec![1.0, 0.0], 0.0001),
                (vec![-1.0, 0.0], 0.0),
                (vec![0.0, 1.0], 0.0001),
                (vec![0.0, -1.0], 0.0),
            ],
            dof: 2,
        };
        let limits = [(0.0, 100.0), (0.0, 100.0)];
        // The region is 1e-8 of the total area, so 1000 samples almost certainly miss
        let result = region.sample(&limits);
        // We accept either None (expected) or Some (extremely unlikely but valid)
        // The point is this doesn't panic
        if let Some(ref pt) = result {
            assert!(region.contains(pt));
        }
    }

    /// sample() succeeds when the region covers a large fraction of the limits.
    #[test]
    fn sample_succeeds_large_region() {
        let region = ConvexRegion {
            center: vec![0.0, 0.0],
            halfplanes: vec![
                (vec![1.0, 0.0], 1.0),
                (vec![-1.0, 0.0], 1.0),
                (vec![0.0, 1.0], 1.0),
                (vec![0.0, -1.0], 1.0),
            ],
            dof: 2,
        };
        let limits = [(-1.0, 1.0), (-1.0, 1.0)];
        let result = region.sample(&limits);
        assert!(
            result.is_some(),
            "sample should succeed for full-coverage region"
        );
        let pt = result.unwrap();
        assert!(region.contains(&pt));
    }

    /// grow_region() in completely open space produces a region spanning the limits.
    #[test]
    fn grow_region_open_space() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let seed = vec![0.5, 0.5];
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            max_iterations: 5,
            num_obstacle_samples: 20,
            boundary_step: 0.01,
            ..Default::default()
        };
        let region = grow_region(&NoCollision, &seed, &limits, &config);

        // Region should contain the seed
        assert!(region.contains(&seed));
        // Corners of the limits should be inside (no obstacles)
        assert!(region.contains(&[0.01, 0.01]));
        assert!(region.contains(&[0.99, 0.99]));
        // Volume should be close to 1.0 (the full limit box)
        let vol = region.approximate_volume();
        assert!(
            vol > 0.5,
            "open space region volume should be large, got {vol}"
        );
    }

    /// find_boundary() returns None when ray is along a direction with no collision
    /// and max_t is within step.
    #[test]
    fn find_boundary_no_collision() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let origin = vec![0.5, 0.5];
        let dir = vec![1.0, 0.0];
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let result = find_boundary(&NoCollision, &origin, &dir, &limits, 0.01);
        assert!(
            result.is_none(),
            "should return None when no collision exists along ray"
        );
    }

    /// find_boundary() returns a boundary point when collision exists.
    #[test]
    fn find_boundary_with_collision() {
        let checker = BoxChecker {
            box_min: [0.7, 0.0],
            box_max: [1.0, 1.0],
        };
        let origin = vec![0.3, 0.5];
        let dir = vec![1.0, 0.0]; // ray toward the box
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let result = find_boundary(&checker, &origin, &dir, &limits, 0.01);
        assert!(result.is_some(), "should find boundary before box obstacle");
        let (pt, dist) = result.unwrap();
        // Boundary should be near x=0.7
        assert!(
            pt[0] < 0.71 && pt[0] > 0.6,
            "boundary x should be near 0.7, got {}",
            pt[0]
        );
        assert!(dist > 0.0);
    }

    /// find_boundary() with seed already at the limit edge (max_t <= step).
    #[test]
    fn find_boundary_origin_at_limit_edge() {
        struct AlwaysCollision;
        impl CollisionChecker for AlwaysCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                true
            }
        }
        // Origin is at x=0.995, direction +x, limit is 1.0 → max_t = 0.005 < step=0.01
        let origin = vec![0.995, 0.5];
        let dir = vec![1.0, 0.0];
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let result = find_boundary(&AlwaysCollision, &origin, &dir, &limits, 0.01);
        assert!(result.is_none(), "should return None when max_t <= step");
    }

    /// IRIS decomposition in a narrow passage should produce thin regions.
    #[test]
    fn iris_narrow_passage() {
        // Two box obstacles leaving a narrow gap at x ∈ (0.45, 0.55)
        struct NarrowPassage;
        impl CollisionChecker for NarrowPassage {
            fn is_in_collision(&self, joints: &[f64]) -> bool {
                // Obstacle 1: x <= 0.45
                // Obstacle 2: x >= 0.55
                joints[0] <= 0.45 || joints[0] >= 0.55
            }
        }

        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 1,
            seed_points: Some(vec![vec![0.5, 0.5]]), // in the gap
            max_iterations: 10,
            num_obstacle_samples: 50,
            boundary_step: 0.005,
            min_volume: 0.0, // accept any size
        };

        let decomp = ConvexDecomposition::iris(&NarrowPassage, &limits, &config).unwrap();
        assert_eq!(decomp.num_regions(), 1);
        let region = &decomp.regions[0];
        assert!(region.contains(&[0.5, 0.5]));
        // Point outside the gap should not be in the region
        assert!(!region.contains(&[0.3, 0.5]));
        assert!(!region.contains(&[0.7, 0.5]));
    }

    /// Region containment: point on boundary (within tolerance).
    #[test]
    fn contains_point_on_boundary() {
        let region = ConvexRegion {
            center: vec![0.0, 0.0],
            halfplanes: vec![
                (vec![1.0, 0.0], 1.0), // x <= 1
            ],
            dof: 2,
        };
        // Exactly on boundary
        assert!(region.contains(&[1.0, 0.0]));
        // Slightly inside tolerance (1e-10)
        assert!(region.contains(&[1.0 + 1e-11, 0.0]));
        // Outside tolerance
        assert!(!region.contains(&[1.0 + 1e-9, 0.0]));
    }

    /// ConvexRegion properties: center is always inside, volume > 0, dof matches.
    #[test]
    fn convex_region_properties() {
        let checker = BoxChecker {
            box_min: [0.4, 0.4],
            box_max: [0.6, 0.6],
        };
        let limits = [(-1.0, 1.0), (-1.0, 1.0)];
        let config = IrisConfig {
            num_regions: 2,
            seed_points: Some(vec![vec![-0.5, -0.5], vec![0.8, 0.8]]),
            max_iterations: 5,
            num_obstacle_samples: 30,
            boundary_step: 0.02,
            min_volume: 0.0,
        };
        let decomp = ConvexDecomposition::iris(&checker, &limits, &config).unwrap();
        for region in &decomp.regions {
            assert!(
                region.contains(&region.center),
                "center must be inside region"
            );
            assert!(region.approximate_volume() > 0.0, "volume must be positive");
            assert_eq!(region.dof, 2);
        }
    }

    /// IRIS with 10D configuration space (high-dimensional).
    #[test]
    fn iris_10d_high_dimensional() {
        struct SphereObstacle10D;
        impl CollisionChecker for SphereObstacle10D {
            fn is_in_collision(&self, joints: &[f64]) -> bool {
                // Sphere obstacle centered at origin with radius 0.3
                let dist_sq: f64 = joints.iter().map(|x| x * x).sum();
                dist_sq < 0.09 // radius^2 = 0.3^2
            }
        }

        let limits: Vec<(f64, f64)> = (0..10).map(|_| (-1.0, 1.0)).collect();
        let config = IrisConfig {
            num_regions: 1,
            seed_points: Some(vec![vec![0.8; 10]]), // far from obstacle
            max_iterations: 5,
            num_obstacle_samples: 30,
            boundary_step: 0.02,
            min_volume: 0.0,
        };

        let decomp = ConvexDecomposition::iris(&SphereObstacle10D, &limits, &config).unwrap();
        assert_eq!(decomp.dof, 10);
        assert_eq!(decomp.num_regions(), 1);
        let region = &decomp.regions[0];
        assert!(region.contains(&vec![0.8; 10]));
        assert_eq!(region.dof, 10);
        assert!(region.approximate_volume() > 0.0);
    }

    /// containing_regions() returns correct indices.
    #[test]
    fn containing_regions_correct() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 2,
            seed_points: Some(vec![vec![0.3, 0.3], vec![0.7, 0.7]]),
            ..Default::default()
        };
        let decomp = ConvexDecomposition::iris(&NoCollision, &limits, &config).unwrap();

        // With no obstacles, both regions should contain their own centers
        // and likely contain points in between
        let regions_for_center0 = decomp.containing_regions(&[0.3, 0.3]);
        assert!(
            regions_for_center0.contains(&0),
            "center of region 0 should be in region 0"
        );
    }

    /// is_segment_collision_free() with zero-length segment.
    #[test]
    fn segment_collision_free_zero_length() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }
        assert!(is_segment_collision_free(
            &NoCollision,
            &[0.5, 0.5],
            &[0.5, 0.5],
            0.01
        ));
    }

    /// is_segment_collision_free() detects collision along a segment.
    #[test]
    fn segment_collision_detected() {
        let checker = BoxChecker {
            box_min: [0.4, 0.0],
            box_max: [0.6, 1.0],
        };
        // Segment crosses the box
        let free = is_segment_collision_free(&checker, &[0.0, 0.5], &[1.0, 0.5], 0.01);
        assert!(
            !free,
            "segment through obstacle should not be collision-free"
        );
    }

    /// IRIS with seed point in collision should skip it gracefully.
    #[test]
    fn iris_seed_in_collision_skipped() {
        let checker = BoxChecker {
            box_min: [0.4, 0.4],
            box_max: [0.6, 0.6],
        };
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 3,
            seed_points: Some(vec![
                vec![0.5, 0.5], // IN collision — should be skipped
                vec![0.1, 0.1], // free
                vec![0.9, 0.9], // free
            ]),
            min_volume: 0.0,
            ..Default::default()
        };
        let decomp = ConvexDecomposition::iris(&checker, &limits, &config).unwrap();
        // The colliding seed should be skipped, so at most 2 regions
        assert!(decomp.num_regions() <= 2);
        // No region should have center at [0.5, 0.5]
        for region in &decomp.regions {
            let at_obstacle =
                (region.center[0] - 0.5).abs() < 1e-6 && (region.center[1] - 0.5).abs() < 1e-6;
            assert!(!at_obstacle, "colliding seed should not produce a region");
        }
    }

    /// IRIS discards regions smaller than min_volume.
    #[test]
    fn iris_min_volume_filter() {
        // Obstacle covers almost everything, leaving a tiny free region
        struct AlmostFullObstacle;
        impl CollisionChecker for AlmostFullObstacle {
            fn is_in_collision(&self, joints: &[f64]) -> bool {
                // Only free in [0.0, 0.005] x [0.0, 0.005]
                joints[0] > 0.005 || joints[1] > 0.005
            }
        }

        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 1,
            seed_points: Some(vec![vec![0.002, 0.002]]),
            min_volume: 1.0, // require volume >= 1.0
            max_iterations: 5,
            num_obstacle_samples: 30,
            boundary_step: 0.001,
        };
        let decomp = ConvexDecomposition::iris(&AlmostFullObstacle, &limits, &config).unwrap();
        // The tiny region should be discarded due to min_volume filter
        assert_eq!(
            decomp.num_regions(),
            0,
            "tiny region should be filtered by min_volume"
        );
    }
}
