//! Point cloud processing utilities.
//!
//! Voxel grid downsampling, RANSAC plane removal, workspace cropping,
//! and statistical outlier removal.

use kinetic_collision::AABB;

/// Voxel grid downsampling.
///
/// Replaces all points within each voxel cell by their centroid.
/// Reduces point count while preserving spatial distribution.
pub fn voxel_downsample(points: &[[f64; 3]], voxel_size: f64) -> Vec<[f64; 3]> {
    use std::collections::HashMap;

    if points.is_empty() || voxel_size <= 0.0 {
        return points.to_vec();
    }

    let inv = 1.0 / voxel_size;
    type VoxelAccum = (f64, f64, f64, usize);
    // Map voxel index → (sum_x, sum_y, sum_z, count)
    let mut buckets: HashMap<(i64, i64, i64), VoxelAccum> = HashMap::new();

    for p in points {
        let ix = (p[0] * inv).floor() as i64;
        let iy = (p[1] * inv).floor() as i64;
        let iz = (p[2] * inv).floor() as i64;

        let entry = buckets.entry((ix, iy, iz)).or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += p[0];
        entry.1 += p[1];
        entry.2 += p[2];
        entry.3 += 1;
    }

    buckets
        .values()
        .map(|(sx, sy, sz, n)| {
            let n = *n as f64;
            [sx / n, sy / n, sz / n]
        })
        .collect()
}

/// RANSAC plane removal.
///
/// Fits a dominant plane (e.g., floor/table) and returns the remaining points
/// and optionally the plane normal (unit vector).
/// `distance_threshold` is the max distance from plane for a point to be an inlier.
/// `max_iterations` controls how many random plane samples to try.
pub(crate) fn ransac_remove_plane(
    points: &[[f64; 3]],
    distance_threshold: f64,
    max_iterations: usize,
) -> (Vec<[f64; 3]>, Option<[f64; 3]>) {
    if points.len() < 3 {
        return (points.to_vec(), None);
    }

    let n = points.len();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_normal = [0.0f64; 3];

    // Simple deterministic pseudo-random using the point count as seed
    let mut rng_state: u64 = n as u64 ^ 0xDEAD_BEEF;
    let next_rand = |state: &mut u64| -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n
    };

    for _ in 0..max_iterations {
        // Pick 3 random non-degenerate points
        let i0 = next_rand(&mut rng_state);
        let i1 = next_rand(&mut rng_state);
        let i2 = next_rand(&mut rng_state);
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        let p0 = points[i0];
        let p1 = points[i1];
        let p2 = points[i2];

        // Compute plane normal via cross product
        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let nx = v1[1] * v2[2] - v1[2] * v2[1];
        let ny = v1[2] * v2[0] - v1[0] * v2[2];
        let nz = v1[0] * v2[1] - v1[1] * v2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();

        if len < 1e-12 {
            continue; // Degenerate (collinear)
        }

        let normal = [nx / len, ny / len, nz / len];
        let offset = normal[0] * p0[0] + normal[1] * p0[1] + normal[2] * p0[2];

        // Count inliers
        let mut inliers = Vec::new();
        for (idx, p) in points.iter().enumerate() {
            let dist = (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] - offset).abs();
            if dist <= distance_threshold {
                inliers.push(idx);
            }
        }

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_normal = normal;
        }

        // Early exit if we found a very good plane (>60% of points)
        if best_inliers.len() > n * 3 / 5 {
            break;
        }
    }

    if best_inliers.is_empty() {
        return (points.to_vec(), None);
    }

    // Build set of inlier indices for fast lookup
    let mut is_inlier = vec![false; n];
    for &idx in &best_inliers {
        is_inlier[idx] = true;
    }

    let remaining: Vec<[f64; 3]> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_inlier[*i])
        .map(|(_, p)| *p)
        .collect();

    (remaining, Some(best_normal))
}

/// Crop points to an axis-aligned bounding box.
pub fn crop_to_aabb(points: &[[f64; 3]], bounds: &AABB) -> Vec<[f64; 3]> {
    points
        .iter()
        .filter(|p| bounds.contains(p[0], p[1], p[2]))
        .copied()
        .collect()
}

/// Statistical outlier removal using KD-tree for O(n log n) neighbor queries.
///
/// For each point, computes the mean distance to its `k` nearest neighbors.
/// Points whose mean distance exceeds `mean + std_dev_multiplier * std` are removed.
///
/// This matches the PCL `StatisticalOutlierRemoval` algorithm.
pub fn statistical_outlier_removal(
    points: &[[f64; 3]],
    k: usize,
    std_dev_multiplier: f64,
) -> Vec<[f64; 3]> {
    if points.len() <= k {
        return points.to_vec();
    }

    let n = points.len();

    // Build KD-tree from points — O(n log n)
    // Add tiny jitter based on index to avoid kiddo's bucket overflow
    // when many points share the same coordinate on one axis.
    let mut tree: kiddo::KdTree<f64, 3> = kiddo::KdTree::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        let jitter = (i as f64) * 1e-15;
        tree.add(&[p[0] + jitter, p[1] + jitter, p[2] + jitter], i as u64);
    }

    // Query k+1 nearest neighbors for each point (includes self) — O(n * k * log n)
    let mut mean_dists = Vec::with_capacity(n);
    for p in points {
        let neighbors = tree.nearest_n::<kiddo::SquaredEuclidean>(p, k + 1);
        // Skip the first neighbor (self, distance=0), sum distances of the rest
        let knn_sum: f64 = neighbors
            .iter()
            .filter(|nb| nb.distance > 0.0)
            .take(k)
            .map(|nb| nb.distance.sqrt())
            .sum();
        mean_dists.push(knn_sum / k as f64);
    }

    // Compute global mean and std of mean-knn-distances
    let global_mean: f64 = mean_dists.iter().sum::<f64>() / n as f64;
    let variance: f64 = mean_dists
        .iter()
        .map(|d| (d - global_mean) * (d - global_mean))
        .sum::<f64>()
        / n as f64;
    let global_std = variance.sqrt();

    let threshold = global_mean + std_dev_multiplier * global_std;

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| mean_dists[*i] <= threshold)
        .map(|(_, p)| *p)
        .collect()
}

/// Radius-based outlier removal using KD-tree.
///
/// For each point, counts neighbors within `radius`. Points with fewer
/// than `min_neighbors` are considered outliers and removed.
pub fn radius_outlier_removal(
    points: &[[f64; 3]],
    radius: f64,
    min_neighbors: usize,
) -> Vec<[f64; 3]> {
    if points.is_empty() || radius <= 0.0 {
        return points.to_vec();
    }

    let n = points.len();

    // Build KD-tree with tiny jitter to avoid bucket overflow on degenerate data
    let mut tree: kiddo::KdTree<f64, 3> = kiddo::KdTree::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        let jitter = (i as f64) * 1e-15;
        tree.add(&[p[0] + jitter, p[1] + jitter, p[2] + jitter], i as u64);
    }

    let radius_sq = radius * radius;

    points
        .iter()
        .filter(|p| {
            // Count neighbors within radius (excluding self)
            let neighbors = tree.within::<kiddo::SquaredEuclidean>(p, radius_sq);
            let count = neighbors.iter().filter(|nb| nb.distance > 0.0).count();
            count >= min_neighbors
        })
        .copied()
        .collect()
}

/// Uniform random downsampling to a maximum number of points.
///
/// Uses a deterministic stride-based selection to avoid
/// needing a random number generator.
pub fn uniform_downsample(points: &[[f64; 3]], max_points: usize) -> Vec<[f64; 3]> {
    if points.len() <= max_points || max_points == 0 {
        return points.to_vec();
    }

    let stride = points.len() as f64 / max_points as f64;
    let mut result = Vec::with_capacity(max_points);
    let mut idx = 0.0f64;

    while result.len() < max_points {
        let i = idx as usize;
        if i >= points.len() {
            break;
        }
        result.push(points[i]);
        idx += stride;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_downsample_reduces_count() {
        let mut points = Vec::new();
        // Dense grid of points within a small cube
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    points.push([i as f64 * 0.001, j as f64 * 0.001, k as f64 * 0.001]);
                }
            }
        }
        let result = voxel_downsample(&points, 0.005);
        assert!(result.len() < points.len());
        assert!(!result.is_empty());
    }

    #[test]
    fn voxel_downsample_empty() {
        let result = voxel_downsample(&[], 0.01);
        assert!(result.is_empty());
    }

    #[test]
    fn ransac_finds_floor_plane() {
        let mut points = Vec::new();
        // Floor at z=0 (100 points)
        for i in 0..10 {
            for j in 0..10 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.0]);
            }
        }
        // Some objects above floor (20 points)
        for i in 0..4 {
            for j in 0..5 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.5 + i as f64 * 0.01]);
            }
        }

        let (remaining, normal) = ransac_remove_plane(&points, 0.02, 100);

        assert!(normal.is_some());
        let normal = normal.unwrap();
        // Normal should be approximately [0, 0, ±1]
        assert!(normal[2].abs() > 0.9);
        // Most floor points removed
        assert!(remaining.len() < points.len());
        assert!(remaining.len() <= 30); // at most the 20 object points + some noise
    }

    #[test]
    fn ransac_too_few_points() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let (remaining, normal) = ransac_remove_plane(&points, 0.01, 50);
        assert!(normal.is_none());
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn crop_to_aabb_filters() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [2.0, 2.0, 2.0],
        ];
        let bounds = AABB::new(0.0, 0.0, 0.0, 1.5, 1.5, 1.5);
        let result = crop_to_aabb(&points, &bounds);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn statistical_outlier_removal_works() {
        let mut points = Vec::new();
        // Cluster of points
        for i in 0..20 {
            points.push([i as f64 * 0.01, 0.0, 0.0]);
        }
        // Outlier far away
        points.push([100.0, 0.0, 0.0]);

        let result = statistical_outlier_removal(&points, 5, 1.0);
        assert!(result.len() < points.len());
        // The outlier should be removed
        assert!(!result.iter().any(|p| p[0] > 50.0));
    }

    #[test]
    fn uniform_downsample_to_max() {
        let points: Vec<[f64; 3]> = (0..1000).map(|i| [i as f64, 0.0, 0.0]).collect();
        let result = uniform_downsample(&points, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn uniform_downsample_small_input() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = uniform_downsample(&points, 100);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn radius_outlier_removal_works() {
        let mut points = Vec::new();
        // Dense cluster: 20 points within 0.2m of each other
        for i in 0..20 {
            points.push([i as f64 * 0.01, 0.0, 0.0]);
        }
        // Outlier far away
        points.push([100.0, 0.0, 0.0]);

        let result = radius_outlier_removal(&points, 0.25, 3);
        assert!(result.len() < points.len());
        // The far outlier should be removed
        assert!(!result.iter().any(|p| p[0] > 50.0));
        // Most cluster points should remain
        assert!(result.len() >= 15);
    }

    #[test]
    fn radius_outlier_removal_empty() {
        let result = radius_outlier_removal(&[], 1.0, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn radius_outlier_removal_all_isolated() {
        // Points spaced far apart — all should be removed (need at least 2 neighbors)
        let points = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]];
        let result = radius_outlier_removal(&points, 1.0, 2);
        assert!(
            result.is_empty(),
            "Expected empty, got {} points",
            result.len()
        );
    }

    #[test]
    fn statistical_outlier_kdtree_matches_expected() {
        // Tight cluster with one obvious outlier
        let mut points = Vec::new();
        for i in 0..50 {
            points.push([i as f64 * 0.01, 0.0, 0.0]);
        }
        points.push([500.0, 0.0, 0.0]); // extreme outlier

        let result = statistical_outlier_removal(&points, 10, 1.0);
        // Outlier should be removed
        assert!(!result.iter().any(|p| p[0] > 1.0));
        // Most cluster points should remain
        assert!(result.len() >= 40);
    }
}
