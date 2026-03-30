//! Scalar (non-SIMD) reference implementation.
//!
//! Used as fallback on platforms without SIMD support, and as the
//! reference implementation for verifying SIMD correctness.

use crate::capt::{CollisionPointTree, GridParams};
use crate::soa::SpheresSoA;

/// Check if any robot sphere collides with the CAPT tree.
///
/// Returns `true` if collision detected.
pub fn check_robot_collision_scalar(spheres: &SpheresSoA, tree: &CollisionPointTree) -> bool {
    for i in 0..spheres.len() {
        if tree.is_collision(spheres.x[i], spheres.y[i], spheres.z[i], spheres.radius[i]) {
            return true;
        }
    }
    false
}

/// Compute minimum signed distance between two sphere sets.
///
/// Returns the smallest gap between any pair of spheres.
/// Negative values indicate penetration.
/// Returns f64::INFINITY if either set is empty.
pub fn min_distance_scalar(a: &SpheresSoA, b: &SpheresSoA) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let mut best = f64::INFINITY;

    for i in 0..a.len() {
        let ax = a.x[i];
        let ay = a.y[i];
        let az = a.z[i];
        let ar = a.radius[i];

        for j in 0..b.len() {
            let dx = ax - b.x[j];
            let dy = ay - b.y[j];
            let dz = az - b.z[j];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() - ar - b.radius[j];
            if dist < best {
                best = dist;
            }
        }
    }

    best
}

/// Check if any sphere pair between two sets overlaps.
///
/// Returns `true` if any pair has negative signed distance.
pub fn any_collision_scalar(a: &SpheresSoA, b: &SpheresSoA) -> bool {
    for i in 0..a.len() {
        let ax = a.x[i];
        let ay = a.y[i];
        let az = a.z[i];
        let ar = a.radius[i];

        for j in 0..b.len() {
            let dx = ax - b.x[j];
            let dy = ay - b.y[j];
            let dz = az - b.z[j];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let threshold = ar + b.radius[j];
            if dist_sq < threshold * threshold {
                return true;
            }
        }
    }

    false
}

/// Batch check: for each sphere in `spheres`, check collision against CAPT tree.
///
/// Returns a bitmask-like vec of bools (one per sphere).
pub fn batch_check_scalar(spheres: &SpheresSoA, tree: &CollisionPointTree) -> Vec<bool> {
    (0..spheres.len())
        .map(|i| tree.is_collision(spheres.x[i], spheres.y[i], spheres.z[i], spheres.radius[i]))
        .collect()
}

/// Check robot collision using grid params directly (for SIMD-matching signature).
///
/// This takes the same parameters that SIMD kernels would take.
pub fn check_spheres_against_grid_scalar(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    r: &[f64],
    clearance: &[f64],
    params: &GridParams,
) -> bool {
    let n = x.len();
    for i in 0..n {
        let fx = (x[i] - params.min_x) * params.inv_resolution;
        let fy = (y[i] - params.min_y) * params.inv_resolution;
        let fz = (z[i] - params.min_z) * params.inv_resolution;

        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            return true; // out of bounds = collision
        }

        let ix = fx as usize;
        let iy = fy as usize;
        let iz = fz as usize;

        if ix >= params.nx || iy >= params.ny || iz >= params.nz {
            return true; // out of bounds
        }

        let idx = ix * params.stride_x + iy * params.stride_y + iz;
        if clearance[idx] < r[i] {
            return true; // collision
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_distance_empty() {
        let a = SpheresSoA::new();
        let b = SpheresSoA::new();
        assert!(min_distance_scalar(&a, &b).is_infinite());
    }

    #[test]
    fn min_distance_separated() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.1, 0);
        let mut b = SpheresSoA::new();
        b.push(1.0, 0.0, 0.0, 0.1, 0);

        let d = min_distance_scalar(&a, &b);
        assert!((d - 0.8).abs() < 1e-10);
    }

    #[test]
    fn min_distance_overlapping() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);
        let mut b = SpheresSoA::new();
        b.push(0.6, 0.0, 0.0, 0.5, 0);

        let d = min_distance_scalar(&a, &b);
        assert!(d < 0.0, "Expected negative distance, got {}", d);
        assert!((d - (-0.4)).abs() < 1e-10);
    }

    #[test]
    fn any_collision_true() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);
        let mut b = SpheresSoA::new();
        b.push(0.6, 0.0, 0.0, 0.5, 0);
        assert!(any_collision_scalar(&a, &b));
    }

    #[test]
    fn any_collision_false() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.1, 0);
        let mut b = SpheresSoA::new();
        b.push(1.0, 0.0, 0.0, 0.1, 0);
        assert!(!any_collision_scalar(&a, &b));
    }
}
