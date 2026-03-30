//! SIMD-accelerated collision kernels with runtime dispatch.
//!
//! Provides three tiers of SIMD support:
//! - **AVX2** (x86_64): 4 f64 spheres per cycle (256-bit)
//! - **NEON** (aarch64): 2 f64 spheres per cycle (128-bit, always available)
//! - **Scalar**: Portable fallback for all platforms
//!
//! Runtime detection on x86_64 selects the widest available tier.
//! On aarch64, NEON is always used (it's mandatory on the platform).

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::capt::CollisionPointTree;
use crate::soa::SpheresSoA;

/// SIMD tier detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    /// AVX2: 256-bit, 4 f64 per cycle.
    Avx2,
    /// NEON: 128-bit, 2 f64 per cycle (aarch64).
    Neon,
    /// Scalar fallback.
    Scalar,
}

impl std::fmt::Display for SimdTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimdTier::Avx2 => write!(f, "AVX2 (4×f64)"),
            SimdTier::Neon => write!(f, "NEON (2×f64)"),
            SimdTier::Scalar => write!(f, "Scalar"),
        }
    }
}

/// Detect the best available SIMD tier at runtime.
pub fn detect_simd_tier() -> SimdTier {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return SimdTier::Avx2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return SimdTier::Neon;
    }

    #[allow(unreachable_code)]
    SimdTier::Scalar
}

/// Check if any sphere pair between two sets overlaps.
///
/// Automatically dispatches to the best available SIMD tier.
pub fn any_collision(a: &SpheresSoA, b: &SpheresSoA) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::any_collision_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::any_collision_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar::any_collision_scalar(a, b)
}

/// Compute minimum signed distance between two sphere sets.
///
/// Automatically dispatches to the best available SIMD tier.
pub fn min_distance(a: &SpheresSoA, b: &SpheresSoA) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::min_distance_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::min_distance_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar::min_distance_scalar(a, b)
}

/// Check robot spheres against CAPT tree.
///
/// Returns `true` if any robot sphere collides with obstacles.
/// Automatically dispatches to the best available SIMD tier.
pub fn check_robot_collision(spheres: &SpheresSoA, tree: &CollisionPointTree) -> bool {
    if spheres.is_empty() {
        return false;
    }

    let params = tree.grid_params();
    let clearance = tree.clearance_data();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                avx2::check_spheres_against_grid_avx2(
                    &spheres.x,
                    &spheres.y,
                    &spheres.z,
                    &spheres.radius,
                    clearance,
                    &params,
                )
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            neon::check_spheres_against_grid_neon(
                &spheres.x,
                &spheres.y,
                &spheres.z,
                &spheres.radius,
                clearance,
                &params,
            )
        };
    }

    #[allow(unreachable_code)]
    scalar::check_spheres_against_grid_scalar(
        &spheres.x,
        &spheres.y,
        &spheres.z,
        &spheres.radius,
        clearance,
        &params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capt::AABB;

    #[test]
    fn detect_tier() {
        let tier = detect_simd_tier();
        // Just ensure it doesn't panic and returns a valid tier
        let _ = tier;
    }

    #[test]
    fn dispatch_any_collision_no_overlap() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.1, 0);
        let mut b = SpheresSoA::new();
        b.push(2.0, 0.0, 0.0, 0.1, 0);
        assert!(!any_collision(&a, &b));
    }

    #[test]
    fn dispatch_any_collision_overlap() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);
        let mut b = SpheresSoA::new();
        b.push(0.5, 0.0, 0.0, 0.5, 0);
        assert!(any_collision(&a, &b));
    }

    #[test]
    fn dispatch_min_distance() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.1, 0);
        let mut b = SpheresSoA::new();
        b.push(1.0, 0.0, 0.0, 0.1, 0);

        let d = min_distance(&a, &b);
        assert!((d - 0.8).abs() < 1e-10);
    }

    #[test]
    fn dispatch_min_distance_many_spheres() {
        // Test with enough spheres to exercise SIMD lanes
        let mut a = SpheresSoA::new();
        for i in 0..10 {
            a.push(i as f64 * 0.5, 0.0, 0.0, 0.05, 0);
        }
        let mut b = SpheresSoA::new();
        b.push(1.0, 1.0, 0.0, 0.05, 0);

        let d_dispatch = min_distance(&a, &b);
        let d_scalar = scalar::min_distance_scalar(&a, &b);
        assert!(
            (d_dispatch - d_scalar).abs() < 1e-10,
            "Dispatch {} != scalar {}",
            d_dispatch,
            d_scalar
        );
    }

    #[test]
    fn dispatch_robot_collision_no_obstacles() {
        let tree = CollisionPointTree::empty(0.05, AABB::symmetric(2.0));
        let mut spheres = SpheresSoA::new();
        spheres.push(0.0, 0.0, 0.0, 0.1, 0);
        assert!(!check_robot_collision(&spheres, &tree));
    }

    #[test]
    fn dispatch_robot_collision_with_obstacle() {
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.5, 0);
        let tree = CollisionPointTree::build(&obstacles, 0.05, AABB::symmetric(2.0));

        let mut spheres = SpheresSoA::new();
        spheres.push(0.0, 0.0, 0.0, 0.1, 0); // inside obstacle
        assert!(check_robot_collision(&spheres, &tree));

        let mut spheres_far = SpheresSoA::new();
        spheres_far.push(1.5, 0.0, 0.0, 0.1, 0); // far from obstacle
        assert!(!check_robot_collision(&spheres_far, &tree));
    }

    #[test]
    fn dispatch_empty_spheres() {
        let tree = CollisionPointTree::empty(0.05, AABB::symmetric(1.0));
        let spheres = SpheresSoA::new();
        assert!(!check_robot_collision(&spheres, &tree));
    }

    #[test]
    fn simd_matches_scalar_comprehensive() {
        // Comprehensive test: many spheres, verify SIMD dispatch matches scalar
        let mut a = SpheresSoA::new();
        for i in 0..17 {
            // 17 = not divisible by 4 or 2, tests remainder paths
            let angle = i as f64 * 0.4;
            a.push(angle.cos(), angle.sin(), i as f64 * 0.1, 0.05, 0);
        }

        let mut b = SpheresSoA::new();
        for j in 0..5 {
            b.push(j as f64 * 0.3, 0.5, 0.0, 0.1, 0);
        }

        let d_dispatch = min_distance(&a, &b);
        let d_scalar = scalar::min_distance_scalar(&a, &b);
        assert!(
            (d_dispatch - d_scalar).abs() < 1e-10,
            "SIMD dispatch ({}) != scalar ({})",
            d_dispatch,
            d_scalar
        );

        let c_dispatch = any_collision(&a, &b);
        let c_scalar = scalar::any_collision_scalar(&a, &b);
        assert_eq!(c_dispatch, c_scalar);
    }

    // ─── New SIMD edge case tests ───

    /// Test non-4-aligned sphere counts (1, 2, 3, 5, 7, 13) —
    /// exercises AVX2 remainder loop (len%4 != 0) and NEON remainder (len%2 != 0).
    #[test]
    fn simd_non_aligned_sphere_counts() {
        let counts = [1, 2, 3, 5, 7, 13];

        for &count in &counts {
            let mut a = SpheresSoA::new();
            for i in 0..count {
                a.push(i as f64 * 0.5, 0.0, 0.0, 0.05, 0);
            }

            let mut b = SpheresSoA::new();
            b.push(1.0, 1.0, 0.0, 0.1, 0);

            let d_dispatch = min_distance(&a, &b);
            let d_scalar = scalar::min_distance_scalar(&a, &b);
            assert!(
                (d_dispatch - d_scalar).abs() < 1e-10,
                "count={}: dispatch ({}) != scalar ({})",
                count,
                d_dispatch,
                d_scalar
            );

            let c_dispatch = any_collision(&a, &b);
            let c_scalar = scalar::any_collision_scalar(&a, &b);
            assert_eq!(c_dispatch, c_scalar, "count={}: collision mismatch", count);
        }
    }

    /// Test with exactly 4 spheres (perfect AVX2 lane alignment).
    #[test]
    fn simd_exact_4_alignment() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.1, 0);
        a.push(1.0, 0.0, 0.0, 0.1, 1);
        a.push(2.0, 0.0, 0.0, 0.1, 2);
        a.push(3.0, 0.0, 0.0, 0.1, 3);

        let mut b = SpheresSoA::new();
        b.push(1.5, 0.0, 0.0, 0.1, 0);

        let d_dispatch = min_distance(&a, &b);
        let d_scalar = scalar::min_distance_scalar(&a, &b);
        assert!(
            (d_dispatch - d_scalar).abs() < 1e-10,
            "4-aligned: dispatch ({}) != scalar ({})",
            d_dispatch,
            d_scalar
        );
    }

    /// Test with 8 spheres (2 full AVX2 iterations, no remainder).
    #[test]
    fn simd_exact_8_alignment() {
        let mut a = SpheresSoA::new();
        for i in 0..8 {
            a.push(i as f64 * 0.3, 0.0, 0.0, 0.05, i);
        }

        let mut b = SpheresSoA::new();
        b.push(0.5, 0.5, 0.0, 0.1, 0);

        let d_dispatch = min_distance(&a, &b);
        let d_scalar = scalar::min_distance_scalar(&a, &b);
        assert!(
            (d_dispatch - d_scalar).abs() < 1e-10,
            "8-aligned: dispatch ({}) != scalar ({})",
            d_dispatch,
            d_scalar
        );
    }

    /// Scalar fallback produces correct results (exercise scalar path directly).
    #[test]
    fn scalar_fallback_correctness() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.3, 0);
        a.push(1.0, 0.0, 0.0, 0.3, 1);
        a.push(2.0, 0.0, 0.0, 0.3, 2);

        let mut b = SpheresSoA::new();
        b.push(0.3, 0.0, 0.0, 0.3, 0); // dist=0.3, threshold=0.6 → overlaps
        b.push(1.3, 0.0, 0.0, 0.3, 1); // dist=0.3, threshold=0.6 → overlaps

        // Scalar: any_collision should be true (spheres overlap)
        assert!(scalar::any_collision_scalar(&a, &b));

        // Scalar: min_distance should be negative
        let d = scalar::min_distance_scalar(&a, &b);
        assert!(
            d < 0.0,
            "overlapping spheres should have negative distance: {d}"
        );

        // Non-overlapping
        let mut c = SpheresSoA::new();
        c.push(10.0, 10.0, 10.0, 0.1, 0);
        assert!(!scalar::any_collision_scalar(&a, &c));
        let d2 = scalar::min_distance_scalar(&a, &c);
        assert!(
            d2 > 0.0,
            "separated spheres should have positive distance: {d2}"
        );
    }

    /// Test with very large sphere count (10k) — verify no panic or precision issues.
    #[test]
    fn simd_large_sphere_count() {
        let mut a = SpheresSoA::new();
        for i in 0..10_000 {
            let x = (i as f64 * 0.01) % 100.0;
            let y = (i as f64 * 0.007) % 50.0;
            let z = (i as f64 * 0.003) % 20.0;
            a.push(x, y, z, 0.001, i % 20);
        }

        let mut b = SpheresSoA::new();
        b.push(50.0, 25.0, 10.0, 0.01, 0);

        // Should not panic
        let d_dispatch = min_distance(&a, &b);
        let c_dispatch = any_collision(&a, &b);

        // Verify against scalar
        let d_scalar = scalar::min_distance_scalar(&a, &b);
        let c_scalar = scalar::any_collision_scalar(&a, &b);

        assert!(
            (d_dispatch - d_scalar).abs() < 1e-8,
            "10k spheres: dispatch ({}) != scalar ({})",
            d_dispatch,
            d_scalar
        );
        assert_eq!(c_dispatch, c_scalar, "10k spheres collision mismatch");
    }

    /// Test with NaN coordinates — NaN comparison is always false, so
    /// min_distance returns Inf (NaN doesn't beat the initial best=Inf).
    /// any_collision returns false (NaN dist_sq < threshold^2 is false).
    /// This documents the correct behavior: NaN spheres are effectively ignored.
    #[test]
    fn simd_nan_behavior() {
        let mut a = SpheresSoA::new();
        a.push(f64::NAN, 0.0, 0.0, 0.1, 0);

        let mut b = SpheresSoA::new();
        b.push(0.0, 0.0, 0.0, 0.1, 0);

        // NaN comparisons are always false, so NaN distance never beats Inf
        let d = scalar::min_distance_scalar(&a, &b);
        assert!(
            d.is_nan() || d.is_infinite(),
            "NaN input should produce NaN or Inf, got {d}"
        );

        // NaN dist_sq < threshold^2 is false → no collision detected
        assert!(
            !scalar::any_collision_scalar(&a, &b),
            "NaN sphere should not report collision"
        );
    }

    /// Test with Inf coordinates.
    #[test]
    fn simd_inf_coordinates() {
        let mut a = SpheresSoA::new();
        a.push(f64::INFINITY, 0.0, 0.0, 0.1, 0);

        let mut b = SpheresSoA::new();
        b.push(0.0, 0.0, 0.0, 0.1, 0);

        // Inf - 0.0 = Inf, sqrt(Inf) = Inf, Inf - 0.2 = Inf
        let d = scalar::min_distance_scalar(&a, &b);
        assert!(
            d.is_infinite(),
            "Inf input should produce Inf distance, got {d}"
        );
    }

    /// Empty sets: any_collision should return false, min_distance should return Inf.
    #[test]
    fn simd_empty_sets() {
        let empty = SpheresSoA::new();
        let mut one = SpheresSoA::new();
        one.push(0.0, 0.0, 0.0, 0.1, 0);

        assert!(!any_collision(&empty, &one));
        assert!(!any_collision(&one, &empty));
        assert!(!any_collision(&empty, &empty));

        assert!(min_distance(&empty, &one).is_infinite());
        assert!(min_distance(&one, &empty).is_infinite());
        assert!(min_distance(&empty, &empty).is_infinite());
    }

    /// SIMD tier display doesn't panic.
    #[test]
    fn simd_tier_display() {
        let tier = detect_simd_tier();
        let s = format!("{tier}");
        assert!(!s.is_empty());

        // Test all variants
        assert_eq!(format!("{}", SimdTier::Avx2), "AVX2 (4×f64)");
        assert_eq!(format!("{}", SimdTier::Neon), "NEON (2×f64)");
        assert_eq!(format!("{}", SimdTier::Scalar), "Scalar");
    }
}
