//! AVX2 SIMD collision kernels (x86_64).
//!
//! Processes 4 f64 sphere coordinates simultaneously using 256-bit AVX2
//! instructions. Provides ~4x throughput over scalar for sphere-sphere
//! distance computations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::capt::GridParams;
use crate::soa::SpheresSoA;

/// Check if any sphere pair between two sets overlaps, using AVX2.
///
/// Processes 4 spheres from `a` simultaneously against each sphere in `b`.
/// Falls through to scalar for the remaining 0-3 spheres.
///
/// # Safety
/// Caller must verify AVX2 is available via `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn any_collision_avx2(a: &SpheresSoA, b: &SpheresSoA) -> bool {
    let na = a.len();
    let nb = b.len();

    if na == 0 || nb == 0 {
        return false;
    }

    // Process 4 spheres from `a` at a time
    let chunks = na / 4;
    let remainder = na % 4;

    for j in 0..nb {
        let bx = _mm256_set1_pd(b.x[j]);
        let by = _mm256_set1_pd(b.y[j]);
        let bz = _mm256_set1_pd(b.z[j]);
        let br = _mm256_set1_pd(b.radius[j]);

        for chunk in 0..chunks {
            let base = chunk * 4;

            let ax = _mm256_loadu_pd(a.x.as_ptr().add(base));
            let ay = _mm256_loadu_pd(a.y.as_ptr().add(base));
            let az = _mm256_loadu_pd(a.z.as_ptr().add(base));
            let ar = _mm256_loadu_pd(a.radius.as_ptr().add(base));

            let dx = _mm256_sub_pd(ax, bx);
            let dy = _mm256_sub_pd(ay, by);
            let dz = _mm256_sub_pd(az, bz);

            // dist_sq = dx*dx + dy*dy + dz*dz
            let dist_sq = _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            );

            // threshold = (ar + br)^2
            let sum_r = _mm256_add_pd(ar, br);
            let threshold_sq = _mm256_mul_pd(sum_r, sum_r);

            // collision = dist_sq < threshold_sq
            let cmp = _mm256_cmp_pd(dist_sq, threshold_sq, _CMP_LT_OQ);
            let mask = _mm256_movemask_pd(cmp);

            if mask != 0 {
                return true;
            }
        }

        // Handle remainder with scalar
        let base = chunks * 4;
        for i in 0..remainder {
            let idx = base + i;
            let dx = a.x[idx] - b.x[j];
            let dy = a.y[idx] - b.y[j];
            let dz = a.z[idx] - b.z[j];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let threshold = a.radius[idx] + b.radius[j];
            if dist_sq < threshold * threshold {
                return true;
            }
        }
    }

    false
}

/// Compute minimum signed distance between two sphere sets using AVX2.
///
/// Processes 4 spheres from `a` per cycle. For each chunk, computes
/// the center-to-center distance squared and tracks the minimum.
///
/// # Safety
/// Caller must verify AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn min_distance_avx2(a: &SpheresSoA, b: &SpheresSoA) -> f64 {
    let na = a.len();
    let nb = b.len();

    if na == 0 || nb == 0 {
        return f64::INFINITY;
    }

    let mut best = f64::INFINITY;
    let chunks = na / 4;
    let remainder = na % 4;

    for j in 0..nb {
        let bx = _mm256_set1_pd(b.x[j]);
        let by = _mm256_set1_pd(b.y[j]);
        let bz = _mm256_set1_pd(b.z[j]);
        let br_val = b.radius[j];

        for chunk in 0..chunks {
            let base = chunk * 4;

            let ax = _mm256_loadu_pd(a.x.as_ptr().add(base));
            let ay = _mm256_loadu_pd(a.y.as_ptr().add(base));
            let az = _mm256_loadu_pd(a.z.as_ptr().add(base));

            let dx = _mm256_sub_pd(ax, bx);
            let dy = _mm256_sub_pd(ay, by);
            let dz = _mm256_sub_pd(az, bz);

            let dist_sq = _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            );

            // Extract 4 squared distances and compute actual distances
            let mut dist_arr = [0.0f64; 4];
            _mm256_storeu_pd(dist_arr.as_mut_ptr(), dist_sq);

            #[allow(clippy::needless_range_loop)]
            for k in 0..4 {
                let dist = dist_arr[k].sqrt() - a.radius[base + k] - br_val;
                if dist < best {
                    best = dist;
                }
            }
        }

        // Remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let idx = base + i;
            let dx = a.x[idx] - b.x[j];
            let dy = a.y[idx] - b.y[j];
            let dz = a.z[idx] - b.z[j];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() - a.radius[idx] - br_val;
            if dist < best {
                best = dist;
            }
        }
    }

    best
}

/// Check robot spheres against CAPT grid using AVX2.
///
/// Processes 4 spheres per iteration: computes grid indices,
/// gathers clearance values, and compares against radii.
///
/// # Safety
/// Caller must verify AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn check_spheres_against_grid_avx2(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    r: &[f64],
    clearance: &[f64],
    params: &GridParams,
) -> bool {
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let min_x = _mm256_set1_pd(params.min_x);
    let min_y = _mm256_set1_pd(params.min_y);
    let min_z = _mm256_set1_pd(params.min_z);
    let inv_res = _mm256_set1_pd(params.inv_resolution);
    let zero = _mm256_setzero_pd();

    for chunk in 0..chunks {
        let base = chunk * 4;

        let vx = _mm256_loadu_pd(x.as_ptr().add(base));
        let vy = _mm256_loadu_pd(y.as_ptr().add(base));
        let vz = _mm256_loadu_pd(z.as_ptr().add(base));
        let vr = _mm256_loadu_pd(r.as_ptr().add(base));

        // Compute fractional grid coordinates
        let fx = _mm256_mul_pd(_mm256_sub_pd(vx, min_x), inv_res);
        let fy = _mm256_mul_pd(_mm256_sub_pd(vy, min_y), inv_res);
        let fz = _mm256_mul_pd(_mm256_sub_pd(vz, min_z), inv_res);

        // Check bounds: any negative → out of bounds
        let oob_x_lo = _mm256_cmp_pd(fx, zero, _CMP_LT_OQ);
        let oob_y_lo = _mm256_cmp_pd(fy, zero, _CMP_LT_OQ);
        let oob_z_lo = _mm256_cmp_pd(fz, zero, _CMP_LT_OQ);
        let oob_lo = _mm256_or_pd(_mm256_or_pd(oob_x_lo, oob_y_lo), oob_z_lo);

        if _mm256_movemask_pd(oob_lo) != 0 {
            return true; // At least one sphere out of bounds
        }

        // Extract indices and check bounds + gather clearance (scalar, since gather is complex)
        let mut fx_arr = [0.0f64; 4];
        let mut fy_arr = [0.0f64; 4];
        let mut fz_arr = [0.0f64; 4];
        let mut r_arr = [0.0f64; 4];
        _mm256_storeu_pd(fx_arr.as_mut_ptr(), fx);
        _mm256_storeu_pd(fy_arr.as_mut_ptr(), fy);
        _mm256_storeu_pd(fz_arr.as_mut_ptr(), fz);
        _mm256_storeu_pd(r_arr.as_mut_ptr(), vr);

        for k in 0..4 {
            let ix = fx_arr[k] as usize;
            let iy = fy_arr[k] as usize;
            let iz = fz_arr[k] as usize;

            if ix >= params.nx || iy >= params.ny || iz >= params.nz {
                return true;
            }

            let idx = ix * params.stride_x + iy * params.stride_y + iz;
            if clearance[idx] < r_arr[k] {
                return true;
            }
        }
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let fx = (x[idx] - params.min_x) * params.inv_resolution;
        let fy = (y[idx] - params.min_y) * params.inv_resolution;
        let fz = (z[idx] - params.min_z) * params.inv_resolution;

        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            return true;
        }

        let ix = fx as usize;
        let iy = fy as usize;
        let iz = fz as usize;

        if ix >= params.nx || iy >= params.ny || iz >= params.nz {
            return true;
        }

        let grid_idx = ix * params.stride_x + iy * params.stride_y + iz;
        if clearance[grid_idx] < r[idx] {
            return true;
        }
    }

    false
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::simd::scalar;

    #[test]
    fn avx2_collision_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut a = SpheresSoA::new();
        for i in 0..10 {
            a.push(i as f64 * 0.3, 0.0, 0.0, 0.1, 0);
        }

        let mut b = SpheresSoA::new();
        b.push(0.45, 0.0, 0.0, 0.2, 0);
        b.push(5.0, 0.0, 0.0, 0.1, 0);

        let scalar_result = scalar::any_collision_scalar(&a, &b);
        let avx2_result = unsafe { any_collision_avx2(&a, &b) };
        assert_eq!(
            scalar_result, avx2_result,
            "AVX2 collision result differs from scalar"
        );
    }

    #[test]
    fn avx2_min_distance_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut a = SpheresSoA::new();
        for i in 0..7 {
            a.push(i as f64 * 0.5, 0.1 * i as f64, 0.0, 0.05, 0);
        }

        let mut b = SpheresSoA::new();
        b.push(1.0, 0.0, 0.0, 0.1, 0);
        b.push(2.0, 1.0, 0.0, 0.2, 0);

        let scalar_d = scalar::min_distance_scalar(&a, &b);
        let avx2_d = unsafe { min_distance_avx2(&a, &b) };
        assert!(
            (scalar_d - avx2_d).abs() < 1e-10,
            "AVX2 distance {} differs from scalar {}",
            avx2_d,
            scalar_d
        );
    }
}
