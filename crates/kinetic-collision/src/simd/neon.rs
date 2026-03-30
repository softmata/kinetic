//! ARM NEON SIMD collision kernels (aarch64).
//!
//! Processes 2 f64 sphere coordinates simultaneously using 128-bit NEON
//! instructions. Available on all aarch64 platforms (Apple Silicon, ARM servers).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::capt::GridParams;
use crate::soa::SpheresSoA;

/// Check if any sphere pair between two sets overlaps, using NEON.
///
/// Processes 2 spheres from `a` simultaneously against each sphere in `b`.
///
/// # Safety
/// Only callable on aarch64 targets (NEON is always available on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn any_collision_neon(a: &SpheresSoA, b: &SpheresSoA) -> bool {
    let na = a.len();
    let nb = b.len();

    if na == 0 || nb == 0 {
        return false;
    }

    let chunks = na / 2;
    let remainder = na % 2;

    for j in 0..nb {
        let bx = vdupq_n_f64(b.x[j]);
        let by = vdupq_n_f64(b.y[j]);
        let bz = vdupq_n_f64(b.z[j]);
        let br = vdupq_n_f64(b.radius[j]);

        for chunk in 0..chunks {
            let base = chunk * 2;

            let ax = vld1q_f64(a.x.as_ptr().add(base));
            let ay = vld1q_f64(a.y.as_ptr().add(base));
            let az = vld1q_f64(a.z.as_ptr().add(base));
            let ar = vld1q_f64(a.radius.as_ptr().add(base));

            let dx = vsubq_f64(ax, bx);
            let dy = vsubq_f64(ay, by);
            let dz = vsubq_f64(az, bz);

            // dist_sq = dx*dx + dy*dy + dz*dz
            let dist_sq = vaddq_f64(
                vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)),
                vmulq_f64(dz, dz),
            );

            // threshold = (ar + br)^2
            let sum_r = vaddq_f64(ar, br);
            let threshold_sq = vmulq_f64(sum_r, sum_r);

            // collision = dist_sq < threshold_sq
            let cmp = vcltq_f64(dist_sq, threshold_sq);

            // Check if any lane has a collision
            if vmaxvq_u64(cmp) != 0 {
                return true;
            }
        }

        // Handle remainder
        let base = chunks * 2;
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

/// Compute minimum signed distance between two sphere sets using NEON.
///
/// Processes 2 spheres from `a` per cycle.
///
/// # Safety
/// Only callable on aarch64 targets.
#[cfg(target_arch = "aarch64")]
pub unsafe fn min_distance_neon(a: &SpheresSoA, b: &SpheresSoA) -> f64 {
    let na = a.len();
    let nb = b.len();

    if na == 0 || nb == 0 {
        return f64::INFINITY;
    }

    let mut best = f64::INFINITY;
    let chunks = na / 2;
    let remainder = na % 2;

    for j in 0..nb {
        let bx = vdupq_n_f64(b.x[j]);
        let by = vdupq_n_f64(b.y[j]);
        let bz = vdupq_n_f64(b.z[j]);
        let br_val = b.radius[j];

        for chunk in 0..chunks {
            let base = chunk * 2;

            let ax = vld1q_f64(a.x.as_ptr().add(base));
            let ay = vld1q_f64(a.y.as_ptr().add(base));
            let az = vld1q_f64(a.z.as_ptr().add(base));

            let dx = vsubq_f64(ax, bx);
            let dy = vsubq_f64(ay, by);
            let dz = vsubq_f64(az, bz);

            let dist_sq = vaddq_f64(
                vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)),
                vmulq_f64(dz, dz),
            );

            // Extract and compute actual distances
            let mut dist_arr = [0.0f64; 2];
            vst1q_f64(dist_arr.as_mut_ptr(), dist_sq);

            for k in 0..2 {
                let dist = dist_arr[k].sqrt() - a.radius[base + k] - br_val;
                if dist < best {
                    best = dist;
                }
            }
        }

        // Remainder
        let base = chunks * 2;
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

/// Check robot spheres against CAPT grid using NEON.
///
/// Processes 2 spheres per iteration.
///
/// # Safety
/// Only callable on aarch64 targets.
#[cfg(target_arch = "aarch64")]
pub unsafe fn check_spheres_against_grid_neon(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    r: &[f64],
    clearance: &[f64],
    params: &GridParams,
) -> bool {
    let n = x.len();
    let chunks = n / 2;
    let remainder = n % 2;

    let min_x = vdupq_n_f64(params.min_x);
    let min_y = vdupq_n_f64(params.min_y);
    let min_z = vdupq_n_f64(params.min_z);
    let inv_res = vdupq_n_f64(params.inv_resolution);
    let zero = vdupq_n_f64(0.0);

    for chunk in 0..chunks {
        let base = chunk * 2;

        let vx = vld1q_f64(x.as_ptr().add(base));
        let vy = vld1q_f64(y.as_ptr().add(base));
        let vz = vld1q_f64(z.as_ptr().add(base));

        // Compute fractional grid coordinates
        let fx = vmulq_f64(vsubq_f64(vx, min_x), inv_res);
        let fy = vmulq_f64(vsubq_f64(vy, min_y), inv_res);
        let fz = vmulq_f64(vsubq_f64(vz, min_z), inv_res);

        // Check for negative (out of bounds)
        let oob_x = vcltq_f64(fx, zero);
        let oob_y = vcltq_f64(fy, zero);
        let oob_z = vcltq_f64(fz, zero);
        let oob = vorrq_u64(vorrq_u64(oob_x, oob_y), oob_z);

        if vmaxvq_u64(oob) != 0 {
            return true;
        }

        // Extract and do scalar index computation + clearance lookup
        let mut fx_arr = [0.0f64; 2];
        let mut fy_arr = [0.0f64; 2];
        let mut fz_arr = [0.0f64; 2];
        vst1q_f64(fx_arr.as_mut_ptr(), fx);
        vst1q_f64(fy_arr.as_mut_ptr(), fy);
        vst1q_f64(fz_arr.as_mut_ptr(), fz);

        for k in 0..2 {
            let ix = fx_arr[k] as usize;
            let iy = fy_arr[k] as usize;
            let iz = fz_arr[k] as usize;

            if ix >= params.nx || iy >= params.ny || iz >= params.nz {
                return true;
            }

            let idx = ix * params.stride_x + iy * params.stride_y + iz;
            if clearance[idx] < r[base + k] {
                return true;
            }
        }
    }

    // Remainder
    let base = chunks * 2;
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
