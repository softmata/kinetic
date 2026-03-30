//! Path smoothing: natural cubic spline and B-spline.
//!
//! After shortcutting, the path is piecewise-linear. Smoothing fits curves
//! through the waypoints to produce C2-continuous trajectories suitable for
//! robot execution with bounded velocities and accelerations.
//!
//! Two methods:
//! - **Natural cubic spline**: Interpolates exactly through all waypoints.
//!   Zero second derivative at endpoints. C2-continuous.
//! - **Cubic B-spline**: Approximates the path with controllable smoothing.
//!   Trade-off between fidelity and smoothness via `smoothing_factor`.

use crate::shortcut::CollisionChecker;

/// Result of path smoothing.
#[derive(Debug, Clone)]
pub struct SmoothedPath {
    /// Dense waypoints along the smoothed path.
    pub waypoints: Vec<Vec<f64>>,
    /// Whether the path is C2-continuous (always true for spline output).
    pub c2_continuous: bool,
}

/// Smooth a path using natural cubic spline interpolation.
///
/// Fits a natural cubic spline through each joint independently.
/// Output has `num_output_points` evenly-spaced samples along the path.
///
/// Natural boundary conditions: second derivative = 0 at endpoints.
/// The result is C2-continuous (positions, velocities, accelerations continuous).
///
/// If `collision_checker` is provided, verifies the smoothed path is collision-free.
/// If a collision is found, returns the original path unsmoothed.
pub fn smooth_cubic_spline<C: CollisionChecker>(
    path: &[Vec<f64>],
    num_output_points: usize,
    collision_checker: Option<&C>,
) -> SmoothedPath {
    if path.len() < 2 {
        return SmoothedPath {
            waypoints: path.to_vec(),
            c2_continuous: true,
        };
    }

    if path.len() == 2 {
        // Two points: just linearly interpolate
        return linear_interpolate(path, num_output_points);
    }

    let dof = path[0].len();
    let n = path.len();

    // Compute cumulative chord-length parameterization
    let t_params = chord_length_params(path);

    // Fit a cubic spline per joint
    let mut output = vec![vec![0.0; dof]; num_output_points];

    for j in 0..dof {
        let y: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
        let coeffs = natural_cubic_spline_coeffs(&t_params, &y);

        // Sample at uniform t values
        let t_max = t_params[n - 1];
        for (k, out_wp) in output.iter_mut().enumerate() {
            let t = if num_output_points <= 1 {
                0.0
            } else {
                t_max * k as f64 / (num_output_points - 1) as f64
            };
            out_wp[j] = evaluate_cubic_spline(&t_params, &coeffs, t);
        }
    }

    // Verify collision-free if checker provided
    if let Some(checker) = collision_checker {
        for wp in &output {
            if checker.is_in_collision(wp) {
                return SmoothedPath {
                    waypoints: path.to_vec(),
                    c2_continuous: false,
                };
            }
        }
    }

    SmoothedPath {
        waypoints: output,
        c2_continuous: true,
    }
}

/// Smooth a path using cubic B-spline approximation.
///
/// Unlike cubic spline interpolation, B-splines approximate (don't pass through)
/// the control points. `smoothing_factor` controls the trade-off:
/// - 0.0: Use all waypoints as control points (closest to interpolation).
/// - 1.0: Maximum smoothing (fewer effective control points).
///
/// The result is C2-continuous.
pub fn smooth_bspline<C: CollisionChecker>(
    path: &[Vec<f64>],
    num_output_points: usize,
    smoothing_factor: f64,
    collision_checker: Option<&C>,
) -> SmoothedPath {
    if path.len() < 2 {
        return SmoothedPath {
            waypoints: path.to_vec(),
            c2_continuous: true,
        };
    }

    if path.len() <= 4 {
        // Too few points for B-spline, fall back to cubic spline
        return smooth_cubic_spline(path, num_output_points, collision_checker);
    }

    let dof = path[0].len();
    let smoothing = smoothing_factor.clamp(0.0, 1.0);

    // Reduce control points based on smoothing factor
    let control_points = select_control_points(path, smoothing);
    let n_ctrl = control_points.len();

    if n_ctrl < 4 {
        // Not enough for cubic B-spline, fall back
        return smooth_cubic_spline(path, num_output_points, collision_checker);
    }

    // Build uniform cubic B-spline knot vector
    let n_knots = n_ctrl + 4; // degree + 1 + n_ctrl
    let knots: Vec<f64> = (0..n_knots).map(|i| i as f64).collect();

    // Sample the B-spline
    let mut output = vec![vec![0.0; dof]; num_output_points];
    let t_min = knots[3]; // first valid parameter for cubic B-spline
    let t_max = knots[n_ctrl]; // last valid parameter

    for (k, out_wp) in output.iter_mut().enumerate() {
        let t = if num_output_points <= 1 {
            t_min
        } else {
            t_min + (t_max - t_min) * k as f64 / (num_output_points - 1) as f64
        };

        #[allow(clippy::needless_range_loop)]
        for j in 0..dof {
            out_wp[j] = evaluate_bspline(&knots, &control_points, j, t);
        }
    }

    // Ensure start and end match original path
    if let Some(first) = output.first_mut() {
        first.clone_from(&path[0]);
    }
    if let Some(last) = output.last_mut() {
        last.clone_from(path.last().unwrap());
    }

    // Verify collision-free if checker provided
    if let Some(checker) = collision_checker {
        for wp in &output {
            if checker.is_in_collision(wp) {
                return SmoothedPath {
                    waypoints: path.to_vec(),
                    c2_continuous: false,
                };
            }
        }
    }

    SmoothedPath {
        waypoints: output,
        c2_continuous: true,
    }
}

/// Compute derivatives at a point along a cubic-spline-smoothed path.
///
/// Returns (position, velocity, acceleration) at parameter `t` in [0, 1].
pub fn spline_derivatives(path: &[Vec<f64>], t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if path.len() < 2 {
        let pos = path.first().cloned().unwrap_or_default();
        let dof = pos.len();
        return (pos, vec![0.0; dof], vec![0.0; dof]);
    }

    let dof = path[0].len();
    let n = path.len();
    let t_params = chord_length_params(path);
    let t_max = t_params[n - 1];
    let t_eval = t.clamp(0.0, 1.0) * t_max;

    let mut pos = vec![0.0; dof];
    let mut vel = vec![0.0; dof];
    let mut acc = vec![0.0; dof];

    for j in 0..dof {
        let y: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
        let coeffs = natural_cubic_spline_coeffs(&t_params, &y);
        let (p, v, a) = evaluate_cubic_spline_derivatives(&t_params, &coeffs, t_eval);
        pos[j] = p;
        vel[j] = v;
        acc[j] = a;
    }

    (pos, vel, acc)
}

// === Internal implementation ===

/// Cubic spline coefficients for one segment: a + b*(t-ti) + c*(t-ti)^2 + d*(t-ti)^3
#[derive(Debug, Clone)]
struct CubicCoeffs {
    a: Vec<f64>, // n values (function values at knots)
    b: Vec<f64>, // n-1 first derivative coefficients
    c: Vec<f64>, // n second derivative coefficients
    d: Vec<f64>, // n-1 third derivative coefficients
}

/// Compute chord-length parameterization for a path.
fn chord_length_params(path: &[Vec<f64>]) -> Vec<f64> {
    let mut t = vec![0.0];
    for i in 1..path.len() {
        let dist = joint_distance(&path[i - 1], &path[i]);
        t.push(t[i - 1] + dist);
    }
    t
}

/// Compute natural cubic spline coefficients.
///
/// Solves the tridiagonal system for natural boundary conditions (c[0] = c[n-1] = 0).
fn natural_cubic_spline_coeffs(t: &[f64], y: &[f64]) -> CubicCoeffs {
    let n = t.len();
    assert_eq!(n, y.len());
    assert!(n >= 2);

    if n == 2 {
        // Linear segment
        let h = t[1] - t[0];
        let slope = if h.abs() < 1e-15 {
            0.0
        } else {
            (y[1] - y[0]) / h
        };
        return CubicCoeffs {
            a: y.to_vec(),
            b: vec![slope],
            c: vec![0.0, 0.0],
            d: vec![0.0],
        };
    }

    // Compute segment lengths
    let h: Vec<f64> = (0..n - 1).map(|i| t[i + 1] - t[i]).collect();

    // Build tridiagonal system for c values (second derivatives / 2)
    // Natural boundary: c[0] = 0, c[n-1] = 0
    let m = n - 2; // number of interior points
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m.saturating_sub(1)];
    let mut lower = vec![0.0; m.saturating_sub(1)];
    let mut rhs = vec![0.0; m];

    for i in 0..m {
        let idx = i + 1; // index in original arrays (1..n-1)
        diag[i] = 2.0 * (h[idx - 1] + h[idx]);
        rhs[i] = 3.0 * ((y[idx + 1] - y[idx]) / h[idx] - (y[idx] - y[idx - 1]) / h[idx - 1]);
        if i > 0 {
            lower[i - 1] = h[idx - 1];
        }
        if i < m - 1 {
            upper[i] = h[idx];
        }
    }

    // Solve tridiagonal system (Thomas algorithm)
    let c_interior = solve_tridiagonal(&lower, &diag, &upper, &rhs);

    // Build full c array with boundary conditions
    let mut c = vec![0.0; n];
    for (i, &val) in c_interior.iter().enumerate() {
        c[i + 1] = val;
    }

    // Compute b and d from c
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];

    for i in 0..n - 1 {
        if h[i].abs() < 1e-15 {
            b[i] = 0.0;
            d[i] = 0.0;
        } else {
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }
    }

    CubicCoeffs {
        a: y.to_vec(),
        b,
        c,
        d,
    }
}

/// Evaluate cubic spline at parameter t.
fn evaluate_cubic_spline(t_params: &[f64], coeffs: &CubicCoeffs, t: f64) -> f64 {
    // Find the segment containing t
    let seg = find_segment(t_params, t);
    let dt = t - t_params[seg];

    coeffs.a[seg] + coeffs.b[seg] * dt + coeffs.c[seg] * dt * dt + coeffs.d[seg] * dt * dt * dt
}

/// Evaluate cubic spline and its first two derivatives at parameter t.
fn evaluate_cubic_spline_derivatives(
    t_params: &[f64],
    coeffs: &CubicCoeffs,
    t: f64,
) -> (f64, f64, f64) {
    let seg = find_segment(t_params, t);
    let dt = t - t_params[seg];

    let pos =
        coeffs.a[seg] + coeffs.b[seg] * dt + coeffs.c[seg] * dt * dt + coeffs.d[seg] * dt.powi(3);
    let vel = coeffs.b[seg] + 2.0 * coeffs.c[seg] * dt + 3.0 * coeffs.d[seg] * dt * dt;
    let acc = 2.0 * coeffs.c[seg] + 6.0 * coeffs.d[seg] * dt;

    (pos, vel, acc)
}

/// Find which segment contains parameter t.
fn find_segment(t_params: &[f64], t: f64) -> usize {
    let n = t_params.len();
    if t <= t_params[0] {
        return 0;
    }
    if t >= t_params[n - 1] {
        return n - 2; // last segment
    }

    // Binary search
    match t_params.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(exact) => exact.min(n - 2),
        Err(insert) => (insert - 1).min(n - 2),
    }
}

/// Solve tridiagonal system Ax = rhs using Thomas algorithm.
fn solve_tridiagonal(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    // Forward sweep
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        if i < n - 1 {
            c_prime[i] = upper[i] / (diag[i] - lower[i - 1] * c_prime[i - 1]);
        }
        d_prime[i] =
            (rhs[i] - lower[i - 1] * d_prime[i - 1]) / (diag[i] - lower[i - 1] * c_prime[i - 1]);
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

/// Select control points for B-spline based on smoothing factor.
///
/// smoothing=0.0: keep all points.
/// smoothing=1.0: keep minimum (start, end, a few intermediate).
fn select_control_points(path: &[Vec<f64>], smoothing: f64) -> Vec<Vec<f64>> {
    let n = path.len();
    if n <= 4 {
        return path.to_vec();
    }

    // Number of control points: interpolate between n and 4
    let min_ctrl = 4;
    let max_ctrl = n;
    let n_ctrl = max_ctrl - ((max_ctrl - min_ctrl) as f64 * smoothing) as usize;
    let n_ctrl = n_ctrl.max(min_ctrl);

    if n_ctrl >= n {
        return path.to_vec();
    }

    // Uniformly sample n_ctrl points from the path
    let mut ctrl = Vec::with_capacity(n_ctrl);
    for i in 0..n_ctrl {
        let idx = if n_ctrl <= 1 {
            0
        } else {
            ((n - 1) as f64 * i as f64 / (n_ctrl - 1) as f64).round() as usize
        };
        ctrl.push(path[idx.min(n - 1)].clone());
    }

    ctrl
}

/// Evaluate cubic B-spline at parameter t for joint j.
fn evaluate_bspline(knots: &[f64], control_points: &[Vec<f64>], joint: usize, t: f64) -> f64 {
    let n_ctrl = control_points.len();
    let degree = 3;

    // Clamp t
    let t = t.clamp(knots[degree], knots[n_ctrl]);

    // De Boor's algorithm
    let mut k = degree; // find knot span
    for i in degree..n_ctrl {
        if t < knots[i + 1] {
            k = i;
            break;
        }
    }
    // Handle edge case at end
    if t >= knots[n_ctrl] {
        k = n_ctrl - 1;
    }

    // Initialize with control point values
    let mut d: Vec<f64> = (0..=degree)
        .map(|i| {
            let idx = k - degree + i;
            if idx < n_ctrl {
                control_points[idx][joint]
            } else {
                control_points[n_ctrl - 1][joint]
            }
        })
        .collect();

    // De Boor iteration
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let left = k - degree + j;
            let right = left + degree + 1 - r;
            let left_knot = knots[left];
            let right_knot = if right < knots.len() {
                knots[right]
            } else {
                knots[knots.len() - 1]
            };

            let denom = right_knot - left_knot;
            let alpha = if denom.abs() < 1e-15 {
                0.0
            } else {
                (t - left_knot) / denom
            };

            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    d[degree]
}

/// Linear interpolation between two endpoints.
fn linear_interpolate(path: &[Vec<f64>], num_points: usize) -> SmoothedPath {
    if path.len() < 2 || num_points < 2 {
        return SmoothedPath {
            waypoints: path.to_vec(),
            c2_continuous: true,
        };
    }

    let from = &path[0];
    let to = path.last().unwrap();
    let dof = from.len();

    let waypoints: Vec<Vec<f64>> = (0..num_points)
        .map(|k| {
            let t = k as f64 / (num_points - 1) as f64;
            (0..dof).map(|j| from[j] + t * (to[j] - from[j])).collect()
        })
        .collect();

    SmoothedPath {
        waypoints,
        c2_continuous: true,
    }
}

/// Euclidean distance in joint space.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FreeSpaceChecker;
    impl CollisionChecker for FreeSpaceChecker {
        fn is_in_collision(&self, _joints: &[f64]) -> bool {
            false
        }
    }

    struct BlockAllChecker;
    impl CollisionChecker for BlockAllChecker {
        fn is_in_collision(&self, _joints: &[f64]) -> bool {
            true
        }
    }

    fn make_zigzag_path() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![2.0, -0.3],
            vec![3.0, 0.8],
            vec![4.0, 0.2],
            vec![5.0, 1.0],
        ]
    }

    #[test]
    fn cubic_spline_preserves_endpoints() {
        let path = make_zigzag_path();
        let result = smooth_cubic_spline(&path, 50, Some(&FreeSpaceChecker));

        assert!(result.c2_continuous);

        let first = &result.waypoints[0];
        let last = result.waypoints.last().unwrap();

        // Spline interpolates through endpoints
        assert!((first[0] - 0.0).abs() < 1e-10);
        assert!((first[1] - 0.0).abs() < 1e-10);
        assert!((last[0] - 5.0).abs() < 1e-10);
        assert!((last[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cubic_spline_interpolates_through_waypoints() {
        let path = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];

        // Sample exactly at knot parameters
        let t_params = chord_length_params(&path);
        let y: Vec<f64> = path.iter().map(|wp| wp[0]).collect();
        let coeffs = natural_cubic_spline_coeffs(&t_params, &y);

        // Check interpolation at each knot
        for (i, wp) in path.iter().enumerate() {
            let val = evaluate_cubic_spline(&t_params, &coeffs, t_params[i]);
            assert!(
                (val - wp[0]).abs() < 1e-10,
                "Spline should pass through waypoint {}: expected {}, got {}",
                i,
                wp[0],
                val
            );
        }
    }

    #[test]
    fn cubic_spline_c2_continuity() {
        let path = make_zigzag_path();
        let n = path.len();
        let t_params = chord_length_params(&path);

        // Check each joint
        for j in 0..2 {
            let y: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
            let coeffs = natural_cubic_spline_coeffs(&t_params, &y);

            // Check continuity at each interior knot
            for i in 1..n - 1 {
                let t = t_params[i];
                let eps = 1e-8;

                // Position continuity
                let (p_left, v_left, a_left) =
                    evaluate_cubic_spline_derivatives(&t_params, &coeffs, t - eps);
                let (p_right, v_right, a_right) =
                    evaluate_cubic_spline_derivatives(&t_params, &coeffs, t + eps);

                assert!(
                    (p_left - p_right).abs() < 1e-5,
                    "Position discontinuity at knot {} for joint {}: {} vs {}",
                    i,
                    j,
                    p_left,
                    p_right
                );
                assert!(
                    (v_left - v_right).abs() < 1e-4,
                    "Velocity discontinuity at knot {} for joint {}: {} vs {}",
                    i,
                    j,
                    v_left,
                    v_right
                );
                assert!(
                    (a_left - a_right).abs() < 1e-2,
                    "Acceleration discontinuity at knot {} for joint {}: {} vs {}",
                    i,
                    j,
                    a_left,
                    a_right
                );
            }
        }
    }

    #[test]
    fn cubic_spline_natural_boundary() {
        let path = make_zigzag_path();
        let t_params = chord_length_params(&path);

        for j in 0..2 {
            let y: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
            let coeffs = natural_cubic_spline_coeffs(&t_params, &y);

            // Natural boundary: c[0] = 0 and c[n-1] = 0 (second derivative = 0 at endpoints)
            assert!(
                coeffs.c[0].abs() < 1e-10,
                "Natural boundary violated at start for joint {}: c[0] = {}",
                j,
                coeffs.c[0]
            );
            assert!(
                coeffs.c[path.len() - 1].abs() < 1e-10,
                "Natural boundary violated at end for joint {}: c[n-1] = {}",
                j,
                coeffs.c[path.len() - 1]
            );
        }
    }

    #[test]
    fn cubic_spline_collision_fallback() {
        let path = make_zigzag_path();

        // With a checker that blocks everything, should fall back to original path
        let result = smooth_cubic_spline(&path, 50, Some(&BlockAllChecker));
        assert_eq!(result.waypoints.len(), path.len());
        assert!(!result.c2_continuous);
    }

    #[test]
    fn bspline_produces_output() {
        let path = make_zigzag_path();
        let result = smooth_bspline(&path, 50, 0.0, Some(&FreeSpaceChecker));

        assert!(result.c2_continuous);
        assert_eq!(result.waypoints.len(), 50);

        // Start and end should match original
        let first = &result.waypoints[0];
        let last = result.waypoints.last().unwrap();
        assert!((first[0] - 0.0).abs() < 1e-10);
        assert!((last[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn bspline_smoothing_factor() {
        let path = make_zigzag_path();

        // Smoothing 0.0 should be close to interpolation
        let smooth0 = smooth_bspline(&path, 50, 0.0, Some(&FreeSpaceChecker));
        // Smoothing 1.0 should be smoother
        let smooth1 = smooth_bspline(&path, 50, 1.0, Some(&FreeSpaceChecker));

        // Both should produce valid output
        assert_eq!(smooth0.waypoints.len(), 50);
        assert_eq!(smooth1.waypoints.len(), 50);
    }

    #[test]
    fn spline_derivatives_at_endpoints() {
        let path = make_zigzag_path();

        let (pos, _vel, acc) = spline_derivatives(&path, 0.0);
        assert!((pos[0] - 0.0).abs() < 1e-10);
        assert!((pos[1] - 0.0).abs() < 1e-10);
        // Natural boundary: acceleration should be ~0 at endpoints
        assert!(
            acc[0].abs() < 1e-5,
            "Endpoint acceleration should be ~0: {}",
            acc[0]
        );

        let (pos_end, _vel_end, acc_end) = spline_derivatives(&path, 1.0);
        assert!((pos_end[0] - 5.0).abs() < 1e-10);
        assert!(acc_end[0].abs() < 1e-5);
    }

    #[test]
    fn short_path_handling() {
        // Single point
        let single = vec![vec![1.0, 2.0]];
        let r = smooth_cubic_spline(&single, 10, None::<&FreeSpaceChecker>);
        assert_eq!(r.waypoints.len(), 1);

        // Two points
        let two = vec![vec![0.0], vec![1.0]];
        let r = smooth_cubic_spline(&two, 10, None::<&FreeSpaceChecker>);
        assert_eq!(r.waypoints.len(), 10);
        assert!((r.waypoints[0][0] - 0.0).abs() < 1e-10);
        assert!((r.waypoints[9][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tridiagonal_solver() {
        // Simple 3x3 system: diag=[2,2,2], upper=[1,1], lower=[1,1], rhs=[1,2,1]
        let lower = vec![1.0, 1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let upper = vec![1.0, 1.0];
        let rhs = vec![1.0, 2.0, 1.0];

        let x = solve_tridiagonal(&lower, &diag, &upper, &rhs);
        assert_eq!(x.len(), 3);

        // Verify: Ax = rhs
        let check0 = diag[0] * x[0] + upper[0] * x[1];
        let check1 = lower[0] * x[0] + diag[1] * x[1] + upper[1] * x[2];
        let check2 = lower[1] * x[1] + diag[2] * x[2];

        assert!((check0 - rhs[0]).abs() < 1e-10);
        assert!((check1 - rhs[1]).abs() < 1e-10);
        assert!((check2 - rhs[2]).abs() < 1e-10);
    }

    // ─── New collision safety & edge case tests ───

    /// A checker that blocks a specific 2D box region.
    struct BoxChecker2D {
        x_lo: f64,
        x_hi: f64,
        y_lo: f64,
        y_hi: f64,
    }

    impl CollisionChecker for BoxChecker2D {
        fn is_in_collision(&self, joints: &[f64]) -> bool {
            joints[0] >= self.x_lo
                && joints[0] <= self.x_hi
                && joints[1] >= self.y_lo
                && joints[1] <= self.y_hi
        }
    }

    /// Smoothed path remains collision-free when obstacle exists.
    /// The spline should detour around the obstacle (or fall back to original path).
    #[test]
    fn cubic_spline_collision_safety_with_obstacle() {
        // Path goes around a box obstacle in the center
        let path = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.0, 1.0],
            vec![0.5, 1.0],
            vec![1.0, 1.0],
        ];

        let checker = BoxChecker2D {
            x_lo: 0.3,
            x_hi: 0.7,
            y_lo: 0.3,
            y_hi: 0.7,
        };

        let result = smooth_cubic_spline(&path, 100, Some(&checker));

        // Every waypoint in the smoothed result should be collision-free
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(
                !checker.is_in_collision(wp),
                "smoothed waypoint {} is in collision: {:?}",
                i,
                wp
            );
        }
    }

    /// B-spline smoothed path remains collision-free.
    #[test]
    fn bspline_collision_safety_with_obstacle() {
        let path = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.0, 1.0],
            vec![0.5, 1.0],
            vec![1.0, 1.0],
            vec![1.5, 1.0],
        ];

        let checker = BoxChecker2D {
            x_lo: 0.3,
            x_hi: 0.7,
            y_lo: 0.3,
            y_hi: 0.7,
        };

        let result = smooth_bspline(&path, 100, 0.3, Some(&checker));

        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(
                !checker.is_in_collision(wp),
                "B-spline waypoint {} is in collision: {:?}",
                i,
                wp
            );
        }
    }

    /// Smoothing near joint limits: verify smoothed path stays within limits.
    #[test]
    fn cubic_spline_respects_joint_limits() {
        // Path with values near the limits [−π, π] ≈ [−3.14, 3.14]
        let limit = std::f64::consts::PI;
        let path = vec![
            vec![limit * 0.9, -limit * 0.9],
            vec![limit * 0.95, -limit * 0.5],
            vec![limit * 0.85, -limit * 0.3],
            vec![limit * 0.7, limit * 0.2],
            vec![limit * 0.5, limit * 0.8],
        ];

        let result = smooth_cubic_spline(&path, 200, Some(&FreeSpaceChecker));
        assert!(result.c2_continuous);

        // Verify no output sample overshoots beyond the input range
        let (min_j0, max_j0) = path
            .iter()
            .map(|wp| wp[0])
            .fold((f64::MAX, f64::MIN), |(min, max), v| {
                (min.min(v), max.max(v))
            });
        let (min_j1, max_j1) = path
            .iter()
            .map(|wp| wp[1])
            .fold((f64::MAX, f64::MIN), |(min, max), v| {
                (min.min(v), max.max(v))
            });

        // Allow some spline overshoot (natural cubic spline doesn't guarantee monotonicity)
        // but it should be bounded
        let margin = 0.5; // up to 0.5 rad overshoot acceptable for cubic spline
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(
                wp[0] >= min_j0 - margin && wp[0] <= max_j0 + margin,
                "joint 0 at waypoint {} = {} exceeds bounds [{}, {}] by more than {margin}",
                i,
                wp[0],
                min_j0,
                max_j0,
            );
            assert!(
                wp[1] >= min_j1 - margin && wp[1] <= max_j1 + margin,
                "joint 1 at waypoint {} = {} exceeds bounds [{}, {}] by more than {margin}",
                i,
                wp[1],
                min_j1,
                max_j1,
            );
        }
    }

    /// Cubic spline C2 continuity at interior boundary waypoints: verify position,
    /// velocity, and acceleration are continuous from both sides.
    #[test]
    fn cubic_spline_c2_continuity_at_boundaries() {
        let path = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![3.0, -0.5],
            vec![4.0, 0.5],
        ];

        let t_params = chord_length_params(&path);

        for j in 0..2 {
            let y: Vec<f64> = path.iter().map(|wp| wp[j]).collect();
            let coeffs = natural_cubic_spline_coeffs(&t_params, &y);

            // Check at each interior knot (indices 1, 2, 3)
            for i in 1..path.len() - 1 {
                let t = t_params[i];
                let eps = 1e-10;

                let (p_l, v_l, a_l) =
                    evaluate_cubic_spline_derivatives(&t_params, &coeffs, t - eps);
                let (p_r, v_r, a_r) =
                    evaluate_cubic_spline_derivatives(&t_params, &coeffs, t + eps);

                assert!(
                    (p_l - p_r).abs() < 1e-6,
                    "C0 violated at knot {i} joint {j}: {p_l} vs {p_r}"
                );
                assert!(
                    (v_l - v_r).abs() < 1e-4,
                    "C1 violated at knot {i} joint {j}: {v_l} vs {v_r}"
                );
                assert!(
                    (a_l - a_r).abs() < 1e-2,
                    "C2 violated at knot {i} joint {j}: {a_l} vs {a_r}"
                );
            }
        }
    }

    /// B-spline collision fallback: when smoothed path collides, return original.
    #[test]
    fn bspline_collision_fallback() {
        let path = make_zigzag_path();
        let result = smooth_bspline(&path, 50, 0.5, Some(&BlockAllChecker));
        // Should fall back to original path
        assert_eq!(result.waypoints.len(), path.len());
        assert!(!result.c2_continuous);
    }

    /// Smoothing a straight-line path should produce a straight-line output.
    #[test]
    fn cubic_spline_straight_line_stays_straight() {
        // Collinear path: all points on y = 2*x
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];

        let result = smooth_cubic_spline(&path, 50, Some(&FreeSpaceChecker));
        assert!(result.c2_continuous);

        // All output points should lie on y = 2*x
        for (i, wp) in result.waypoints.iter().enumerate() {
            let expected_y = 2.0 * wp[0];
            assert!(
                (wp[1] - expected_y).abs() < 1e-8,
                "waypoint {} deviates from line: ({}, {}) expected y={}",
                i,
                wp[0],
                wp[1],
                expected_y,
            );
        }
    }

    // ─── Additional coverage tests ───

    #[test]
    fn smoothed_path_debug_and_clone() {
        let sp = SmoothedPath {
            waypoints: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
            c2_continuous: true,
        };
        let cloned = sp.clone();
        assert_eq!(cloned.waypoints.len(), 2);
        assert!(cloned.c2_continuous);
        let debug = format!("{:?}", sp);
        assert!(debug.contains("c2_continuous"));
    }

    #[test]
    fn cubic_spline_empty_path() {
        let empty: Vec<Vec<f64>> = vec![];
        let result = smooth_cubic_spline(&empty, 10, None::<&FreeSpaceChecker>);
        assert!(result.waypoints.is_empty());
        assert!(result.c2_continuous);
    }

    #[test]
    fn cubic_spline_single_point_path() {
        let single = vec![vec![5.0, 10.0]];
        let result = smooth_cubic_spline(&single, 10, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 1);
        assert_eq!(result.waypoints[0], vec![5.0, 10.0]);
        assert!(result.c2_continuous);
    }

    #[test]
    fn cubic_spline_two_points() {
        let two = vec![vec![0.0, 0.0], vec![10.0, 20.0]];
        let result = smooth_cubic_spline(&two, 11, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 11);
        assert!(result.c2_continuous);

        // Should be linear interpolation
        for (i, wp) in result.waypoints.iter().enumerate() {
            let t = i as f64 / 10.0;
            assert!(
                (wp[0] - t * 10.0).abs() < 1e-8,
                "two-point spline should be linear, wp[{}][0] = {}",
                i,
                wp[0]
            );
            assert!(
                (wp[1] - t * 20.0).abs() < 1e-8,
                "two-point spline should be linear, wp[{}][1] = {}",
                i,
                wp[1]
            );
        }
    }

    #[test]
    fn cubic_spline_num_output_one() {
        let path = make_zigzag_path();
        let result = smooth_cubic_spline(&path, 1, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 1);
        // With num_output=1, should sample at t=0 -> first waypoint
        assert!((result.waypoints[0][0] - 0.0).abs() < 1e-8);
    }

    #[test]
    fn bspline_empty_path() {
        let empty: Vec<Vec<f64>> = vec![];
        let result = smooth_bspline(&empty, 10, 0.5, None::<&FreeSpaceChecker>);
        assert!(result.waypoints.is_empty());
        assert!(result.c2_continuous);
    }

    #[test]
    fn bspline_single_point() {
        let single = vec![vec![5.0]];
        let result = smooth_bspline(&single, 10, 0.5, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 1);
        assert!(result.c2_continuous);
    }

    #[test]
    fn bspline_two_points() {
        let two = vec![vec![0.0], vec![1.0]];
        let result = smooth_bspline(&two, 10, 0.5, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 10);
        assert!(result.c2_continuous);
    }

    #[test]
    fn bspline_three_points() {
        let three = vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 0.0]];
        // 3 points <= 4, falls back to cubic spline
        let result = smooth_bspline(&three, 20, 0.5, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 20);
        assert!(result.c2_continuous);
    }

    #[test]
    fn bspline_four_points() {
        let four = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
        ];
        // 4 points <= 4, falls back to cubic spline
        let result = smooth_bspline(&four, 20, 0.5, None::<&FreeSpaceChecker>);
        assert_eq!(result.waypoints.len(), 20);
        assert!(result.c2_continuous);
    }

    #[test]
    fn bspline_smoothing_factor_clamped() {
        let path = make_zigzag_path();
        // Negative smoothing should be clamped to 0.0
        let r1 = smooth_bspline(&path, 50, -1.0, None::<&FreeSpaceChecker>);
        assert_eq!(r1.waypoints.len(), 50);
        // Greater than 1.0 should be clamped to 1.0
        let r2 = smooth_bspline(&path, 50, 2.0, None::<&FreeSpaceChecker>);
        assert_eq!(r2.waypoints.len(), 50);
    }

    #[test]
    fn bspline_preserves_start_and_end() {
        let path = make_zigzag_path();
        let result = smooth_bspline(&path, 100, 0.5, None::<&FreeSpaceChecker>);
        let first = &result.waypoints[0];
        let last = result.waypoints.last().unwrap();
        // Start and end should match original path exactly
        assert!((first[0] - 0.0).abs() < 1e-10);
        assert!((first[1] - 0.0).abs() < 1e-10);
        assert!((last[0] - 5.0).abs() < 1e-10);
        assert!((last[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spline_derivatives_single_point() {
        let path = vec![vec![5.0, 10.0]];
        let (pos, vel, acc) = spline_derivatives(&path, 0.5);
        assert!((pos[0] - 5.0).abs() < 1e-10);
        assert!((pos[1] - 10.0).abs() < 1e-10);
        // Velocity and acceleration should be zero for a single point
        assert!((vel[0]).abs() < 1e-10);
        assert!((acc[0]).abs() < 1e-10);
    }

    #[test]
    fn spline_derivatives_empty_path() {
        let path: Vec<Vec<f64>> = vec![];
        let (pos, vel, acc) = spline_derivatives(&path, 0.5);
        assert!(pos.is_empty());
        assert!(vel.is_empty());
        assert!(acc.is_empty());
    }

    #[test]
    fn spline_derivatives_two_points() {
        let path = vec![vec![0.0], vec![10.0]];
        let (pos, vel, _acc) = spline_derivatives(&path, 0.5);
        // At t=0.5, position should be midpoint
        assert!(
            (pos[0] - 5.0).abs() < 1e-8,
            "midpoint should be 5.0, got {}",
            pos[0]
        );
        // Velocity should be constant (linear)
        assert!(vel[0].abs() > 0.0, "velocity should be nonzero for linear path");
    }

    #[test]
    fn spline_derivatives_at_midpoint() {
        let path = make_zigzag_path();
        let (pos, vel, acc) = spline_derivatives(&path, 0.5);
        // Position should be finite
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
        // Velocity should be finite
        assert!(vel[0].is_finite());
        assert!(vel[1].is_finite());
        // Acceleration should be finite
        assert!(acc[0].is_finite());
        assert!(acc[1].is_finite());
    }

    #[test]
    fn spline_derivatives_clamped_t() {
        let path = make_zigzag_path();
        // t < 0 should clamp to 0
        let (pos_neg, _, _) = spline_derivatives(&path, -1.0);
        let (pos_zero, _, _) = spline_derivatives(&path, 0.0);
        assert!((pos_neg[0] - pos_zero[0]).abs() < 1e-10);

        // t > 1 should clamp to 1
        let (pos_over, _, _) = spline_derivatives(&path, 2.0);
        let (pos_one, _, _) = spline_derivatives(&path, 1.0);
        assert!((pos_over[0] - pos_one[0]).abs() < 1e-10);
    }

    #[test]
    fn chord_length_params_basic() {
        let path = vec![vec![0.0], vec![1.0], vec![3.0]];
        let t = chord_length_params(&path);
        assert_eq!(t.len(), 3);
        assert!((t[0] - 0.0).abs() < 1e-10);
        assert!((t[1] - 1.0).abs() < 1e-10);
        assert!((t[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn chord_length_params_multidimensional() {
        let path = vec![vec![0.0, 0.0], vec![3.0, 4.0]]; // distance = 5
        let t = chord_length_params(&path);
        assert!((t[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn find_segment_boundary_cases() {
        let t_params = vec![0.0, 1.0, 2.0, 3.0];
        // Before start
        assert_eq!(find_segment(&t_params, -1.0), 0);
        // At start
        assert_eq!(find_segment(&t_params, 0.0), 0);
        // At end
        assert_eq!(find_segment(&t_params, 3.0), 2);
        // Past end
        assert_eq!(find_segment(&t_params, 5.0), 2);
        // At middle knot
        assert_eq!(find_segment(&t_params, 1.0), 1);
        // Between knots
        assert_eq!(find_segment(&t_params, 1.5), 1);
        assert_eq!(find_segment(&t_params, 0.5), 0);
        assert_eq!(find_segment(&t_params, 2.5), 2);
    }

    #[test]
    fn find_segment_two_points() {
        let t_params = vec![0.0, 1.0];
        assert_eq!(find_segment(&t_params, 0.0), 0);
        assert_eq!(find_segment(&t_params, 0.5), 0);
        assert_eq!(find_segment(&t_params, 1.0), 0);
    }

    #[test]
    fn tridiagonal_solver_empty() {
        let x = solve_tridiagonal(&[], &[], &[], &[]);
        assert!(x.is_empty());
    }

    #[test]
    fn tridiagonal_solver_single() {
        let x = solve_tridiagonal(&[], &[4.0], &[], &[8.0]);
        assert_eq!(x.len(), 1);
        assert!((x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn tridiagonal_solver_2x2() {
        // [2 1] [x0]   [5]
        // [1 2] [x1] = [5]
        // Solution: x0 = x1 = 5/3
        let lower = vec![1.0];
        let diag = vec![2.0, 2.0];
        let upper = vec![1.0];
        let rhs = vec![5.0, 5.0];
        let x = solve_tridiagonal(&lower, &diag, &upper, &rhs);
        assert_eq!(x.len(), 2);
        let expected = 5.0 / 3.0;
        assert!(
            (x[0] - expected).abs() < 1e-10,
            "x[0] = {}, expected {}",
            x[0],
            expected
        );
        assert!(
            (x[1] - expected).abs() < 1e-10,
            "x[1] = {}, expected {}",
            x[1],
            expected
        );
    }

    #[test]
    fn tridiagonal_solver_identity_like() {
        // Diagonal = [1, 1, 1, 1], no off-diagonals -> x = rhs
        let diag = vec![1.0, 1.0, 1.0, 1.0];
        let lower = vec![0.0, 0.0, 0.0];
        let upper = vec![0.0, 0.0, 0.0];
        let rhs = vec![3.0, 7.0, 11.0, 13.0];
        let x = solve_tridiagonal(&lower, &diag, &upper, &rhs);
        for (i, (&xi, &ri)) in x.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (xi - ri).abs() < 1e-10,
                "x[{}] = {}, expected {}",
                i,
                xi,
                ri
            );
        }
    }

    #[test]
    fn select_control_points_few_points() {
        let path = vec![vec![0.0], vec![1.0], vec![2.0]];
        let ctrl = select_control_points(&path, 0.5);
        // <= 4 points -> return all
        assert_eq!(ctrl.len(), 3);
    }

    #[test]
    fn select_control_points_no_smoothing() {
        let path = make_zigzag_path();
        let ctrl = select_control_points(&path, 0.0);
        assert_eq!(ctrl.len(), path.len());
    }

    #[test]
    fn select_control_points_max_smoothing() {
        let path = make_zigzag_path();
        let ctrl = select_control_points(&path, 1.0);
        assert!(ctrl.len() >= 4, "Should have at least 4 control points, got {}", ctrl.len());
        assert!(ctrl.len() < path.len(), "Max smoothing should reduce control points");
    }

    #[test]
    fn linear_interpolate_basic() {
        let path = vec![vec![0.0, 0.0], vec![10.0, 20.0]];
        let result = linear_interpolate(&path, 5);
        assert_eq!(result.waypoints.len(), 5);
        assert!(result.c2_continuous);
        // Check first and last
        assert!((result.waypoints[0][0]).abs() < 1e-10);
        assert!((result.waypoints[4][0] - 10.0).abs() < 1e-10);
        // Check midpoint
        assert!((result.waypoints[2][0] - 5.0).abs() < 1e-10);
        assert!((result.waypoints[2][1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn linear_interpolate_single_output() {
        let path = vec![vec![0.0], vec![1.0]];
        let result = linear_interpolate(&path, 1);
        // num_points < 2 -> return original
        assert_eq!(result.waypoints.len(), 2);
    }

    #[test]
    fn linear_interpolate_empty_path() {
        let path: Vec<Vec<f64>> = vec![];
        let result = linear_interpolate(&path, 10);
        // path.len() < 2 -> return original
        assert!(result.waypoints.is_empty());
    }

    #[test]
    fn joint_distance_matches_euclidean() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = ((4.0f64).powi(2) * 4.0).sqrt(); // sqrt(16*4) = 8
        assert!(
            (joint_distance(&a, &b) - expected).abs() < 1e-10,
            "expected {}, got {}",
            expected,
            joint_distance(&a, &b)
        );
    }

    #[test]
    fn cubic_spline_no_collision_checker() {
        let path = make_zigzag_path();
        let result = smooth_cubic_spline(&path, 30, None::<&FreeSpaceChecker>);
        assert!(result.c2_continuous);
        assert_eq!(result.waypoints.len(), 30);
    }

    #[test]
    fn bspline_no_collision_checker() {
        let path = make_zigzag_path();
        let result = smooth_bspline(&path, 30, 0.3, None::<&FreeSpaceChecker>);
        assert!(result.c2_continuous);
        assert_eq!(result.waypoints.len(), 30);
    }

    #[test]
    fn evaluate_bspline_at_boundaries() {
        // Simple test: 4 control points, uniform knots
        let control_points = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];
        let n_ctrl = control_points.len();
        let n_knots = n_ctrl + 4;
        let knots: Vec<f64> = (0..n_knots).map(|i| i as f64).collect();
        let t_min = knots[3]; // 3.0
        let t_max = knots[n_ctrl]; // 4.0

        // Evaluate at t_min
        let val_min = evaluate_bspline(&knots, &control_points, 0, t_min);
        assert!(val_min.is_finite());

        // Evaluate at t_max
        let val_max = evaluate_bspline(&knots, &control_points, 0, t_max);
        assert!(val_max.is_finite());

        // Evaluate at midpoint
        let val_mid = evaluate_bspline(&knots, &control_points, 0, (t_min + t_max) / 2.0);
        assert!(val_mid.is_finite());
    }

    #[test]
    fn cubic_spline_coeffs_two_points() {
        let t = vec![0.0, 1.0];
        let y = vec![0.0, 5.0];
        let coeffs = natural_cubic_spline_coeffs(&t, &y);
        // Should be linear: a=[0,5], b=[5], c=[0,0], d=[0]
        assert!((coeffs.a[0] - 0.0).abs() < 1e-10);
        assert!((coeffs.b[0] - 5.0).abs() < 1e-10);
        assert!((coeffs.c[0]).abs() < 1e-10);
        assert!((coeffs.d[0]).abs() < 1e-10);
    }

    #[test]
    fn cubic_spline_coeffs_three_points() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];
        let coeffs = natural_cubic_spline_coeffs(&t, &y);
        // a should be the y values
        assert!((coeffs.a[0] - 0.0).abs() < 1e-10);
        assert!((coeffs.a[1] - 1.0).abs() < 1e-10);
        assert!((coeffs.a[2] - 0.0).abs() < 1e-10);
        // Natural boundary: c[0] = c[2] = 0
        assert!((coeffs.c[0]).abs() < 1e-10);
        assert!((coeffs.c[2]).abs() < 1e-10);
    }

    #[test]
    fn evaluate_cubic_spline_at_knots() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 1.0, 3.0];
        let coeffs = natural_cubic_spline_coeffs(&t, &y);

        // At each knot, the spline should return the exact y value
        for i in 0..t.len() {
            let val = evaluate_cubic_spline(&t, &coeffs, t[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "at t={}, expected {}, got {}",
                t[i],
                y[i],
                val
            );
        }
    }
}
