//! Paden-Kahan subproblem solvers for analytical IK.
//!
//! Three canonical subproblems from the robotics literature (Murray, Li, Sastry):
//! - SP1: Rotation about a single axis to move a point to a target
//! - SP2: Two sequential rotations about distinct axes
//! - SP3: Rotation about a single axis to achieve a given distance

use nalgebra::{Matrix3, Vector3};

const TOLERANCE: f64 = 1e-10;

/// Subproblem 1: Find angle θ such that Rot(ω, θ) applied at point `r`
/// moves point `p` to point `q`.
///
/// Returns `Some(θ)` if feasible, `None` if the rotation cannot produce the
/// desired mapping (the points don't lie on the same circle about the axis).
pub fn subproblem1(
    omega: &Vector3<f64>,
    r: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
) -> Option<f64> {
    let omega_n = omega.normalize();

    // Translate so rotation axis passes through origin
    let u_prime = p - r;
    let v_prime = q - r;

    // Project onto plane perpendicular to ω
    let u = u_prime - omega_n * omega_n.dot(&u_prime);
    let v = v_prime - omega_n * omega_n.dot(&v_prime);

    let u_norm = u.norm();
    let v_norm = v.norm();

    // Feasibility: both projections must have equal length
    if u_norm < TOLERANCE || v_norm < TOLERANCE {
        return None;
    }
    if (u_norm - v_norm).abs() > 1e-6 * u_norm.max(v_norm).max(1.0) {
        return None;
    }

    // Also check that the component along ω matches
    if (omega_n.dot(&u_prime) - omega_n.dot(&v_prime)).abs() > 1e-6 {
        return None;
    }

    let theta = f64::atan2(omega_n.dot(&u.cross(&v)), u.dot(&v));
    Some(theta)
}

/// Subproblem 2: Find angles (θ1, θ2) such that
/// Rot(ω1, θ1) at r1 · Rot(ω2, θ2) at r2 · p = q.
///
/// Returns up to 2 solution pairs.
pub fn subproblem2(
    omega1: &Vector3<f64>,
    omega2: &Vector3<f64>,
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
) -> Vec<(f64, f64)> {
    let w1 = omega1.normalize();
    let w2 = omega2.normalize();

    // The intermediate point c = Rot(ω2, θ2) · p must also satisfy
    // Rot(ω1, θ1) · c = q. So c lies on:
    // - Circle C2: p rotated about ω2 through r2
    // - Circle C1: q rotated about ω1 through r1 (backwards)
    //
    // Strategy: parameterize c and find where the two constraints intersect.

    let u = p - r2;
    let v = q - r1;

    // Project u onto plane ⊥ ω2 and v onto plane ⊥ ω1
    let u_par = w2 * w2.dot(&u);
    let u_perp = u - u_par;
    let v_par = w1 * w1.dot(&v);
    let v_perp = v - v_par;

    let ru = u_perp.norm();
    let rv = v_perp.norm();

    if ru < TOLERANCE || rv < TOLERANCE {
        return Vec::new();
    }

    // The intermediate point c (relative to r2) has the form:
    // c - r2 = u_par + ru * (cos(θ2) * e_u + sin(θ2) * (w2 × e_u))
    // where e_u = u_perp / |u_perp|
    //
    // Similarly for c relative to r1 (from q side):
    // c - r1 = v_par + rv * (cos(θ1) * e_v + sin(θ1) * (w1 × e_v))

    // We need |c - r1|² for the c on circle C2 to equal the radius of circle C1.
    // This gives us a constraint on θ2.

    // Distance from r1 to the axis-parallel component of c on circle 2:
    // c = r2 + u_par + ru * (cos θ2 * e_u + sin θ2 * f_u)
    // where e_u = u_perp/ru, f_u = w2 × e_u

    let e_u = u_perp / ru;
    let f_u = w2.cross(&e_u);

    // The c point on circle 2:
    // c(θ2) = r2 + u_par + ru * cos(θ2) * e_u + ru * sin(θ2) * f_u

    // The constraint: c must also lie on circle 1 about r1 with axis w1.
    // The component of (c - r1) along w1 must equal v_par = w1 * w1·v:
    // w1 · (c(θ2) - r1) = w1 · v = w1 · (q - r1)

    let base = r2 + u_par - r1;
    let w1_base = w1.dot(&base);
    let w1_eu = w1.dot(&e_u);
    let w1_fu = w1.dot(&f_u);
    let target_proj = w1.dot(&v);

    // w1_base + ru * (cos θ2 * w1_eu + sin θ2 * w1_fu) = target_proj
    let rhs = target_proj - w1_base;
    let amp = (ru * ru * (w1_eu * w1_eu + w1_fu * w1_fu)).sqrt();

    if amp < TOLERANCE {
        return Vec::new();
    }

    let sin_cos_ratio = rhs / amp;
    if sin_cos_ratio.abs() > 1.0 + 1e-8 {
        return Vec::new();
    }
    let sin_cos_ratio = sin_cos_ratio.clamp(-1.0, 1.0);

    let phi = f64::atan2(ru * w1_fu, ru * w1_eu);
    let alpha = f64::acos(sin_cos_ratio);

    let theta2_candidates = [phi + alpha, phi - alpha];

    let mut solutions = Vec::new();

    for &theta2 in &theta2_candidates {
        // Compute intermediate point c for this θ2
        let c = r2 + u_par + ru * (theta2.cos() * e_u + theta2.sin() * f_u);

        // Now solve SP1: Rot(ω1, θ1) at r1 maps c to q
        if let Some(theta1) = subproblem1(&w1, r1, &c, q) {
            // Verify: also check that Rot(ω2, θ2) at r2 maps p to c
            let rot2 = axis_angle_rotation(&w2, theta2);
            let p_rot = r2 + rot2 * (p - r2);
            if (p_rot - c).norm() < 1e-4 {
                solutions.push((theta1, theta2));
            }
        }
    }

    solutions
}

/// Subproblem 3: Find angle θ such that |Rot(ω, θ) at r · p - q| = delta.
///
/// Returns up to 2 solutions.
pub fn subproblem3(
    omega: &Vector3<f64>,
    r: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
    delta: f64,
) -> Vec<f64> {
    let w = omega.normalize();

    // Translate so axis passes through origin
    let u = p - r;
    let d = q - r;

    // Component along axis (invariant under rotation)
    let u_par = w * w.dot(&u);
    let u_perp = u - u_par;
    let d_par = w * w.dot(&d);
    let d_perp = d - d_par;

    let rho = u_perp.norm();
    if rho < TOLERANCE {
        return Vec::new();
    }

    // After rotation: the rotated point is at u_par + rho*(cos θ * e_u + sin θ * f_u)
    // Distance squared to d:
    // |u_par - d_par|² + |rho*(cos θ * e_u + sin θ * f_u) - d_perp|²
    //
    // Let's expand:
    // = |u_par - d_par|² + rho² + |d_perp|² - 2*rho*(cos θ * e_u·d_perp + sin θ * f_u·d_perp)
    //
    // Set this equal to delta²:

    let e_u = u_perp / rho;
    let f_u = w.cross(&e_u);

    let axis_diff_sq = (u_par - d_par).norm_squared();
    let d_perp_sq = d_perp.norm_squared();

    let a = e_u.dot(&d_perp);
    let b = f_u.dot(&d_perp);

    // axis_diff_sq + rho² + d_perp_sq - 2*rho*(cos θ * a + sin θ * b) = delta²
    // cos θ * a + sin θ * b = (axis_diff_sq + rho² + d_perp_sq - delta²) / (2*rho)

    let rhs = (axis_diff_sq + rho * rho + d_perp_sq - delta * delta) / (2.0 * rho);
    let amp = (a * a + b * b).sqrt();

    if amp < TOLERANCE {
        return Vec::new();
    }

    let ratio = rhs / amp;
    if ratio.abs() > 1.0 + 1e-8 {
        return Vec::new();
    }
    let ratio = ratio.clamp(-1.0, 1.0);

    let phi = f64::atan2(b, a);
    let alpha = f64::acos(ratio);

    let mut solutions = Vec::new();
    let t1 = phi + alpha;
    let t2 = phi - alpha;

    solutions.push(t1);
    if (t1 - t2).abs() > 1e-10 {
        solutions.push(t2);
    }

    solutions
}

/// Build a 3x3 rotation matrix from axis-angle.
pub fn axis_angle_rotation(axis: &Vector3<f64>, angle: f64) -> Matrix3<f64> {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let x = axis.x;
    let y = axis.y;
    let z = axis.z;

    Matrix3::new(
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    )
}

/// Decompose a rotation matrix into ZYZ Euler angles.
/// Returns (α, β, γ) such that R = Rz(α) · Ry(β) · Rz(γ).
pub fn euler_zyz_decompose(r: &Matrix3<f64>) -> Vec<(f64, f64, f64)> {
    let r33 = r[(2, 2)].clamp(-1.0, 1.0);
    let beta = r33.acos();

    if beta.abs() < 1e-10 {
        // Gimbal lock: β ≈ 0, only α + γ is determined
        let alpha = f64::atan2(r[(1, 0)], r[(0, 0)]);
        return vec![(alpha, 0.0, 0.0)];
    }
    if (beta - std::f64::consts::PI).abs() < 1e-10 {
        // Gimbal lock: β ≈ π, only α - γ is determined
        let alpha = f64::atan2(-r[(1, 0)], -r[(0, 0)]);
        return vec![(alpha, std::f64::consts::PI, 0.0)];
    }

    let sin_beta = beta.sin();
    let alpha = f64::atan2(r[(1, 2)] / sin_beta, r[(0, 2)] / sin_beta);
    let gamma = f64::atan2(r[(2, 1)] / sin_beta, -r[(2, 0)] / sin_beta);

    // Also return the second solution with β → -β
    let alpha2 = f64::atan2(-r[(1, 2)] / sin_beta, -r[(0, 2)] / sin_beta);
    let gamma2 = f64::atan2(-r[(2, 1)] / sin_beta, r[(2, 0)] / sin_beta);

    vec![(alpha, beta, gamma), (alpha2, -beta, gamma2)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sp1_rotation_about_z() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(0.0, 1.0, 0.0);

        let theta = subproblem1(&omega, &r, &p, &q).unwrap();
        assert!((theta - PI / 2.0).abs() < 1e-8);
    }

    #[test]
    fn sp1_rotation_about_offset_axis() {
        let omega = Vector3::z();
        let r = Vector3::new(1.0, 0.0, 0.0);
        let p = Vector3::new(2.0, 0.0, 0.0);
        let q = Vector3::new(1.0, 1.0, 0.0);

        let theta = subproblem1(&omega, &r, &p, &q).unwrap();
        let rot = axis_angle_rotation(&omega, theta);
        let p_rot = r + rot * (p - r);
        assert!((p_rot - q).norm() < 1e-8);
    }

    #[test]
    fn sp1_infeasible() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(2.0, 0.0, 0.0); // different radius

        assert!(subproblem1(&omega, &r, &p, &q).is_none());
    }

    #[test]
    fn sp3_distance_constraint() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        // q offset from the rotation axis so the problem is non-degenerate
        let q = Vector3::new(2.0, 0.0, 0.0);
        let delta = 1.5;

        let solutions = subproblem3(&omega, &r, &p, &q, delta);
        assert!(!solutions.is_empty());

        for theta in &solutions {
            let rot = axis_angle_rotation(&omega, *theta);
            let p_rot = r + rot * (p - r);
            let dist = (p_rot - q).norm();
            assert!(
                (dist - delta).abs() < 1e-6,
                "distance {} != delta {}",
                dist,
                delta
            );
        }
    }

    #[test]
    fn sp3_no_solution() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(0.0, 0.0, 0.0);
        let delta = 5.0; // too far

        let solutions = subproblem3(&omega, &r, &p, &q, delta);
        assert!(solutions.is_empty());
    }

    #[test]
    fn euler_zyz_roundtrip() {
        let alpha = 0.5;
        let beta = 1.0;
        let gamma = -0.3;
        let rz_a = axis_angle_rotation(&Vector3::z(), alpha);
        let ry_b = axis_angle_rotation(&Vector3::y(), beta);
        let rz_g = axis_angle_rotation(&Vector3::z(), gamma);
        let r = rz_a * ry_b * rz_g;

        let solutions = euler_zyz_decompose(&r);
        assert!(!solutions.is_empty());

        let (a, b, g) = solutions[0];
        let r_check = axis_angle_rotation(&Vector3::z(), a)
            * axis_angle_rotation(&Vector3::y(), b)
            * axis_angle_rotation(&Vector3::z(), g);
        assert!((r - r_check).norm() < 1e-8);
    }

    // ─── Paden-Kahan edge case tests ───

    /// SP1 with zero rotation (p and q are identical after projection): angle ≈ 0.
    #[test]
    fn sp1_zero_rotation() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(1.0, 0.0, 0.0); // same as p

        let theta = subproblem1(&omega, &r, &p, &q).unwrap();
        assert!(
            theta.abs() < 1e-8,
            "Same point should give zero rotation, got {}",
            theta
        );
    }

    /// SP1 with anti-parallel vectors (angle ≈ π): 180° rotation.
    #[test]
    fn sp1_antiparallel_pi_rotation() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(-1.0, 0.0, 0.0); // opposite direction

        let theta = subproblem1(&omega, &r, &p, &q).unwrap();
        assert!(
            (theta.abs() - PI).abs() < 1e-8,
            "Opposite point should give π rotation, got {}",
            theta
        );

        // Verify the rotation actually works
        let rot = axis_angle_rotation(&omega, theta);
        let p_rot = r + rot * (p - r);
        assert!((p_rot - q).norm() < 1e-8);
    }

    /// SP1 with point on the rotation axis: should return None (zero projection norm).
    #[test]
    fn sp1_point_on_axis() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(0.0, 0.0, 5.0); // on Z axis
        let q = Vector3::new(0.0, 0.0, 5.0);

        // Both points are on the axis → perpendicular projection is zero
        // This is degenerate: any angle works, but our implementation returns None
        let result = subproblem1(&omega, &r, &p, &q);
        assert!(
            result.is_none(),
            "Points on axis should be degenerate (None)"
        );
    }

    /// SP3 tangent case: delta equals the max achievable distance.
    /// At the tangent, solutions should coincide or produce the correct distance.
    #[test]
    fn sp3_tangent_solutions_coincide() {
        let omega = Vector3::z();
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);

        let q2 = Vector3::new(2.0, 0.0, 0.0); // outside the circle
                                              // Distance range: [1.0, 3.0]
                                              // At delta=3.0 (max), solutions should coincide at angle π
        let solutions = subproblem3(&omega, &r, &p, &q2, 3.0);
        assert!(
            !solutions.is_empty(),
            "Tangent delta should have solution(s)"
        );

        // All solutions should give the correct distance
        for &theta in &solutions {
            let rot = axis_angle_rotation(&omega, theta);
            let p_rot = r + rot * (p - r);
            let dist = (p_rot - q2).norm();
            assert!(
                (dist - 3.0).abs() < 1e-6,
                "Distance should be 3.0, got {}",
                dist
            );
        }

        // If two solutions, they should coincide modulo 2π (tangent case)
        if solutions.len() == 2 {
            let diff = ((solutions[0] - solutions[1]) % (2.0 * PI)).abs();
            let diff_wrapped = diff.min((2.0 * PI) - diff);
            assert!(
                diff_wrapped < 0.01,
                "Tangent solutions should nearly coincide (mod 2π), diff={}",
                diff_wrapped
            );
        }
    }

    /// SP2 basic test: two sequential rotations.
    #[test]
    fn sp2_basic_two_rotations() {
        let w1 = Vector3::z();
        let w2 = Vector3::y();
        let r1 = Vector3::zeros();
        let r2 = Vector3::zeros();

        // Start at (1,0,0), rotate by θ2 about Y then θ1 about Z to reach target
        let p = Vector3::new(1.0, 0.0, 0.0);

        // Apply known angles: θ1=π/4, θ2=π/3
        let rot2 = axis_angle_rotation(&w2, PI / 3.0);
        let rot1 = axis_angle_rotation(&w1, PI / 4.0);
        let q = r1 + rot1 * (r2 + rot2 * (p - r2) - r1);

        let solutions = subproblem2(&w1, &w2, &r1, &r2, &p, &q);

        // Should find at least one solution
        assert!(
            !solutions.is_empty(),
            "SP2 should find solutions for a valid two-rotation problem"
        );

        // Verify at least one solution reproduces the target
        let found = solutions.iter().any(|&(t1, t2)| {
            let r2_mat = axis_angle_rotation(&w2, t2);
            let r1_mat = axis_angle_rotation(&w1, t1);
            let result = r1 + r1_mat * (r2 + r2_mat * (p - r2) - r1);
            (result - q).norm() < 1e-4
        });
        assert!(found, "At least one SP2 solution should match target");
    }

    /// Euler ZYZ gimbal lock at β ≈ 0: only α+γ is determined.
    #[test]
    fn euler_zyz_gimbal_lock_beta_zero() {
        // R = Rz(α) · Ry(0) · Rz(γ) = Rz(α+γ) (since Ry(0) = I)
        let alpha = 0.7;
        let gamma = 0.3;
        let r = axis_angle_rotation(&Vector3::z(), alpha + gamma);

        let solutions = euler_zyz_decompose(&r);
        assert!(!solutions.is_empty());

        // β should be ~0
        let (a, b, g) = solutions[0];
        assert!(
            b.abs() < 1e-8,
            "Beta should be ~0 for gimbal lock, got {}",
            b
        );

        // α + γ ≈ original sum (but individually may differ)
        let r_check = axis_angle_rotation(&Vector3::z(), a)
            * axis_angle_rotation(&Vector3::y(), b)
            * axis_angle_rotation(&Vector3::z(), g);
        assert!((r - r_check).norm() < 1e-8);
    }

    /// Euler ZYZ gimbal lock at β ≈ π: only α-γ is determined.
    #[test]
    fn euler_zyz_gimbal_lock_beta_pi() {
        let alpha = 0.5;
        let gamma = -0.2;
        let rz_a = axis_angle_rotation(&Vector3::z(), alpha);
        let ry_pi = axis_angle_rotation(&Vector3::y(), PI);
        let rz_g = axis_angle_rotation(&Vector3::z(), gamma);
        let r = rz_a * ry_pi * rz_g;

        let solutions = euler_zyz_decompose(&r);
        assert!(!solutions.is_empty());

        let (a, b, g) = solutions[0];
        assert!(
            (b.abs() - PI).abs() < 1e-8,
            "Beta should be ~π for gimbal lock, got {}",
            b
        );

        let r_check = axis_angle_rotation(&Vector3::z(), a)
            * axis_angle_rotation(&Vector3::y(), b)
            * axis_angle_rotation(&Vector3::z(), g);
        assert!(
            (r - r_check).norm() < 1e-6,
            "Gimbal lock reconstruction error: {}",
            (r - r_check).norm()
        );
    }
}
