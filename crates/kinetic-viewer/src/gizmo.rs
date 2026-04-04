//! 3D gizmo rendering and ray-based interaction for interactive markers.
//!
//! Provides:
//! - [`screen_to_ray`]: unproject screen coordinates to a world-space ray
//! - [`gizmo_lines`]: generate line vertices for translation arrows + rotation rings
//! - [`hit_test_gizmo`]: check if a ray intersects a gizmo handle
//! - [`drag_along_axis`]: project mouse delta onto a constrained axis

use crate::pipeline::LineVertex;
use crate::Camera;
use nalgebra::{Matrix4, Point3, Vector3};

/// A ray in world space (origin + direction).
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

/// Which handle of a gizmo was hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoHandle {
    TranslateX,
    TranslateY,
    TranslateZ,
    RotateX,
    RotateY,
    RotateZ,
}

impl GizmoHandle {
    /// The world-space axis for this handle.
    pub fn axis(&self) -> Vector3<f32> {
        match self {
            Self::TranslateX | Self::RotateX => Vector3::x(),
            Self::TranslateY | Self::RotateY => Vector3::y(),
            Self::TranslateZ | Self::RotateZ => Vector3::z(),
        }
    }

    /// Color for this handle (RGB matching axis).
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::TranslateX | Self::RotateX => [1.0, 0.2, 0.2, 1.0],
            Self::TranslateY | Self::RotateY => [0.2, 1.0, 0.2, 1.0],
            Self::TranslateZ | Self::RotateZ => [0.2, 0.2, 1.0, 1.0],
        }
    }
}

/// Unproject screen coordinates to a world-space ray.
pub fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    camera: &Camera,
    aspect: f32,
) -> Ray {
    let view = camera.view_matrix();
    let proj = match camera.projection {
        crate::Projection::Perspective { fov_y, near, far } => {
            Matrix4::new_perspective(aspect, fov_y.to_radians(), near, far)
        }
        crate::Projection::Orthographic { scale, near, far } => {
            Matrix4::new_orthographic(-scale * aspect, scale * aspect, -scale, scale, near, far)
        }
    };

    let inv_vp = (proj * view)
        .try_inverse()
        .unwrap_or_else(Matrix4::identity);

    // NDC coordinates
    let ndc_x = (2.0 * screen_x / viewport_width) - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_y / viewport_height);

    let near_ndc = nalgebra::Vector4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far_ndc = nalgebra::Vector4::new(ndc_x, ndc_y, 1.0, 1.0);

    let near_world = inv_vp * near_ndc;
    let far_world = inv_vp * far_ndc;

    let near_pos = Point3::new(
        near_world.x / near_world.w,
        near_world.y / near_world.w,
        near_world.z / near_world.w,
    );
    let far_pos = Point3::new(
        far_world.x / far_world.w,
        far_world.y / far_world.w,
        far_world.z / far_world.w,
    );

    let direction = (far_pos - near_pos).normalize();

    Ray {
        origin: near_pos,
        direction,
    }
}

/// Generate line vertices for a 6-DOF translation gizmo (3 arrows) at a position.
///
/// `length` controls arrow size. Returns line vertices in pairs (start, end).
pub fn translation_gizmo_lines(center: [f32; 3], length: f32) -> Vec<LineVertex> {
    let mut lines = Vec::with_capacity(6);
    let c = center;

    // X arrow (red)
    lines.push(LineVertex { position: c, color: [1.0, 0.2, 0.2, 1.0] });
    lines.push(LineVertex { position: [c[0] + length, c[1], c[2]], color: [1.0, 0.2, 0.2, 1.0] });

    // Y arrow (green)
    lines.push(LineVertex { position: c, color: [0.2, 1.0, 0.2, 1.0] });
    lines.push(LineVertex { position: [c[0], c[1] + length, c[2]], color: [0.2, 1.0, 0.2, 1.0] });

    // Z arrow (blue)
    lines.push(LineVertex { position: c, color: [0.2, 0.2, 1.0, 1.0] });
    lines.push(LineVertex { position: [c[0], c[1], c[2] + length], color: [0.2, 0.2, 1.0, 1.0] });

    lines
}

/// Generate line vertices for rotation rings around each axis.
pub fn rotation_gizmo_lines(center: [f32; 3], radius: f32, segments: usize) -> Vec<LineVertex> {
    let mut lines = Vec::with_capacity(segments * 2 * 3);
    let c = center;

    for axis in 0..3 {
        let color = match axis {
            0 => [1.0, 0.2, 0.2, 0.6], // X ring = red
            1 => [0.2, 1.0, 0.2, 0.6], // Y ring = green
            _ => [0.2, 0.2, 1.0, 0.6], // Z ring = blue
        };

        for i in 0..segments {
            let t0 = std::f32::consts::TAU * i as f32 / segments as f32;
            let t1 = std::f32::consts::TAU * (i + 1) as f32 / segments as f32;

            let (p0, p1) = match axis {
                0 => (
                    [c[0], c[1] + radius * t0.cos(), c[2] + radius * t0.sin()],
                    [c[0], c[1] + radius * t1.cos(), c[2] + radius * t1.sin()],
                ),
                1 => (
                    [c[0] + radius * t0.cos(), c[1], c[2] + radius * t0.sin()],
                    [c[0] + radius * t1.cos(), c[1], c[2] + radius * t1.sin()],
                ),
                _ => (
                    [c[0] + radius * t0.cos(), c[1] + radius * t0.sin(), c[2]],
                    [c[0] + radius * t1.cos(), c[1] + radius * t1.sin(), c[2]],
                ),
            };

            lines.push(LineVertex { position: p0, color });
            lines.push(LineVertex { position: p1, color });
        }
    }

    lines
}

/// Hit-test a ray against a gizmo at `center` with `arrow_length` and `ring_radius`.
///
/// Returns the closest handle hit, or None.
pub fn hit_test_gizmo(
    ray: &Ray,
    center: [f32; 3],
    arrow_length: f32,
    ring_radius: f32,
    threshold: f32,
) -> Option<GizmoHandle> {
    let c = Point3::from(center);
    // best = (handle, distance_to_handle) — pick smallest distance
    let mut best: Option<(GizmoHandle, f32)> = None;

    // Test translation arrows (line segments from center along each axis)
    let axes = [
        (GizmoHandle::TranslateX, Vector3::x()),
        (GizmoHandle::TranslateY, Vector3::y()),
        (GizmoHandle::TranslateZ, Vector3::z()),
    ];

    for (handle, axis) in &axes {
        let end = c + axis * arrow_length;
        let dist = ray_line_distance(ray, &c, &end);
        if dist < threshold {
            if best.is_none() || dist < best.unwrap().1 {
                best = Some((*handle, dist));
            }
        }
    }

    // Test rotation rings (circles around each axis)
    let ring_axes = [
        (GizmoHandle::RotateX, Vector3::x()),
        (GizmoHandle::RotateY, Vector3::y()),
        (GizmoHandle::RotateZ, Vector3::z()),
    ];

    for (handle, normal) in &ring_axes {
        let dist = ray_circle_distance(ray, &c, normal, ring_radius);
        if dist < threshold {
            if best.is_none() || dist < best.unwrap().1 {
                best = Some((*handle, dist));
            }
        }
    }

    best.map(|(h, _)| h)
}

/// Project a mouse delta onto a world-space axis, returning the displacement along that axis.
pub fn drag_along_axis(
    ray: &Ray,
    prev_ray: &Ray,
    center: &Point3<f32>,
    axis: &Vector3<f32>,
) -> f32 {
    // Find closest points on the ray and the axis line
    let t_curr = ray_axis_param(ray, center, axis);
    let t_prev = ray_axis_param(prev_ray, center, axis);
    t_curr - t_prev
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Minimum distance from a ray to a line segment (p0, p1).
fn ray_line_distance(ray: &Ray, p0: &Point3<f32>, p1: &Point3<f32>) -> f32 {
    let d = *p1 - *p0;
    let w = ray.origin - *p0;
    let a = ray.direction.dot(&ray.direction);
    let b = ray.direction.dot(&d);
    let c = d.dot(&d);
    let dd = ray.direction.dot(&w);
    let e = d.dot(&w);

    let denom = a * c - b * b;
    if denom.abs() < 1e-10 {
        return w.norm(); // parallel
    }

    let s = (b * e - c * dd) / denom;
    let t = (a * e - b * dd) / denom;
    let t = t.clamp(0.0, 1.0);
    let s = s.max(0.0);

    let closest_ray = ray.origin + ray.direction * s;
    let closest_seg = *p0 + d * t;

    (closest_ray - closest_seg).norm()
}

/// Distance from a ray to the nearest point on a circle (center, normal, radius).
fn ray_circle_distance(
    ray: &Ray,
    center: &Point3<f32>,
    normal: &Vector3<f32>,
    radius: f32,
) -> f32 {
    // Intersect ray with the plane of the circle
    let denom = ray.direction.dot(normal);
    if denom.abs() < 1e-10 {
        return f32::MAX; // parallel to plane
    }
    let t = (*center - ray.origin).dot(normal) / denom;
    if t < 0.0 {
        return f32::MAX; // behind camera
    }
    let hit = ray.origin + ray.direction * t;
    let dist_from_center = (hit - *center).norm();

    // Distance from the hit point to the ring
    (dist_from_center - radius).abs()
}

/// Parameter t along the ray for the closest approach to a point.
fn ray_closest_t(ray: &Ray, point: &Point3<f32>) -> f32 {
    let w = *point - ray.origin;
    w.dot(&ray.direction) / ray.direction.dot(&ray.direction)
}

/// Parameter along the axis line for the closest point to the ray.
fn ray_axis_param(ray: &Ray, center: &Point3<f32>, axis: &Vector3<f32>) -> f32 {
    let w = ray.origin - *center;
    let a = ray.direction.dot(&ray.direction);
    let b = ray.direction.dot(axis);
    let c = axis.dot(axis);
    let d = ray.direction.dot(&w);
    let e = axis.dot(&w);

    let denom = a * c - b * b;
    if denom.abs() < 1e-10 {
        return 0.0;
    }
    (a * e - b * d) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_creation() {
        let camera = Camera::perspective([2.0, 1.0, 2.0], [0.0, 0.0, 0.0], 45.0);
        let ray = screen_to_ray(640.0, 360.0, 1280.0, 720.0, &camera, 16.0 / 9.0);
        // Center of screen should point roughly toward camera target (origin)
        assert!(ray.direction.norm() > 0.99);
    }

    #[test]
    fn translation_gizmo_has_6_vertices() {
        let lines = translation_gizmo_lines([0.0, 0.0, 0.0], 0.5);
        assert_eq!(lines.len(), 6); // 3 arrows x 2 vertices
    }

    #[test]
    fn rotation_gizmo_has_correct_vertices() {
        let lines = rotation_gizmo_lines([0.0, 0.0, 0.0], 0.3, 16);
        assert_eq!(lines.len(), 16 * 2 * 3); // 16 segments x 2 vertices x 3 axes
    }

    #[test]
    fn hit_test_translate_x() {
        // Ray passing near the midpoint of the X arrow shaft, well away from Y/Z
        let ray = Ray {
            origin: Point3::new(0.25, 0.02, -1.0),
            direction: Vector3::new(0.0, 0.0, 1.0),
        };
        let hit = hit_test_gizmo(&ray, [0.0, 0.0, 0.0], 0.5, 0.3, 0.05);
        assert_eq!(hit, Some(GizmoHandle::TranslateX));
    }

    #[test]
    fn hit_test_miss() {
        // Ray far from gizmo
        let ray = Ray {
            origin: Point3::new(0.0, 10.0, 0.0),
            direction: Vector3::new(0.0, 0.0, 1.0),
        };
        let hit = hit_test_gizmo(&ray, [0.0, 0.0, 0.0], 0.5, 0.3, 0.05);
        assert!(hit.is_none());
    }

    #[test]
    fn drag_along_axis_moves() {
        let ray1 = Ray {
            origin: Point3::new(0.0, 1.0, 0.0),
            direction: Vector3::new(0.0, -1.0, 0.0),
        };
        let ray2 = Ray {
            origin: Point3::new(0.5, 1.0, 0.0),
            direction: Vector3::new(0.0, -1.0, 0.0),
        };
        let center = Point3::new(0.0, 0.0, 0.0);
        let axis = Vector3::x();
        let delta = drag_along_axis(&ray2, &ray1, &center, &axis);
        assert!((delta - 0.5).abs() < 0.1);
    }
}
