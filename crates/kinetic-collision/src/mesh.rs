//! Mesh collision backend using parry3d-f64 for exact geometry queries.
//!
//! Provides precise distance and contact queries for URDF collision geometries.
//! This backend converts URDF primitive shapes (Box, Cylinder, Sphere) and
//! mesh references into parry3d-f64 shapes for exact collision detection.
//!
//! Used as the "slow path" in the two-tier collision checker — only invoked
//! when the SIMD sphere-tree reports near-collision within safety margin.

use std::collections::HashMap;

use nalgebra::{Isometry3, Point3};
use parry3d_f64::shape::{Ball, Cuboid, Cylinder, SharedShape, TriMesh};

use kinetic_core::Pose;
use kinetic_robot::{GeometryShape, Robot};

use crate::soa::SpheresSoA;

/// A contact point between two shapes.
#[derive(Debug, Clone)]
pub struct ContactPoint {
    /// Point on the robot link surface (world frame).
    pub point_robot: nalgebra::Point3<f64>,
    /// Point on the obstacle surface (world frame).
    pub point_obstacle: nalgebra::Point3<f64>,
    /// Signed distance: negative = penetration.
    pub distance: f64,
    /// Index of the robot link involved.
    pub link_idx: usize,
}

/// Exact mesh collision backend wrapping parry3d-f64 shapes.
///
/// Stores one `SharedShape` per robot link, built from the link's URDF
/// collision geometry. Links without collision geometry get no shape entry.
#[derive(Debug, Clone)]
pub struct MeshCollisionBackend {
    /// parry3d-f64 shape per link index. Links without geometry are absent.
    shapes: HashMap<usize, LinkShape>,
}

/// A link's collision shape with its local-frame offset.
#[derive(Debug, Clone)]
struct LinkShape {
    shape: SharedShape,
    /// Transform from link origin to geometry origin.
    local_transform: Isometry3<f64>,
}

impl MeshCollisionBackend {
    /// Build mesh collision shapes from a robot model.
    ///
    /// Converts each link's collision geometry to parry3d-f64 shapes.
    /// If a link has multiple collision geometries, they are merged into
    /// a compound shape.
    pub fn from_robot(robot: &Robot) -> Self {
        let mut shapes = HashMap::new();

        for (link_idx, link) in robot.links.iter().enumerate() {
            if link.collision_geometry.is_empty() {
                continue;
            }

            if link.collision_geometry.len() == 1 {
                // Single geometry → direct shape
                let geom = &link.collision_geometry[0];
                if let Some(shape) = convert_geometry(&geom.shape) {
                    shapes.insert(
                        link_idx,
                        LinkShape {
                            shape,
                            local_transform: geom.origin.0,
                        },
                    );
                }
            } else {
                // Multiple geometries → compound shape
                let mut parts: Vec<(Isometry3<f64>, SharedShape)> = Vec::new();
                for geom in &link.collision_geometry {
                    if let Some(shape) = convert_geometry(&geom.shape) {
                        parts.push((geom.origin.0, shape));
                    }
                }
                if !parts.is_empty() {
                    let compound = SharedShape::compound(parts);
                    shapes.insert(
                        link_idx,
                        LinkShape {
                            shape: compound,
                            local_transform: Isometry3::identity(),
                        },
                    );
                }
            }
        }

        Self { shapes }
    }

    /// Number of links that have collision shapes.
    pub fn num_shapes(&self) -> usize {
        self.shapes.len()
    }

    /// Check if a specific link has an exact collision shape.
    pub fn has_shape(&self, link_idx: usize) -> bool {
        self.shapes.contains_key(&link_idx)
    }

    /// Compute exact minimum distance between robot and an obstacle shape.
    ///
    /// `link_transforms`: world-frame transform for each link (indexed by link index).
    /// `obstacle`: parry3d-f64 shape representing the obstacle.
    /// `obstacle_pose`: world-frame transform of the obstacle.
    ///
    /// Returns the minimum signed distance across all robot links.
    /// Positive = separated, zero = touching, negative values not returned
    /// (use `contact_points` for penetration queries).
    pub fn min_distance_exact(
        &self,
        link_transforms: &[Isometry3<f64>],
        obstacle: &SharedShape,
        obstacle_pose: &Isometry3<f64>,
    ) -> f64 {
        let mut min_dist = f64::INFINITY;

        for (&link_idx, link_shape) in &self.shapes {
            if link_idx >= link_transforms.len() {
                continue;
            }

            let world_transform = link_transforms[link_idx] * link_shape.local_transform;

            let dist = parry3d_f64::query::distance(
                &world_transform,
                link_shape.shape.as_ref(),
                obstacle_pose,
                obstacle.as_ref(),
            )
            .unwrap_or(f64::INFINITY);

            if dist < min_dist {
                min_dist = dist;
            }
        }

        min_dist
    }

    /// Compute exact minimum distance between two robot links.
    ///
    /// Used for exact self-collision distance when sphere check is ambiguous.
    pub fn link_distance(
        &self,
        link_a: usize,
        transform_a: &Isometry3<f64>,
        link_b: usize,
        transform_b: &Isometry3<f64>,
    ) -> f64 {
        let shape_a = match self.shapes.get(&link_a) {
            Some(s) => s,
            None => return f64::INFINITY,
        };
        let shape_b = match self.shapes.get(&link_b) {
            Some(s) => s,
            None => return f64::INFINITY,
        };

        let world_a = *transform_a * shape_a.local_transform;
        let world_b = *transform_b * shape_b.local_transform;

        parry3d_f64::query::distance(
            &world_a,
            shape_a.shape.as_ref(),
            &world_b,
            shape_b.shape.as_ref(),
        )
        .unwrap_or(f64::INFINITY)
    }

    /// Find contact points between robot and obstacle within a margin.
    ///
    /// Returns all contacts where distance < margin (including penetrations).
    pub fn contact_points(
        &self,
        link_transforms: &[Isometry3<f64>],
        obstacle: &SharedShape,
        obstacle_pose: &Isometry3<f64>,
        margin: f64,
    ) -> Vec<ContactPoint> {
        let mut contacts = Vec::new();

        for (&link_idx, link_shape) in &self.shapes {
            if link_idx >= link_transforms.len() {
                continue;
            }

            let world_transform = link_transforms[link_idx] * link_shape.local_transform;

            if let Ok(Some(contact)) = parry3d_f64::query::contact(
                &world_transform,
                link_shape.shape.as_ref(),
                obstacle_pose,
                obstacle.as_ref(),
                margin,
            ) {
                contacts.push(ContactPoint {
                    point_robot: world_transform * contact.point1,
                    point_obstacle: *obstacle_pose * contact.point2,
                    distance: contact.dist,
                    link_idx,
                });
            }
        }

        contacts
    }

    /// Compute exact minimum distance from robot to obstacle spheres.
    ///
    /// Convenience method that creates Ball shapes from obstacle sphere data.
    pub fn min_distance_to_spheres(
        &self,
        link_transforms: &[Isometry3<f64>],
        obstacle_centers: &[(f64, f64, f64)],
        obstacle_radii: &[f64],
    ) -> f64 {
        let mut min_dist = f64::INFINITY;

        for (center, &radius) in obstacle_centers.iter().zip(obstacle_radii.iter()) {
            let obs_shape = SharedShape::ball(radius);
            let obs_pose = Isometry3::translation(center.0, center.1, center.2);

            let d = self.min_distance_exact(link_transforms, &obs_shape, &obs_pose);
            if d < min_dist {
                min_dist = d;
            }
        }

        min_dist
    }
}

/// Convert link transforms from Pose slice to Isometry3 slice.
pub fn poses_to_isometries(poses: &[Pose]) -> Vec<Isometry3<f64>> {
    poses.iter().map(|p| p.0).collect()
}

/// Convert a URDF geometry shape to a parry3d-f64 SharedShape.
fn convert_geometry(shape: &GeometryShape) -> Option<SharedShape> {
    match shape {
        GeometryShape::Sphere { radius } => Some(SharedShape::new(Ball::new(*radius))),

        GeometryShape::Box { x, y, z } => {
            // GeometryShape::Box stores half-extents
            Some(SharedShape::new(Cuboid::new(nalgebra::Vector3::new(
                *x, *y, *z,
            ))))
        }

        GeometryShape::Cylinder { radius, length } => {
            // parry3d-f64 Cylinder is centered at origin, half-height
            Some(SharedShape::new(Cylinder::new(*length / 2.0, *radius)))
        }

        GeometryShape::Mesh { filename, scale } => {
            // For mesh files, we attempt to load STL/OBJ.
            // If the file doesn't exist or can't be loaded, return None
            // and fall back to sphere approximation.
            load_mesh(filename, scale)
        }
    }
}

/// Attempt to load a mesh file and convert to a parry3d-f64 TriMesh.
///
/// Supports STL format. Returns None if the file can't be loaded.
fn load_mesh(filename: &str, scale: &[f64; 3]) -> Option<SharedShape> {
    // Try to load as STL
    let data = std::fs::read(filename).ok()?;
    let mesh = stl_io::read_stl(&mut std::io::Cursor::new(data)).ok()?;

    let vertices: Vec<nalgebra::Point3<f64>> = mesh
        .vertices
        .iter()
        .map(|v| {
            nalgebra::Point3::new(
                v[0] as f64 * scale[0],
                v[1] as f64 * scale[1],
                v[2] as f64 * scale[2],
            )
        })
        .collect();

    let indices: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .map(|f| {
            [
                f.vertices[0] as u32,
                f.vertices[1] as u32,
                f.vertices[2] as u32,
            ]
        })
        .collect();

    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let trimesh = TriMesh::new(vertices, indices);
    Some(SharedShape::new(trimesh))
}

/// Create a SharedShape from common primitive descriptions (convenience).
pub fn shape_from_sphere(radius: f64) -> SharedShape {
    SharedShape::ball(radius)
}

/// Create a box obstacle shape from half-extents.
pub fn shape_from_box(hx: f64, hy: f64, hz: f64) -> SharedShape {
    SharedShape::cuboid(hx, hy, hz)
}

/// Create a cylinder obstacle shape.
pub fn shape_from_cylinder(radius: f64, half_height: f64) -> SharedShape {
    SharedShape::cylinder(half_height, radius)
}

/// Configuration for approximate convex decomposition of concave meshes.
#[derive(Debug, Clone)]
pub struct ConvexDecompConfig {
    /// Maximum number of convex hulls to produce. Default: 16.
    pub max_hulls: usize,
    /// Voxel resolution for decomposition. Default: 64.
    pub resolution: u32,
    /// Maximum concavity tolerance. Default: 0.01.
    pub max_concavity: f64,
}

impl Default for ConvexDecompConfig {
    fn default() -> Self {
        Self {
            max_hulls: 16,
            resolution: 64,
            max_concavity: 0.01,
        }
    }
}

/// Decompose a concave triangle mesh into an approximate convex compound shape.
///
/// Returns `None` if the input is degenerate (no vertices, no triangles, or
/// fewer than 4 unique vertices).
///
/// The result is a `SharedShape` that can be used directly with parry3d-f64
/// distance/contact queries.
pub fn convex_decomposition(
    vertices: &[Point3<f64>],
    indices: &[[u32; 3]],
    _config: &ConvexDecompConfig,
) -> Option<SharedShape> {
    if vertices.len() < 4 || indices.is_empty() {
        return None;
    }

    // Build a convex hull from all vertices as a conservative approximation.
    // A full VHACD decomposition would produce tighter hulls for concave shapes,
    // but the convex hull is correct and conservative for collision checking.
    Some(SharedShape::convex_hull(vertices)?)
}

/// Convert a point cloud into collision spheres.
///
/// Each point becomes a sphere with the given `sphere_radius`.
///
/// If `voxel_size` is `Some(size)`, points are voxel-grid downsampled first:
/// each occupied voxel produces one sphere at the voxel centroid with an
/// inflated radius that conservatively covers the voxel diagonal.
///
/// # Arguments
/// - `points`: 3D point positions `[x, y, z]`.
/// - `sphere_radius`: radius for each point sphere (meters).
/// - `voxel_size`: if `Some`, downsample into voxels of this size first.
///
/// # Returns
/// A `SpheresSoA` containing the resulting collision spheres.
pub fn pointcloud_to_spheres(
    points: &[[f64; 3]],
    sphere_radius: f64,
    voxel_size: Option<f64>,
) -> SpheresSoA {
    let mut out = SpheresSoA::new();

    if points.is_empty() {
        return out;
    }

    match voxel_size {
        None => {
            // Direct mode: one sphere per point
            for p in points {
                out.push(p[0], p[1], p[2], sphere_radius, 0);
            }
        }
        Some(vs) => {
            // Voxel-grid downsampling: group points into voxels, emit centroid per voxel
            let inv_vs = 1.0 / vs;
            let mut voxels: HashMap<(i64, i64, i64), (f64, f64, f64, usize)> = HashMap::new();

            for p in points {
                let vx = (p[0] * inv_vs).floor() as i64;
                let vy = (p[1] * inv_vs).floor() as i64;
                let vz = (p[2] * inv_vs).floor() as i64;

                let entry = voxels.entry((vx, vy, vz)).or_insert((0.0, 0.0, 0.0, 0));
                entry.0 += p[0];
                entry.1 += p[1];
                entry.2 += p[2];
                entry.3 += 1;
            }

            // Effective radius: covers voxel half-diagonal + point sphere radius
            // Half-diagonal of a cube with side `vs` = vs * sqrt(3)/2 ≈ vs * 0.866
            let effective_radius = vs * 0.866 + sphere_radius;

            for (_, (sx, sy, sz, count)) in &voxels {
                let n = *count as f64;
                out.push(sx / n, sy / n, sz / n, effective_radius, 0);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_robot::Robot;

    const GEOM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_geom">
  <link name="base_link">
    <collision>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </collision>
  </link>
  <link name="link2">
    <collision>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

    #[test]
    fn build_mesh_backend() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        // 3 links have geometry (base, link1, link2), ee_link has none
        assert_eq!(backend.num_shapes(), 3);
        assert!(backend.has_shape(0)); // base_link
        assert!(backend.has_shape(1)); // link1
        assert!(backend.has_shape(2)); // link2
        assert!(!backend.has_shape(3)); // ee_link
    }

    #[test]
    fn exact_distance_far_obstacle() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        let obstacle = shape_from_sphere(0.1);
        let obstacle_pose = Isometry3::translation(5.0, 5.0, 5.0);

        let dist = backend.min_distance_exact(&transforms, &obstacle, &obstacle_pose);
        assert!(dist > 1.0, "Expected large distance, got {}", dist);
    }

    #[test]
    fn exact_distance_close_obstacle() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        // Sphere obstacle touching the base_link box
        // Base box is 0.2x0.2x0.1 (half-extents 0.1, 0.1, 0.05)
        // Place obstacle just outside: at (0.2, 0, 0) with radius 0.05
        let obstacle = shape_from_sphere(0.05);
        let obstacle_pose = Isometry3::translation(0.2, 0.0, 0.0);

        let dist = backend.min_distance_exact(&transforms, &obstacle, &obstacle_pose);
        // Distance from box face at x=0.1 to sphere surface at x=0.15 → ~0.05
        assert!(dist >= 0.0, "Expected non-negative distance, got {}", dist);
        assert!(dist < 0.2, "Expected small distance, got {}", dist);
    }

    #[test]
    fn exact_distance_penetrating_obstacle() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        // Large obstacle at origin → overlapping
        let obstacle = shape_from_sphere(0.5);
        let obstacle_pose = Isometry3::translation(0.0, 0.0, 0.0);

        let dist = backend.min_distance_exact(&transforms, &obstacle, &obstacle_pose);
        // parry3d distance returns 0.0 for penetrating shapes
        assert!(
            dist <= 0.01,
            "Expected zero/penetration distance, got {}",
            dist
        );
    }

    #[test]
    fn contact_points_within_margin() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        // Obstacle near base_link
        let obstacle = shape_from_sphere(0.05);
        let obstacle_pose = Isometry3::translation(0.2, 0.0, 0.0);

        let contacts = backend.contact_points(&transforms, &obstacle, &obstacle_pose, 1.0);
        // With 1m margin, should find contacts with nearby links
        assert!(!contacts.is_empty(), "Expected contacts within margin");
    }

    #[test]
    fn link_to_link_distance() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let t_base = Isometry3::identity();
        let t_link2 = Isometry3::translation(0.0, 0.0, 0.5);

        let dist = backend.link_distance(0, &t_base, 2, &t_link2);
        assert!(
            dist > 0.0,
            "Expected positive distance between separated links, got {}",
            dist
        );
    }

    #[test]
    fn convenience_shape_constructors() {
        let sphere = shape_from_sphere(0.1);
        let box_shape = shape_from_box(0.1, 0.2, 0.3);
        let cyl = shape_from_cylinder(0.05, 0.15);

        // Verify shapes are created (non-null AABB)
        let aabb_s = sphere.compute_aabb(&Isometry3::identity());
        assert!(aabb_s.extents().x > 0.0);

        let aabb_b = box_shape.compute_aabb(&Isometry3::identity());
        assert!(aabb_b.extents().x > 0.0);

        let aabb_c = cyl.compute_aabb(&Isometry3::identity());
        assert!(aabb_c.extents().x > 0.0);
    }

    // ─── Edge case & degenerate geometry tests ───

    /// link_distance returns INFINITY when one link has no collision shape.
    #[test]
    fn link_distance_missing_shape_returns_infinity() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let t = Isometry3::identity();

        // link index 3 = ee_link (no geometry)
        let dist = backend.link_distance(0, &t, 3, &t);
        assert_eq!(dist, f64::INFINITY, "Missing shape should give INFINITY");

        // Both missing
        let dist2 = backend.link_distance(3, &t, 3, &t);
        assert_eq!(dist2, f64::INFINITY);
    }

    /// min_distance_to_spheres convenience: distance to multiple obstacle spheres.
    #[test]
    fn min_distance_to_spheres_convenience() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        let centers = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0)];
        let radii = [0.1, 0.1];

        let dist = backend.min_distance_to_spheres(&transforms, &centers, &radii);
        // Both spheres are 10m away; robot geometry at origin is ~0.1m extent
        assert!(
            dist > 5.0,
            "Far spheres should give large distance, got {}",
            dist
        );
    }

    /// min_distance_exact with empty transforms list: should return INFINITY
    /// (no links can be checked).
    #[test]
    fn min_distance_empty_transforms() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let empty_transforms: Vec<Isometry3<f64>> = vec![];
        let obstacle = shape_from_sphere(0.1);
        let obstacle_pose = Isometry3::translation(0.0, 0.0, 0.0);

        let dist = backend.min_distance_exact(&empty_transforms, &obstacle, &obstacle_pose);
        assert_eq!(
            dist,
            f64::INFINITY,
            "Empty transforms should yield INFINITY"
        );
    }

    /// Contact points with far-away obstacle: should return no contacts.
    #[test]
    fn contact_points_far_obstacle_no_contacts() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        let transforms: Vec<Isometry3<f64>> = (0..robot.links.len())
            .map(|_| Isometry3::identity())
            .collect();

        let obstacle = shape_from_sphere(0.01);
        let obstacle_pose = Isometry3::translation(50.0, 50.0, 50.0);

        // With a small margin (0.01m), far obstacle should yield no contacts
        let contacts = backend.contact_points(&transforms, &obstacle, &obstacle_pose, 0.01);
        assert!(
            contacts.is_empty(),
            "Far obstacle should produce no contacts, got {}",
            contacts.len()
        );
    }

    /// Robot with no collision geometry links: backend should have zero shapes.
    #[test]
    fn backend_no_geometry_robot() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="bare">
  <link name="base"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
</robot>"#;

        let robot = Robot::from_urdf_string(urdf).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        assert_eq!(backend.num_shapes(), 0);
        assert!(!backend.has_shape(0));
        assert!(!backend.has_shape(1));

        // Distance queries with no shapes should return INFINITY
        let transforms = vec![Isometry3::identity(); 2];
        let obs = shape_from_sphere(1.0);
        let dist = backend.min_distance_exact(&transforms, &obs, &Isometry3::identity());
        assert_eq!(dist, f64::INFINITY);
    }

    /// Self-collision distance between overlapping links at the same position.
    #[test]
    fn link_distance_overlapping_links() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = MeshCollisionBackend::from_robot(&robot);

        // Both at origin — base_link (box) and link1 (cylinder) overlap
        let t = Isometry3::identity();
        let dist = backend.link_distance(0, &t, 1, &t);

        // Overlapping shapes → distance should be 0 (penetration)
        assert!(
            dist <= 0.01,
            "Overlapping links should have ~0 distance, got {}",
            dist
        );
    }
}
