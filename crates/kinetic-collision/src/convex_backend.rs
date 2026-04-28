//! Convex-hull collision backend (the medium-fidelity LOD tier).
//!
//! Pulled out of `lod.rs` so the LOD dispatcher there is just selection
//! logic, not also the convex-hull shape implementation. The public
//! [`ConvexCollisionBackend`] type is re-exported from `lod` to keep
//! external paths (`kinetic_collision::lod::ConvexCollisionBackend`)
//! stable.

use std::collections::HashMap;

use nalgebra::Isometry3;
use parry3d_f64::shape::SharedShape;

use kinetic_robot::{GeometryShape, Robot};

/// Per-link convex hull shapes for the medium-fidelity tier.
///
/// For primitive shapes (sphere, box, cylinder), the convex hull IS the exact
/// shape — no approximation. For mesh files, a convex hull is computed from
/// the mesh vertices, which is less conservative than sphere approximation
/// but faster than full mesh collision.
#[derive(Debug, Clone)]
pub struct ConvexCollisionBackend {
    /// Convex hull shape per link index. Links without geometry are absent.
    shapes: HashMap<usize, ConvexLinkShape>,
}

/// A link's convex hull shape with its local-frame offset.
#[derive(Debug, Clone)]
struct ConvexLinkShape {
    shape: SharedShape,
    local_transform: Isometry3<f64>,
}

impl ConvexCollisionBackend {
    /// Build convex hull shapes from a robot model.
    ///
    /// Primitives (sphere, box, cylinder) produce exact shapes.
    /// Mesh files produce convex hulls from their vertices.
    pub fn from_robot(robot: &Robot) -> Self {
        let mut shapes = HashMap::new();

        for (link_idx, link) in robot.links.iter().enumerate() {
            if link.collision_geometry.is_empty() {
                continue;
            }

            if link.collision_geometry.len() == 1 {
                let geom = &link.collision_geometry[0];
                if let Some(shape) = convert_to_convex(&geom.shape) {
                    shapes.insert(
                        link_idx,
                        ConvexLinkShape {
                            shape,
                            local_transform: geom.origin.0,
                        },
                    );
                }
            } else {
                let mut parts: Vec<(Isometry3<f64>, SharedShape)> = Vec::new();
                for geom in &link.collision_geometry {
                    if let Some(shape) = convert_to_convex(&geom.shape) {
                        parts.push((geom.origin.0, shape));
                    }
                }
                if !parts.is_empty() {
                    shapes.insert(
                        link_idx,
                        ConvexLinkShape {
                            shape: SharedShape::compound(parts),
                            local_transform: Isometry3::identity(),
                        },
                    );
                }
            }
        }

        Self { shapes }
    }

    /// Number of links that have convex collision shapes.
    pub fn num_shapes(&self) -> usize {
        self.shapes.len()
    }

    /// Check if a specific link has a convex collision shape.
    pub fn has_shape(&self, link_idx: usize) -> bool {
        self.shapes.contains_key(&link_idx)
    }

    /// Compute minimum distance between robot and an obstacle shape.
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

    /// Compute minimum distance between two robot links using convex shapes.
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
}

/// Convert a URDF geometry shape to a convex hull shape.
///
/// Primitives (sphere, box, cylinder) are exact shapes.
/// Meshes produce convex hulls from their vertex set.
fn convert_to_convex(shape: &GeometryShape) -> Option<SharedShape> {
    match shape {
        GeometryShape::Sphere { radius } => Some(SharedShape::ball(*radius)),
        GeometryShape::Box { x, y, z } => Some(SharedShape::cuboid(*x, *y, *z)),
        GeometryShape::Cylinder { radius, length } => {
            Some(SharedShape::cylinder(*length / 2.0, *radius))
        }
        GeometryShape::Mesh { filename, scale } => load_mesh_as_convex(filename, scale),
    }
}

/// Load a mesh file and compute its convex hull.
fn load_mesh_as_convex(filename: &str, scale: &[f64; 3]) -> Option<SharedShape> {
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

    if vertices.len() < 4 {
        return None;
    }

    parry3d_f64::shape::ConvexPolyhedron::from_convex_hull(&vertices)
        .map(SharedShape::new)
}
