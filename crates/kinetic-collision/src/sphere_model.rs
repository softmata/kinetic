//! Robot sphere model — approximate robot geometry with bounding spheres.
//!
//! Generates sphere approximations from URDF collision geometry primitives
//! (Box, Cylinder, Sphere, Mesh) and maintains both local-frame and
//! world-frame sphere representations for fast collision queries.

use kinetic_core::Pose;
use kinetic_robot::{GeometryShape, Link, Robot};

use crate::soa::SpheresSoA;

/// Configuration for sphere generation density.
#[derive(Debug, Clone, Copy)]
pub struct SphereGenConfig {
    /// Maximum sphere radius. Smaller values → more spheres, tighter fit.
    pub max_radius: f64,
    /// Minimum number of spheres per geometry primitive.
    pub min_spheres_per_geom: usize,
    /// Maximum number of spheres per geometry primitive.
    pub max_spheres_per_geom: usize,
}

impl Default for SphereGenConfig {
    fn default() -> Self {
        Self {
            max_radius: 0.05,
            min_spheres_per_geom: 1,
            max_spheres_per_geom: 32,
        }
    }
}

impl SphereGenConfig {
    /// Low-fidelity config: fewer, larger spheres (faster but less accurate).
    pub fn coarse() -> Self {
        Self {
            max_radius: 0.1,
            min_spheres_per_geom: 1,
            max_spheres_per_geom: 8,
        }
    }

    /// High-fidelity config: many small spheres (slower but more accurate).
    pub fn fine() -> Self {
        Self {
            max_radius: 0.02,
            min_spheres_per_geom: 4,
            max_spheres_per_geom: 64,
        }
    }
}

/// Pre-computed sphere model for a robot in local link frames.
///
/// Each link's collision geometry is approximated by a set of spheres.
/// The `link_ranges` field maps each link index to its range of spheres
/// in the `local` SoA storage.
#[derive(Debug, Clone)]
pub struct RobotSphereModel {
    /// All spheres in their local link frames.
    pub local: SpheresSoA,
    /// Range [start, end) into `local` for each link index.
    /// Length = number of links in the robot.
    pub link_ranges: Vec<(usize, usize)>,
    /// Total number of links.
    pub num_links: usize,
}

impl RobotSphereModel {
    /// Build a sphere model from a robot's collision geometry.
    pub fn from_robot(robot: &Robot, config: &SphereGenConfig) -> Self {
        let num_links = robot.links.len();
        let mut local = SpheresSoA::new();
        let mut link_ranges = Vec::with_capacity(num_links);

        for (link_idx, link) in robot.links.iter().enumerate() {
            let start = local.len();
            generate_link_spheres(link, link_idx, config, &mut local);
            let end = local.len();
            link_ranges.push((start, end));
        }

        Self {
            local,
            link_ranges,
            num_links,
        }
    }

    /// Build using default configuration.
    pub fn from_robot_default(robot: &Robot) -> Self {
        Self::from_robot(robot, &SphereGenConfig::default())
    }

    /// Total number of spheres across all links.
    pub fn total_spheres(&self) -> usize {
        self.local.len()
    }

    /// Number of spheres for a specific link.
    pub fn link_sphere_count(&self, link_idx: usize) -> usize {
        let (start, end) = self.link_ranges[link_idx];
        end - start
    }

    /// Create a runtime sphere set for world-frame queries.
    pub fn create_runtime(&self) -> RobotSpheres<'_> {
        RobotSpheres {
            world: SpheresSoA::with_capacity(self.local.len()),
            model: self,
        }
    }
}

/// Per-link collision padding and scaling configuration.
///
/// Allows inflating/deflating collision spheres per-link for safety margins,
/// conservative planning, or fine-grained collision tuning.
#[derive(Debug, Clone)]
pub struct LinkCollisionConfig {
    /// Global padding added to all sphere radii (meters). Default: 0.0.
    pub default_padding: f64,
    /// Global scale factor for all sphere radii. Default: 1.0.
    pub default_scale: f64,
    /// Per-link overrides: (link_index, padding, scale).
    overrides: Vec<(usize, f64, f64)>,
}

impl LinkCollisionConfig {
    /// Create a new config with no padding or scaling.
    pub fn new() -> Self {
        Self {
            default_padding: 0.0,
            default_scale: 1.0,
            overrides: Vec::new(),
        }
    }

    /// Set global padding (meters added to all sphere radii).
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.default_padding = padding;
        self
    }

    /// Set global scale factor for all sphere radii.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.default_scale = scale;
        self
    }

    /// Set per-link padding and scale override.
    pub fn set(&mut self, link_index: usize, padding: f64, scale: f64) {
        if let Some(entry) = self.overrides.iter_mut().find(|(idx, _, _)| *idx == link_index) {
            entry.1 = padding;
            entry.2 = scale;
        } else {
            self.overrides.push((link_index, padding, scale));
        }
    }

    /// Get padding and scale for a given link index.
    pub fn get(&self, link_index: usize) -> (f64, f64) {
        for &(idx, padding, scale) in &self.overrides {
            if idx == link_index {
                return (padding, scale);
            }
        }
        (self.default_padding, self.default_scale)
    }
}

impl Default for LinkCollisionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime world-frame spheres, updated from FK link poses.
///
/// Holds a reference to the pre-built model and maintains world-frame
/// sphere positions that are updated each time `update()` is called
/// with new link poses.
#[derive(Debug)]
pub struct RobotSpheres<'a> {
    /// World-frame spheres (updated by `update()`).
    pub world: SpheresSoA,
    /// Reference to the pre-built local-frame model.
    model: &'a RobotSphereModel,
}

impl<'a> RobotSpheres<'a> {
    /// Update world-frame spheres from link poses.
    ///
    /// `link_poses` must have one entry per link in the robot.
    /// Each local sphere is transformed by its owning link's world pose.
    pub fn update(&mut self, link_poses: &[Pose]) {
        self.world.clear();

        let local = &self.model.local;
        for i in 0..local.len() {
            let link_idx = local.link_id[i];
            let pose = &link_poses[link_idx];
            let iso = &pose.0;

            // Transform sphere center from link-local to world frame
            let local_pt = nalgebra::Point3::new(local.x[i], local.y[i], local.z[i]);
            let world_pt = iso.transform_point(&local_pt);

            self.world.push(
                world_pt.x,
                world_pt.y,
                world_pt.z,
                local.radius[i],
                link_idx,
            );
        }
    }

    /// Update world-frame spheres from link poses with per-link padding/scaling.
    ///
    /// Like `update()`, but applies the `LinkCollisionConfig` padding and scale
    /// to each sphere's radius. Padding is added after scaling. Negative effective
    /// radii are clamped to zero.
    pub fn update_with_config(&mut self, link_poses: &[Pose], config: &LinkCollisionConfig) {
        self.world.clear();

        let local = &self.model.local;
        for i in 0..local.len() {
            let link_idx = local.link_id[i];
            let pose = &link_poses[link_idx];
            let iso = &pose.0;

            let local_pt = nalgebra::Point3::new(local.x[i], local.y[i], local.z[i]);
            let world_pt = iso.transform_point(&local_pt);

            let (padding, scale) = config.get(link_idx);
            let effective_radius = (local.radius[i] * scale + padding).max(0.0);

            self.world.push(
                world_pt.x,
                world_pt.y,
                world_pt.z,
                effective_radius,
                link_idx,
            );
        }
    }

    /// Check if this robot collides with another set of spheres.
    pub fn collides_with(&self, other: &SpheresSoA) -> bool {
        self.world.any_overlap(other)
    }

    /// Check if this robot collides with another set of spheres with margin.
    pub fn collides_with_margin(&self, other: &SpheresSoA, margin: f64) -> bool {
        self.world.any_overlap_with_margin(other, margin)
    }

    /// Check for self-collision between links.
    ///
    /// `skip_pairs` is a set of (link_a, link_b) pairs to skip (e.g., adjacent links).
    /// Both orderings are checked, so only provide one direction.
    pub fn self_collision(&self, skip_pairs: &[(usize, usize)]) -> bool {
        let w = &self.world;
        for i in 0..w.len() {
            for j in (i + 1)..w.len() {
                let link_a = w.link_id[i];
                let link_b = w.link_id[j];

                // Skip same-link pairs
                if link_a == link_b {
                    continue;
                }

                // Skip allowed adjacent pairs
                if skip_pairs
                    .iter()
                    .any(|&(a, b)| (a == link_a && b == link_b) || (a == link_b && b == link_a))
                {
                    continue;
                }

                if w.overlaps(i, w, j) {
                    return true;
                }
            }
        }
        false
    }

    /// Self-collision check with margin.
    pub fn self_collision_with_margin(&self, skip_pairs: &[(usize, usize)], margin: f64) -> bool {
        let w = &self.world;
        for i in 0..w.len() {
            for j in (i + 1)..w.len() {
                let link_a = w.link_id[i];
                let link_b = w.link_id[j];

                if link_a == link_b {
                    continue;
                }

                if skip_pairs
                    .iter()
                    .any(|&(a, b)| (a == link_a && b == link_b) || (a == link_b && b == link_a))
                {
                    continue;
                }

                if w.overlaps_with_margin(i, w, j, margin) {
                    return true;
                }
            }
        }
        false
    }
}

/// Generate spheres for a single link's collision geometry.
fn generate_link_spheres(
    link: &Link,
    link_idx: usize,
    config: &SphereGenConfig,
    out: &mut SpheresSoA,
) {
    if link.collision_geometry.is_empty() {
        return;
    }

    for geom in &link.collision_geometry {
        let origin = &geom.origin;
        match &geom.shape {
            GeometryShape::Sphere { radius } => {
                generate_sphere_spheres(*radius, origin, link_idx, out);
            }
            GeometryShape::Box { x, y, z } => {
                generate_box_spheres(*x, *y, *z, origin, link_idx, config, out);
            }
            GeometryShape::Cylinder { radius, length } => {
                generate_cylinder_spheres(*radius, *length, origin, link_idx, config, out);
            }
            GeometryShape::Mesh { scale, .. } => {
                // For meshes without loaded geometry, use a conservative bounding sphere.
                // The scale factor gives us a rough size estimate.
                let max_scale = scale[0].abs().max(scale[1].abs()).max(scale[2].abs());
                let bounding_radius = 0.1 * max_scale; // conservative default
                generate_sphere_spheres(bounding_radius, origin, link_idx, out);
            }
        }
    }
}

/// A sphere primitive maps to exactly one sphere.
fn generate_sphere_spheres(radius: f64, origin: &Pose, link_idx: usize, out: &mut SpheresSoA) {
    let t = origin.translation();
    out.push(t.x, t.y, t.z, radius, link_idx);
}

/// Approximate a box with spheres.
///
/// Places spheres in a grid pattern covering the box volume.
/// Half-extents are (hx, hy, hz).
fn generate_box_spheres(
    hx: f64,
    hy: f64,
    hz: f64,
    origin: &Pose,
    link_idx: usize,
    config: &SphereGenConfig,
    out: &mut SpheresSoA,
) {
    let max_dim = hx.max(hy).max(hz);

    // Determine grid resolution based on config
    let sphere_r = config.max_radius.min(max_dim);
    let step = sphere_r * 2.0;

    let nx = ((2.0 * hx / step).ceil() as usize).max(1);
    let ny = ((2.0 * hy / step).ceil() as usize).max(1);
    let nz = ((2.0 * hz / step).ceil() as usize).max(1);

    let total = nx * ny * nz;
    let total = total.clamp(config.min_spheres_per_geom, config.max_spheres_per_geom);

    if total == 1 {
        // Single bounding sphere
        let bounding_r = (hx * hx + hy * hy + hz * hz).sqrt();
        let t = origin.translation();
        out.push(t.x, t.y, t.z, bounding_r, link_idx);
        return;
    }

    // Recompute grid to match allowed total
    let ratio = (total as f64).cbrt();
    let nx = (ratio * hx / max_dim).ceil().max(1.0) as usize;
    let ny = (ratio * hy / max_dim).ceil().max(1.0) as usize;
    let nz = (ratio * hz / max_dim).ceil().max(1.0) as usize;

    let iso = &origin.0;

    for ix in 0..nx {
        let fx = if nx > 1 {
            -hx + (2.0 * hx * ix as f64) / (nx - 1) as f64
        } else {
            0.0
        };
        for iy in 0..ny {
            let fy = if ny > 1 {
                -hy + (2.0 * hy * iy as f64) / (ny - 1) as f64
            } else {
                0.0
            };
            for iz in 0..nz {
                let fz = if nz > 1 {
                    -hz + (2.0 * hz * iz as f64) / (nz - 1) as f64
                } else {
                    0.0
                };

                let local_pt = nalgebra::Point3::new(fx, fy, fz);
                let world_pt = iso.transform_point(&local_pt);

                // Sphere radius: enough to cover the cell
                let cell_hx = if nx > 1 { hx / (nx - 1) as f64 } else { hx };
                let cell_hy = if ny > 1 { hy / (ny - 1) as f64 } else { hy };
                let cell_hz = if nz > 1 { hz / (nz - 1) as f64 } else { hz };
                let r = (cell_hx * cell_hx + cell_hy * cell_hy + cell_hz * cell_hz).sqrt();

                out.push(world_pt.x, world_pt.y, world_pt.z, r, link_idx);
            }
        }
    }
}

/// Approximate a cylinder with spheres along its axis.
///
/// Places spheres at intervals along the cylinder length,
/// each with radius equal to the cylinder radius.
fn generate_cylinder_spheres(
    radius: f64,
    length: f64,
    origin: &Pose,
    link_idx: usize,
    config: &SphereGenConfig,
    out: &mut SpheresSoA,
) {
    let half_len = length / 2.0;

    // Number of spheres along the axis
    let step = config.max_radius.min(radius) * 2.0;
    let n = ((length / step).ceil() as usize)
        .max(1)
        .clamp(config.min_spheres_per_geom, config.max_spheres_per_geom);

    let iso = &origin.0;

    if n == 1 {
        // Single bounding sphere at center
        let bounding_r = (radius * radius + half_len * half_len).sqrt();
        let t = origin.translation();
        out.push(t.x, t.y, t.z, bounding_r, link_idx);
        return;
    }

    // URDF cylinders are along Z axis
    for i in 0..n {
        let fz = if n > 1 {
            -half_len + length * i as f64 / (n - 1) as f64
        } else {
            0.0
        };

        let local_pt = nalgebra::Point3::new(0.0, 0.0, fz);
        let world_pt = iso.transform_point(&local_pt);

        out.push(world_pt.x, world_pt.y, world_pt.z, radius, link_idx);
    }
}

/// Build the default skip-pairs for a robot (adjacent links in the kinematic tree).
///
/// Returns pairs of link indices that should be skipped during self-collision
/// checking because they are directly connected by a joint.
pub fn adjacent_link_pairs(robot: &Robot) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for joint in &robot.joints {
        pairs.push((joint.parent_link, joint.child_link));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_robot::Robot;

    const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
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
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

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
    fn no_geometry_no_spheres() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);
        assert_eq!(model.total_spheres(), 0);
        assert_eq!(model.num_links, 4);
    }

    #[test]
    fn geometry_generates_spheres() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);

        // Should have spheres for base (box), link1 (cylinder), link2 (sphere)
        assert!(model.total_spheres() > 0, "Expected some spheres");

        // link2 has a sphere primitive → exactly 1 sphere
        let link2_idx = robot.link_index("link2").unwrap();
        assert_eq!(model.link_sphere_count(link2_idx), 1);

        // ee_link has no geometry → 0 spheres
        let ee_idx = robot.link_index("ee_link").unwrap();
        assert_eq!(model.link_sphere_count(ee_idx), 0);
    }

    #[test]
    fn sphere_geometry_correct_radius() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);

        let link2_idx = robot.link_index("link2").unwrap();
        let (start, end) = model.link_ranges[link2_idx];
        assert_eq!(end - start, 1);
        assert!((model.local.radius[start] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn runtime_update_transforms_spheres() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);
        let mut runtime = model.create_runtime();

        // Create identity poses for all links
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();

        runtime.update(&poses);
        assert_eq!(runtime.world.len(), model.total_spheres());
    }

    #[test]
    fn runtime_translation() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);
        let mut runtime = model.create_runtime();

        let link2_idx = robot.link_index("link2").unwrap();
        let (start, _) = model.link_ranges[link2_idx];

        // Place link2 at (1, 2, 3)
        let mut poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        poses[link2_idx] = Pose::from_xyz(1.0, 2.0, 3.0);

        runtime.update(&poses);

        // The sphere for link2 should be at (1, 2, 3) since the geometry origin
        // is at the link origin (identity) for this URDF
        assert!((runtime.world.x[start] - 1.0).abs() < 1e-10);
        assert!((runtime.world.y[start] - 2.0).abs() < 1e-10);
        assert!((runtime.world.z[start] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn adjacent_pairs() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let pairs = adjacent_link_pairs(&robot);
        assert_eq!(pairs.len(), 3); // 3 joints = 3 adjacent pairs
    }

    #[test]
    fn self_collision_with_skip() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);
        let mut runtime = model.create_runtime();
        let skip = adjacent_link_pairs(&robot);

        // At identity all links are at origin → spheres likely overlap
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // With skip pairs for adjacent links, self-collision should be filtered
        // (base-link1 adjacent, link1-link2 adjacent, link2-ee adjacent)
        // Only non-adjacent pairs like base-link2 or base-ee could collide
        // This tests the mechanism works — the actual result depends on geometry
        let _result = runtime.self_collision(&skip);
        // Just verifying it doesn't panic
    }

    #[test]
    fn coarse_fewer_spheres_than_fine() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let coarse = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let fine = RobotSphereModel::from_robot(&robot, &SphereGenConfig::fine());
        assert!(
            coarse.total_spheres() <= fine.total_spheres(),
            "Coarse ({}) should have <= spheres than fine ({})",
            coarse.total_spheres(),
            fine.total_spheres()
        );
    }

    #[test]
    fn collides_with_external() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot_default(&robot);
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // Obstacle far away → no collision
        let mut far_obstacle = SpheresSoA::new();
        far_obstacle.push(100.0, 100.0, 100.0, 0.1, 0);
        assert!(!runtime.collides_with(&far_obstacle));

        // Obstacle at origin → collision (base_link box is at origin)
        let mut near_obstacle = SpheresSoA::new();
        near_obstacle.push(0.0, 0.0, 0.0, 0.01, 0);
        if model.total_spheres() > 0 {
            // Should collide since base box is at origin
            assert!(runtime.collides_with(&near_obstacle));
        }
    }
}
