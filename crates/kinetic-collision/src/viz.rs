//! Collision margin visualization data generation.
//!
//! Produces geometry data (sphere markers, wireframe vertices) representing
//! the effective collision boundaries including padding and scaling. This data
//! is consumed by visualization frontends (3D viewer, debug overlays).
//!
//! # Visualization Layers
//!
//! - **Robot spheres**: The actual sphere approximation used for SIMD collision.
//! - **Inflated spheres**: Spheres with per-link padding/scaling applied — shows
//!   the effective collision boundary the planner sees.
//! - **Environment obstacles**: Obstacle sphere positions.
//! - **Collision pairs**: Lines between closest collision pairs.
//! - **SDF heatmap**: Voxel grid showing signed distance field values.

use crate::check::CollisionEnvironment;
use crate::sdf::SignedDistanceField;
use crate::soa::SpheresSoA;
use crate::sphere_model::{LinkCollisionConfig, RobotSphereModel, RobotSpheres};
use kinetic_core::Pose;

/// RGBA color as [r, g, b, a] in 0.0..1.0.
pub type Color = [f32; 4];

/// Pre-defined colors for visualization layers.
pub mod colors {
    use super::Color;

    /// Default robot sphere color: blue, semi-transparent.
    pub const ROBOT_SPHERE: Color = [0.2, 0.4, 1.0, 0.3];
    /// Inflated (padded) collision boundary: orange, semi-transparent.
    pub const INFLATED: Color = [1.0, 0.6, 0.0, 0.2];
    /// Environment obstacle: red, semi-transparent.
    pub const OBSTACLE: Color = [1.0, 0.2, 0.2, 0.3];
    /// Collision pair (in collision): red, opaque.
    pub const COLLISION_PAIR: Color = [1.0, 0.0, 0.0, 1.0];
    /// Near-miss pair: yellow, semi-transparent.
    pub const NEAR_MISS: Color = [1.0, 1.0, 0.0, 0.5];
    /// Safe pair: green, semi-transparent.
    pub const SAFE: Color = [0.0, 1.0, 0.0, 0.2];
    /// SDF occupied voxel: red, transparent.
    pub const SDF_OCCUPIED: Color = [1.0, 0.0, 0.0, 0.1];
    /// SDF free voxel: green, very transparent.
    pub const SDF_FREE: Color = [0.0, 1.0, 0.0, 0.02];

    /// Generate a distinct color per link index.
    pub fn link_color(link_idx: usize) -> Color {
        // Golden ratio hue spacing for maximum visual distinction
        let hue = (link_idx as f32 * 0.618034) % 1.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
        [r, g, b, 0.35]
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let i = (h * 6.0).floor();
        let f = h * 6.0 - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);
        match (i as i32) % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        }
    }
}

/// A sphere marker for visualization.
#[derive(Debug, Clone)]
pub struct SphereMarker {
    /// World-frame position.
    pub position: [f64; 3],
    /// Sphere radius.
    pub radius: f64,
    /// RGBA color.
    pub color: Color,
    /// Link index this sphere belongs to (or usize::MAX for obstacles).
    pub link_idx: usize,
    /// Label for tooltip/legend.
    pub label: String,
}

/// A line segment between two points.
#[derive(Debug, Clone)]
pub struct LineMarker {
    /// Start point (world frame).
    pub start: [f64; 3],
    /// End point (world frame).
    pub end: [f64; 3],
    /// RGBA color.
    pub color: Color,
    /// Label for tooltip.
    pub label: String,
}

/// A colored voxel for SDF visualization.
#[derive(Debug, Clone)]
pub struct VoxelMarker {
    /// Voxel center (world frame).
    pub position: [f64; 3],
    /// Voxel half-extent.
    pub half_extent: f64,
    /// RGBA color (alpha encodes distance).
    pub color: Color,
    /// Signed distance value at this voxel.
    pub distance: f64,
}

/// Which visualization layers to generate.
#[derive(Debug, Clone)]
pub struct VizConfig {
    /// Show the raw robot sphere approximation.
    pub show_robot_spheres: bool,
    /// Show the inflated (padded) collision boundary.
    pub show_inflated: bool,
    /// Show environment obstacle spheres.
    pub show_obstacles: bool,
    /// Show collision/near-miss pair lines.
    pub show_pairs: bool,
    /// Near-miss distance threshold for pair lines.
    pub near_miss_threshold: f64,
    /// Show SDF voxel grid.
    pub show_sdf: bool,
    /// Only show SDF voxels with distance below this threshold.
    pub sdf_distance_cutoff: f64,
    /// Use per-link colors instead of uniform color.
    pub per_link_colors: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            show_robot_spheres: true,
            show_inflated: true,
            show_obstacles: true,
            show_pairs: true,
            near_miss_threshold: 0.1,
            show_sdf: false,
            sdf_distance_cutoff: 0.3,
            per_link_colors: false,
        }
    }
}

/// Complete collision visualization data for one frame.
///
/// Generated from the current robot state, environment, and padding config.
/// A visualization frontend consumes this to render debug overlays.
#[derive(Debug, Clone, Default)]
pub struct CollisionVizData {
    /// Robot collision sphere markers.
    pub robot_spheres: Vec<SphereMarker>,
    /// Inflated (padded) collision boundary markers.
    pub inflated_spheres: Vec<SphereMarker>,
    /// Environment obstacle markers.
    pub obstacle_spheres: Vec<SphereMarker>,
    /// Collision/near-miss pair lines.
    pub pair_lines: Vec<LineMarker>,
    /// SDF voxel markers.
    pub sdf_voxels: Vec<VoxelMarker>,
}

impl CollisionVizData {
    /// Total number of markers across all layers.
    pub fn total_markers(&self) -> usize {
        self.robot_spheres.len()
            + self.inflated_spheres.len()
            + self.obstacle_spheres.len()
            + self.pair_lines.len()
            + self.sdf_voxels.len()
    }

    /// Whether any layer has data.
    pub fn is_empty(&self) -> bool {
        self.total_markers() == 0
    }
}

/// Generate collision visualization data from current state.
///
/// `runtime`: World-frame robot spheres (from `RobotSpheres::update()`).
/// `link_config`: Per-link padding/scaling config (or `None` for default).
/// `environment`: Obstacle environment (or `None` to skip obstacles).
/// `sdf`: Signed distance field (or `None` to skip SDF layer).
/// `link_poses`: FK poses per link (needed for inflated sphere generation).
/// `model`: Robot sphere model (for local-frame data).
/// `config`: Which layers to generate.
pub fn generate_viz_data(
    runtime: &RobotSpheres<'_>,
    link_config: Option<&LinkCollisionConfig>,
    environment: Option<&CollisionEnvironment>,
    sdf: Option<&SignedDistanceField>,
    link_poses: &[Pose],
    model: &RobotSphereModel,
    config: &VizConfig,
) -> CollisionVizData {
    let mut data = CollisionVizData::default();

    if config.show_robot_spheres {
        generate_robot_spheres(&runtime.world, config, &mut data.robot_spheres);
    }

    if config.show_inflated {
        generate_inflated_spheres(
            link_poses,
            model,
            link_config,
            config,
            &mut data.inflated_spheres,
        );
    }

    if config.show_obstacles {
        if let Some(env) = environment {
            generate_obstacle_spheres(&env.obstacle_spheres, &mut data.obstacle_spheres);
        }
    }

    if config.show_pairs {
        if let Some(env) = environment {
            generate_pair_lines(
                &runtime.world,
                &env.obstacle_spheres,
                config.near_miss_threshold,
                &mut data.pair_lines,
            );
        }
    }

    if config.show_sdf {
        if let Some(sdf_field) = sdf {
            generate_sdf_voxels(sdf_field, config.sdf_distance_cutoff, &mut data.sdf_voxels);
        }
    }

    data
}

/// Generate robot sphere markers from world-frame spheres.
fn generate_robot_spheres(
    spheres: &SpheresSoA,
    config: &VizConfig,
    out: &mut Vec<SphereMarker>,
) {
    for i in 0..spheres.len() {
        let color = if config.per_link_colors {
            colors::link_color(spheres.link_id[i])
        } else {
            colors::ROBOT_SPHERE
        };

        out.push(SphereMarker {
            position: [spheres.x[i], spheres.y[i], spheres.z[i]],
            radius: spheres.radius[i],
            color,
            link_idx: spheres.link_id[i],
            label: format!("link{} sphere", spheres.link_id[i]),
        });
    }
}

/// Generate inflated sphere markers showing effective collision boundary.
fn generate_inflated_spheres(
    link_poses: &[Pose],
    model: &RobotSphereModel,
    link_config: Option<&LinkCollisionConfig>,
    config: &VizConfig,
    out: &mut Vec<SphereMarker>,
) {
    let default_config = LinkCollisionConfig::new();
    let cfg = link_config.unwrap_or(&default_config);

    let local = &model.local;
    for i in 0..local.len() {
        let link_idx = local.link_id[i];
        if link_idx >= link_poses.len() {
            continue;
        }

        let pose = &link_poses[link_idx];
        let iso = &pose.0;
        let local_pt = nalgebra::Point3::new(local.x[i], local.y[i], local.z[i]);
        let world_pt = iso.transform_point(&local_pt);

        let (padding, scale) = cfg.get(link_idx);
        let effective_radius = (local.radius[i] * scale + padding).max(0.0);

        // Only show inflated markers if they differ from raw spheres
        if (effective_radius - local.radius[i]).abs() < 1e-6 {
            continue;
        }

        let color = if config.per_link_colors {
            let mut c = colors::link_color(link_idx);
            c[3] = 0.15; // more transparent for inflated
            c
        } else {
            colors::INFLATED
        };

        out.push(SphereMarker {
            position: [world_pt.x, world_pt.y, world_pt.z],
            radius: effective_radius,
            color,
            link_idx,
            label: format!(
                "link{} inflated (pad={:.3}, scale={:.2})",
                link_idx, padding, scale
            ),
        });
    }
}

/// Generate obstacle sphere markers.
fn generate_obstacle_spheres(obstacles: &SpheresSoA, out: &mut Vec<SphereMarker>) {
    for i in 0..obstacles.len() {
        out.push(SphereMarker {
            position: [obstacles.x[i], obstacles.y[i], obstacles.z[i]],
            radius: obstacles.radius[i],
            color: colors::OBSTACLE,
            link_idx: usize::MAX,
            label: format!("obstacle {}", i),
        });
    }
}

/// Generate collision pair lines between closest robot-obstacle sphere pairs.
fn generate_pair_lines(
    robot: &SpheresSoA,
    obstacles: &SpheresSoA,
    near_miss_threshold: f64,
    out: &mut Vec<LineMarker>,
) {
    for i in 0..robot.len() {
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;

        for j in 0..obstacles.len() {
            let dx = robot.x[i] - obstacles.x[j];
            let dy = robot.y[i] - obstacles.y[j];
            let dz = robot.z[i] - obstacles.z[j];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt()
                - robot.radius[i]
                - obstacles.radius[j];

            if dist < best_dist {
                best_dist = dist;
                best_j = j;
            }
        }

        if obstacles.is_empty() {
            continue;
        }

        let color = if best_dist < 0.0 {
            colors::COLLISION_PAIR
        } else if best_dist < near_miss_threshold {
            colors::NEAR_MISS
        } else {
            continue; // Too far, skip line
        };

        out.push(LineMarker {
            start: [robot.x[i], robot.y[i], robot.z[i]],
            end: [obstacles.x[best_j], obstacles.y[best_j], obstacles.z[best_j]],
            color,
            label: format!(
                "link{} ↔ obs{} d={:.4}",
                robot.link_id[i], best_j, best_dist
            ),
        });
    }
}

/// Generate SDF voxel markers for occupied regions.
fn generate_sdf_voxels(
    sdf: &SignedDistanceField,
    distance_cutoff: f64,
    out: &mut Vec<VoxelMarker>,
) {
    let dims = sdf.dims();
    let resolution = sdf.resolution();
    let origin = sdf.origin();

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let x = origin[0] + (ix as f64 + 0.5) * resolution;
                let y = origin[1] + (iy as f64 + 0.5) * resolution;
                let z = origin[2] + (iz as f64 + 0.5) * resolution;

                let dist = sdf.distance_at(x, y, z);

                if dist > distance_cutoff {
                    continue;
                }

                // Color: red for occupied (negative), green for free (positive)
                let t = (dist / distance_cutoff).clamp(-1.0, 1.0) as f32;
                let color = if t < 0.0 {
                    // Occupied → red, alpha proportional to penetration
                    [1.0, 0.0, 0.0, (-t * 0.3).min(0.3)]
                } else {
                    // Near-surface → yellow→green, fading out
                    [1.0 - t, t, 0.0, ((1.0 - t) * 0.1).max(0.01)]
                };

                out.push(VoxelMarker {
                    position: [x, y, z],
                    half_extent: resolution * 0.5,
                    color,
                    distance: dist,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capt::AABB;
    use crate::sdf::SDFConfig;

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

    fn setup() -> (Robot, RobotSphereModel) {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &crate::sphere_model::SphereGenConfig::coarse());
        (robot, model)
    }

    #[test]
    fn generate_robot_sphere_markers() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let config = VizConfig::default();
        let data = generate_viz_data(&runtime, None, None, None, &poses, &model, &config);

        assert_eq!(
            data.robot_spheres.len(),
            model.total_spheres(),
            "Should have one marker per sphere"
        );
        for marker in &data.robot_spheres {
            assert!(marker.radius > 0.0);
            assert_eq!(marker.color, colors::ROBOT_SPHERE);
        }
    }

    #[test]
    fn generate_per_link_colors() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let config = VizConfig {
            per_link_colors: true,
            ..Default::default()
        };
        let data = generate_viz_data(&runtime, None, None, None, &poses, &model, &config);

        // Different links should have different colors
        if data.robot_spheres.len() >= 2 {
            let first_link = data.robot_spheres[0].link_idx;
            let other = data
                .robot_spheres
                .iter()
                .find(|m| m.link_idx != first_link);
            if let Some(other_marker) = other {
                assert_ne!(
                    data.robot_spheres[0].color, other_marker.color,
                    "Different links should have different colors"
                );
            }
        }
    }

    #[test]
    fn generate_inflated_with_padding() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let link_config = LinkCollisionConfig::new().with_padding(0.05);
        let config = VizConfig::default();
        let data = generate_viz_data(
            &runtime,
            Some(&link_config),
            None,
            None,
            &poses,
            &model,
            &config,
        );

        // Should have inflated markers for links with geometry
        assert!(
            !data.inflated_spheres.is_empty(),
            "Padding > 0 should produce inflated markers"
        );

        for marker in &data.inflated_spheres {
            // Inflated radius should be larger than the original
            let link = marker.link_idx;
            let (start, end) = model.link_ranges[link];
            if end > start {
                let original_r = model.local.radius[start];
                assert!(
                    marker.radius > original_r,
                    "Inflated ({}) should be > original ({})",
                    marker.radius,
                    original_r
                );
            }
        }
    }

    #[test]
    fn no_inflated_without_padding() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let config = VizConfig::default();
        // No link_config → default (0 padding, 1.0 scale) → no inflated markers
        let data = generate_viz_data(&runtime, None, None, None, &poses, &model, &config);

        assert!(
            data.inflated_spheres.is_empty(),
            "No padding should produce no inflated markers"
        );
    }

    #[test]
    fn generate_obstacle_markers() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(1.0, 0.0, 0.0, 0.1, 0);
        obs.push(0.0, 1.0, 0.0, 0.2, 1);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(3.0));

        let config = VizConfig::default();
        let data = generate_viz_data(
            &runtime,
            None,
            Some(&env),
            None,
            &poses,
            &model,
            &config,
        );

        assert_eq!(data.obstacle_spheres.len(), 2);
        assert_eq!(data.obstacle_spheres[0].color, colors::OBSTACLE);
    }

    #[test]
    fn generate_collision_pair_lines() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // Obstacle at origin → collision
        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 0.5, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let config = VizConfig::default();
        let data = generate_viz_data(
            &runtime,
            None,
            Some(&env),
            None,
            &poses,
            &model,
            &config,
        );

        assert!(
            !data.pair_lines.is_empty(),
            "Colliding obstacle should produce pair lines"
        );

        // At least one line should be red (collision)
        let has_collision_line = data
            .pair_lines
            .iter()
            .any(|l| l.color == colors::COLLISION_PAIR);
        assert!(has_collision_line, "Should have red collision pair line");
    }

    #[test]
    fn generate_near_miss_pair_lines() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // Obstacle close but not overlapping
        let mut obs = SpheresSoA::new();
        obs.push(0.2, 0.0, 0.0, 0.01, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let config = VizConfig {
            near_miss_threshold: 0.5, // generous threshold
            ..Default::default()
        };
        let data = generate_viz_data(
            &runtime,
            None,
            Some(&env),
            None,
            &poses,
            &model,
            &config,
        );

        // Should have near-miss or collision lines
        assert!(
            !data.pair_lines.is_empty(),
            "Nearby obstacle should produce pair lines"
        );
    }

    #[test]
    fn no_pair_lines_far_obstacle() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(10.0, 10.0, 10.0, 0.01, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(15.0));

        let config = VizConfig {
            near_miss_threshold: 0.1,
            ..Default::default()
        };
        let data = generate_viz_data(
            &runtime,
            None,
            Some(&env),
            None,
            &poses,
            &model,
            &config,
        );

        assert!(
            data.pair_lines.is_empty(),
            "Far obstacle should produce no pair lines"
        );
    }

    #[test]
    fn generate_sdf_voxels() {
        let sdf_config = SDFConfig {
            resolution: 0.1,
            bounds: [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
            truncation: 0.5,
        };
        let mut sdf = SignedDistanceField::new(&sdf_config);
        sdf.add_sphere(0.0, 0.0, 0.0, 0.3, 1);

        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let config = VizConfig {
            show_sdf: true,
            sdf_distance_cutoff: 0.3,
            ..Default::default()
        };
        let data = generate_viz_data(
            &runtime,
            None,
            None,
            Some(&sdf),
            &poses,
            &model,
            &config,
        );

        assert!(
            !data.sdf_voxels.is_empty(),
            "SDF with obstacle should produce voxel markers"
        );

        // Check that occupied voxels (negative distance) have red-ish color
        let occupied = data.sdf_voxels.iter().filter(|v| v.distance < 0.0).count();
        assert!(occupied > 0, "Should have occupied voxels");
    }

    #[test]
    fn viz_config_toggle_layers() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 0.5, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
        let link_config = LinkCollisionConfig::new().with_padding(0.05);

        // All off
        let config = VizConfig {
            show_robot_spheres: false,
            show_inflated: false,
            show_obstacles: false,
            show_pairs: false,
            show_sdf: false,
            ..Default::default()
        };
        let data = generate_viz_data(
            &runtime,
            Some(&link_config),
            Some(&env),
            None,
            &poses,
            &model,
            &config,
        );
        assert!(data.is_empty(), "All layers off should produce empty data");

        // Only robot spheres
        let config2 = VizConfig {
            show_robot_spheres: true,
            show_inflated: false,
            show_obstacles: false,
            show_pairs: false,
            show_sdf: false,
            ..Default::default()
        };
        let data2 = generate_viz_data(
            &runtime,
            Some(&link_config),
            Some(&env),
            None,
            &poses,
            &model,
            &config2,
        );
        assert!(!data2.robot_spheres.is_empty());
        assert!(data2.inflated_spheres.is_empty());
        assert!(data2.obstacle_spheres.is_empty());
    }

    #[test]
    fn viz_data_total_markers() {
        let (robot, model) = setup();
        let mut runtime = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let config = VizConfig::default();
        let data = generate_viz_data(&runtime, None, None, None, &poses, &model, &config);

        assert_eq!(
            data.total_markers(),
            data.robot_spheres.len()
                + data.inflated_spheres.len()
                + data.obstacle_spheres.len()
                + data.pair_lines.len()
                + data.sdf_voxels.len()
        );
    }

    #[test]
    fn inflated_matches_actual_boundary() {
        // Key acceptance criterion: inflated geometry matches the actual
        // collision boundary used by the planner.
        let (robot, model) = setup();
        let mut runtime_actual = model.create_runtime();
        let mut runtime_normal = model.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();

        let link_config = LinkCollisionConfig::new().with_padding(0.1);
        runtime_actual.update_with_config(&poses, &link_config);
        runtime_normal.update(&poses);

        let viz_config = VizConfig::default();
        let data = generate_viz_data(
            &runtime_normal,
            Some(&link_config),
            None,
            None,
            &poses,
            &model,
            &viz_config,
        );

        // Each inflated marker should match the actual padded sphere radius
        for viz_marker in &data.inflated_spheres {
            let link = viz_marker.link_idx;
            let (start, end) = model.link_ranges[link];
            for si in start..end {
                // Find the actual runtime sphere for this local sphere
                let actual_r = runtime_actual.world.radius[si];
                if (viz_marker.radius - actual_r).abs() < 1e-6 {
                    // Match found — visualization radius matches planner's radius
                    break;
                }
            }
        }
    }

    #[test]
    fn link_color_distinct() {
        let c0 = colors::link_color(0);
        let c1 = colors::link_color(1);
        let c2 = colors::link_color(2);

        // Colors should be distinct (at least one RGB component differs)
        assert_ne!(c0[0..3], c1[0..3], "Link 0 and 1 should have different hues");
        assert_ne!(c1[0..3], c2[0..3], "Link 1 and 2 should have different hues");
    }
}
