//! Collision visualization rendering: converts collision debug data to line/mesh commands.
//!
//! Bridges the data structures from `kinetic-collision/src/viz.rs` (SphereMarker,
//! LineMarker, VoxelMarker) to viewer render commands (DrawLine, DrawMesh).
//! Each collision viz layer can be toggled independently.

use crate::pipeline::LineVertex;

/// Configuration for which collision visualization layers to show.
#[derive(Debug, Clone)]
pub struct CollisionVizConfig {
    pub show_robot_spheres: bool,
    pub show_inflated: bool,
    pub show_obstacles: bool,
    pub show_pair_lines: bool,
    pub show_sdf_voxels: bool,
    pub near_miss_threshold: f32,
    pub sdf_distance_cutoff: f32,
}

impl Default for CollisionVizConfig {
    fn default() -> Self {
        Self {
            show_robot_spheres: false,
            show_inflated: false,
            show_obstacles: false,
            show_pair_lines: true,
            show_sdf_voxels: false,
            near_miss_threshold: 0.1,
            sdf_distance_cutoff: 0.3,
        }
    }
}

/// A sphere marker for visualization (mirrors kinetic-collision's SphereMarker).
#[derive(Debug, Clone)]
pub struct SphereViz {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

/// A line between two points for collision pair visualization.
#[derive(Debug, Clone)]
pub struct PairLineViz {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
}

/// A voxel for SDF visualization.
#[derive(Debug, Clone)]
pub struct VoxelViz {
    pub center: [f32; 3],
    pub half_extent: f32,
    pub color: [f32; 4],
}

/// All collision visualization data for one frame.
#[derive(Debug, Clone, Default)]
pub struct CollisionVizData {
    pub robot_spheres: Vec<SphereViz>,
    pub inflated_spheres: Vec<SphereViz>,
    pub obstacle_spheres: Vec<SphereViz>,
    pub pair_lines: Vec<PairLineViz>,
    pub sdf_voxels: Vec<VoxelViz>,
}

/// Generate line vertices for collision pair lines.
pub fn pair_lines_to_vertices(pairs: &[PairLineViz]) -> Vec<LineVertex> {
    let mut lines = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        lines.push(LineVertex {
            position: pair.start,
            color: pair.color,
        });
        lines.push(LineVertex {
            position: pair.end,
            color: pair.color,
        });
    }
    lines
}

/// Generate wireframe lines for a sphere (3 orthogonal circles).
pub fn sphere_wireframe_lines(center: [f32; 3], radius: f32, color: [f32; 4], segments: usize) -> Vec<LineVertex> {
    let mut lines = Vec::with_capacity(segments * 2 * 3);
    let c = center;

    for axis in 0..3 {
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

/// Generate wireframe lines for a voxel (cube edges).
pub fn voxel_wireframe_lines(center: [f32; 3], half: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let c = center;
    let h = half;
    let corners = [
        [c[0] - h, c[1] - h, c[2] - h],
        [c[0] + h, c[1] - h, c[2] - h],
        [c[0] + h, c[1] + h, c[2] - h],
        [c[0] - h, c[1] + h, c[2] - h],
        [c[0] - h, c[1] - h, c[2] + h],
        [c[0] + h, c[1] - h, c[2] + h],
        [c[0] + h, c[1] + h, c[2] + h],
        [c[0] - h, c[1] + h, c[2] + h],
    ];

    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom
        (4, 5), (5, 6), (6, 7), (7, 4), // top
        (0, 4), (1, 5), (2, 6), (3, 7), // verticals
    ];

    let mut lines = Vec::with_capacity(24);
    for (a, b) in &edges {
        lines.push(LineVertex { position: corners[*a], color });
        lines.push(LineVertex { position: corners[*b], color });
    }
    lines
}

/// Collect all collision viz line vertices based on config.
pub fn collect_collision_lines(data: &CollisionVizData, config: &CollisionVizConfig) -> Vec<LineVertex> {
    let mut lines = Vec::new();

    if config.show_robot_spheres {
        for s in &data.robot_spheres {
            lines.extend(sphere_wireframe_lines(s.center, s.radius, s.color, 12));
        }
    }

    if config.show_inflated {
        for s in &data.inflated_spheres {
            lines.extend(sphere_wireframe_lines(s.center, s.radius, s.color, 12));
        }
    }

    if config.show_obstacles {
        for s in &data.obstacle_spheres {
            lines.extend(sphere_wireframe_lines(s.center, s.radius, s.color, 12));
        }
    }

    if config.show_pair_lines {
        lines.extend(pair_lines_to_vertices(&data.pair_lines));
    }

    if config.show_sdf_voxels {
        for v in &data.sdf_voxels {
            lines.extend(voxel_wireframe_lines(v.center, v.half_extent, v.color));
        }
    }

    lines
}

/// Draw collision debug egui panel.
#[cfg(feature = "visual")]
pub fn collision_debug_panel_ui(ctx: &egui::Context, config: &mut CollisionVizConfig) {
    egui::Window::new("Collision Debug")
        .default_pos([1040.0, 400.0])
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.checkbox(&mut config.show_robot_spheres, "Robot spheres");
            ui.checkbox(&mut config.show_inflated, "Inflated boundaries");
            ui.checkbox(&mut config.show_obstacles, "Obstacles");
            ui.checkbox(&mut config.show_pair_lines, "Collision pairs");
            ui.checkbox(&mut config.show_sdf_voxels, "SDF voxels");

            ui.separator();

            ui.add(
                egui::Slider::new(&mut config.near_miss_threshold, 0.01..=1.0)
                    .text("Near-miss (m)")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut config.sdf_distance_cutoff, 0.05..=2.0)
                    .text("SDF cutoff (m)")
                    .logarithmic(true),
            );

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Show All").clicked() {
                    config.show_robot_spheres = true;
                    config.show_inflated = true;
                    config.show_obstacles = true;
                    config.show_pair_lines = true;
                    config.show_sdf_voxels = true;
                }
                if ui.button("Hide All").clicked() {
                    config.show_robot_spheres = false;
                    config.show_inflated = false;
                    config.show_obstacles = false;
                    config.show_pair_lines = false;
                    config.show_sdf_voxels = false;
                }
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_lines_generate_vertices() {
        let pairs = vec![PairLineViz {
            start: [0.0, 0.0, 0.0],
            end: [1.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        }];
        let verts = pair_lines_to_vertices(&pairs);
        assert_eq!(verts.len(), 2);
    }

    #[test]
    fn sphere_wireframe_generates_lines() {
        let lines = sphere_wireframe_lines([0.0, 0.0, 0.0], 0.1, [1.0, 0.0, 0.0, 0.3], 8);
        // 3 circles x 8 segments x 2 vertices = 48
        assert_eq!(lines.len(), 48);
    }

    #[test]
    fn voxel_wireframe_generates_24_vertices() {
        let lines = voxel_wireframe_lines([0.0, 0.0, 0.0], 0.05, [1.0, 0.0, 0.0, 0.5]);
        // 12 edges x 2 vertices = 24
        assert_eq!(lines.len(), 24);
    }

    #[test]
    fn collect_respects_config() {
        let data = CollisionVizData {
            robot_spheres: vec![SphereViz {
                center: [0.0, 0.0, 0.0],
                radius: 0.1,
                color: [0.2, 0.4, 1.0, 0.3],
            }],
            pair_lines: vec![PairLineViz {
                start: [0.0, 0.0, 0.0],
                end: [1.0, 0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            }],
            ..Default::default()
        };

        // Default config: pair_lines on, robot_spheres off
        let config = CollisionVizConfig::default();
        let lines = collect_collision_lines(&data, &config);
        assert_eq!(lines.len(), 2); // just pair lines

        // Enable robot spheres
        let mut config2 = config.clone();
        config2.show_robot_spheres = true;
        let lines2 = collect_collision_lines(&data, &config2);
        assert!(lines2.len() > 2); // pair lines + sphere wireframes
    }
}
