//! Perception visualization: point clouds, octree voxels, dynamic obstacles.
//!
//! Manages multiple named perception sources (lidar, depth camera, etc.),
//! each with independent visibility, color, and rendering style.

use crate::pipeline::LineVertex;

/// A named perception source with its data and display settings.
#[derive(Debug, Clone)]
pub struct PerceptionSource {
    pub name: String,
    pub source_type: SourceType,
    pub visible: bool,
    pub color: [f32; 4],
    pub point_count: usize,
}

/// Type of perception data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    PointCloud,
    Octree,
    DepthCamera,
}

/// Point cloud data ready for visualization.
#[derive(Debug, Clone)]
pub struct PointCloudViz {
    pub source_name: String,
    pub points: Vec<[f32; 3]>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub uniform_color: [f32; 4],
    pub point_size: f32,
    pub visible: bool,
}

impl PointCloudViz {
    pub fn new(name: &str, points: Vec<[f32; 3]>, color: [f32; 4]) -> Self {
        Self {
            source_name: name.to_string(),
            points,
            colors: None,
            uniform_color: color,
            point_size: 2.0,
            visible: true,
        }
    }

    pub fn with_per_point_colors(mut self, colors: Vec<[f32; 4]>) -> Self {
        self.colors = Some(colors);
        self
    }
}

/// Octree voxel data for visualization.
#[derive(Debug, Clone)]
pub struct OctreeViz {
    pub source_name: String,
    pub voxels: Vec<OctreeVoxel>,
    pub visible: bool,
    pub max_render_count: usize,
}

/// A single octree voxel.
#[derive(Debug, Clone, Copy)]
pub struct OctreeVoxel {
    pub center: [f32; 3],
    pub half_size: f32,
    pub occupancy: f32,
}

impl OctreeViz {
    pub fn new(name: &str, voxels: Vec<OctreeVoxel>) -> Self {
        Self {
            source_name: name.to_string(),
            voxels,
            visible: true,
            max_render_count: 50_000,
        }
    }
}

/// Dynamic obstacle in the scene.
#[derive(Debug, Clone)]
pub struct DynamicObstacle {
    pub name: String,
    pub shape: ObstacleShape,
    pub pose: [f32; 16], // column-major 4x4
    pub color: [f32; 4],
    pub visible: bool,
}

/// Obstacle shape types.
#[derive(Debug, Clone)]
pub enum ObstacleShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Cylinder { radius: f32, half_height: f32 },
}

/// Attached object that moves with a robot link.
#[derive(Debug, Clone)]
pub struct AttachedObject {
    pub name: String,
    pub shape: ObstacleShape,
    pub parent_link: String,
    pub grasp_transform: [f32; 16],
    pub color: [f32; 4],
}

/// Manages all perception data sources.
pub struct PerceptionManager {
    pub point_clouds: Vec<PointCloudViz>,
    pub octrees: Vec<OctreeViz>,
    pub obstacles: Vec<DynamicObstacle>,
    pub attached_objects: Vec<AttachedObject>,
}

impl Default for PerceptionManager {
    fn default() -> Self {
        Self {
            point_clouds: Vec::new(),
            octrees: Vec::new(),
            obstacles: Vec::new(),
            attached_objects: Vec::new(),
        }
    }
}

impl PerceptionManager {
    pub fn add_point_cloud(&mut self, cloud: PointCloudViz) {
        // Replace if same source name exists
        self.point_clouds.retain(|c| c.source_name != cloud.source_name);
        self.point_clouds.push(cloud);
    }

    pub fn add_octree(&mut self, octree: OctreeViz) {
        self.octrees.retain(|o| o.source_name != octree.source_name);
        self.octrees.push(octree);
    }

    pub fn add_obstacle(&mut self, obstacle: DynamicObstacle) {
        self.obstacles.retain(|o| o.name != obstacle.name);
        self.obstacles.push(obstacle);
    }

    pub fn remove_obstacle(&mut self, name: &str) {
        self.obstacles.retain(|o| o.name != name);
    }

    pub fn attach_object(&mut self, obj: AttachedObject) {
        self.attached_objects.retain(|a| a.name != obj.name);
        self.attached_objects.push(obj);
    }

    pub fn detach_object(&mut self, name: &str) -> Option<AttachedObject> {
        let idx = self.attached_objects.iter().position(|a| a.name == name)?;
        Some(self.attached_objects.remove(idx))
    }

    pub fn clear_source(&mut self, name: &str) {
        self.point_clouds.retain(|c| c.source_name != name);
        self.octrees.retain(|o| o.source_name != name);
    }

    /// List all active sources with metadata.
    pub fn sources(&self) -> Vec<PerceptionSource> {
        let mut sources = Vec::new();
        for cloud in &self.point_clouds {
            sources.push(PerceptionSource {
                name: cloud.source_name.clone(),
                source_type: SourceType::PointCloud,
                visible: cloud.visible,
                color: cloud.uniform_color,
                point_count: cloud.points.len(),
            });
        }
        for octree in &self.octrees {
            sources.push(PerceptionSource {
                name: octree.source_name.clone(),
                source_type: SourceType::Octree,
                visible: octree.visible,
                color: [0.3, 0.3, 0.8, 0.5],
                point_count: octree.voxels.len(),
            });
        }
        sources
    }

    /// Generate point cloud cross markers for all visible point clouds.
    ///
    /// Each point becomes 3 short perpendicular line segments (a small cross),
    /// making individual points visible in the line renderer.
    pub fn collect_point_cloud_lines(&self) -> Vec<LineVertex> {
        let mut lines = Vec::new();
        let cross_size = 0.005; // 5mm arms
        for cloud in &self.point_clouds {
            if !cloud.visible {
                continue;
            }
            for (pi, pt) in cloud.points.iter().enumerate() {
                let color = cloud.colors.as_ref()
                    .and_then(|c| c.get(pi).copied())
                    .unwrap_or(cloud.uniform_color);
                // X arm
                lines.push(LineVertex { position: [pt[0] - cross_size, pt[1], pt[2]], color });
                lines.push(LineVertex { position: [pt[0] + cross_size, pt[1], pt[2]], color });
                // Y arm
                lines.push(LineVertex { position: [pt[0], pt[1] - cross_size, pt[2]], color });
                lines.push(LineVertex { position: [pt[0], pt[1] + cross_size, pt[2]], color });
                // Z arm
                lines.push(LineVertex { position: [pt[0], pt[1], pt[2] - cross_size], color });
                lines.push(LineVertex { position: [pt[0], pt[1], pt[2] + cross_size], color });
            }
        }
        lines
    }

    /// Generate octree voxel wireframe lines for all visible octrees.
    pub fn collect_octree_lines(&self) -> Vec<LineVertex> {
        let mut lines = Vec::new();
        for octree in &self.octrees {
            if !octree.visible {
                continue;
            }
            let limit = octree.voxels.len().min(octree.max_render_count);
            for voxel in &octree.voxels[..limit] {
                let occ = voxel.occupancy;
                let color = [
                    0.2 + 0.6 * occ,
                    0.2 + 0.3 * (1.0 - occ),
                    0.8 * (1.0 - occ),
                    0.2 + 0.4 * occ,
                ];
                lines.extend(crate::collision_viz::voxel_wireframe_lines(
                    voxel.center,
                    voxel.half_size,
                    color,
                ));
            }
        }
        lines
    }
}

/// Draw perception sources panel.
#[cfg(feature = "visual")]
pub fn perception_panel_ui(ctx: &egui::Context, manager: &mut PerceptionManager) {
    egui::Window::new("Perception Sources")
        .default_pos([1040.0, 200.0])
        .default_width(220.0)
        .show(ctx, |ui| {
            if manager.point_clouds.is_empty() && manager.octrees.is_empty() {
                ui.colored_label(egui::Color32::GRAY, "No perception sources active.");
                return;
            }

            for cloud in &mut manager.point_clouds {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut cloud.visible, "");
                    ui.label(format!("{} ({} pts)", cloud.source_name, cloud.points.len()));
                });
            }

            for octree in &mut manager.octrees {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut octree.visible, "");
                    ui.label(format!(
                        "{} ({} voxels)",
                        octree.source_name,
                        octree.voxels.len()
                    ));
                });
            }

            ui.separator();

            // Obstacles
            if !manager.obstacles.is_empty() {
                ui.label(
                    egui::RichText::new("Dynamic Obstacles")
                        .strong(),
                );
                let mut to_remove = None;
                for (i, obs) in manager.obstacles.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut obs.visible, "");
                        ui.label(&obs.name);
                        if ui.button("X").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }
                if let Some(idx) = to_remove {
                    manager.obstacles.remove(idx);
                }
            }

            // Attached objects
            if !manager.attached_objects.is_empty() {
                ui.separator();
                ui.label(
                    egui::RichText::new("Attached Objects")
                        .strong()
                        .color(egui::Color32::YELLOW),
                );
                for obj in &manager.attached_objects {
                    ui.label(format!("  {} → {}", obj.name, obj.parent_link));
                }
            }
        });
}

/// Scene state for save/load.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SceneSnapshot {
    pub robot_name: String,
    pub joint_values: Vec<f64>,
    pub obstacles: Vec<SceneObstacleData>,
    pub grid_visible: bool,
    pub axes_visible: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SceneObstacleData {
    pub name: String,
    pub shape_type: String,
    pub dimensions: Vec<f64>,
    pub position: [f64; 3],
}

impl SceneSnapshot {
    /// Save to JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load from JSON file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_remove_point_cloud() {
        let mut mgr = PerceptionManager::default();
        mgr.add_point_cloud(PointCloudViz::new(
            "lidar",
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [0.2, 0.8, 0.2, 1.0],
        ));
        assert_eq!(mgr.sources().len(), 1);
        assert_eq!(mgr.sources()[0].point_count, 2);

        mgr.clear_source("lidar");
        assert!(mgr.sources().is_empty());
    }

    #[test]
    fn add_replaces_same_name() {
        let mut mgr = PerceptionManager::default();
        mgr.add_point_cloud(PointCloudViz::new("lidar", vec![[0.0; 3]], [1.0; 4]));
        mgr.add_point_cloud(PointCloudViz::new("lidar", vec![[0.0; 3]; 5], [1.0; 4]));
        assert_eq!(mgr.point_clouds.len(), 1);
        assert_eq!(mgr.point_clouds[0].points.len(), 5);
    }

    #[test]
    fn obstacle_lifecycle() {
        let mut mgr = PerceptionManager::default();
        mgr.add_obstacle(DynamicObstacle {
            name: "box1".into(),
            shape: ObstacleShape::Box { half_extents: [0.1, 0.1, 0.1] },
            pose: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            color: [0.5, 0.5, 0.5, 1.0],
            visible: true,
        });
        assert_eq!(mgr.obstacles.len(), 1);
        mgr.remove_obstacle("box1");
        assert!(mgr.obstacles.is_empty());
    }

    #[test]
    fn attach_detach() {
        let mut mgr = PerceptionManager::default();
        mgr.attach_object(AttachedObject {
            name: "cup".into(),
            shape: ObstacleShape::Cylinder { radius: 0.04, half_height: 0.06 },
            parent_link: "ee_link".into(),
            grasp_transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            color: [0.8, 0.4, 0.1, 1.0],
        });
        assert_eq!(mgr.attached_objects.len(), 1);
        let detached = mgr.detach_object("cup");
        assert!(detached.is_some());
        assert!(mgr.attached_objects.is_empty());
    }

    #[test]
    fn scene_snapshot_roundtrip() {
        let snapshot = SceneSnapshot {
            robot_name: "ur5e".into(),
            joint_values: vec![0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
            obstacles: vec![SceneObstacleData {
                name: "table".into(),
                shape_type: "box".into(),
                dimensions: vec![0.5, 0.5, 0.02],
                position: [0.0, 0.0, -0.01],
            }],
            grid_visible: true,
            axes_visible: true,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let loaded: SceneSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.robot_name, "ur5e");
        assert_eq!(loaded.obstacles.len(), 1);
    }
}
