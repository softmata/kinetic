//! Interactive planning interface: markers, goal setting, planning panel, constraints.
//!
//! Provides the data model and interaction logic for interactive planning.
//! This module is UI-framework-agnostic — it produces interaction state that
//! egui/imgui/custom UI can consume and render.

use std::collections::HashMap;
use std::time::Duration;

use nalgebra::Isometry3;
use kinetic_core::{Goal, Pose};

// ═══════════════════════════════════════════════════════════════════════════
// Interactive Markers
// ═══════════════════════════════════════════════════════════════════════════

/// An interactive 3D marker that can be dragged to set goals.
#[derive(Debug, Clone)]
pub struct InteractiveMarker {
    pub id: String,
    pub pose: Isometry3<f64>,
    pub marker_type: MarkerType,
    pub scale: f64,
    pub visible: bool,
    pub color: [f32; 4],
    /// Which axes can be dragged.
    pub interaction_mode: InteractionMode,
}

/// Type of interactive marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerType {
    /// 6-DOF pose marker (translate + rotate).
    Pose6DOF,
    /// Position-only marker (translate only).
    Position3D,
    /// Single-axis marker (translate along one axis).
    SingleAxis,
    /// Joint slider marker.
    JointSlider,
    /// Point marker (no interaction).
    Point,
}

/// Which degrees of freedom are interactive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionMode {
    /// All 6 DOF (translate + rotate).
    Full6DOF,
    /// Translate only.
    TranslateOnly,
    /// Rotate only.
    RotateOnly,
    /// No interaction (display only).
    None,
}

impl InteractiveMarker {
    /// Create a 6-DOF pose marker at a given position.
    pub fn pose_marker(id: &str, pose: Isometry3<f64>) -> Self {
        Self {
            id: id.to_string(),
            pose,
            marker_type: MarkerType::Pose6DOF,
            scale: 0.1,
            visible: true,
            color: [0.2, 0.6, 1.0, 0.8],
            interaction_mode: InteractionMode::Full6DOF,
        }
    }

    /// Create a position-only marker.
    pub fn position_marker(id: &str, position: [f64; 3]) -> Self {
        Self {
            id: id.to_string(),
            pose: Isometry3::translation(position[0], position[1], position[2]),
            marker_type: MarkerType::Position3D,
            scale: 0.08,
            visible: true,
            color: [1.0, 0.4, 0.1, 0.8],
            interaction_mode: InteractionMode::TranslateOnly,
        }
    }

    /// Get the position.
    pub fn position(&self) -> [f64; 3] {
        let t = self.pose.translation.vector;
        [t.x, t.y, t.z]
    }

    /// Convert to a Goal::Pose.
    pub fn to_goal(&self) -> Goal {
        Goal::Pose(Pose(self.pose))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Planning Request Panel
// ═══════════════════════════════════════════════════════════════════════════

/// State for the planning request UI panel.
#[derive(Debug, Clone)]
pub struct PlanningPanel {
    /// Selected planner.
    pub planner_id: String,
    /// Available planners.
    pub available_planners: Vec<String>,
    /// Planning time limit.
    pub planning_time: f64,
    /// Velocity scaling.
    pub velocity_scale: f64,
    /// Acceleration scaling.
    pub acceleration_scale: f64,
    /// Number of planning attempts.
    pub num_attempts: usize,
    /// Whether to smooth the result.
    pub smooth: bool,
    /// Whether to time-parameterize.
    pub time_parameterize: bool,
    /// Goal type selection.
    pub goal_type: GoalType,
    /// Status message.
    pub status: PlanningStatus,
    /// Last planning time.
    pub last_planning_time: Option<Duration>,
}

/// Goal type for the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalType {
    Joint,
    Pose,
    Named,
}

/// Planning status for the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningStatus {
    Idle,
    Planning,
    Succeeded,
    Failed(String),
}

impl Default for PlanningPanel {
    fn default() -> Self {
        Self {
            planner_id: "rrt_connect".into(),
            available_planners: vec![
                "rrt_connect".into(), "rrt_star".into(), "bi_rrt_star".into(),
                "prm".into(), "est".into(), "kpiece".into(), "bitrrt".into(),
            ],
            planning_time: 5.0,
            velocity_scale: 1.0,
            acceleration_scale: 1.0,
            num_attempts: 1,
            smooth: true,
            time_parameterize: true,
            goal_type: GoalType::Joint,
            status: PlanningStatus::Idle,
            last_planning_time: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constraint Visualization
// ═══════════════════════════════════════════════════════════════════════════

/// Visual representation of a planning constraint.
#[derive(Debug, Clone)]
pub struct ConstraintViz {
    pub name: String,
    pub constraint_type: ConstraintVizType,
    pub visible: bool,
    pub color: [f32; 4],
}

/// Types of constraint visualizations.
#[derive(Debug, Clone)]
pub enum ConstraintVizType {
    /// Orientation cone: axis + tolerance angle.
    OrientationCone { axis: [f64; 3], tolerance: f64, link_name: String },
    /// Position box: AABB bounds.
    PositionBox { min: [f64; 3], max: [f64; 3], link_name: String },
    /// Joint range: arc visualization at joint.
    JointRange { joint_index: usize, min: f64, max: f64 },
    /// Keep-out zone: sphere that must be avoided.
    KeepOutZone { center: [f64; 3], radius: f64 },
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene Object Manipulation
// ═══════════════════════════════════════════════════════════════════════════

/// A manipulable scene object in the UI.
#[derive(Debug, Clone)]
pub struct SceneObjectUI {
    pub name: String,
    pub shape: SceneShapeUI,
    pub pose: Isometry3<f64>,
    pub selected: bool,
    pub color: [f32; 4],
}

/// Simplified shape for UI display.
#[derive(Debug, Clone)]
pub enum SceneShapeUI {
    Box { half_extents: [f64; 3] },
    Sphere { radius: f64 },
    Cylinder { radius: f64, half_height: f64 },
    Mesh { vertex_count: usize },
}

// ═══════════════════════════════════════════════════════════════════════════
// Servo Overlay
// ═══════════════════════════════════════════════════════════════════════════

/// State for real-time servo overlay visualization.
#[derive(Debug, Clone)]
pub struct ServoOverlay {
    /// Whether servo is active.
    pub active: bool,
    /// Current twist command [vx, vy, vz, wx, wy, wz].
    pub twist: [f64; 6],
    /// Commanded end-effector velocity magnitude.
    pub velocity_magnitude: f64,
    /// Safety scaling factor applied.
    pub safety_scale: f64,
    /// Distance to nearest collision.
    pub collision_distance: f64,
    /// Whether in singularity warning zone.
    pub near_singularity: bool,
}

impl Default for ServoOverlay {
    fn default() -> Self {
        Self {
            active: false,
            twist: [0.0; 6],
            velocity_magnitude: 0.0,
            safety_scale: 1.0,
            collision_distance: f64::INFINITY,
            near_singularity: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Interaction Manager
// ═══════════════════════════════════════════════════════════════════════════

/// Manages all interactive elements in the planning interface.
pub struct InteractionManager {
    /// Interactive markers.
    pub markers: HashMap<String, InteractiveMarker>,
    /// Planning panel state.
    pub planning_panel: PlanningPanel,
    /// Constraint visualizations.
    pub constraints: Vec<ConstraintViz>,
    /// Scene objects.
    pub scene_objects: Vec<SceneObjectUI>,
    /// Servo overlay.
    pub servo: ServoOverlay,
    /// Currently selected marker.
    pub selected_marker: Option<String>,
    /// Currently selected scene object.
    pub selected_object: Option<String>,
    /// Undo history: (marker_id, previous_pose).
    undo_stack: Vec<(String, Isometry3<f64>)>,
}

impl InteractionManager {
    pub fn new() -> Self {
        Self {
            markers: HashMap::new(),
            planning_panel: PlanningPanel::default(),
            constraints: Vec::new(),
            scene_objects: Vec::new(),
            servo: ServoOverlay::default(),
            selected_marker: None,
            selected_object: None,
            undo_stack: Vec::new(),
        }
    }

    /// Add an interactive marker.
    pub fn add_marker(&mut self, marker: InteractiveMarker) {
        self.markers.insert(marker.id.clone(), marker);
    }

    /// Remove a marker.
    pub fn remove_marker(&mut self, id: &str) -> Option<InteractiveMarker> {
        self.markers.remove(id)
    }

    /// Move a marker to a new pose (with undo support).
    pub fn move_marker(&mut self, id: &str, new_pose: Isometry3<f64>) {
        if let Some(marker) = self.markers.get_mut(id) {
            self.undo_stack.push((id.to_string(), marker.pose));
            marker.pose = new_pose;
        }
    }

    /// Undo the last marker move.
    pub fn undo(&mut self) -> bool {
        if let Some((id, pose)) = self.undo_stack.pop() {
            if let Some(marker) = self.markers.get_mut(&id) {
                marker.pose = pose;
                return true;
            }
        }
        false
    }

    /// Get the goal from the active goal marker.
    pub fn get_goal(&self) -> Option<Goal> {
        self.markers.get("goal").map(|m| m.to_goal())
    }

    /// Set the goal marker to a specific pose.
    pub fn set_goal_pose(&mut self, pose: Isometry3<f64>) {
        if let Some(marker) = self.markers.get_mut("goal") {
            marker.pose = pose;
        } else {
            self.add_marker(InteractiveMarker::pose_marker("goal", pose));
        }
    }

    /// Add a scene object.
    pub fn add_scene_object(&mut self, obj: SceneObjectUI) {
        self.scene_objects.push(obj);
    }

    /// Remove a scene object by name.
    pub fn remove_scene_object(&mut self, name: &str) {
        self.scene_objects.retain(|o| o.name != name);
    }

    /// Select an object by name.
    pub fn select_object(&mut self, name: &str) {
        for obj in &mut self.scene_objects {
            obj.selected = obj.name == name;
        }
        self.selected_object = Some(name.to_string());
    }

    /// Deselect all objects.
    pub fn deselect_all(&mut self) {
        for obj in &mut self.scene_objects {
            obj.selected = false;
        }
        self.selected_object = None;
        self.selected_marker = None;
    }

    /// Add a constraint visualization.
    pub fn add_constraint(&mut self, viz: ConstraintViz) {
        self.constraints.push(viz);
    }

    /// Clear all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Number of interactive elements.
    pub fn num_markers(&self) -> usize { self.markers.len() }
    pub fn num_objects(&self) -> usize { self.scene_objects.len() }
    pub fn num_constraints(&self) -> usize { self.constraints.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_creation() {
        let m = InteractiveMarker::pose_marker("goal", Isometry3::identity());
        assert_eq!(m.id, "goal");
        assert_eq!(m.marker_type, MarkerType::Pose6DOF);
        assert!(m.visible);
    }

    #[test]
    fn marker_to_goal() {
        let m = InteractiveMarker::pose_marker("g", Isometry3::translation(1.0, 2.0, 3.0));
        let goal = m.to_goal();
        match goal {
            Goal::Pose(p) => {
                let t = p.translation();
                assert!((t.x - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Pose goal"),
        }
    }

    #[test]
    fn interaction_manager_markers() {
        let mut mgr = InteractionManager::new();
        mgr.add_marker(InteractiveMarker::pose_marker("goal", Isometry3::identity()));
        mgr.add_marker(InteractiveMarker::position_marker("point", [1.0, 0.0, 0.0]));

        assert_eq!(mgr.num_markers(), 2);
        assert!(mgr.get_goal().is_some());

        mgr.remove_marker("point");
        assert_eq!(mgr.num_markers(), 1);
    }

    #[test]
    fn marker_move_and_undo() {
        let mut mgr = InteractionManager::new();
        mgr.add_marker(InteractiveMarker::pose_marker("goal", Isometry3::identity()));

        let new_pose = Isometry3::translation(1.0, 0.0, 0.0);
        mgr.move_marker("goal", new_pose);

        let pos = mgr.markers["goal"].position();
        assert!((pos[0] - 1.0).abs() < 1e-10);

        assert!(mgr.undo());
        let pos = mgr.markers["goal"].position();
        assert!((pos[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scene_object_selection() {
        let mut mgr = InteractionManager::new();
        mgr.add_scene_object(SceneObjectUI {
            name: "table".into(),
            shape: SceneShapeUI::Box { half_extents: [0.5, 0.5, 0.02] },
            pose: Isometry3::identity(),
            selected: false,
            color: [0.6, 0.4, 0.2, 1.0],
        });
        mgr.add_scene_object(SceneObjectUI {
            name: "cup".into(),
            shape: SceneShapeUI::Cylinder { radius: 0.04, half_height: 0.06 },
            pose: Isometry3::translation(0.3, 0.0, 0.1),
            selected: false,
            color: [0.8, 0.2, 0.2, 1.0],
        });

        mgr.select_object("cup");
        assert_eq!(mgr.selected_object.as_deref(), Some("cup"));
        assert!(mgr.scene_objects.iter().find(|o| o.name == "cup").unwrap().selected);
        assert!(!mgr.scene_objects.iter().find(|o| o.name == "table").unwrap().selected);

        mgr.deselect_all();
        assert!(mgr.selected_object.is_none());
    }

    #[test]
    fn planning_panel_defaults() {
        let panel = PlanningPanel::default();
        assert_eq!(panel.planner_id, "rrt_connect");
        assert!(panel.smooth);
        assert_eq!(panel.status, PlanningStatus::Idle);
        assert_eq!(panel.available_planners.len(), 7);
    }

    #[test]
    fn constraint_visualization() {
        let mut mgr = InteractionManager::new();
        mgr.add_constraint(ConstraintViz {
            name: "keep_upright".into(),
            constraint_type: ConstraintVizType::OrientationCone {
                axis: [0.0, 0.0, 1.0],
                tolerance: 0.3,
                link_name: "ee_link".into(),
            },
            visible: true,
            color: [0.0, 1.0, 0.0, 0.3],
        });

        assert_eq!(mgr.num_constraints(), 1);
        mgr.clear_constraints();
        assert_eq!(mgr.num_constraints(), 0);
    }

    #[test]
    fn servo_overlay_default() {
        let servo = ServoOverlay::default();
        assert!(!servo.active);
        assert_eq!(servo.twist, [0.0; 6]);
        assert!(!servo.near_singularity);
    }

    #[test]
    fn set_goal_creates_marker() {
        let mut mgr = InteractionManager::new();
        mgr.set_goal_pose(Isometry3::translation(0.5, 0.3, 0.4));
        assert!(mgr.markers.contains_key("goal"));
        let pos = mgr.markers["goal"].position();
        assert!((pos[0] - 0.5).abs() < 1e-10);
    }
}
