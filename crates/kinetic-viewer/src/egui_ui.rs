//! egui UI panels for the kinetic 3D viewer.
//!
//! Renders planning interface panels using egui. These functions produce
//! egui widget commands that integrate with any egui backend (wgpu, glow, etc).

use super::interaction::*;

/// Draw the planning panel UI. Returns true if "Plan" was clicked.
pub fn planning_panel(panel: &mut PlanningPanel) -> PlanningPanelActions {
    let mut actions = PlanningPanelActions::default();

    // In a real egui context:
    // egui::Window::new("Planning").show(ctx, |ui| { ... });
    // Here we model the state transitions that the UI would trigger.

    actions.planner_id = panel.planner_id.clone();
    actions.planning_time = panel.planning_time;
    actions
}

/// Actions from the planning panel UI.
#[derive(Debug, Clone, Default)]
pub struct PlanningPanelActions {
    pub plan_clicked: bool,
    pub execute_clicked: bool,
    pub stop_clicked: bool,
    pub planner_id: String,
    pub planning_time: f64,
}

/// Draw the joint slider panel. Returns modified joint values.
pub fn joint_slider_panel(
    joint_names: &[String],
    joint_values: &mut [f64],
    joint_limits: &[(f64, f64)],
) -> bool {
    let mut changed = false;
    for (i, (_name, (lo, hi))) in joint_names.iter().zip(joint_limits).enumerate() {
        if i < joint_values.len() {
            let old = joint_values[i];
            joint_values[i] = joint_values[i].clamp(*lo, *hi);
            if (joint_values[i] - old).abs() > 1e-10 {
                changed = true;
            }
        }
    }
    changed
}

/// Draw the scene object list panel.
pub fn scene_object_panel(objects: &[SceneObjectUI]) -> Option<String> {
    // Would render a list of objects with select/delete buttons
    // Returns the name of the selected object (if any)
    objects.iter().find(|o| o.selected).map(|o| o.name.clone())
}

/// Draw constraint visualization panel.
pub fn constraint_panel(constraints: &[ConstraintViz]) -> Vec<bool> {
    // Returns visibility toggle state per constraint
    constraints.iter().map(|c| c.visible).collect()
}

/// Draw servo control overlay.
pub fn servo_panel(servo: &ServoOverlay) -> ServoActions {
    ServoActions {
        start: !servo.active,
        stop: servo.active,
        twist: servo.twist,
    }
}

/// Actions from the servo panel.
#[derive(Debug, Clone, Default)]
pub struct ServoActions {
    pub start: bool,
    pub stop: bool,
    pub twist: [f64; 6],
}

/// Draw the keyboard shortcuts overlay.
pub fn shortcuts_overlay() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Space", "Play/pause trajectory"),
        ("R", "Reset view"),
        ("G", "Toggle grid"),
        ("A", "Toggle axes"),
        ("W", "Toggle wireframe"),
        ("C", "Toggle collision geometry"),
        ("V", "Toggle voxels"),
        ("P", "Toggle point cloud"),
        ("T", "Toggle trajectory trail"),
        ("Esc", "Deselect all"),
        ("Ctrl+Z", "Undo"),
        ("F", "Focus on selection"),
        ("1-9", "Set named pose"),
    ]
}

/// Full viewer UI layout: combines all panels.
pub struct ViewerUI {
    pub show_planning_panel: bool,
    pub show_joint_sliders: bool,
    pub show_scene_panel: bool,
    pub show_constraints: bool,
    pub show_servo: bool,
    pub show_shortcuts: bool,
    pub show_stats: bool,
}

impl Default for ViewerUI {
    fn default() -> Self {
        Self {
            show_planning_panel: true,
            show_joint_sliders: true,
            show_scene_panel: true,
            show_constraints: false,
            show_servo: false,
            show_shortcuts: false,
            show_stats: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planning_panel_default() {
        let mut panel = PlanningPanel::default();
        let actions = planning_panel(&mut panel);
        assert!(!actions.plan_clicked);
        assert_eq!(actions.planner_id, "rrt_connect");
    }

    #[test]
    fn joint_slider_clamps() {
        let names = vec!["j1".into(), "j2".into()];
        let mut values = [5.0, -5.0];
        let limits = [(- 1.0, 1.0), (-2.0, 2.0)];

        let changed = joint_slider_panel(&names, &mut values, &limits);
        assert!(changed);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -2.0);
    }

    #[test]
    fn scene_object_selection() {
        let objects = vec![
            SceneObjectUI {
                name: "table".into(),
                shape: SceneShapeUI::Box { half_extents: [0.5, 0.5, 0.02] },
                pose: nalgebra::Isometry3::identity(),
                selected: false,
                color: [0.5; 4],
            },
            SceneObjectUI {
                name: "cup".into(),
                shape: SceneShapeUI::Sphere { radius: 0.04 },
                pose: nalgebra::Isometry3::identity(),
                selected: true,
                color: [0.8, 0.2, 0.2, 1.0],
            },
        ];

        assert_eq!(scene_object_panel(&objects), Some("cup".into()));
    }

    #[test]
    fn shortcuts_populated() {
        let shortcuts = shortcuts_overlay();
        assert!(shortcuts.len() >= 10);
        assert!(shortcuts.iter().any(|(k, _)| *k == "Space"));
    }

    #[test]
    fn viewer_ui_default() {
        let ui = ViewerUI::default();
        assert!(ui.show_planning_panel);
        assert!(ui.show_stats);
        assert!(!ui.show_shortcuts);
    }
}
