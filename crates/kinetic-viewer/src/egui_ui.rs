//! egui UI panels for the kinetic 3D viewer.
//!
//! Each function draws a panel using real egui widgets and returns actions
//! that the app loop should handle (plan clicked, joint changed, etc.).

use super::interaction::*;

// ═══════════════════════════════════════════════════════════════════════════
// Planning Panel
// ═══════════════════════════════════════════════════════════════════════════

/// Actions from the planning panel UI.
#[derive(Debug, Clone, Default)]
pub struct PlanningPanelActions {
    pub plan_clicked: bool,
    pub execute_clicked: bool,
    pub stop_clicked: bool,
    pub planner_id: String,
    pub planning_time: f64,
}

/// Draw the planning panel.
#[cfg(feature = "visual")]
pub fn planning_panel_ui(ctx: &egui::Context, panel: &mut PlanningPanel) -> PlanningPanelActions {
    let mut actions = PlanningPanelActions::default();

    egui::Window::new("Planning")
        .default_pos([8.0, 8.0])
        .default_width(220.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Planner selection
            ui.label("Planner:");
            egui::ComboBox::from_id_salt("planner_select")
                .selected_text(&panel.planner_id)
                .show_ui(ui, |ui| {
                    for p in &panel.available_planners {
                        ui.selectable_value(&mut panel.planner_id, p.clone(), p);
                    }
                });

            ui.separator();

            // Parameters
            ui.add(
                egui::Slider::new(&mut panel.planning_time, 0.1..=30.0)
                    .text("Time (s)")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut panel.velocity_scale, 0.01..=1.0).text("Vel scale"),
            );
            ui.add(
                egui::Slider::new(&mut panel.acceleration_scale, 0.01..=1.0).text("Accel scale"),
            );

            ui.horizontal(|ui| {
                ui.label("Attempts:");
                ui.add(egui::DragValue::new(&mut panel.num_attempts).range(1..=100));
            });

            ui.checkbox(&mut panel.smooth, "Smooth");
            ui.checkbox(&mut panel.time_parameterize, "Time parameterize");

            ui.separator();

            // Buttons
            ui.horizontal(|ui| {
                let plan_btn = egui::Button::new(
                    egui::RichText::new("Plan").color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(40, 140, 60));
                if ui.add(plan_btn).clicked() {
                    actions.plan_clicked = true;
                }

                let exec_btn = egui::Button::new(
                    egui::RichText::new("Execute").color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(40, 80, 160));
                if ui.add(exec_btn).clicked() {
                    actions.execute_clicked = true;
                }

                let stop_btn = egui::Button::new(
                    egui::RichText::new("Stop").color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(160, 40, 40));
                if ui.add(stop_btn).clicked() {
                    actions.stop_clicked = true;
                }
            });

            // Status
            let (status_text, status_color) = match &panel.status {
                PlanningStatus::Idle => ("Idle", egui::Color32::GRAY),
                PlanningStatus::Planning => ("Planning...", egui::Color32::YELLOW),
                PlanningStatus::Succeeded => ("Succeeded", egui::Color32::GREEN),
                PlanningStatus::Failed(msg) => (msg.as_str(), egui::Color32::RED),
            };
            ui.colored_label(status_color, status_text);

            if let Some(t) = panel.last_planning_time {
                ui.label(format!("Last: {:.1}ms", t.as_secs_f64() * 1000.0));
            }
        });

    actions.planner_id = panel.planner_id.clone();
    actions.planning_time = panel.planning_time;
    actions
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint Slider Panel
// ═══════════════════════════════════════════════════════════════════════════

/// Draw the joint slider panel. Returns true if any joint changed.
#[cfg(feature = "visual")]
pub fn joint_slider_panel_ui(
    ctx: &egui::Context,
    joint_names: &[String],
    joint_values: &mut [f64],
    joint_limits: &[(f64, f64)],
) -> bool {
    let mut changed = false;

    egui::Window::new("Joints")
        .default_pos([8.0, 350.0])
        .default_width(250.0)
        .resizable(true)
        .show(ctx, |ui| {
            if ui.button("Reset All").clicked() {
                for v in joint_values.iter_mut() {
                    *v = 0.0;
                }
                changed = true;
            }
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for (i, name) in joint_names.iter().enumerate() {
                        if i >= joint_values.len() || i >= joint_limits.len() {
                            break;
                        }
                        let (lo, hi) = joint_limits[i];
                        let deg = joint_values[i].to_degrees();

                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(name).monospace().strong(),
                            );
                            ui.label(format!("{:.1}\u{00b0}", deg));
                        });

                        let slider = egui::Slider::new(&mut joint_values[i], lo..=hi)
                            .show_value(false)
                            .step_by(0.01);
                        if ui.add(slider).changed() {
                            changed = true;
                        }
                    }
                });
        });

    changed
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene Objects Panel
// ═══════════════════════════════════════════════════════════════════════════

/// Draw the scene object list panel.
#[cfg(feature = "visual")]
pub fn scene_object_panel_ui(
    ctx: &egui::Context,
    objects: &mut Vec<SceneObjectUI>,
) -> Option<String> {
    let mut selected = None;

    egui::Window::new("Scene Objects")
        .default_pos([1040.0, 8.0])
        .default_width(200.0)
        .resizable(true)
        .show(ctx, |ui| {
            if objects.is_empty() {
                ui.label("No objects in scene.");
            }

            let mut to_remove = None;
            for (i, obj) in objects.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    let label = if obj.selected {
                        egui::RichText::new(&obj.name)
                            .strong()
                            .color(egui::Color32::YELLOW)
                    } else {
                        egui::RichText::new(&obj.name)
                    };

                    if ui.selectable_label(obj.selected, label).clicked() {
                        obj.selected = !obj.selected;
                        if obj.selected {
                            selected = Some(obj.name.clone());
                        }
                    }

                    if ui
                        .button(egui::RichText::new("X").color(egui::Color32::RED))
                        .clicked()
                    {
                        to_remove = Some(i);
                    }
                });
            }

            if let Some(idx) = to_remove {
                objects.remove(idx);
            }
        });

    selected
}

// ═══════════════════════════════════════════════════════════════════════════
// Constraint Panel
// ═══════════════════════════════════════════════════════════════════════════

/// Draw constraint visualization panel.
#[cfg(feature = "visual")]
pub fn constraint_panel_ui(ctx: &egui::Context, constraints: &mut [ConstraintViz]) {
    if constraints.is_empty() {
        return;
    }

    egui::Window::new("Constraints")
        .default_pos([1040.0, 300.0])
        .default_width(200.0)
        .show(ctx, |ui| {
            for c in constraints.iter_mut() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut c.visible, "");
                    ui.label(&c.name);
                });
            }
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// Servo Overlay
// ═══════════════════════════════════════════════════════════════════════════

/// Actions from the servo panel.
#[derive(Debug, Clone, Default)]
pub struct ServoActions {
    pub start: bool,
    pub stop: bool,
    pub twist: [f64; 6],
}

/// Draw servo control overlay.
#[cfg(feature = "visual")]
pub fn servo_panel_ui(ctx: &egui::Context, servo: &mut ServoOverlay) -> ServoActions {
    let mut actions = ServoActions::default();

    egui::Window::new("Servo Control")
        .default_pos([8.0, 550.0])
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if !servo.active {
                    if ui.button(egui::RichText::new("Start").color(egui::Color32::GREEN)).clicked() {
                        actions.start = true;
                    }
                } else {
                    if ui.button(egui::RichText::new("Stop").color(egui::Color32::RED)).clicked() {
                        actions.stop = true;
                    }
                }
            });

            if servo.active {
                ui.separator();
                let labels = ["vx", "vy", "vz", "wx", "wy", "wz"];
                for (i, label) in labels.iter().enumerate() {
                    ui.add(
                        egui::Slider::new(&mut servo.twist[i], -1.0..=1.0)
                            .text(*label)
                            .fixed_decimals(2),
                    );
                }

                ui.separator();

                // Safety indicators
                let dist_color = if servo.collision_distance > 0.1 {
                    egui::Color32::GREEN
                } else if servo.collision_distance > 0.02 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(
                    dist_color,
                    format!("Collision dist: {:.3}m", servo.collision_distance),
                );

                // Velocity magnitude (GAP 11)
                ui.label(format!("Velocity: {:.3} m/s", servo.velocity_magnitude));

                let sing_color = if servo.near_singularity {
                    egui::Color32::RED
                } else {
                    egui::Color32::GREEN
                };
                ui.colored_label(
                    sing_color,
                    if servo.near_singularity {
                        "SINGULARITY WARNING"
                    } else {
                        "Singularity: OK"
                    },
                );

                ui.label(format!("Safety scale: {:.0}%", servo.safety_scale * 100.0));
            }
        });

    actions.twist = servo.twist;
    actions
}

// ═══════════════════════════════════════════════════════════════════════════
// Stats & Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

/// Draw stats overlay.
#[cfg(feature = "visual")]
pub fn stats_panel_ui(ctx: &egui::Context, settings: &super::ViewerSettings) {
    egui::Window::new("Stats")
        .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
        .resizable(false)
        .collapsible(false)
        .title_bar(false)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("KINETIC Viewer")
                    .strong()
                    .color(egui::Color32::from_rgb(100, 180, 255)),
            );
            ui.separator();
            ui.label(format!(
                "Grid: {}  Axes: {}",
                if settings.show_grid { "ON" } else { "off" },
                if settings.show_axes { "ON" } else { "off" },
            ));
            ui.label("F1: shortcuts  F3: toggle stats");
        });
}

/// Draw keyboard shortcuts overlay.
#[cfg(feature = "visual")]
pub fn shortcuts_panel_ui(ctx: &egui::Context) {
    egui::Window::new("Keyboard Shortcuts")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            egui::Grid::new("shortcuts_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (key, desc) in shortcuts_overlay() {
                        ui.label(
                            egui::RichText::new(key)
                                .monospace()
                                .color(egui::Color32::from_rgb(255, 200, 100)),
                        );
                        ui.label(desc);
                        ui.end_row();
                    }
                });
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-visual stubs (kept for GPU-free builds and existing tests)
// ═══════════════════════════════════════════════════════════════════════════

/// Draw the planning panel UI (non-visual / test stub).
pub fn planning_panel(panel: &mut PlanningPanel) -> PlanningPanelActions {
    let mut actions = PlanningPanelActions::default();
    actions.planner_id = panel.planner_id.clone();
    actions.planning_time = panel.planning_time;
    actions
}

/// Draw the joint slider panel (non-visual stub).
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

/// Draw the scene object list panel (non-visual stub).
pub fn scene_object_panel(objects: &[SceneObjectUI]) -> Option<String> {
    objects.iter().find(|o| o.selected).map(|o| o.name.clone())
}

/// Draw constraint visualization panel (non-visual stub).
pub fn constraint_panel(constraints: &[ConstraintViz]) -> Vec<bool> {
    constraints.iter().map(|c| c.visible).collect()
}

/// Draw servo control overlay (non-visual stub).
pub fn servo_panel(servo: &ServoOverlay) -> ServoActions {
    ServoActions {
        start: !servo.active,
        stop: servo.active,
        twist: servo.twist,
    }
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
        ("Esc", "Deselect all / Quit"),
        ("Z", "Undo"),
        ("F", "Focus on selection"),
        ("F1", "Toggle shortcuts"),
        ("F3", "Toggle stats"),
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
    /// Whether to show the collision debug panel (toggle with 'c' key).
    pub show_collision_debug: bool,
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
            show_collision_debug: false,
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
        let limits = [(-1.0, 1.0), (-2.0, 2.0)];

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
