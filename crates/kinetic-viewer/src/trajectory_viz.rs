//! Trajectory visualization: trail rendering, playback UI, ghost robots.
//!
//! Generates line vertices for trajectory trails (EE path as colored line strip)
//! and provides egui widgets for playback controls (play/pause, speed, scrubber).

use crate::pipeline::LineVertex;
use crate::TrajectoryPlayer;
use kinetic_core::Trajectory;

/// Generate trail line vertices by sampling the trajectory at regular intervals.
///
/// Since we don't have FK here (it's in kinetic-kinematics), we use the first 3
/// joint values as a proxy XYZ position. Callers with FK should use
/// [`trail_from_positions`] instead.
pub fn trail_from_trajectory(trajectory: &Trajectory, num_samples: usize) -> Vec<LineVertex> {
    if trajectory.is_empty() || trajectory.dof < 3 || num_samples < 2 {
        return Vec::new();
    }

    let mut line_vertices = Vec::with_capacity(num_samples * 2);

    for i in 0..num_samples {
        let t = i as f64 / (num_samples - 1) as f64;
        let jv = trajectory.sample(t);

        // Use first 3 joints as proxy XYZ (real usage would FK → EE position)
        let pos = [jv[0] as f32, jv[1] as f32, jv[2] as f32];

        // Color gradient: green → blue
        let frac = t as f32;
        let color = [
            0.1 * (1.0 - frac),
            0.8 * (1.0 - frac) + 0.2 * frac,
            0.2 * (1.0 - frac) + 0.9 * frac,
            0.8,
        ];

        if i > 0 {
            // End previous line segment
            line_vertices.push(LineVertex { position: pos, color });
        }
        if i < num_samples - 1 {
            // Start next line segment
            line_vertices.push(LineVertex { position: pos, color });
        }
    }

    line_vertices
}

/// Generate trail from pre-computed EE positions (e.g., from FK).
pub fn trail_from_positions(positions: &[[f32; 3]]) -> Vec<LineVertex> {
    if positions.len() < 2 {
        return Vec::new();
    }

    let mut line_vertices = Vec::with_capacity(positions.len() * 2);
    let total = positions.len() as f32;

    for (i, pos) in positions.iter().enumerate() {
        let frac = i as f32 / total;
        let color = [
            0.1 * (1.0 - frac),
            0.8 * (1.0 - frac) + 0.2 * frac,
            0.2 * (1.0 - frac) + 0.9 * frac,
            0.8,
        ];

        if i > 0 {
            line_vertices.push(LineVertex { position: *pos, color });
        }
        if i < positions.len() - 1 {
            line_vertices.push(LineVertex { position: *pos, color });
        }
    }

    line_vertices
}

/// State for managing multiple trajectory displays.
pub struct TrajectoryVizState {
    /// Active trajectory player (if any).
    pub player: Option<TrajectoryPlayer>,
    /// Stored trajectories for comparison.
    pub stored_trails: Vec<StoredTrail>,
    /// Whether trail is visible.
    pub show_trail: bool,
    /// Whether ghost robots are visible.
    pub show_ghosts: bool,
    /// Playback speed multiplier (persisted across frames).
    pub playback_speed: f64,
    /// Whether playback should loop (persisted across frames).
    pub looping: bool,
}

/// A stored trajectory with metadata.
pub struct StoredTrail {
    pub name: String,
    pub trail_vertices: Vec<LineVertex>,
    pub visible: bool,
    pub color_idx: usize,
}

impl Default for TrajectoryVizState {
    fn default() -> Self {
        Self {
            player: None,
            stored_trails: Vec::new(),
            show_trail: true,
            show_ghosts: true,
            playback_speed: 1.0,
            looping: false,
        }
    }
}

impl TrajectoryVizState {
    /// Set a new trajectory for playback.
    pub fn set_trajectory(&mut self, trajectory: Trajectory, name: &str) {
        let trail = trail_from_trajectory(&trajectory, 100);
        let idx = self.stored_trails.len();
        self.stored_trails.push(StoredTrail {
            name: name.to_string(),
            trail_vertices: trail,
            visible: true,
            color_idx: idx,
        });
        self.player = Some(TrajectoryPlayer::new(trajectory));
    }

    /// Collect all visible trail line vertices.
    pub fn collect_trail_lines(&self) -> Vec<LineVertex> {
        let mut lines = Vec::new();
        for trail in &self.stored_trails {
            if trail.visible {
                lines.extend_from_slice(&trail.trail_vertices);
            }
        }
        lines
    }
}

/// Draw the trajectory playback egui panel.
#[cfg(feature = "visual")]
pub fn playback_panel_ui(ctx: &egui::Context, viz: &mut TrajectoryVizState) {
    let has_player = viz.player.is_some();

    egui::TopBottomPanel::bottom("playback_bar")
        .resizable(false)
        .min_height(32.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(player) = &mut viz.player {
                    // Play/pause button
                    let icon = if player.is_playing() { "||" } else { ">" };
                    if ui.button(egui::RichText::new(icon).monospace().size(16.0)).clicked() {
                        if player.is_playing() {
                            player.pause();
                        } else {
                            player.play();
                        }
                    }

                    // Reset button
                    if ui.button("Reset").clicked() {
                        player.reset();
                    }

                    // Speed slider — persisted in TrajectoryVizState (GAP 6)
                    ui.label("Speed:");
                    if ui.add(egui::Slider::new(&mut viz.playback_speed, 0.1..=5.0).show_value(false)).changed() {
                        player.set_speed(viz.playback_speed);
                    }

                    // Loop toggle — persisted in TrajectoryVizState (GAP 8)
                    if ui.checkbox(&mut viz.looping, "Loop").changed() {
                        player.set_looping(viz.looping);
                    }

                    ui.separator();

                    // Progress bar / scrubber (GAP 7)
                    let mut progress = player.progress();
                    let bar_response = ui.add(
                        egui::Slider::new(&mut progress, 0.0..=1.0)
                            .text("t")
                            .show_value(true)
                            .fixed_decimals(2),
                    );
                    if bar_response.changed() {
                        player.seek(progress);
                    }
                } else {
                    ui.colored_label(
                        egui::Color32::GRAY,
                        "No trajectory loaded. Plan a motion to see playback controls.",
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut viz.show_trail, "Trail");
                    ui.checkbox(&mut viz.show_ghosts, "Ghosts");
                });
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trail_from_trajectory_generates_lines() {
        let mut traj = Trajectory::with_dof(3);
        traj.push_waypoint(&[0.0, 0.0, 0.0]);
        traj.push_waypoint(&[1.0, 0.5, 0.0]);
        traj.push_waypoint(&[2.0, 1.0, 0.0]);

        let trail = trail_from_trajectory(&traj, 10);
        // 10 sample points → 9 line segments → 18 vertices
        assert_eq!(trail.len(), 18);
    }

    #[test]
    fn trail_from_positions_generates_lines() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let trail = trail_from_positions(&positions);
        // 3 positions → 2 segments → 4 vertices
        assert_eq!(trail.len(), 4);
    }

    #[test]
    fn empty_trajectory_gives_empty_trail() {
        let traj = Trajectory::with_dof(3);
        let trail = trail_from_trajectory(&traj, 10);
        assert!(trail.is_empty());
    }

    #[test]
    fn viz_state_set_trajectory() {
        let mut state = TrajectoryVizState::default();
        let mut traj = Trajectory::with_dof(3);
        traj.push_waypoint(&[0.0, 0.0, 0.0]);
        traj.push_waypoint(&[1.0, 1.0, 1.0]);

        state.set_trajectory(traj, "test_plan");
        assert!(state.player.is_some());
        assert_eq!(state.stored_trails.len(), 1);
        assert_eq!(state.stored_trails[0].name, "test_plan");
    }

    #[test]
    fn collect_trail_lines_respects_visibility() {
        let mut state = TrajectoryVizState::default();

        let mut t1 = Trajectory::with_dof(3);
        t1.push_waypoint(&[0.0, 0.0, 0.0]);
        t1.push_waypoint(&[1.0, 1.0, 1.0]);
        state.set_trajectory(t1, "trail1");

        let mut t2 = Trajectory::with_dof(3);
        t2.push_waypoint(&[0.0, 0.0, 0.0]);
        t2.push_waypoint(&[2.0, 2.0, 2.0]);
        state.set_trajectory(t2, "trail2");

        let all_lines = state.collect_trail_lines();
        assert!(!all_lines.is_empty());

        // Hide first trail
        state.stored_trails[0].visible = false;
        let hidden_lines = state.collect_trail_lines();
        assert!(hidden_lines.len() < all_lines.len());
    }
}
