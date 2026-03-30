//! TUI (terminal UI) setup wizard for robot configuration.
//!
//! Provides a step-by-step terminal-based wizard that guides users through
//! robot configuration. Works over SSH without any GUI.
//!
//! Steps:
//! 1. Load URDF → validate
//! 2. Auto-detect planning groups → confirm/edit
//! 3. Auto-detect ACM → confirm
//! 4. Define end-effectors
//! 5. Set named poses (teach pendant style)
//! 6. Choose IK solver
//! 7. Write config file

use crate::setup::*;
use crate::Robot;

/// A step in the setup wizard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WizardStep {
    LoadURDF,
    PlanningGroups,
    CollisionMatrix,
    EndEffectors,
    NamedPoses,
    IKSolver,
    Review,
    WriteConfig,
    Done,
}

impl WizardStep {
    pub fn title(&self) -> &str {
        match self {
            Self::LoadURDF => "Load URDF",
            Self::PlanningGroups => "Planning Groups",
            Self::CollisionMatrix => "Collision Matrix (ACM)",
            Self::EndEffectors => "End-Effectors",
            Self::NamedPoses => "Named Poses",
            Self::IKSolver => "IK Solver Selection",
            Self::Review => "Review Configuration",
            Self::WriteConfig => "Write Config File",
            Self::Done => "Done",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::LoadURDF => 0,
            Self::PlanningGroups => 1,
            Self::CollisionMatrix => 2,
            Self::EndEffectors => 3,
            Self::NamedPoses => 4,
            Self::IKSolver => 5,
            Self::Review => 6,
            Self::WriteConfig => 7,
            Self::Done => 8,
        }
    }

    pub fn total_steps() -> usize { 8 }

    pub fn next(&self) -> Self {
        match self {
            Self::LoadURDF => Self::PlanningGroups,
            Self::PlanningGroups => Self::CollisionMatrix,
            Self::CollisionMatrix => Self::EndEffectors,
            Self::EndEffectors => Self::NamedPoses,
            Self::NamedPoses => Self::IKSolver,
            Self::IKSolver => Self::Review,
            Self::Review => Self::WriteConfig,
            Self::WriteConfig => Self::Done,
            Self::Done => Self::Done,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            Self::LoadURDF => Self::LoadURDF,
            Self::PlanningGroups => Self::LoadURDF,
            Self::CollisionMatrix => Self::PlanningGroups,
            Self::EndEffectors => Self::CollisionMatrix,
            Self::NamedPoses => Self::EndEffectors,
            Self::IKSolver => Self::NamedPoses,
            Self::Review => Self::IKSolver,
            Self::WriteConfig => Self::Review,
            Self::Done => Self::WriteConfig,
        }
    }
}

/// State of the setup wizard.
pub struct WizardState {
    pub step: WizardStep,
    pub robot: Option<Robot>,
    pub config: SetupConfig,
    pub urdf_path: String,
    pub output_path: String,
    pub validation_issues: Vec<ValidationIssue>,
    pub status_message: String,
}

impl WizardState {
    pub fn new() -> Self {
        Self {
            step: WizardStep::LoadURDF,
            robot: None,
            config: SetupConfig::new(),
            urdf_path: String::new(),
            output_path: "kinetic.toml".into(),
            validation_issues: Vec::new(),
            status_message: "Welcome to the Kinetic Setup Wizard".into(),
        }
    }

    /// Load a URDF and auto-detect configuration.
    pub fn load_urdf(&mut self, urdf_content: &str) -> bool {
        match Robot::from_urdf_string(urdf_content) {
            Ok(robot) => {
                self.config = generate_config(&robot);
                self.validation_issues = validate_config(&robot, &self.config);
                let errors = self.validation_issues.iter()
                    .filter(|i| i.severity == Severity::Error).count();
                self.status_message = format!(
                    "Loaded '{}': {} links, {} joints, {} DOF. {} issues ({} errors).",
                    robot.name, robot.links.len(), robot.joints.len(), robot.dof,
                    self.validation_issues.len(), errors,
                );
                self.robot = Some(robot);
                true
            }
            Err(e) => {
                self.status_message = format!("Failed to load URDF: {}", e);
                false
            }
        }
    }

    /// Advance to next step.
    pub fn next_step(&mut self) {
        self.step = self.step.next();
    }

    /// Go back to previous step.
    pub fn prev_step(&mut self) {
        self.step = self.step.prev();
    }

    /// Generate config TOML output.
    pub fn generate_output(&self) -> Option<String> {
        self.robot.as_ref().map(|r| generate_config_toml(r, &self.config))
    }

    /// Progress as fraction (0.0..1.0).
    pub fn progress(&self) -> f64 {
        self.step.index() as f64 / WizardStep::total_steps() as f64
    }

    /// Summary for the review step.
    pub fn summary(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(ref robot) = self.robot {
            lines.push(format!("Robot: {}", robot.name));
            lines.push(format!("DOF: {}", robot.dof));
        }
        lines.push(format!("Planning groups: {}", self.config.planning_groups.len()));
        lines.push(format!("ACM entries: {}", self.config.acm_entries.len()));
        lines.push(format!("End-effectors: {}", self.config.end_effectors.len()));
        lines.push(format!("Named poses: {}", self.config.named_poses.len()));
        lines.push(format!("Output: {}", self.output_path));
        lines
    }
}

/// Format for terminal display (no ratatui dependency needed for data model).
pub fn format_step_header(step: &WizardStep) -> String {
    format!(
        "╔══ Step {}/{}: {} ══╗",
        step.index() + 1,
        WizardStep::total_steps(),
        step.title()
    )
}

pub fn format_progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}] {:.0}%", "█".repeat(filled), "░".repeat(empty), progress * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base"><collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision></link>
  <link name="tip"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2" effort="10"/>
  </joint>
</robot>"#;

    #[test]
    fn wizard_step_navigation() {
        let mut step = WizardStep::LoadURDF;
        assert_eq!(step.index(), 0);

        step = step.next();
        assert_eq!(step, WizardStep::PlanningGroups);

        step = step.prev();
        assert_eq!(step, WizardStep::LoadURDF);

        // Can't go before first
        step = step.prev();
        assert_eq!(step, WizardStep::LoadURDF);
    }

    #[test]
    fn wizard_load_urdf() {
        let mut state = WizardState::new();
        assert!(state.load_urdf(URDF));
        assert!(state.robot.is_some());
        assert!(!state.config.planning_groups.is_empty());
        assert!(state.status_message.contains("test_arm"));
    }

    #[test]
    fn wizard_load_invalid_urdf() {
        let mut state = WizardState::new();
        assert!(!state.load_urdf("not xml"));
        assert!(state.robot.is_none());
        assert!(state.status_message.contains("Failed"));
    }

    #[test]
    fn wizard_generate_output() {
        let mut state = WizardState::new();
        state.load_urdf(URDF);
        let output = state.generate_output();
        assert!(output.is_some());
        assert!(output.unwrap().contains("[planning_groups"));
    }

    #[test]
    fn wizard_summary() {
        let mut state = WizardState::new();
        state.load_urdf(URDF);
        let summary = state.summary();
        assert!(summary.len() >= 5);
        assert!(summary[0].contains("test_arm"));
    }

    #[test]
    fn wizard_progress() {
        let state = WizardState::new();
        assert_eq!(state.progress(), 0.0);
    }

    #[test]
    fn format_helpers() {
        let header = format_step_header(&WizardStep::PlanningGroups);
        assert!(header.contains("Planning Groups"));
        assert!(header.contains("2/8"));

        let bar = format_progress_bar(0.5, 20);
        assert!(bar.contains("50%"));
    }
}
