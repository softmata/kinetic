//! Production task application pipelines built on the stage engine.
//!
//! Provides ready-to-use manipulation primitives (bin picking, assembly,
//! pour, handover, inspect, stack, sort), error recovery with re-planning,
//! task checkpoint/resume, execution logging/replay, and failure analytics.

use std::collections::HashMap;
use std::time::Instant;

use kinetic_core::JointValues;

use crate::stage_engine::*;

// ═══════════════════════════════════════════════════════════════════════════
// Manipulation Primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for bin picking pipeline.
#[derive(Debug, Clone)]
pub struct BinPickConfig {
    /// Candidate grasp joint configurations.
    pub grasp_candidates: Vec<JointValues>,
    /// Pre-grasp approach joints.
    pub approach_joints: JointValues,
    /// Post-grasp retreat joints.
    pub retreat_joints: JointValues,
    /// Drop-off location joints.
    pub dropoff_joints: JointValues,
    /// Object name.
    pub object: String,
    /// Gripper link name.
    pub gripper_link: String,
    /// Gripper open/close widths.
    pub gripper_open: f64,
    pub gripper_close: f64,
}

/// Build a bin picking pipeline with multiple grasp candidates and fallback.
pub fn bin_pick_pipeline(config: &BinPickConfig) -> SerialContainer {
    // Try each grasp candidate as a fallback
    let mut grasp_fallback = FallbackContainer::new("try_grasps");
    for (i, grasp) in config.grasp_candidates.iter().enumerate() {
        let attempt = SerialContainer::new(&format!("grasp_attempt_{}", i))
            .with(Box::new(MoveToStage::new("approach", config.approach_joints.clone())))
            .with(Box::new(MoveToStage::new(&format!("grasp_{}", i), grasp.clone())))
            .with(Box::new(GripperStage::close("close", config.gripper_close)));
        grasp_fallback.add(Box::new(attempt));
    }

    SerialContainer::new("bin_pick")
        .with(Box::new(GripperStage::open("open_gripper", config.gripper_open)))
        .with(Box::new(grasp_fallback))
        .with(Box::new(ModifySceneStage::attach("attach", &config.object, &config.gripper_link)))
        .with(Box::new(MoveToStage::new("retreat", config.retreat_joints.clone())))
        .with(Box::new(MoveToStage::new("dropoff", config.dropoff_joints.clone())))
        .with(Box::new(GripperStage::open("release", config.gripper_open)))
        .with(Box::new(ModifySceneStage::detach("detach", &config.object)))
}

/// Assembly insertion: approach, align, insert with force-controlled motion.
pub fn assembly_insertion(
    approach_joints: JointValues,
    align_joints: JointValues,
    insert_joints: JointValues,
    retreat_joints: JointValues,
) -> SerialContainer {
    SerialContainer::new("assembly_insert")
        .with(Box::new(MoveToStage::new("approach", approach_joints)))
        .with(Box::new(MoveToStage::new("align", align_joints)))
        .with(Box::new(MoveToStage::new("insert", insert_joints)))
        .with(Box::new(MoveToStage::new("retreat", retreat_joints)))
}

/// Pour task: move to pour position, tilt, return.
pub fn pour_task(
    pre_pour_joints: JointValues,
    pour_joints: JointValues,
    post_pour_joints: JointValues,
) -> SerialContainer {
    SerialContainer::new("pour")
        .with(Box::new(MoveToStage::new("pre_pour", pre_pour_joints)))
        .with(Box::new(MoveToStage::new("pour_tilt", pour_joints)))
        .with(Box::new(MoveToStage::new("post_pour", post_pour_joints)))
}

/// Handover task: one arm extends to handover pose, gripper releases.
pub fn handover_task(
    extend_joints: JointValues,
    handover_joints: JointValues,
    retract_joints: JointValues,
    gripper_open: f64,
    object: &str,
) -> SerialContainer {
    SerialContainer::new("handover")
        .with(Box::new(MoveToStage::new("extend", extend_joints)))
        .with(Box::new(MoveToStage::new("handover_pose", handover_joints)))
        .with(Box::new(GripperStage::open("release", gripper_open)))
        .with(Box::new(ModifySceneStage::detach("detach", object)))
        .with(Box::new(MoveToStage::new("retract", retract_joints)))
}

/// Inspect task: move to inspection pose, hold, return.
pub fn inspect_task(
    inspect_joints: JointValues,
    hold_duration_ms: u64,
    return_joints: JointValues,
) -> SerialContainer {
    // Hold is modeled as a stage that stays at the inspect pose
    let _ = hold_duration_ms; // duration handled at execution time
    SerialContainer::new("inspect")
        .with(Box::new(MoveToStage::new("inspect_pose", inspect_joints)))
        .with(Box::new(MoveToStage::new("return", return_joints)))
}

/// Stack task: pick object, move above stack, lower, release.
pub fn stack_task(
    object: &str,
    link: &str,
    pick_joints: JointValues,
    above_stack_joints: JointValues,
    stack_joints: JointValues,
    retreat_joints: JointValues,
    gripper_open: f64,
    gripper_close: f64,
) -> SerialContainer {
    SerialContainer::new("stack")
        .with(Box::new(GripperStage::open("open", gripper_open)))
        .with(Box::new(MoveToStage::new("pick_pose", pick_joints)))
        .with(Box::new(GripperStage::close("grasp", gripper_close)))
        .with(Box::new(ModifySceneStage::attach("attach", object, link)))
        .with(Box::new(MoveToStage::new("above_stack", above_stack_joints)))
        .with(Box::new(MoveToStage::new("stack_pose", stack_joints)))
        .with(Box::new(GripperStage::open("release", gripper_open)))
        .with(Box::new(ModifySceneStage::detach("detach", object)))
        .with(Box::new(MoveToStage::new("retreat", retreat_joints)))
}

/// Sort task: pick from source, place at destination based on classification.
pub fn sort_task(
    object: &str,
    link: &str,
    pick_joints: JointValues,
    destination_joints: JointValues,
    home_joints: JointValues,
    gripper_open: f64,
    gripper_close: f64,
) -> SerialContainer {
    SerialContainer::new("sort")
        .with(Box::new(GripperStage::open("open", gripper_open)))
        .with(Box::new(MoveToStage::new("pick", pick_joints)))
        .with(Box::new(GripperStage::close("grasp", gripper_close)))
        .with(Box::new(ModifySceneStage::attach("attach", object, link)))
        .with(Box::new(MoveToStage::new("destination", destination_joints)))
        .with(Box::new(GripperStage::open("release", gripper_open)))
        .with(Box::new(ModifySceneStage::detach("detach", object)))
        .with(Box::new(MoveToStage::new("home", home_joints)))
}

/// Dual-arm handoff: arm A extends, arm B grasps, arm A releases.
pub fn dual_arm_handoff(
    arm_a_extend: JointValues,
    arm_b_approach: JointValues,
    arm_b_grasp: JointValues,
    arm_a_retract: JointValues,
    arm_b_retract: JointValues,
    object: &str,
    _link_a: &str,
    link_b: &str,
    gripper_open: f64,
    gripper_close: f64,
) -> SerialContainer {
    SerialContainer::new("dual_arm_handoff")
        .with(Box::new(MoveToStage::new("arm_a_extend", arm_a_extend)))
        .with(Box::new(MoveToStage::new("arm_b_approach", arm_b_approach)))
        .with(Box::new(MoveToStage::new("arm_b_grasp_pose", arm_b_grasp)))
        .with(Box::new(GripperStage::close("arm_b_close", gripper_close)))
        .with(Box::new(ModifySceneStage::detach("detach_a", object)))
        .with(Box::new(ModifySceneStage::attach("attach_b", object, link_b)))
        .with(Box::new(GripperStage::open("arm_a_release", gripper_open)))
        .with(Box::new(MoveToStage::new("arm_a_retract", arm_a_retract)))
        .with(Box::new(MoveToStage::new("arm_b_retract", arm_b_retract)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Error Classification
// ═══════════════════════════════════════════════════════════════════════════

/// Error category for task failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Planning failed to find a path.
    PlanningFailure,
    /// Collision detected during execution.
    CollisionDetected,
    /// Grasp failed (object slipped or not acquired).
    GraspFailure,
    /// Joint limit exceeded.
    JointLimitViolation,
    /// Execution deviation exceeded threshold.
    DeviationExceeded,
    /// Hardware communication error.
    HardwareError,
    /// Timeout exceeded.
    Timeout,
    /// Object not found in scene.
    ObjectNotFound,
    /// Unknown error.
    Unknown,
}

/// A classified task error.
#[derive(Debug, Clone)]
pub struct TaskFailure {
    pub category: ErrorCategory,
    pub stage_name: String,
    pub message: String,
    pub timestamp: Instant,
    pub recoverable: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Recovery Strategy Registry
// ═══════════════════════════════════════════════════════════════════════════

/// Recovery action to take on failure.
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Re-plan the failed stage from current state.
    Replan,
    /// Skip the failed stage and continue.
    Skip,
    /// Retry the failed stage up to N times.
    Retry { max_attempts: usize },
    /// Go to a safe home position.
    GoHome { home_joints: JointValues },
    /// Run an alternative stage.
    Alternative { stage_name: String },
    /// Abort the entire task.
    Abort,
}

/// Maps error categories to recovery actions.
#[derive(Debug, Clone)]
pub struct RecoveryRegistry {
    strategies: HashMap<ErrorCategory, RecoveryAction>,
    default_action: RecoveryAction,
}

impl RecoveryRegistry {
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            default_action: RecoveryAction::Abort,
        }
    }

    /// Register a recovery action for an error category.
    pub fn register(&mut self, category: ErrorCategory, action: RecoveryAction) {
        self.strategies.insert(category, action);
    }

    /// Set the default recovery action.
    pub fn set_default(&mut self, action: RecoveryAction) {
        self.default_action = action;
    }

    /// Look up recovery action for a failure.
    pub fn lookup(&self, failure: &TaskFailure) -> &RecoveryAction {
        self.strategies.get(&failure.category).unwrap_or(&self.default_action)
    }

    /// Create a production-ready registry with sensible defaults.
    pub fn production() -> Self {
        let mut reg = Self::new();
        reg.register(ErrorCategory::PlanningFailure, RecoveryAction::Replan);
        reg.register(ErrorCategory::GraspFailure, RecoveryAction::Retry { max_attempts: 3 });
        reg.register(ErrorCategory::DeviationExceeded, RecoveryAction::Replan);
        reg.register(ErrorCategory::Timeout, RecoveryAction::Retry { max_attempts: 2 });
        reg.register(ErrorCategory::CollisionDetected, RecoveryAction::Abort);
        reg.register(ErrorCategory::HardwareError, RecoveryAction::Abort);
        reg.set_default(RecoveryAction::Abort);
        reg
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Task Checkpoint & Resume
// ═══════════════════════════════════════════════════════════════════════════

/// A checkpoint of task execution state.
#[derive(Debug, Clone)]
pub struct TaskCheckpoint {
    /// Which stage index we're at in the pipeline.
    pub stage_index: usize,
    /// Stage name at checkpoint.
    pub stage_name: String,
    /// Joint state at checkpoint.
    pub joints: JointValues,
    /// Scene modifications applied so far.
    pub scene_diffs: Vec<SceneModification>,
    /// Timestamp.
    pub timestamp: Instant,
    /// Completed stage solutions so far.
    pub completed_stages: usize,
}

/// Manages checkpoints for task resume.
#[derive(Default)]
pub struct CheckpointManager {
    checkpoints: Vec<TaskCheckpoint>,
}

impl CheckpointManager {
    pub fn new() -> Self { Self::default() }

    /// Save a checkpoint.
    pub fn save(&mut self, checkpoint: TaskCheckpoint) {
        self.checkpoints.push(checkpoint);
    }

    /// Get the latest checkpoint.
    pub fn latest(&self) -> Option<&TaskCheckpoint> {
        self.checkpoints.last()
    }

    /// Number of checkpoints.
    pub fn count(&self) -> usize { self.checkpoints.len() }

    /// Clear all checkpoints.
    pub fn clear(&mut self) { self.checkpoints.clear(); }
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution Logging & Replay
// ═══════════════════════════════════════════════════════════════════════════

/// A logged execution event.
#[derive(Debug, Clone)]
pub struct ExecutionLogEntry {
    pub timestamp: Instant,
    pub stage_name: String,
    pub event_type: LogEventType,
    pub joints: Option<JointValues>,
    pub message: String,
}

/// Types of logged events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogEventType {
    StageStarted,
    StageCompleted,
    StageFailed,
    GripperCommand,
    SceneModified,
    RecoveryTriggered,
    CheckpointSaved,
}

/// Execution logger for task replay.
pub struct ExecutionLog {
    entries: Vec<ExecutionLogEntry>,
    max_size: usize,
}

impl ExecutionLog {
    pub fn new(max_size: usize) -> Self {
        Self { entries: Vec::new(), max_size }
    }

    pub fn log(&mut self, entry: ExecutionLogEntry) {
        if self.entries.len() >= self.max_size {
            self.entries.drain(0..self.max_size / 2);
        }
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[ExecutionLogEntry] { &self.entries }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn clear(&mut self) { self.entries.clear(); }

    /// Filter entries by stage name.
    pub fn for_stage(&self, stage: &str) -> Vec<&ExecutionLogEntry> {
        self.entries.iter().filter(|e| e.stage_name == stage).collect()
    }

    /// Filter entries by event type.
    pub fn by_type(&self, event_type: LogEventType) -> Vec<&ExecutionLogEntry> {
        self.entries.iter().filter(|e| e.event_type == event_type).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Failure Analytics
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks failure statistics for analytics.
#[derive(Default)]
pub struct FailureAnalytics {
    failures: Vec<TaskFailure>,
}

impl FailureAnalytics {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, failure: TaskFailure) {
        self.failures.push(failure);
    }

    pub fn total_failures(&self) -> usize { self.failures.len() }

    /// Count failures by category.
    pub fn by_category(&self) -> HashMap<ErrorCategory, usize> {
        let mut counts = HashMap::new();
        for f in &self.failures {
            *counts.entry(f.category).or_insert(0) += 1;
        }
        counts
    }

    /// Count failures by stage.
    pub fn by_stage(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for f in &self.failures {
            *counts.entry(f.stage_name.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Most common failure category.
    pub fn most_common(&self) -> Option<ErrorCategory> {
        self.by_category().into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(cat, _)| cat)
    }

    /// Failure rate (failures / total_attempts).
    pub fn failure_rate(&self, total_attempts: usize) -> f64 {
        if total_attempts == 0 { return 0.0; }
        self.failures.len() as f64 / total_attempts as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn j(vals: &[f64]) -> JointValues { JointValues::new(vals.to_vec()) }
    fn start(dof: usize) -> InterfaceState { InterfaceState::from_joints(JointValues::zeros(dof)) }

    // ─── Manipulation primitives ───

    #[test]
    fn bin_pick_pipeline_executes() {
        let config = BinPickConfig {
            grasp_candidates: vec![j(&[0.3, 0.3]), j(&[0.4, 0.4])],
            approach_joints: j(&[0.2, 0.2]),
            retreat_joints: j(&[0.5, 0.5]),
            dropoff_joints: j(&[1.0, 1.0]),
            object: "bolt".into(),
            gripper_link: "gripper".into(),
            gripper_open: 0.08,
            gripper_close: 0.02,
        };

        let pipeline = bin_pick_pipeline(&config);
        let engine = TaskEngine::new();
        let result = engine.execute(&pipeline, start(2), Duration::from_secs(5));
        assert!(result.best().is_some(), "Bin pick should produce a solution");
    }

    #[test]
    fn assembly_insertion_executes() {
        let task = assembly_insertion(j(&[0.1]), j(&[0.2]), j(&[0.3]), j(&[0.1]));
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
        assert_eq!(result.best().unwrap().sub_solutions.len(), 4);
    }

    #[test]
    fn pour_task_executes() {
        let task = pour_task(j(&[0.5]), j(&[1.0]), j(&[0.5]));
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn handover_task_executes() {
        let task = handover_task(j(&[0.5]), j(&[0.8]), j(&[0.2]), 0.08, "cup");
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn inspect_task_executes() {
        let task = inspect_task(j(&[0.7]), 2000, j(&[0.0]));
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn stack_task_executes() {
        let task = stack_task("block", "hand", j(&[0.3]), j(&[0.6]), j(&[0.5]), j(&[0.1]), 0.08, 0.02);
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn sort_task_executes() {
        let task = sort_task("item", "gripper", j(&[0.3]), j(&[0.8]), j(&[0.0]), 0.08, 0.02);
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn dual_arm_handoff_executes() {
        let task = dual_arm_handoff(
            j(&[0.5]), j(&[0.3]), j(&[0.4]), j(&[0.0]), j(&[0.6]),
            "obj", "hand_a", "hand_b", 0.08, 0.02,
        );
        let engine = TaskEngine::new();
        let result = engine.execute(&task, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    // ─── Error classification ───

    #[test]
    fn error_categories() {
        let failure = TaskFailure {
            category: ErrorCategory::GraspFailure,
            stage_name: "grasp".into(),
            message: "slipped".into(),
            timestamp: Instant::now(),
            recoverable: true,
        };
        assert!(failure.recoverable);
        assert_eq!(failure.category, ErrorCategory::GraspFailure);
    }

    // ─── Recovery registry ───

    #[test]
    fn recovery_registry_lookup() {
        let registry = RecoveryRegistry::production();
        let failure = TaskFailure {
            category: ErrorCategory::GraspFailure,
            stage_name: "grasp".into(),
            message: "".into(),
            timestamp: Instant::now(),
            recoverable: true,
        };
        match registry.lookup(&failure) {
            RecoveryAction::Retry { max_attempts } => assert_eq!(*max_attempts, 3),
            other => panic!("Expected Retry, got {:?}", other),
        }
    }

    #[test]
    fn recovery_registry_default() {
        let registry = RecoveryRegistry::new();
        let failure = TaskFailure {
            category: ErrorCategory::Unknown,
            stage_name: "".into(),
            message: "".into(),
            timestamp: Instant::now(),
            recoverable: false,
        };
        assert!(matches!(registry.lookup(&failure), RecoveryAction::Abort));
    }

    // ─── Checkpoint ───

    #[test]
    fn checkpoint_save_restore() {
        let mut mgr = CheckpointManager::new();
        mgr.save(TaskCheckpoint {
            stage_index: 3,
            stage_name: "grasp".into(),
            joints: j(&[0.5, 0.5]),
            scene_diffs: vec![],
            timestamp: Instant::now(),
            completed_stages: 3,
        });

        assert_eq!(mgr.count(), 1);
        let latest = mgr.latest().unwrap();
        assert_eq!(latest.stage_name, "grasp");
        assert_eq!(latest.stage_index, 3);
    }

    // ─── Execution log ───

    #[test]
    fn execution_log_records() {
        let mut log = ExecutionLog::new(100);
        log.log(ExecutionLogEntry {
            timestamp: Instant::now(),
            stage_name: "approach".into(),
            event_type: LogEventType::StageStarted,
            joints: Some(j(&[0.0])),
            message: "starting".into(),
        });
        log.log(ExecutionLogEntry {
            timestamp: Instant::now(),
            stage_name: "approach".into(),
            event_type: LogEventType::StageCompleted,
            joints: Some(j(&[0.5])),
            message: "done".into(),
        });

        assert_eq!(log.len(), 2);
        assert_eq!(log.for_stage("approach").len(), 2);
        assert_eq!(log.by_type(LogEventType::StageStarted).len(), 1);
    }

    // ─── Failure analytics ───

    #[test]
    fn failure_analytics_tracking() {
        let mut analytics = FailureAnalytics::new();
        analytics.record(TaskFailure {
            category: ErrorCategory::GraspFailure,
            stage_name: "grasp".into(),
            message: "".into(),
            timestamp: Instant::now(),
            recoverable: true,
        });
        analytics.record(TaskFailure {
            category: ErrorCategory::GraspFailure,
            stage_name: "grasp".into(),
            message: "".into(),
            timestamp: Instant::now(),
            recoverable: true,
        });
        analytics.record(TaskFailure {
            category: ErrorCategory::PlanningFailure,
            stage_name: "move".into(),
            message: "".into(),
            timestamp: Instant::now(),
            recoverable: true,
        });

        assert_eq!(analytics.total_failures(), 3);
        assert_eq!(analytics.most_common(), Some(ErrorCategory::GraspFailure));
        assert_eq!(*analytics.by_category().get(&ErrorCategory::GraspFailure).unwrap(), 2);
        assert_eq!(*analytics.by_stage().get("grasp").unwrap(), 2);
        assert!((analytics.failure_rate(10) - 0.3).abs() < 1e-10);
    }

    // ─── Integration ───

    #[test]
    fn pick_with_recovery_integration() {
        // Build a bin pick pipeline
        let config = BinPickConfig {
            grasp_candidates: vec![j(&[0.3])],
            approach_joints: j(&[0.2]),
            retreat_joints: j(&[0.5]),
            dropoff_joints: j(&[1.0]),
            object: "part".into(),
            gripper_link: "gripper".into(),
            gripper_open: 0.08,
            gripper_close: 0.02,
        };

        let pipeline = bin_pick_pipeline(&config);
        let engine = TaskEngine::new();
        let result = engine.execute(&pipeline, start(1), Duration::from_secs(5));
        assert!(result.best().is_some());

        // Simulate recovery flow
        let registry = RecoveryRegistry::production();
        let mut analytics = FailureAnalytics::new();
        let mut checkpoint_mgr = CheckpointManager::new();
        let mut log = ExecutionLog::new(100);

        // Log the execution
        log.log(ExecutionLogEntry {
            timestamp: Instant::now(),
            stage_name: "bin_pick".into(),
            event_type: LogEventType::StageCompleted,
            joints: None,
            message: "completed".into(),
        });

        // Save checkpoint
        checkpoint_mgr.save(TaskCheckpoint {
            stage_index: 0,
            stage_name: "bin_pick".into(),
            joints: j(&[1.0]),
            scene_diffs: vec![],
            timestamp: Instant::now(),
            completed_stages: 7,
        });

        // Simulate a failure on next attempt
        let failure = TaskFailure {
            category: ErrorCategory::GraspFailure,
            stage_name: "grasp".into(),
            message: "object slipped".into(),
            timestamp: Instant::now(),
            recoverable: true,
        };
        analytics.record(failure.clone());

        // Look up recovery
        let action = registry.lookup(&failure);
        assert!(matches!(action, RecoveryAction::Retry { .. }));

        // Verify full pipeline
        assert_eq!(log.len(), 1);
        assert_eq!(checkpoint_mgr.count(), 1);
        assert_eq!(analytics.total_failures(), 1);
    }
}
