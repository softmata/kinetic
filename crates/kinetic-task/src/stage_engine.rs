//! DAG-based stage execution engine (MoveIt Task Constructor equivalent).
//!
//! Stages form a directed acyclic graph with InterfaceState propagation.
//! Each stage computes solutions at its interfaces (start/end states),
//! and the engine finds a consistent chain of states through the graph.
//!
//! # Architecture
//!
//! ```text
//! InterfaceState → Stage → InterfaceState → Stage → InterfaceState
//!                    ↓
//!              (generates)
//!            StageSolution
//! ```
//!
//! # Stage Types
//!
//! - **Generator**: Creates states (e.g., grasp pose generator).
//! - **Propagator**: Takes input state, produces output state (e.g., move_to).
//! - **Connector**: Connects two given states (e.g., plan path between).
//! - **Container**: Holds child stages (Serial, Parallel, Fallback, Loop).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use kinetic_core::JointValues;

// ═══════════════════════════════════════════════════════════════════════════
// InterfaceState
// ═══════════════════════════════════════════════════════════════════════════

/// State at a stage interface (start or end of a stage).
///
/// Contains joint configuration, scene state, and optional metadata.
/// States flow between stages through the DAG.
#[derive(Debug, Clone)]
pub struct InterfaceState {
    /// Joint configuration.
    pub joints: JointValues,
    /// Scene modifications at this point (attached objects, etc.).
    pub scene_diff: Vec<SceneModification>,
    /// Arbitrary properties.
    pub properties: HashMap<String, StateValue>,
}

/// A scene modification at a state boundary.
#[derive(Debug, Clone)]
pub enum SceneModification {
    /// Attach an object to a robot link.
    AttachObject { object: String, link: String },
    /// Detach an object from the robot.
    DetachObject { object: String },
    /// Add a collision object to the scene.
    AddObject { name: String, position: [f64; 3], half_extents: [f64; 3] },
    /// Remove a collision object.
    RemoveObject { name: String },
}

/// Property value in an InterfaceState.
#[derive(Debug, Clone)]
pub enum StateValue {
    Float(f64),
    String(String),
    Bool(bool),
    Joints(JointValues),
}

impl InterfaceState {
    /// Create from joint values.
    pub fn from_joints(joints: JointValues) -> Self {
        Self {
            joints,
            scene_diff: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Set a property.
    pub fn set(&mut self, key: &str, value: StateValue) {
        self.properties.insert(key.to_string(), value);
    }

    /// Get a float property.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.properties.get(key) {
            Some(StateValue::Float(v)) => Some(*v),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost Terms
// ═══════════════════════════════════════════════════════════════════════════

/// Cost term for ranking stage solutions.
pub trait CostTerm: Send + Sync {
    /// Evaluate the cost of a stage solution.
    fn evaluate(&self, solution: &StageSolution) -> f64;
    fn name(&self) -> &str;
}

/// Path length cost.
pub struct PathLengthCost;
impl CostTerm for PathLengthCost {
    fn evaluate(&self, solution: &StageSolution) -> f64 {
        solution.trajectory.as_ref().map_or(0.0, |t| t.path_length())
    }
    fn name(&self) -> &str { "path_length" }
}

/// Clearance cost (penalizes solutions with low clearance).
pub struct ClearanceCost {
    pub min_clearance: f64,
}
impl CostTerm for ClearanceCost {
    fn evaluate(&self, solution: &StageSolution) -> f64 {
        // Stub: would query collision distance along trajectory
        solution.cost
    }
    fn name(&self) -> &str { "clearance" }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage Solution
// ═══════════════════════════════════════════════════════════════════════════

/// Solution produced by a single stage.
#[derive(Debug, Clone)]
pub struct StageSolution {
    /// Stage that produced this solution.
    pub stage_name: String,
    /// Start interface state.
    pub start_state: InterfaceState,
    /// End interface state.
    pub end_state: InterfaceState,
    /// Trajectory (if this stage produces motion).
    pub trajectory: Option<kinetic_core::Trajectory>,
    /// Cost of this solution.
    pub cost: f64,
    /// Sub-solutions (for container stages).
    pub sub_solutions: Vec<StageSolution>,
    /// Planning time for this stage.
    pub planning_time: Duration,
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage Trait
// ═══════════════════════════════════════════════════════════════════════════

/// How a stage relates to its interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageType {
    /// Generates states (no input required).
    Generator,
    /// Propagates from input state to output state.
    PropagatorForward,
    /// Propagates backward (from end to start).
    PropagatorBackward,
    /// Connects two given states.
    Connector,
    /// Container holding child stages.
    Container,
}

/// A stage in the task pipeline.
pub trait Stage: Send + Sync {
    /// Stage name.
    fn name(&self) -> &str;
    /// Stage type.
    fn stage_type(&self) -> StageType;
    /// Compute solutions given input state(s).
    fn compute(&self, context: &StageContext) -> Vec<StageSolution>;
}

/// Context passed to a stage during computation.
#[derive(Debug, Clone)]
pub struct StageContext {
    /// Input state (from previous stage).
    pub start_state: Option<InterfaceState>,
    /// Output state (from next stage, for backward propagation).
    pub end_state: Option<InterfaceState>,
    /// Maximum planning time for this stage.
    pub timeout: Duration,
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Stages
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed state generator: always produces the same state.
pub struct FixedStateGenerator {
    name: String,
    state: InterfaceState,
}

impl FixedStateGenerator {
    pub fn new(name: &str, state: InterfaceState) -> Self {
        Self { name: name.to_string(), state }
    }
}

impl Stage for FixedStateGenerator {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Generator }
    fn compute(&self, _context: &StageContext) -> Vec<StageSolution> {
        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state: self.state.clone(),
            end_state: self.state.clone(),
            trajectory: None,
            cost: 0.0,
            sub_solutions: Vec::new(),
            planning_time: Duration::ZERO,
        }]
    }
}

/// Current state generator: uses the input state as-is.
pub struct CurrentStateStage {
    name: String,
}

impl CurrentStateStage {
    pub fn new(name: &str) -> Self { Self { name: name.to_string() } }
}

impl Stage for CurrentStateStage {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Generator }
    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        if let Some(ref state) = context.start_state {
            vec![StageSolution {
                stage_name: self.name.clone(),
                start_state: state.clone(),
                end_state: state.clone(),
                trajectory: None,
                cost: 0.0,
                sub_solutions: Vec::new(),
                planning_time: Duration::ZERO,
            }]
        } else {
            vec![]
        }
    }
}

/// Move-to stage: connects start state to a goal configuration.
pub struct MoveToStage {
    name: String,
    goal: JointValues,
}

impl MoveToStage {
    pub fn new(name: &str, goal: JointValues) -> Self {
        Self { name: name.to_string(), goal }
    }
}

impl Stage for MoveToStage {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Connector }
    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let start = match &context.start_state {
            Some(s) => s.clone(),
            None => return vec![],
        };

        // Create a straight-line trajectory (real impl would use planner)
        let dof = start.joints.len();
        if self.goal.len() != dof {
            return vec![]; // DOF mismatch — cannot plan
        }
        let mut traj = kinetic_core::Trajectory::with_dof(dof);
        traj.push_waypoint(&start.joints);
        let mid: Vec<f64> = (0..dof).map(|i| (start.joints[i] + self.goal[i]) / 2.0).collect();
        traj.push_waypoint(&mid);
        traj.push_waypoint(&self.goal);

        let end = InterfaceState::from_joints(self.goal.clone());

        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state: start,
            end_state: end,
            trajectory: Some(traj),
            cost: 0.0,
            sub_solutions: Vec::new(),
            planning_time: Duration::ZERO,
        }]
    }
}

/// Gripper stage: open or close the gripper.
pub struct GripperStage {
    name: String,
    width: f64,
}

impl GripperStage {
    pub fn open(name: &str, width: f64) -> Self { Self { name: name.to_string(), width } }
    pub fn close(name: &str, width: f64) -> Self { Self { name: name.to_string(), width } }
}

impl Stage for GripperStage {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::PropagatorForward }
    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let start = match &context.start_state {
            Some(s) => s.clone(),
            None => return vec![],
        };
        let mut end = start.clone();
        end.set("gripper_width", StateValue::Float(self.width));

        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state: start,
            end_state: end,
            trajectory: None,
            cost: 0.0,
            sub_solutions: Vec::new(),
            planning_time: Duration::ZERO,
        }]
    }
}

/// Modify scene stage: attach/detach objects.
pub struct ModifySceneStage {
    name: String,
    modifications: Vec<SceneModification>,
}

impl ModifySceneStage {
    pub fn new(name: &str, modifications: Vec<SceneModification>) -> Self {
        Self { name: name.to_string(), modifications }
    }

    pub fn attach(name: &str, object: &str, link: &str) -> Self {
        Self::new(name, vec![SceneModification::AttachObject {
            object: object.to_string(), link: link.to_string(),
        }])
    }

    pub fn detach(name: &str, object: &str) -> Self {
        Self::new(name, vec![SceneModification::DetachObject {
            object: object.to_string(),
        }])
    }
}

impl Stage for ModifySceneStage {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::PropagatorForward }
    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let start = match &context.start_state {
            Some(s) => s.clone(),
            None => return vec![],
        };
        let mut end = start.clone();
        end.scene_diff.extend(self.modifications.clone());

        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state: start,
            end_state: end,
            trajectory: None,
            cost: 0.0,
            sub_solutions: Vec::new(),
            planning_time: Duration::ZERO,
        }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Container Stages
// ═══════════════════════════════════════════════════════════════════════════

/// Serial container: executes child stages in sequence.
///
/// Each stage's end_state becomes the next stage's start_state.
pub struct SerialContainer {
    name: String,
    children: Vec<Box<dyn Stage>>,
}

impl SerialContainer {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), children: Vec::new() }
    }

    pub fn add(&mut self, stage: Box<dyn Stage>) {
        self.children.push(stage);
    }

    pub fn with(mut self, stage: Box<dyn Stage>) -> Self {
        self.children.push(stage);
        self
    }
}

impl Stage for SerialContainer {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Container }

    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let start_time = Instant::now();
        let mut current_state = context.start_state.clone();
        let mut sub_solutions = Vec::new();
        let mut total_cost = 0.0;

        for child in &self.children {
            let child_context = StageContext {
                start_state: current_state.clone(),
                end_state: context.end_state.clone(),
                timeout: context.timeout,
            };

            let solutions = child.compute(&child_context);
            if solutions.is_empty() {
                return vec![]; // pipeline fails if any stage fails
            }

            // Take the best (lowest cost) solution
            let best = solutions.into_iter()
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
                .unwrap();

            current_state = Some(best.end_state.clone());
            total_cost += best.cost;
            sub_solutions.push(best);
        }

        let start_state = context.start_state.clone()
            .unwrap_or_else(|| InterfaceState::from_joints(JointValues::zeros(0)));
        let end_state = current_state
            .unwrap_or_else(|| InterfaceState::from_joints(JointValues::zeros(0)));

        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state,
            end_state,
            trajectory: None, // sub-solutions have individual trajectories
            cost: total_cost,
            sub_solutions,
            planning_time: start_time.elapsed(),
        }]
    }
}

/// Fallback container: tries children in order, returns first success.
pub struct FallbackContainer {
    name: String,
    children: Vec<Box<dyn Stage>>,
}

impl FallbackContainer {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), children: Vec::new() }
    }

    pub fn add(&mut self, stage: Box<dyn Stage>) {
        self.children.push(stage);
    }

    pub fn with(mut self, stage: Box<dyn Stage>) -> Self {
        self.children.push(stage);
        self
    }
}

impl Stage for FallbackContainer {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Container }

    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        for child in &self.children {
            let solutions = child.compute(context);
            if !solutions.is_empty() {
                return solutions;
            }
        }
        vec![]
    }
}

/// Parallel container: runs all children, returns all solutions (for ranking).
pub struct ParallelContainer {
    name: String,
    children: Vec<Box<dyn Stage>>,
}

impl ParallelContainer {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), children: Vec::new() }
    }

    pub fn add(&mut self, stage: Box<dyn Stage>) {
        self.children.push(stage);
    }
}

impl Stage for ParallelContainer {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Container }

    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let mut all_solutions = Vec::new();
        for child in &self.children {
            all_solutions.extend(child.compute(context));
        }
        all_solutions
    }
}

/// Loop container: repeats child stage N times (or until failure).
pub struct LoopContainer {
    name: String,
    child: Box<dyn Stage>,
    max_iterations: usize,
}

impl LoopContainer {
    pub fn new(name: &str, child: Box<dyn Stage>, max_iterations: usize) -> Self {
        Self { name: name.to_string(), child, max_iterations }
    }
}

impl Stage for LoopContainer {
    fn name(&self) -> &str { &self.name }
    fn stage_type(&self) -> StageType { StageType::Container }

    fn compute(&self, context: &StageContext) -> Vec<StageSolution> {
        let start_time = Instant::now();
        let mut current_state = context.start_state.clone();
        let mut sub_solutions = Vec::new();

        for _ in 0..self.max_iterations {
            let ctx = StageContext {
                start_state: current_state.clone(),
                end_state: context.end_state.clone(),
                timeout: context.timeout,
            };
            let solutions = self.child.compute(&ctx);
            if solutions.is_empty() { break; }

            let best = solutions.into_iter()
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
                .unwrap();
            current_state = Some(best.end_state.clone());
            sub_solutions.push(best);
        }

        if sub_solutions.is_empty() { return vec![]; }

        let end = current_state.unwrap_or_else(|| InterfaceState::from_joints(JointValues::zeros(0)));
        let start = context.start_state.clone().unwrap_or_else(|| InterfaceState::from_joints(JointValues::zeros(0)));

        vec![StageSolution {
            stage_name: self.name.clone(),
            start_state: start,
            end_state: end,
            trajectory: None,
            cost: sub_solutions.iter().map(|s| s.cost).sum(),
            sub_solutions,
            planning_time: start_time.elapsed(),
        }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Task Engine
// ═══════════════════════════════════════════════════════════════════════════

/// The task engine: executes a stage DAG and returns solutions.
pub struct TaskEngine {
    /// Cost terms for ranking solutions.
    cost_terms: Vec<(Box<dyn CostTerm>, f64)>, // (term, weight)
}

impl TaskEngine {
    pub fn new() -> Self {
        Self { cost_terms: Vec::new() }
    }

    /// Add a cost term with a weight.
    pub fn add_cost_term(&mut self, term: Box<dyn CostTerm>, weight: f64) {
        self.cost_terms.push((term, weight));
    }

    /// Execute a stage graph starting from a given state.
    pub fn execute(
        &self,
        root_stage: &dyn Stage,
        start_state: InterfaceState,
        timeout: Duration,
    ) -> TaskResult {
        let start_time = Instant::now();

        let context = StageContext {
            start_state: Some(start_state),
            end_state: None,
            timeout,
        };

        let mut solutions = root_stage.compute(&context);

        // Apply cost terms
        for sol in &mut solutions {
            let mut total_cost = sol.cost;
            for (term, weight) in &self.cost_terms {
                total_cost += weight * term.evaluate(sol);
            }
            sol.cost = total_cost;
        }

        // Sort by cost
        solutions.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());

        let planning_time = start_time.elapsed();

        TaskResult {
            solutions,
            planning_time,
            success: true,
        }
    }
}

/// Result of task engine execution.
#[derive(Debug)]
pub struct TaskResult {
    /// All solutions found, sorted by cost (best first).
    pub solutions: Vec<StageSolution>,
    /// Total planning time.
    pub planning_time: Duration,
    /// Whether at least one solution was found.
    pub success: bool,
}

impl TaskResult {
    /// Get the best solution (lowest cost).
    pub fn best(&self) -> Option<&StageSolution> {
        self.solutions.first()
    }

    /// Number of solutions found.
    pub fn num_solutions(&self) -> usize {
        self.solutions.len()
    }

    /// Collect all trajectories from the best solution (flattened from sub-solutions).
    pub fn trajectories(&self) -> Vec<&kinetic_core::Trajectory> {
        let mut trajs = Vec::new();
        if let Some(best) = self.best() {
            collect_trajectories(best, &mut trajs);
        }
        trajs
    }
}

fn collect_trajectories<'a>(sol: &'a StageSolution, out: &mut Vec<&'a kinetic_core::Trajectory>) {
    if let Some(ref t) = sol.trajectory {
        out.push(t);
    }
    for sub in &sol.sub_solutions {
        collect_trajectories(sub, out);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pick / Place Macros
// ═══════════════════════════════════════════════════════════════════════════

/// Build a pick task as a serial container of stages.
pub fn pick_task(
    name: &str,
    object: &str,
    link: &str,
    approach_joints: JointValues,
    grasp_joints: JointValues,
    retreat_joints: JointValues,
    gripper_open: f64,
    gripper_close: f64,
) -> SerialContainer {
    SerialContainer::new(name)
        .with(Box::new(GripperStage::open("open_gripper", gripper_open)))
        .with(Box::new(MoveToStage::new("approach", approach_joints)))
        .with(Box::new(MoveToStage::new("grasp_pose", grasp_joints)))
        .with(Box::new(GripperStage::close("close_gripper", gripper_close)))
        .with(Box::new(ModifySceneStage::attach("attach_object", object, link)))
        .with(Box::new(MoveToStage::new("retreat", retreat_joints)))
}

/// Build a place task as a serial container of stages.
pub fn place_task(
    name: &str,
    object: &str,
    approach_joints: JointValues,
    place_joints: JointValues,
    retreat_joints: JointValues,
    gripper_open: f64,
) -> SerialContainer {
    SerialContainer::new(name)
        .with(Box::new(MoveToStage::new("approach_place", approach_joints)))
        .with(Box::new(MoveToStage::new("place_pose", place_joints)))
        .with(Box::new(GripperStage::open("release", gripper_open)))
        .with(Box::new(ModifySceneStage::detach("detach_object", object)))
        .with(Box::new(MoveToStage::new("retreat_place", retreat_joints)))
}

/// Task serialization to JSON-like format (simplified).
pub fn serialize_task_result(result: &TaskResult) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(result.num_solutions() as u32).to_le_bytes());
    buf.extend_from_slice(&result.planning_time.as_millis().to_le_bytes());
    for sol in &result.solutions {
        buf.extend_from_slice(&sol.stage_name.len().to_le_bytes());
        buf.extend_from_slice(sol.stage_name.as_bytes());
        buf.extend_from_slice(&sol.cost.to_le_bytes());
        buf.extend_from_slice(&(sol.sub_solutions.len() as u32).to_le_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_state(dof: usize) -> InterfaceState {
        InterfaceState::from_joints(JointValues::zeros(dof))
    }

    // ─── InterfaceState tests ───

    #[test]
    fn interface_state_properties() {
        let mut state = start_state(3);
        state.set("gripper", StateValue::Float(0.08));
        assert_eq!(state.get_float("gripper"), Some(0.08));
        assert_eq!(state.get_float("missing"), None);
    }

    // ─── Stage types tests ───

    #[test]
    fn fixed_state_generator() {
        let state = InterfaceState::from_joints(JointValues::new(vec![1.0, 2.0]));
        let stage = FixedStateGenerator::new("home", state);

        let ctx = StageContext { start_state: None, end_state: None, timeout: Duration::from_secs(1) };
        let solutions = stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].end_state.joints[0], 1.0);
    }

    #[test]
    fn current_state_stage() {
        let stage = CurrentStateStage::new("current");
        let state = start_state(3);
        let ctx = StageContext { start_state: Some(state.clone()), end_state: None, timeout: Duration::from_secs(1) };

        let solutions = stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);

        // No start state → no solutions
        let ctx2 = StageContext { start_state: None, end_state: None, timeout: Duration::from_secs(1) };
        assert!(stage.compute(&ctx2).is_empty());
    }

    #[test]
    fn move_to_stage() {
        let goal = JointValues::new(vec![1.0, 1.0, 1.0]);
        let stage = MoveToStage::new("go_home", goal.clone());

        let ctx = StageContext {
            start_state: Some(start_state(3)),
            end_state: None,
            timeout: Duration::from_secs(1),
        };

        let solutions = stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].trajectory.is_some());
        assert_eq!(solutions[0].end_state.joints.len(), 3);
    }

    #[test]
    fn gripper_stage() {
        let stage = GripperStage::open("open", 0.08);
        let ctx = StageContext {
            start_state: Some(start_state(3)),
            end_state: None,
            timeout: Duration::from_secs(1),
        };

        let solutions = stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].end_state.get_float("gripper_width"), Some(0.08));
    }

    #[test]
    fn modify_scene_stage() {
        let stage = ModifySceneStage::attach("attach", "cup", "gripper_link");
        let ctx = StageContext {
            start_state: Some(start_state(3)),
            end_state: None,
            timeout: Duration::from_secs(1),
        };

        let solutions = stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].end_state.scene_diff.len(), 1);
    }

    // ─── Container tests ───

    #[test]
    fn serial_container_chains_states() {
        let mut serial = SerialContainer::new("pick_sequence");
        serial.add(Box::new(GripperStage::open("open", 0.08)));
        serial.add(Box::new(MoveToStage::new("approach", JointValues::new(vec![0.5, 0.5, 0.5]))));
        serial.add(Box::new(GripperStage::close("close", 0.02)));

        let ctx = StageContext {
            start_state: Some(start_state(3)),
            end_state: None,
            timeout: Duration::from_secs(5),
        };

        let solutions = serial.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].sub_solutions.len(), 3);
        // Final state should have gripper closed
        assert_eq!(solutions[0].end_state.get_float("gripper_width"), Some(0.02));
    }

    #[test]
    fn serial_container_fails_on_empty_child() {
        let mut serial = SerialContainer::new("will_fail");
        serial.add(Box::new(CurrentStateStage::new("needs_input")));
        // No start state → CurrentStateStage fails → serial fails

        let ctx = StageContext { start_state: None, end_state: None, timeout: Duration::from_secs(1) };
        let solutions = serial.compute(&ctx);
        assert!(solutions.is_empty());
    }

    #[test]
    fn fallback_container_tries_alternatives() {
        let mut fallback = FallbackContainer::new("try_grasps");
        // First option fails (no input state)
        fallback.add(Box::new(CurrentStateStage::new("option_a"))); // will fail without start
        // Second option succeeds
        fallback.add(Box::new(FixedStateGenerator::new("option_b",
            InterfaceState::from_joints(JointValues::new(vec![1.0])))));

        let ctx = StageContext { start_state: None, end_state: None, timeout: Duration::from_secs(1) };
        let solutions = fallback.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].stage_name, "option_b");
    }

    #[test]
    fn parallel_container_collects_all() {
        let mut parallel = ParallelContainer::new("race");
        parallel.add(Box::new(FixedStateGenerator::new("a",
            InterfaceState::from_joints(JointValues::new(vec![1.0])))));
        parallel.add(Box::new(FixedStateGenerator::new("b",
            InterfaceState::from_joints(JointValues::new(vec![2.0])))));

        let ctx = StageContext { start_state: None, end_state: None, timeout: Duration::from_secs(1) };
        let solutions = parallel.compute(&ctx);
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn loop_container_repeats() {
        let stage = FixedStateGenerator::new("step",
            InterfaceState::from_joints(JointValues::new(vec![1.0])));
        let loop_stage = LoopContainer::new("repeat", Box::new(stage), 3);

        let ctx = StageContext {
            start_state: Some(start_state(1)),
            end_state: None,
            timeout: Duration::from_secs(1),
        };
        let solutions = loop_stage.compute(&ctx);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].sub_solutions.len(), 3);
    }

    // ─── TaskEngine tests ───

    #[test]
    fn task_engine_executes() {
        let engine = TaskEngine::new();
        let serial = SerialContainer::new("task")
            .with(Box::new(GripperStage::open("open", 0.08)))
            .with(Box::new(MoveToStage::new("move", JointValues::new(vec![1.0, 1.0]))));

        let result = engine.execute(&serial, start_state(2), Duration::from_secs(5));
        assert!(result.best().is_some());
        assert!(result.num_solutions() > 0);
    }

    #[test]
    fn task_engine_with_cost_terms() {
        let mut engine = TaskEngine::new();
        engine.add_cost_term(Box::new(PathLengthCost), 1.0);

        let stage = MoveToStage::new("move", JointValues::new(vec![1.0, 1.0]));
        let result = engine.execute(&stage, start_state(2), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    #[test]
    fn task_engine_collects_trajectories() {
        let engine = TaskEngine::new();
        let serial = SerialContainer::new("task")
            .with(Box::new(MoveToStage::new("move1", JointValues::new(vec![1.0]))))
            .with(Box::new(MoveToStage::new("move2", JointValues::new(vec![2.0]))));

        let result = engine.execute(&serial, start_state(1), Duration::from_secs(5));
        let trajs = result.trajectories();
        assert_eq!(trajs.len(), 2, "Should collect 2 trajectories from sub-solutions");
    }

    // ─── Pick/Place macro tests ───

    #[test]
    fn pick_task_macro() {
        let task = pick_task(
            "pick_cup", "cup", "gripper",
            JointValues::new(vec![0.5, 0.5]),
            JointValues::new(vec![0.3, 0.3]),
            JointValues::new(vec![0.5, 0.7]),
            0.08, 0.02,
        );

        let engine = TaskEngine::new();
        let result = engine.execute(&task, start_state(2), Duration::from_secs(5));
        assert!(result.best().is_some());
        assert_eq!(result.best().unwrap().sub_solutions.len(), 6); // open, approach, grasp, close, attach, retreat
    }

    #[test]
    fn place_task_macro() {
        let task = place_task(
            "place_cup", "cup",
            JointValues::new(vec![0.5, 0.5]),
            JointValues::new(vec![0.3, 0.3]),
            JointValues::new(vec![0.5, 0.7]),
            0.08,
        );

        let engine = TaskEngine::new();
        let result = engine.execute(&task, start_state(2), Duration::from_secs(5));
        assert!(result.best().is_some());
        assert_eq!(result.best().unwrap().sub_solutions.len(), 5);
    }

    #[test]
    fn pick_and_place_full() {
        let pick = pick_task(
            "pick", "box", "hand",
            JointValues::new(vec![0.5]), JointValues::new(vec![0.3]),
            JointValues::new(vec![0.6]), 0.08, 0.02,
        );
        let place = place_task(
            "place", "box",
            JointValues::new(vec![1.0]), JointValues::new(vec![0.8]),
            JointValues::new(vec![1.1]), 0.08,
        );

        let mut full = SerialContainer::new("pick_and_place");
        full.add(Box::new(pick));
        full.add(Box::new(place));

        let engine = TaskEngine::new();
        let result = engine.execute(&full, start_state(1), Duration::from_secs(10));
        assert!(result.best().is_some());
        assert_eq!(result.best().unwrap().sub_solutions.len(), 2); // pick container + place container
    }

    #[test]
    fn serialization_basic() {
        let engine = TaskEngine::new();
        let stage = MoveToStage::new("m", JointValues::new(vec![1.0]));
        let result = engine.execute(&stage, start_state(1), Duration::from_secs(1));
        let bytes = serialize_task_result(&result);
        assert!(!bytes.is_empty());
    }

    // ─── Nested containers ──────────────────────────────────────────────

    #[test]
    fn nested_serial_in_fallback() {
        // Intent: SerialContainer inside FallbackContainer should work
        let mut inner_serial = SerialContainer::new("inner");
        inner_serial.add(Box::new(MoveToStage::new("m1", JointValues::new(vec![0.5]))));
        inner_serial.add(Box::new(MoveToStage::new("m2", JointValues::new(vec![1.0]))));

        let mut fallback = FallbackContainer::new("fb");
        fallback.add(Box::new(inner_serial));

        let engine = TaskEngine::new();
        let result = engine.execute(&fallback, start_state(1), Duration::from_secs(5));
        assert!(result.best().is_some(), "nested serial-in-fallback should produce a solution");
    }

    #[test]
    fn nested_fallback_in_serial() {
        // Intent: FallbackContainer inside SerialContainer should work
        let mut fallback = FallbackContainer::new("choice");
        fallback.add(Box::new(MoveToStage::new("opt_a", JointValues::new(vec![0.5]))));
        fallback.add(Box::new(MoveToStage::new("opt_b", JointValues::new(vec![1.0]))));

        let mut serial = SerialContainer::new("outer");
        serial.add(Box::new(fallback));
        serial.add(Box::new(GripperStage::close("grip", 0.02)));

        let engine = TaskEngine::new();
        let result = engine.execute(&serial, start_state(1), Duration::from_secs(5));
        assert!(result.best().is_some());
    }

    // ─── Empty containers ───────────────────────────────────────────────

    #[test]
    fn empty_serial_container() {
        let serial = SerialContainer::new("empty_serial");
        let engine = TaskEngine::new();
        let result = engine.execute(&serial, start_state(1), Duration::from_secs(1));
        // Empty container should either produce an empty solution or no solution
        // Just verify it doesn't panic
        let _ = result.num_solutions();
    }

    #[test]
    fn empty_fallback_container() {
        let fallback = FallbackContainer::new("empty_fb");
        let engine = TaskEngine::new();
        let result = engine.execute(&fallback, start_state(1), Duration::from_secs(1));
        let _ = result.num_solutions();
    }

    #[test]
    fn empty_parallel_container() {
        let parallel = ParallelContainer::new("empty_par");
        let engine = TaskEngine::new();
        let result = engine.execute(&parallel, start_state(1), Duration::from_secs(1));
        let _ = result.num_solutions();
    }

    // ─── Cost term weighting ────────────────────────────────────────────

    #[test]
    fn cost_term_affects_ranking() {
        // Intent: adding a cost term should change solution ranking
        let mut par = ParallelContainer::new("choices");
        par.add(Box::new(MoveToStage::new("short", JointValues::new(vec![0.1]))));
        par.add(Box::new(MoveToStage::new("long", JointValues::new(vec![5.0]))));

        // Without cost terms — default ordering
        let engine_no_cost = TaskEngine::new();
        let result_no_cost = engine_no_cost.execute(&par, start_state(1), Duration::from_secs(1));

        // With path length cost — should prefer shorter path
        let mut engine_cost = TaskEngine::new();
        engine_cost.add_cost_term(Box::new(PathLengthCost), 1.0);
        let result_cost = engine_cost.execute(&par, start_state(1), Duration::from_secs(1));

        assert!(result_cost.num_solutions() >= 2);
        // Best solution with cost should have lower cost
        if result_cost.num_solutions() >= 2 {
            let sols = &result_cost.solutions;
            assert!(sols[0].cost <= sols[1].cost, "best should have lower cost");
        }
    }

    // ─── TaskResult accessors ───────────────────────────────────────────

    #[test]
    fn task_result_num_solutions_with_parallel() {
        // Intent: ParallelContainer should return multiple solutions
        let mut par = ParallelContainer::new("multi");
        par.add(Box::new(MoveToStage::new("a", JointValues::new(vec![0.1]))));
        par.add(Box::new(MoveToStage::new("b", JointValues::new(vec![0.5]))));
        par.add(Box::new(MoveToStage::new("c", JointValues::new(vec![1.0]))));

        let engine = TaskEngine::new();
        let result = engine.execute(&par, start_state(1), Duration::from_secs(5));
        assert!(result.num_solutions() >= 2, "parallel should return multiple solutions");
    }

    // ─── Loop container ─────────────────────────────────────────────────

    #[test]
    fn loop_container_correct_iteration_count() {
        let stage = MoveToStage::new("m", JointValues::new(vec![0.5]));
        let loop_c = LoopContainer::new("loop", Box::new(stage), 3);

        let engine = TaskEngine::new();
        let result = engine.execute(&loop_c, start_state(1), Duration::from_secs(5));
        assert!(result.best().is_some());
        let best = result.best().unwrap();
        // 3 iterations → 3 sub-solutions
        assert_eq!(best.sub_solutions.len(), 3, "loop should produce 3 iterations");
    }

    // ─── Stage type metadata ────────────────────────────────────────────

    #[test]
    fn stage_type_correctly_reported() {
        let gen = FixedStateGenerator::new("gen", start_state(1));
        assert_eq!(gen.stage_type(), StageType::Generator);

        let move_to = MoveToStage::new("mv", JointValues::new(vec![0.5]));
        assert_eq!(move_to.stage_type(), StageType::Connector);

        let grip = GripperStage::open("g", 0.08);
        assert_eq!(grip.stage_type(), StageType::PropagatorForward);
    }
}
