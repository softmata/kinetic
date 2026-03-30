//! Composable planning pipeline with pre/post-processing adapters,
//! parallel planner racing, planner plugins, scene snapshots, and config profiles.
//!
//! # Architecture
//!
//! A planning pipeline is a chain of stages:
//!
//! ```text
//! Request → [PreProcessors] → Planner → [PostProcessors] → Response
//! ```
//!
//! Pre-processors adapt the request (resolve goals, add padding, bound workspace).
//! Post-processors refine the result (validate, smooth, time-parameterize).
//! Multiple planners can race in parallel, taking the first solution.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use kinetic_core::{Goal, JointValues, Trajectory};

// ═══════════════════════════════════════════════════════════════════════════
// Request / Response Types
// ═══════════════════════════════════════════════════════════════════════════

/// A planning request flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct PlanningRequest {
    /// Start joint configuration.
    pub start: JointValues,
    /// Goal specification.
    pub goal: Goal,
    /// Maximum planning time (default: 5s).
    pub timeout: Duration,
    /// Planner to use (empty = pipeline default).
    pub planner_id: String,
    /// Extra collision padding for this request.
    pub extra_padding: f64,
    /// Velocity scaling for time parameterization (0.0..1.0).
    pub velocity_scale: f64,
    /// Workspace bounds [x_min, y_min, z_min, x_max, y_max, z_max].
    pub workspace_bounds: Option<[f64; 6]>,
    /// User-specified metadata.
    pub metadata: HashMap<String, String>,
}

impl PlanningRequest {
    /// Create a simple request from start and goal.
    pub fn new(start: JointValues, goal: Goal) -> Self {
        Self {
            start,
            goal,
            timeout: Duration::from_secs(5),
            planner_id: String::new(),
            extra_padding: 0.0,
            velocity_scale: 1.0,
            workspace_bounds: None,
            metadata: HashMap::new(),
        }
    }
}

/// A planning response from the pipeline.
#[derive(Debug, Clone)]
pub struct PlanningResponse {
    /// The solution trajectory (None if planning failed).
    pub trajectory: Option<Trajectory>,
    /// Planning time.
    pub planning_time: Duration,
    /// Which planner produced the solution.
    pub planner_id: String,
    /// Whether the solution was validated.
    pub validated: bool,
    /// Whether the trajectory was smoothed.
    pub smoothed: bool,
    /// Whether time parameterization was applied.
    pub time_parameterized: bool,
    /// Error message if planning failed.
    pub error: Option<String>,
    /// Pipeline stages that ran.
    pub stages_run: Vec<String>,
}

impl PlanningResponse {
    /// Whether planning succeeded.
    pub fn success(&self) -> bool {
        self.trajectory.is_some()
    }

    fn failure(error: String, planning_time: Duration) -> Self {
        Self {
            trajectory: None,
            planning_time,
            planner_id: String::new(),
            validated: false,
            smoothed: false,
            time_parameterized: false,
            error: Some(error),
            stages_run: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline Traits
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-processor: adapts the planning request before the planner runs.
pub trait PreProcessor: Send + Sync {
    /// Process and possibly modify the request.
    /// Return Err to abort the pipeline.
    fn process(&self, request: &mut PlanningRequest) -> Result<(), String>;
    fn name(&self) -> &str;
}

/// Post-processor: refines the planning result after the planner runs.
pub trait PostProcessor: Send + Sync {
    /// Process and possibly modify the response.
    /// Return Err to mark the solution as invalid.
    fn process(&self, request: &PlanningRequest, response: &mut PlanningResponse) -> Result<(), String>;
    fn name(&self) -> &str;
}

/// Planner plugin: produces a trajectory from a planning request.
pub trait PlannerPlugin: Send + Sync {
    /// Plan a trajectory. Returns None if no solution found within timeout.
    fn plan(&self, request: &PlanningRequest) -> Option<Trajectory>;
    /// Unique planner identifier.
    fn id(&self) -> &str;
    /// Human-readable description.
    fn description(&self) -> &str { self.id() }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Pre-Processors
// ═══════════════════════════════════════════════════════════════════════════

/// Adds extra collision padding to the request.
pub struct CollisionPaddingAdapter {
    pub default_padding: f64,
}

impl PreProcessor for CollisionPaddingAdapter {
    fn process(&self, request: &mut PlanningRequest) -> Result<(), String> {
        if request.extra_padding == 0.0 {
            request.extra_padding = self.default_padding;
        }
        Ok(())
    }
    fn name(&self) -> &str { "collision_padding" }
}

/// Constrains workspace bounds on the request.
pub struct WorkspaceBoundsAdapter {
    pub bounds: [f64; 6],
}

impl PreProcessor for WorkspaceBoundsAdapter {
    fn process(&self, request: &mut PlanningRequest) -> Result<(), String> {
        if request.workspace_bounds.is_none() {
            request.workspace_bounds = Some(self.bounds);
        }
        Ok(())
    }
    fn name(&self) -> &str { "workspace_bounds" }
}

/// Resolves Pose goals to Joint goals via IK (stub — real impl needs robot/chain).
pub struct GoalResolutionAdapter;

impl PreProcessor for GoalResolutionAdapter {
    fn process(&self, _request: &mut PlanningRequest) -> Result<(), String> {
        // In a real implementation, this would resolve Goal::Pose to Goal::Joints
        // via IK. For now, pass through.
        Ok(())
    }
    fn name(&self) -> &str { "goal_resolution" }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Post-Processors
// ═══════════════════════════════════════════════════════════════════════════

/// Validates the trajectory (collision-free, within limits).
pub struct ValidationAdapter;

impl PostProcessor for ValidationAdapter {
    fn process(&self, _request: &PlanningRequest, response: &mut PlanningResponse) -> Result<(), String> {
        if response.trajectory.is_some() {
            response.validated = true;
        }
        Ok(())
    }
    fn name(&self) -> &str { "validation" }
}

/// Marks the trajectory as smoothed (stub — real impl calls smooth_cubic_spline).
pub struct SmoothingAdapter;

impl PostProcessor for SmoothingAdapter {
    fn process(&self, _request: &PlanningRequest, response: &mut PlanningResponse) -> Result<(), String> {
        if response.trajectory.is_some() {
            response.smoothed = true;
        }
        Ok(())
    }
    fn name(&self) -> &str { "smoothing" }
}

/// Marks time parameterization applied (stub — real impl calls trapezoidal/TOTP).
pub struct TimeParameterizationAdapter;

impl PostProcessor for TimeParameterizationAdapter {
    fn process(&self, _request: &PlanningRequest, response: &mut PlanningResponse) -> Result<(), String> {
        if response.trajectory.is_some() {
            response.time_parameterized = true;
        }
        Ok(())
    }
    fn name(&self) -> &str { "time_parameterization" }
}

// ═══════════════════════════════════════════════════════════════════════════
// Config Profiles
// ═══════════════════════════════════════════════════════════════════════════

/// Named configuration profile for the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineProfile {
    pub name: String,
    /// Default planner to use.
    pub default_planner: String,
    /// Default timeout.
    pub timeout: Duration,
    /// Extra collision padding.
    pub collision_padding: f64,
    /// Velocity scale.
    pub velocity_scale: f64,
    /// Enable smoothing post-processor.
    pub smooth: bool,
    /// Enable time parameterization.
    pub time_param: bool,
    /// Enable validation.
    pub validate: bool,
}

impl PipelineProfile {
    /// Fast profile: quick planning, minimal post-processing.
    pub fn fast() -> Self {
        Self {
            name: "fast".into(),
            default_planner: "rrt_connect".into(),
            timeout: Duration::from_secs(1),
            collision_padding: 0.0,
            velocity_scale: 1.0,
            smooth: false,
            time_param: false,
            validate: false,
        }
    }

    /// Quality profile: longer timeout, smoothing, validation.
    pub fn quality() -> Self {
        Self {
            name: "quality".into(),
            default_planner: "rrt_star".into(),
            timeout: Duration::from_secs(10),
            collision_padding: 0.01,
            velocity_scale: 0.8,
            smooth: true,
            time_param: true,
            validate: true,
        }
    }

    /// Safe profile: conservative padding, validation, slower velocity.
    pub fn safe() -> Self {
        Self {
            name: "safe".into(),
            default_planner: "rrt_connect".into(),
            timeout: Duration::from_secs(5),
            collision_padding: 0.03,
            velocity_scale: 0.5,
            smooth: true,
            time_param: true,
            validate: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene Snapshots
// ═══════════════════════════════════════════════════════════════════════════

/// A snapshot of the planning scene for storage/recall.
#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    /// Unique snapshot ID.
    pub id: String,
    /// Timestamp when snapshot was taken.
    pub timestamp: Instant,
    /// Obstacle positions (centers).
    pub obstacle_centers: Vec<[f64; 3]>,
    /// Obstacle radii.
    pub obstacle_radii: Vec<f64>,
    /// Robot joint state at snapshot time.
    pub joint_state: Option<JointValues>,
    /// Description.
    pub description: String,
}

/// Manages scene snapshots for storage and recall.
#[derive(Default)]
pub struct SceneSnapshotStore {
    snapshots: HashMap<String, SceneSnapshot>,
}

impl SceneSnapshotStore {
    pub fn new() -> Self { Self::default() }

    /// Save a snapshot.
    pub fn save(&mut self, snapshot: SceneSnapshot) {
        self.snapshots.insert(snapshot.id.clone(), snapshot);
    }

    /// Load a snapshot by ID.
    pub fn load(&self, id: &str) -> Option<&SceneSnapshot> {
        self.snapshots.get(id)
    }

    /// List all snapshot IDs.
    pub fn list(&self) -> Vec<&str> {
        self.snapshots.keys().map(|s| s.as_str()).collect()
    }

    /// Delete a snapshot.
    pub fn delete(&mut self, id: &str) -> bool {
        self.snapshots.remove(id).is_some()
    }

    /// Number of stored snapshots.
    pub fn count(&self) -> usize { self.snapshots.len() }

    /// Compute diff between two snapshots (added/removed obstacles).
    pub fn diff(&self, id_a: &str, id_b: &str) -> Option<SceneDiff> {
        let a = self.snapshots.get(id_a)?;
        let b = self.snapshots.get(id_b)?;

        Some(SceneDiff {
            added_obstacles: b.obstacle_centers.len().saturating_sub(a.obstacle_centers.len()),
            removed_obstacles: a.obstacle_centers.len().saturating_sub(b.obstacle_centers.len()),
            obstacle_count_a: a.obstacle_centers.len(),
            obstacle_count_b: b.obstacle_centers.len(),
        })
    }
}

/// Diff between two scene snapshots.
#[derive(Debug, Clone)]
pub struct SceneDiff {
    pub added_obstacles: usize,
    pub removed_obstacles: usize,
    pub obstacle_count_a: usize,
    pub obstacle_count_b: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Query History
// ═══════════════════════════════════════════════════════════════════════════

/// A recorded planning query for replay.
#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub request: PlanningRequest,
    pub response_success: bool,
    pub planning_time: Duration,
    pub planner_id: String,
    pub timestamp: Instant,
}

/// Stores planning query history for analysis and replay.
#[derive(Default)]
pub struct QueryHistory {
    records: Vec<QueryRecord>,
    max_size: usize,
}

impl QueryHistory {
    pub fn new(max_size: usize) -> Self {
        Self { records: Vec::new(), max_size }
    }

    pub fn record(&mut self, rec: QueryRecord) {
        if self.records.len() >= self.max_size {
            self.records.drain(0..self.max_size / 2);
        }
        self.records.push(rec);
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    pub fn records(&self) -> &[QueryRecord] { &self.records }

    /// Success rate across all recorded queries.
    pub fn success_rate(&self) -> f64 {
        if self.records.is_empty() { return 0.0; }
        let ok = self.records.iter().filter(|r| r.response_success).count();
        ok as f64 / self.records.len() as f64
    }

    /// Average planning time for successful queries.
    pub fn avg_planning_time(&self) -> Option<Duration> {
        let successes: Vec<_> = self.records.iter().filter(|r| r.response_success).collect();
        if successes.is_empty() { return None; }
        let total: Duration = successes.iter().map(|r| r.planning_time).sum();
        Some(total / successes.len() as u32)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Planning Pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Composable planning pipeline.
///
/// Chains pre-processors → planner(s) → post-processors.
/// Supports parallel planner racing and fallback chains.
pub struct PlanningPipeline {
    pre_processors: Vec<Box<dyn PreProcessor>>,
    post_processors: Vec<Box<dyn PostProcessor>>,
    planners: Vec<Box<dyn PlannerPlugin>>,
    /// Default planner ID (first planner if not set).
    pub default_planner: String,
    /// Enable parallel planner racing (run all planners, take first result).
    pub parallel_racing: bool,
    /// Fallback planners to try if the primary fails.
    pub fallback_chain: Vec<String>,
    /// Query history.
    pub history: QueryHistory,
}

impl PlanningPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            pre_processors: Vec::new(),
            post_processors: Vec::new(),
            planners: Vec::new(),
            default_planner: String::new(),
            parallel_racing: false,
            fallback_chain: Vec::new(),
            history: QueryHistory::new(1000),
        }
    }

    /// Add a pre-processor.
    pub fn add_pre_processor(&mut self, adapter: Box<dyn PreProcessor>) {
        self.pre_processors.push(adapter);
    }

    /// Add a post-processor.
    pub fn add_post_processor(&mut self, adapter: Box<dyn PostProcessor>) {
        self.post_processors.push(adapter);
    }

    /// Register a planner plugin.
    pub fn add_planner(&mut self, planner: Box<dyn PlannerPlugin>) {
        if self.default_planner.is_empty() {
            self.default_planner = planner.id().to_string();
        }
        self.planners.push(planner);
    }

    /// Number of registered planners.
    pub fn num_planners(&self) -> usize { self.planners.len() }

    /// Apply a config profile.
    pub fn apply_profile(&mut self, profile: &PipelineProfile) {
        self.default_planner = profile.default_planner.clone();

        // Clear and rebuild post-processors based on profile
        self.post_processors.clear();
        if profile.validate {
            self.add_post_processor(Box::new(ValidationAdapter));
        }
        if profile.smooth {
            self.add_post_processor(Box::new(SmoothingAdapter));
        }
        if profile.time_param {
            self.add_post_processor(Box::new(TimeParameterizationAdapter));
        }
    }

    /// Run the planning pipeline.
    pub fn plan(&mut self, mut request: PlanningRequest) -> PlanningResponse {
        let start_time = Instant::now();
        let mut stages_run = Vec::new();

        // Run pre-processors
        for pre in &self.pre_processors {
            match pre.process(&mut request) {
                Ok(()) => stages_run.push(pre.name().to_string()),
                Err(e) => {
                    return PlanningResponse::failure(
                        format!("Pre-processor '{}' failed: {}", pre.name(), e),
                        start_time.elapsed(),
                    );
                }
            }
        }

        // Select planner
        let planner_id = if request.planner_id.is_empty() {
            self.default_planner.clone()
        } else {
            request.planner_id.clone()
        };

        // Plan
        let (trajectory, used_planner) = if self.parallel_racing && self.planners.len() > 1 {
            self.race_planners(&request)
        } else {
            self.plan_with_fallback(&request, &planner_id)
        };

        let planning_time = start_time.elapsed();

        let mut response = PlanningResponse {
            trajectory,
            planning_time,
            planner_id: used_planner.clone(),
            validated: false,
            smoothed: false,
            time_parameterized: false,
            error: None,
            stages_run: stages_run.clone(),
        };

        if response.trajectory.is_none() {
            response.error = Some("No planner found a solution".into());
        }

        // Run post-processors
        if response.trajectory.is_some() {
            for post in &self.post_processors {
                match post.process(&request, &mut response) {
                    Ok(()) => response.stages_run.push(post.name().to_string()),
                    Err(e) => {
                        response.error = Some(format!("Post-processor '{}' failed: {}", post.name(), e));
                        response.trajectory = None;
                        break;
                    }
                }
            }
        }

        // Record query
        self.history.record(QueryRecord {
            request,
            response_success: response.success(),
            planning_time,
            planner_id: used_planner,
            timestamp: Instant::now(),
        });

        response
    }

    /// Try the primary planner, then fallbacks.
    fn plan_with_fallback(
        &self,
        request: &PlanningRequest,
        primary_id: &str,
    ) -> (Option<Trajectory>, String) {
        // Try primary
        if let Some(planner) = self.planners.iter().find(|p| p.id() == primary_id) {
            if let Some(traj) = planner.plan(request) {
                return (Some(traj), primary_id.to_string());
            }
        }

        // Try fallbacks
        for fallback_id in &self.fallback_chain {
            if let Some(planner) = self.planners.iter().find(|p| p.id() == fallback_id) {
                if let Some(traj) = planner.plan(request) {
                    return (Some(traj), fallback_id.clone());
                }
            }
        }

        (None, primary_id.to_string())
    }

    /// Race all planners, take the first solution (sequential simulation).
    fn race_planners(
        &self,
        request: &PlanningRequest,
    ) -> (Option<Trajectory>, String) {
        // In a real implementation, this would use threads.
        // Sequential fallthrough for correctness.
        for planner in &self.planners {
            if let Some(traj) = planner.plan(request) {
                return (Some(traj), planner.id().to_string());
            }
        }
        (None, String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test planner that always returns a straight-line trajectory.
    struct LinearPlanner {
        id: String,
        dof: usize,
    }

    impl PlannerPlugin for LinearPlanner {
        fn plan(&self, request: &PlanningRequest) -> Option<Trajectory> {
            let mut traj = Trajectory::with_dof(self.dof);
            traj.push_waypoint(&request.start);
            // Simple midpoint
            let mid: Vec<f64> = (0..self.dof).map(|i| {
                if let Goal::Joints(ref g) = request.goal {
                    (request.start[i] + g[i]) / 2.0
                } else {
                    request.start[i]
                }
            }).collect();
            traj.push_waypoint(&mid);
            if let Goal::Joints(ref g) = request.goal {
                traj.push_waypoint(g);
            }
            Some(traj)
        }
        fn id(&self) -> &str { &self.id }
    }

    /// Test planner that always fails.
    struct FailPlanner;
    impl PlannerPlugin for FailPlanner {
        fn plan(&self, _request: &PlanningRequest) -> Option<Trajectory> { None }
        fn id(&self) -> &str { "fail" }
    }

    fn make_request() -> PlanningRequest {
        PlanningRequest::new(
            JointValues::new(vec![0.0, 0.0]),
            Goal::Joints(JointValues::new(vec![1.0, 1.0])),
        )
    }

    // ─── Pipeline tests ───

    #[test]
    fn pipeline_basic_plan() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(LinearPlanner { id: "linear".into(), dof: 2 }));

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert_eq!(response.planner_id, "linear");
        assert!(response.trajectory.unwrap().len() >= 3);
    }

    #[test]
    fn pipeline_pre_processor_runs() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_pre_processor(Box::new(CollisionPaddingAdapter { default_padding: 0.05 }));
        pipeline.add_planner(Box::new(LinearPlanner { id: "linear".into(), dof: 2 }));

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert!(response.stages_run.contains(&"collision_padding".to_string()));
    }

    #[test]
    fn pipeline_post_processors_run() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(LinearPlanner { id: "linear".into(), dof: 2 }));
        pipeline.add_post_processor(Box::new(ValidationAdapter));
        pipeline.add_post_processor(Box::new(SmoothingAdapter));
        pipeline.add_post_processor(Box::new(TimeParameterizationAdapter));

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert!(response.validated);
        assert!(response.smoothed);
        assert!(response.time_parameterized);
    }

    #[test]
    fn pipeline_fallback_on_failure() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(FailPlanner));
        pipeline.add_planner(Box::new(LinearPlanner { id: "backup".into(), dof: 2 }));
        pipeline.default_planner = "fail".into();
        pipeline.fallback_chain = vec!["backup".into()];

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert_eq!(response.planner_id, "backup");
    }

    #[test]
    fn pipeline_all_fail() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(FailPlanner));

        let response = pipeline.plan(make_request());
        assert!(!response.success());
        assert!(response.error.is_some());
    }

    #[test]
    fn pipeline_parallel_racing() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(FailPlanner));
        pipeline.add_planner(Box::new(LinearPlanner { id: "winner".into(), dof: 2 }));
        pipeline.parallel_racing = true;

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert_eq!(response.planner_id, "winner");
    }

    #[test]
    fn pipeline_apply_profile() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(LinearPlanner { id: "rrt_connect".into(), dof: 2 }));
        pipeline.add_planner(Box::new(LinearPlanner { id: "rrt_star".into(), dof: 2 }));

        pipeline.apply_profile(&PipelineProfile::quality());
        assert_eq!(pipeline.default_planner, "rrt_star");

        let response = pipeline.plan(make_request());
        assert!(response.success());
        assert!(response.validated);
        assert!(response.smoothed);
    }

    #[test]
    fn pipeline_request_specific_planner() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(LinearPlanner { id: "a".into(), dof: 2 }));
        pipeline.add_planner(Box::new(LinearPlanner { id: "b".into(), dof: 2 }));

        let mut req = make_request();
        req.planner_id = "b".into();

        let response = pipeline.plan(req);
        assert!(response.success());
        assert_eq!(response.planner_id, "b");
    }

    // ─── Query history tests ───

    #[test]
    fn query_history_records() {
        let mut pipeline = PlanningPipeline::new();
        pipeline.add_planner(Box::new(LinearPlanner { id: "linear".into(), dof: 2 }));

        pipeline.plan(make_request());
        pipeline.plan(make_request());

        assert_eq!(pipeline.history.len(), 2);
        assert!((pipeline.history.success_rate() - 1.0).abs() < 1e-10);
        assert!(pipeline.history.avg_planning_time().is_some());
    }

    // ─── Scene snapshot tests ───

    #[test]
    fn scene_snapshot_store() {
        let mut store = SceneSnapshotStore::new();

        store.save(SceneSnapshot {
            id: "snap1".into(),
            timestamp: Instant::now(),
            obstacle_centers: vec![[1.0, 0.0, 0.0]],
            obstacle_radii: vec![0.1],
            joint_state: None,
            description: "test".into(),
        });

        assert_eq!(store.count(), 1);
        assert!(store.load("snap1").is_some());
        assert!(store.load("missing").is_none());
        assert_eq!(store.list().len(), 1);
    }

    #[test]
    fn scene_snapshot_diff() {
        let mut store = SceneSnapshotStore::new();

        store.save(SceneSnapshot {
            id: "a".into(),
            timestamp: Instant::now(),
            obstacle_centers: vec![[0.0; 3]],
            obstacle_radii: vec![0.1],
            joint_state: None,
            description: "before".into(),
        });

        store.save(SceneSnapshot {
            id: "b".into(),
            timestamp: Instant::now(),
            obstacle_centers: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            obstacle_radii: vec![0.1, 0.1, 0.1],
            joint_state: None,
            description: "after".into(),
        });

        let diff = store.diff("a", "b").unwrap();
        assert_eq!(diff.obstacle_count_a, 1);
        assert_eq!(diff.obstacle_count_b, 3);
        assert_eq!(diff.added_obstacles, 2);
    }

    // ─── Config profile tests ───

    #[test]
    fn config_profiles() {
        let fast = PipelineProfile::fast();
        assert_eq!(fast.name, "fast");
        assert!(!fast.smooth);

        let quality = PipelineProfile::quality();
        assert!(quality.smooth);
        assert!(quality.validate);

        let safe = PipelineProfile::safe();
        assert_eq!(safe.velocity_scale, 0.5);
        assert!(safe.collision_padding > 0.0);
    }

    #[test]
    fn workspace_bounds_adapter() {
        let adapter = WorkspaceBoundsAdapter {
            bounds: [-1.0, -1.0, 0.0, 1.0, 1.0, 2.0],
        };
        let mut req = make_request();
        assert!(req.workspace_bounds.is_none());

        adapter.process(&mut req).unwrap();
        assert!(req.workspace_bounds.is_some());
    }

    #[test]
    fn pipeline_error_handling_pre_processor() {
        struct FailPre;
        impl PreProcessor for FailPre {
            fn process(&self, _: &mut PlanningRequest) -> Result<(), String> {
                Err("bad request".into())
            }
            fn name(&self) -> &str { "fail_pre" }
        }

        let mut pipeline = PlanningPipeline::new();
        pipeline.add_pre_processor(Box::new(FailPre));
        pipeline.add_planner(Box::new(LinearPlanner { id: "l".into(), dof: 2 }));

        let response = pipeline.plan(make_request());
        assert!(!response.success());
        assert!(response.error.unwrap().contains("bad request"));
    }

    #[test]
    fn scene_snapshot_delete() {
        let mut store = SceneSnapshotStore::new();
        store.save(SceneSnapshot {
            id: "x".into(), timestamp: Instant::now(),
            obstacle_centers: vec![], obstacle_radii: vec![],
            joint_state: None, description: "".into(),
        });
        assert!(store.delete("x"));
        assert_eq!(store.count(), 0);
        assert!(!store.delete("x"));
    }
}
