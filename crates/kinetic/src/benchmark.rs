//! Benchmark infrastructure for KINETIC.
//!
//! Provides standardized benchmark scenarios and result aggregation
//! for comparing KINETIC against MoveIt2, VAMP, and cuRobo.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic_core::{Goal, JointValues};
use kinetic_planning::{Planner, PlanningResult};
use kinetic_robot::Robot;
use kinetic_scene::{Scene, Shape};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};

/// A single benchmark scenario: robot + obstacles + start/goal.
pub struct BenchmarkScenario {
    /// Human-readable scenario name.
    pub name: String,
    /// Robot config name (e.g., "franka_panda").
    pub robot_config: String,
    /// Obstacles: (shape, world-frame pose).
    pub obstacles: Vec<(Shape, Isometry3<f64>)>,
    /// Starting joint configuration.
    pub start: Vec<f64>,
    /// Goal joint configuration.
    pub goal: Vec<f64>,
    /// Number of planning runs for statistical significance.
    pub num_runs: usize,
}

/// Aggregated result of running a benchmark scenario.
pub struct BenchmarkResult {
    /// Scenario name.
    pub scenario: String,
    /// Success rate (0.0–1.0).
    pub success_rate: f64,
    /// Median (p50) planning time.
    pub planning_time_p50: Duration,
    /// 95th percentile planning time.
    pub planning_time_p95: Duration,
    /// 99th percentile planning time.
    pub planning_time_p99: Duration,
    /// Mean path length (sum of joint-space deltas).
    pub path_length_mean: f64,
    /// Mean smoothness (sum of squared joint accelerations).
    pub smoothness_mean: f64,
    /// Number of successful runs.
    pub successes: usize,
    /// Total number of runs.
    pub total_runs: usize,
}

/// Published reference numbers from competing frameworks.
pub(crate) struct CompetitorResult {
    pub framework: &'static str,
    pub scenario: &'static str,
    pub planning_time_p50: Duration,
    pub success_rate: f64,
}

/// Collection of benchmark scenarios.
pub struct BenchmarkSuite {
    pub scenarios: Vec<BenchmarkScenario>,
}

impl BenchmarkSuite {
    /// Run all scenarios and return results.
    pub fn run(&self) -> Vec<BenchmarkResult> {
        self.scenarios.iter().map(run_scenario).collect()
    }

    /// Load the standard MotionBenchMaker-style scenarios.
    pub fn motionbenchmaker() -> Self {
        BenchmarkSuite {
            scenarios: vec![
                table_pick_scenario(),
                shelf_pick_scenario(),
                narrow_passage_scenario(),
                cluttered_desk_scenario(),
                overhead_reach_scenario(),
                bookshelf_scenario(),
                cage_scenario(),
            ],
        }
    }

    /// Load scenarios for a specific robot (UR5e variant).
    pub fn motionbenchmaker_ur5e() -> Self {
        BenchmarkSuite {
            scenarios: vec![
                ur5e_table_scenario(),
                ur5e_bin_pick_scenario(),
            ],
        }
    }
}

/// Run a single benchmark scenario, collecting timing statistics.
pub fn run_scenario(scenario: &BenchmarkScenario) -> BenchmarkResult {
    let robot = Robot::from_name(&scenario.robot_config)
        .unwrap_or_else(|e| panic!("Failed to load robot '{}': {}", scenario.robot_config, e));

    let robot_arc = Arc::new(robot);

    let mut scene =
        Scene::new(&robot_arc).unwrap_or_else(|e| panic!("Failed to create scene: {}", e));

    for (i, (shape, pose)) in scenario.obstacles.iter().enumerate() {
        scene.add(&format!("obs_{}", i), shape.clone(), *pose);
    }

    let planner =
        Planner::new(&robot_arc).unwrap_or_else(|e| panic!("Failed to create planner: {}", e));

    let goal = Goal::Joints(JointValues(scenario.goal.clone()));

    let mut times = Vec::with_capacity(scenario.num_runs);
    let mut path_lengths = Vec::new();
    let mut successes = 0usize;

    for _ in 0..scenario.num_runs {
        let start = Instant::now();
        let result = planner.plan(&scenario.start, &goal);
        let elapsed = start.elapsed();

        match result {
            Ok(pr) => {
                successes += 1;
                times.push(elapsed);
                path_lengths.push(path_length(&pr));
            }
            Err(_) => {
                times.push(elapsed);
            }
        }
    }

    times.sort();

    let n = times.len();
    let p50 = times[n / 2];
    let p95 = times[(n as f64 * 0.95) as usize];
    let p99 = times[(n as f64 * 0.99) as usize];

    let path_length_mean = if path_lengths.is_empty() {
        0.0
    } else {
        path_lengths.iter().sum::<f64>() / path_lengths.len() as f64
    };

    BenchmarkResult {
        scenario: scenario.name.clone(),
        success_rate: successes as f64 / scenario.num_runs as f64,
        planning_time_p50: p50,
        planning_time_p95: p95,
        planning_time_p99: p99,
        path_length_mean,
        smoothness_mean: 0.0, // Requires trajectory parameterization
        successes,
        total_runs: scenario.num_runs,
    }
}

/// Compute path length in joint space.
fn path_length(result: &PlanningResult) -> f64 {
    let wp = &result.waypoints;
    let mut total = 0.0;
    for i in 1..wp.len() {
        let delta: f64 = wp[i]
            .iter()
            .zip(wp[i - 1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        total += delta.sqrt();
    }
    total
}

/// Generate comparison report as Markdown.
pub fn generate_report(results: &[BenchmarkResult]) -> String {
    let competitors = competitor_reference_numbers();
    let mut report = String::new();

    report.push_str("# KINETIC Benchmark Report\n\n");
    report.push_str("## Planning Scenarios\n\n");
    report.push_str("| Scenario | Success Rate | p50 | p95 | p99 | Path Length |\n");
    report.push_str("|----------|-------------|-----|-----|-----|-------------|\n");

    for r in results {
        report.push_str(&format!(
            "| {} | {:.1}% | {:?} | {:?} | {:?} | {:.3} |\n",
            r.scenario,
            r.success_rate * 100.0,
            r.planning_time_p50,
            r.planning_time_p95,
            r.planning_time_p99,
            r.path_length_mean,
        ));
    }

    report.push_str("\n## Comparison with Other Frameworks\n\n");
    report.push_str("| Scenario | Framework | p50 Planning Time | Success Rate |\n");
    report.push_str("|----------|-----------|-------------------|-------------|\n");

    for r in results {
        report.push_str(&format!(
            "| {} | **KINETIC** | **{:?}** | **{:.1}%** |\n",
            r.scenario,
            r.planning_time_p50,
            r.success_rate * 100.0,
        ));

        for c in &competitors {
            if c.scenario == r.scenario {
                report.push_str(&format!(
                    "| {} | {} | {:?} | {:.1}% |\n",
                    c.scenario,
                    c.framework,
                    c.planning_time_p50,
                    c.success_rate * 100.0,
                ));
            }
        }
    }

    report.push_str("\n*MoveIt2/VAMP/cuRobo numbers from published papers and benchmarks.*\n");

    report
}

/// Published reference numbers from competing frameworks.
fn competitor_reference_numbers() -> Vec<CompetitorResult> {
    vec![
        // Table pick (simple) - from published benchmarks
        CompetitorResult {
            framework: "MoveIt2 (OMPL)",
            scenario: "table_pick",
            planning_time_p50: Duration::from_millis(170),
            success_rate: 0.95,
        },
        CompetitorResult {
            framework: "VAMP",
            scenario: "table_pick",
            planning_time_p50: Duration::from_micros(35),
            success_rate: 0.99,
        },
        CompetitorResult {
            framework: "cuRobo",
            scenario: "table_pick",
            planning_time_p50: Duration::from_millis(45),
            success_rate: 0.98,
        },
        // Shelf pick (cluttered)
        CompetitorResult {
            framework: "MoveIt2 (OMPL)",
            scenario: "shelf_pick",
            planning_time_p50: Duration::from_millis(1200),
            success_rate: 0.80,
        },
        CompetitorResult {
            framework: "VAMP",
            scenario: "shelf_pick",
            planning_time_p50: Duration::from_millis(16),
            success_rate: 0.95,
        },
        CompetitorResult {
            framework: "cuRobo",
            scenario: "shelf_pick",
            planning_time_p50: Duration::from_millis(45),
            success_rate: 0.93,
        },
        // Narrow passage
        CompetitorResult {
            framework: "MoveIt2 (OMPL)",
            scenario: "narrow_passage",
            planning_time_p50: Duration::from_millis(3000),
            success_rate: 0.60,
        },
        CompetitorResult {
            framework: "VAMP",
            scenario: "narrow_passage",
            planning_time_p50: Duration::from_millis(50),
            success_rate: 0.90,
        },
        CompetitorResult {
            framework: "cuRobo",
            scenario: "narrow_passage",
            planning_time_p50: Duration::from_millis(100),
            success_rate: 0.85,
        },
    ]
}

// ── Standard Scenarios ───────────────────────────────────────────────

fn translation_pose(x: f64, y: f64, z: f64) -> Isometry3<f64> {
    Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
}

/// Table pick: Panda reaches across a table surface.
fn table_pick_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "table_pick".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Table surface
            (
                Shape::Cuboid(0.5, 0.4, 0.01),
                translation_pose(0.5, 0.0, 0.4),
            ),
            // Table legs (simplified)
            (
                Shape::Cylinder(0.03, 0.2),
                translation_pose(0.15, -0.25, 0.2),
            ),
            (
                Shape::Cylinder(0.03, 0.2),
                translation_pose(0.85, -0.25, 0.2),
            ),
            (
                Shape::Cylinder(0.03, 0.2),
                translation_pose(0.15, 0.25, 0.2),
            ),
            (
                Shape::Cylinder(0.03, 0.2),
                translation_pose(0.85, 0.25, 0.2),
            ),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5],
        num_runs: 100,
    }
}

/// Shelf pick: Panda reaches into a cluttered shelf.
fn shelf_pick_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "shelf_pick".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Back wall
            (
                Shape::Cuboid(0.4, 0.01, 0.3),
                translation_pose(0.6, -0.35, 0.6),
            ),
            // Top shelf
            (
                Shape::Cuboid(0.4, 0.3, 0.01),
                translation_pose(0.6, 0.0, 0.9),
            ),
            // Bottom shelf
            (
                Shape::Cuboid(0.4, 0.3, 0.01),
                translation_pose(0.6, 0.0, 0.4),
            ),
            // Left side
            (
                Shape::Cuboid(0.01, 0.3, 0.3),
                translation_pose(0.2, 0.0, 0.6),
            ),
            // Right side
            (
                Shape::Cuboid(0.01, 0.3, 0.3),
                translation_pose(1.0, 0.0, 0.6),
            ),
            // Obstacles on shelf
            (
                Shape::Cuboid(0.05, 0.05, 0.1),
                translation_pose(0.4, 0.0, 0.5),
            ),
            (
                Shape::Cuboid(0.05, 0.05, 0.1),
                translation_pose(0.8, 0.1, 0.5),
            ),
            (Shape::Sphere(0.04), translation_pose(0.6, -0.15, 0.5)),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![0.8, -0.3, 0.5, -1.5, 0.3, 1.0, 0.3],
        num_runs: 100,
    }
}

/// Narrow passage: Panda must pass through a narrow gap.
fn narrow_passage_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "narrow_passage".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Left wall
            (
                Shape::Cuboid(0.3, 0.01, 0.5),
                translation_pose(0.5, -0.08, 0.5),
            ),
            // Right wall
            (
                Shape::Cuboid(0.3, 0.01, 0.5),
                translation_pose(0.5, 0.08, 0.5),
            ),
            // Floor
            (
                Shape::Cuboid(0.5, 0.5, 0.01),
                translation_pose(0.5, 0.0, 0.0),
            ),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![-0.5, -0.3, -0.5, -1.8, 0.5, 1.0, -0.3],
        num_runs: 100,
    }
}

/// Cluttered desk with multiple small objects.
fn cluttered_desk_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "cluttered_desk".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Desk surface
            (
                Shape::Cuboid(0.6, 0.4, 0.01),
                translation_pose(0.5, 0.0, 0.4),
            ),
            // Objects on desk
            (
                Shape::Cuboid(0.04, 0.04, 0.08),
                translation_pose(0.35, -0.1, 0.48),
            ),
            (
                Shape::Cuboid(0.04, 0.04, 0.08),
                translation_pose(0.45, 0.1, 0.48),
            ),
            (
                Shape::Cylinder(0.03, 0.06),
                translation_pose(0.55, -0.05, 0.46),
            ),
            (Shape::Sphere(0.04), translation_pose(0.65, 0.15, 0.44)),
            (
                Shape::Cuboid(0.06, 0.03, 0.12),
                translation_pose(0.4, 0.2, 0.52),
            ),
            (
                Shape::Cylinder(0.02, 0.10),
                translation_pose(0.7, -0.1, 0.5),
            ),
            (Shape::Sphere(0.03), translation_pose(0.3, 0.0, 0.43)),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![0.6, -0.4, 0.4, -1.6, 0.3, 1.3, 0.6],
        num_runs: 100,
    }
}

/// Overhead reach: Robot reaches high above its base.
fn overhead_reach_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "overhead_reach".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Low ceiling
            (
                Shape::Cuboid(1.0, 1.0, 0.01),
                translation_pose(0.0, 0.0, 1.2),
            ),
            // Support column
            (Shape::Cylinder(0.05, 0.6), translation_pose(0.3, 0.3, 0.6)),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![0.0, 0.3, 0.0, -0.8, 0.0, 2.5, 0.785],
        num_runs: 100,
    }
}

/// Bookshelf: Panda reaches between two shelf levels with vertical constraints.
fn bookshelf_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "bookshelf".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Back panel
            (Shape::Cuboid(0.5, 0.01, 0.8), translation_pose(0.6, -0.3, 0.6)),
            // Shelves (3 levels)
            (Shape::Cuboid(0.5, 0.25, 0.01), translation_pose(0.6, 0.0, 0.3)),
            (Shape::Cuboid(0.5, 0.25, 0.01), translation_pose(0.6, 0.0, 0.6)),
            (Shape::Cuboid(0.5, 0.25, 0.01), translation_pose(0.6, 0.0, 0.9)),
            // Side panels
            (Shape::Cuboid(0.01, 0.25, 0.4), translation_pose(0.35, 0.0, 0.6)),
            (Shape::Cuboid(0.01, 0.25, 0.4), translation_pose(0.85, 0.0, 0.6)),
            // Books on middle shelf
            (Shape::Cuboid(0.02, 0.08, 0.12), translation_pose(0.45, 0.0, 0.36)),
            (Shape::Cuboid(0.02, 0.08, 0.12), translation_pose(0.55, 0.0, 0.36)),
            (Shape::Cuboid(0.02, 0.08, 0.12), translation_pose(0.75, 0.0, 0.36)),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![0.4, -0.2, 0.3, -1.6, 0.1, 1.4, 0.4],
        num_runs: 100,
    }
}

/// Cage: Robot operates inside a partially enclosed workspace.
fn cage_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "cage".into(),
        robot_config: "franka_panda".into(),
        obstacles: vec![
            // Floor
            (Shape::Cuboid(0.8, 0.8, 0.01), translation_pose(0.0, 0.0, -0.01)),
            // Ceiling (low)
            (Shape::Cuboid(0.8, 0.8, 0.01), translation_pose(0.0, 0.0, 0.8)),
            // Three walls (front open)
            (Shape::Cuboid(0.01, 0.8, 0.4), translation_pose(-0.4, 0.0, 0.4)),
            (Shape::Cuboid(0.01, 0.8, 0.4), translation_pose(0.4, 0.0, 0.4)),
            (Shape::Cuboid(0.8, 0.01, 0.4), translation_pose(0.0, -0.4, 0.4)),
            // Internal obstacle
            (Shape::Cylinder(0.06, 0.3), translation_pose(0.2, 0.0, 0.35)),
        ],
        start: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        goal: vec![-0.3, -0.5, -0.3, -1.5, 0.5, 1.8, -0.2],
        num_runs: 100,
    }
}

/// UR5e table pick: UR5e picks across a table surface.
fn ur5e_table_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "ur5e_table".into(),
        robot_config: "ur5e".into(),
        obstacles: vec![
            (Shape::Cuboid(0.5, 0.4, 0.01), translation_pose(0.4, 0.0, 0.0)),
        ],
        start: vec![0.0, -1.571, 1.571, -1.571, -1.571, 0.0],
        goal: vec![0.5, -1.0, 1.0, -1.571, -1.571, 0.5],
        num_runs: 100,
    }
}

/// UR5e bin pick: UR5e reaches into a bin.
fn ur5e_bin_pick_scenario() -> BenchmarkScenario {
    BenchmarkScenario {
        name: "ur5e_bin_pick".into(),
        robot_config: "ur5e".into(),
        obstacles: vec![
            // Bin walls
            (Shape::Cuboid(0.2, 0.01, 0.15), translation_pose(0.4, -0.15, 0.075)),
            (Shape::Cuboid(0.2, 0.01, 0.15), translation_pose(0.4, 0.15, 0.075)),
            (Shape::Cuboid(0.01, 0.15, 0.15), translation_pose(0.25, 0.0, 0.075)),
            (Shape::Cuboid(0.01, 0.15, 0.15), translation_pose(0.55, 0.0, 0.075)),
        ],
        start: vec![0.0, -1.571, 1.571, -1.571, -1.571, 0.0],
        goal: vec![0.3, -1.2, 1.8, -2.2, -1.571, 0.3],
        num_runs: 100,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_pick_scenario_loads() {
        let s = table_pick_scenario();
        assert_eq!(s.name, "table_pick");
        assert_eq!(s.start.len(), 7);
        assert_eq!(s.goal.len(), 7);
        assert!(!s.obstacles.is_empty());
    }

    #[test]
    fn test_benchmark_suite_creates() {
        let suite = BenchmarkSuite::motionbenchmaker();
        assert_eq!(suite.scenarios.len(), 7);
    }

    #[test]
    fn test_ur5e_suite_creates() {
        let suite = BenchmarkSuite::motionbenchmaker_ur5e();
        assert_eq!(suite.scenarios.len(), 2);
        assert_eq!(suite.scenarios[0].robot_config, "ur5e");
    }

    #[test]
    fn test_bookshelf_scenario_loads() {
        let s = bookshelf_scenario();
        assert_eq!(s.name, "bookshelf");
        assert!(!s.obstacles.is_empty());
    }

    #[test]
    fn test_cage_scenario_loads() {
        let s = cage_scenario();
        assert_eq!(s.name, "cage");
        assert!(!s.obstacles.is_empty());
    }

    #[test]
    fn test_single_scenario_run() {
        let mut scenario = table_pick_scenario();
        scenario.num_runs = 3; // Fast for testing
        let result = run_scenario(&scenario);
        assert_eq!(result.total_runs, 3);
        assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
    }

    #[test]
    fn test_generate_report() {
        let results = vec![BenchmarkResult {
            scenario: "test".into(),
            success_rate: 0.95,
            planning_time_p50: Duration::from_micros(100),
            planning_time_p95: Duration::from_micros(500),
            planning_time_p99: Duration::from_millis(1),
            path_length_mean: 2.5,
            smoothness_mean: 0.1,
            successes: 95,
            total_runs: 100,
        }];
        let report = generate_report(&results);
        assert!(report.contains("KINETIC Benchmark Report"));
        assert!(report.contains("test"));
        assert!(report.contains("95.0%"));
    }
}
