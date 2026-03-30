//! Benchmark integration tests — run all scenarios with reduced count for CI.
//!
//! Full benchmarks (100 runs each) via: `cargo bench -p kinetic --bench motionbenchmaker`
//! These tests verify the infrastructure works and produce a report.

use kinetic::benchmark::{run_scenario, generate_report, BenchmarkSuite};

#[test]
fn all_panda_scenarios_run() {
    let suite = BenchmarkSuite::motionbenchmaker();
    let mut results = vec![];

    for mut scenario in suite.scenarios {
        scenario.num_runs = 3; // fast for CI
        let result = run_scenario(&scenario);

        assert_eq!(result.total_runs, 3, "{}: wrong run count", result.scenario);
        assert!(
            result.success_rate >= 0.0 && result.success_rate <= 1.0,
            "{}: invalid success rate {}",
            result.scenario, result.success_rate
        );
        assert!(
            result.planning_time_p50.as_nanos() > 0,
            "{}: zero planning time",
            result.scenario
        );

        eprintln!(
            "  {}: {:.0}% success, p50={:?}, path={:.3}",
            result.scenario,
            result.success_rate * 100.0,
            result.planning_time_p50,
            result.path_length_mean,
        );
        results.push(result);
    }

    assert_eq!(results.len(), 7, "should have 7 Panda scenarios");

    // Generate and verify report
    let report = generate_report(&results);
    assert!(report.contains("KINETIC Benchmark Report"));
    assert!(report.contains("table_pick"));
    assert!(report.contains("bookshelf"));
    assert!(report.contains("cage"));
    assert!(report.contains("MoveIt2"));
    eprintln!("\n{report}");
}

#[test]
fn ur5e_scenarios_run() {
    let suite = BenchmarkSuite::motionbenchmaker_ur5e();

    for mut scenario in suite.scenarios {
        scenario.num_runs = 3;
        let result = run_scenario(&scenario);

        assert_eq!(result.total_runs, 3);
        eprintln!(
            "  {}: {:.0}% success, p50={:?}",
            result.scenario,
            result.success_rate * 100.0,
            result.planning_time_p50,
        );
    }
}

#[test]
fn benchmark_report_includes_competitors() {
    let suite = BenchmarkSuite::motionbenchmaker();
    let mut results = vec![];
    for mut scenario in suite.scenarios {
        scenario.num_runs = 1;
        results.push(run_scenario(&scenario));
    }

    let report = generate_report(&results);
    assert!(report.contains("MoveIt2 (OMPL)"), "report should include MoveIt2");
    assert!(report.contains("VAMP"), "report should include VAMP");
    assert!(report.contains("cuRobo"), "report should include cuRobo");
}

#[test]
fn success_rate_above_minimum_threshold() {
    // table_pick should be easy — verify minimum 33% success with 3 runs
    let mut scenario = BenchmarkSuite::motionbenchmaker().scenarios.into_iter()
        .find(|s| s.name == "table_pick").unwrap();
    scenario.num_runs = 3;
    let result = run_scenario(&scenario);
    assert!(
        result.successes >= 1,
        "table_pick should succeed at least once out of 3: got {}/{}",
        result.successes, result.total_runs
    );
}
