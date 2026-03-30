//! MotionBenchMaker-style scenario benchmarks.
//!
//! Measures planning performance on standardized scenarios
//! comparable to published MoveIt2, VAMP, and cuRobo results.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::benchmark::{run_scenario, BenchmarkSuite};

fn bench_table_pick(c: &mut Criterion) {
    let suite = BenchmarkSuite::motionbenchmaker();
    let scenario = &suite.scenarios[0]; // table_pick
    assert_eq!(scenario.name, "table_pick");

    c.bench_function("mbm_table_pick", |b| {
        let s = kinetic::benchmark::BenchmarkScenario {
            name: scenario.name.clone(),
            robot_config: scenario.robot_config.clone(),
            obstacles: scenario.obstacles.clone(),
            start: scenario.start.clone(),
            goal: scenario.goal.clone(),
            num_runs: 1,
        };
        b.iter(|| {
            black_box(run_scenario(&s));
        })
    });
}

fn bench_shelf_pick(c: &mut Criterion) {
    let suite = BenchmarkSuite::motionbenchmaker();
    let scenario = &suite.scenarios[1]; // shelf_pick
    assert_eq!(scenario.name, "shelf_pick");

    c.bench_function("mbm_shelf_pick", |b| {
        let s = kinetic::benchmark::BenchmarkScenario {
            name: scenario.name.clone(),
            robot_config: scenario.robot_config.clone(),
            obstacles: scenario.obstacles.clone(),
            start: scenario.start.clone(),
            goal: scenario.goal.clone(),
            num_runs: 1,
        };
        b.iter(|| {
            black_box(run_scenario(&s));
        })
    });
}

fn bench_narrow_passage(c: &mut Criterion) {
    let suite = BenchmarkSuite::motionbenchmaker();
    let scenario = &suite.scenarios[2]; // narrow_passage
    assert_eq!(scenario.name, "narrow_passage");

    c.bench_function("mbm_narrow_passage", |b| {
        let s = kinetic::benchmark::BenchmarkScenario {
            name: scenario.name.clone(),
            robot_config: scenario.robot_config.clone(),
            obstacles: scenario.obstacles.clone(),
            start: scenario.start.clone(),
            goal: scenario.goal.clone(),
            num_runs: 1,
        };
        b.iter(|| {
            black_box(run_scenario(&s));
        })
    });
}

fn bench_cluttered_desk(c: &mut Criterion) {
    let suite = BenchmarkSuite::motionbenchmaker();
    let scenario = &suite.scenarios[3]; // cluttered_desk
    assert_eq!(scenario.name, "cluttered_desk");

    c.bench_function("mbm_cluttered_desk", |b| {
        let s = kinetic::benchmark::BenchmarkScenario {
            name: scenario.name.clone(),
            robot_config: scenario.robot_config.clone(),
            obstacles: scenario.obstacles.clone(),
            start: scenario.start.clone(),
            goal: scenario.goal.clone(),
            num_runs: 1,
        };
        b.iter(|| {
            black_box(run_scenario(&s));
        })
    });
}

fn bench_full_suite_3_runs(c: &mut Criterion) {
    c.bench_function("mbm_full_suite_3_runs_each", |b| {
        b.iter(|| {
            let mut suite = BenchmarkSuite::motionbenchmaker();
            // Reduce runs for benchmark timing
            for s in &mut suite.scenarios {
                s.num_runs = 3;
            }
            black_box(suite.run());
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_table_pick, bench_shelf_pick, bench_narrow_passage, bench_cluttered_desk, bench_full_suite_3_runs
);
criterion_main!(benches);
