//! Full pipeline benchmarks: plan → parameterize → sample.
//!
//! Performance targets:
//! - Full pipeline simple: <200 us
//! - Full pipeline with trajectory: <1 ms

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("../examples/panda_urdf.txt");

fn bench_full_pipeline_simple(c: &mut Criterion) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    c.bench_function("full_pipeline_plan_only", |b| {
        b.iter(|| {
            let result = planner.plan(&start, &goal).unwrap();
            black_box(result);
        })
    });
}

fn bench_full_pipeline_with_trajectory(c: &mut Criterion) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    c.bench_function("full_pipeline_plan_and_trapezoidal", |b| {
        b.iter(|| {
            let result = planner.plan(&start, &goal).unwrap();
            let timed = trapezoidal(&result.waypoints, 1.0, 2.0).unwrap();
            black_box(timed);
        })
    });
}

fn bench_full_pipeline_plan_totp(c: &mut Criterion) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    let vel_limits = vec![2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610];
    let accel_limits = vec![15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0];

    c.bench_function("full_pipeline_plan_and_totp", |b| {
        b.iter(|| {
            let result = planner.plan(&start, &goal).unwrap();
            let timed = totp(&result.waypoints, &vel_limits, &accel_limits, 0.01).unwrap();
            black_box(timed);
        })
    });
}

fn bench_fk_ik_roundtrip(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(PANDA_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

    let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let config = IKConfig::dls()
        .with_seed(robot.mid_configuration().to_vec())
        .with_max_iterations(300);

    c.bench_function("fk_ik_roundtrip", |b| {
        b.iter(|| {
            let pose = forward_kinematics(&robot, &chain, &q).unwrap();
            let sol = solve_ik(&robot, &chain, &pose, &config).unwrap();
            black_box(sol);
        })
    });
}

fn bench_trajectory_sample(c: &mut Criterion) {
    let path: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let t = i as f64 / 19.0;
            vec![
                t * 0.5,
                t * -0.3,
                t * 0.2,
                t * -0.8,
                t * 0.2,
                t * 0.5,
                t * 0.3,
            ]
        })
        .collect();

    let timed = trapezoidal(&path, 1.0, 2.0).unwrap();
    let duration = timed.duration.as_secs_f64();

    c.bench_function("trajectory_sample_at", |b| {
        b.iter(|| {
            // Sample at 100 points
            for i in 0..100 {
                let t = std::time::Duration::from_secs_f64(duration * (i as f64 / 99.0));
                black_box(timed.sample_at(t));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_full_pipeline_simple,
    bench_full_pipeline_with_trajectory,
    bench_full_pipeline_plan_totp,
    bench_fk_ik_roundtrip,
    bench_trajectory_sample,
);
criterion_main!(benches);
