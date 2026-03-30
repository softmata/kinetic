//! Trajectory time parameterization benchmarks.
//!
//! Performance targets:
//! - TOTP: <1 ms
//! - Trapezoidal: <100 us
//! - Trapezoidal per-joint: <200 us

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::prelude::*;
use kinetic::trajectory::{
    blend, cubic_spline_time, jerk_limited, trapezoidal_per_joint, TrajectoryValidator,
    ValidationConfig,
};

fn make_path(num_waypoints: usize, dof: usize) -> Vec<Vec<f64>> {
    (0..num_waypoints)
        .map(|i| {
            let t = i as f64 / (num_waypoints - 1) as f64;
            (0..dof).map(|j| t * (j as f64 + 1.0) * 0.5).collect()
        })
        .collect()
}

fn bench_trapezoidal(c: &mut Criterion) {
    let path = make_path(10, 7);

    c.bench_function("trapezoidal_7dof_10wp", |b| {
        b.iter(|| {
            black_box(trapezoidal(&path, 1.0, 2.0).unwrap());
        })
    });
}

fn bench_trapezoidal_per_joint(c: &mut Criterion) {
    let path = make_path(10, 7);
    let vel_limits = vec![2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610];
    let accel_limits = vec![15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0];

    c.bench_function("trapezoidal_per_joint_7dof_10wp", |b| {
        b.iter(|| {
            black_box(trapezoidal_per_joint(&path, &vel_limits, &accel_limits).unwrap());
        })
    });
}

fn bench_totp(c: &mut Criterion) {
    let path = make_path(20, 7);
    let vel_limits = vec![2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610];
    let accel_limits = vec![15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0];

    c.bench_function("totp_7dof_20wp", |b| {
        b.iter(|| {
            black_box(totp(&path, &vel_limits, &accel_limits, 0.01).unwrap());
        })
    });
}

fn bench_jerk_limited(c: &mut Criterion) {
    let path = make_path(10, 7);

    c.bench_function("jerk_limited_7dof_10wp", |b| {
        b.iter(|| {
            black_box(jerk_limited(&path, 1.0, 2.0, 10.0).unwrap());
        })
    });
}

fn bench_cubic_spline(c: &mut Criterion) {
    let path = make_path(10, 7);

    c.bench_function("cubic_spline_7dof_10wp", |b| {
        b.iter(|| {
            black_box(cubic_spline_time(&path, Some(3.0), None).unwrap());
        })
    });
}

fn bench_blend(c: &mut Criterion) {
    let path1 = make_path(5, 7);
    let path2: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            let t = i as f64 / 4.0;
            (0..7).map(|j| 0.5 + t * (j as f64 + 1.0) * 0.3).collect()
        })
        .collect();

    let traj1 = trapezoidal(&path1, 1.0, 2.0).unwrap();
    let traj2 = trapezoidal(&path2, 1.0, 2.0).unwrap();

    c.bench_function("blend_two_trajectories", |b| {
        b.iter(|| {
            black_box(blend(&traj1, &traj2, 0.1).unwrap());
        })
    });
}

fn bench_trapezoidal_large_path(c: &mut Criterion) {
    let path = make_path(100, 7);

    c.bench_function("trapezoidal_7dof_100wp", |b| {
        b.iter(|| {
            black_box(trapezoidal(&path, 1.0, 2.0).unwrap());
        })
    });
}

fn bench_validation(c: &mut Criterion) {
    let path = make_path(50, 7);
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();
    let validator = TrajectoryValidator::new(
        &[-std::f64::consts::PI; 7],
        &[std::f64::consts::PI; 7],
        &[2.0; 7],
        &[4.0; 7],
        ValidationConfig::default(),
    );

    c.bench_function("validation_7dof_50wp", |b| {
        b.iter(|| {
            let _ = black_box(validator.validate(&traj));
        })
    });
}

criterion_group!(
    benches,
    bench_trapezoidal,
    bench_trapezoidal_per_joint,
    bench_totp,
    bench_jerk_limited,
    bench_cubic_spline,
    bench_blend,
    bench_trapezoidal_large_path,
    bench_validation,
);
criterion_main!(benches);
