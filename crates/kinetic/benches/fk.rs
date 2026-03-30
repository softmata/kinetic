//! Forward kinematics benchmarks.
//!
//! Performance targets:
//! - FK single 7-DOF: <1 µs
//! - FK batch 8 configs: <200 ns/config
//! - Jacobian 7-DOF: <2 µs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::kinematics::{fk_batch, manipulability};
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("../examples/panda_urdf.txt");

fn setup() -> (Robot, KinematicChain, Vec<f64>) {
    let robot = Robot::from_urdf_string(PANDA_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();
    let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    (robot, chain, q)
}

fn bench_fk_single(c: &mut Criterion) {
    let (robot, chain, q) = setup();

    c.bench_function("fk_7dof_single", |b| {
        b.iter(|| {
            black_box(forward_kinematics(&robot, &chain, &q).unwrap());
        })
    });
}

fn bench_forward_kinematics_all(c: &mut Criterion) {
    let (robot, chain, q) = setup();

    c.bench_function("fk_7dof_all_links", |b| {
        b.iter(|| {
            black_box(forward_kinematics_all(&robot, &chain, &q).unwrap());
        })
    });
}

fn bench_fk_batch(c: &mut Criterion) {
    let (robot, chain, _) = setup();

    // fk_batch takes a flat &[f64] with num_configs * dof elements
    let num_configs = 8;
    let mut flat_configs = Vec::with_capacity(num_configs * chain.dof);
    for i in 0..num_configs {
        let offset = i as f64 * 0.1;
        flat_configs.extend_from_slice(&[0.3 + offset, -0.5 + offset, 0.2, -1.5, 0.1, 1.0, 0.5]);
    }

    c.bench_function("fk_7dof_batch_8", |b| {
        b.iter(|| {
            black_box(fk_batch(&robot, &chain, &flat_configs, num_configs).unwrap());
        })
    });
}

fn bench_jacobian(c: &mut Criterion) {
    let (robot, chain, q) = setup();

    c.bench_function("jacobian_7dof", |b| {
        b.iter(|| {
            black_box(jacobian(&robot, &chain, &q).unwrap());
        })
    });
}

fn bench_manipulability(c: &mut Criterion) {
    let (robot, chain, q) = setup();

    c.bench_function("manipulability_7dof", |b| {
        b.iter(|| {
            black_box(manipulability(&robot, &chain, &q).unwrap());
        })
    });
}

criterion_group!(
    benches,
    bench_fk_single,
    bench_forward_kinematics_all,
    bench_fk_batch,
    bench_jacobian,
    bench_manipulability,
);
criterion_main!(benches);
