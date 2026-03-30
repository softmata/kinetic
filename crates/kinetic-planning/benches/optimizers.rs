//! Benchmarks for CHOMP and STOMP trajectory optimizers.

use criterion::{criterion_group, criterion_main, Criterion};

use kinetic_core::Trajectory;
use kinetic_planning::cost::{CompositeCost, SmoothnessCost, TrajectoryCost, VelocityCost};
use kinetic_planning::chomp::{CHOMP, CHOMPConfig, InitStrategy};
use kinetic_planning::stomp::{STOMP, STOMPConfig};

fn make_zigzag(dof: usize, waypoints: usize) -> Trajectory {
    let mut traj = Trajectory::with_dof(dof);
    for i in 0..waypoints {
        let t = i as f64 / (waypoints - 1) as f64;
        let wp: Vec<f64> = (0..dof).map(|j| {
            t + if i % 2 == 0 { 0.3 } else { -0.3 } * (j as f64 * 0.1 + 0.5)
        }).collect();
        traj.push_waypoint(&wp);
    }
    traj
}

fn bench_chomp(c: &mut Criterion) {
    let dof = 6;
    let traj = make_zigzag(dof, 20);

    c.bench_function("chomp_6dof_20wp_smooth", |b| {
        let cost = Box::new(SmoothnessCost::new(dof));
        let chomp = CHOMP::new(cost, dof, CHOMPConfig {
            max_iterations: 50,
            learning_rate: 0.01,
            ..Default::default()
        });
        b.iter(|| chomp.optimize(&traj))
    });

    c.bench_function("chomp_6dof_20wp_composite", |b| {
        let mut composite = CompositeCost::new(dof);
        composite.add("smooth", Box::new(SmoothnessCost::new(dof)), 1.0);
        composite.add("vel", Box::new(VelocityCost::new(dof)), 0.5);
        let chomp = CHOMP::new(Box::new(composite), dof, CHOMPConfig {
            max_iterations: 50,
            learning_rate: 0.01,
            ..Default::default()
        });
        b.iter(|| chomp.optimize(&traj))
    });
}

fn bench_stomp(c: &mut Criterion) {
    let dof = 6;
    let traj = make_zigzag(dof, 20);

    c.bench_function("stomp_6dof_20wp_10samples", |b| {
        let cost = Box::new(SmoothnessCost::new(dof));
        let stomp = STOMP::new(cost, dof, STOMPConfig {
            num_samples: 10,
            max_iterations: 30,
            ..Default::default()
        });
        b.iter(|| stomp.optimize(&traj))
    });
}

fn bench_cost_eval(c: &mut Criterion) {
    let dof = 6;
    let traj = make_zigzag(dof, 50);

    c.bench_function("smoothness_cost_6dof_50wp", |b| {
        let cost = SmoothnessCost::new(dof);
        b.iter(|| cost.evaluate(&traj))
    });

    c.bench_function("smoothness_gradient_6dof_50wp", |b| {
        let cost = SmoothnessCost::new(dof);
        b.iter(|| cost.gradient(&traj))
    });
}

criterion_group!(benches, bench_chomp, bench_stomp, bench_cost_eval);
criterion_main!(benches);
