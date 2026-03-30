//! Benchmarks for Bio-IK, SQP, and Cached IK solvers.

use criterion::{criterion_group, criterion_main, Criterion};

use kinetic_core::Pose;
use kinetic_kinematics::{
    forward_kinematics, solve_bio_ik, solve_sqp, BioIKConfig, CacheConfig, IKCache,
    KinematicChain, SQPConfig,
};
use kinetic_robot::Robot;

const URDF: &str = r#"<?xml version="1.0"?>
<robot name="bench_3dof">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="tip"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="10"/>
  </joint>
</robot>"#;

fn setup() -> (Robot, KinematicChain, Pose) {
    let robot = Robot::from_urdf_string(URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
    let target = forward_kinematics(&robot, &chain, &[0.5, 0.3, -0.2]).unwrap();
    (robot, chain, target)
}

fn bench_bio_ik(c: &mut Criterion) {
    let (robot, chain, target) = setup();

    c.bench_function("bio_ik_3dof_pop30_gen50", |b| {
        b.iter(|| {
            solve_bio_ik(&robot, &chain, &target, &[0.0; 3], &BioIKConfig {
                population_size: 30,
                max_generations: 50,
                ..Default::default()
            })
        })
    });
}

fn bench_sqp(c: &mut Criterion) {
    let (robot, chain, target) = setup();

    c.bench_function("sqp_3dof_100iter", |b| {
        b.iter(|| solve_sqp(&robot, &chain, &target, &[0.0; 3], &SQPConfig::default()))
    });
}

fn bench_cached_ik(c: &mut Criterion) {
    let (robot, chain, target) = setup();
    let mut cache = IKCache::new(CacheConfig {
        warm_samples: 500,
        ..Default::default()
    });
    cache.warm(&robot, &chain);

    c.bench_function("cached_ik_lookup", |b| {
        b.iter(|| cache.lookup(&target))
    });
}

criterion_group!(benches, bench_bio_ik, bench_sqp, bench_cached_ik);
criterion_main!(benches);
