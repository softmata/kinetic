//! Inverse kinematics benchmarks.
//!
//! Performance targets:
//! - DLS 7-DOF convergence: <500 µs
//! - FABRIK 3-DOF: <300 µs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("../examples/panda_urdf.txt");

const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

fn bench_dls_7dof(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(PANDA_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

    let q_target = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

    let config = IKConfig::dls()
        .with_seed(robot.mid_configuration().to_vec())
        .with_max_iterations(300);

    c.bench_function("ik_dls_7dof", |b| {
        b.iter(|| {
            black_box(solve_ik(&robot, &chain, &target, &config).unwrap());
        })
    });
}

fn bench_fabrik_3dof(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let q_target = vec![0.3, 0.5, -0.2];
    let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

    let config = IKConfig::fabrik()
        .with_seed(vec![0.0, 0.0, 0.0])
        .with_max_iterations(200);

    c.bench_function("ik_fabrik_3dof", |b| {
        b.iter(|| {
            black_box(solve_ik(&robot, &chain, &target, &config).unwrap());
        })
    });
}

fn bench_dls_3dof(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let q_target = vec![0.3, 0.5, -0.2];
    let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

    let config = IKConfig::dls().with_seed(vec![0.0, 0.0, 0.0]);

    c.bench_function("ik_dls_3dof", |b| {
        b.iter(|| {
            black_box(solve_ik(&robot, &chain, &target, &config).unwrap());
        })
    });
}

criterion_group!(benches, bench_dls_7dof, bench_fabrik_3dof, bench_dls_3dof);
criterion_main!(benches);
