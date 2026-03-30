//! Collision detection benchmarks.
//!
//! Performance targets:
//! - SIMD collision check 3-DOF + 10 obstacles: <500 ns
//! - Self-collision check: <200 ns
//! - 100 environment spheres: <2 us
//! - 1000 environment spheres: <20 us

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::collision::{RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic::prelude::*;

const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link">
    <collision><geometry><cylinder radius="0.05" length="0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.035" length="0.25"/></geometry></collision>
  </link>
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

fn bench_self_collision(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::default());
    let mut spheres = sphere_model.create_runtime();

    let q = vec![0.0, 0.5, -0.3];
    let link_poses = forward_kinematics_all(&robot, &chain, &q).unwrap();
    spheres.update(&link_poses);

    c.bench_function("collision_self_check", |b| {
        b.iter(|| {
            black_box(spheres.self_collision(&[]));
        })
    });
}

fn bench_env_collision_10_obstacles(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::default());
    let mut spheres = sphere_model.create_runtime();

    let q = vec![0.0, 0.5, -0.3];
    let link_poses = forward_kinematics_all(&robot, &chain, &q).unwrap();
    spheres.update(&link_poses);

    let mut obstacles = SpheresSoA::new();
    for i in 0..10 {
        let x = 0.3 + (i as f64) * 0.05;
        obstacles.push(x, 0.0, 0.3, 0.02, i);
    }

    c.bench_function("collision_env_10_obstacles", |b| {
        b.iter(|| {
            black_box(spheres.collides_with(&obstacles));
        })
    });
}

fn bench_env_collision_100_obstacles(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::default());
    let mut spheres = sphere_model.create_runtime();

    let q = vec![0.0, 0.5, -0.3];
    let link_poses = forward_kinematics_all(&robot, &chain, &q).unwrap();
    spheres.update(&link_poses);

    let mut obstacles = SpheresSoA::new();
    for i in 0..100 {
        let x = -1.0 + (i as f64 % 10.0) * 0.2;
        let y = -1.0 + ((i / 10) as f64) * 0.2;
        obstacles.push(x, y, 0.3, 0.02, i);
    }

    c.bench_function("collision_env_100_obstacles", |b| {
        b.iter(|| {
            black_box(spheres.collides_with(&obstacles));
        })
    });
}

fn bench_env_collision_1000_obstacles(c: &mut Criterion) {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

    let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::default());
    let mut spheres = sphere_model.create_runtime();

    let q = vec![0.0, 0.5, -0.3];
    let link_poses = forward_kinematics_all(&robot, &chain, &q).unwrap();
    spheres.update(&link_poses);

    let mut obstacles = SpheresSoA::new();
    for i in 0..1000 {
        let x = -2.0 + (i as f64 % 10.0) * 0.4;
        let y = -2.0 + ((i / 10) as f64 % 10.0) * 0.4;
        let z = ((i / 100) as f64) * 0.2;
        obstacles.push(x, y, z, 0.02, i);
    }

    c.bench_function("collision_env_1000_obstacles", |b| {
        b.iter(|| {
            black_box(spheres.collides_with(&obstacles));
        })
    });
}

fn bench_simd_sphere_check(c: &mut Criterion) {
    // Direct SoA sphere-sphere check
    let mut robot_spheres = SpheresSoA::new();
    for i in 0..20 {
        let x = (i as f64) * 0.1;
        robot_spheres.push(x, 0.0, 0.5, 0.03, i);
    }

    let mut obstacles = SpheresSoA::new();
    for i in 0..10 {
        let x = 0.3 + (i as f64) * 0.05;
        obstacles.push(x, 0.0, 0.3, 0.02, i);
    }

    c.bench_function("simd_sphere_20v10", |b| {
        b.iter(|| {
            black_box(robot_spheres.any_overlap_with_margin(&obstacles, 0.0));
        })
    });
}

fn bench_simd_min_distance(c: &mut Criterion) {
    let mut set_a = SpheresSoA::new();
    for i in 0..20 {
        set_a.push((i as f64) * 0.1, 0.0, 0.0, 0.03, i);
    }

    let mut set_b = SpheresSoA::new();
    for i in 0..10 {
        set_b.push(0.5 + (i as f64) * 0.05, 0.3, 0.0, 0.02, i);
    }

    c.bench_function("simd_min_distance_20v10", |b| {
        b.iter(|| {
            black_box(set_a.min_distance(&set_b));
        })
    });
}

criterion_group!(
    benches,
    bench_self_collision,
    bench_env_collision_10_obstacles,
    bench_env_collision_100_obstacles,
    bench_env_collision_1000_obstacles,
    bench_simd_sphere_check,
    bench_simd_min_distance,
);
criterion_main!(benches);
