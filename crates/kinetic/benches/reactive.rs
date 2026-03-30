//! Reactive control (servo / RMP) benchmarks.
//!
//! Performance targets:
//! - RMP tick: <200 us
//! - Servo send_twist: <500 us (includes collision check)

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::core::Twist;
use kinetic::prelude::*;
use kinetic::reactive::{PolicyType, Servo, ServoConfig, RMP};

const PANDA_URDF: &str = include_str!("../examples/panda_urdf.txt");

fn setup_servo() -> (Servo, Vec<f64>) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = ServoConfig::default();

    let mut servo = Servo::new(&robot, &scene, config).unwrap();

    // Set non-singular initial state
    let init_pos = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let init_vel = vec![0.0; 7];
    servo.set_state(&init_pos, &init_vel).unwrap();

    (servo, init_pos)
}

fn bench_servo_twist(c: &mut Criterion) {
    let (mut servo, _) = setup_servo();

    let twist = Twist::new(Vector3::new(0.05, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0));

    c.bench_function("servo_send_twist", |b| {
        b.iter(|| {
            black_box(servo.send_twist(&twist).unwrap());
        })
    });
}

fn bench_servo_joint_jog(c: &mut Criterion) {
    let (mut servo, _) = setup_servo();

    c.bench_function("servo_joint_jog", |b| {
        b.iter(|| {
            black_box(servo.send_joint_jog(0, 0.1).unwrap());
        })
    });
}

fn bench_servo_state(c: &mut Criterion) {
    let (servo, _) = setup_servo();

    c.bench_function("servo_state", |b| {
        b.iter(|| {
            black_box(servo.state());
        })
    });
}

fn bench_rmp_tick(c: &mut Criterion) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

    let q = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let qd = vec![0.0; 7];

    // Target position for attractor
    let target_q = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
    let target_pose = forward_kinematics(&robot, &chain, &target_q).unwrap();

    let mut rmp = RMP::new(&robot).unwrap();
    rmp.add(PolicyType::ReachTarget {
        target_pose: target_pose.0,
        gain: 10.0,
    });

    c.bench_function("rmp_tick", |b| {
        b.iter(|| {
            black_box(rmp.compute(&q, &qd, 0.002).unwrap());
        })
    });
}

fn bench_rmp_combined(c: &mut Criterion) {
    let robot = Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap());
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

    let q = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let qd = vec![0.0; 7];

    let target_q = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
    let target_pose = forward_kinematics(&robot, &chain, &target_q).unwrap();

    let mut rmp = RMP::new(&robot).unwrap();
    rmp.add(PolicyType::ReachTarget {
        target_pose: target_pose.0,
        gain: 10.0,
    });
    rmp.add(PolicyType::JointLimitAvoidance {
        margin: 0.1,
        gain: 5.0,
    });
    rmp.add(PolicyType::Damping { coefficient: 2.0 });

    c.bench_function("rmp_combined_3_policies", |b| {
        b.iter(|| {
            black_box(rmp.compute(&q, &qd, 0.002).unwrap());
        })
    });
}

criterion_group!(
    benches,
    bench_servo_twist,
    bench_servo_joint_jog,
    bench_servo_state,
    bench_rmp_tick,
    bench_rmp_combined,
);
criterion_main!(benches);
