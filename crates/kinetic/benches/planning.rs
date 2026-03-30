//! Motion planning benchmarks.
//!
//! Performance targets:
//! - RRT simple (no obstacles): <100 us p50
//! - RRT cluttered (10 obstacles): <20 ms p50
//! - Cartesian 20cm: <500 us
//! - GCS build+plan: <50 ms
//! - ConstrainedRRT: <50 ms

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kinetic::collision::CollisionEnvironment;
use kinetic::planning::{
    CartesianConfig, CartesianPlanner, CollisionChecker, ConstrainedRRT, GCSPlanner, IrisConfig,
};
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("../examples/panda_urdf.txt");

fn setup_robot() -> Arc<Robot> {
    Arc::new(Robot::from_urdf_string(PANDA_URDF).unwrap())
}

fn bench_plan_simple(c: &mut Criterion) {
    let robot = setup_robot();
    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    c.bench_function("plan_simple_no_obstacles", |b| {
        b.iter(|| {
            black_box(planner.plan(&start, &goal).unwrap());
        })
    });
}

fn bench_plan_with_table(c: &mut Criterion) {
    let robot = setup_robot();

    let mut scene = Scene::new(&robot).unwrap();
    let table_pose = Isometry3::from_parts(
        nalgebra::Translation3::new(0.5, 0.0, 0.4),
        UnitQuaternion::identity(),
    );
    scene.add("table", Shape::Cuboid(0.5, 0.4, 0.01), table_pose);

    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    c.bench_function("plan_with_table", |b| {
        b.iter(|| {
            black_box(planner.plan(&start, &goal).unwrap());
        })
    });
}

fn bench_plan_cluttered(c: &mut Criterion) {
    let robot = setup_robot();

    let mut scene = Scene::new(&robot).unwrap();

    // Add 10 obstacles around the workspace
    for i in 0..10 {
        let x = 0.3 + (i as f64 % 5.0) * 0.12;
        let y = -0.2 + (i as f64 / 5.0).floor() * 0.2;
        let z = 0.3 + (i as f64 % 3.0) * 0.15;
        let pose = Isometry3::from_parts(
            nalgebra::Translation3::new(x, y, z),
            UnitQuaternion::identity(),
        );
        scene.add(&format!("obs_{i}"), Shape::Sphere(0.04), pose);
    }

    let planner = Planner::new(&robot).unwrap();

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    c.bench_function("plan_cluttered_10_obstacles", |b| {
        b.iter(|| {
            // Planning may fail in cluttered env; we measure time regardless
            let _ = black_box(planner.plan(&start, &goal));
        })
    });
}

fn bench_cartesian_linear(c: &mut Criterion) {
    let robot = setup_robot();
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();
    let cartesian = CartesianPlanner::new(robot.clone(), chain);

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    // 10cm linear move in z
    let direction = Vector3::new(0.0, 0.0, 0.10);
    let config = CartesianConfig::default();

    c.bench_function("cartesian_linear_10cm", |b| {
        b.iter(|| {
            let _ = black_box(cartesian.plan_relative(&start, &direction, &config));
        })
    });
}

fn bench_gcs_build_and_plan(c: &mut Criterion) {
    // GCS needs a collision checker and joint limits
    struct NeverCollides;
    impl CollisionChecker for NeverCollides {
        fn is_in_collision(&self, _joints: &[f64]) -> bool {
            false
        }
    }

    let limits: Vec<(f64, f64)> = vec![
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ];

    let config = IrisConfig {
        num_regions: 5,
        max_iterations: 10,
        ..Default::default()
    };

    let mut group = c.benchmark_group("gcs");
    group.sample_size(10);

    group.bench_function("gcs_build_and_plan", |b| {
        b.iter(|| {
            let planner = GCSPlanner::build(&NeverCollides, &limits, &config, 0.1).unwrap();
            let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
            let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
            let _ = black_box(planner.plan(&start, &goal));
        })
    });

    group.finish();
}

fn bench_constrained_rrt(c: &mut Criterion) {
    let robot = setup_robot();
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8").unwrap();

    let env = CollisionEnvironment::empty(0.05, AABB::symmetric(10.0));
    let planner_config = PlannerConfig {
        timeout: Duration::from_secs(5),
        max_iterations: 5000,
        collision_margin: 0.01,
        shortcut_iterations: 10,
        smooth: false,
    };
    let rrt_config = RRTConfig {
        step_size: 0.2,
        goal_bias: 0.1,
    };

    // Orientation constraint: keep end-effector z-axis upright
    let constraints = vec![Constraint::Orientation {
        link: "panda_link8".to_string(),
        axis: Vector3::new(0.0, 0.0, 1.0),
        tolerance: 0.3,
    }];

    let crrt = ConstrainedRRT::new(
        robot.clone(),
        chain,
        env,
        planner_config,
        rrt_config,
        constraints,
    );

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5]));

    let mut group = c.benchmark_group("constrained_rrt");
    group.sample_size(10);

    group.bench_function("constrained_rrt_orientation", |b| {
        b.iter(|| {
            let _ = black_box(crrt.plan(&start, &goal));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_plan_simple,
    bench_plan_with_table,
    bench_plan_cluttered,
    bench_cartesian_linear,
    bench_gcs_build_and_plan,
    bench_constrained_rrt,
);
criterion_main!(benches);
