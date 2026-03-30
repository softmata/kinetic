//! Benchmarks for octree insertion, query, and raycasting.

use criterion::{criterion_group, criterion_main, Criterion};
use kinetic_perception::octree::{Octree, OctreeConfig};
use kinetic_perception::depth::{deproject_u16, CameraIntrinsics, DepthConfig, DistortionModel};

fn bench_octree_insertion(c: &mut Criterion) {
    c.bench_function("octree_insert_1000_points", |b| {
        b.iter(|| {
            let mut octree = Octree::new(OctreeConfig {
                max_depth: 8,
                root_half_size: 5.0,
                ..Default::default()
            });
            for i in 0..1000 {
                let t = i as f64 * 0.004;
                octree.insert_point(t.cos() * 2.0, t.sin() * 2.0, t * 0.1);
            }
        })
    });
}

fn bench_octree_query(c: &mut Criterion) {
    let mut octree = Octree::new(OctreeConfig {
        max_depth: 8,
        root_half_size: 5.0,
        ..Default::default()
    });
    for i in 0..5000 {
        let t = i as f64 * 0.001;
        octree.insert_point(t.cos() * 2.0, t.sin() * 2.0, t * 0.05);
    }

    c.bench_function("octree_query_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let t = i as f64 * 0.003;
                octree.query(t, 0.0, 0.0);
            }
        })
    });

    c.bench_function("octree_radius_query", |b| {
        b.iter(|| octree.query_radius(0.0, 0.0, 0.0, 1.0))
    });
}

fn bench_octree_raycasting(c: &mut Criterion) {
    c.bench_function("octree_insert_ray_100", |b| {
        b.iter(|| {
            let mut octree = Octree::new(OctreeConfig {
                max_depth: 6,
                root_half_size: 3.0,
                ..Default::default()
            });
            for i in 0..100 {
                let angle = i as f64 * 0.063;
                octree.insert_ray(
                    [0.0, 0.0, 0.0],
                    [angle.cos() * 2.0, angle.sin() * 2.0, 0.5],
                );
            }
        })
    });
}

fn bench_depth_deproject(c: &mut Criterion) {
    let intrinsics = CameraIntrinsics::new(525.0, 525.0, 319.5, 239.5, 640, 480);
    let mut depth = vec![0u16; 640 * 480];
    for i in 0..640 * 480 {
        depth[i] = 1000 + (i % 3000) as u16;
    }

    c.bench_function("deproject_640x480_u16", |b| {
        b.iter(|| deproject_u16(&depth, &intrinsics, &DistortionModel::None, 0.001, &DepthConfig::default()))
    });
}

criterion_group!(benches, bench_octree_insertion, bench_octree_query, bench_octree_raycasting, bench_depth_deproject);
criterion_main!(benches);
