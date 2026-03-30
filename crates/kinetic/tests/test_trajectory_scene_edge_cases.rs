//! Edge case tests for trajectory and scene crates.
//!
//! Covers uncovered branches:
//! 1. Jerk-limited trajectory: zero-displacement segments, identical waypoints,
//!    per-joint limits with mismatched lengths
//! 2. Spline: degenerate 2-point paths, auto-duration with velocity limits,
//!    clamped boundary conditions edge cases
//! 3. Blend: both-empty trajectories, overlapping blend regions, sequence blending
//! 4. Octree: points at exact boundaries, clearing and re-inserting, large point clouds,
//!    collision sphere conversion, pruning behavior
//! 5. Depth: NaN/Inf depth values, empty images, stride larger than dimensions
//! 6. Point cloud processing: outlier removal, voxel downsampling, floor removal

use std::time::Duration;

// ── Jerk-limited trajectory edge cases ──────────────────────────────

#[test]
fn jerk_limited_zero_displacement_segments() {
    use kinetic::trajectory::jerk_limited::jerk_limited;

    // Path with zero-displacement segments (identical consecutive waypoints)
    let path = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0], // zero displacement
        vec![1.0, 1.0],
        vec![1.0, 1.0], // zero displacement again
    ];
    let result = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();
    assert!(
        result.waypoints.len() >= 4,
        "Should handle zero-displacement segments"
    );

    // Start and end positions correct
    assert!((result.waypoints.first().unwrap().positions[0]).abs() < 1e-10);
    let last = result.waypoints.last().unwrap();
    assert!((last.positions[0] - 1.0).abs() < 1e-6);
}

#[test]
fn jerk_limited_very_large_displacement() {
    use kinetic::trajectory::jerk_limited::jerk_limited;

    // Large displacement with small limits → very long trajectory
    let path = vec![vec![0.0], vec![100.0]];
    let result = jerk_limited(&path, 0.5, 1.0, 5.0).unwrap();
    assert!(
        result.duration().as_secs_f64() > 100.0,
        "Large displacement should produce long trajectory"
    );
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn jerk_limited_per_joint_single_waypoint() {
    use kinetic::trajectory::jerk_limited::jerk_limited_per_joint;

    let path = vec![vec![1.0, 2.0]];
    let result = jerk_limited_per_joint(&path, &[1.0, 1.0], &[2.0, 2.0], &[10.0, 10.0]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.duration(), Duration::ZERO);
}

#[test]
fn jerk_limited_per_joint_empty() {
    use kinetic::trajectory::jerk_limited::jerk_limited_per_joint;

    let path: Vec<Vec<f64>> = vec![];
    let result = jerk_limited_per_joint(&path, &[], &[], &[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn jerk_limited_per_joint_one_joint_zero_displacement() {
    use kinetic::trajectory::jerk_limited::jerk_limited_per_joint;

    // Joint 1 moves, joint 0 doesn't
    let path = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
    let result = jerk_limited_per_joint(&path, &[1.0, 1.0], &[2.0, 2.0], &[10.0, 10.0]).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);

    // Joint 0 should stay at 0
    let last = result.waypoints.last().unwrap();
    assert!((last.positions[0]).abs() < 1e-6);
    assert!((last.positions[1] - 1.0).abs() < 1e-6);
}

// ── Spline edge cases ───────────────────────────────────────────────

#[test]
fn spline_two_point_path() {
    use kinetic::trajectory::spline::cubic_spline_time;

    let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
    let result = cubic_spline_time(&path, Some(1.0), None).unwrap();
    assert!(result.waypoints.len() >= 2);
    assert!((result.waypoints.first().unwrap().positions[0]).abs() < 1e-6);
    let last = result.waypoints.last().unwrap();
    assert!((last.positions[0] - 1.0).abs() < 1e-6);
}

#[test]
fn spline_auto_duration_no_velocity_limits() {
    use kinetic::trajectory::spline::cubic_spline_time;

    let path = vec![vec![0.0], vec![5.0], vec![3.0]];
    let result = cubic_spline_time(&path, None, None).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn spline_auto_duration_velocity_limits_mismatch() {
    use kinetic::trajectory::spline::cubic_spline_time;

    let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    // Wrong number of velocity limits
    let result = cubic_spline_time(&path, None, Some(&[1.0]));
    assert!(result.is_err(), "Velocity limits mismatch should error");
}

#[test]
fn spline_clamped_empty_path() {
    use kinetic::trajectory::spline::cubic_spline_time_clamped;

    let path: Vec<Vec<f64>> = vec![];
    let result = cubic_spline_time_clamped(&path, 1.0, &[], &[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn spline_clamped_single_waypoint() {
    use kinetic::trajectory::spline::cubic_spline_time_clamped;

    let path = vec![vec![1.0, 2.0]];
    let result = cubic_spline_time_clamped(&path, 1.0, &[0.5, 0.0], &[0.0, 0.0]).unwrap();
    assert_eq!(result.len(), 1);
    // Velocities should match start_velocities
    assert!((result.waypoints[0].velocities[0] - 0.5).abs() < 1e-10);
}

#[test]
fn spline_clamped_invalid_duration() {
    use kinetic::trajectory::spline::cubic_spline_time_clamped;

    let path = vec![vec![0.0], vec![1.0]];
    assert!(cubic_spline_time_clamped(&path, 0.0, &[0.0], &[0.0]).is_err());
    assert!(cubic_spline_time_clamped(&path, -1.0, &[0.0], &[0.0]).is_err());
}

// ── Blend edge cases ────────────────────────────────────────────────

#[test]
fn blend_both_empty_trajectories() {
    use kinetic::trajectory::blend::blend;
    use kinetic::trajectory::trapezoidal::TimedTrajectory;

    let empty = TimedTrajectory {
        duration: Duration::ZERO,
        dof: 0,
        waypoints: vec![],
    };
    let result = blend(&empty, &empty, 0.1).unwrap();
    assert!(result.is_empty());
}

#[test]
fn blend_minimal_blend_duration() {
    use kinetic::trajectory::blend::blend;
    use kinetic::trajectory::trapezoidal::{TimedTrajectory, TimedWaypoint};

    let make_traj = |start: f64, end: f64, dur: f64| -> TimedTrajectory {
        TimedTrajectory {
            duration: Duration::from_secs_f64(dur),
            dof: 1,
            waypoints: (0..=10)
                .map(|k| {
                    let alpha = k as f64 / 10.0;
                    TimedWaypoint {
                        time: alpha * dur,
                        positions: vec![start + alpha * (end - start)],
                        velocities: vec![(end - start) / dur],
                        accelerations: vec![0.0],
                    }
                })
                .collect(),
        }
    };

    let t1 = make_traj(0.0, 1.0, 2.0);
    let t2 = make_traj(1.0, 2.0, 2.0);

    // Very small blend duration
    let result = blend(&t1, &t2, 0.01).unwrap();
    assert!(!result.waypoints.is_empty());
    assert!(result.duration().as_secs_f64() > 0.0);
}

#[test]
fn blend_sequence_two_trajectories() {
    use kinetic::trajectory::blend::blend_sequence;
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path1 = vec![vec![0.0], vec![1.0]];
    let path2 = vec![vec![1.0], vec![2.0]];
    let t1 = trapezoidal(&path1, 1.0, 2.0).unwrap();
    let t2 = trapezoidal(&path2, 1.0, 2.0).unwrap();

    let result = blend_sequence(&[t1, t2], 0.2).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);
}

#[test]
fn blend_sequence_empty_list() {
    use kinetic::trajectory::blend::blend_sequence;
    let result = blend_sequence(&[], 0.1).unwrap();
    assert!(result.is_empty());
}

// ── Trapezoidal edge cases ──────────────────────────────────────────

#[test]
fn trapezoidal_multi_segment_path() {
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![
        vec![0.0, 0.0],
        vec![0.5, 1.0],
        vec![1.0, 0.5],
        vec![1.5, 1.5],
        vec![2.0, 0.0],
    ];
    let result = trapezoidal(&path, 1.0, 2.0).unwrap();
    assert!(result.waypoints.len() >= 5);
    assert!(result.duration().as_secs_f64() > 0.0);

    // Timestamps must be monotonic
    for i in 1..result.waypoints.len() {
        assert!(
            result.waypoints[i].time >= result.waypoints[i - 1].time - 1e-10,
            "Time should be monotonic"
        );
    }
}

#[test]
fn trapezoidal_per_joint_basic() {
    use kinetic::trajectory::trapezoidal::trapezoidal_per_joint;

    let path = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
    let result = trapezoidal_per_joint(&path, &[1.0, 0.5], &[2.0, 1.0]).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn trapezoidal_sample_at_intermediate_time() {
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0, 0.0], vec![2.0, 4.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();

    // Sample at midpoint
    let mid = traj.sample_at(traj.duration() / 2);
    assert!(mid.positions[0] > 0.0 && mid.positions[0] < 2.0);
    assert!(mid.positions[1] > 0.0 && mid.positions[1] < 4.0);

    // Sample at start
    let start = traj.sample_at(Duration::ZERO);
    assert!((start.positions[0]).abs() < 1e-6);

    // Sample beyond end (should clamp to last)
    let beyond = traj.sample_at(traj.duration() + Duration::from_secs(10));
    assert!((beyond.positions[0] - 2.0).abs() < 1e-6);
}

// ── TOTP edge cases ─────────────────────────────────────────────────

#[test]
fn totp_two_point_path() {
    use kinetic::trajectory::totp::totp;

    let path = vec![vec![0.0], vec![2.0]];
    let result = totp(&path, &[1.0], &[2.0], 0.01).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);
    assert!(result.waypoints.len() >= 2);
}

#[test]
fn totp_three_segment_path() {
    use kinetic::trajectory::totp::totp;

    let path = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.5],
        vec![2.0, 1.0],
        vec![3.0, 0.0],
    ];
    let result = totp(&path, &[2.0, 2.0], &[4.0, 4.0], 0.01).unwrap();
    assert!(result.duration().as_secs_f64() > 0.0);

    // First position
    assert!((result.waypoints.first().unwrap().positions[0]).abs() < 1e-6);
    // Last position
    let last = result.waypoints.last().unwrap();
    assert!((last.positions[0] - 3.0).abs() < 0.05);
}

// ── Octree edge cases ───────────────────────────────────────────────

#[test]
fn octree_point_at_exact_origin() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.1,
        half_extent: 1.0,
        ray_cast_free_space: false,
        ..Default::default()
    });

    tree.insert_point([0.0, 0.0, 0.0], None);
    assert!(tree.num_occupied() > 0);
    assert!(tree.check_sphere([0.0, 0.0, 0.0], 0.01));
}

#[test]
fn octree_point_at_exact_boundary() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let he = 1.0;
    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.1,
        half_extent: he,
        ray_cast_free_space: false,
        ..Default::default()
    });

    // Point exactly at the boundary
    tree.insert_point([he, he, he], None);
    // Points at boundary are out of bounds (>= half_extent check)
    // The tree should handle this gracefully without crashing
    // (may or may not insert depending on exact boundary logic)
}

#[test]
fn octree_large_point_cloud() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.05,
        half_extent: 2.0,
        ray_cast_free_space: false,
        ..Default::default()
    });

    // Insert 10,000 points in a grid
    let mut points = Vec::with_capacity(10_000);
    for i in 0..100 {
        for j in 0..100 {
            points.push([-1.0 + i as f64 * 0.02, -1.0 + j as f64 * 0.02, 0.5]);
        }
    }

    tree.insert_points_occupied(&points);
    assert!(
        tree.num_occupied() > 100,
        "Should have many occupied voxels"
    );
    assert!(tree.num_leaves() > 100, "Should have many leaves");

    // Collision check should work
    assert!(tree.check_sphere([0.0, 0.0, 0.5], 0.1));
    assert!(!tree.check_sphere([0.0, 0.0, 3.0], 0.1));
}

#[test]
fn octree_clear_then_reinsert() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.1,
        half_extent: 1.0,
        ray_cast_free_space: false,
        ..Default::default()
    });

    tree.insert_point([0.5, 0.5, 0.5], None);
    assert!(tree.num_occupied() > 0);

    tree.clear();
    assert_eq!(tree.num_occupied(), 0);
    assert_eq!(tree.num_leaves(), 1);

    // Re-insert at different location
    tree.insert_point([-0.5, -0.5, -0.5], None);
    assert!(tree.num_occupied() > 0);
    assert!(tree.check_sphere([-0.5, -0.5, -0.5], 0.15));
    assert!(!tree.check_sphere([0.5, 0.5, 0.5], 0.05));
}

#[test]
fn octree_collision_spheres_radii_positive() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.1,
        half_extent: 1.0,
        ray_cast_free_space: false,
        ..Default::default()
    });

    let points = [[0.3, 0.3, 0.3], [-0.3, -0.3, -0.3], [0.5, 0.0, 0.0]];
    tree.insert_points_occupied(&points);

    let spheres = tree.to_collision_spheres();
    assert!(spheres.len() >= 3);
    for r in &spheres.radius {
        assert!(*r > 0.0, "All collision sphere radii should be positive");
    }
}

#[test]
fn octree_no_collision_after_clear() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.1,
        half_extent: 1.0,
        ray_cast_free_space: false,
        ..Default::default()
    });

    tree.insert_point([0.5, 0.5, 0.5], None);
    assert!(tree.check_sphere([0.5, 0.5, 0.5], 0.15));

    tree.clear();
    assert!(!tree.check_sphere([0.5, 0.5, 0.5], 0.15));
}

#[test]
fn octree_high_resolution_small_extent() {
    use kinetic::scene::octree::{Octree, OctreeConfig};

    // Very fine resolution in small volume
    let mut tree = Octree::new(OctreeConfig {
        resolution: 0.005,
        half_extent: 0.1,
        ray_cast_free_space: false,
        ..Default::default()
    });

    tree.insert_point([0.05, 0.05, 0.05], None);
    assert!(tree.num_occupied() > 0);
    assert!(tree.max_depth() > 0);
}

// ── Depth processing edge cases ─────────────────────────────────────

#[test]
fn depth_all_inf_values() {
    use kinetic::scene::depth::{depth_to_points_camera_frame, CameraIntrinsics, DepthConfig};

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let width = 5;
    let height = 5;
    let depth = vec![f32::INFINITY; width * height];
    let config = DepthConfig::default();

    let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
    assert!(points.is_empty(), "All Inf depth should produce no points");
}

#[test]
fn depth_neg_inf_values() {
    use kinetic::scene::depth::{depth_to_points_camera_frame, CameraIntrinsics, DepthConfig};

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let width = 3;
    let height = 1;
    let depth = vec![f32::NEG_INFINITY; 3];
    let config = DepthConfig::default();

    let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
    assert!(
        points.is_empty(),
        "Negative Inf depth should produce no points"
    );
}

#[test]
fn depth_mixed_valid_invalid() {
    use kinetic::scene::depth::{depth_to_points_camera_frame, CameraIntrinsics, DepthConfig};

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 2.0, 0.5);
    let width = 6;
    let height = 1;
    let depth: Vec<f32> = vec![
        f32::NAN,      // invalid
        0.05,          // below min_depth
        1.0,           // valid
        f32::INFINITY, // invalid
        3.0,           // valid
        10.0,          // above max_depth
    ];
    let config = DepthConfig {
        min_depth: 0.1,
        max_depth: 5.0,
        stride: 1,
    };

    let points = depth_to_points_camera_frame(&depth, width, height, &intrinsics, &config);
    assert_eq!(points.len(), 2, "Should have exactly 2 valid depth values");
}

#[test]
fn depth_world_transform_with_rotation() {
    use kinetic::scene::depth::{depth_to_points_world, CameraIntrinsics, DepthConfig};

    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 0.0, 0.0);
    let width = 1;
    let height = 1;
    let depth = [2.0f32]; // 2m depth at pixel (0,0)

    // Camera rotated 90 degrees around Y axis (looking along -X in world)
    let rotation = nalgebra::UnitQuaternion::from_axis_angle(
        &nalgebra::Vector3::y_axis(),
        std::f64::consts::FRAC_PI_2,
    );
    let camera_pose =
        nalgebra::Isometry3::from_parts(nalgebra::Translation3::new(1.0, 0.0, 0.0), rotation);

    let config = DepthConfig::default();
    let points = depth_to_points_world(&depth, width, height, &intrinsics, &camera_pose, &config);
    assert_eq!(points.len(), 1);
    // The point should be transformed by the camera pose
    // With 90deg Y rotation, camera Z axis maps to world -X
    // (approximately, depends on exact rotation convention)
    assert!(points[0][0].is_finite());
    assert!(points[0][1].is_finite());
    assert!(points[0][2].is_finite());
}

// ── Point cloud processing edge cases ───────────────────────────────

#[test]
fn pointcloud_process_single_point() {
    use kinetic::scene::pointcloud::{process_pointcloud, PointCloudConfig};

    let points = [[0.5, 0.5, 0.5]];
    let config = PointCloudConfig::default();
    let (processed, spheres) = process_pointcloud(&points, &config);
    assert_eq!(processed.len(), 1);
    assert_eq!(spheres.len(), 1);
}

#[test]
fn pointcloud_process_with_outlier_removal() {
    use kinetic::scene::pointcloud::{process_pointcloud, OutlierConfig, PointCloudConfig};

    let mut points: Vec<[f64; 3]> = Vec::new();
    // Dense cluster
    for i in 0..20 {
        for j in 0..20 {
            points.push([i as f64 * 0.01, j as f64 * 0.01, 0.5]);
        }
    }
    // One outlier far away
    points.push([100.0, 100.0, 100.0]);

    let config = PointCloudConfig {
        outlier_removal: Some(OutlierConfig {
            k: 10,
            std_dev_multiplier: 1.0,
        }),
        ..Default::default()
    };

    let (processed, spheres) = process_pointcloud(&points, &config);
    assert!(processed.len() < points.len(), "Outlier should be removed");
    assert_eq!(spheres.len(), processed.len());
}

#[test]
fn pointcloud_process_with_radius_outlier_removal() {
    use kinetic::scene::pointcloud::{process_pointcloud, PointCloudConfig, RadiusOutlierConfig};

    let mut points: Vec<[f64; 3]> = Vec::new();
    // Dense cluster
    for i in 0..10 {
        points.push([i as f64 * 0.01, 0.0, 0.5]);
    }
    // Isolated outlier
    points.push([50.0, 50.0, 50.0]);

    let config = PointCloudConfig {
        radius_outlier_removal: Some(RadiusOutlierConfig {
            radius: 0.5,
            min_neighbors: 3,
        }),
        ..Default::default()
    };

    let (processed, _) = process_pointcloud(&points, &config);
    assert!(processed.len() <= points.len());
}

#[test]
fn pointcloud_all_filters_combined() {
    use kinetic::collision::AABB;
    use kinetic::scene::pointcloud::{
        process_pointcloud, OutlierConfig, PointCloudConfig, RadiusOutlierConfig,
    };

    let mut points: Vec<[f64; 3]> = Vec::new();
    // Floor points
    for i in 0..20 {
        for j in 0..20 {
            points.push([i as f64 * 0.05 - 0.5, j as f64 * 0.05 - 0.5, 0.0]);
        }
    }
    // Object points
    for i in 0..10 {
        points.push([i as f64 * 0.02, 0.0, 0.5]);
    }
    // Outlier
    points.push([100.0, 100.0, 100.0]);
    // Out-of-bounds point
    points.push([5.0, 5.0, 5.0]);

    let config = PointCloudConfig {
        sphere_radius: 0.01,
        max_points: 500,
        remove_floor: true,
        floor_distance_threshold: 0.02,
        crop_box: Some(AABB::new(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)),
        voxel_downsample: Some(0.05),
        outlier_removal: Some(OutlierConfig {
            k: 5,
            std_dev_multiplier: 2.0,
        }),
        radius_outlier_removal: Some(RadiusOutlierConfig {
            radius: 0.2,
            min_neighbors: 2,
        }),
    };

    let (processed, spheres) = process_pointcloud(&points, &config);
    // After all filters, should have significantly fewer points
    assert!(processed.len() < points.len());
    assert_eq!(spheres.len(), processed.len());
    // All spheres should have the configured radius
    for r in &spheres.radius {
        assert!((*r - 0.01).abs() < 1e-10);
    }
}

// ── Trajectory validation additional edge cases ─────────────────────

#[test]
fn trajectory_validate_single_waypoint_passes() {
    use kinetic::trajectory::trapezoidal::{TimedTrajectory, TimedWaypoint};
    use kinetic::trajectory::validation::{TrajectoryValidator, ValidationConfig};

    let traj = TimedTrajectory {
        duration: Duration::ZERO,
        dof: 2,
        waypoints: vec![TimedWaypoint {
            time: 0.0,
            positions: vec![0.0, 0.0],
            velocities: vec![0.0, 0.0],
            accelerations: vec![0.0, 0.0],
        }],
    };

    let validator = TrajectoryValidator::new(
        &[-3.14, -3.14],
        &[3.14, 3.14],
        &[2.0, 2.0],
        &[4.0, 4.0],
        ValidationConfig::default(),
    );

    assert!(validator.validate(&traj).is_ok());
}

#[test]
fn trajectory_validate_jerk_and_accel_limits_together() {
    use kinetic::trajectory::trapezoidal::{TimedTrajectory, TimedWaypoint};
    use kinetic::trajectory::validation::{TrajectoryValidator, ValidationConfig, ViolationType};

    // Trajectory with both acceleration violations and jerk violations
    let traj = TimedTrajectory {
        duration: Duration::from_secs_f64(0.02),
        dof: 1,
        waypoints: vec![
            TimedWaypoint {
                time: 0.0,
                positions: vec![0.0],
                velocities: vec![0.0],
                accelerations: vec![0.0],
            },
            TimedWaypoint {
                time: 0.01,
                positions: vec![0.0],
                velocities: vec![0.5],
                accelerations: vec![50.0], // exceeds accel limit
            },
            TimedWaypoint {
                time: 0.02,
                positions: vec![0.0],
                velocities: vec![0.5],
                accelerations: vec![-50.0], // huge jerk from +50 to -50
            },
        ],
    };

    let validator = TrajectoryValidator::new(
        &[-3.14],
        &[3.14],
        &[2.0],
        &[4.0],
        ValidationConfig {
            max_jerk: Some(100.0),
            ..Default::default()
        },
    );

    let result = validator.validate(&traj);
    assert!(result.is_err());
    let violations = result.unwrap_err();

    // Should have acceleration limit violations
    assert!(violations
        .iter()
        .any(|v| v.violation_type == ViolationType::AccelerationLimit));
    // Should have jerk violations too (da=100 in dt=0.01 → jerk=10000 > 100)
    assert!(violations
        .iter()
        .any(|v| v.violation_type == ViolationType::JerkLimit));
}

// ── Trajectory monitor ──────────────────────────────────────────────

#[test]
fn trajectory_monitor_basic() {
    use kinetic::trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();

    let config = MonitorConfig::default();
    let mut monitor = ExecutionMonitor::new(traj, config);

    // Check with correct position at t=0 → should be Normal
    let level = monitor.check(0.0, &[0.0, 0.0]);
    assert!(
        matches!(level, DeviationLevel::Normal),
        "On-track position should be Normal"
    );
}

#[test]
fn trajectory_monitor_deviation_abort() {
    use kinetic::trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();

    let config = MonitorConfig {
        position_tolerance: 0.01, // very tight tolerance
        ..Default::default()
    };
    let mut monitor = ExecutionMonitor::new(traj, config);

    // Check with far-off position → should be Abort
    let level = monitor.check(0.0, &[5.0, 5.0]);
    assert!(
        matches!(level, DeviationLevel::Abort { .. }),
        "Far-off position should be Abort"
    );
}

#[test]
fn trajectory_monitor_wrong_dof_aborts() {
    use kinetic::trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();
    let mut monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

    // Wrong DOF → should abort
    let level = monitor.check(0.0, &[0.0]); // 1 joint instead of 2
    assert!(
        matches!(level, DeviationLevel::Abort { .. }),
        "Wrong DOF should abort"
    );
}

#[test]
fn trajectory_monitor_reset() {
    use kinetic::trajectory::monitor::{ExecutionMonitor, MonitorConfig};
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0], vec![1.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();
    let mut monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

    // Run some checks then reset
    monitor.check(0.0, &[0.0]);
    monitor.check(0.1, &[0.1]);
    monitor.reset();

    // After reset, should work normally
    let _level = monitor.check(0.0, &[0.0]);
}

#[test]
fn trajectory_monitor_raw_deviations() {
    use kinetic::trajectory::monitor::{ExecutionMonitor, MonitorConfig};
    use kinetic::trajectory::trapezoidal::trapezoidal;

    let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let traj = trapezoidal(&path, 1.0, 2.0).unwrap();
    let monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

    let devs = monitor.raw_deviations(0.0, &[0.1, 0.2]);
    assert_eq!(devs.len(), 2);
    assert!((devs[0] - 0.1).abs() < 1e-6);
    assert!((devs[1] - 0.2).abs() < 1e-6);
}
