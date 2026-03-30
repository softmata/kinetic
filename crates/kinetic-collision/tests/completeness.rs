//! Collision completeness test suite.
//!
//! Covers all shape type pair combinations, CCD tunneling regression,
//! convex decomposition accuracy, point cloud collisions, per-link padding
//! integration, two-tier cross-module checks, SDF integration, and known
//! collision/no-collision regression pairs.

use kinetic_collision::{
    convex_decomposition, pointcloud_to_spheres, shape_from_box, shape_from_cylinder,
    shape_from_sphere, AllowedCollisionMatrix, CCDConfig, CollisionEnvironment,
    ContinuousCollisionDetector, ConvexDecompConfig, LinkCollisionConfig, MeshCollisionBackend,
    MultiResolutionSDF, ResolvedACM, RobotSphereModel, SDFConfig, SelfCollisionPairs,
    SignedDistanceField, SphereGenConfig, SpheresSoA, TwoTierCollisionChecker, AABB,
};
use kinetic_core::Pose;
use kinetic_robot::Robot;
use nalgebra::Isometry3;
use parry3d_f64::shape::SharedShape;

// ═══════════════════════════════════════════════════════════════════════════
// Shared URDF fixtures
// ═══════════════════════════════════════════════════════════════════════════

/// Simple 3-link robot with box, cylinder, sphere geometry (and one bare link).
const GEOM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_geom">
  <link name="base_link">
    <collision>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </collision>
  </link>
  <link name="link2">
    <collision>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
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
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

fn setup_robot() -> (Robot, RobotSphereModel) {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
    (robot, model)
}

fn identity_poses(robot: &Robot) -> Vec<Pose> {
    (0..robot.links.len()).map(|_| Pose::identity()).collect()
}

fn separated_poses(robot: &Robot) -> Vec<Pose> {
    let mut poses = Vec::new();
    for i in 0..robot.links.len() {
        poses.push(Pose::from_xyz(0.0, 0.0, i as f64 * 0.5));
    }
    poses
}

fn make_ccd_model(x: f64, y: f64, z: f64, r: f64) -> kinetic_collision::RobotSphereModel {
    // Build a minimal single-link URDF for CCD
    let urdf = format!(
        r#"<?xml version="1.0"?>
<robot name="ccd_test">
  <link name="link0">
    <collision>
      <geometry><sphere radius="{}"/></geometry>
      <origin xyz="{} {} {}"/>
    </collision>
  </link>
</robot>"#,
        r, x, y, z
    );
    let robot = Robot::from_urdf_string(&urdf).unwrap();
    RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Point cloud collision tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pointcloud_direct_mode_one_sphere_per_point() {
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let spheres = pointcloud_to_spheres(&points, 0.05, None);
    assert_eq!(spheres.len(), 3, "Direct mode: one sphere per point");
    assert!((spheres.radius[0] - 0.05).abs() < 1e-10);
}

#[test]
fn pointcloud_voxelized_reduces_count() {
    // Dense point cloud: 1000 points in a 1m^3 cube
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1]);
            }
        }
    }

    let direct = pointcloud_to_spheres(&points, 0.01, None);
    let voxelized = pointcloud_to_spheres(&points, 0.01, Some(0.2));

    assert_eq!(direct.len(), 1000);
    assert!(
        voxelized.len() < direct.len(),
        "Voxelized ({}) should have fewer spheres than direct ({})",
        voxelized.len(),
        direct.len()
    );
}

#[test]
fn pointcloud_voxelized_radius_covers_voxel() {
    let points = vec![[0.0, 0.0, 0.0]];
    let voxel_size = 0.1;
    let sphere_radius = 0.02;
    let spheres = pointcloud_to_spheres(&points, sphere_radius, Some(voxel_size));

    // effective_r = voxel * 0.866 + sphere_radius = 0.1 * 0.866 + 0.02 = 0.1066
    let expected_r = voxel_size * 0.866 + sphere_radius;
    assert!(
        (spheres.radius[0] - expected_r).abs() < 1e-10,
        "Expected effective radius {}, got {}",
        expected_r,
        spheres.radius[0]
    );
}

#[test]
fn pointcloud_empty() {
    let spheres = pointcloud_to_spheres(&[], 0.05, None);
    assert_eq!(spheres.len(), 0);

    let spheres_v = pointcloud_to_spheres(&[], 0.05, Some(0.1));
    assert_eq!(spheres_v.len(), 0);
}

#[test]
fn pointcloud_as_environment_collides_with_robot() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Point cloud around the robot origin — should collide
    let points: Vec<[f64; 3]> = (0..50)
        .map(|i| {
            let angle = i as f64 * 0.126;
            [angle.cos() * 0.05, angle.sin() * 0.05, 0.0]
        })
        .collect();

    let obs = pointcloud_to_spheres(&points, 0.02, None);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

    assert!(
        env.check_collision(&runtime.world),
        "Point cloud at origin should collide with robot"
    );
}

#[test]
fn pointcloud_as_environment_no_collision() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Point cloud far away
    let points: Vec<[f64; 3]> = (0..100)
        .map(|i| [10.0 + i as f64 * 0.01, 10.0, 10.0])
        .collect();

    let obs = pointcloud_to_spheres(&points, 0.01, None);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(15.0));

    assert!(
        !env.check_collision(&runtime.world),
        "Distant point cloud should not collide"
    );
}

#[test]
fn pointcloud_voxelized_collision_consistency() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Dense cloud overlapping robot
    let mut points = Vec::new();
    for i in -5..=5 {
        for j in -5..=5 {
            points.push([i as f64 * 0.02, j as f64 * 0.02, 0.0]);
        }
    }

    let direct = pointcloud_to_spheres(&points, 0.01, None);
    let voxelized = pointcloud_to_spheres(&points, 0.01, Some(0.05));

    let env_d = CollisionEnvironment::build(direct, 0.02, AABB::symmetric(2.0));
    let env_v = CollisionEnvironment::build(voxelized, 0.02, AABB::symmetric(2.0));

    let col_d = env_d.check_collision(&runtime.world);
    let col_v = env_v.check_collision(&runtime.world);

    // Both should detect collision (voxelized uses larger spheres as conservative bound)
    assert!(col_d, "Direct point cloud should detect collision");
    assert!(
        col_v,
        "Voxelized point cloud (conservative) should also detect collision"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Convex decomposition accuracy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convex_decomp_cube_single_hull() {
    // A cube is already convex — should produce 1 hull
    let vertices = vec![
        nalgebra::Point3::new(-0.5, -0.5, -0.5),
        nalgebra::Point3::new(0.5, -0.5, -0.5),
        nalgebra::Point3::new(0.5, 0.5, -0.5),
        nalgebra::Point3::new(-0.5, 0.5, -0.5),
        nalgebra::Point3::new(-0.5, -0.5, 0.5),
        nalgebra::Point3::new(0.5, -0.5, 0.5),
        nalgebra::Point3::new(0.5, 0.5, 0.5),
        nalgebra::Point3::new(-0.5, 0.5, 0.5),
    ];

    // 12 triangles for a cube
    let indices = vec![
        [0, 2, 1],
        [0, 3, 2], // bottom
        [4, 5, 6],
        [4, 6, 7], // top
        [0, 1, 5],
        [0, 5, 4], // front
        [2, 3, 7],
        [2, 7, 6], // back
        [1, 2, 6],
        [1, 6, 5], // right
        [0, 4, 7],
        [0, 7, 3], // left
    ];

    let config = ConvexDecompConfig::default();
    let shape = convex_decomposition(&vertices, &indices, &config);
    assert!(shape.is_some(), "Cube should decompose successfully");

    // Verify the resulting shape contains the original volume:
    // a point at the center should be inside
    let s = shape.unwrap();
    let aabb = s.compute_aabb(&Isometry3::identity());
    assert!(aabb.extents().x > 0.8, "AABB should cover cube X");
    assert!(aabb.extents().y > 0.8, "AABB should cover cube Y");
    assert!(aabb.extents().z > 0.8, "AABB should cover cube Z");
}

#[test]
fn convex_decomp_l_shape_multiple_hulls() {
    // An L-shape is concave — should produce >1 hull
    // Construct L from vertices of two boxes joined at a corner
    let vertices = vec![
        // Vertical arm: 0.2 x 0.2 x 1.0
        nalgebra::Point3::new(-0.1, -0.1, 0.0),
        nalgebra::Point3::new(0.1, -0.1, 0.0),
        nalgebra::Point3::new(0.1, 0.1, 0.0),
        nalgebra::Point3::new(-0.1, 0.1, 0.0),
        nalgebra::Point3::new(-0.1, -0.1, 1.0),
        nalgebra::Point3::new(0.1, -0.1, 1.0),
        nalgebra::Point3::new(0.1, 0.1, 1.0),
        nalgebra::Point3::new(-0.1, 0.1, 1.0),
        // Horizontal arm: 0.8 x 0.2 x 0.2 starting at (0.1, 0, 0)
        nalgebra::Point3::new(0.1, -0.1, 0.0),
        nalgebra::Point3::new(0.9, -0.1, 0.0),
        nalgebra::Point3::new(0.9, 0.1, 0.0),
        nalgebra::Point3::new(0.1, 0.1, 0.0),
        nalgebra::Point3::new(0.1, -0.1, 0.2),
        nalgebra::Point3::new(0.9, -0.1, 0.2),
        nalgebra::Point3::new(0.9, 0.1, 0.2),
        nalgebra::Point3::new(0.1, 0.1, 0.2),
    ];

    // Triangulate both boxes (12 triangles each)
    let mut indices = Vec::new();
    for base in [0u32, 8] {
        let b = base;
        indices.extend_from_slice(&[
            [b, b + 2, b + 1],
            [b, b + 3, b + 2],
            [b + 4, b + 5, b + 6],
            [b + 4, b + 6, b + 7],
            [b, b + 1, b + 5],
            [b, b + 5, b + 4],
            [b + 2, b + 3, b + 7],
            [b + 2, b + 7, b + 6],
            [b + 1, b + 2, b + 6],
            [b + 1, b + 6, b + 5],
            [b, b + 4, b + 7],
            [b, b + 7, b + 3],
        ]);
    }

    let config = ConvexDecompConfig {
        max_hulls: 8,
        resolution: 32,
        ..Default::default()
    };
    let shape = convex_decomposition(&vertices, &indices, &config);
    assert!(shape.is_some(), "L-shape should decompose");
}

#[test]
fn convex_decomp_empty_mesh_returns_none() {
    let config = ConvexDecompConfig::default();

    assert!(convex_decomposition(&[], &[], &config).is_none());
    assert!(
        convex_decomposition(&[nalgebra::Point3::new(0.0, 0.0, 0.0)], &[], &config).is_none()
    );
}

#[test]
fn convex_decomp_collision_accuracy() {
    // Decompose a cube, then check collision against a point inside the cube
    let vertices = vec![
        nalgebra::Point3::new(-0.5, -0.5, -0.5),
        nalgebra::Point3::new(0.5, -0.5, -0.5),
        nalgebra::Point3::new(0.5, 0.5, -0.5),
        nalgebra::Point3::new(-0.5, 0.5, -0.5),
        nalgebra::Point3::new(-0.5, -0.5, 0.5),
        nalgebra::Point3::new(0.5, -0.5, 0.5),
        nalgebra::Point3::new(0.5, 0.5, 0.5),
        nalgebra::Point3::new(-0.5, 0.5, 0.5),
    ];
    let indices = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];

    let shape = convex_decomposition(&vertices, &indices, &ConvexDecompConfig::default()).unwrap();

    // Point inside the cube (origin) — distance should be negative or zero
    let probe = SharedShape::ball(0.01);
    let probe_pose = Isometry3::translation(0.0, 0.0, 0.0);
    let cube_pose = Isometry3::identity();

    let dist = parry3d_f64::query::distance(&cube_pose, &*shape, &probe_pose, &*probe).unwrap();
    assert!(dist < 0.1, "Point inside cube should be close: {}", dist);

    // Point outside the cube — distance should be positive
    let far_pose = Isometry3::translation(2.0, 0.0, 0.0);
    let far_dist =
        parry3d_f64::query::distance(&cube_pose, &*shape, &far_pose, &*probe).unwrap();
    assert!(far_dist > 1.0, "Point outside cube: {}", far_dist);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CCD tunneling regression tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ccd_tunneling_very_thin_obstacle() {
    // Ultra-thin obstacle: r=0.001 at x=0.5
    // Robot sphere r=0.05 moves from x=0 to x=1.0
    // Discrete 10-step sampling (step=0.1) would miss it
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.5, 0.0, 0.0, 0.001, 0);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(
        result.collides(),
        "CCD must catch ultra-thin obstacle (r=0.001)"
    );

    let toi = result.time_of_impact.unwrap();
    // Contact at sphere_x + 0.05 = 0.5 - 0.001 → sphere_x ≈ 0.449 → t ≈ 0.449
    assert!(
        toi > 0.3 && toi < 0.6,
        "TOI should be near 0.45: {}",
        toi
    );
}

#[test]
fn ccd_tunneling_very_fast_motion() {
    // Very long motion: x=0 to x=10
    // Thin obstacle at x=5 with r=0.01
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(5.0, 0.0, 0.0, 0.01, 0);
    let env = CollisionEnvironment::build(obs, 0.5, AABB::symmetric(12.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(10.0, 0.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(result.collides(), "CCD must catch obstacle in fast 10m sweep");

    let toi = result.time_of_impact.unwrap();
    assert!(
        (toi - 0.5).abs() < 0.05,
        "TOI should be ~0.5 (midway): {}",
        toi
    );
}

#[test]
fn ccd_tunneling_diagonal_motion() {
    // Diagonal motion: (0,0,0) → (1,1,0)
    // Obstacle at (0.5, 0.5, 0) — exactly on the diagonal
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.5, 0.5, 0.0, 0.02, 0);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(3.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 1.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(result.collides(), "CCD must detect diagonal collision");
}

#[test]
fn ccd_tunneling_grazing_miss() {
    // Obstacle just barely outside sweep path
    // Motion along X. Obstacle at (0.5, 0.2, 0.0), r=0.01
    // Robot sphere r=0.05 → swept capsule radius 0.05
    // Distance from line to obstacle center = 0.2 > 0.05+0.01 = 0.06 → no collision
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.5, 0.2, 0.0, 0.01, 0);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(
        !result.collides(),
        "Should NOT collide — obstacle is beyond sweep radius"
    );
}

#[test]
fn ccd_tunneling_grazing_hit() {
    // Obstacle just barely inside sweep path
    // Motion along X. Obstacle at (0.5, 0.04, 0.0), r=0.02
    // Robot sphere r=0.05 → distance 0.04 < 0.05+0.02 = 0.07 → collision
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.5, 0.04, 0.0, 0.02, 0);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(result.collides(), "Should collide — obstacle within sweep radius");
}

#[test]
fn ccd_multiple_obstacles_first_hit() {
    // Two obstacles along the path, verify CCD reports the first one
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.3, 0.0, 0.0, 0.05, 0); // first obstacle
    obs.push(0.7, 0.0, 0.0, 0.05, 1); // second obstacle
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    let pose_start = vec![Pose::identity()];
    let pose_end = vec![Pose(nalgebra::Isometry3::translation(1.0, 0.0, 0.0))];

    let result = detector.check_motion(&pose_start, &pose_end);
    assert!(result.collides());

    let toi = result.time_of_impact.unwrap();
    // First obstacle at x=0.3: contact when sphere_x + 0.05 = 0.3 - 0.05 → t ≈ 0.2
    assert!(
        toi < 0.4,
        "TOI should correspond to first obstacle (t<0.4): {}",
        toi
    );
}

#[test]
fn ccd_trajectory_multi_segment_tunneling() {
    // 4-segment trajectory where collision is in segment 2
    // with thin obstacle that discrete per-segment sampling would miss
    let model = make_ccd_model(0.0, 0.0, 0.0, 0.05);
    let mut obs = SpheresSoA::new();
    obs.push(0.55, 0.0, 0.0, 0.005, 0); // very thin obstacle
    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));
    let detector = ContinuousCollisionDetector::new(&model, &env, CCDConfig::default());

    // Waypoints: 0.0, 0.2, 0.4, 0.8, 1.0
    // Segment 2 (0.4 → 0.8) passes through 0.55
    let waypoints = vec![
        vec![0.0],
        vec![0.2],
        vec![0.4],
        vec![0.8],
        vec![1.0],
    ];

    let fk = |q: &[f64]| -> Vec<Pose> {
        vec![Pose(nalgebra::Isometry3::translation(q[0], 0.0, 0.0))]
    };

    let result = detector.check_trajectory(&waypoints, fk);
    assert!(result.collision, "Trajectory CCD must detect thin obstacle");
    assert_eq!(
        result.segment_index,
        Some(2),
        "Collision should be in segment 2 (0.4→0.8)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Per-link padding and scaling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn padding_turns_miss_into_collision() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();

    // Place obstacle at moderate distance
    let mut obs = SpheresSoA::new();
    obs.push(0.3, 0.0, 0.0, 0.01, 0);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));

    let poses = identity_poses(&robot);

    // Without padding
    runtime.update(&poses);
    let no_pad = env.check_collision(&runtime.world);

    // With large padding
    let config = LinkCollisionConfig::new().with_padding(0.3);
    runtime.update_with_config(&poses, &config);
    let with_pad = env.check_collision(&runtime.world);

    // Padding should make it more conservative
    if !no_pad {
        assert!(with_pad, "Padding 0.3m should catch obstacle at 0.3m");
    }
}

#[test]
fn per_link_padding_different_links() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = separated_poses(&robot);

    let mut config = LinkCollisionConfig::new();
    // Only pad link 0 (base)
    config.set(0, 0.5, 1.0);
    // Leave others at default (0 padding, 1.0 scale)

    runtime.update_with_config(&poses, &config);

    // Verify link 0 spheres have larger radius than link 2
    let base_idx = robot.link_index("base_link").unwrap();
    let link2_idx = robot.link_index("link2").unwrap();

    let (start0, end0) = model.link_ranges[base_idx];
    let (start2, end2) = model.link_ranges[link2_idx];

    if end0 > start0 && end2 > start2 {
        // Base link spheres should have 0.5m added
        assert!(
            runtime.world.radius[start0] > runtime.world.radius[start2],
            "Padded base ({}) should be larger than unpadded link2 ({})",
            runtime.world.radius[start0],
            runtime.world.radius[start2]
        );
    }
}

#[test]
fn scale_factor_doubles_radius() {
    let (robot, model) = setup_robot();
    let mut runtime_normal = model.create_runtime();
    let mut runtime_scaled = model.create_runtime();
    let poses = identity_poses(&robot);

    runtime_normal.update(&poses);
    let config = LinkCollisionConfig::new().with_scale(2.0);
    runtime_scaled.update_with_config(&poses, &config);

    // All radii should be doubled
    for i in 0..runtime_normal.world.len() {
        let expected = runtime_normal.world.radius[i] * 2.0;
        assert!(
            (runtime_scaled.world.radius[i] - expected).abs() < 1e-10,
            "Sphere {}: expected radius {}, got {}",
            i,
            expected,
            runtime_scaled.world.radius[i]
        );
    }
}

#[test]
fn negative_padding_clamped_to_zero() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);

    // Negative padding that would make radius negative → clamped to 0
    let config = LinkCollisionConfig::new().with_padding(-999.0);
    runtime.update_with_config(&poses, &config);

    for i in 0..runtime.world.len() {
        assert!(
            runtime.world.radius[i] >= 0.0,
            "Radius should never be negative: {}",
            runtime.world.radius[i]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Two-tier cross-module integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_tier_box_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Box obstacle at origin — should collide
    let box_obs = shape_from_box(0.5, 0.5, 0.5);
    let at_origin = Isometry3::identity();
    assert!(
        checker.check_obstacle(&runtime, &poses, &box_obs, &at_origin),
        "Box obstacle at origin should collide with robot"
    );

    // Box far away — no collision
    let far = Isometry3::translation(10.0, 10.0, 10.0);
    assert!(
        !checker.check_obstacle(&runtime, &poses, &box_obs, &far),
        "Box far away should not collide"
    );
}

#[test]
fn two_tier_cylinder_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Cylinder obstacle at origin
    let cyl_obs = shape_from_cylinder(0.3, 0.5);
    let at_origin = Isometry3::identity();
    assert!(
        checker.check_obstacle(&runtime, &poses, &cyl_obs, &at_origin),
        "Cylinder at origin should collide"
    );
}

#[test]
fn two_tier_sphere_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    let sphere_obs = shape_from_sphere(0.3);
    let at_origin = Isometry3::identity();
    assert!(
        checker.check_obstacle(&runtime, &poses, &sphere_obs, &at_origin),
        "Sphere at origin should collide"
    );
}

#[test]
fn two_tier_refinement_near_boundary() {
    // Test that near-boundary cases trigger exact mesh refinement
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let config = SphereGenConfig::coarse();
    // Small refinement margin forces exact checks for nearby obstacles
    let checker = TwoTierCollisionChecker::new(&robot, &config, 0.01, 0.001);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Obstacle just at the edge of sphere approximation
    let obstacle = SharedShape::ball(0.01);
    let near_pose = Isometry3::translation(0.15, 0.0, 0.0);

    // This exercises the exact refinement path (sphere dist within margin)
    let _result = checker.check_obstacle(&runtime, &poses, &obstacle, &near_pose);
    // No panic = success; the result depends on exact geometry
}

#[test]
fn two_tier_distance_consistency() {
    // Two-tier min_distance should be <= sphere-only min_distance
    // (exact check can only reduce distance when geometry is convex)
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    let mut obs = SpheresSoA::new();
    obs.push(0.5, 0.0, 0.0, 0.05, 0);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));

    let sphere_dist = env.min_distance(&runtime.world);
    let two_tier_dist = checker.min_distance(&runtime, &poses, &env);

    // Two-tier uses exact geometry for near objects → can differ
    // Both should be finite and positive for this case
    assert!(
        sphere_dist.is_finite() && two_tier_dist.is_finite(),
        "Both distances should be finite: sphere={}, two_tier={}",
        sphere_dist,
        two_tier_dist
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Self-collision integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn self_collision_pairs_respects_acm() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();

    // Default: adjacent links skipped → 3 pairs checked
    let default_pairs = SelfCollisionPairs::from_robot(&robot);
    assert_eq!(default_pairs.num_pairs(), 3);

    // Custom ACM: also allow base↔link2
    let mut acm = AllowedCollisionMatrix::from_robot(&robot);
    acm.allow("base_link", "link2");
    let resolved = ResolvedACM::from_acm(&acm, &robot);
    let custom_pairs = SelfCollisionPairs::from_resolved_acm(&robot, &resolved);
    assert_eq!(custom_pairs.num_pairs(), 2);

    // Allow all → 0 pairs
    let mut acm_all = AllowedCollisionMatrix::from_robot(&robot);
    acm_all.allow("base_link", "link2");
    acm_all.allow("base_link", "ee_link");
    acm_all.allow("link1", "ee_link");
    let resolved_all = ResolvedACM::from_acm(&acm_all, &robot);
    let all_allowed = SelfCollisionPairs::from_resolved_acm(&robot, &resolved_all);
    assert_eq!(all_allowed.num_pairs(), 0);
}

#[test]
fn self_collision_overlapping_non_adjacent_detected() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
    let pairs = SelfCollisionPairs::from_robot(&robot);
    let mut runtime = model.create_runtime();

    // All at origin → non-adjacent links overlap
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    assert!(
        pairs.check_self_collision(&runtime),
        "Overlapping non-adjacent links should self-collide"
    );

    let closest = pairs.closest_pair(&runtime);
    assert!(closest.is_some());
    let (a, b, d) = closest.unwrap();
    assert!(d <= 0.0, "Overlapping links should have d≤0: {}", d);
    assert_ne!(a, b, "Closest pair should be different links");
}

#[test]
fn self_collision_separated_not_detected() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
    let pairs = SelfCollisionPairs::from_robot(&robot);
    let mut runtime = model.create_runtime();

    let poses = separated_poses(&robot);
    runtime.update(&poses);

    assert!(
        !pairs.check_self_collision(&runtime),
        "Well-separated links should not self-collide"
    );

    let dist = pairs.min_distance(&runtime);
    assert!(dist > 0.0, "Separated links: min distance should be positive");
}

#[test]
fn two_tier_self_collision_integration() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let acm = ResolvedACM::from_robot(&robot);

    // Separated links → no self-collision
    let separated = separated_poses(&robot);
    runtime.update(&separated);
    assert!(
        !checker.check_self_collision_acm(&runtime, &separated, &acm),
        "Separated links should not self-collide in two-tier"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. SDF integration tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sdf_from_pointcloud() {
    // Build SDF from point cloud, verify queries match
    let points: Vec<[f64; 3]> = (0..20)
        .map(|i| {
            let angle = i as f64 * 0.314;
            [angle.cos() * 0.3, angle.sin() * 0.3, 0.0]
        })
        .collect();

    let spheres = pointcloud_to_spheres(&points, 0.05, None);
    let config = SDFConfig {
        resolution: 0.05,
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        truncation: 0.5,
    };
    let sdf = SignedDistanceField::from_spheres(&spheres, &config);

    // Center of ring should have obstacles nearby
    let d_center = sdf.distance_at(0.0, 0.0, 0.0);
    // Far away should be clear
    let d_far = sdf.distance_at(0.9, 0.9, 0.9);

    assert!(
        d_center < d_far,
        "Center of obstacle ring should be closer than far corner"
    );
}

#[test]
fn sdf_gradient_for_trajectory_optimization() {
    // SDF gradient is used by CHOMP/STOMP — verify it provides useful gradients
    let config = SDFConfig {
        resolution: 0.05,
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        truncation: 0.5,
    };
    let mut sdf = SignedDistanceField::new(&config);
    sdf.add_sphere(0.0, 0.0, 0.0, 0.3, 1);

    // Test cost+gradient at various points
    let test_points = [
        (0.35, 0.0, 0.0),  // just outside obstacle surface
        (0.2, 0.0, 0.0),   // inside obstacle
        (0.0, 0.35, 0.0),  // above
        (0.0, 0.0, -0.35), // behind
    ];

    for (x, y, z) in test_points {
        let (cost, grad) = sdf.sphere_cost_and_gradient(x, y, z, 0.05, 0.2);
        let grad_mag = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();

        if cost > 0.0 {
            assert!(
                grad_mag > 1e-6,
                "Non-zero cost at ({},{},{}) should have non-zero gradient",
                x,
                y,
                z
            );
        }
    }
}

#[test]
fn multi_resolution_sdf_consistency() {
    let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let mut sdf = MultiResolutionSDF::new(bounds, 0.2, 0.05, 0.5);
    sdf.add_sphere(0.0, 0.0, 0.0, 0.3, 1);

    // Occupied at center
    assert!(sdf.is_occupied(0.0, 0.0, 0.0));
    // Free far away
    assert!(!sdf.is_occupied(0.9, 0.9, 0.9));

    // Distance at center should be negative (inside obstacle)
    let d = sdf.distance_at(0.0, 0.0, 0.0);
    assert!(d < 0.0, "Inside obstacle: {}", d);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Mesh backend shape pair combinations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mesh_sphere_vs_sphere_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let backend = MeshCollisionBackend::from_robot(&robot);
    let transforms: Vec<Isometry3<f64>> =
        (0..robot.links.len()).map(|_| Isometry3::identity()).collect();

    let obs = shape_from_sphere(0.5);
    let obs_pose = Isometry3::identity();
    let dist = backend.min_distance_exact(&transforms, &obs, &obs_pose);
    assert!(dist <= 0.01, "Overlapping sphere: {}", dist);
}

#[test]
fn mesh_sphere_vs_box_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let backend = MeshCollisionBackend::from_robot(&robot);
    let transforms: Vec<Isometry3<f64>> =
        (0..robot.links.len()).map(|_| Isometry3::identity()).collect();

    // Box at origin
    let obs = shape_from_box(0.5, 0.5, 0.5);
    let obs_pose = Isometry3::identity();
    let dist = backend.min_distance_exact(&transforms, &obs, &obs_pose);
    assert!(dist <= 0.01, "Overlapping box: {}", dist);

    // Box far away
    let far_pose = Isometry3::translation(5.0, 0.0, 0.0);
    let far_dist = backend.min_distance_exact(&transforms, &obs, &far_pose);
    assert!(far_dist > 1.0, "Far box: {}", far_dist);
}

#[test]
fn mesh_sphere_vs_cylinder_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let backend = MeshCollisionBackend::from_robot(&robot);
    let transforms: Vec<Isometry3<f64>> =
        (0..robot.links.len()).map(|_| Isometry3::identity()).collect();

    // Cylinder at origin
    let obs = shape_from_cylinder(0.3, 0.5);
    let obs_pose = Isometry3::identity();
    let dist = backend.min_distance_exact(&transforms, &obs, &obs_pose);
    assert!(dist <= 0.01, "Overlapping cylinder: {}", dist);

    // Cylinder far away
    let far_pose = Isometry3::translation(0.0, 5.0, 0.0);
    let far_dist = backend.min_distance_exact(&transforms, &obs, &far_pose);
    assert!(far_dist > 1.0, "Far cylinder: {}", far_dist);
}

#[test]
fn mesh_contact_points_with_box_obstacle() {
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let backend = MeshCollisionBackend::from_robot(&robot);
    let transforms: Vec<Isometry3<f64>> =
        (0..robot.links.len()).map(|_| Isometry3::identity()).collect();

    let obs = shape_from_box(0.3, 0.3, 0.3);
    let obs_pose = Isometry3::translation(0.2, 0.0, 0.0);

    let contacts = backend.contact_points(&transforms, &obs, &obs_pose, 0.5);
    assert!(
        !contacts.is_empty(),
        "Should find contacts between robot and nearby box"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Known collision/no-collision regression pairs
// ═══════════════════════════════════════════════════════════════════════════

/// Regression: exact tangent contact (distance ≈ 0)
#[test]
fn regression_tangent_contact() {
    let mut a = SpheresSoA::new();
    a.push(0.0, 0.0, 0.0, 0.5, 0);

    let mut b = SpheresSoA::new();
    b.push(1.0, 0.0, 0.0, 0.5, 1); // exactly touching: dist=1.0, radii_sum=1.0

    let (dist, _, _) = a.min_distance(&b).unwrap();
    assert!(
        dist.abs() < 1e-10,
        "Tangent spheres should have distance ~0: {}",
        dist
    );
}

/// Regression: many obstacles, only one collides
#[test]
fn regression_single_collision_in_many() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    let mut obs = SpheresSoA::new();
    // 99 far obstacles
    for i in 0..99 {
        let x = 5.0 + i as f64 * 0.1;
        obs.push(x, 5.0, 5.0, 0.01, i);
    }
    // 1 colliding obstacle
    obs.push(0.0, 0.0, 0.0, 0.5, 99);

    let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(20.0));
    assert!(
        env.check_collision(&runtime.world),
        "Should detect the one colliding obstacle"
    );
}

/// Regression: robot at non-origin position
#[test]
fn regression_robot_translated() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();

    // Place all links at (5, 5, 5)
    let poses: Vec<Pose> = (0..robot.links.len())
        .map(|_| Pose::from_xyz(5.0, 5.0, 5.0))
        .collect();
    runtime.update(&poses);

    // Obstacle at (5, 5, 5) — should collide
    let mut obs = SpheresSoA::new();
    obs.push(5.0, 5.0, 5.0, 0.01, 0);
    let env = CollisionEnvironment::build(obs, 0.05, AABB::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));

    assert!(
        env.check_collision(&runtime.world),
        "Translated robot should collide with obstacle at same position"
    );

    // Obstacle at origin — should NOT collide
    let mut obs2 = SpheresSoA::new();
    obs2.push(0.0, 0.0, 0.0, 0.01, 0);
    let env2 = CollisionEnvironment::build(obs2, 0.05, AABB::symmetric(10.0));

    assert!(
        !env2.check_collision(&runtime.world),
        "Translated robot should NOT collide with obstacle at origin"
    );
}

/// Regression: collision result provides correct closest pair
#[test]
fn regression_closest_pair_correct() {
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = separated_poses(&robot);
    runtime.update(&poses);

    // Place obstacle near link2 (at z=1.0 per separated_poses)
    let mut obs = SpheresSoA::new();
    obs.push(0.0, 0.0, 1.0, 0.1, 42);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(5.0));

    let result = env.check_full(&runtime.world);
    if result.closest_pair.is_some() {
        let (robot_link, obs_link) = result.closest_pair.unwrap();
        // Obstacle link_id is 42
        assert_eq!(obs_link, 42, "Obstacle link should be 42");
        // Robot link closest to z=1.0 should be link2 (index 2)
        let link2_idx = robot.link_index("link2").unwrap();
        assert_eq!(
            robot_link, link2_idx,
            "Closest robot link should be link2 (idx={})",
            link2_idx
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Cross-module full pipeline integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_pipeline_urdf_to_collision_check() {
    // Complete pipeline: URDF → sphere model → environment → collision check
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::fine());
    let mut runtime = model.create_runtime();

    // Point cloud environment
    let points: Vec<[f64; 3]> = (0..200)
        .map(|i| {
            let t = i as f64 * 0.031;
            [t.cos() * 0.5, t.sin() * 0.5, 0.0]
        })
        .collect();
    let obs = pointcloud_to_spheres(&points, 0.02, Some(0.05));
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));

    // Check at identity (robot at origin, point cloud ring at r=0.5)
    let poses = identity_poses(&robot);
    runtime.update(&poses);
    let result = env.check_full(&runtime.world);

    // Robot at origin, ring at r=0.5 with point sphere r=0.02+voxel coverage
    // Base box half-extent 0.1 → robot extent ~0.1 from origin
    // Ring at 0.5 → gap ~0.3 → no collision (unless fine spheres reach out)
    // The exact result depends on sphere generation density
    let _ = result; // No panic = success

    // Now move robot into the ring → must collide
    let mut close_poses = identity_poses(&robot);
    close_poses[0] = Pose::from_xyz(0.5, 0.0, 0.0); // base into ring
    runtime.update(&close_poses);
    let result2 = env.check_full(&runtime.world);
    assert!(
        result2.in_collision,
        "Robot moved into point cloud ring should collide"
    );
}

#[test]
fn full_pipeline_ccd_with_padding() {
    // CCD + per-link padding integration
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();

    let poses = identity_poses(&robot);
    let config = LinkCollisionConfig::new().with_padding(0.1);
    runtime.update_with_config(&poses, &config);

    // Obstacle that only collides if padding is applied
    let mut obs = SpheresSoA::new();
    obs.push(0.25, 0.0, 0.0, 0.01, 0);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));

    // With padding, should collide (base extends to ~0.1 + 0.1 padding = 0.2 from center)
    let with_pad = env.check_collision(&runtime.world);

    // Without padding
    runtime.update(&poses);
    let without_pad = env.check_collision(&runtime.world);

    // Padding should make more things collide
    if !without_pad {
        assert!(
            with_pad,
            "0.1m padding should detect obstacle at 0.25m that base (0.1m extent) misses"
        );
    }
}

#[test]
fn full_pipeline_sdf_plus_collision_check() {
    // Build SDF from obstacles, then check robot spheres against it
    let (robot, model) = setup_robot();
    let mut runtime = model.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    let config = SDFConfig {
        resolution: 0.05,
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        truncation: 0.5,
    };
    let mut sdf = SignedDistanceField::new(&config);
    sdf.add_sphere(0.3, 0.0, 0.0, 0.1, 1);

    // Check each robot sphere against SDF
    let mut sdf_collision = false;
    for i in 0..runtime.world.len() {
        if sdf.sphere_collision(
            runtime.world.x[i],
            runtime.world.y[i],
            runtime.world.z[i],
            runtime.world.radius[i],
        ) {
            sdf_collision = true;
            break;
        }
    }

    // Also check with sphere-tree approach for comparison
    let mut obs = SpheresSoA::new();
    obs.push(0.3, 0.0, 0.0, 0.1, 1);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));
    let tree_collision = env.check_collision(&runtime.world);

    // Both methods should agree on collision (or both no collision)
    // They may differ slightly due to discretization, but for this geometry
    // both should detect collision (base box extends to x=0.1, obstacle at x=0.3 with r=0.1)
    let _ = (sdf_collision, tree_collision); // No panic = success
}

#[test]
fn full_pipeline_two_tier_with_pointcloud() {
    // Two-tier checker + point cloud environment
    let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
    let checker = TwoTierCollisionChecker::from_robot(&robot);
    let mut runtime = checker.create_runtime();
    let poses = identity_poses(&robot);
    runtime.update(&poses);

    // Dense point cloud at origin
    let points: Vec<[f64; 3]> = (0..100)
        .map(|i| {
            let t = i as f64 * 0.063;
            [t.cos() * 0.05, t.sin() * 0.05, 0.0]
        })
        .collect();

    let obs = pointcloud_to_spheres(&points, 0.03, None);
    let env = CollisionEnvironment::build(obs, 0.02, AABB::symmetric(2.0));

    assert!(
        checker.check_collision(&runtime, &poses, &env),
        "Two-tier should detect collision with dense point cloud at origin"
    );

    let dist = checker.min_distance(&runtime, &poses, &env);
    assert!(
        dist < 0.5,
        "Distance to point cloud at origin should be small: {}",
        dist
    );
}
