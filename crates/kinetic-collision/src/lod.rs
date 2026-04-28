//! Collision shape Level of Detail (LOD) system.
//!
//! Provides three tiers of collision fidelity per robot link:
//!
//! 1. **Sphere** (~500ns): SIMD-vectorized sphere-tree. Used when obstacle
//!    distance is clearly beyond any collision risk.
//!
//! 2. **ConvexHull** (~5-20µs): Per-link convex hull shapes via parry3d-f64.
//!    Used for moderate-distance queries where spheres are too conservative
//!    but full mesh is unnecessary. All primitives (box, cylinder, sphere)
//!    are already convex, so the convex hull IS the exact shape for them.
//!    For meshes, a convex hull is computed from the vertices.
//!
//! 3. **Mesh** (~50µs): Full parry3d-f64 exact shapes (TriMesh for mesh files).
//!    Used only when very close to obstacles and maximum precision is needed.
//!
//! # Planning Phase Support
//!
//! Different planning phases have different accuracy needs:
//!
//! - **Exploration** (RRT/PRM sampling): Force `Sphere` for maximum speed.
//! - **Refinement** (trajectory optimization): Auto LOD with convex hull fallback.
//! - **Validation** (final check): Force `Mesh` for exact geometry.
//!
//! # Conservative Guarantee
//!
//! When transitioning from a coarser LOD to a finer one, there are no false
//! negatives: sphere approximations are always conservative (inflate geometry),
//! so if spheres say "no collision", there truly is no collision. Conversely,
//! if spheres say "collision", the finer LOD may reveal it was a false positive.

use nalgebra::Isometry3;
use parry3d_f64::shape::SharedShape;

use kinetic_core::Pose;
use kinetic_robot::Robot;

use crate::check::CollisionEnvironment;
use crate::mesh::{poses_to_isometries, ContactPoint, MeshCollisionBackend};
use crate::sphere_model::{RobotSphereModel, RobotSpheres, SphereGenConfig};

// Re-export the convex-hull backend so external paths stay stable
// (`kinetic_collision::lod::ConvexCollisionBackend`).
pub use crate::convex_backend::ConvexCollisionBackend;

/// Collision fidelity level, ordered from cheapest to most precise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollisionLOD {
    /// SIMD sphere-tree (~500ns). Conservative bounding volumes.
    Sphere,
    /// Per-link convex hulls (~5-20µs). Exact for primitives, convex hull for meshes.
    ConvexHull,
    /// Full mesh geometry (~50µs). TriMesh for mesh files, exact primitives.
    Mesh,
}

impl std::fmt::Display for CollisionLOD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollisionLOD::Sphere => write!(f, "Sphere"),
            CollisionLOD::ConvexHull => write!(f, "ConvexHull"),
            CollisionLOD::Mesh => write!(f, "Mesh"),
        }
    }
}

/// Configuration for LOD distance thresholds.
///
/// When the sphere-based distance to nearest obstacle is:
/// - `> convex_threshold`: use Sphere (fast path)
/// - `convex_threshold .. mesh_threshold`: use ConvexHull (medium path)
/// - `< mesh_threshold`: use Mesh (exact path)
///
/// Thresholds include the safety margin: `effective_threshold = threshold + safety_margin`.
#[derive(Debug, Clone)]
pub struct LODConfig {
    /// Distance below which ConvexHull is used instead of Sphere.
    /// Default: 0.10m (10cm).
    pub convex_threshold: f64,
    /// Distance below which Mesh is used instead of ConvexHull.
    /// Default: 0.02m (2cm).
    pub mesh_threshold: f64,
    /// Minimum clearance required. Added to all distance checks.
    /// Default: 0.02m (2cm).
    pub safety_margin: f64,
}

impl Default for LODConfig {
    fn default() -> Self {
        Self {
            convex_threshold: 0.10,
            mesh_threshold: 0.02,
            safety_margin: 0.02,
        }
    }
}

impl LODConfig {
    /// Config for exploration phase: sphere-only for maximum speed.
    pub fn exploration() -> Self {
        Self {
            convex_threshold: 0.0,
            mesh_threshold: 0.0,
            safety_margin: 0.02,
        }
    }

    /// Config for refinement phase: auto LOD with convex hull.
    pub fn refinement() -> Self {
        Self::default()
    }

    /// Config for validation phase: always use mesh for exact checks.
    pub fn validation() -> Self {
        Self {
            convex_threshold: f64::INFINITY,
            mesh_threshold: f64::INFINITY,
            safety_margin: 0.0,
        }
    }

    /// Select LOD based on sphere-distance to nearest obstacle.
    ///
    /// Thresholds of 0.0 disable that LOD tier (useful for forced modes).
    pub fn select_lod(&self, sphere_distance: f64) -> CollisionLOD {
        if self.mesh_threshold > 0.0
            && sphere_distance < self.mesh_threshold + self.safety_margin
        {
            CollisionLOD::Mesh
        } else if self.convex_threshold > 0.0
            && sphere_distance < self.convex_threshold + self.safety_margin
        {
            CollisionLOD::ConvexHull
        } else {
            CollisionLOD::Sphere
        }
    }

    /// Force a specific LOD level regardless of distance.
    pub fn forced(lod: CollisionLOD) -> Self {
        match lod {
            CollisionLOD::Sphere => Self::exploration(),
            CollisionLOD::ConvexHull => Self {
                convex_threshold: f64::INFINITY,
                mesh_threshold: 0.0,
                safety_margin: 0.02,
            },
            CollisionLOD::Mesh => Self::validation(),
        }
    }
}

/// Three-tier LOD collision checker.
///
/// Combines sphere, convex hull, and mesh backends with automatic LOD
/// selection based on distance to nearest obstacle.
///
/// # Usage
///
/// ```ignore
/// let checker = LODCollisionChecker::from_robot(&robot);
///
/// // Auto LOD (default thresholds)
/// checker.check_collision(&runtime, &poses, &env);
///
/// // Force specific phase
/// checker.set_config(LODConfig::exploration()); // sphere only
/// checker.set_config(LODConfig::validation());  // mesh only
/// ```
#[derive(Debug, Clone)]
pub struct LODCollisionChecker {
    /// SIMD sphere model for broadphase (always evaluated first).
    sphere_model: RobotSphereModel,
    /// Convex hull shapes per link for medium-fidelity checks.
    convex_backend: ConvexCollisionBackend,
    /// Full mesh shapes per link for exact checks.
    mesh_backend: MeshCollisionBackend,
    /// LOD distance thresholds and safety margin.
    pub config: LODConfig,
}

impl LODCollisionChecker {
    /// Build an LOD checker from a robot model.
    pub fn new(robot: &Robot, sphere_config: &SphereGenConfig, config: LODConfig) -> Self {
        Self {
            sphere_model: RobotSphereModel::from_robot(robot, sphere_config),
            convex_backend: ConvexCollisionBackend::from_robot(robot),
            mesh_backend: MeshCollisionBackend::from_robot(robot),
            config,
        }
    }

    /// Build with default settings (coarse spheres, default LOD thresholds).
    pub fn from_robot(robot: &Robot) -> Self {
        Self::new(robot, &SphereGenConfig::coarse(), LODConfig::default())
    }

    /// Update the LOD configuration (e.g., when switching planning phases).
    pub fn set_config(&mut self, config: LODConfig) {
        self.config = config;
    }

    /// Access the sphere model.
    pub fn sphere_model(&self) -> &RobotSphereModel {
        &self.sphere_model
    }

    /// Access the convex backend.
    pub fn convex_backend(&self) -> &ConvexCollisionBackend {
        &self.convex_backend
    }

    /// Access the mesh backend.
    pub fn mesh_backend(&self) -> &MeshCollisionBackend {
        &self.mesh_backend
    }

    /// Create a runtime sphere set for world-frame queries.
    pub fn create_runtime(&self) -> RobotSpheres<'_> {
        self.sphere_model.create_runtime()
    }

    /// LOD collision check against environment.
    ///
    /// 1. Always runs sphere check first (fast path).
    /// 2. If sphere distance is within `convex_threshold`, refines with convex hull.
    /// 3. If convex distance is within `mesh_threshold`, refines with exact mesh.
    ///
    /// Returns `(collision: bool, lod_used: CollisionLOD)`.
    pub fn check_collision(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        environment: &CollisionEnvironment,
    ) -> (bool, CollisionLOD) {
        // Step 1: Sphere fast path (always)
        let sphere_dist = environment.min_distance(&runtime.world);

        if sphere_dist > self.config.convex_threshold + self.config.safety_margin {
            return (false, CollisionLOD::Sphere);
        }

        let selected = self.config.select_lod(sphere_dist);

        match selected {
            CollisionLOD::Sphere => {
                // Sphere says close but config says sphere-only
                let collides = sphere_dist < self.config.safety_margin;
                (collides, CollisionLOD::Sphere)
            }
            CollisionLOD::ConvexHull => {
                // Refine with convex hull
                let isometries = poses_to_isometries(link_poses);
                let obs = &environment.obstacle_spheres;
                let convex_dist = self.min_distance_at_lod(
                    CollisionLOD::ConvexHull,
                    &isometries,
                    obs,
                );
                (convex_dist < self.config.safety_margin, CollisionLOD::ConvexHull)
            }
            CollisionLOD::Mesh => {
                // Full exact check
                let isometries = poses_to_isometries(link_poses);
                let obs = &environment.obstacle_spheres;
                let mesh_dist =
                    self.min_distance_at_lod(CollisionLOD::Mesh, &isometries, obs);
                (mesh_dist < self.config.safety_margin, CollisionLOD::Mesh)
            }
        }
    }

    /// LOD collision check against a single obstacle shape.
    pub fn check_obstacle(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        obstacle: &SharedShape,
        obstacle_pose: &Isometry3<f64>,
    ) -> (bool, CollisionLOD) {
        // Sphere approximation of obstacle
        let aabb = obstacle.compute_aabb(obstacle_pose);
        let obs_center = aabb.center();
        let obs_radius = (aabb.extents() / 2.0).norm();

        let mut min_sphere_dist = f64::INFINITY;
        let w = &runtime.world;
        for i in 0..w.len() {
            let dx = w.x[i] - obs_center.x;
            let dy = w.y[i] - obs_center.y;
            let dz = w.z[i] - obs_center.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() - w.radius[i] - obs_radius;
            if dist < min_sphere_dist {
                min_sphere_dist = dist;
            }
        }

        if min_sphere_dist > self.config.convex_threshold + self.config.safety_margin {
            return (false, CollisionLOD::Sphere);
        }

        let selected = self.config.select_lod(min_sphere_dist);
        let isometries = poses_to_isometries(link_poses);

        let dist = match selected {
            CollisionLOD::Sphere => min_sphere_dist,
            CollisionLOD::ConvexHull => {
                self.convex_backend
                    .min_distance_exact(&isometries, obstacle, obstacle_pose)
            }
            CollisionLOD::Mesh => {
                self.mesh_backend
                    .min_distance_exact(&isometries, obstacle, obstacle_pose)
            }
        };

        (dist < self.config.safety_margin, selected)
    }

    /// Compute minimum distance using a specific LOD level.
    ///
    /// Useful when you want to force a particular fidelity level.
    pub fn min_distance_with_lod(
        &self,
        lod: CollisionLOD,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        environment: &CollisionEnvironment,
    ) -> f64 {
        match lod {
            CollisionLOD::Sphere => environment.min_distance(&runtime.world),
            CollisionLOD::ConvexHull | CollisionLOD::Mesh => {
                let isometries = poses_to_isometries(link_poses);
                self.min_distance_at_lod(lod, &isometries, &environment.obstacle_spheres)
            }
        }
    }

    /// Auto-select LOD and compute minimum distance.
    pub fn min_distance(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        environment: &CollisionEnvironment,
    ) -> (f64, CollisionLOD) {
        let sphere_dist = environment.min_distance(&runtime.world);

        if sphere_dist > self.config.convex_threshold + self.config.safety_margin {
            return (sphere_dist, CollisionLOD::Sphere);
        }

        let selected = self.config.select_lod(sphere_dist);
        let isometries = poses_to_isometries(link_poses);
        let obs = &environment.obstacle_spheres;

        let dist = match selected {
            CollisionLOD::Sphere => sphere_dist,
            _ => self.min_distance_at_lod(selected, &isometries, obs),
        };

        (dist, selected)
    }

    /// Get contact points using mesh backend (always exact).
    pub fn contact_points(
        &self,
        link_poses: &[Pose],
        obstacle: &SharedShape,
        obstacle_pose: &Isometry3<f64>,
        margin: f64,
    ) -> Vec<ContactPoint> {
        let isometries = poses_to_isometries(link_poses);
        self.mesh_backend
            .contact_points(&isometries, obstacle, obstacle_pose, margin)
    }

    /// Self-collision check with LOD.
    ///
    /// Uses sphere check as broadphase, then refines close pairs with the
    /// appropriate LOD backend.
    pub fn check_self_collision(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        skip_pairs: &[(usize, usize)],
    ) -> bool {
        // Broadphase: sphere self-collision with combined margin
        let total_margin = self.config.convex_threshold + self.config.safety_margin;
        if !runtime.self_collision_with_margin(skip_pairs, total_margin) {
            return false;
        }

        // Some pair is close — refine with appropriate LOD
        let isometries = poses_to_isometries(link_poses);
        let w = &runtime.world;

        for i in 0..w.len() {
            for j in (i + 1)..w.len() {
                let link_a = w.link_id[i];
                let link_b = w.link_id[j];

                if link_a == link_b {
                    continue;
                }
                if skip_pairs
                    .iter()
                    .any(|&(a, b)| (a == link_a && b == link_b) || (a == link_b && b == link_a))
                {
                    continue;
                }

                let sphere_dist = w.signed_distance(i, w, j);
                let lod = self.config.select_lod(sphere_dist);

                let exact_dist = match lod {
                    CollisionLOD::Sphere => {
                        if sphere_dist < self.config.safety_margin {
                            return true;
                        }
                        continue;
                    }
                    CollisionLOD::ConvexHull => self.convex_backend.link_distance(
                        link_a,
                        &isometries[link_a],
                        link_b,
                        &isometries[link_b],
                    ),
                    CollisionLOD::Mesh => self.mesh_backend.link_distance(
                        link_a,
                        &isometries[link_a],
                        link_b,
                        &isometries[link_b],
                    ),
                };

                if exact_dist < self.config.safety_margin {
                    return true;
                }
            }
        }

        false
    }

    /// Internal: compute min distance to obstacle spheres at a given LOD.
    fn min_distance_at_lod(
        &self,
        lod: CollisionLOD,
        link_transforms: &[Isometry3<f64>],
        obstacles: &crate::soa::SpheresSoA,
    ) -> f64 {
        let mut min_dist = f64::INFINITY;

        for i in 0..obstacles.len() {
            let obs_shape = SharedShape::ball(obstacles.radius[i]);
            let obs_pose =
                Isometry3::translation(obstacles.x[i], obstacles.y[i], obstacles.z[i]);

            let dist = match lod {
                CollisionLOD::Sphere => unreachable!(),
                CollisionLOD::ConvexHull => {
                    self.convex_backend
                        .min_distance_exact(link_transforms, &obs_shape, &obs_pose)
                }
                CollisionLOD::Mesh => {
                    self.mesh_backend
                        .min_distance_exact(link_transforms, &obs_shape, &obs_pose)
                }
            };

            if dist < min_dist {
                min_dist = dist;
            }
        }

        min_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capt::AABB;
    use crate::soa::SpheresSoA;
    use crate::sphere_model::adjacent_link_pairs;

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

    fn setup() -> (Robot, LODCollisionChecker) {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let checker = LODCollisionChecker::from_robot(&robot);
        (robot, checker)
    }

    fn identity_poses(robot: &Robot) -> Vec<Pose> {
        (0..robot.links.len()).map(|_| Pose::identity()).collect()
    }

    // --- LODConfig tests ---

    #[test]
    fn lod_config_select_sphere_for_far() {
        let config = LODConfig::default();
        assert_eq!(config.select_lod(1.0), CollisionLOD::Sphere);
    }

    #[test]
    fn lod_config_select_convex_for_medium() {
        let config = LODConfig::default();
        // convex_threshold=0.10, mesh_threshold=0.02, safety=0.02
        // 0.05 < 0.10 + 0.02 = 0.12, but > 0.02 + 0.02 = 0.04
        assert_eq!(config.select_lod(0.05), CollisionLOD::ConvexHull);
    }

    #[test]
    fn lod_config_select_mesh_for_close() {
        let config = LODConfig::default();
        // 0.01 < 0.02 + 0.02 = 0.04
        assert_eq!(config.select_lod(0.01), CollisionLOD::Mesh);
    }

    #[test]
    fn lod_config_exploration_always_sphere() {
        let config = LODConfig::exploration();
        assert_eq!(config.select_lod(0.0), CollisionLOD::Sphere);
        assert_eq!(config.select_lod(-1.0), CollisionLOD::Sphere);
    }

    #[test]
    fn lod_config_validation_always_mesh() {
        let config = LODConfig::validation();
        assert_eq!(config.select_lod(100.0), CollisionLOD::Mesh);
        assert_eq!(config.select_lod(0.001), CollisionLOD::Mesh);
    }

    #[test]
    fn lod_config_forced() {
        assert_eq!(
            LODConfig::forced(CollisionLOD::ConvexHull).select_lod(0.5),
            CollisionLOD::ConvexHull
        );
    }

    // --- ConvexCollisionBackend tests ---

    #[test]
    fn convex_backend_builds_shapes() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = ConvexCollisionBackend::from_robot(&robot);
        // 3 links with geometry
        assert_eq!(backend.num_shapes(), 3);
        assert!(backend.has_shape(0));
        assert!(backend.has_shape(1));
        assert!(backend.has_shape(2));
        assert!(!backend.has_shape(3)); // ee_link
    }

    #[test]
    fn convex_backend_distance() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = ConvexCollisionBackend::from_robot(&robot);
        let transforms: Vec<Isometry3<f64>> =
            (0..robot.links.len()).map(|_| Isometry3::identity()).collect();

        // Overlapping obstacle
        let obs = SharedShape::ball(0.5);
        let obs_pose = Isometry3::identity();
        let dist = backend.min_distance_exact(&transforms, &obs, &obs_pose);
        assert!(dist <= 0.01, "Overlapping: {}", dist);

        // Far obstacle
        let far_pose = Isometry3::translation(5.0, 5.0, 5.0);
        let far_dist = backend.min_distance_exact(&transforms, &obs, &far_pose);
        assert!(far_dist > 1.0, "Far: {}", far_dist);
    }

    #[test]
    fn convex_backend_link_distance() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let backend = ConvexCollisionBackend::from_robot(&robot);

        let t_base = Isometry3::identity();
        let t_link2 = Isometry3::translation(0.0, 0.0, 1.0);
        let dist = backend.link_distance(0, &t_base, 2, &t_link2);
        assert!(dist > 0.0, "Separated links: {}", dist);

        // Missing link
        let dist2 = backend.link_distance(0, &t_base, 3, &t_link2);
        assert_eq!(dist2, f64::INFINITY);
    }

    // --- LODCollisionChecker tests ---

    #[test]
    fn lod_checker_far_obstacle_uses_sphere() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(5.0, 5.0, 5.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));

        let (collides, lod) = checker.check_collision(&runtime, &poses, &env);
        assert!(!collides);
        assert_eq!(lod, CollisionLOD::Sphere, "Far obstacle should use Sphere");
    }

    #[test]
    fn lod_checker_overlapping_uses_mesh() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 0.5, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(2.0));

        let (collides, lod) = checker.check_collision(&runtime, &poses, &env);
        assert!(collides);
        assert_eq!(lod, CollisionLOD::Mesh, "Overlapping should use Mesh");
    }

    #[test]
    fn lod_checker_exploration_mode() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let mut checker = LODCollisionChecker::from_robot(&robot);
        checker.set_config(LODConfig::exploration());

        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(5.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));

        let (_, lod) = checker.check_collision(&runtime, &poses, &env);
        assert_eq!(lod, CollisionLOD::Sphere, "Exploration always sphere");
    }

    #[test]
    fn lod_checker_validation_mode() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let mut checker = LODCollisionChecker::from_robot(&robot);
        checker.set_config(LODConfig::validation());

        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(5.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));

        let (_, lod) = checker.check_collision(&runtime, &poses, &env);
        assert_eq!(lod, CollisionLOD::Mesh, "Validation always mesh");
    }

    #[test]
    fn lod_checker_min_distance_forced() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(2.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(5.0));

        let d_sphere = checker.min_distance_with_lod(
            CollisionLOD::Sphere,
            &runtime,
            &poses,
            &env,
        );
        let d_convex = checker.min_distance_with_lod(
            CollisionLOD::ConvexHull,
            &runtime,
            &poses,
            &env,
        );
        let d_mesh = checker.min_distance_with_lod(
            CollisionLOD::Mesh,
            &runtime,
            &poses,
            &env,
        );

        // All should be positive (obstacle at 2m)
        assert!(d_sphere > 0.0, "sphere: {}", d_sphere);
        assert!(d_convex > 0.0, "convex: {}", d_convex);
        assert!(d_mesh > 0.0, "mesh: {}", d_mesh);

        // Sphere approximation inflates robot geometry → obstacle appears closer
        // so d_sphere <= d_convex <= d_mesh (spheres are most conservative)
        assert!(
            d_sphere <= d_convex + 0.01,
            "Sphere ({}) should be <= convex ({}) (conservative inflated model)",
            d_sphere,
            d_convex
        );
    }

    #[test]
    fn lod_checker_obstacle_shape() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        // Far obstacle
        let obs = SharedShape::ball(0.1);
        let far = Isometry3::translation(5.0, 5.0, 5.0);
        let (collides, lod) = checker.check_obstacle(&runtime, &poses, &obs, &far);
        assert!(!collides);
        assert_eq!(lod, CollisionLOD::Sphere);

        // Overlapping obstacle
        let near = Isometry3::identity();
        let big = SharedShape::ball(0.5);
        let (collides2, _) = checker.check_obstacle(&runtime, &poses, &big, &near);
        assert!(collides2);
    }

    #[test]
    fn lod_checker_self_collision() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let skip = adjacent_link_pairs(&robot);

        // Separated poses → no self-collision
        let mut poses = identity_poses(&robot);
        poses[0] = Pose::from_xyz(0.0, 0.0, 0.0);
        poses[1] = Pose::from_xyz(0.0, 0.0, 0.5);
        poses[2] = Pose::from_xyz(0.0, 0.0, 1.0);
        poses[3] = Pose::from_xyz(0.0, 0.0, 1.5);
        runtime.update(&poses);

        assert!(
            !checker.check_self_collision(&runtime, &poses, &skip),
            "Separated links should not self-collide"
        );
    }

    #[test]
    fn lod_display() {
        assert_eq!(format!("{}", CollisionLOD::Sphere), "Sphere");
        assert_eq!(format!("{}", CollisionLOD::ConvexHull), "ConvexHull");
        assert_eq!(format!("{}", CollisionLOD::Mesh), "Mesh");
    }

    #[test]
    fn lod_ordering() {
        assert!(CollisionLOD::Sphere < CollisionLOD::ConvexHull);
        assert!(CollisionLOD::ConvexHull < CollisionLOD::Mesh);
    }

    #[test]
    fn no_false_negatives_sphere_to_mesh() {
        // Key guarantee: if sphere says no collision, mesh also says no collision.
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let poses = identity_poses(&robot);
        runtime.update(&poses);

        let mut obs = SpheresSoA::new();
        obs.push(3.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(5.0));

        let d_sphere = checker.min_distance_with_lod(CollisionLOD::Sphere, &runtime, &poses, &env);
        let d_mesh = checker.min_distance_with_lod(CollisionLOD::Mesh, &runtime, &poses, &env);

        // Sphere is conservative (inflated): d_sphere <= d_mesh
        // If sphere says "no collision" (d > 0), mesh should also say "no collision"
        if d_sphere > 0.0 {
            assert!(
                d_mesh > -0.01,
                "Sphere says safe ({}), but mesh says collision ({})",
                d_sphere,
                d_mesh
            );
        }
    }
}
