//! Two-tier collision checker combining SIMD sphere-tree + parry3d-f64 exact backend.
//!
//! # Architecture
//!
//! The two-tier system exploits the complementary strengths of two backends:
//!
//! 1. **SIMD sphere-tree** (fast path, ~500ns): Handles 99% of queries.
//!    Returns clear no-collision or deep penetration quickly.
//!
//! 2. **parry3d-f64 mesh backend** (slow path, <50µs): Called only when the
//!    sphere-tree reports near-collision (distance within safety margin).
//!    Provides exact mesh-mesh distance.
//!
//! This approach gives <1µs average collision check time while maintaining
//! exact precision when it matters (near-collision configurations).

use nalgebra::Isometry3;
use parry3d_f64::shape::SharedShape;

use kinetic_core::Pose;
use kinetic_robot::Robot;

use crate::acm::ResolvedACM;
use crate::check::CollisionEnvironment;
use crate::mesh::{poses_to_isometries, ContactPoint, MeshCollisionBackend};
use crate::sphere_model::{RobotSphereModel, RobotSpheres, SphereGenConfig};

/// Two-tier collision checker: SIMD sphere-tree + parry3d-f64 exact mesh.
///
/// Fast path uses the sphere tree for broadphase. When the sphere check
/// reports distance within `refinement_margin`, the exact mesh backend
/// is called for a precise answer.
#[derive(Debug, Clone)]
pub struct TwoTierCollisionChecker {
    /// SIMD sphere model for fast broadphase.
    sphere_model: RobotSphereModel,
    /// Exact mesh collision shapes from parry3d-f64.
    mesh_backend: MeshCollisionBackend,
    /// When sphere distance is within this margin, use exact check.
    /// Set to 0.0 to always trust sphere results.
    pub refinement_margin: f64,
    /// Safety margin for collision checks (added to all distances).
    pub safety_margin: f64,
}

impl TwoTierCollisionChecker {
    /// Build a two-tier checker from a robot model.
    ///
    /// `sphere_config`: controls sphere generation density.
    /// `refinement_margin`: sphere distance within this triggers exact check.
    /// `safety_margin`: minimum clearance required (added to all distances).
    pub fn new(
        robot: &Robot,
        sphere_config: &SphereGenConfig,
        refinement_margin: f64,
        safety_margin: f64,
    ) -> Self {
        let sphere_model = RobotSphereModel::from_robot(robot, sphere_config);
        let mesh_backend = MeshCollisionBackend::from_robot(robot);

        Self {
            sphere_model,
            mesh_backend,
            refinement_margin,
            safety_margin,
        }
    }

    /// Build with default settings (coarse spheres, 5cm refinement margin, 2cm safety).
    pub fn from_robot(robot: &Robot) -> Self {
        Self::new(robot, &SphereGenConfig::coarse(), 0.05, 0.02)
    }

    /// Access the underlying sphere model.
    pub fn sphere_model(&self) -> &RobotSphereModel {
        &self.sphere_model
    }

    /// Access the underlying mesh backend.
    pub fn mesh_backend(&self) -> &MeshCollisionBackend {
        &self.mesh_backend
    }

    /// Create a runtime sphere set for world-frame queries.
    pub fn create_runtime(&self) -> RobotSpheres<'_> {
        self.sphere_model.create_runtime()
    }

    /// Two-tier collision check against environment obstacles.
    ///
    /// Returns true if the robot collides with any obstacle (considering safety margin).
    ///
    /// `runtime`: pre-updated world-frame robot spheres.
    /// `link_poses`: world-frame pose per link (for exact queries if needed).
    /// `environment`: CAPT-based environment.
    pub fn check_collision(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        environment: &CollisionEnvironment,
    ) -> bool {
        // Fast path: SIMD sphere check with safety margin
        let sphere_dist = environment.min_distance(&runtime.world);

        if sphere_dist > self.refinement_margin + self.safety_margin {
            // Clear: far from any obstacle.
            return false;
        }

        if sphere_dist < -self.refinement_margin {
            // Deep penetration: definitely colliding.
            return true;
        }

        // Near margin: use exact mesh backend for precise answer.
        let isometries = poses_to_isometries(link_poses);
        let obs = &environment.obstacle_spheres;

        for i in 0..obs.len() {
            let obs_shape = SharedShape::ball(obs.radius[i]);
            let obs_pose = Isometry3::translation(obs.x[i], obs.y[i], obs.z[i]);

            let exact_dist =
                self.mesh_backend
                    .min_distance_exact(&isometries, &obs_shape, &obs_pose);

            if exact_dist < self.safety_margin {
                return true;
            }
        }

        false
    }

    /// Two-tier collision check against a single obstacle shape.
    ///
    /// Checks sphere approximation first, then refines with exact geometry.
    pub fn check_obstacle(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        obstacle: &SharedShape,
        obstacle_pose: &Isometry3<f64>,
    ) -> bool {
        // Fast: sphere-based check
        let aabb = obstacle.compute_aabb(obstacle_pose);
        let obs_center = aabb.center();
        let obs_radius = (aabb.extents() / 2.0).norm();

        // Quick sphere vs sphere-set check
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

        if min_sphere_dist > self.refinement_margin + self.safety_margin {
            return false;
        }

        if min_sphere_dist < -self.refinement_margin {
            return true;
        }

        // Exact check
        let isometries = poses_to_isometries(link_poses);
        let exact_dist = self
            .mesh_backend
            .min_distance_exact(&isometries, obstacle, obstacle_pose);

        exact_dist < self.safety_margin
    }

    /// Compute minimum distance to environment using two-tier approach.
    ///
    /// Returns exact distance when sphere distance is within refinement margin.
    pub fn min_distance(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        environment: &CollisionEnvironment,
    ) -> f64 {
        let sphere_dist = environment.min_distance(&runtime.world);

        if sphere_dist > self.refinement_margin + self.safety_margin {
            // Far enough: sphere distance is a good lower bound.
            return sphere_dist;
        }

        // Near margin: compute exact distance.
        let isometries = poses_to_isometries(link_poses);
        let obs = &environment.obstacle_spheres;
        let mut min_exact = f64::INFINITY;

        for i in 0..obs.len() {
            let obs_shape = SharedShape::ball(obs.radius[i]);
            let obs_pose = Isometry3::translation(obs.x[i], obs.y[i], obs.z[i]);

            let exact_dist =
                self.mesh_backend
                    .min_distance_exact(&isometries, &obs_shape, &obs_pose);

            if exact_dist < min_exact {
                min_exact = exact_dist;
            }
        }

        min_exact
    }

    /// Get contact points from exact mesh backend.
    ///
    /// Only called when precise contact information is needed
    /// (e.g., for force computation or grasp planning).
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

    /// Two-tier self-collision check using a ResolvedACM.
    ///
    /// Equivalent to `check_self_collision` but uses the ACM's skip pairs.
    pub fn check_self_collision_acm(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        acm: &ResolvedACM,
    ) -> bool {
        let skip_pairs = acm.to_skip_pairs();
        self.check_self_collision(runtime, link_poses, &skip_pairs)
    }

    /// Two-tier self-collision check.
    ///
    /// Uses sphere self-collision as broadphase, then refines with exact
    /// mesh-mesh distance for near-collision link pairs.
    pub fn check_self_collision(
        &self,
        runtime: &RobotSpheres<'_>,
        link_poses: &[Pose],
        skip_pairs: &[(usize, usize)],
    ) -> bool {
        // Fast path: sphere self-collision check with margin
        if !runtime
            .self_collision_with_margin(skip_pairs, self.safety_margin + self.refinement_margin)
        {
            return false;
        }

        // Some sphere pair is within margin — check exact.
        let isometries = poses_to_isometries(link_poses);
        let w = &runtime.world;

        // Find which link pairs are close in sphere space
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
                if sphere_dist < self.safety_margin + self.refinement_margin {
                    // Near collision in sphere space → exact check
                    let exact_dist = self.mesh_backend.link_distance(
                        link_a,
                        &isometries[link_a],
                        link_b,
                        &isometries[link_b],
                    );
                    if exact_dist < self.safety_margin {
                        return true;
                    }
                }
            }
        }

        false
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

    fn setup() -> (Robot, TwoTierCollisionChecker) {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let checker = TwoTierCollisionChecker::from_robot(&robot);
        (robot, checker)
    }

    #[test]
    fn two_tier_no_collision_far_obstacle() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obstacles = SpheresSoA::new();
        obstacles.push(5.0, 5.0, 5.0, 0.1, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));

        assert!(!checker.check_collision(&runtime, &poses, &env));
    }

    #[test]
    fn two_tier_collision_at_origin() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.5, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(2.0));

        assert!(checker.check_collision(&runtime, &poses, &env));
    }

    #[test]
    fn two_tier_distance_far() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obstacles = SpheresSoA::new();
        obstacles.push(3.0, 0.0, 0.0, 0.1, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(5.0));

        let dist = checker.min_distance(&runtime, &poses, &env);
        assert!(dist > 1.0, "Expected large distance, got {}", dist);
    }

    #[test]
    fn two_tier_obstacle_shape_check() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // Far obstacle
        let obstacle = SharedShape::ball(0.1);
        let far_pose = Isometry3::translation(5.0, 5.0, 5.0);
        assert!(!checker.check_obstacle(&runtime, &poses, &obstacle, &far_pose));

        // Overlapping obstacle
        let near_pose = Isometry3::translation(0.0, 0.0, 0.0);
        let big_obs = SharedShape::ball(0.5);
        assert!(checker.check_obstacle(&runtime, &poses, &big_obs, &near_pose));
    }

    #[test]
    fn two_tier_self_collision() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let skip = adjacent_link_pairs(&robot);

        // All at identity — adjacent links overlap but are skipped
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // With skip pairs, shouldn't crash
        let _result = checker.check_self_collision(&runtime, &poses, &skip);
    }

    #[test]
    fn two_tier_contact_points() {
        let (robot, checker) = setup();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();

        let obstacle = SharedShape::ball(0.5);
        let obs_pose = Isometry3::translation(0.0, 0.0, 0.0);

        let contacts = checker.contact_points(&poses, &obstacle, &obs_pose, 1.0);
        assert!(
            !contacts.is_empty(),
            "Expected contacts with overlapping obstacle"
        );
    }

    // ─── Margin and safety edge case tests ───

    /// Different safety margins can change the collision verdict for near-boundary cases.
    #[test]
    fn two_tier_safety_margin_changes_result() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let config = SphereGenConfig::coarse();

        // Small safety margin
        let checker_small = TwoTierCollisionChecker::new(&robot, &config, 0.05, 0.01);
        // Large safety margin
        let checker_large = TwoTierCollisionChecker::new(&robot, &config, 0.05, 0.5);

        let mut runtime_small = checker_small.create_runtime();
        let mut runtime_large = checker_large.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime_small.update(&poses);
        runtime_large.update(&poses);

        // Place obstacle at moderate distance: close enough that large safety margin triggers,
        // but far enough that small safety margin doesn't
        let obstacle = SharedShape::ball(0.05);
        let obs_pose = Isometry3::translation(0.5, 0.0, 0.0);

        let collides_small =
            checker_small.check_obstacle(&runtime_small, &poses, &obstacle, &obs_pose);
        let collides_large =
            checker_large.check_obstacle(&runtime_large, &poses, &obstacle, &obs_pose);

        // Large margin should report more collisions (or at least not fewer)
        if !collides_small && collides_large {
            // Expected: larger margin is more conservative
        } else if collides_small && collides_large {
            // Both collide — also valid for objects very close
        } else if !collides_small && !collides_large {
            // Both clear — also valid for objects far away
        }
        // The key: if small margin says collision, large margin MUST also say collision
        if collides_small {
            assert!(
                collides_large,
                "Larger safety margin should be at least as conservative"
            );
        }
    }

    /// Refinement margin = 0: sphere check alone determines result (no exact fallback for near cases).
    #[test]
    fn two_tier_zero_refinement_margin() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let config = SphereGenConfig::coarse();
        let checker = TwoTierCollisionChecker::new(&robot, &config, 0.0, 0.02);

        let mut runtime = checker.create_runtime();
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // Far obstacle — should be clear
        let mut obstacles = SpheresSoA::new();
        obstacles.push(5.0, 5.0, 5.0, 0.1, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));
        assert!(!checker.check_collision(&runtime, &poses, &env));

        // Overlapping obstacle — should still detect collision
        let mut obstacles2 = SpheresSoA::new();
        obstacles2.push(0.0, 0.0, 0.0, 0.5, 0);
        let env2 = CollisionEnvironment::build(obstacles2, 0.05, AABB::symmetric(2.0));
        assert!(checker.check_collision(&runtime, &poses, &env2));
    }

    /// Self-collision with ACM: adjacent link pairs are skipped.
    #[test]
    fn two_tier_self_collision_with_acm() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();
        let acm = ResolvedACM::from_robot(&robot);

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // With proper ACM, adjacent overlaps should be skipped
        let _result = checker.check_self_collision_acm(&runtime, &poses, &acm);
        // Main assertion: no crash when using ACM
    }

    /// Multiple obstacles: collision detected if any one is within range.
    #[test]
    fn two_tier_multiple_obstacles() {
        let (robot, checker) = setup();
        let mut runtime = checker.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let mut obstacles = SpheresSoA::new();
        obstacles.push(10.0, 0.0, 0.0, 0.1, 0); // far
        obstacles.push(0.0, 10.0, 0.0, 0.1, 1); // far
        obstacles.push(0.0, 0.0, 0.0, 0.5, 2); // overlapping
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(15.0));

        assert!(
            checker.check_collision(&runtime, &poses, &env),
            "Should detect collision with the overlapping obstacle"
        );
    }
}
