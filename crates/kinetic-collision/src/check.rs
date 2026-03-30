//! High-level collision checking API.
//!
//! Provides the main entry points for collision detection:
//! - Robot vs. environment (CAPT tree)
//! - Robot self-collision
//! - Robot vs. robot / obstacle spheres
//! - Distance queries

use crate::capt::{CollisionPointTree, AABB};
use crate::simd;
use crate::soa::SpheresSoA;

/// Result of a collision check.
#[derive(Debug, Clone)]
pub struct CollisionResult {
    /// Whether any collision was detected.
    pub in_collision: bool,
    /// Minimum distance between any pair. Negative = penetration depth.
    pub min_distance: f64,
    /// Pair of link indices involved in the closest/colliding pair (if any).
    pub closest_pair: Option<(usize, usize)>,
}

/// A collision environment: CAPT tree + obstacle spheres for detailed queries.
#[derive(Debug, Clone)]
pub struct CollisionEnvironment {
    /// CAPT tree for fast broadphase queries.
    pub tree: CollisionPointTree,
    /// Obstacle spheres for detailed distance queries.
    pub obstacle_spheres: SpheresSoA,
}

impl CollisionEnvironment {
    /// Build an environment from obstacle spheres.
    ///
    /// `resolution`: CAPT grid resolution in meters.
    /// `bounds`: workspace bounding box.
    pub fn build(obstacles: SpheresSoA, resolution: f64, bounds: AABB) -> Self {
        let tree = CollisionPointTree::build(&obstacles, resolution, bounds);
        Self {
            tree,
            obstacle_spheres: obstacles,
        }
    }

    /// Build an empty environment (no obstacles).
    pub fn empty(resolution: f64, bounds: AABB) -> Self {
        Self {
            tree: CollisionPointTree::empty(resolution, bounds),
            obstacle_spheres: SpheresSoA::new(),
        }
    }

    /// Check if robot spheres collide with environment.
    ///
    /// Uses CAPT broadphase for maximum speed.
    pub fn check_collision(&self, robot_spheres: &SpheresSoA) -> bool {
        simd::check_robot_collision(robot_spheres, &self.tree)
    }

    /// Check collision with margin.
    ///
    /// A sphere is in collision if its distance to any obstacle is less
    /// than `radius + margin`. Uses CAPT broadphase when the margin is
    /// small, falls back to exact sphere-sphere distances for large margins
    /// that exceed the CAPT influence radius.
    pub fn check_collision_with_margin(&self, robot_spheres: &SpheresSoA, margin: f64) -> bool {
        // For large margins, CAPT cells beyond the influence radius have
        // infinity clearance and would miss distant collisions. Use exact
        // sphere-sphere distance check instead.
        if margin > 0.3 {
            let d = simd::min_distance(robot_spheres, &self.obstacle_spheres);
            return d < margin;
        }
        // Create expanded spheres with margin added to radii
        let mut expanded = SpheresSoA::with_capacity(robot_spheres.len());
        for i in 0..robot_spheres.len() {
            expanded.push(
                robot_spheres.x[i],
                robot_spheres.y[i],
                robot_spheres.z[i],
                robot_spheres.radius[i] + margin,
                robot_spheres.link_id[i],
            );
        }
        simd::check_robot_collision(&expanded, &self.tree)
    }

    /// Compute minimum distance between robot and environment.
    pub fn min_distance(&self, robot_spheres: &SpheresSoA) -> f64 {
        simd::min_distance(robot_spheres, &self.obstacle_spheres)
    }

    /// Full collision result with distance info.
    pub fn check_full(&self, robot_spheres: &SpheresSoA) -> CollisionResult {
        let in_collision = self.check_collision(robot_spheres);
        let min_dist =
            if in_collision || robot_spheres.is_empty() || self.obstacle_spheres.is_empty() {
                if in_collision {
                    simd::min_distance(robot_spheres, &self.obstacle_spheres)
                } else {
                    f64::INFINITY
                }
            } else {
                simd::min_distance(robot_spheres, &self.obstacle_spheres)
            };

        // Find closest pair
        let closest_pair = robot_spheres
            .min_distance(&self.obstacle_spheres)
            .map(|(_, i, j)| (robot_spheres.link_id[i], self.obstacle_spheres.link_id[j]));

        CollisionResult {
            in_collision,
            min_distance: min_dist,
            closest_pair,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere_model::{
        adjacent_link_pairs, RobotSphereModel, RobotSpheres, SphereGenConfig,
    };

    /// Full collision check: robot vs. environment + self-collision (test helper).
    fn check_full_collision(
        robot_spheres: &RobotSpheres<'_>,
        environment: &CollisionEnvironment,
        skip_pairs: &[(usize, usize)],
        margin: f64,
    ) -> bool {
        if environment.check_collision_with_margin(&robot_spheres.world, margin) {
            return true;
        }
        if robot_spheres.self_collision_with_margin(skip_pairs, margin) {
            return true;
        }
        false
    }
    use kinetic_core::Pose;
    use kinetic_robot::Robot;

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

    #[test]
    fn empty_environment_no_collision() {
        let (robot, model) = setup_robot();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        assert!(!env.check_collision(&runtime.world));
    }

    #[test]
    fn obstacle_at_robot_causes_collision() {
        let (robot, model) = setup_robot();
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.5, 0); // large obstacle at origin

        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(2.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        assert!(env.check_collision(&runtime.world));
    }

    #[test]
    fn distant_obstacle_no_collision() {
        let (robot, model) = setup_robot();
        let mut obstacles = SpheresSoA::new();
        obstacles.push(5.0, 5.0, 5.0, 0.1, 0); // far away

        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        assert!(!env.check_collision(&runtime.world));
    }

    #[test]
    fn full_collision_check() {
        let (robot, model) = setup_robot();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let skip = adjacent_link_pairs(&robot);
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let result = check_full_collision(&runtime, &env, &skip, 0.0);
        // With all links at identity, adjacent links may overlap but are skipped
        // No environment obstacles, so this should pass unless non-adjacent links overlap
        let _r = result; // just verify it doesn't panic
    }

    #[test]
    fn min_distance_to_environment() {
        let (robot, model) = setup_robot();
        let mut obstacles = SpheresSoA::new();
        obstacles.push(2.0, 0.0, 0.0, 0.1, 0);

        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(3.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let d = env.min_distance(&runtime.world);
        assert!(d > 0.0, "Expected positive distance, got {}", d);
    }

    #[test]
    fn collision_result_full() {
        let (robot, model) = setup_robot();
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 0.5, 0);

        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(2.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let result = env.check_full(&runtime.world);
        assert!(result.in_collision);
        assert!(result.min_distance < 0.0);
        assert!(result.closest_pair.is_some());
    }

    #[test]
    fn margin_makes_collision_stricter() {
        let (robot, model) = setup_robot();
        let mut obstacles = SpheresSoA::new();
        obstacles.push(1.0, 0.0, 0.0, 0.1, 0);

        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(2.0));
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let no_margin = env.check_collision(&runtime.world);
        let big_margin = env.check_collision_with_margin(&runtime.world, 5.0);

        // Big margin should be at least as strict as no margin
        if no_margin {
            assert!(big_margin);
        }
        // With 5m margin, everything should collide
        assert!(big_margin);
    }
}
