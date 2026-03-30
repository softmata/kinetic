//! Self-collision detection for robot configurations.
//!
//! Generates non-adjacent link pairs from the URDF kinematic tree and
//! provides fast self-collision checking using the sphere model and ACM.

use crate::acm::ResolvedACM;
use crate::soa::SpheresSoA;
use crate::sphere_model::RobotSpheres;

use kinetic_robot::Robot;

/// Pre-computed self-collision pairs for a robot.
///
/// Contains only the non-adjacent link pairs that should actually be
/// checked for self-collision. Adjacent links (connected by a joint)
/// are excluded since they always appear to overlap at the joint.
#[derive(Debug, Clone)]
pub struct SelfCollisionPairs {
    /// Pairs of link indices to check: (link_a, link_b) with a < b.
    pairs: Vec<(usize, usize)>,
}

impl SelfCollisionPairs {
    /// Generate self-collision pairs from a robot model.
    ///
    /// Includes all non-adjacent link pairs. Adjacent links (connected
    /// by a joint) are automatically excluded.
    pub fn from_robot(robot: &Robot) -> Self {
        let resolved = ResolvedACM::from_robot(robot);
        Self::from_resolved_acm(robot, &resolved)
    }

    /// Generate self-collision pairs using a custom ACM.
    ///
    /// All link pairs NOT in the ACM's allowed set will be checked.
    pub fn from_resolved_acm(robot: &Robot, acm: &ResolvedACM) -> Self {
        let num_links = robot.links.len();
        let mut pairs = Vec::new();

        for a in 0..num_links {
            for b in (a + 1)..num_links {
                if !acm.is_allowed(a, b) {
                    pairs.push((a, b));
                }
            }
        }

        Self { pairs }
    }

    /// Number of pairs to check.
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Get the pairs as a slice.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Check for self-collision using sphere approximations.
    ///
    /// Returns true if any non-allowed pair of links has overlapping spheres.
    pub fn check_self_collision(&self, robot_spheres: &RobotSpheres<'_>) -> bool {
        self.check_with_margin(robot_spheres, 0.0)
    }

    /// Check for self-collision with a safety margin.
    ///
    /// Returns true if any non-allowed pair of links is within `margin` distance.
    pub fn check_with_margin(&self, robot_spheres: &RobotSpheres<'_>, margin: f64) -> bool {
        let w = &robot_spheres.world;

        for &(link_a, link_b) in &self.pairs {
            if check_link_pair_collision(w, link_a, link_b, margin) {
                return true;
            }
        }

        false
    }

    /// Compute minimum distance across all self-collision pairs.
    ///
    /// Returns `f64::INFINITY` if no pairs to check or no spheres.
    pub fn min_distance(&self, robot_spheres: &RobotSpheres<'_>) -> f64 {
        let w = &robot_spheres.world;
        let mut best = f64::INFINITY;

        for &(link_a, link_b) in &self.pairs {
            let d = link_pair_min_distance(w, link_a, link_b);
            if d < best {
                best = d;
            }
        }

        best
    }

    /// Find the closest self-collision pair and their distance.
    ///
    /// Returns `Some((link_a, link_b, distance))` or `None` if no pairs.
    pub fn closest_pair(&self, robot_spheres: &RobotSpheres<'_>) -> Option<(usize, usize, f64)> {
        let w = &robot_spheres.world;
        let mut best_dist = f64::INFINITY;
        let mut best_pair = None;

        for &(link_a, link_b) in &self.pairs {
            let d = link_pair_min_distance(w, link_a, link_b);
            if d < best_dist {
                best_dist = d;
                best_pair = Some((link_a, link_b, d));
            }
        }

        best_pair
    }
}

/// Check if any spheres from two different links overlap (with margin).
fn check_link_pair_collision(
    spheres: &SpheresSoA,
    link_a: usize,
    link_b: usize,
    margin: f64,
) -> bool {
    for i in 0..spheres.len() {
        if spheres.link_id[i] != link_a {
            continue;
        }
        for j in 0..spheres.len() {
            if spheres.link_id[j] != link_b {
                continue;
            }
            if margin > 0.0 {
                if spheres.overlaps_with_margin(i, spheres, j, margin) {
                    return true;
                }
            } else if spheres.overlaps(i, spheres, j) {
                return true;
            }
        }
    }
    false
}

/// Minimum distance between spheres of two specific links.
fn link_pair_min_distance(spheres: &SpheresSoA, link_a: usize, link_b: usize) -> f64 {
    let mut best = f64::INFINITY;

    for i in 0..spheres.len() {
        if spheres.link_id[i] != link_a {
            continue;
        }
        for j in 0..spheres.len() {
            if spheres.link_id[j] != link_b {
                continue;
            }
            let d = spheres.signed_distance(i, spheres, j);
            if d < best {
                best = d;
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acm::AllowedCollisionMatrix;
    use crate::sphere_model::{RobotSphereModel, SphereGenConfig};
    use kinetic_core::Pose;

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

    #[test]
    fn self_collision_pairs_from_robot() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let pairs = SelfCollisionPairs::from_robot(&robot);

        // 4 links: base(0), link1(1), link2(2), ee(3)
        // Adjacent (excluded): (0,1), (1,2), (2,3)
        // Non-adjacent (checked): (0,2), (0,3), (1,3)
        assert_eq!(pairs.num_pairs(), 3);
    }

    #[test]
    fn self_collision_with_custom_acm() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let mut acm = AllowedCollisionMatrix::from_robot(&robot);

        // Also allow base_link <-> link2
        acm.allow("base_link", "link2");
        let resolved = ResolvedACM::from_acm(&acm, &robot);
        let pairs = SelfCollisionPairs::from_resolved_acm(&robot, &resolved);

        // Now only (0,3) and (1,3) should be checked
        assert_eq!(pairs.num_pairs(), 2);
    }

    #[test]
    fn self_collision_check_separated_links() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let pairs = SelfCollisionPairs::from_robot(&robot);
        let mut runtime = model.create_runtime();

        // Place links far apart — no self-collision
        let mut poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        poses[0] = Pose::from_xyz(0.0, 0.0, 0.0);
        poses[1] = Pose::from_xyz(0.0, 0.0, 0.5);
        poses[2] = Pose::from_xyz(0.0, 0.0, 1.0);
        poses[3] = Pose::from_xyz(0.0, 0.0, 1.5);
        runtime.update(&poses);

        assert!(
            !pairs.check_self_collision(&runtime),
            "Separated links should not self-collide"
        );
    }

    #[test]
    fn self_collision_check_overlapping_links() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let pairs = SelfCollisionPairs::from_robot(&robot);
        let mut runtime = model.create_runtime();

        // Place non-adjacent links at the same position — self-collision
        // base(0) and link2(2) are non-adjacent, put them at the same place
        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        // At identity, all links at origin — non-adjacent links with geometry overlap
        // base(0) has a box at origin, link2(2) has a sphere at origin
        // These are non-adjacent (0,2) → should trigger self-collision
        let result = pairs.check_self_collision(&runtime);
        assert!(result, "Overlapping non-adjacent links should self-collide");
    }

    #[test]
    fn self_collision_min_distance() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let pairs = SelfCollisionPairs::from_robot(&robot);
        let mut runtime = model.create_runtime();

        let mut poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        poses[0] = Pose::from_xyz(0.0, 0.0, 0.0);
        poses[1] = Pose::from_xyz(0.0, 0.0, 0.5);
        poses[2] = Pose::from_xyz(0.0, 0.0, 1.0);
        poses[3] = Pose::from_xyz(0.0, 0.0, 1.5);
        runtime.update(&poses);

        let dist = pairs.min_distance(&runtime);
        assert!(dist > 0.0, "Expected positive distance for separated links");
    }

    #[test]
    fn self_collision_closest_pair() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let pairs = SelfCollisionPairs::from_robot(&robot);
        let mut runtime = model.create_runtime();

        let poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        runtime.update(&poses);

        let closest = pairs.closest_pair(&runtime);
        assert!(closest.is_some(), "Should find closest pair");
    }

    #[test]
    fn self_collision_with_margin() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let pairs = SelfCollisionPairs::from_robot(&robot);
        let mut runtime = model.create_runtime();

        // Separated links
        let mut poses: Vec<Pose> = (0..robot.links.len()).map(|_| Pose::identity()).collect();
        poses[0] = Pose::from_xyz(0.0, 0.0, 0.0);
        poses[1] = Pose::from_xyz(0.0, 0.0, 0.5);
        poses[2] = Pose::from_xyz(0.0, 0.0, 1.0);
        poses[3] = Pose::from_xyz(0.0, 0.0, 1.5);
        runtime.update(&poses);

        let no_margin = pairs.check_with_margin(&runtime, 0.0);
        let big_margin = pairs.check_with_margin(&runtime, 10.0);

        // Big margin should be at least as strict
        if no_margin {
            assert!(big_margin);
        }
        // 10m margin should catch everything
        assert!(big_margin, "Huge margin should detect self-collision");
    }
}
