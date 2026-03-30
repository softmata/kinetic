//! Allowed Collision Matrix (ACM) — configurable collision pair exclusions.
//!
//! The ACM tracks which pairs of objects are allowed to be in collision
//! (i.e., should be skipped during collision checking). Common uses:
//!
//! - **Adjacent links**: links connected by a joint always "overlap" at the joint.
//! - **Grasped objects**: the gripper link + grasped object should not collide.
//! - **Mounted objects**: robot base + table surface.
//!
//! The ACM stores pairs by name (string-based) for user-facing API,
//! and can be resolved to index-based pairs for fast runtime checks.

use std::collections::HashSet;

use kinetic_robot::Robot;

/// Allowed Collision Matrix — tracks pairs that should skip collision checks.
///
/// Pairs are stored as ordered (min, max) name tuples for consistent lookup.
#[derive(Debug, Clone)]
pub struct AllowedCollisionMatrix {
    /// Set of (name_a, name_b) pairs that are allowed to collide.
    /// Always stored with name_a <= name_b lexicographically.
    allowed: HashSet<(String, String)>,
}

impl Default for AllowedCollisionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl AllowedCollisionMatrix {
    /// Create an empty ACM (no pairs allowed).
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Build default ACM from a robot's URDF and configuration.
    ///
    /// Adjacent links (connected by a joint) are always in contact at the
    /// joint origin, so checking them for collision is wasteful.
    /// Also incorporates `collision_preference.skip_pairs` from SRDF/config.
    pub fn from_robot(robot: &Robot) -> Self {
        let mut acm = Self::new();

        // Adjacent links from URDF joints
        for joint in &robot.joints {
            let parent_name = &robot.links[joint.parent_link].name;
            let child_name = &robot.links[joint.child_link].name;
            acm.allow(parent_name, child_name);
        }

        // Skip pairs from robot configuration (SRDF disabled_collisions / TOML config)
        if let Some(pref) = &robot.collision_preference {
            for (a, b) in &pref.skip_pairs {
                acm.allow(a, b);
            }
        }

        acm
    }

    /// Allow collision between two named objects (skip collision checking).
    pub fn allow(&mut self, a: &str, b: &str) {
        let pair = ordered_pair(a, b);
        self.allowed.insert(pair);
    }

    /// Disallow collision (re-enable collision checking between these objects).
    pub fn disallow(&mut self, a: &str, b: &str) {
        let pair = ordered_pair(a, b);
        self.allowed.remove(&pair);
    }

    /// Check if collision between two named objects is allowed (should be skipped).
    pub fn is_allowed(&self, a: &str, b: &str) -> bool {
        let pair = ordered_pair(a, b);
        self.allowed.contains(&pair)
    }

    /// Number of allowed pairs.
    pub fn num_allowed(&self) -> usize {
        self.allowed.len()
    }

    /// Iterate over all allowed pairs.
    pub fn allowed_pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.allowed.iter().map(|(a, b)| (a.as_str(), b.as_str()))
    }

    /// Resolve the ACM to index-based pairs for a given robot.
    ///
    /// Returns `(link_a_idx, link_b_idx)` pairs corresponding to the
    /// allowed name pairs. Pairs where either name is not found in the
    /// robot are silently skipped (they may refer to external objects).
    pub fn resolve_to_indices(&self, robot: &Robot) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for (name_a, name_b) in &self.allowed {
            if let (Ok(idx_a), Ok(idx_b)) = (robot.link_index(name_a), robot.link_index(name_b)) {
                pairs.push((idx_a, idx_b));
            }
        }

        pairs
    }

    /// Merge another ACM into this one (union of allowed pairs).
    pub fn merge(&mut self, other: &AllowedCollisionMatrix) {
        for pair in &other.allowed {
            self.allowed.insert(pair.clone());
        }
    }

    /// Check if a pair of link indices is allowed, given a robot for name lookup.
    ///
    /// This is a convenience for checking by index without pre-resolving.
    /// For hot paths, use `resolve_to_indices` and check the index set directly.
    pub fn is_allowed_by_index(&self, robot: &Robot, link_a: usize, link_b: usize) -> bool {
        if link_a >= robot.links.len() || link_b >= robot.links.len() {
            return false;
        }
        let name_a = &robot.links[link_a].name;
        let name_b = &robot.links[link_b].name;
        self.is_allowed(name_a, name_b)
    }
}

/// Order a pair lexicographically for consistent lookup.
fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Pre-resolved index-based skip set for fast runtime self-collision checks.
///
/// Built from an ACM by resolving names to link indices once upfront.
/// Checking a pair is O(1) with a flat boolean matrix for small robots.
#[derive(Debug, Clone)]
pub struct ResolvedACM {
    /// Set of allowed (link_a, link_b) index pairs, stored with a <= b.
    allowed_pairs: HashSet<(usize, usize)>,
}

impl ResolvedACM {
    /// Resolve an ACM against a robot model.
    pub fn from_acm(acm: &AllowedCollisionMatrix, robot: &Robot) -> Self {
        let index_pairs = acm.resolve_to_indices(robot);
        let mut allowed_pairs = HashSet::with_capacity(index_pairs.len());

        for (a, b) in index_pairs {
            let pair = if a <= b { (a, b) } else { (b, a) };
            allowed_pairs.insert(pair);
        }

        Self { allowed_pairs }
    }

    /// Build a resolved ACM directly from a robot (default adjacent-link exclusions).
    pub fn from_robot(robot: &Robot) -> Self {
        let acm = AllowedCollisionMatrix::from_robot(robot);
        Self::from_acm(&acm, robot)
    }

    /// Check if a pair of link indices is allowed (should be skipped).
    ///
    /// O(1) hash lookup. Both orderings work.
    #[inline]
    pub fn is_allowed(&self, link_a: usize, link_b: usize) -> bool {
        if link_a == link_b {
            return true; // same link always allowed
        }
        let pair = if link_a <= link_b {
            (link_a, link_b)
        } else {
            (link_b, link_a)
        };
        self.allowed_pairs.contains(&pair)
    }

    /// Convert to a Vec of skip pairs (for compatibility with existing API).
    pub fn to_skip_pairs(&self) -> Vec<(usize, usize)> {
        self.allowed_pairs.iter().copied().collect()
    }

    /// Number of allowed pairs.
    pub fn num_allowed(&self) -> usize {
        self.allowed_pairs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn empty_acm() {
        let acm = AllowedCollisionMatrix::new();
        assert_eq!(acm.num_allowed(), 0);
        assert!(!acm.is_allowed("a", "b"));
    }

    #[test]
    fn allow_and_check() {
        let mut acm = AllowedCollisionMatrix::new();
        acm.allow("link_a", "link_b");

        assert!(acm.is_allowed("link_a", "link_b"));
        assert!(acm.is_allowed("link_b", "link_a")); // order doesn't matter
        assert!(!acm.is_allowed("link_a", "link_c"));
    }

    #[test]
    fn disallow_removes_pair() {
        let mut acm = AllowedCollisionMatrix::new();
        acm.allow("link_a", "link_b");
        assert!(acm.is_allowed("link_a", "link_b"));

        acm.disallow("link_b", "link_a"); // reverse order should work
        assert!(!acm.is_allowed("link_a", "link_b"));
    }

    #[test]
    fn from_robot_adjacent_links() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let acm = AllowedCollisionMatrix::from_robot(&robot);

        // Adjacent pairs should be allowed
        assert!(acm.is_allowed("base_link", "link1"));
        assert!(acm.is_allowed("link1", "link2"));
        assert!(acm.is_allowed("link2", "ee_link"));

        // Non-adjacent pairs should NOT be allowed
        assert!(!acm.is_allowed("base_link", "link2"));
        assert!(!acm.is_allowed("base_link", "ee_link"));
        assert!(!acm.is_allowed("link1", "ee_link"));

        assert_eq!(acm.num_allowed(), 3); // 3 joints = 3 adjacent pairs
    }

    #[test]
    fn resolve_to_indices() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let acm = AllowedCollisionMatrix::from_robot(&robot);

        let pairs = acm.resolve_to_indices(&robot);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn resolved_acm_from_robot() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let resolved = ResolvedACM::from_robot(&robot);

        let base_idx = robot.link_index("base_link").unwrap();
        let link1_idx = robot.link_index("link1").unwrap();
        let link2_idx = robot.link_index("link2").unwrap();
        let ee_idx = robot.link_index("ee_link").unwrap();

        // Adjacent → allowed
        assert!(resolved.is_allowed(base_idx, link1_idx));
        assert!(resolved.is_allowed(link1_idx, link2_idx));
        assert!(resolved.is_allowed(link2_idx, ee_idx));

        // Same link → always allowed
        assert!(resolved.is_allowed(base_idx, base_idx));

        // Non-adjacent → not allowed
        assert!(!resolved.is_allowed(base_idx, link2_idx));
        assert!(!resolved.is_allowed(base_idx, ee_idx));
        assert!(!resolved.is_allowed(link1_idx, ee_idx));
    }

    #[test]
    fn resolved_acm_to_skip_pairs() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let resolved = ResolvedACM::from_robot(&robot);

        let pairs = resolved.to_skip_pairs();
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn merge_acms() {
        let mut acm1 = AllowedCollisionMatrix::new();
        acm1.allow("link_a", "link_b");

        let mut acm2 = AllowedCollisionMatrix::new();
        acm2.allow("link_c", "link_d");

        acm1.merge(&acm2);
        assert_eq!(acm1.num_allowed(), 2);
        assert!(acm1.is_allowed("link_a", "link_b"));
        assert!(acm1.is_allowed("link_c", "link_d"));
    }

    #[test]
    fn is_allowed_by_index() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let acm = AllowedCollisionMatrix::from_robot(&robot);

        let base_idx = robot.link_index("base_link").unwrap();
        let link1_idx = robot.link_index("link1").unwrap();
        let link2_idx = robot.link_index("link2").unwrap();

        assert!(acm.is_allowed_by_index(&robot, base_idx, link1_idx));
        assert!(!acm.is_allowed_by_index(&robot, base_idx, link2_idx));
    }

    #[test]
    fn acm_custom_allows() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let mut acm = AllowedCollisionMatrix::from_robot(&robot);

        // Custom allow: base_link + ee_link (e.g., robot folded onto base)
        acm.allow("base_link", "ee_link");
        assert!(acm.is_allowed("base_link", "ee_link"));
        assert_eq!(acm.num_allowed(), 4); // 3 adjacent + 1 custom

        // Also allow an external object
        acm.allow("base_link", "table");
        assert!(acm.is_allowed("base_link", "table"));
        assert_eq!(acm.num_allowed(), 5);

        // External pairs resolve to nothing in index resolution
        let pairs = acm.resolve_to_indices(&robot);
        assert_eq!(pairs.len(), 4); // 3 adjacent + base-ee, "table" not in robot
    }

    // ─── Branching tree, merge, and edge case tests ───

    /// URDF with branching kinematic tree (two children from one parent).
    const BRANCHING_URDF: &str = r#"<?xml version="1.0"?>
<robot name="branching">
  <link name="base"/>
  <link name="left_arm"/>
  <link name="left_hand"/>
  <link name="right_arm"/>
  <link name="right_hand"/>

  <joint name="j_left" type="revolute">
    <parent link="base"/><child link="left_arm"/>
    <origin xyz="0 0.2 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="j_left_hand" type="revolute">
    <parent link="left_arm"/><child link="left_hand"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="j_right" type="revolute">
    <parent link="base"/><child link="right_arm"/>
    <origin xyz="0 -0.2 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="j_right_hand" type="revolute">
    <parent link="right_arm"/><child link="right_hand"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
</robot>
"#;

    #[test]
    fn acm_branching_tree() {
        let robot = Robot::from_urdf_string(BRANCHING_URDF).unwrap();
        let acm = AllowedCollisionMatrix::from_robot(&robot);

        // 4 joints → 4 adjacent pairs
        assert_eq!(acm.num_allowed(), 4);

        // Adjacent pairs
        assert!(acm.is_allowed("base", "left_arm"));
        assert!(acm.is_allowed("left_arm", "left_hand"));
        assert!(acm.is_allowed("base", "right_arm"));
        assert!(acm.is_allowed("right_arm", "right_hand"));

        // Cross-branch pairs should NOT be allowed
        assert!(!acm.is_allowed("left_arm", "right_arm"));
        assert!(!acm.is_allowed("left_hand", "right_hand"));
        assert!(!acm.is_allowed("left_arm", "right_hand"));
        assert!(!acm.is_allowed("right_arm", "left_hand"));

        // Non-adjacent parent-child pairs
        assert!(!acm.is_allowed("base", "left_hand"));
        assert!(!acm.is_allowed("base", "right_hand"));
    }

    #[test]
    fn acm_merge_overlapping_pairs() {
        let mut acm1 = AllowedCollisionMatrix::new();
        acm1.allow("a", "b");
        acm1.allow("b", "c");

        let mut acm2 = AllowedCollisionMatrix::new();
        acm2.allow("b", "c"); // overlap with acm1
        acm2.allow("c", "d");

        acm1.merge(&acm2);

        // Union: {(a,b), (b,c), (c,d)} — 3 unique pairs
        assert_eq!(acm1.num_allowed(), 3);
        assert!(acm1.is_allowed("a", "b"));
        assert!(acm1.is_allowed("b", "c"));
        assert!(acm1.is_allowed("c", "d"));
    }

    #[test]
    fn acm_idempotent_allow() {
        let mut acm = AllowedCollisionMatrix::new();
        acm.allow("x", "y");
        acm.allow("x", "y"); // duplicate
        acm.allow("y", "x"); // same pair, reversed order

        // HashSet deduplication → should have exactly 1 pair
        assert_eq!(acm.num_allowed(), 1);
    }

    #[test]
    fn acm_is_allowed_by_index_out_of_bounds() {
        let robot = Robot::from_urdf_string(GEOM_URDF).unwrap();
        let acm = AllowedCollisionMatrix::from_robot(&robot);

        // Out-of-bounds indices should return false
        assert!(!acm.is_allowed_by_index(&robot, 999, 0));
        assert!(!acm.is_allowed_by_index(&robot, 0, 999));
        assert!(!acm.is_allowed_by_index(&robot, 999, 999));
    }

    #[test]
    fn resolved_acm_branching_tree() {
        let robot = Robot::from_urdf_string(BRANCHING_URDF).unwrap();
        let resolved = ResolvedACM::from_robot(&robot);

        // 4 adjacent pairs
        assert_eq!(resolved.num_allowed(), 4);

        let base_idx = robot.link_index("base").unwrap();
        let left_idx = robot.link_index("left_arm").unwrap();
        let right_idx = robot.link_index("right_arm").unwrap();

        // Adjacent → allowed
        assert!(resolved.is_allowed(base_idx, left_idx));
        assert!(resolved.is_allowed(base_idx, right_idx));

        // Cross-branch → not allowed
        assert!(!resolved.is_allowed(left_idx, right_idx));

        // Same link → always allowed
        assert!(resolved.is_allowed(base_idx, base_idx));
    }

    #[test]
    fn acm_allowed_pairs_iterator() {
        let mut acm = AllowedCollisionMatrix::new();
        acm.allow("alpha", "beta");
        acm.allow("gamma", "delta");

        let pairs: Vec<(&str, &str)> = acm.allowed_pairs().collect();
        assert_eq!(pairs.len(), 2);

        // Both pairs present (order within pair is normalized: a <= b)
        let has_ab = pairs.iter().any(|&(a, b)| a == "alpha" && b == "beta");
        let has_gd = pairs.iter().any(|&(a, b)| a == "delta" && b == "gamma");
        assert!(has_ab, "Missing (alpha, beta)");
        assert!(has_gd, "Missing (delta, gamma)");
    }
}
