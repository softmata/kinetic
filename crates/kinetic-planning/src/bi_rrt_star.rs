//! Bidirectional RRT* (Jordan & Perez, 2013).
//!
//! Grows two RRT* trees — one from start, one from goal — and attempts to
//! connect them. Each tree independently performs rewiring for local optimality.
//! Faster convergence than unidirectional RRT* for most problems.
//!
//! # Algorithm
//!
//! 1. Initialize tree_a at start, tree_b at goal.
//! 2. Each iteration:
//!    a. Sample random config.
//!    b. Extend tree_a toward sample with RRT* (best parent + rewire).
//!    c. If extended, try to connect tree_b to the new node.
//!    d. If connected, compute total path cost. Keep best.
//!    e. Swap trees.
//! 3. Continue improving until timeout (anytime).

use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;

use kinetic_collision::{
    AllowedCollisionMatrix, CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig,
};
use kinetic_core::{Goal, KineticError, PlannerConfig, Result};
use kinetic_kinematics::{forward_kinematics_all, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;

use crate::rrt_star::RRTStarConfig;
use crate::shortcut::{self, CollisionChecker};
use crate::smooth;

/// Node in a BiRRT* tree.
#[derive(Debug, Clone)]
struct BiNode {
    joints: Vec<f64>,
    parent: Option<usize>,
    cost: f64,
}

/// A single tree in the BiRRT* algorithm.
#[derive(Debug)]
struct BiTree {
    nodes: Vec<BiNode>,
    dof: usize,
}

impl BiTree {
    fn new(root: Vec<f64>) -> Self {
        let dof = root.len();
        Self {
            nodes: vec![BiNode {
                joints: root,
                parent: None,
                cost: 0.0,
            }],
            dof,
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, joints: Vec<f64>, parent: usize, cost: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(BiNode {
            joints,
            parent: Some(parent),
            cost,
        });
        idx
    }

    fn nearest(&self, target: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist_sq = f64::INFINITY;
        for (i, node) in self.nodes.iter().enumerate() {
            let dist_sq = joint_distance_sq(&node.joints, target);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = i;
            }
        }
        best_idx
    }

    fn near(&self, target: &[f64], radius: f64) -> Vec<usize> {
        let radius_sq = radius * radius;
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| joint_distance_sq(&node.joints, target) <= radius_sq)
            .map(|(i, _)| i)
            .collect()
    }

    fn near_radius(&self, gamma: f64) -> f64 {
        let n = self.len() as f64;
        let d = self.dof as f64;
        if n <= 1.0 {
            return f64::INFINITY;
        }
        gamma * (n.ln() / n).powf(1.0 / d)
    }

    fn rewire(&mut self, node_idx: usize, new_parent: usize, new_cost: f64) {
        let delta = new_cost - self.nodes[node_idx].cost;
        self.nodes[node_idx].parent = Some(new_parent);
        self.nodes[node_idx].cost = new_cost;
        self.propagate_cost(node_idx, delta);
    }

    fn propagate_cost(&mut self, parent_idx: usize, delta: f64) {
        let children: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(parent_idx))
            .map(|(i, _)| i)
            .collect();
        for child in children {
            self.nodes[child].cost += delta;
            self.propagate_cost(child, delta);
        }
    }

    fn path_to_root(&self, mut idx: usize) -> Vec<Vec<f64>> {
        let mut path = Vec::new();
        loop {
            path.push(self.nodes[idx].joints.clone());
            match self.nodes[idx].parent {
                Some(p) => idx = p,
                None => break,
            }
        }
        path.reverse();
        path
    }
}

fn joint_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
}

fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    joint_distance_sq(a, b).sqrt()
}

/// Result from Bidirectional RRT*.
#[derive(Debug, Clone)]
pub struct BiRRTStarResult {
    pub waypoints: Vec<Vec<f64>>,
    pub planning_time: Duration,
    pub iterations: usize,
    pub tree_size: usize,
    pub path_cost: f64,
    pub rewire_count: usize,
}

/// Bidirectional RRT* planner.
pub struct BiRRTStar {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    config: RRTStarConfig,
}

impl CollisionChecker for BiRRTStar {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

impl BiRRTStar {
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        config: RRTStarConfig,
    ) -> Self {
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let acm = ResolvedACM::from_robot(&robot);
        Self {
            robot,
            chain,
            sphere_model,
            acm,
            environment,
            planner_config,
            config,
        }
    }

    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<BiRRTStarResult> {
        let start_time = Instant::now();

        let goal_configs = self.resolve_goal(goal)?;
        if goal_configs.is_empty() {
            return Err(KineticError::GoalUnreachable);
        }

        if self.is_in_collision(start) {
            return Err(KineticError::StartInCollision);
        }

        let mut last_err = KineticError::GoalUnreachable;

        for goal_joints in &goal_configs {
            if self.is_in_collision(goal_joints) {
                continue;
            }

            match self.plan_to_joints(start, goal_joints, start_time) {
                Ok(mut result) => {
                    if self.planner_config.shortcut_iterations > 0 {
                        result.waypoints = shortcut::shortcut(
                            &result.waypoints,
                            self,
                            self.planner_config.shortcut_iterations,
                            self.config.step_size,
                        );
                    }
                    if self.planner_config.smooth && result.waypoints.len() > 2 {
                        let num_output = result.waypoints.len() * 10;
                        let smoothed =
                            smooth::smooth_cubic_spline(&result.waypoints, num_output, Some(self));
                        if smoothed.c2_continuous {
                            result.waypoints = smoothed.waypoints;
                        }
                    }
                    result.planning_time = start_time.elapsed();
                    return Ok(result);
                }
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    fn plan_to_joints(
        &self,
        start: &[f64],
        goal: &[f64],
        start_time: Instant,
    ) -> Result<BiRRTStarResult> {
        let mut tree_a = BiTree::new(start.to_vec());
        let mut tree_b = BiTree::new(goal.to_vec());
        let mut rng = rand::thread_rng();
        let joint_limits = self.get_joint_limits();

        let mut best_path: Option<Vec<Vec<f64>>> = None;
        let mut best_cost = f64::INFINITY;
        let mut rewire_count: usize = 0;
        let mut first_solution_time: Option<Instant> = None;
        let mut total_iterations: usize = 0;

        for _iter in 0..self.planner_config.max_iterations {
            total_iterations += 1;

            if start_time.elapsed() > self.planner_config.timeout {
                break;
            }

            if let Some(first_time) = first_solution_time {
                if !self.config.anytime {
                    break;
                }
                if let Some(imp_timeout) = self.config.improvement_timeout {
                    if first_time.elapsed() > imp_timeout {
                        break;
                    }
                }
            }

            // Sample
            let sample = self.random_sample(&joint_limits, &mut rng);

            // Extend tree_a toward sample with RRT* logic
            let nearest_a = tree_a.nearest(&sample);
            let new_joints = self.steer(&tree_a.nodes[nearest_a].joints, &sample);

            if self.is_in_collision(&new_joints) {
                std::mem::swap(&mut tree_a, &mut tree_b);
                continue;
            }

            // Choose best parent from near nodes
            let radius = tree_a
                .near_radius(self.config.gamma)
                .min(self.config.step_size * 3.0);
            let near_a = tree_a.near(&new_joints, radius);
            let (best_parent, parent_cost) =
                self.choose_best_parent(&tree_a, &near_a, nearest_a, &new_joints);

            if !self.is_edge_free(&tree_a.nodes[best_parent].joints, &new_joints) {
                std::mem::swap(&mut tree_a, &mut tree_b);
                continue;
            }

            let new_a = tree_a.add(new_joints.clone(), best_parent, parent_cost);

            // Rewire tree_a
            for &near_idx in &near_a {
                if near_idx == best_parent {
                    continue;
                }
                let cost_through_new =
                    parent_cost + joint_distance(&new_joints, &tree_a.nodes[near_idx].joints);
                if cost_through_new < tree_a.nodes[near_idx].cost
                    && self.is_edge_free(&new_joints, &tree_a.nodes[near_idx].joints)
                {
                    tree_a.rewire(near_idx, new_a, cost_through_new);
                    rewire_count += 1;
                }
            }

            // Try to connect tree_b to the new node via greedy extend
            let nearest_b = tree_b.nearest(&new_joints);
            let connect_result = self.try_connect(&mut tree_b, nearest_b, &new_joints);

            if let Some(connect_idx) = connect_result {
                // Connection! Compute total cost
                let dist_b = joint_distance(
                    &tree_b.nodes[connect_idx].joints,
                    &tree_a.nodes[new_a].joints,
                );
                let total_cost = tree_a.nodes[new_a].cost + dist_b + tree_b.nodes[connect_idx].cost;

                if total_cost < best_cost {
                    // Extract combined path
                    let path_a = tree_a.path_to_root(new_a);
                    let mut path_b = tree_b.path_to_root(connect_idx);
                    path_b.reverse();

                    let mut full_path = path_a;
                    // Skip duplicate at connection point
                    if !path_b.is_empty() {
                        if let Some(last) = full_path.last() {
                            if joint_distance(last, &path_b[0]) < 1e-8 {
                                path_b.remove(0);
                            }
                        }
                    }
                    full_path.extend(path_b);

                    best_cost = total_cost;
                    best_path = Some(full_path);
                    if first_solution_time.is_none() {
                        first_solution_time = Some(Instant::now());
                    }
                }
            }

            // Swap trees
            std::mem::swap(&mut tree_a, &mut tree_b);
        }

        match best_path {
            Some(waypoints) => Ok(BiRRTStarResult {
                waypoints,
                planning_time: start_time.elapsed(),
                iterations: total_iterations,
                tree_size: tree_a.len() + tree_b.len(),
                path_cost: best_cost,
                rewire_count,
            }),
            None => Err(KineticError::PlanningTimeout {
                elapsed: start_time.elapsed(),
                iterations: total_iterations,
            }),
        }
    }

    /// Greedily extend tree toward target, returning the final node index if reached.
    fn try_connect(&self, tree: &mut BiTree, start_idx: usize, target: &[f64]) -> Option<usize> {
        let mut current_idx = start_idx;

        for _ in 0..100 {
            // cap iterations to avoid infinite loop
            let current = &tree.nodes[current_idx].joints;
            let dist = joint_distance(current, target);

            if dist < 1e-8 {
                return Some(current_idx); // reached
            }

            let new_joints = self.steer(current, target);

            if self.is_in_collision(&new_joints) {
                return None; // trapped
            }

            let edge_len = joint_distance(current, &new_joints);
            let new_cost = tree.nodes[current_idx].cost + edge_len;
            current_idx = tree.add(new_joints.clone(), current_idx, new_cost);

            if joint_distance(&new_joints, target) < 1e-8 {
                return Some(current_idx);
            }

            // If we didn't make progress (step was same as current), stop
            if edge_len < 1e-10 {
                return None;
            }
        }

        // Got close enough? Check final distance
        let final_dist = joint_distance(&tree.nodes[current_idx].joints, target);
        if final_dist <= self.config.step_size {
            Some(current_idx)
        } else {
            None
        }
    }

    fn steer(&self, from: &[f64], to: &[f64]) -> Vec<f64> {
        let dist = joint_distance(from, to);
        if dist <= self.config.step_size {
            return to.to_vec();
        }
        let ratio = self.config.step_size / dist;
        from.iter()
            .zip(to.iter())
            .map(|(a, b)| a + ratio * (b - a))
            .collect()
    }

    fn choose_best_parent(
        &self,
        tree: &BiTree,
        near: &[usize],
        default: usize,
        new_joints: &[f64],
    ) -> (usize, f64) {
        let mut best = default;
        let mut best_cost =
            tree.nodes[default].cost + joint_distance(&tree.nodes[default].joints, new_joints);

        for &idx in near {
            let cost =
                tree.nodes[idx].cost + joint_distance(&tree.nodes[idx].joints, new_joints);
            if cost < best_cost && self.is_edge_free(&tree.nodes[idx].joints, new_joints) {
                best = idx;
                best_cost = cost;
            }
        }
        (best, best_cost)
    }

    fn is_edge_free(&self, from: &[f64], to: &[f64]) -> bool {
        let dist = joint_distance(from, to);
        let n = (dist / self.config.step_size).ceil() as usize;
        if n == 0 {
            return true;
        }
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let interp: Vec<f64> = from
                .iter()
                .zip(to.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect();
            if self.is_in_collision(&interp) {
                return false;
            }
        }
        true
    }

    fn is_in_collision(&self, joints: &[f64]) -> bool {
        let link_poses = match forward_kinematics_all(&self.robot, &self.chain, joints) {
            Ok(poses) => poses,
            Err(_) => return true,
        };
        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);
        if self
            .environment
            .check_collision_with_margin(&runtime.world, self.planner_config.collision_margin)
        {
            return true;
        }
        let skip_pairs = self.acm.to_skip_pairs();
        runtime.self_collision_with_margin(&skip_pairs, self.planner_config.collision_margin)
    }

    fn resolve_goal(&self, goal: &Goal) -> Result<Vec<Vec<f64>>> {
        match goal {
            Goal::Joints(jv) => Ok(vec![jv.0.clone()]),
            Goal::Pose(target_pose) => {
                let ik_config = IKConfig {
                    num_restarts: 8,
                    ..Default::default()
                };
                match solve_ik(&self.robot, &self.chain, target_pose, &ik_config) {
                    Ok(sol) => Ok(vec![sol.joints]),
                    Err(_) => Err(KineticError::NoIKSolution),
                }
            }
            Goal::Named(name) => Err(KineticError::NamedConfigNotFound(name.clone())),
            Goal::Relative(_) => Err(KineticError::UnsupportedGoal(
                "Relative goals not supported in BiRRT* planner".into(),
            )),
        }
    }

    fn random_sample(&self, limits: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
        limits
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo..=hi))
            .collect()
    }

    fn get_joint_limits(&self) -> Vec<(f64, f64)> {
        self.chain
            .active_joints
            .iter()
            .map(|&joint_idx| {
                if let Some(limits) = &self.robot.joints[joint_idx].limits {
                    (limits.lower, limits.upper)
                } else {
                    (-std::f64::consts::PI, std::f64::consts::PI)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::capt::AABB;
    use kinetic_core::JointValues;

    fn setup_bi_rrt_star() -> BiRRTStar {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        BiRRTStar::new(
            robot,
            chain,
            env,
            config,
            RRTStarConfig {
                anytime: false,
                ..Default::default()
            },
        )
    }

    #[test]
    fn bi_rrt_star_free_space() {
        let planner = setup_bi_rrt_star();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.path_cost > 0.0);
    }

    #[test]
    fn bi_rrt_star_path_collision_free() {
        let planner = setup_bi_rrt_star();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!planner.is_in_collision(wp), "Waypoint {} in collision", i);
        }
    }

    #[test]
    fn bi_rrt_star_start_in_collision() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let mut obs = kinetic_collision::SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 5.0, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));

        let planner = BiRRTStar::new(
            robot,
            chain,
            env,
            PlannerConfig::default(),
            RRTStarConfig::default(),
        );

        let start = vec![0.0; 6];
        let goal = Goal::Joints(JointValues(vec![1.0, -1.0, 0.5, 0.0, 0.0, 0.0]));

        match planner.plan(&start, &goal) {
            Err(KineticError::StartInCollision) => {}
            other => panic!("Expected StartInCollision, got {:?}", other),
        }
    }
}
