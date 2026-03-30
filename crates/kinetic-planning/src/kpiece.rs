//! KPIECE (Kinodynamic Planning by Interior-Exterior Cell Exploration)
//! (Sucan & Kavraki, 2009).
//!
//! Projects C-space onto a lower-dimensional discretization grid and tracks
//! which cells are "interior" (well-explored) vs "exterior" (frontier).
//! Expansion is biased toward exterior cells, enabling efficient exploration
//! of high-dimensional spaces.
//!
//! # Key idea
//!
//! Full C-space is too large to discretize. KPIECE projects configurations
//! onto a low-dimensional grid (default: first 2-3 joints) and uses cell
//! occupancy to guide exploration. Cells with few samples relative to their
//! neighbors are "exterior" (frontier) and get priority.
//!
//! # Algorithm
//!
//! 1. Initialize tree rooted at start. Project start into grid cell.
//! 2. Each iteration:
//!    a. Select a cell weighted by `importance = exterior_score / (1 + coverage)`.
//!    b. Pick a random node from that cell.
//!    c. Expand by random perturbation.
//!    d. If collision-free, add node, update cell grid.
//!    e. Check if goal is reachable from new node.
//! 3. Return path when goal reached.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;

use kinetic_collision::{
    AllowedCollisionMatrix, CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig,
};
use kinetic_core::{Goal, KineticError, PlannerConfig, Result};
use kinetic_kinematics::{forward_kinematics_all, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;

use crate::shortcut::{self, CollisionChecker};
use crate::smooth;

/// KPIECE-specific configuration.
#[derive(Debug, Clone)]
pub struct KPIECEConfig {
    /// Expansion range from selected node (default: 0.3 radians).
    pub expansion_range: f64,
    /// Step size for connect/edge collision checking (default: 0.1 radians).
    pub step_size: f64,
    /// Grid cell size in each projected dimension (default: 0.5 radians).
    pub cell_size: f64,
    /// Number of projection dimensions (default: 2, max: DOF).
    /// Projects onto the first N joint dimensions.
    pub projection_dims: usize,
    /// Goal bias probability (default: 0.05).
    pub goal_bias: f64,
    /// Threshold: a cell is "interior" when it has >= this many nodes (default: 5).
    pub interior_threshold: usize,
}

impl Default for KPIECEConfig {
    fn default() -> Self {
        Self {
            expansion_range: 0.3,
            step_size: 0.1,
            cell_size: 0.5,
            projection_dims: 2,
            goal_bias: 0.05,
            interior_threshold: 5,
        }
    }
}

/// A cell in the discretized projection grid.
#[derive(Debug, Clone)]
struct GridCell {
    /// Indices of tree nodes that project into this cell.
    node_indices: Vec<usize>,
    /// Number of neighboring cells that are occupied.
    occupied_neighbors: usize,
    /// Number of total possible neighbors (2*d for d projection dims).
    total_neighbors: usize,
}

impl GridCell {
    fn new(total_neighbors: usize) -> Self {
        Self {
            node_indices: Vec::new(),
            occupied_neighbors: 0,
            total_neighbors,
        }
    }

    fn count(&self) -> usize {
        self.node_indices.len()
    }

    /// Importance score: exterior cells with few nodes get high importance.
    /// exterior_ratio = 1 - (occupied_neighbors / total_neighbors).
    /// importance = exterior_ratio / (1 + count).
    fn importance(&self, interior_threshold: usize) -> f64 {
        let exterior_ratio = if self.total_neighbors > 0 {
            1.0 - (self.occupied_neighbors as f64 / self.total_neighbors as f64)
        } else {
            1.0
        };

        // Cells below interior threshold get a boost
        let coverage_penalty = if self.count() >= interior_threshold {
            self.count() as f64
        } else {
            self.count() as f64 * 0.5
        };

        (1.0 + exterior_ratio) / (1.0 + coverage_penalty)
    }
}

/// A node in the KPIECE tree.
#[derive(Debug, Clone)]
struct KPIECENode {
    joints: Vec<f64>,
    parent: Option<usize>,
    /// Grid cell key this node projects into.
    #[allow(dead_code)]
    cell_key: Vec<i64>,
}

/// KPIECE planner result.
#[derive(Debug, Clone)]
pub struct KPIECEResult {
    pub waypoints: Vec<Vec<f64>>,
    pub planning_time: Duration,
    pub iterations: usize,
    pub tree_size: usize,
    /// Number of cells explored.
    pub cells_explored: usize,
}

/// KPIECE planner.
pub struct KPIECE {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    kpiece_config: KPIECEConfig,
}

impl CollisionChecker for KPIECE {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.check_collision(joints)
    }
}

impl KPIECE {
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        kpiece_config: KPIECEConfig,
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
            kpiece_config,
        }
    }

    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Plan using grid-projected exploration.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<KPIECEResult> {
        let start_time = Instant::now();
        let joint_limits = self.get_joint_limits();
        let mut rng = rand::thread_rng();

        let goal_joints = self.resolve_goal(goal)?;

        if self.check_collision(start) {
            return Err(KineticError::StartInCollision);
        }
        if self.check_collision(&goal_joints) {
            return Err(KineticError::GoalInCollision);
        }

        let proj_dims = self.kpiece_config.projection_dims.min(start.len());

        // Initialize tree and grid
        let start_key = self.project(start, proj_dims);
        let mut nodes = vec![KPIECENode {
            joints: start.to_vec(),
            parent: None,
            cell_key: start_key.clone(),
        }];

        let total_neighbors = 2 * proj_dims;
        let mut grid: HashMap<Vec<i64>, GridCell> = HashMap::new();
        let start_cell = grid
            .entry(start_key)
            .or_insert_with(|| GridCell::new(total_neighbors));
        start_cell.node_indices.push(0);

        let goal_tolerance = self.kpiece_config.step_size * 2.0;

        for iteration in 0..self.planner_config.max_iterations {
            if start_time.elapsed() >= self.planner_config.timeout {
                break;
            }

            // Update neighbor counts periodically
            if iteration % 50 == 0 {
                self.update_neighbor_counts(&mut grid, proj_dims);
            }

            // Select a cell weighted by importance
            let selected_cell_key = self.select_cell(&grid, &mut rng);
            let selected_cell_key = match selected_cell_key {
                Some(k) => k,
                None => continue,
            };

            // Pick a random node from the selected cell
            let node_idx = {
                let cell = &grid[&selected_cell_key];
                cell.node_indices[rng.gen_range(0..cell.node_indices.len())]
            };
            let selected_joints = nodes[node_idx].joints.clone();

            // Expand: random perturbation or goal bias
            let new_joints: Vec<f64> = if rng.gen::<f64>() < self.kpiece_config.goal_bias {
                let dist = joint_distance(&selected_joints, &goal_joints);
                if dist < 1e-10 {
                    goal_joints.clone()
                } else {
                    let t = (self.kpiece_config.expansion_range / dist).min(1.0);
                    selected_joints
                        .iter()
                        .zip(goal_joints.iter())
                        .map(|(a, b)| a + t * (b - a))
                        .collect()
                }
            } else {
                selected_joints
                    .iter()
                    .zip(joint_limits.iter())
                    .map(|(&s, &(lo, hi))| {
                        let delta = rng.gen_range(
                            -self.kpiece_config.expansion_range..=self.kpiece_config.expansion_range,
                        );
                        (s + delta).clamp(lo, hi)
                    })
                    .collect()
            };

            if self.check_collision(&new_joints) {
                continue;
            }

            // Add node
            let new_key = self.project(&new_joints, proj_dims);
            let new_idx = nodes.len();
            nodes.push(KPIECENode {
                joints: new_joints.clone(),
                parent: Some(node_idx),
                cell_key: new_key.clone(),
            });

            let cell = grid
                .entry(new_key)
                .or_insert_with(|| GridCell::new(total_neighbors));
            cell.node_indices.push(new_idx);

            // Check if goal is reachable
            let dist_to_goal = joint_distance(&new_joints, &goal_joints);
            if dist_to_goal <= goal_tolerance
                || self.can_connect(&new_joints, &goal_joints)
            {
                // Build path
                let mut path = vec![goal_joints.clone()];
                let mut idx = new_idx;
                loop {
                    path.push(nodes[idx].joints.clone());
                    match nodes[idx].parent {
                        Some(p) => idx = p,
                        None => break,
                    }
                }
                path.reverse();

                let mut waypoints = path;

                // Post-process
                if self.planner_config.shortcut_iterations > 0 {
                    waypoints = shortcut::shortcut(
                        &waypoints,
                        self,
                        self.planner_config.shortcut_iterations,
                        self.kpiece_config.step_size,
                    );
                }
                if self.planner_config.smooth && waypoints.len() > 2 {
                    let n = waypoints.len() * 10;
                    let smoothed = smooth::smooth_cubic_spline(&waypoints, n, Some(self));
                    if smoothed.c2_continuous {
                        waypoints = smoothed.waypoints;
                    }
                }

                return Ok(KPIECEResult {
                    waypoints,
                    planning_time: start_time.elapsed(),
                    iterations: iteration + 1,
                    tree_size: nodes.len(),
                    cells_explored: grid.len(),
                });
            }
        }

        Err(KineticError::PlanningFailed(
            "KPIECE: timeout or iteration limit reached".into(),
        ))
    }

    /// Project a configuration onto the grid: discretize first N joints.
    fn project(&self, joints: &[f64], dims: usize) -> Vec<i64> {
        joints
            .iter()
            .take(dims)
            .map(|&v| (v / self.kpiece_config.cell_size).floor() as i64)
            .collect()
    }

    /// Select a cell weighted by importance score.
    fn select_cell(&self, grid: &HashMap<Vec<i64>, GridCell>, rng: &mut impl Rng) -> Option<Vec<i64>> {
        if grid.is_empty() {
            return None;
        }

        let threshold = self.kpiece_config.interior_threshold;
        let entries: Vec<(&Vec<i64>, f64)> = grid
            .iter()
            .filter(|(_, cell)| !cell.node_indices.is_empty())
            .map(|(key, cell)| (key, cell.importance(threshold)))
            .collect();

        if entries.is_empty() {
            return None;
        }

        let total: f64 = entries.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return Some(entries[0].0.clone());
        }

        let mut r = rng.gen::<f64>() * total;
        for (key, w) in &entries {
            r -= w;
            if r <= 0.0 {
                return Some((*key).clone());
            }
        }
        Some(entries.last().unwrap().0.clone())
    }

    /// Update neighbor counts for all cells.
    fn update_neighbor_counts(&self, grid: &mut HashMap<Vec<i64>, GridCell>, dims: usize) {
        let keys: Vec<Vec<i64>> = grid.keys().cloned().collect();
        for key in &keys {
            let mut occupied = 0;
            for d in 0..dims {
                for delta in [-1i64, 1] {
                    let mut neighbor_key = key.clone();
                    neighbor_key[d] += delta;
                    if grid.contains_key(&neighbor_key) {
                        occupied += 1;
                    }
                }
            }
            if let Some(cell) = grid.get_mut(key) {
                cell.occupied_neighbors = occupied;
                cell.total_neighbors = 2 * dims;
            }
        }
    }

    /// Check if a straight-line path between two configs is collision-free.
    fn can_connect(&self, from: &[f64], to: &[f64]) -> bool {
        let dist = joint_distance(from, to);
        let n = (dist / self.kpiece_config.step_size).ceil() as usize;
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
            if self.check_collision(&interp) {
                return false;
            }
        }
        true
    }

    fn check_collision(&self, joints: &[f64]) -> bool {
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

    fn resolve_goal(&self, goal: &Goal) -> Result<Vec<f64>> {
        match goal {
            Goal::Joints(jv) => Ok(jv.0.clone()),
            Goal::Pose(target_pose) => {
                let ik_config = IKConfig {
                    num_restarts: 8,
                    ..Default::default()
                };
                match solve_ik(&self.robot, &self.chain, target_pose, &ik_config) {
                    Ok(sol) => Ok(sol.joints),
                    Err(_) => Err(KineticError::NoIKSolution),
                }
            }
            Goal::Named(name) => Err(KineticError::NamedConfigNotFound(name.clone())),
            Goal::Relative(_) => Err(KineticError::UnsupportedGoal(
                "Relative goals not supported in KPIECE".into(),
            )),
        }
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

fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::capt::AABB;
    use kinetic_core::JointValues;

    fn setup_kpiece() -> KPIECE {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(30),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        KPIECE::new(robot, chain, env, config, KPIECEConfig::default())
    }

    #[test]
    fn kpiece_plan_free_space() {
        let planner = setup_kpiece();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.iterations > 0);
        assert!(result.cells_explored >= 1);
    }

    #[test]
    fn kpiece_path_collision_free() {
        let planner = setup_kpiece();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!planner.check_collision(wp), "Waypoint {} in collision", i);
        }
    }

    #[test]
    fn kpiece_cell_projection() {
        let planner = setup_kpiece();
        // cell_size=0.5, so joint value 1.3 → floor(1.3/0.5) = floor(2.6) = 2
        let key = planner.project(&[1.3, -0.7, 0.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(key, vec![2, -2]);
    }

    #[test]
    fn kpiece_cell_importance() {
        let mut cell = GridCell::new(4);
        // Empty frontier cell with no occupied neighbors → high importance
        let imp_empty = cell.importance(5);

        cell.node_indices = vec![0, 1, 2, 3, 4, 5];
        cell.occupied_neighbors = 4;
        let imp_interior = cell.importance(5);

        assert!(
            imp_empty > imp_interior,
            "Frontier cell should have higher importance: {} vs {}",
            imp_empty,
            imp_interior
        );
    }

    #[test]
    fn kpiece_config_defaults() {
        let config = KPIECEConfig::default();
        assert_eq!(config.expansion_range, 0.3);
        assert_eq!(config.cell_size, 0.5);
        assert_eq!(config.projection_dims, 2);
        assert_eq!(config.interior_threshold, 5);
    }

    #[test]
    fn kpiece_larger_distance() {
        let planner = setup_kpiece();
        let start = vec![0.0, -1.5, 1.0, 0.0, 0.5, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.5, -0.3, -0.5, 1.0, -0.5, 1.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.cells_explored >= 1);
    }
}
