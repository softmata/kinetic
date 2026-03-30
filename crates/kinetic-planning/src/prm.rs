//! PRM (Probabilistic Roadmap) planner (Kavraki et al., 1996).
//!
//! Multi-query planner: builds a roadmap of collision-free configurations
//! connected by collision-free edges. Queries are answered via A* search
//! on the roadmap. The same roadmap can be reused for multiple start/goal pairs.
//!
//! # Two phases
//!
//! 1. **Construction**: Sample N random collision-free configs, connect each to
//!    K nearest neighbors if the edge is collision-free. Store as adjacency list.
//! 2. **Query**: Connect start and goal to roadmap, run A* with Euclidean heuristic.

use std::collections::BinaryHeap;
use std::io::{self, Read as IoRead, Write as IoWrite};
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

/// Connection strategy for the PRM roadmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionStrategy {
    /// Fixed K-nearest neighbor connections (classic PRM).
    KNearest(usize),
    /// Adaptive K-nearest for asymptotic optimality (PRM*, OMPL-style).
    ///
    /// `k = ceil(k_multiplier * e * (1 + 1/d) * ln(n))` where:
    /// - `n` = number of nodes in the roadmap
    /// - `d` = C-space dimension (DOF)
    /// - `k_multiplier` = tuning constant (default: 1.0)
    ///
    /// This is equivalent to the radius-based PRM* formulation but avoids
    /// needing to know C-space volume. As samples increase, connectivity
    /// grows logarithmically, ensuring asymptotic optimality
    /// (Karaman & Frazzoli, 2011; OMPL implementation).
    AdaptiveK {
        /// Multiplier for the k_PRM* formula. Higher values = more connections.
        /// Default: 1.0 (theoretical minimum for optimality).
        k_multiplier: f64,
    },
    /// Radius-based connections for asymptotic optimality (PRM*).
    ///
    /// Connection radius: `r = gamma * (log(n)/n)^(1/d)`.
    /// Requires gamma to be calibrated for the C-space volume.
    Radius {
        /// Gamma constant. Must exceed the dimension-dependent threshold.
        gamma: f64,
    },
}

/// PRM-specific configuration.
#[derive(Debug, Clone)]
pub struct PRMConfig {
    /// Number of samples in the roadmap (default: 500).
    pub num_samples: usize,
    /// Connection strategy (default: KNearest(10)).
    ///
    /// Use `ConnectionStrategy::KNearest(k)` for classic PRM with fixed k neighbors.
    /// Use `ConnectionStrategy::Radius { gamma }` for PRM* with asymptotically optimal
    /// connectivity.
    pub connection: ConnectionStrategy,
    /// Step size for edge collision checking (default: 0.05 radians).
    pub edge_step: f64,
    /// Whether to use lazy collision checking (defer edge checks to query time).
    pub lazy: bool,
}

impl PRMConfig {
    /// Create a PRM* configuration with adaptive K-nearest connections.
    ///
    /// Uses `k = ceil(k_multiplier * e * (1 + 1/d) * ln(n))` which ensures
    /// asymptotic optimality without needing C-space volume calibration.
    pub fn prm_star(num_samples: usize) -> Self {
        Self {
            num_samples,
            connection: ConnectionStrategy::AdaptiveK { k_multiplier: 1.0 },
            edge_step: 0.05,
            lazy: false,
        }
    }

    /// Create a PRM* configuration with radius-based connections.
    ///
    /// Requires `gamma` to be calibrated for the C-space volume.
    /// For most use cases, prefer [`prm_star()`] which auto-adapts.
    pub fn prm_star_radius(num_samples: usize, gamma: f64) -> Self {
        Self {
            num_samples,
            connection: ConnectionStrategy::Radius { gamma },
            edge_step: 0.05,
            lazy: false,
        }
    }
}

impl Default for PRMConfig {
    fn default() -> Self {
        Self {
            num_samples: 500,
            connection: ConnectionStrategy::KNearest(10),
            edge_step: 0.05,
            lazy: false,
        }
    }
}

/// A node in the PRM roadmap.
#[derive(Debug, Clone)]
struct PRMNode {
    joints: Vec<f64>,
    /// Adjacency list: (neighbor_index, edge_cost).
    edges: Vec<(usize, f64)>,
}

/// The PRM roadmap graph.
#[derive(Debug)]
pub struct Roadmap {
    nodes: Vec<PRMNode>,
    dof: usize,
    construction_time: Duration,
}

impl Roadmap {
    /// Number of nodes in the roadmap.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the roadmap.
    pub fn num_edges(&self) -> usize {
        self.nodes.iter().map(|n| n.edges.len()).sum::<usize>() / 2
    }

    /// Construction time.
    pub fn construction_time(&self) -> Duration {
        self.construction_time
    }

    /// Save the roadmap to a writer in compact binary format.
    ///
    /// Format: magic(4) + version(u32) + scene_hash(u64) + dof(u32) + num_nodes(u32)
    ///         + [joints: f64 * dof] * num_nodes
    ///         + [num_edges(u32) + [(neighbor: u32, cost: f64)] * num_edges] * num_nodes
    pub fn save<W: IoWrite>(&self, writer: &mut W, scene_hash: u64) -> io::Result<()> {
        // Magic bytes: "KPRM"
        writer.write_all(b"KPRM")?;
        // Version 1
        writer.write_all(&1u32.to_le_bytes())?;
        // Scene hash for invalidation
        writer.write_all(&scene_hash.to_le_bytes())?;
        // DOF
        writer.write_all(&(self.dof as u32).to_le_bytes())?;
        // Number of nodes
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;

        // Node joints (flat array)
        for node in &self.nodes {
            for &v in &node.joints {
                writer.write_all(&v.to_le_bytes())?;
            }
        }

        // Edge adjacency lists
        for node in &self.nodes {
            writer.write_all(&(node.edges.len() as u32).to_le_bytes())?;
            for &(neighbor, cost) in &node.edges {
                writer.write_all(&(neighbor as u32).to_le_bytes())?;
                writer.write_all(&cost.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Save the roadmap to a file path.
    pub fn save_to_file(&self, path: &std::path::Path, scene_hash: u64) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.save(&mut file, scene_hash)
    }

    /// Load a roadmap from a reader. Returns `(roadmap, scene_hash)`.
    ///
    /// The caller should compare the scene_hash against the current environment
    /// to decide whether to revalidate or discard the roadmap.
    pub fn load<R: IoRead>(reader: &mut R) -> io::Result<(Self, u64)> {
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"KPRM" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a KPRM roadmap file"));
        }

        // Version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported KPRM version: {}", version),
            ));
        }

        // Scene hash
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let scene_hash = u64::from_le_bytes(buf8);

        // DOF
        reader.read_exact(&mut buf4)?;
        let dof = u32::from_le_bytes(buf4) as usize;

        // Number of nodes
        reader.read_exact(&mut buf4)?;
        let num_nodes = u32::from_le_bytes(buf4) as usize;

        // Read joints
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut joints = Vec::with_capacity(dof);
            for _ in 0..dof {
                reader.read_exact(&mut buf8)?;
                joints.push(f64::from_le_bytes(buf8));
            }
            nodes.push(PRMNode {
                joints,
                edges: Vec::new(),
            });
        }

        // Read edges
        for i in 0..num_nodes {
            reader.read_exact(&mut buf4)?;
            let num_edges = u32::from_le_bytes(buf4) as usize;
            let mut edges = Vec::with_capacity(num_edges);
            for _ in 0..num_edges {
                reader.read_exact(&mut buf4)?;
                let neighbor = u32::from_le_bytes(buf4) as usize;
                reader.read_exact(&mut buf8)?;
                let cost = f64::from_le_bytes(buf8);
                edges.push((neighbor, cost));
            }
            nodes[i].edges = edges;
        }

        let roadmap = Roadmap {
            nodes,
            dof,
            construction_time: Duration::ZERO,
        };

        Ok((roadmap, scene_hash))
    }

    /// Load a roadmap from a file path. Returns `(roadmap, scene_hash)`.
    pub fn load_from_file(path: &std::path::Path) -> io::Result<(Self, u64)> {
        let mut file = std::fs::File::open(path)?;
        Self::load(&mut file)
    }
}

/// Result from a PRM query.
#[derive(Debug, Clone)]
pub struct PRMResult {
    pub waypoints: Vec<Vec<f64>>,
    pub planning_time: Duration,
    pub path_cost: f64,
    pub nodes_expanded: usize,
}

/// Stirling-based approximation of the Gamma function for positive values.
#[allow(dead_code)]
fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Use exact values for small integers and half-integers
    if (x - 1.0).abs() < 1e-10 {
        return 1.0;
    }
    if (x - 2.0).abs() < 1e-10 {
        return 1.0;
    }
    if (x - 3.0).abs() < 1e-10 {
        return 2.0;
    }
    if (x - 4.0).abs() < 1e-10 {
        return 6.0;
    }
    // Stirling's approximation: Gamma(x) ≈ sqrt(2*pi/x) * (x/e)^x
    let e = std::f64::consts::E;
    (2.0 * std::f64::consts::PI / x).sqrt() * (x / e).powf(x)
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

/// A* search node for the priority queue.
#[derive(Debug, Clone)]
struct AStarNode {
    index: usize,
    g_cost: f64,
    f_cost: f64,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse ordering
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// PRM planner — multi-query planning via pre-computed roadmap.
pub struct PRM {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    prm_config: PRMConfig,
    roadmap: Option<Roadmap>,
}

impl CollisionChecker for PRM {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

impl PRM {
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        prm_config: PRMConfig,
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
            prm_config,
            roadmap: None,
        }
    }

    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Save the current roadmap to a file.
    ///
    /// The `scene_hash` should represent the current collision environment.
    /// When loading, compare scene hashes to detect environment changes.
    pub fn save_roadmap(&self, path: &std::path::Path, scene_hash: u64) -> io::Result<()> {
        match &self.roadmap {
            Some(roadmap) => roadmap.save_to_file(path, scene_hash),
            None => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "No roadmap to save. Call build_roadmap() first.",
            )),
        }
    }

    /// Load a roadmap from a file.
    ///
    /// Returns the scene hash stored with the roadmap. Compare against the
    /// current environment to decide if the roadmap is still valid.
    ///
    /// If the scene has changed, call `revalidate_roadmap()` to re-check edges,
    /// or discard and rebuild.
    pub fn load_roadmap(&mut self, path: &std::path::Path) -> io::Result<u64> {
        let (roadmap, scene_hash) = Roadmap::load_from_file(path)?;
        if roadmap.dof != self.chain.active_joints.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Roadmap DOF ({}) doesn't match robot chain DOF ({})",
                    roadmap.dof,
                    self.chain.active_joints.len()
                ),
            ));
        }
        self.roadmap = Some(roadmap);
        Ok(scene_hash)
    }

    /// Re-validate all edges in the loaded roadmap against the current collision environment.
    ///
    /// Removes edges that are now in collision. Use after loading a roadmap
    /// when the scene has changed.
    pub fn revalidate_roadmap(&mut self) {
        let roadmap = match self.roadmap.as_ref() {
            Some(r) => r,
            None => return,
        };

        // Collect all edges as (node_idx, edge_idx, from_joints, to_joints)
        let mut edges_to_check: Vec<(usize, usize, Vec<f64>, Vec<f64>)> = Vec::new();

        for (i, node) in roadmap.nodes.iter().enumerate() {
            for (edge_idx, &(neighbor, _)) in node.edges.iter().enumerate() {
                // Only check each undirected edge once (i < neighbor)
                if i < neighbor {
                    edges_to_check.push((
                        i,
                        edge_idx,
                        node.joints.clone(),
                        roadmap.nodes[neighbor].joints.clone(),
                    ));
                }
            }
        }

        // Now check collisions (no borrow on self.roadmap)
        let mut invalid_pairs: Vec<(usize, usize)> = Vec::new();
        for (node_idx, edge_idx, from, to) in &edges_to_check {
            if !self.is_edge_collision_free(from, to) {
                invalid_pairs.push((*node_idx, *edge_idx));
            }
        }

        // Build set of invalid (i, j) node pairs for bidirectional removal
        let roadmap = self.roadmap.as_mut().unwrap();
        let mut invalid_node_pairs: Vec<(usize, usize)> = Vec::new();
        for &(node_idx, edge_idx) in &invalid_pairs {
            let (neighbor, _) = roadmap.nodes[node_idx].edges[edge_idx];
            invalid_node_pairs.push((node_idx, neighbor));
        }

        // Remove edges in both directions
        for &(i, j) in &invalid_node_pairs {
            roadmap.nodes[i].edges.retain(|&(n, _)| n != j);
            roadmap.nodes[j].edges.retain(|&(n, _)| n != i);
        }
    }

    /// Build the roadmap. Must be called before `query()`.
    pub fn build_roadmap(&mut self) -> &Roadmap {
        let start_time = Instant::now();
        let joint_limits = self.get_joint_limits();
        let mut rng = rand::thread_rng();

        // Phase 1: Sample collision-free configurations
        let mut nodes: Vec<PRMNode> = Vec::with_capacity(self.prm_config.num_samples);

        while nodes.len() < self.prm_config.num_samples {
            let sample = self.random_sample(&joint_limits, &mut rng);
            if !self.is_in_collision(&sample) {
                nodes.push(PRMNode {
                    joints: sample,
                    edges: Vec::new(),
                });
            }
        }

        let dof = joint_limits.len();

        // Phase 2: Connect nodes using the configured strategy
        let n = nodes.len();
        for i in 0..n {
            let mut distances: Vec<(usize, f64)> = nodes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, node)| (j, joint_distance(&nodes[i].joints, &node.joints)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let candidates = self.select_candidates(&distances, n, dof);

            for &(j, dist) in candidates {
                if nodes[i].edges.iter().any(|&(neighbor, _)| neighbor == j) {
                    continue;
                }

                let edge_free = self.prm_config.lazy
                    || self.is_edge_collision_free(&nodes[i].joints, &nodes[j].joints);

                if edge_free {
                    nodes[i].edges.push((j, dist));
                    nodes[j].edges.push((i, dist));
                }
            }
        }

        self.roadmap = Some(Roadmap {
            nodes,
            dof,
            construction_time: start_time.elapsed(),
        });

        self.roadmap.as_ref().unwrap()
    }

    /// Query the roadmap for a path from start to goal.
    ///
    /// Connects start and goal to the roadmap, then runs A*.
    pub fn query(&self, start: &[f64], goal: &Goal) -> Result<PRMResult> {
        let start_time = Instant::now();

        let roadmap = self
            .roadmap
            .as_ref()
            .ok_or_else(|| KineticError::PlanningFailed("Roadmap not built. Call build_roadmap() first.".into()))?;

        let goal_configs = self.resolve_goal(goal)?;
        if goal_configs.is_empty() {
            return Err(KineticError::GoalUnreachable);
        }

        if self.is_in_collision(start) {
            return Err(KineticError::StartInCollision);
        }

        for goal_joints in &goal_configs {
            if self.is_in_collision(goal_joints) {
                continue;
            }

            // Connect start and goal to nearest roadmap nodes
            let start_neighbors = self.find_connectable_neighbors(start, roadmap);
            let goal_neighbors = self.find_connectable_neighbors(goal_joints, roadmap);

            if start_neighbors.is_empty() || goal_neighbors.is_empty() {
                continue;
            }

            // Run A* on the roadmap
            match self.astar(
                start,
                goal_joints,
                &start_neighbors,
                &goal_neighbors,
                roadmap,
            ) {
                Ok(mut result) => {
                    // Post-process
                    if self.planner_config.shortcut_iterations > 0 {
                        result.waypoints = shortcut::shortcut(
                            &result.waypoints,
                            self,
                            self.planner_config.shortcut_iterations,
                            self.prm_config.edge_step,
                        );
                    }
                    if self.planner_config.smooth && result.waypoints.len() > 2 {
                        let n = result.waypoints.len() * 10;
                        let smoothed =
                            smooth::smooth_cubic_spline(&result.waypoints, n, Some(self));
                        if smoothed.c2_continuous {
                            result.waypoints = smoothed.waypoints;
                        }
                    }
                    result.planning_time = start_time.elapsed();
                    return Ok(result);
                }
                Err(_) => continue,
            }
        }

        Err(KineticError::GoalUnreachable)
    }

    /// Plan using the roadmap (convenience: build if needed, then query).
    pub fn plan(&mut self, start: &[f64], goal: &Goal) -> Result<PRMResult> {
        if self.roadmap.is_none() {
            self.build_roadmap();
        }
        self.query(start, goal)
    }

    /// Find roadmap nodes connectable from a given configuration.
    fn find_connectable_neighbors(
        &self,
        config: &[f64],
        roadmap: &Roadmap,
    ) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = roadmap
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, joint_distance(config, &n.joints)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected = self.select_candidates(&distances, roadmap.nodes.len(), roadmap.dof);

        selected
            .iter()
            .filter(|&&(i, _)| {
                self.prm_config.lazy
                    || self.is_edge_collision_free(config, &roadmap.nodes[i].joints)
            })
            .copied()
            .collect()
    }

    /// A* search on the roadmap.
    ///
    /// `start_neighbors`: roadmap nodes connected to start (index, cost).
    /// `goal_neighbors`: roadmap nodes connected to goal (index, cost).
    fn astar(
        &self,
        start: &[f64],
        goal: &[f64],
        start_neighbors: &[(usize, f64)],
        goal_neighbors: &[(usize, f64)],
        roadmap: &Roadmap,
    ) -> Result<PRMResult> {
        let n = roadmap.nodes.len();
        let mut g_costs = vec![f64::INFINITY; n];
        let mut came_from = vec![None::<usize>; n];
        let mut closed = vec![false; n];
        let mut heap = BinaryHeap::new();
        let mut nodes_expanded: usize = 0;

        // Goal set for fast lookup
        let goal_set: std::collections::HashSet<usize> =
            goal_neighbors.iter().map(|&(i, _)| i).collect();

        // Initialize with start neighbors
        for &(idx, dist_from_start) in start_neighbors {
            if dist_from_start < g_costs[idx] {
                g_costs[idx] = dist_from_start;
                let h = joint_distance(&roadmap.nodes[idx].joints, goal);
                heap.push(AStarNode {
                    index: idx,
                    g_cost: dist_from_start,
                    f_cost: dist_from_start + h,
                });
            }
        }

        while let Some(current) = heap.pop() {
            if closed[current.index] {
                continue;
            }
            closed[current.index] = true;
            nodes_expanded += 1;

            // Check if we reached a goal neighbor
            if goal_set.contains(&current.index) {
                // Reconstruct path
                let mut path = vec![goal.to_vec()];

                let mut idx = current.index;
                loop {
                    path.push(roadmap.nodes[idx].joints.clone());
                    match came_from[idx] {
                        Some(prev) => idx = prev,
                        None => break,
                    }
                }
                path.push(start.to_vec());
                path.reverse();

                let goal_dist = joint_distance(&roadmap.nodes[current.index].joints, goal);
                return Ok(PRMResult {
                    waypoints: path,
                    planning_time: Duration::ZERO, // filled by caller
                    path_cost: current.g_cost + goal_dist,
                    nodes_expanded,
                });
            }

            // Expand neighbors
            for &(neighbor, edge_cost) in &roadmap.nodes[current.index].edges {
                if closed[neighbor] {
                    continue;
                }

                // Lazy collision check
                if self.prm_config.lazy
                    && !self.is_edge_collision_free(
                        &roadmap.nodes[current.index].joints,
                        &roadmap.nodes[neighbor].joints,
                    )
                {
                    continue;
                }

                let new_g = current.g_cost + edge_cost;
                if new_g < g_costs[neighbor] {
                    g_costs[neighbor] = new_g;
                    came_from[neighbor] = Some(current.index);
                    let h = joint_distance(&roadmap.nodes[neighbor].joints, goal);
                    heap.push(AStarNode {
                        index: neighbor,
                        g_cost: new_g,
                        f_cost: new_g + h,
                    });
                }
            }
        }

        Err(KineticError::PlanningFailed("No path found in roadmap".into()))
    }

    /// Add samples to existing roadmap incrementally.
    pub fn add_samples(&mut self, num_additional: usize) {
        let joint_limits = self.get_joint_limits();
        let mut rng = rand::thread_rng();

        // First collect new collision-free samples
        let mut new_samples = Vec::with_capacity(num_additional);
        while new_samples.len() < num_additional {
            let sample = self.random_sample(&joint_limits, &mut rng);
            if !self.is_in_collision(&sample) {
                new_samples.push(sample);
            }
        }

        let roadmap = match self.roadmap.as_mut() {
            Some(r) => r,
            None => return,
        };

        let existing_count = roadmap.nodes.len();

        for sample in new_samples {
            roadmap.nodes.push(PRMNode {
                joints: sample,
                edges: Vec::new(),
            });
        }

        // Collect candidate edges using connection strategy
        let total_nodes = roadmap.nodes.len();
        let dof = roadmap.dof;
        let connection = self.prm_config.connection;
        let lazy = self.prm_config.lazy;

        let mut candidates: Vec<(usize, usize, f64, Vec<f64>, Vec<f64>)> = Vec::new();

        for i in existing_count..total_nodes {
            let mut distances: Vec<(usize, f64)> = roadmap
                .nodes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, n)| (j, joint_distance(&roadmap.nodes[i].joints, &n.joints)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let count = match connection {
                ConnectionStrategy::KNearest(k) => k.min(distances.len()),
                ConnectionStrategy::AdaptiveK { k_multiplier } => {
                    Self::prm_star_k(k_multiplier, total_nodes, dof).min(distances.len())
                }
                ConnectionStrategy::Radius { gamma } => {
                    let radius = Self::connection_radius(gamma, total_nodes, dof);
                    distances.iter().take_while(|&&(_, d)| d <= radius).count()
                }
            };

            for &(j, dist) in distances.iter().take(count) {
                if roadmap.nodes[i].edges.iter().any(|&(n, _)| n == j) {
                    continue;
                }
                candidates.push((
                    i,
                    j,
                    dist,
                    roadmap.nodes[i].joints.clone(),
                    roadmap.nodes[j].joints.clone(),
                ));
            }
        }

        // Now we can drop the mutable borrow on roadmap and check collisions
        let _ = roadmap;

        let mut edges_to_add: Vec<(usize, usize, f64)> = Vec::new();
        for (i, j, dist, from, to) in candidates {
            let edge_free = lazy || self.is_edge_collision_free(&from, &to);
            if edge_free {
                edges_to_add.push((i, j, dist));
            }
        }

        // Re-borrow roadmap mutably to add edges
        let roadmap = self.roadmap.as_mut().unwrap();
        for (i, j, dist) in edges_to_add {
            roadmap.nodes[i].edges.push((j, dist));
            roadmap.nodes[j].edges.push((i, dist));
        }
    }

    /// Select candidate neighbors based on the connection strategy.
    ///
    /// For KNearest: returns the first `k` entries from sorted distances.
    /// For AdaptiveK (PRM*): returns `k = ceil(m * e * (1+1/d) * ln(n))` nearest.
    /// For Radius (PRM*): returns all entries within `r = gamma * (log(n)/n)^(1/d)`.
    fn select_candidates<'a>(
        &self,
        sorted_distances: &'a [(usize, f64)],
        num_nodes: usize,
        dof: usize,
    ) -> &'a [(usize, f64)] {
        match self.prm_config.connection {
            ConnectionStrategy::KNearest(k) => {
                let k = k.min(sorted_distances.len());
                &sorted_distances[..k]
            }
            ConnectionStrategy::AdaptiveK { k_multiplier } => {
                let k = Self::prm_star_k(k_multiplier, num_nodes, dof);
                let k = k.min(sorted_distances.len());
                &sorted_distances[..k]
            }
            ConnectionStrategy::Radius { gamma } => {
                let radius = Self::connection_radius(gamma, num_nodes, dof);
                let count = sorted_distances
                    .iter()
                    .take_while(|&&(_, dist)| dist <= radius)
                    .count();
                &sorted_distances[..count]
            }
        }
    }

    /// Compute the PRM* connection radius: `r = gamma * (log(n)/n)^(1/d)`.
    ///
    /// This radius ensures asymptotic optimality as `n → ∞` (Karaman & Frazzoli, 2011).
    fn connection_radius(gamma: f64, num_nodes: usize, dof: usize) -> f64 {
        if num_nodes <= 1 || dof == 0 {
            return f64::INFINITY;
        }
        let n = num_nodes as f64;
        let d = dof as f64;
        gamma * (n.ln() / n).powf(1.0 / d)
    }

    /// Compute the minimum gamma for asymptotic optimality given joint limits.
    ///
    /// gamma_min = 2 * (1 + 1/d)^(1/d) * (mu_free / zeta_d)^(1/d)
    /// where mu_free = product of joint ranges and zeta_d = volume of unit d-ball.
    #[allow(dead_code)]
    fn auto_gamma(joint_limits: &[(f64, f64)]) -> f64 {
        let d = joint_limits.len() as f64;
        if d == 0.0 {
            return 1.5;
        }
        // Volume of free C-space (product of joint ranges)
        let mu_free: f64 = joint_limits
            .iter()
            .map(|&(lo, hi)| (hi - lo).abs())
            .product();
        // Volume of unit d-ball: pi^(d/2) / Gamma(d/2 + 1)
        let zeta_d = std::f64::consts::PI.powf(d / 2.0) / gamma_fn(d / 2.0 + 1.0);
        // gamma_min with 1.1x safety margin
        let gamma = 2.0 * (1.0 + 1.0 / d).powf(1.0 / d) * (mu_free / zeta_d).powf(1.0 / d);
        gamma * 1.1
    }

    /// Compute PRM* adaptive k: `k = ceil(k_multiplier * e * (1 + 1/d) * ln(n))`.
    ///
    /// This ensures asymptotic optimality without needing to know the C-space volume.
    fn prm_star_k(k_multiplier: f64, num_nodes: usize, dof: usize) -> usize {
        if num_nodes <= 1 || dof == 0 {
            return num_nodes;
        }
        let n = num_nodes as f64;
        let d = dof as f64;
        let k = k_multiplier * std::f64::consts::E * (1.0 + 1.0 / d) * n.ln();
        (k.ceil() as usize).max(1)
    }

    /// Get the current adaptive k value (for PRM* AdaptiveK mode). Returns None for other modes.
    pub fn current_k(&self) -> Option<usize> {
        match self.prm_config.connection {
            ConnectionStrategy::AdaptiveK { k_multiplier } => {
                let roadmap = self.roadmap.as_ref()?;
                Some(Self::prm_star_k(k_multiplier, roadmap.nodes.len(), roadmap.dof))
            }
            _ => None,
        }
    }

    /// Get the current connection radius (for PRM* Radius mode). Returns None for K-based modes.
    pub fn current_radius(&self) -> Option<f64> {
        match self.prm_config.connection {
            ConnectionStrategy::Radius { gamma } => {
                let roadmap = self.roadmap.as_ref()?;
                Some(Self::connection_radius(gamma, roadmap.nodes.len(), roadmap.dof))
            }
            _ => None,
        }
    }

    fn is_edge_collision_free(&self, from: &[f64], to: &[f64]) -> bool {
        let dist = joint_distance(from, to);
        let n = (dist / self.prm_config.edge_step).ceil() as usize;
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
                "Relative goals not supported in PRM planner".into(),
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

    fn setup_prm() -> PRM {
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

        PRM::new(
            robot,
            chain,
            env,
            config,
            PRMConfig {
                num_samples: 200,
                connection: ConnectionStrategy::KNearest(8),
                ..Default::default()
            },
        )
    }

    fn setup_prm_star() -> PRM {
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

        // PRM* with adaptive K: k = ceil(e * (1 + 1/6) * ln(50)) ≈ ceil(2.718 * 1.167 * 3.912) ≈ 13
        // Use 50 samples to keep debug-mode tests fast.
        PRM::new(
            robot,
            chain,
            env,
            config,
            PRMConfig::prm_star(50),
        )
    }

    #[test]
    fn prm_build_roadmap() {
        let mut prm = setup_prm();
        let roadmap = prm.build_roadmap();
        assert_eq!(roadmap.num_nodes(), 200);
        assert!(roadmap.num_edges() > 0, "Roadmap should have edges");
        assert!(roadmap.construction_time() > Duration::ZERO);
    }

    #[test]
    fn prm_query_free_space() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = prm.query(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.path_cost > 0.0);
        assert!(result.nodes_expanded > 0);
    }

    #[test]
    fn prm_multi_query_reuses_roadmap() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let start1 = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal1 = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let start2 = vec![0.3, -0.5, 0.5, 0.0, 0.0, 0.0];
        let goal2 = Goal::Joints(JointValues(vec![-0.3, -1.2, 0.6, 0.0, 0.0, 0.0]));

        let r1 = prm.query(&start1, &goal1).unwrap();
        let r2 = prm.query(&start2, &goal2).unwrap();

        assert!(r1.waypoints.len() >= 2);
        assert!(r2.waypoints.len() >= 2);
        // Second query should be fast (no roadmap rebuild)
    }

    #[test]
    fn prm_path_collision_free() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = prm.query(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!prm.is_in_collision(wp), "Waypoint {} in collision", i);
        }
    }

    #[test]
    fn prm_start_in_collision() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let mut obs = kinetic_collision::SpheresSoA::new();
        obs.push(0.0, 0.0, 0.0, 5.0, 0);
        let env = CollisionEnvironment::build(obs, 0.05, AABB::symmetric(10.0));

        let mut prm = PRM::new(
            robot,
            chain,
            env,
            PlannerConfig::default(),
            PRMConfig {
                num_samples: 50,
                ..Default::default()
            },
        );
        prm.build_roadmap();

        let start = vec![0.0; 6];
        let goal = Goal::Joints(JointValues(vec![1.0, -1.0, 0.5, 0.0, 0.0, 0.0]));

        match prm.query(&start, &goal) {
            Err(KineticError::StartInCollision) => {}
            other => panic!("Expected StartInCollision, got {:?}", other),
        }
    }

    #[test]
    fn prm_incremental_growth() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let initial_nodes = prm.roadmap.as_ref().unwrap().num_nodes();
        prm.add_samples(50);
        let new_nodes = prm.roadmap.as_ref().unwrap().num_nodes();
        assert_eq!(new_nodes, initial_nodes + 50);
    }

    #[test]
    fn prm_lazy_mode() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(30),
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        };

        let mut prm = PRM::new(
            robot,
            chain,
            env,
            config,
            PRMConfig {
                num_samples: 200,
                connection: ConnectionStrategy::KNearest(8),
                lazy: true,
                ..Default::default()
            },
        );
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        // Lazy PRM defers collision checks to query time
        let result = prm.query(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    // --- PRM* tests ---

    #[test]
    fn prm_star_build_roadmap() {
        let mut prm = setup_prm_star();
        let roadmap = prm.build_roadmap();
        assert_eq!(roadmap.num_nodes(), 50);
        assert!(roadmap.num_edges() > 0, "PRM* roadmap should have edges");
    }

    #[test]
    fn prm_star_query_free_space() {
        let mut prm = setup_prm_star();
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = prm.query(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.path_cost > 0.0);
    }

    #[test]
    fn prm_star_connection_radius_formula() {
        // r = gamma * (log(n)/n)^(1/d)
        // With gamma=1.5, n=500, d=6:
        let r = PRM::connection_radius(1.5, 500, 6);
        // (ln(500)/500)^(1/6) ≈ (6.2146/500)^(1/6) ≈ (0.01243)^(0.1667) ≈ 0.484
        // r ≈ 1.5 * 0.484 ≈ 0.726
        assert!(r > 0.0, "Radius must be positive");
        assert!(r < 5.0, "Radius should be reasonable: {}", r);

        // Radius should decrease as n increases (denser roadmap needs smaller radius)
        let r_small = PRM::connection_radius(1.5, 100, 6);
        let r_large = PRM::connection_radius(1.5, 10000, 6);
        assert!(r_small > r_large, "Radius should shrink with more samples");
    }

    #[test]
    fn prm_star_radius_scales_with_dimension() {
        // For a fixed gamma and n, higher d → exponent 1/d is smaller →
        // (log(n)/n)^(1/d) is closer to 1 → larger radius.
        // This makes sense: higher-dimensional spaces need larger radii for connectivity.
        let r_3dof = PRM::connection_radius(1.5, 500, 3);
        let r_6dof = PRM::connection_radius(1.5, 500, 6);
        assert!(
            r_6dof > r_3dof,
            "6-DOF radius ({}) should be larger than 3-DOF ({})",
            r_6dof,
            r_3dof
        );
    }

    #[test]
    fn prm_star_current_k() {
        let mut prm = setup_prm_star();
        assert!(prm.current_k().is_none(), "No roadmap yet");

        prm.build_roadmap();
        let k = prm.current_k().unwrap();
        // k = ceil(e * (1 + 1/6) * ln(200)) = ceil(2.718 * 1.167 * 5.298) ≈ 17
        assert!(k > 5 && k < 50, "Expected k ~17 for 200 nodes 6-DOF, got {}", k);
    }

    #[test]
    fn prm_star_incremental_grows_k() {
        let mut prm = setup_prm_star();
        prm.build_roadmap();

        let k_before = prm.current_k().unwrap();
        prm.add_samples(50);
        let k_after = prm.current_k().unwrap();

        // More samples → larger k (k grows as O(log n))
        assert!(
            k_after > k_before,
            "k should grow after adding samples: before={}, after={}",
            k_before,
            k_after
        );
    }

    #[test]
    fn prm_star_multi_query() {
        let mut prm = setup_prm_star();
        prm.build_roadmap();

        let start1 = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal1 = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let start2 = vec![0.3, -0.5, 0.5, 0.0, 0.0, 0.0];
        let goal2 = Goal::Joints(JointValues(vec![-0.3, -1.2, 0.6, 0.0, 0.0, 0.0]));

        let r1 = prm.query(&start1, &goal1).unwrap();
        let r2 = prm.query(&start2, &goal2).unwrap();

        assert!(r1.waypoints.len() >= 2);
        assert!(r2.waypoints.len() >= 2);
    }

    #[test]
    fn prm_star_path_collision_free() {
        let mut prm = setup_prm_star();
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = prm.query(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!prm.is_in_collision(wp), "PRM* waypoint {} in collision", i);
        }
    }

    #[test]
    fn prm_star_config_convenience() {
        let config = PRMConfig::prm_star(1000);
        assert_eq!(config.num_samples, 1000);
        assert_eq!(config.connection, ConnectionStrategy::AdaptiveK { k_multiplier: 1.0 });
        assert!(!config.lazy);

        let config_r = PRMConfig::prm_star_radius(500, 2.0);
        assert_eq!(config_r.num_samples, 500);
        assert_eq!(config_r.connection, ConnectionStrategy::Radius { gamma: 2.0 });
    }

    #[test]
    fn connection_radius_edge_cases() {
        // n=0 or n=1 → infinity (connect to everything)
        assert!(PRM::connection_radius(1.5, 0, 6).is_infinite());
        assert!(PRM::connection_radius(1.5, 1, 6).is_infinite());
        // d=0 → infinity
        assert!(PRM::connection_radius(1.5, 100, 0).is_infinite());
        // n=2 should produce a finite radius
        let r = PRM::connection_radius(1.5, 2, 6);
        assert!(r.is_finite());
        assert!(r > 0.0);
    }

    // --- Persistence tests ---

    #[test]
    fn roadmap_save_load_roundtrip() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let original = prm.roadmap.as_ref().unwrap();
        let orig_nodes = original.num_nodes();
        let orig_edges = original.num_edges();

        // Save to buffer
        let mut buf = Vec::new();
        original.save(&mut buf, 0xDEADBEEF).unwrap();
        assert!(!buf.is_empty());

        // Load back
        let mut cursor = std::io::Cursor::new(&buf);
        let (loaded, scene_hash) = Roadmap::load(&mut cursor).unwrap();

        assert_eq!(scene_hash, 0xDEADBEEF);
        assert_eq!(loaded.num_nodes(), orig_nodes);
        assert_eq!(loaded.num_edges(), orig_edges);
        assert_eq!(loaded.dof, original.dof);

        // Check node joints match
        for (i, (orig, load)) in original.nodes.iter().zip(loaded.nodes.iter()).enumerate() {
            assert_eq!(orig.joints, load.joints, "Node {} joints mismatch", i);
            assert_eq!(orig.edges.len(), load.edges.len(), "Node {} edge count mismatch", i);
        }
    }

    #[test]
    fn roadmap_save_load_file() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let tmp = std::env::temp_dir().join("kinetic_test_roadmap.kprm");
        prm.save_roadmap(&tmp, 42).unwrap();

        assert!(tmp.exists());
        let metadata = std::fs::metadata(&tmp).unwrap();
        assert!(metadata.len() > 0);

        let scene_hash = prm.load_roadmap(&tmp).unwrap();
        assert_eq!(scene_hash, 42);
        assert_eq!(prm.roadmap.as_ref().unwrap().num_nodes(), 200);

        // Clean up
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn roadmap_loaded_produces_same_results() {
        let mut prm = setup_prm();
        prm.build_roadmap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result1 = prm.query(&start, &goal).unwrap();

        // Save and reload
        let mut buf = Vec::new();
        prm.roadmap.as_ref().unwrap().save(&mut buf, 0).unwrap();

        let (loaded, _) = Roadmap::load(&mut std::io::Cursor::new(&buf)).unwrap();
        prm.roadmap = Some(loaded);

        let result2 = prm.query(&start, &goal).unwrap();

        // Same path cost (same roadmap = same A* result)
        assert!(
            (result1.path_cost - result2.path_cost).abs() < 1e-10,
            "Path costs differ: {} vs {}",
            result1.path_cost,
            result2.path_cost
        );
        assert_eq!(result1.waypoints.len(), result2.waypoints.len());
    }

    #[test]
    fn roadmap_invalid_magic() {
        let buf = b"NOPE1234";
        let result = Roadmap::load(&mut std::io::Cursor::new(buf));
        assert!(result.is_err());
    }

    #[test]
    fn roadmap_version_check() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KPRM");
        buf.extend_from_slice(&99u32.to_le_bytes()); // unsupported version
        let result = Roadmap::load(&mut std::io::Cursor::new(&buf));
        assert!(result.is_err());
    }

    #[test]
    fn prm_save_without_roadmap_errors() {
        let prm = setup_prm();
        let tmp = std::env::temp_dir().join("kinetic_test_no_roadmap.kprm");
        let result = prm.save_roadmap(&tmp, 0);
        assert!(result.is_err());
    }

    #[test]
    fn prm_load_dof_mismatch() {
        // Build a roadmap, save it, then try to load into a different DOF chain
        let mut prm = setup_prm();
        prm.build_roadmap();

        let mut buf = Vec::new();
        prm.roadmap.as_ref().unwrap().save(&mut buf, 0).unwrap();

        // Create PRM with a 3-DOF test robot
        let robot = Arc::new(Robot::from_urdf_string(
            r#"<?xml version="1.0"?>
            <robot name="test">
              <link name="base"/>
              <link name="l1"/>
              <link name="l2"/>
              <link name="ee"/>
              <joint name="j1" type="revolute">
                <parent link="base"/><child link="l1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" velocity="2" effort="100"/>
              </joint>
              <joint name="j2" type="revolute">
                <parent link="l1"/><child link="l2"/>
                <origin xyz="0 0 0.3"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2" upper="2" velocity="2" effort="80"/>
              </joint>
              <joint name="j3" type="revolute">
                <parent link="l2"/><child link="ee"/>
                <origin xyz="0 0 0.3"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2.5" upper="2.5" velocity="3" effort="50"/>
              </joint>
            </robot>"#,
        ).unwrap());

        let chain = KinematicChain::extract(&robot, "base", "ee").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));
        let mut prm3 = PRM::new(robot, chain, env, PlannerConfig::default(), PRMConfig::default());

        // Save the 6-DOF roadmap to a file, try to load into 3-DOF PRM
        let tmp = std::env::temp_dir().join("kinetic_test_dof_mismatch.kprm");
        std::fs::write(&tmp, &buf).unwrap();
        let result = prm3.load_roadmap(&tmp);
        assert!(result.is_err(), "Should fail on DOF mismatch");
        std::fs::remove_file(&tmp).ok();
    }
}
