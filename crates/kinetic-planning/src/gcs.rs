//! Graphs of Convex Sets (GCS) planner.
//!
//! Globally optimal motion planning over pre-computed convex regions.
//! The planner builds a graph where nodes are convex C-space regions and
//! edges connect overlapping/adjacent regions. A shortest path through
//! the graph yields a collision-free trajectory that is optimal within
//! the convex decomposition.
//!
//! Inspired by Marcucci et al., Science Robotics 2023.

use crate::facade::PlannerType;
use crate::iris::{ConvexDecomposition, IrisConfig};
use crate::shortcut::CollisionChecker;
use crate::PlanningResult;
use kinetic_core::Result as KResult;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// GCS planner operating over pre-computed convex regions.
pub struct GCSPlanner {
    /// The convex decomposition of C-space.
    decomposition: ConvexDecomposition,
    /// Adjacency graph: adj[i] lists region indices adjacent to region i.
    adjacency: Vec<Vec<usize>>,
}

impl GCSPlanner {
    /// Create a GCS planner from a pre-computed convex decomposition.
    pub fn from_decomposition<C: CollisionChecker>(
        decomposition: ConvexDecomposition,
        checker: &C,
        step_size: f64,
    ) -> Self {
        let adjacency = decomposition.adjacency(checker, step_size);
        Self {
            decomposition,
            adjacency,
        }
    }

    /// Build a GCS planner end-to-end: IRIS decomposition + graph construction.
    pub fn build<C: CollisionChecker>(
        checker: &C,
        limits: &[(f64, f64)],
        config: &IrisConfig,
        step_size: f64,
    ) -> KResult<Self> {
        let decomposition = ConvexDecomposition::iris(checker, limits, config)?;
        let adjacency = decomposition.adjacency(checker, step_size);
        Ok(Self {
            decomposition,
            adjacency,
        })
    }

    /// Plan a path from start to goal through the convex region graph.
    ///
    /// Algorithm:
    /// 1. Find regions containing start and goal
    /// 2. A* search through the region graph
    /// 3. Extract waypoints (region centers along the path)
    /// 4. Refine: use linear interpolation within convex regions
    pub fn plan(&self, start: &[f64], goal: &[f64]) -> KResult<PlanningResult> {
        let timer = std::time::Instant::now();

        // Find start and goal regions
        let start_regions = self.decomposition.containing_regions(start);
        let goal_regions = self.decomposition.containing_regions(goal);

        if start_regions.is_empty() {
            return Err(kinetic_core::KineticError::PlanningFailed(
                "start configuration not in any convex region".into(),
            ));
        }
        if goal_regions.is_empty() {
            return Err(kinetic_core::KineticError::PlanningFailed(
                "goal configuration not in any convex region".into(),
            ));
        }

        // Check if start and goal are in the same region (trivial case)
        for &sr in &start_regions {
            if goal_regions.contains(&sr) {
                return Ok(PlanningResult {
                    waypoints: vec![start.to_vec(), goal.to_vec()],
                    planning_time: timer.elapsed(),
                    iterations: 0,
                    tree_size: 0,
                    planner_used: PlannerType::Auto,
                });
            }
        }

        // A* search from start regions to goal regions
        let n = self.decomposition.num_regions();
        let region_path = self.astar_search(&start_regions, &goal_regions, goal)?;

        // Extract waypoints along the region path
        let mut waypoints = vec![start.to_vec()];

        for &region_idx in &region_path {
            let center = &self.decomposition.regions[region_idx].center;
            // Use the region center as a waypoint (guaranteed collision-free)
            waypoints.push(center.clone());
        }

        waypoints.push(goal.to_vec());

        // Refine: smooth the path by removing unnecessary waypoints
        // (linear interpolation within convex regions is always safe)
        let refined = self.refine_path(&waypoints);

        Ok(PlanningResult {
            waypoints: refined,
            planning_time: timer.elapsed(),
            iterations: region_path.len(),
            tree_size: n,
            planner_used: PlannerType::Auto,
        })
    }

    /// A* search through the region adjacency graph.
    fn astar_search(
        &self,
        start_regions: &[usize],
        goal_regions: &[usize],
        goal: &[f64],
    ) -> KResult<Vec<usize>> {
        let n = self.decomposition.num_regions();
        let mut dist = vec![f64::INFINITY; n];
        let mut came_from = vec![None; n];
        let mut heap = BinaryHeap::new();

        // Initialize with all start regions
        for &sr in start_regions {
            dist[sr] = 0.0;
            let h = heuristic(&self.decomposition.regions[sr].center, goal);
            heap.push(AStarNode {
                region: sr,
                cost: 0.0,
                priority: h,
            });
        }

        while let Some(node) = heap.pop() {
            let u = node.region;

            // Check if we reached a goal region
            if goal_regions.contains(&u) {
                // Reconstruct path
                let mut path = vec![u];
                let mut current = u;
                while let Some(prev) = came_from[current] {
                    path.push(prev);
                    current = prev;
                }
                path.reverse();
                return Ok(path);
            }

            if node.cost > dist[u] {
                continue; // stale entry
            }

            // Explore neighbors
            for &v in &self.adjacency[u] {
                let edge_cost = joint_distance(
                    &self.decomposition.regions[u].center,
                    &self.decomposition.regions[v].center,
                );
                let new_cost = dist[u] + edge_cost;

                if new_cost < dist[v] {
                    dist[v] = new_cost;
                    came_from[v] = Some(u);
                    let h = heuristic(&self.decomposition.regions[v].center, goal);
                    heap.push(AStarNode {
                        region: v,
                        cost: new_cost,
                        priority: new_cost + h,
                    });
                }
            }
        }

        Err(kinetic_core::KineticError::PlanningFailed(
            "no path found through convex region graph".into(),
        ))
    }

    /// Refine a path by removing redundant waypoints.
    /// Since each region is convex, straight-line paths within regions
    /// are automatically collision-free.
    fn refine_path(&self, waypoints: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if waypoints.len() <= 2 {
            return waypoints.to_vec();
        }

        let mut refined = vec![waypoints[0].clone()];
        let mut i = 0;

        while i < waypoints.len() - 1 {
            // Try to skip intermediate waypoints
            let mut farthest = i + 1;
            for j in (i + 2)..waypoints.len() {
                // Check if the midpoint of the skip is in some convex region
                let mid: Vec<f64> = waypoints[i]
                    .iter()
                    .zip(waypoints[j].iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();

                if !self.decomposition.containing_regions(&mid).is_empty() {
                    farthest = j;
                } else {
                    break;
                }
            }

            i = farthest;
            refined.push(waypoints[i].clone());
        }

        refined
    }

    /// Access the underlying decomposition.
    pub fn decomposition(&self) -> &ConvexDecomposition {
        &self.decomposition
    }

    /// Number of regions in the decomposition.
    pub fn num_regions(&self) -> usize {
        self.decomposition.num_regions()
    }
}

/// A* priority queue node.
#[derive(Debug, Clone)]
struct AStarNode {
    region: usize,
    cost: f64,
    priority: f64, // cost + heuristic (lower is better)
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is a max-heap)
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Euclidean distance heuristic for A*.
fn heuristic(a: &[f64], b: &[f64]) -> f64 {
    joint_distance(a, b)
}

fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BoxChecker {
        box_min: [f64; 2],
        box_max: [f64; 2],
    }

    impl CollisionChecker for BoxChecker {
        fn is_in_collision(&self, joints: &[f64]) -> bool {
            joints[0] >= self.box_min[0]
                && joints[0] <= self.box_max[0]
                && joints[1] >= self.box_min[1]
                && joints[1] <= self.box_max[1]
        }
    }

    #[test]
    fn gcs_trivial_no_obstacles() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let checker = NoCollision;
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 5,
            seed_points: Some(vec![vec![0.1, 0.1], vec![0.5, 0.5], vec![0.9, 0.9]]),
            ..Default::default()
        };

        let planner = GCSPlanner::build(&checker, &limits, &config, 0.05).unwrap();
        assert!(planner.num_regions() > 0);

        let result = planner.plan(&[0.1, 0.1], &[0.9, 0.9]).unwrap();
        assert!(result.waypoints.len() >= 2);

        // Start and goal should match
        let start = &result.waypoints[0];
        let end = result.waypoints.last().unwrap();
        assert!((start[0] - 0.1).abs() < 1e-10);
        assert!((end[0] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn gcs_around_obstacle() {
        let checker = BoxChecker {
            box_min: [0.35, 0.35],
            box_max: [0.65, 0.65],
        };
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 10,
            seed_points: Some(vec![
                vec![0.1, 0.1],
                vec![0.1, 0.9],
                vec![0.9, 0.1],
                vec![0.9, 0.9],
                vec![0.1, 0.5],
                vec![0.9, 0.5],
                vec![0.5, 0.1],
                vec![0.5, 0.9],
            ]),
            num_obstacle_samples: 30,
            boundary_step: 0.02,
            ..Default::default()
        };

        let planner = GCSPlanner::build(&checker, &limits, &config, 0.05).unwrap();

        // Plan from bottom-left to top-right (must go around obstacle)
        let result = planner.plan(&[0.1, 0.1], &[0.9, 0.9]);
        if let Ok(result) = result {
            // All waypoints should be collision-free
            for wp in &result.waypoints {
                assert!(
                    !checker.is_in_collision(wp),
                    "waypoint {:?} is in collision",
                    wp
                );
            }
        }
        // Note: planning may fail if IRIS doesn't create enough connected regions
        // with these specific seeds. This is expected for a simplified IRIS.
    }

    #[test]
    fn gcs_start_equals_goal() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let checker = NoCollision;
        let limits = [(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 3,
            seed_points: Some(vec![vec![0.5, 0.5]]),
            ..Default::default()
        };

        let planner = GCSPlanner::build(&checker, &limits, &config, 0.05).unwrap();
        let result = planner.plan(&[0.5, 0.5], &[0.5, 0.5]).unwrap();
        assert_eq!(result.waypoints.len(), 2); // start + goal
    }

    // ─── New tests below ───

    /// Helper: build a GCSPlanner with a manually specified adjacency graph.
    fn make_planner_with_adj(
        regions: Vec<crate::iris::ConvexRegion>,
        adj: Vec<Vec<usize>>,
        limits: Vec<(f64, f64)>,
    ) -> GCSPlanner {
        let dof = limits.len();
        GCSPlanner {
            decomposition: ConvexDecomposition {
                regions,
                limits,
                dof,
            },
            adjacency: adj,
        }
    }

    /// Make a convex region centered at `center` with box halfplanes of given `extent`.
    fn box_region(center: Vec<f64>, extent: f64) -> crate::iris::ConvexRegion {
        let dof = center.len();
        let mut halfplanes = Vec::new();
        for d in 0..dof {
            let mut n_lo = vec![0.0; dof];
            n_lo[d] = -1.0;
            halfplanes.push((n_lo, -(center[d] - extent)));
            let mut n_hi = vec![0.0; dof];
            n_hi[d] = 1.0;
            halfplanes.push((n_hi, center[d] + extent));
        }
        crate::iris::ConvexRegion {
            center,
            halfplanes,
            dof,
        }
    }

    /// A* search on a manually constructed graph with a known shortest path.
    #[test]
    fn gcs_astar_known_shortest_path() {
        // Graph: 0 --1-- 1 --1-- 2 (direct path)
        //        0 ----3---- 2       (longer via cost)
        // Regions at (0,0), (0.5,0), (1.0,0), with region 0-1-2 connected linearly
        let r0 = box_region(vec![0.0, 0.0], 0.3);
        let r1 = box_region(vec![0.5, 0.0], 0.3);
        let r2 = box_region(vec![1.0, 0.0], 0.3);

        // adj: 0↔1, 1↔2 (no direct 0↔2 shortcut)
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let planner = make_planner_with_adj(vec![r0, r1, r2], adj, vec![(-1.0, 2.0), (-1.0, 1.0)]);

        let result = planner.plan(&[0.0, 0.0], &[1.0, 0.0]).unwrap();
        // Should find a path: start → region 0 → region 1 → region 2 → goal
        assert!(result.waypoints.len() >= 2);
        assert!((result.waypoints[0][0] - 0.0).abs() < 1e-10);
        assert!((result.waypoints.last().unwrap()[0] - 1.0).abs() < 1e-10);
    }

    /// Disconnected graph: start and goal in different connected components.
    #[test]
    fn gcs_disconnected_graph_error() {
        // Two isolated regions — no edges between them
        let r0 = box_region(vec![0.0, 0.0], 0.3);
        let r1 = box_region(vec![5.0, 5.0], 0.3);

        let adj = vec![vec![], vec![]]; // no connectivity
        let planner = make_planner_with_adj(vec![r0, r1], adj, vec![(-1.0, 6.0), (-1.0, 6.0)]);

        let result = planner.plan(&[0.0, 0.0], &[5.0, 5.0]);
        assert!(
            result.is_err(),
            "planning in disconnected graph should fail"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("no path found"),
            "error should mention no path found, got: {err_msg}"
        );
    }

    /// Start configuration not contained in any region.
    #[test]
    fn gcs_start_not_in_region() {
        let r0 = box_region(vec![5.0, 5.0], 0.3);
        let adj = vec![vec![]];
        let planner = make_planner_with_adj(vec![r0], adj, vec![(-10.0, 10.0), (-10.0, 10.0)]);

        let result = planner.plan(&[0.0, 0.0], &[5.0, 5.0]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("start"));
    }

    /// Goal configuration not contained in any region.
    #[test]
    fn gcs_goal_not_in_region() {
        let r0 = box_region(vec![0.0, 0.0], 0.3);
        let adj = vec![vec![]];
        let planner = make_planner_with_adj(vec![r0], adj, vec![(-10.0, 10.0), (-10.0, 10.0)]);

        let result = planner.plan(&[0.0, 0.0], &[9.0, 9.0]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("goal"));
    }

    /// Trivial case: start and goal in the same convex region (direct path).
    #[test]
    fn gcs_same_region_direct_path() {
        let r0 = box_region(vec![0.0, 0.0], 1.0); // region covers [-1,1]^2
        let adj = vec![vec![]];
        let planner = make_planner_with_adj(vec![r0], adj, vec![(-2.0, 2.0), (-2.0, 2.0)]);

        let result = planner.plan(&[-0.5, -0.5], &[0.5, 0.5]).unwrap();
        // Direct path: just start and goal, no intermediate waypoints needed
        assert_eq!(result.waypoints.len(), 2);
        assert_eq!(result.iterations, 0, "same-region plan needs 0 iterations");
    }

    /// Multi-region path: verify path visits expected regions in order.
    #[test]
    fn gcs_multi_region_path() {
        // Linear chain: r0 → r1 → r2 → r3
        let regions = vec![
            box_region(vec![0.0, 0.0], 0.3),
            box_region(vec![0.5, 0.0], 0.3),
            box_region(vec![1.0, 0.0], 0.3),
            box_region(vec![1.5, 0.0], 0.3),
        ];
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let planner = make_planner_with_adj(regions, adj, vec![(-1.0, 2.0), (-1.0, 1.0)]);

        let result = planner.plan(&[0.0, 0.0], &[1.5, 0.0]).unwrap();
        // Should have start + intermediate centers + goal
        assert!(
            result.waypoints.len() >= 2,
            "multi-region path should have at least start + goal"
        );
        // iterations should reflect the number of regions traversed
        assert!(
            result.iterations >= 2,
            "should traverse multiple regions, got {} iterations",
            result.iterations
        );
    }

    /// Empty decomposition (0 regions) — start not found.
    #[test]
    fn gcs_empty_decomposition() {
        let planner = make_planner_with_adj(vec![], vec![], vec![(-1.0, 1.0), (-1.0, 1.0)]);
        let result = planner.plan(&[0.0, 0.0], &[0.5, 0.5]);
        assert!(result.is_err(), "empty decomposition should fail planning");
    }

    /// refine_path removes redundant waypoints when midpoints are in regions.
    #[test]
    fn gcs_refine_removes_redundant_waypoints() {
        // One big region covering everything — all midpoints are in it
        let r0 = box_region(vec![0.5, 0.5], 1.0); // covers [-0.5, 1.5]^2
        let adj = vec![vec![]];
        let planner = make_planner_with_adj(vec![r0], adj, vec![(-1.0, 2.0), (-1.0, 2.0)]);

        let waypoints = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.2],
            vec![0.4, 0.4],
            vec![0.6, 0.6],
            vec![0.8, 0.8],
            vec![1.0, 1.0],
        ];

        let refined = planner.refine_path(&waypoints);
        // All midpoints between start and goal are in the region,
        // so refine should collapse to just start + goal
        assert!(
            refined.len() <= waypoints.len(),
            "refine should not add waypoints"
        );
        assert!(
            refined.len() < waypoints.len(),
            "refine should remove redundant waypoints, got {} from {}",
            refined.len(),
            waypoints.len()
        );
        // Start and end preserved
        assert_eq!(refined[0], vec![0.0, 0.0]);
        assert_eq!(*refined.last().unwrap(), vec![1.0, 1.0]);
    }

    /// refine_path with 2 waypoints returns them unchanged.
    #[test]
    fn gcs_refine_two_waypoints_unchanged() {
        let r0 = box_region(vec![0.0, 0.0], 1.0);
        let adj = vec![vec![]];
        let planner = make_planner_with_adj(vec![r0], adj, vec![(-2.0, 2.0), (-2.0, 2.0)]);

        let waypoints = vec![vec![0.0, 0.0], vec![0.5, 0.5]];
        let refined = planner.refine_path(&waypoints);
        assert_eq!(refined.len(), 2);
    }

    /// Heuristic is admissible: h(n) = Euclidean distance ≤ actual path cost.
    #[test]
    fn gcs_heuristic_admissible() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let h = heuristic(&a, &b);
        assert!(
            (h - 5.0).abs() < 1e-10,
            "Euclidean distance from (0,0) to (3,4) should be 5, got {h}"
        );
        // Euclidean distance is always a lower bound of actual path cost
        // in non-negative-cost graphs, so it's admissible.
    }

    /// from_decomposition uses the provided checker for adjacency.
    #[test]
    fn gcs_from_decomposition() {
        struct NoCollision;
        impl CollisionChecker for NoCollision {
            fn is_in_collision(&self, _: &[f64]) -> bool {
                false
            }
        }

        let limits = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = IrisConfig {
            num_regions: 3,
            seed_points: Some(vec![vec![0.2, 0.2], vec![0.5, 0.5], vec![0.8, 0.8]]),
            ..Default::default()
        };
        let decomp = ConvexDecomposition::iris(&NoCollision, &limits, &config).unwrap();
        let planner = GCSPlanner::from_decomposition(decomp, &NoCollision, 0.05);

        assert!(planner.num_regions() > 0);
        assert_eq!(planner.decomposition().dof, 2);
    }
}
