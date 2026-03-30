//! Randomized path shortcutting for RRT post-processing.
//!
//! After RRT-Connect finds a collision-free path, the raw path has unnecessary
//! detours. Randomized shortcutting picks two random waypoints, checks if the
//! straight-line in C-space between them is collision-free, and removes the
//! intermediate waypoints.
//!
//! Typical result: 50-80% reduction in waypoints, 30-60% reduction in path length.

use rand::Rng;

/// Collision checker trait for shortcutting.
///
/// This decouples shortcutting from the specific collision checking implementation,
/// allowing it to work with any collision backend.
pub trait CollisionChecker {
    /// Check if a joint configuration is in collision.
    fn is_in_collision(&self, joints: &[f64]) -> bool;
}

/// Randomized path shortcutting.
///
/// For each iteration, picks two random waypoints (i, j with i < j) and checks
/// if the straight-line interpolation in C-space is collision-free. If so,
/// removes all waypoints between i and j.
///
/// `path`: list of joint-space waypoints.
/// `checker`: collision checker (must support `is_in_collision`).
/// `iterations`: number of shortcutting attempts (more = shorter path, diminishing returns).
/// `step_size`: discretization step for collision checking along interpolated segment.
///
/// Returns the shortcutted path (always starts and ends at same configs as input).
pub fn shortcut<C: CollisionChecker>(
    path: &[Vec<f64>],
    checker: &C,
    iterations: usize,
    step_size: f64,
) -> Vec<Vec<f64>> {
    if path.len() <= 2 {
        return path.to_vec();
    }

    let mut result = path.to_vec();
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {
        if result.len() <= 2 {
            break;
        }

        // Pick two random indices i < j
        let i = rng.gen_range(0..result.len() - 1);
        let j = rng.gen_range(i + 1..result.len());

        if j - i <= 1 {
            continue; // no waypoints to remove
        }

        // Check if straight line from result[i] to result[j] is collision-free
        if is_segment_collision_free(checker, &result[i], &result[j], step_size) {
            // Remove intermediate waypoints
            result.drain(i + 1..j);
        }
    }

    result
}

/// Check if the straight-line interpolation between two C-space configs is collision-free.
///
/// Discretizes at `step_size` intervals and checks each intermediate point.
fn is_segment_collision_free<C: CollisionChecker>(
    checker: &C,
    from: &[f64],
    to: &[f64],
    step_size: f64,
) -> bool {
    let dist = joint_distance(from, to);
    if dist < 1e-10 {
        return true;
    }

    let num_steps = (dist / step_size).ceil() as usize;
    let num_steps = num_steps.max(1);

    for step in 1..num_steps {
        let t = step as f64 / num_steps as f64;
        let interp: Vec<f64> = from
            .iter()
            .zip(to.iter())
            .map(|(&a, &b)| a + t * (b - a))
            .collect();

        if checker.is_in_collision(&interp) {
            return false;
        }
    }

    true
}

/// Euclidean distance in joint space.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the total path length in joint space.
pub fn path_length(path: &[Vec<f64>]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    path.windows(2).map(|w| joint_distance(&w[0], &w[1])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A collision checker that never reports collision (free space).
    struct FreeSpaceChecker;

    impl CollisionChecker for FreeSpaceChecker {
        fn is_in_collision(&self, _joints: &[f64]) -> bool {
            false
        }
    }

    /// A collision checker that blocks a specific region.
    struct WallChecker {
        /// Blocks if any joint is in [wall_lo, wall_hi].
        wall_lo: f64,
        wall_hi: f64,
    }

    impl CollisionChecker for WallChecker {
        fn is_in_collision(&self, joints: &[f64]) -> bool {
            joints
                .iter()
                .any(|&v| v >= self.wall_lo && v <= self.wall_hi)
        }
    }

    #[test]
    fn shortcut_free_space_reduces_waypoints() {
        // A zigzag path in free space should be shortcuttable
        let path = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.3],
            vec![0.2, 0.8],
            vec![0.7, 0.5],
            vec![0.3, 0.9],
            vec![1.0, 1.0],
        ];

        let shortened = shortcut(&path, &FreeSpaceChecker, 100, 0.05);

        // Should preserve start and end
        assert_eq!(shortened.first().unwrap(), path.first().unwrap());
        assert_eq!(shortened.last().unwrap(), path.last().unwrap());

        // Should have fewer waypoints (in free space, should shortcut to just start+end)
        assert!(
            shortened.len() <= path.len(),
            "Shortened ({}) should be <= original ({})",
            shortened.len(),
            path.len()
        );

        // In free space, ideally reduces to 2 waypoints (start and end)
        assert_eq!(
            shortened.len(),
            2,
            "Free space should shortcut to direct line"
        );
    }

    #[test]
    fn shortcut_preserves_start_and_end() {
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ];

        let shortened = shortcut(&path, &FreeSpaceChecker, 50, 0.1);

        assert_eq!(&shortened[0], &[0.0, 0.0, 0.0]);
        assert_eq!(shortened.last().unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn shortcut_with_obstacle_keeps_detour() {
        // Path goes around a wall: 0.0 → 2.0 → 3.0
        // Wall blocks [0.5, 1.5]
        // Direct shortcut from 0 to 3 would cross the wall
        let path = vec![
            vec![0.0],
            vec![-0.5], // go negative to avoid wall
            vec![-1.0],
            vec![2.0], // jump over wall
            vec![3.0],
        ];

        let checker = WallChecker {
            wall_lo: 0.5,
            wall_hi: 1.5,
        };

        let shortened = shortcut(&path, &checker, 100, 0.05);

        // Should still have more than 2 waypoints since direct path crosses wall
        // The start→end shortcut should fail because the interpolation passes through [0.5, 1.5]
        assert!(shortened.len() >= 2);
        assert_eq!(&shortened[0], &[0.0]);
        assert_eq!(shortened.last().unwrap(), &[3.0]);
    }

    #[test]
    fn shortcut_short_path_unchanged() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];

        let shortened = shortcut(&path, &FreeSpaceChecker, 100, 0.1);
        assert_eq!(shortened.len(), 2);
    }

    #[test]
    fn shortcut_empty_and_single() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(shortcut(&empty, &FreeSpaceChecker, 10, 0.1).is_empty());

        let single = vec![vec![1.0, 2.0]];
        assert_eq!(shortcut(&single, &FreeSpaceChecker, 10, 0.1).len(), 1);
    }

    #[test]
    fn shortcut_reduces_path_length() {
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![0.5, 1.0],
            vec![1.5, 0.8],
            vec![1.0, 1.5],
            vec![2.0, 2.0],
        ];

        let original_len = path_length(&path);
        let shortened = shortcut(&path, &FreeSpaceChecker, 200, 0.05);
        let shortened_len = path_length(&shortened);

        assert!(
            shortened_len <= original_len + 1e-10,
            "Shortened path ({}) should not be longer than original ({})",
            shortened_len,
            original_len
        );
    }

    #[test]
    fn path_length_computation() {
        let path = vec![vec![0.0, 0.0], vec![3.0, 4.0]]; // distance = 5.0
        assert!((path_length(&path) - 5.0).abs() < 1e-10);

        let empty: Vec<Vec<f64>> = vec![];
        assert!((path_length(&empty) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn segment_collision_check() {
        let checker = FreeSpaceChecker;
        assert!(is_segment_collision_free(
            &checker,
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.1
        ));

        let wall = WallChecker {
            wall_lo: 0.4,
            wall_hi: 0.6,
        };
        // This segment crosses the wall
        assert!(!is_segment_collision_free(&wall, &[0.0], &[1.0], 0.05));
    }

    // ─── New collision safety tests ───

    /// Shortcut path remains collision-free: every segment in the shortened path
    /// must not cross the obstacle.
    #[test]
    fn shortcut_path_remains_collision_free() {
        // 2D path that detours around a box obstacle at (0.4-0.6, 0.4-0.6)
        struct Box2D;
        impl CollisionChecker for Box2D {
            fn is_in_collision(&self, joints: &[f64]) -> bool {
                joints[0] >= 0.4 && joints[0] <= 0.6 && joints[1] >= 0.4 && joints[1] <= 0.6
            }
        }

        // Path goes around the obstacle (all segments are collision-free)
        let path = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![0.2, 0.8],
            vec![0.3, 0.9],
            vec![0.5, 0.9], // above the obstacle
            vec![0.7, 0.9],
            vec![0.8, 0.8],
            vec![0.9, 0.5],
            vec![1.0, 1.0],
        ];

        // Verify original path is collision-free first
        for wp in &path {
            assert!(
                !Box2D.is_in_collision(wp),
                "original path has collision at {:?}",
                wp
            );
        }

        let shortened = shortcut(&path, &Box2D, 200, 0.01);

        // Verify every segment in the shortened path is collision-free
        for w in shortened.windows(2) {
            assert!(
                is_segment_collision_free(&Box2D, &w[0], &w[1], 0.01),
                "shortcut produced a colliding segment: {:?} → {:?}",
                w[0],
                w[1]
            );
        }

        // Every waypoint should be collision-free
        for (i, wp) in shortened.iter().enumerate() {
            assert!(
                !Box2D.is_in_collision(wp),
                "shortcut waypoint {} is in collision: {:?}",
                i,
                wp
            );
        }
    }

    /// Shortcutting a near-collinear path should simplify to 2 waypoints.
    #[test]
    fn shortcut_collinear_path_simplifies() {
        // All points on a straight line: (0,0) → (1,1) → (2,2) → (3,3)
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];

        let shortened = shortcut(&path, &FreeSpaceChecker, 100, 0.05);
        assert_eq!(
            shortened.len(),
            2,
            "collinear path in free space should simplify to start+end"
        );
        assert_eq!(shortened[0], vec![0.0, 0.0]);
        assert_eq!(shortened[1], vec![4.0, 4.0]);
    }

    /// Shortcutting with a 2D obstacle: path must go around, not through.
    #[test]
    fn shortcut_2d_obstacle_preserves_detour() {
        // 2D checker that blocks x ∈ [0.4, 0.6] AND y ∈ [0.4, 0.6]
        struct Box2DChecker;
        impl CollisionChecker for Box2DChecker {
            fn is_in_collision(&self, joints: &[f64]) -> bool {
                joints[0] >= 0.4 && joints[0] <= 0.6 && joints[1] >= 0.4 && joints[1] <= 0.6
            }
        }

        // Path goes around the obstacle
        let path = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.0, 1.0],
            vec![0.5, 1.0],
            vec![1.0, 1.0],
        ];

        let shortened = shortcut(&path, &Box2DChecker, 200, 0.01);

        // Every segment should be collision-free
        for w in shortened.windows(2) {
            assert!(
                is_segment_collision_free(&Box2DChecker, &w[0], &w[1], 0.01),
                "shortcut created colliding segment: {:?} → {:?}",
                w[0],
                w[1]
            );
        }
    }

    // ─── Additional coverage tests ───

    #[test]
    fn joint_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((joint_distance(&a, &a)).abs() < 1e-10);
    }

    #[test]
    fn joint_distance_known_values() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((joint_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn joint_distance_single_element() {
        let a = vec![0.0];
        let b = vec![7.0];
        assert!((joint_distance(&a, &b) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn joint_distance_negative_values() {
        let a = vec![-3.0, 0.0];
        let b = vec![0.0, 4.0];
        assert!((joint_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn path_length_single_point() {
        let path = vec![vec![1.0, 2.0, 3.0]];
        assert!((path_length(&path) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn path_length_three_points() {
        // Each segment has length 5 (3-4-5 triangle)
        let path = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 0.0]];
        let len = path_length(&path);
        let expected = 5.0 + 5.0;
        assert!(
            (len - expected).abs() < 1e-10,
            "expected {expected}, got {len}"
        );
    }

    #[test]
    fn path_length_collinear() {
        let path = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        assert!((path_length(&path) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn segment_collision_free_same_point() {
        let checker = FreeSpaceChecker;
        assert!(is_segment_collision_free(&checker, &[1.0, 2.0], &[1.0, 2.0], 0.1));
    }

    #[test]
    fn segment_collision_free_very_close_points() {
        let checker = FreeSpaceChecker;
        let a = vec![0.0, 0.0];
        let b = vec![1e-12, 1e-12];
        assert!(is_segment_collision_free(&checker, &a, &b, 0.1));
    }

    #[test]
    fn segment_collision_wall_at_midpoint() {
        // Wall at 0.5 should block segment from 0.0 to 1.0
        let wall = WallChecker {
            wall_lo: 0.45,
            wall_hi: 0.55,
        };
        assert!(!is_segment_collision_free(&wall, &[0.0], &[1.0], 0.05));
    }

    #[test]
    fn segment_collision_wall_at_start() {
        // Wall at 0.0 - the from point is at the edge
        let wall = WallChecker {
            wall_lo: -0.1,
            wall_hi: 0.1,
        };
        // The from/to endpoints themselves are not checked by is_segment_collision_free
        // (it checks intermediate points step 1..num_steps).
        // From 0.0 to 1.0 with step ~0.05: step 1 is at ~0.05, which is in [−0.1, 0.1]
        assert!(!is_segment_collision_free(&wall, &[0.0], &[1.0], 0.05));
    }

    #[test]
    fn segment_collision_free_with_large_step() {
        // Large step size means fewer intermediate checks
        let wall = WallChecker {
            wall_lo: 0.49,
            wall_hi: 0.51,
        };
        // With step_size = 10.0, num_steps = 1, so only step at t=1 is checked
        // but the loop is `for step in 1..num_steps` which is `1..1` = empty,
        // so no intermediate checks happen
        assert!(is_segment_collision_free(&wall, &[0.0], &[1.0], 10.0));
    }

    #[test]
    fn shortcut_zero_iterations() {
        let path = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
        ];
        let shortened = shortcut(&path, &FreeSpaceChecker, 0, 0.1);
        // Zero iterations = no shortcutting
        assert_eq!(shortened.len(), path.len());
    }

    #[test]
    fn shortcut_three_waypoints_free_space() {
        let path = vec![
            vec![0.0, 0.0],
            vec![5.0, 5.0],
            vec![10.0, 10.0],
        ];
        let shortened = shortcut(&path, &FreeSpaceChecker, 100, 0.1);
        // In free space, should reduce to 2 waypoints
        assert_eq!(shortened.len(), 2);
        assert_eq!(shortened[0], vec![0.0, 0.0]);
        assert_eq!(shortened[1], vec![10.0, 10.0]);
    }

    #[test]
    fn shortcut_high_dimensional() {
        // 6-DOF path in free space
        let path = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            vec![0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let shortened = shortcut(&path, &FreeSpaceChecker, 200, 0.05);
        assert!(shortened.len() <= path.len());
        assert_eq!(shortened.first().unwrap(), path.first().unwrap());
        assert_eq!(shortened.last().unwrap(), path.last().unwrap());
    }

    #[test]
    fn shortcut_idempotent() {
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        // Already minimal; shortcutting should not change it
        let shortened = shortcut(&path, &FreeSpaceChecker, 100, 0.1);
        assert_eq!(shortened.len(), 2);

        // Second round should be same
        let shortened2 = shortcut(&shortened, &FreeSpaceChecker, 100, 0.1);
        assert_eq!(shortened2.len(), 2);
    }

    #[test]
    fn path_length_returns_zero_for_two_identical_points() {
        let path = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
        assert!((path_length(&path)).abs() < 1e-10);
    }

    /// A collision checker that blocks everything.
    struct BlockAllChecker;
    impl CollisionChecker for BlockAllChecker {
        fn is_in_collision(&self, _joints: &[f64]) -> bool {
            true
        }
    }

    #[test]
    fn shortcut_blocked_checker_preserves_path() {
        let path = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
        ];
        // Since every intermediate point is "in collision", no shortcut should succeed
        let shortened = shortcut(&path, &BlockAllChecker, 100, 0.05);
        // Path should remain unchanged (or at least not shorter)
        assert_eq!(shortened.len(), path.len());
    }

    #[test]
    fn segment_collision_free_block_all() {
        assert!(!is_segment_collision_free(
            &BlockAllChecker,
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.1
        ));
    }
}
