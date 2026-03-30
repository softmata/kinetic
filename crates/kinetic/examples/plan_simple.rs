//! Simple planning — plan a joint-space path for a UR5e.
//!
//! Demonstrates the one-liner `plan()` and the `Planner` builder API.
//!
//! ```sh
//! cargo run --example plan_simple -p kinetic
//! ```

use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    println!("=== KINETIC Simple Planning ===\n");

    // --- Method 1: One-liner ---
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let result = plan("ur5e", &start, &goal)?;
    println!(
        "One-liner: {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(),
        result.planning_time,
        result.path_length(),
    );

    // --- Method 2: Planner builder ---
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?.with_config(PlannerConfig::realtime());

    let result = planner.plan(&start, &goal)?;
    println!(
        "Builder:   {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(),
        result.planning_time,
        result.path_length(),
    );

    // --- Method 3: Pose goal (Cartesian target) ---
    let target_joints = vec![0.5, -0.8, 0.5, 0.1, -0.1, 0.3];
    let target_pose = planner.fk(&target_joints)?;
    let t = target_pose.translation();
    println!(
        "\nPlanning to Cartesian pose: ({:.3}, {:.3}, {:.3})",
        t.x, t.y, t.z
    );

    let result = planner.plan(&start, &Goal::Pose(target_pose))?;
    println!(
        "Pose goal: {} waypoints, {:.1?}",
        result.num_waypoints(),
        result.planning_time,
    );

    Ok(())
}
