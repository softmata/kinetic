//! Planning with obstacles — plan around scene objects.
//!
//! Demonstrates the Scene + Planner workflow:
//! 1. Load a robot
//! 2. Create a scene with obstacles
//! 3. Plan a collision-free path
//! 4. Time-parameterize the trajectory
//!
//! ```sh
//! cargo run --example plan_with_obstacles -p kinetic
//! ```

use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    println!("=== KINETIC Planning with Obstacles ===\n");

    // 1. Load robot
    let robot = Robot::from_name("ur5e")?;
    println!("Loaded '{}' ({} DOF)", robot.name, robot.dof);

    // 2. Create a scene with obstacles
    let mut scene = Scene::new(&robot)?;

    // Add a table surface
    scene.add_box("table", [0.4, 0.3, 0.01], [0.5, 0.0, 0.0]);

    // Add a box obstacle on the table
    scene.add_box("box", [0.05, 0.05, 0.1], [0.4, 0.0, 0.11]);

    // Add a cylindrical obstacle
    scene.add_cylinder("pipe", 0.03, 0.15, [0.3, 0.1, 0.15]);

    println!("Scene: {} obstacles", scene.num_objects());

    // 3. Plan around obstacles
    let planner = Planner::new(&robot)?
        .with_scene(&scene)
        .with_config(PlannerConfig::default());

    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let result = planner.plan(&start, &goal)?;
    println!(
        "\nPath found: {} waypoints in {:.1?}",
        result.num_waypoints(),
        result.planning_time,
    );
    println!("Path length: {:.3} rad", result.path_length());

    // 4. Time-parameterize with trapezoidal velocity profile
    let max_vel = 2.0; // rad/s
    let max_acc = 5.0; // rad/s^2
    let timed = trapezoidal(&result.waypoints, max_vel, max_acc).map_err(KineticError::Other)?;

    println!("\nTrajectory duration: {:.3?}", timed.duration());
    println!("Timed waypoints: {}", timed.waypoints.len());

    // Print first and last waypoints
    if let Some(first) = timed.waypoints.first() {
        println!("  t=0.000s: {:6.3?}", &first.positions[..3]);
    }
    if let Some(last) = timed.waypoints.last() {
        println!("  t={:.3}s: {:6.3?}", last.time, &last.positions[..3]);
    }

    Ok(())
}
