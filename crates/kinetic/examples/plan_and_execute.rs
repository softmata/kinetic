//! Complete KINETIC workflow: plan → execute → monitor → export.
//!
//! Demonstrates KINETIC as a standalone motion planning-to-execution system.
//!
//! Run: cargo run --example plan_and_execute -p kinetic

use kinetic::prelude::*;

fn main() -> Result<()> {
    // 1. Load robot
    let robot = Robot::from_name("ur5e")?;
    println!("Loaded {} ({}DOF)", robot.name, robot.dof);

    // 2. Build scene with a table obstacle
    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.3, 0.02], [0.5, 0.0, -0.02]);
    println!("Scene: {} objects", scene.num_objects());

    // 3. Define start and goal
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal_joints = vec![1.0, -1.0, 0.5, -1.0, -0.5, 0.3];

    // 4. Plan with scene awareness
    let planner = Planner::new(&robot)?.with_scene(&scene);
    let plan_result = planner.plan(&start, &Goal::joints(goal_joints.clone()))?;
    println!(
        "Planned: {} waypoints in {:.1}ms",
        plan_result.waypoints.len(),
        plan_result.planning_time.as_secs_f64() * 1000.0
    );

    // 5. Time-parameterize (trapezoidal velocity profile)
    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &plan_result.waypoints,
        &vel_limits,
        &accel_limits,
    )
    .map_err(|e| KineticError::PlanningFailed(e))?;
    println!(
        "Trajectory: {:.3}s, {} waypoints",
        timed.duration.as_secs_f64(),
        timed.waypoints.len()
    );

    // 6. Execute with LogExecutor (records all commands)
    let mut executor = LogExecutor::new(ExecutionConfig {
        rate_hz: 100.0,
        ..Default::default()
    });
    let exec_result = executor
        .execute_and_log(&timed)
        .map_err(|e| KineticError::PlanningFailed(e.to_string()))?;
    println!(
        "Executed: {:?}, {} commands, {:.3}s",
        exec_result.state, exec_result.commands_sent, exec_result.actual_duration.as_secs_f64()
    );

    // 7. Verify all commands are within joint limits
    let commands = executor.commands();
    let mut all_valid = true;
    for cmd in commands {
        for (j, &pos) in cmd.positions.iter().enumerate() {
            if pos < robot.joint_limits[j].lower - 0.001
                || pos > robot.joint_limits[j].upper + 0.001
            {
                println!(
                    "WARNING: joint {} at {:.4} outside limits [{:.4}, {:.4}]",
                    j, pos, robot.joint_limits[j].lower, robot.joint_limits[j].upper
                );
                all_valid = false;
            }
        }
    }
    if all_valid {
        println!("All {} commands within joint limits", commands.len());
    }

    // 8. Frame tree: track coordinate frames
    let tree = FrameTree::new();
    // Set a static camera calibration
    tree.set_static_transform("base_link", "camera", Isometry3::translation(0.0, 0.0, 0.5));

    // Update from FK at final position
    let final_pose = kinetic::kinematics::forward_kinematics(
        &robot,
        &planner.chain(),
        &exec_result.final_positions,
    )?;
    println!(
        "Final EE pose: [{:.4}, {:.4}, {:.4}]",
        final_pose.translation().x,
        final_pose.translation().y,
        final_pose.translation().z
    );

    // 9. Export trajectory to JSON
    let json = trajectory_to_json(&timed);
    println!("Exported trajectory: {} bytes JSON", json.len());

    // 10. One-liner workflow with PlanExecuteLoop
    let planner2 = Planner::new(&robot)?;
    let sim_executor = Box::new(SimExecutor::default());
    let mut pel = PlanExecuteLoop::new(planner2, sim_executor);

    let pel_result = pel.move_to(&start, &Goal::joints(goal_joints))?;
    println!(
        "PlanExecuteLoop: completed in {:.1}ms, {} replans",
        pel_result.total_duration.as_secs_f64() * 1000.0,
        pel_result.replans
    );

    println!("\nKINETIC standalone workflow complete.");
    Ok(())
}
