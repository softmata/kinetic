//! Servo loop — real-time control with twist commands.
//!
//! Simulates a 2-second servo control loop at 100 Hz:
//! - First second: move forward along the X axis
//! - Second second: move sideways along the Y axis
//!
//! Prints joint positions, collision scale factor, and singularity
//! metric at each step, then shows the final end-effector position.
//!
//! ```sh
//! cargo run --example servo_loop -p kinetic
//! ```

use std::sync::Arc;

use kinetic::core::Twist;
use kinetic::prelude::*;
use kinetic::reactive::servo::{Servo, ServoConfig};

/// Inline 6-DOF URDF with collision geometry for servo demonstration.
///
/// Modeled after a UR5e-like kinematic structure with standard industrial
/// axis pattern (Z-Y-Y-Z-Y-Z) and realistic link lengths (~0.85m reach).
const ARM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="servo_demo_arm">
  <link name="base_link">
    <collision><geometry><cylinder radius="0.06" length="0.15"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.05" length="0.425"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.05" length="0.392"/></geometry></collision>
  </link>
  <link name="link3">
    <collision><geometry><cylinder radius="0.04" length="0.1"/></geometry></collision>
  </link>
  <link name="link4">
    <collision><geometry><cylinder radius="0.035" length="0.1"/></geometry></collision>
  </link>
  <link name="link5">
    <collision><geometry><cylinder radius="0.03" length="0.1"/></geometry></collision>
  </link>
  <link name="ee_link"/>

  <!-- Joint 1: Base rotation (Z axis) -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.15"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.175" effort="87"/>
  </joint>
  <!-- Joint 2: Shoulder lift (Y axis) -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.175" effort="87"/>
  </joint>
  <!-- Joint 3: Elbow (Y axis) -->
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0 0 0.425"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.175" effort="87"/>
  </joint>
  <!-- Joint 4: Wrist 1 (Z axis) -->
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/>
    <origin xyz="0 0 0.392"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.07" upper="3.07" velocity="2.61" effort="12"/>
  </joint>
  <!-- Joint 5: Wrist 2 (Y axis) -->
  <joint name="joint5" type="revolute">
    <parent link="link4"/><child link="link5"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.61" effort="12"/>
  </joint>
  <!-- Joint 6: Wrist 3 (Z axis) -->
  <joint name="joint6" type="revolute">
    <parent link="link5"/><child link="ee_link"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.61" effort="12"/>
  </joint>
</robot>
"#;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 1. Load robot and create scene with a table obstacle
    let robot = Arc::new(Robot::from_urdf_string(ARM_URDF)?);
    println!("Loaded '{}' — {} DOF", robot.name, robot.dof);

    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.4, 0.01], [0.6, 0.0, 0.3]);
    let scene = Arc::new(scene);

    // 2. Create Servo controller at 100 Hz
    let config = ServoConfig {
        rate_hz: 100.0,
        singularity_threshold: 0.005, // lower for this compact demo robot
        ..Default::default()
    };
    let mut servo = Servo::new(&robot, &scene, config)?;

    // 3. Set initial joint state (well-bent elbow configuration for good manipulability)
    let initial_joints = vec![0.0, -0.8, 1.0, 0.0, -0.5, 0.0];
    let initial_vel = vec![0.0; robot.dof];
    servo.set_state(&initial_joints, &initial_vel)?;

    let ee = servo.state().ee_pose.translation.vector;
    println!("Start EE position: ({:.4}, {:.4}, {:.4})", ee.x, ee.y, ee.z);
    println!();

    // 4. Phase 1: Move forward along X for 1 second (100 steps at 100 Hz)
    let forward_twist = Twist::new(
        Vector3::new(0.05, 0.0, 0.0), // 5 cm/s forward
        Vector3::zeros(),
    );

    println!("=== Phase 1: Forward motion (X-axis, 1s) ===");
    for step in 0..100 {
        let cmd = servo.send_twist(&forward_twist)?;
        let state = servo.state();

        if step % 20 == 0 {
            print!("  t={:.2}s  joints=[", step as f64 / 100.0);
            for (i, p) in cmd.positions.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.3}", p);
            }
            println!("]");
            println!(
                "          manip={:.4}  near_singularity={}  near_collision={}",
                state.manipulability, state.is_near_singularity, state.is_near_collision,
            );
        }
    }

    let ee = servo.state().ee_pose.translation.vector;
    println!(
        "  After phase 1: EE=({:.4}, {:.4}, {:.4})",
        ee.x, ee.y, ee.z
    );
    println!();

    // 5. Phase 2: Move sideways along Y for 1 second (100 steps at 100 Hz)
    let sideways_twist = Twist::new(
        Vector3::new(0.0, 0.05, 0.0), // 5 cm/s sideways
        Vector3::zeros(),
    );

    println!("=== Phase 2: Sideways motion (Y-axis, 1s) ===");
    for step in 0..100 {
        let cmd = servo.send_twist(&sideways_twist)?;
        let state = servo.state();

        if step % 20 == 0 {
            print!("  t={:.2}s  joints=[", 1.0 + step as f64 / 100.0);
            for (i, p) in cmd.positions.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.3}", p);
            }
            println!("]");
            println!(
                "          manip={:.4}  near_singularity={}  near_collision={}",
                state.manipulability, state.is_near_singularity, state.is_near_collision,
            );
        }
    }

    // 6. Print final state
    let final_state = servo.state();
    let ee = final_state.ee_pose.translation.vector;
    println!();
    println!("=== Final State ===");
    println!("  EE position: ({:.4}, {:.4}, {:.4})", ee.x, ee.y, ee.z);
    println!("  Manipulability: {:.4}", final_state.manipulability);
    println!("  Near singularity: {}", final_state.is_near_singularity);
    println!("  Near collision: {}", final_state.is_near_collision);
    println!(
        "  Min obstacle distance: {:.4} m",
        final_state.min_obstacle_distance
    );

    Ok(())
}
