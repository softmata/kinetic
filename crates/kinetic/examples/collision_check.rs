//! Collision checking — demonstrate sphere-model collision detection.
//!
//! Creates a robot sphere model, updates it with FK poses, and checks
//! for self-collision and environment collision.
//!
//! ```sh
//! cargo run --example collision_check -p kinetic
//! ```

use kinetic::collision::{adjacent_link_pairs, RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic::prelude::*;

const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link">
    <visual>
      <geometry><cylinder radius="0.05" length="0.1"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.05" length="0.1"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <visual>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </collision>
  </link>
  <link name="link2">
    <visual>
      <geometry><cylinder radius="0.035" length="0.25"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.035" length="0.25"/></geometry>
    </collision>
  </link>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
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
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF)?;
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link")?;
    println!("Loaded '{}' — {} DOF", robot.name, robot.dof);

    // Create sphere model from collision geometry
    let sphere_config = SphereGenConfig::default();
    let sphere_model = RobotSphereModel::from_robot(&robot, &sphere_config);
    let mut spheres = sphere_model.create_runtime();

    println!("Sphere model: {} total spheres", spheres.world.len());

    // Test several configurations
    let configs = [
        ("Zero config", vec![0.0, 0.0, 0.0]),
        ("Straight up", vec![0.0, 0.0, 0.0]),
        ("Bent forward", vec![0.0, 1.0, -0.5]),
        ("Max bend", vec![0.0, 1.5, -1.5]),
    ];

    for (name, q) in &configs {
        // Compute FK for all links
        let link_poses = forward_kinematics_all(&robot, &chain, q)?;

        // Update sphere positions from FK
        spheres.update(&link_poses);

        // Check self-collision (skip adjacent link pairs)
        let skip_pairs = adjacent_link_pairs(&robot);
        let self_collision = spheres.self_collision(&skip_pairs);

        // Check environment collision against some obstacles
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.5, 0.0, 0.3, 0.1, 0); // sphere at (0.5, 0, 0.3) r=0.1

        let env_collision = spheres.collides_with(&obstacles);

        let ee_pose = forward_kinematics(&robot, &chain, q)?;
        let t = ee_pose.translation();

        println!(
            "  {}: EE=({:.3}, {:.3}, {:.3})  self_col={:<5}  env_col={}",
            name, t.x, t.y, t.z, self_collision, env_collision
        );
    }

    // Demonstrate SIMD collision checking
    let tier = kinetic::collision::simd::detect_simd_tier();
    println!("\nSIMD tier: {:?}", tier);

    // Benchmark collision speed
    let q = vec![0.0, 0.5, -0.3];
    let link_poses = forward_kinematics_all(&robot, &chain, &q)?;
    spheres.update(&link_poses);

    let mut obstacles = SpheresSoA::new();
    for i in 0..10 {
        let x = 0.3 + (i as f64) * 0.05;
        obstacles.push(x, 0.0, 0.3, 0.02, i);
    }

    let start = std::time::Instant::now();
    let iters = 10_000;
    for _ in 0..iters {
        std::hint::black_box(spheres.collides_with(&obstacles));
    }
    let elapsed = start.elapsed();
    let per_check = elapsed / iters as u32;
    println!(
        "\nCollision check: {:?}/check ({} checks)",
        per_check, iters
    );

    Ok(())
}
