//! Hello World — FK → IK roundtrip in 5 lines.
//!
//! ```sh
//! cargo run --example hello_fk_ik -p kinetic
//! ```

use kinetic::prelude::*;

/// Inline Panda-like URDF for demo (in real usage: Robot::from_urdf("panda.urdf"))
const PANDA_URDF: &str = r#"<?xml version="1.0"?>
<robot name="panda_like">
  <link name="panda_link0"/>
  <link name="panda_link1"/>
  <link name="panda_link2"/>
  <link name="panda_link3"/>
  <link name="panda_link4"/>
  <link name="panda_link5"/>
  <link name="panda_link6"/>
  <link name="panda_link7"/>
  <link name="panda_link8"/>

  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <origin xyz="0 0 0.333"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint2" type="revolute">
    <parent link="panda_link1"/>
    <child link="panda_link2"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.8326" upper="1.8326" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint3" type="revolute">
    <parent link="panda_link2"/>
    <child link="panda_link3"/>
    <origin xyz="0 -0.316 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint4" type="revolute">
    <parent link="panda_link3"/>
    <child link="panda_link4"/>
    <origin xyz="0.0825 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0718" upper="-0.0698" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint5" type="revolute">
    <parent link="panda_link4"/>
    <child link="panda_link5"/>
    <origin xyz="-0.0825 0.384 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint6" type="revolute">
    <parent link="panda_link5"/>
    <child link="panda_link6"/>
    <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.0175" upper="3.7525" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint7" type="revolute">
    <parent link="panda_link6"/>
    <child link="panda_link7"/>
    <origin xyz="0.088 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint8" type="fixed">
    <parent link="panda_link7"/>
    <child link="panda_link8"/>
    <origin xyz="0 0 0.107"/>
  </joint>
</robot>
"#;

fn main() -> kinetic::core::Result<()> {
    // 1. Load robot
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    println!("Loaded '{}' — {} DOF", robot.name, robot.dof);

    // 2. Extract kinematic chain
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

    // 3. FK: compute end-effector pose from joint angles
    let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let start = std::time::Instant::now();
    let pose = forward_kinematics(&robot, &chain, &q)?;
    let fk_time = start.elapsed();

    let t = pose.translation();
    println!(
        "FK → position: ({:.4}, {:.4}, {:.4})  [{:?}]",
        t.x, t.y, t.z, fk_time
    );

    // 4. IK: recover joint angles from pose
    let config = IKConfig::dls()
        .with_seed(robot.mid_configuration().to_vec())
        .with_max_iterations(300);

    let start = std::time::Instant::now();
    let solution = solve_ik(&robot, &chain, &pose, &config)?;
    let ik_time = start.elapsed();

    println!(
        "IK → converged in {} iters, pos_err={:.2e}, orient_err={:.2e}  [{:?}]",
        solution.iterations, solution.position_error, solution.orientation_error, ik_time
    );

    // 5. Verify roundtrip
    let recovered_pose = forward_kinematics(&robot, &chain, &solution.joints)?;
    let rt = recovered_pose.translation();
    println!("Roundtrip FK → ({:.4}, {:.4}, {:.4})", rt.x, rt.y, rt.z);

    let pos_diff = (t - rt).norm();
    println!("Position roundtrip error: {:.2e} m", pos_diff);

    Ok(())
}
