//! Multiple IK solutions — demonstrate random restarts finding different configurations.
//!
//! A 7-DOF robot has infinite IK solutions for most targets (1 redundant DOF).
//! Using random restarts, we can discover multiple distinct joint configurations
//! that all reach the same end-effector pose.
//!
//! ```sh
//! cargo run --example multiple_ik -p kinetic
//! ```

use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("panda_urdf.txt");

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

    // Pick a target pose
    let q_target = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let target = forward_kinematics(&robot, &chain, &q_target)?;
    let t = target.translation();
    println!("Target pose: ({:.4}, {:.4}, {:.4})", t.x, t.y, t.z);
    println!("---");

    // Find multiple solutions using different seeds
    let seeds = [
        robot.mid_configuration().to_vec(),
        vec![0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
        vec![1.0, -1.0, 1.0, -2.0, 0.5, 2.0, -0.5],
        vec![-0.5, 0.5, -0.3, -1.0, -0.5, 0.5, 1.0],
    ];

    let mut solutions = Vec::new();

    for (i, seed) in seeds.iter().enumerate() {
        let config = IKConfig::dls()
            .with_seed(seed.clone())
            .with_max_iterations(300);

        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) => {
                println!(
                    "Solution #{}: {} iters, pos_err={:.2e}",
                    i + 1,
                    sol.iterations,
                    sol.position_error
                );
                print!("  joints: [");
                for (j, &v) in sol.joints.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    print!("{:.3}", v);
                }
                println!("]");
                solutions.push(sol);
            }
            Err(e) => {
                println!("Solution #{}: failed — {}", i + 1, e);
            }
        }
    }

    // Compare solutions pairwise
    if solutions.len() >= 2 {
        println!("\n--- Solution distances (L2 in joint space) ---");
        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let dist: f64 = solutions[i]
                    .joints
                    .iter()
                    .zip(solutions[j].joints.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!("  #{} ↔ #{}: {:.4} rad", i + 1, j + 1, dist);
            }
        }
    }

    Ok(())
}
