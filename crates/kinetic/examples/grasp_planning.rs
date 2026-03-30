//! Grasp planning — generate and rank grasp candidates for common objects.
//!
//! Demonstrates the `kinetic-grasp` module:
//! - Create a parallel jaw gripper
//! - Generate grasps for a cylinder (bottle), cuboid (box), and sphere (ball)
//! - Rank by force closure quality
//! - Print top candidates with type, quality, and approach direction
//!
//! ```sh
//! cargo run --example grasp_planning -p kinetic
//! ```

use kinetic::grasp::{
    GraspConfig, GraspError, GraspGenerator, GraspMetric, GraspType, GripperType,
};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    // 1. Create a parallel jaw gripper
    //    - max_opening: 8 cm (finger gap)
    //    - finger_depth: 3 cm (how far fingers extend past the object)
    let gripper = GripperType::parallel(0.08, 0.03);
    let gen = GraspGenerator::new(gripper);

    println!("=== KINETIC Grasp Planning Demo ===");
    println!("Gripper: parallel jaw, 80mm opening, 30mm finger depth\n");

    // 2. Define three object shapes at different poses
    let objects: Vec<(&str, Shape, Isometry3<f64>)> = vec![
        (
            "Bottle (cylinder)",
            Shape::Cylinder(0.03, 0.08), // radius 3cm, half-height 8cm
            Isometry3::translation(0.4, 0.0, 0.08),
        ),
        (
            "Box (cuboid)",
            Shape::Cuboid(0.04, 0.03, 0.02), // half-extents: 4x3x2 cm
            Isometry3::translation(0.4, 0.2, 0.02),
        ),
        (
            "Ball (sphere)",
            Shape::Sphere(0.025), // radius 2.5 cm
            Isometry3::translation(0.4, -0.2, 0.025),
        ),
    ];

    // 3. For each shape: generate, rank, and display grasps
    for (name, shape, pose) in &objects {
        println!("--- {} ---", name);
        let pos = pose.translation.vector;
        println!("  Position: ({:.2}, {:.2}, {:.2})", pos.x, pos.y, pos.z);

        let config = GraspConfig {
            num_candidates: 50,
            rank_by: GraspMetric::ForceClosureQuality,
            ..Default::default()
        };

        match gen.from_shape(shape, pose, config) {
            Ok(grasps) => {
                println!("  Generated {} grasp candidates", grasps.len());

                // Count by type
                let antipodal = grasps
                    .iter()
                    .filter(|g| g.grasp_type == GraspType::Antipodal)
                    .count();
                let topdown = grasps
                    .iter()
                    .filter(|g| g.grasp_type == GraspType::TopDown)
                    .count();
                let side = grasps
                    .iter()
                    .filter(|g| g.grasp_type == GraspType::SideGrasp)
                    .count();
                println!(
                    "  Types: {} antipodal, {} top-down, {} side",
                    antipodal, topdown, side
                );

                // Show top 3 grasps
                let top_n = grasps.len().min(3);
                println!("  Top {} grasps:", top_n);
                for (i, g) in grasps.iter().take(top_n).enumerate() {
                    let gp = g.grasp_pose.translation.vector;
                    let ad = g.approach_direction;
                    println!(
                        "    #{}: type={:?}, quality={:.3}, approach=({:.2}, {:.2}, {:.2}), pos=({:.3}, {:.3}, {:.3})",
                        i + 1,
                        g.grasp_type,
                        g.quality,
                        ad.x, ad.y, ad.z,
                        gp.x, gp.y, gp.z,
                    );
                }
            }
            Err(GraspError::NoGraspsFound) => {
                println!("  No valid grasps found (object too large for gripper)");
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
        println!();
    }

    // 4. Demonstrate filtering with different ranking metrics
    println!("--- Ranking Comparison (Bottle) ---");
    let bottle = Shape::Cylinder(0.03, 0.08);
    let bottle_pose = Isometry3::translation(0.4, 0.0, 0.08);

    let metrics = [
        ("Force Closure", GraspMetric::ForceClosureQuality),
        ("Distance from CoM", GraspMetric::DistanceFromCenterOfMass),
        ("Approach Angle", GraspMetric::ApproachAngle),
    ];

    for (metric_name, metric) in &metrics {
        let config = GraspConfig {
            num_candidates: 20,
            rank_by: *metric,
            ..Default::default()
        };

        if let Ok(grasps) = gen.from_shape(&bottle, &bottle_pose, config) {
            let best = &grasps[0];
            println!(
                "  {}: best quality={:.3}, type={:?}",
                metric_name, best.quality, best.grasp_type,
            );
        }
    }

    Ok(())
}
