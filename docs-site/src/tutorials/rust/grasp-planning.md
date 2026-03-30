# Grasp Planning

## What You'll Learn
- Create a parallel jaw gripper model with `GripperType::parallel`
- Generate grasp candidates for cylinders, cuboids, and spheres
- Rank grasps by force closure quality, distance from center of mass, or approach angle
- Filter and select the best grasp for execution

## Prerequisites
- [Collision Detection](../../concepts/collision-detection.md)
- [Inverse Kinematics](../../concepts/inverse-kinematics.md)
- [FK and IK Tutorial](fk-and-ik.md)

## Overview

Before a robot can pick up an object, it needs to know *how* to grasp it.
Kinetic's `GraspGenerator` produces ranked grasp candidates from object geometry.
Given a shape (cylinder, cuboid, sphere) and a gripper model, it generates
antipodal, top-down, and side grasps, then ranks them by quality metrics like
force closure. This tutorial generates grasps for three common object shapes and
compares ranking strategies.

## Step 1: Create a Gripper

```rust
use kinetic::grasp::{
    GraspConfig, GraspError, GraspGenerator, GraspMetric, GraspType, GripperType,
};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let gripper = GripperType::parallel(0.08, 0.03);
    let gen = GraspGenerator::new(gripper);
```

**What this does:** Creates a parallel jaw gripper model with 80 mm maximum opening and 30 mm finger depth. The `GraspGenerator` uses these dimensions to determine which grasps are geometrically feasible.

**Why:** The gripper model constrains which grasps are possible. An object wider than `max_opening` cannot be grasped. `finger_depth` determines how far the fingers wrap around the object, affecting grasp stability. Modeling the gripper accurately prevents generating grasps that would fail on real hardware.

## Step 2: Define Objects and Generate Grasps

```rust
    let objects: Vec<(&str, Shape, Isometry3<f64>)> = vec![
        (
            "Bottle (cylinder)",
            Shape::Cylinder(0.03, 0.08),         // radius 3cm, half-height 8cm
            Isometry3::translation(0.4, 0.0, 0.08),
        ),
        (
            "Box (cuboid)",
            Shape::Cuboid(0.04, 0.03, 0.02),     // half-extents: 4x3x2 cm
            Isometry3::translation(0.4, 0.2, 0.02),
        ),
        (
            "Ball (sphere)",
            Shape::Sphere(0.025),                 // radius 2.5 cm
            Isometry3::translation(0.4, -0.2, 0.025),
        ),
    ];

    for (name, shape, pose) in &objects {
        let config = GraspConfig {
            num_candidates: 50,
            rank_by: GraspMetric::ForceClosureQuality,
            ..Default::default()
        };

        match gen.from_shape(shape, pose, config) {
            Ok(grasps) => {
                println!("{}: {} grasps", name, grasps.len());
                // Show top 3
                for (i, g) in grasps.iter().take(3).enumerate() {
                    println!(
                        "  #{}: type={:?}, quality={:.3}, approach=({:.2}, {:.2}, {:.2})",
                        i + 1, g.grasp_type, g.quality,
                        g.approach_direction.x, g.approach_direction.y, g.approach_direction.z,
                    );
                }
            }
            Err(GraspError::NoGraspsFound) => {
                println!("{}: no valid grasps (object too large for gripper)", name);
            }
            Err(e) => println!("{}: error — {}", name, e),
        }
    }
```

**What this does:** For each object, calls `gen.from_shape()` to generate up to 50 grasp candidates ranked by force closure quality. Each grasp has a `grasp_type` (Antipodal, TopDown, or SideGrasp), a `quality` score (0.0 to 1.0), a `grasp_pose` (the gripper TCP pose), and an `approach_direction`.

**Why:** Different shapes produce different grasp distributions. Cylinders favor antipodal grasps around the circumference. Cuboids produce grasps aligned with each face. Spheres are hardest — only small spheres that fit within the gripper opening are graspable. Force closure quality measures how well the grasp resists arbitrary external wrenches.

## Step 3: Count Grasps by Type

```rust
        let antipodal = grasps.iter()
            .filter(|g| g.grasp_type == GraspType::Antipodal).count();
        let topdown = grasps.iter()
            .filter(|g| g.grasp_type == GraspType::TopDown).count();
        let side = grasps.iter()
            .filter(|g| g.grasp_type == GraspType::SideGrasp).count();
        println!("  Types: {} antipodal, {} top-down, {} side", antipodal, topdown, side);
```

**What this does:** Categorizes the generated grasps. Antipodal grasps apply opposing forces through the object's center. Top-down grasps approach from above. Side grasps approach from the side.

**Why:** Grasp type selection depends on the scene. If there is a table below the object, top-down grasps require enough clearance above. If there are objects on both sides, antipodal grasps along the free axis are preferred. Knowing the distribution helps you filter for the right approach direction.

## Step 4: Compare Ranking Metrics

```rust
    let bottle = Shape::Cylinder(0.03, 0.08);
    let bottle_pose = Isometry3::translation(0.4, 0.0, 0.08);

    let metrics = [
        ("Force Closure",       GraspMetric::ForceClosureQuality),
        ("Distance from CoM",   GraspMetric::DistanceFromCenterOfMass),
        ("Approach Angle",      GraspMetric::ApproachAngle),
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
```

**What this does:** Generates grasps for the same bottle using three different ranking metrics and compares which grasp type ranks first under each.

**Why:** `ForceClosureQuality` maximizes resistance to disturbances — best for heavy or slippery objects. `DistanceFromCenterOfMass` minimizes torque during lifting — best for tall objects that might tip. `ApproachAngle` prefers grasps with approach directions aligned with a preferred axis (typically vertical) — best when the scene constrains the approach direction.

## Complete Code

See `examples/grasp_planning.rs` for the full listing combining all steps above.

## What You Learned
- `GripperType::parallel(max_opening, finger_depth)` defines the gripper geometry
- `GraspGenerator::new(gripper)` creates a generator bound to the gripper model
- `gen.from_shape(&shape, &pose, config)` generates ranked grasp candidates
- Three grasp types: `Antipodal`, `TopDown`, `SideGrasp`
- Three ranking metrics: `ForceClosureQuality`, `DistanceFromCenterOfMass`, `ApproachAngle`
- Grasps are returned sorted by quality (best first)

## Try This
- Increase `num_candidates` to 200 and observe more diverse grasp orientations
- Try a shape wider than `max_opening` (e.g., `Shape::Sphere(0.05)` with 0.08 opening) and handle `GraspError::NoGraspsFound`
- Use `GraspConfig { check_collision: Some(scene.clone()), .. }` to filter out grasps that collide with the scene
- Combine with IK: for each grasp candidate, solve IK to check if the arm can reach the grasp pose

## Next
- [Pick and Place](pick-and-place.md) — using grasps in a full pick-place pipeline
- [Task Sequencing](task-sequencing.md) — composing grasps into multi-stage tasks
