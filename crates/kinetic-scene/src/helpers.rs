//! Internal helpers used by [`Scene`] and [`Shape`]:
//! kinematic-chain auto-detection and shape→sphere voxelization.

use nalgebra::{Isometry3, Vector3};

use kinetic_core::KineticError;
use kinetic_kinematics::KinematicChain;
use kinetic_robot::Robot;

/// Auto-detect kinematic chain from robot (planning groups or tree walk).
pub(crate) fn auto_detect_chain(robot: &Robot) -> kinetic_core::Result<KinematicChain> {
    // Try planning groups first
    if let Some((_, group)) = robot.groups.iter().next() {
        return KinematicChain::extract(robot, &group.base_link, &group.tip_link);
    }

    // Fall back: root link -> farthest leaf
    if robot.links.is_empty() {
        return Err(KineticError::NoLinks);
    }

    let root_name = &robot.links[0].name;

    // Find leaf links (links with no children)
    let mut has_child = vec![false; robot.links.len()];
    for joint in &robot.joints {
        has_child[joint.parent_link] = true;
    }

    // Find the leaf farthest from root (most joints in chain)
    let mut best_leaf = robot.links.len() - 1;
    let mut best_depth = 0;
    for (i, _) in robot.links.iter().enumerate() {
        if has_child[i] {
            continue;
        }
        // Count depth (number of joints from root to this leaf)
        let mut depth = 0;
        let mut current = i;
        while current != 0 {
            if let Some(joint_idx) = robot.links[current].parent_joint {
                depth += 1;
                current = robot.joints[joint_idx].parent_link;
            } else {
                break;
            }
        }
        if depth > best_depth {
            best_depth = depth;
            best_leaf = i;
        }
    }

    let tip_name = &robot.links[best_leaf].name;
    KinematicChain::extract(robot, root_name, tip_name)
}

// === Shape → sphere voxelization ===
// Each fn returns Vec<(x, y, z, radius)> in world frame; resolution is the voxel edge length.

pub(crate) fn cuboid_to_spheres(
    hx: f64,
    hy: f64,
    hz: f64,
    pose: &Isometry3<f64>,
    resolution: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let r = resolution / 2.0;
    let mut spheres = Vec::new();

    let nx = ((2.0 * hx / resolution).ceil() as usize).max(1);
    let ny = ((2.0 * hy / resolution).ceil() as usize).max(1);
    let nz = ((2.0 * hz / resolution).ceil() as usize).max(1);

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let lx = -hx + (2.0 * hx) * (ix as f64 + 0.5) / nx as f64;
                let ly = -hy + (2.0 * hy) * (iy as f64 + 0.5) / ny as f64;
                let lz = -hz + (2.0 * hz) * (iz as f64 + 0.5) / nz as f64;

                let world = pose * nalgebra::Point3::new(lx, ly, lz);
                spheres.push((world.x, world.y, world.z, r));
            }
        }
    }

    spheres
}

pub(crate) fn cylinder_to_spheres(
    radius: f64,
    half_height: f64,
    pose: &Isometry3<f64>,
    resolution: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let r = resolution / 2.0;
    let mut spheres = Vec::new();

    let nh = ((2.0 * half_height / resolution).ceil() as usize).max(1);
    let nr = ((2.0 * radius / resolution).ceil() as usize).max(1);

    for ih in 0..nh {
        let z = -half_height + (2.0 * half_height) * (ih as f64 + 0.5) / nh as f64;

        for ix in 0..nr {
            for iy in 0..nr {
                let x = -radius + (2.0 * radius) * (ix as f64 + 0.5) / nr as f64;
                let y = -radius + (2.0 * radius) * (iy as f64 + 0.5) / nr as f64;

                if x * x + y * y <= radius * radius {
                    let world = pose * nalgebra::Point3::new(x, y, z);
                    spheres.push((world.x, world.y, world.z, r));
                }
            }
        }
    }

    if spheres.is_empty() {
        let p = pose.translation;
        spheres.push((p.x, p.y, p.z, radius.max(half_height)));
    }

    spheres
}

pub(crate) fn half_space_to_spheres(
    normal: &Vector3<f64>,
    offset: f64,
    _pose: &Isometry3<f64>,
    _resolution: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let n = normal.normalize();
    let center = n * offset;
    let r = 2.0;

    let t1 = if n.x.abs() < 0.9 {
        Vector3::x().cross(&n).normalize()
    } else {
        Vector3::y().cross(&n).normalize()
    };
    let t2 = n.cross(&t1).normalize();

    let mut spheres = Vec::new();
    let extent = 3.0;
    let step = r * 1.5;

    let mut u = -extent;
    while u <= extent {
        let mut v = -extent;
        while v <= extent {
            let p = center + t1 * u + t2 * v - n * r;
            spheres.push((p.x, p.y, p.z, r));
            v += step;
        }
        u += step;
    }

    spheres
}
