//! Capture headless screenshots of robot models.
//!
//! Renders each robot to a 1280x720 PNG using the HeadlessRenderer (no window).
//! Output goes to `crates/kinetic-viewer/screenshots/{name}.png`.
//!
//! Usage:
//!   cargo run -p kinetic-viewer --example capture_screenshots --features visual
//!
//! With a specific Vulkan driver:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
//!     cargo run -p kinetic-viewer --example capture_screenshots --features visual

use kinetic_core::Pose;
use kinetic_kinematics::{forward_kinematics_all, KinematicChain};
use kinetic_robot::Robot;
use kinetic_viewer::test_utils::HeadlessRenderer;
use kinetic_viewer::{
    robot_scene_graph, update_robot_transforms, Camera, Material, MeshRegistry, SceneNode,
    ViewerSettings,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Per-link color palette: gradient from steel-blue (base) to warm-orange (tip).
const LINK_COLORS: &[[f32; 3]] = &[
    [0.30, 0.45, 0.70],
    [0.35, 0.50, 0.75],
    [0.40, 0.55, 0.65],
    [0.50, 0.60, 0.55],
    [0.60, 0.60, 0.45],
    [0.70, 0.55, 0.35],
    [0.80, 0.50, 0.30],
    [0.85, 0.45, 0.25],
    [0.90, 0.40, 0.20],
    [0.95, 0.35, 0.15],
];

/// Robots to screenshot. Only robots with collision geometry in their URDFs
/// will produce visible shapes. Robots without collision geometry are skipped.
const ROBOTS: &[&str] = &[
    "ur5e",
    "franka_panda",
    "kuka_iiwa14",
    "xarm7",
    "kinova_gen3",
];

/// Apply distinct per-link colors to the scene graph for better visibility.
fn colorize_links(root: &mut SceneNode) {
    for (i, link_node) in root.children.iter_mut().enumerate() {
        let color = LINK_COLORS[i % LINK_COLORS.len()];
        for geom_node in link_node.children.iter_mut() {
            geom_node.material = Material::solid(color[0], color[1], color[2]);
        }
    }
}

/// Compute a camera position that frames the robot's FK poses nicely.
fn auto_camera(link_poses: &[Pose]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];

    for pose in link_poses {
        let t = pose.0.translation.vector;
        for i in 0..3 {
            min[i] = min[i].min(t[i]);
            max[i] = max[i].max(t[i]);
        }
    }

    let center = [
        ((min[0] + max[0]) / 2.0) as f32,
        ((min[1] + max[1]) / 2.0) as f32,
        ((min[2] + max[2]) / 2.0) as f32,
    ];

    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let max_extent = extent[0].max(extent[1]).max(extent[2]).max(0.3) as f32;
    let dist = max_extent * 1.8;

    let cam_pos = [
        center[0] + dist * 0.7,
        center[1] + dist * 0.5,
        center[2] + dist * 0.7,
    ];

    (cam_pos, center)
}

fn main() {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("screenshots");
    std::fs::create_dir_all(&out_dir).expect("create screenshots dir");

    let settings = ViewerSettings {
        grid_size: 1.0,
        grid_divisions: 10,
        ..ViewerSettings::default()
    };

    let mut success = 0;
    let mut skipped = Vec::new();

    for &name in ROBOTS {
        print!("  {name} ... ");

        let robot = match Robot::from_name(name) {
            Ok(r) => r,
            Err(e) => {
                println!("SKIP (load failed: {e})");
                skipped.push(name);
                continue;
            }
        };

        let chain = match KinematicChain::auto_detect(&robot) {
            Ok(c) => c,
            Err(e) => {
                println!("SKIP (chain detect failed: {e})");
                skipped.push(name);
                continue;
            }
        };

        let joint_values = vec![0.0f64; chain.dof];
        let link_poses = forward_kinematics_all(&robot, &chain, &joint_values)
            .unwrap_or_else(|_| vec![Pose::identity(); robot.links.len()]);

        let collision_count: usize = robot
            .links
            .iter()
            .map(|l| l.collision_geometry.len())
            .sum();
        if collision_count == 0 {
            println!("SKIP (no collision geometry in URDF)");
            skipped.push(name);
            continue;
        }

        let mut meshes = MeshRegistry::new();
        let mut scene = robot_scene_graph(&robot, &mut meshes);
        update_robot_transforms(&mut scene, &link_poses);
        colorize_links(&mut scene);

        let (cam_pos, cam_target) = auto_camera(&link_poses);
        let camera = Camera::perspective(cam_pos, cam_target, 45.0);

        let mut renderer = match HeadlessRenderer::new(WIDTH, HEIGHT) {
            Ok(r) => r,
            Err(e) => {
                println!("SKIP (renderer init failed: {e})");
                skipped.push(name);
                continue;
            }
        };
        renderer.upload_meshes(&meshes);
        let pixels = renderer.render(&scene, &camera, &settings);

        let path = out_dir.join(format!("{name}.png"));
        image::save_buffer(&path, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
            .expect("save PNG");

        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!("OK ({} bytes)", file_size);
        success += 1;
    }

    println!("\nDone: {success}/{} rendered", ROBOTS.len());
    if !skipped.is_empty() {
        println!("Skipped: {}", skipped.join(", "));
    }
}
