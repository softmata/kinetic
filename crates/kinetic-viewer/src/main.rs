//! KINETIC 3D Viewer — fully integrated robot visualization and motion planning.
//!
//! Usage:
//!   kinetic-viewer                    # show UR5e at home pose
//!   kinetic-viewer --robot panda      # show Franka Panda
//!   kinetic-viewer --robot path.urdf  # load custom URDF
//!   kinetic-viewer --list-robots      # print available robots

use clap::Parser;
use kinetic_robot::Robot;
use kinetic_viewer::app::{run_viewer, ViewerConfig};
use kinetic_viewer::Camera;

#[derive(Parser)]
#[command(name = "kinetic-viewer", about = "KINETIC 3D robot viewer with integrated motion planning")]
struct Cli {
    /// Robot name (built-in) or path to URDF file.
    #[arg(short, long, default_value = "ur5e")]
    robot: String,

    /// List available built-in robots and exit.
    #[arg(long)]
    list_robots: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    if cli.list_robots {
        print_available_robots();
        return;
    }

    let robot = load_robot(&cli.robot);
    let camera = Camera::perspective([1.5, 1.0, 1.5], [0.0, 0.3, 0.0], 45.0);

    let config = ViewerConfig {
        title: format!("KINETIC Viewer — {}", robot.name),
        ..Default::default()
    };

    if let Err(e) = run_viewer(config, robot, camera) {
        eprintln!("Viewer error: {e}");
        std::process::exit(1);
    }
}

fn load_robot(name_or_path: &str) -> Robot {
    if std::path::Path::new(name_or_path).extension().is_some() {
        match Robot::from_urdf(name_or_path) {
            Ok(r) => return r,
            Err(e) => {
                eprintln!("Failed to load URDF '{}': {}", name_or_path, e);
                std::process::exit(1);
            }
        }
    }
    match Robot::from_name(name_or_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Robot '{}' not found: {}", name_or_path, e);
            eprintln!("Run with --list-robots to see available robots.");
            std::process::exit(1);
        }
    }
}

fn print_available_robots() {
    let candidates = [
        std::path::PathBuf::from("robot_configs"),
        std::path::PathBuf::from("kinetic/robot_configs"),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|ws| ws.join("robot_configs"))
            .unwrap_or_default(),
    ];
    for dir in &candidates {
        if dir.is_dir() {
            let mut names: Vec<String> = std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| {
                    let e = e.ok()?;
                    if e.path().join("kinetic.toml").exists() {
                        Some(e.file_name().to_string_lossy().into_owned())
                    } else {
                        None
                    }
                })
                .collect();
            names.sort();
            println!("Available robots ({}):", names.len());
            for name in &names { println!("  {name}"); }
            return;
        }
    }
    eprintln!("Could not find robot_configs/ directory.");
}
