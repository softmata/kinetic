# 3D Visualization

The `kinetic-viewer` crate provides an interactive 3D viewer for robot
visualization, motion planning, and trajectory playback. It includes both a
windowed GUI (with egui panels) and a headless renderer for CI pipelines and
screenshot generation.

## Quick Start

Launch the viewer for a built-in robot:

```bash
cargo run -p kinetic-viewer --features cli -- --robot ur5e
```

List all 54 supported robots:

```bash
cargo run -p kinetic-viewer --features cli -- --list-robots
```

Load a custom URDF file:

```bash
cargo run -p kinetic-viewer --features cli -- --robot path/to/robot.urdf
```

## Features

### Interactive GUI

The viewer opens a wgpu-rendered window with egui side panels:

- **Joint sliders** -- drag individual joints to see the robot move in real time
- **Planning panel** -- set start/goal configurations and run RRT, PRM, or other planners
- **Servo panel** -- real-time velocity-based end-effector control
- **Constraint panel** -- visualize joint limits and workspace constraints
- **Scene objects** -- add/remove obstacles and inspection targets
- **Trajectory playback** -- animate planned paths with speed control and looping
- **Collision debug** -- toggle sphere model visualization

### Rendering

- Hardware-accelerated via wgpu (Vulkan, Metal, DX12)
- Grid floor, coordinate axes, and wireframe overlays
- Per-link coloring and transparency for ghost robots
- Trajectory trail visualization

### Headless Rendering

The `HeadlessRenderer` renders to an offscreen texture without a window. It
works with software Vulkan (lavapipe) for CI environments and with NVIDIA/AMD
GPUs for production screenshots.

## Keyboard Shortcuts

| Key   | Action                              |
|-------|-------------------------------------|
| G     | Toggle grid                         |
| A     | Toggle coordinate axes              |
| T     | Toggle trajectory trail             |
| C     | Toggle collision debug spheres      |
| W     | Toggle collision wireframe          |
| V     | Toggle voxel display                |
| P     | Toggle point cloud                  |
| R     | Reset camera to default position    |
| F     | Focus camera on goal marker         |
| Z     | Undo last interaction               |
| Space | Play/pause trajectory               |
| F1    | Show/hide keyboard shortcuts        |
| F3    | Show/hide stats overlay             |
| Esc   | Quit                                |

Mouse controls: left-drag to orbit, right-drag to pan, scroll to zoom.

## Screenshots

Headless screenshots of robots at their zero joint configuration, rendered
using collision geometry primitives with per-link color gradients.

| Robot | Screenshot |
|-------|------------|
| UR5e | ![UR5e](../../crates/kinetic-viewer/screenshots/ur5e.png) |
| Franka Panda | ![Franka Panda](../../crates/kinetic-viewer/screenshots/franka_panda.png) |
| KUKA iiwa14 | ![KUKA iiwa14](../../crates/kinetic-viewer/screenshots/kuka_iiwa14.png) |
| xArm7 | ![xArm7](../../crates/kinetic-viewer/screenshots/xarm7.png) |

To regenerate screenshots:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  cargo run -p kinetic-viewer --example capture_screenshots --features visual
```

## API Usage

### Embedding in Custom Applications

Use `kinetic-viewer` as a library to render robots in your own application:

```rust
use kinetic_viewer::{robot_scene_graph, Camera, MeshRegistry, ViewerSettings};
use kinetic_viewer::app::{run_viewer, ViewerConfig};
use kinetic_robot::Robot;

let robot = Robot::from_name("ur5e")?;
let camera = Camera::perspective([1.5, 1.0, 1.5], [0.0, 0.3, 0.0], 45.0);
let config = ViewerConfig {
    title: "My Robot Viewer".into(),
    ..Default::default()
};
run_viewer(config, robot, camera)?;
```

### Headless Rendering (Screenshots, CI)

Render to PNG without a window:

```rust
use kinetic_viewer::test_utils::HeadlessRenderer;
use kinetic_viewer::{robot_scene_graph, Camera, MeshRegistry, ViewerSettings};
use kinetic_robot::Robot;

let robot = Robot::from_name("ur5e").unwrap();
let mut meshes = MeshRegistry::new();
let scene = robot_scene_graph(&robot, &mut meshes);
let camera = Camera::perspective([1.5, 1.0, 1.5], [0.0, 0.3, 0.0], 45.0);
let settings = ViewerSettings::default();

let mut renderer = HeadlessRenderer::new(1280, 720).unwrap();
renderer.upload_meshes(&meshes);
let pixels = renderer.render(&scene, &camera, &settings);

// Save as PNG (requires `image` crate)
image::save_buffer(
    "screenshot.png", &pixels, 1280, 720,
    image::ColorType::Rgba8,
).unwrap();
```

### Visual Regression Testing

Compare two renders for pixel-level differences:

```rust
use kinetic_viewer::test_utils::{compare_screenshots, diff_image};

let result = compare_screenshots(&actual, &expected, 1280, 720, 2);
assert!(result.rmse < 5.0, "Visual regression: RMSE={}", result.rmse);

// Generate a red/green diff image for inspection
let diff = diff_image(&actual, &expected, 1280, 720);
```

## Crate Features

| Feature   | Description                                    |
|-----------|------------------------------------------------|
| `visual`  | GPU rendering, mesh loading, image export      |
| `cli`     | Command-line interface (`--robot`, `--list-robots`) |

The `visual` feature is enabled by default. Use `--no-default-features` for
data-model-only usage (scene graph, camera math, render commands) without GPU
dependencies.

## Architecture

```
kinetic-viewer
  +-- lib.rs         Scene graph, Camera, MeshData, ViewerSettings, RenderCommand
  +-- app.rs         Windowed viewer (winit + wgpu + egui)
  +-- gpu_context.rs wgpu device/surface setup
  +-- pipeline.rs    Mesh, wireframe, and line render pipelines
  +-- gpu_buffers.rs GPU buffer management (instances, lines, uniforms)
  +-- test_utils.rs  HeadlessRenderer + screenshot comparison
  +-- egui_ui.rs     Joint sliders, planning panel, servo panel
  +-- gizmo.rs       3D gizmo interaction (translate/rotate handles)
  +-- interaction.rs Markers, selection, undo
  +-- collision_viz.rs  Collision sphere debug overlays
  +-- trajectory_viz.rs Trajectory playback + ghost robots
  +-- perception_viz.rs Point cloud + voxel display
  +-- web_export.rs  Scene export for web viewers
```
