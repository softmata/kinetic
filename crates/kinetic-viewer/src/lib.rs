//! 3D scene viewer data model and rendering abstraction for KINETIC.
//!
//! Provides the scene graph, camera, mesh loading, and render data structures
//! that a wgpu (or other) renderer consumes. This crate is renderer-agnostic —
//! it produces render commands/data, not pixels.
//!
//! # Architecture
//!
//! ```text
//! SceneGraph → RenderList → [wgpu renderer] → pixels
//!     ↑
//! RobotModel (URDF meshes)
//! TrajectoryPlayer (animation)
//! CollisionDisplay (wireframes)
//! VoxelDisplay (point clouds)
//! ```

#[cfg(feature = "visual")]
pub mod app;
#[cfg(feature = "visual")]
pub mod collision_viz;
pub mod egui_ui;
#[cfg(feature = "visual")]
pub mod gpu_buffers;
#[cfg(feature = "visual")]
pub mod gizmo;
#[cfg(feature = "visual")]
pub mod gpu_context;
pub mod interaction;
#[cfg(feature = "visual")]
pub mod perception_viz;
#[cfg(feature = "visual")]
pub mod pipeline;
#[cfg(feature = "visual")]
pub mod test_utils;
#[cfg(feature = "visual")]
pub mod trajectory_viz;
pub mod web_export;
pub mod dryrun_renderer;

use nalgebra::{Isometry3, Matrix4, Point3, Vector3};

use kinetic_core::{JointValues, Pose, Trajectory};
use kinetic_robot::{GeometryShape, Robot};

// ═══════════════════════════════════════════════════════════════════════════
// Scene Graph
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the scene graph.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub name: String,
    pub transform: Isometry3<f64>,
    pub mesh: Option<MeshHandle>,
    pub material: Material,
    pub visible: bool,
    pub children: Vec<SceneNode>,
}

impl SceneNode {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            transform: Isometry3::identity(),
            mesh: None,
            material: Material::default(),
            visible: true,
            children: Vec::new(),
        }
    }

    pub fn with_mesh(mut self, mesh: MeshHandle) -> Self { self.mesh = Some(mesh); self }
    pub fn with_transform(mut self, t: Isometry3<f64>) -> Self { self.transform = t; self }
    pub fn with_material(mut self, m: Material) -> Self { self.material = m; self }
    pub fn add_child(&mut self, child: SceneNode) { self.children.push(child); }
}

/// Handle to a loaded mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub usize);

/// Material properties for rendering.
#[derive(Debug, Clone)]
pub struct Material {
    pub color: [f32; 4],       // RGBA
    pub metallic: f32,
    pub roughness: f32,
    pub wireframe: bool,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self { color: [0.7, 0.7, 0.7, 1.0], metallic: 0.0, roughness: 0.8, wireframe: false, double_sided: false }
    }
}

impl Material {
    pub fn solid(r: f32, g: f32, b: f32) -> Self { Self { color: [r, g, b, 1.0], ..Default::default() } }
    pub fn transparent(r: f32, g: f32, b: f32, a: f32) -> Self { Self { color: [r, g, b, a], ..Default::default() } }
    pub fn wireframe(r: f32, g: f32, b: f32) -> Self { Self { color: [r, g, b, 1.0], wireframe: true, ..Default::default() } }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mesh Data
// ═══════════════════════════════════════════════════════════════════════════

/// CPU-side mesh data ready for upload to GPU.
#[derive(Debug, Clone)]
pub struct MeshData {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn num_vertices(&self) -> usize { self.vertices.len() }
    pub fn num_triangles(&self) -> usize { self.indices.len() / 3 }
    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }

    /// Generate a unit cube mesh.
    pub fn cube() -> Self {
        let v = [
            [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
            [-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5],
        ];
        let vertices: Vec<[f32; 3]> = v.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        let normals = vec![[0.0, 0.0, 1.0]; 8]; // simplified
        let indices = vec![
            0,1,2,0,2,3, 4,6,5,4,7,6, 0,5,1,0,4,5, 2,7,3,2,6,7, 0,3,7,0,7,4, 1,5,6,1,6,2,
        ];
        Self { vertices, normals, indices }
    }

    /// Generate a sphere mesh.
    pub fn sphere(segments: usize, rings: usize) -> Self {
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..=segments {
                let theta = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
                let x = phi.sin() * theta.cos();
                let y = phi.cos();
                let z = phi.sin() * theta.sin();
                vertices.push([x * 0.5, y * 0.5, z * 0.5]);
                normals.push([x, y, z]);
            }
        }

        for r in 0..rings {
            for s in 0..segments {
                let a = r * (segments + 1) + s;
                let b = a + segments + 1;
                indices.extend_from_slice(&[a as u32, b as u32, (a + 1) as u32]);
                indices.extend_from_slice(&[(a + 1) as u32, b as u32, (b + 1) as u32]);
            }
        }

        Self { vertices, normals, indices }
    }

    /// Generate a cylinder mesh.
    pub fn cylinder(segments: usize) -> Self {
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let x = theta.cos() * 0.5;
            let z = theta.sin() * 0.5;
            vertices.push([x, -0.5, z]); normals.push([theta.cos(), 0.0, theta.sin()]);
            vertices.push([x,  0.5, z]); normals.push([theta.cos(), 0.0, theta.sin()]);
        }

        for i in 0..segments {
            let a = (i * 2) as u32;
            indices.extend_from_slice(&[a, a+2, a+1, a+1, a+2, a+3]);
        }

        Self { vertices, normals, indices }
    }

    /// Load mesh from an STL file. Returns a placeholder cube on failure.
    pub fn from_mesh_file(filename: &str, scale: [f64; 3]) -> Self {
        // Try to load as STL
        #[cfg(feature = "visual")]
        {
            if let Ok(mesh) = Self::from_stl(std::path::Path::new(filename), scale) {
                return mesh;
            }
        }
        // Fallback: placeholder cube at 5cm
        let _ = (filename, scale);
        let mut m = Self::cube();
        for v in &mut m.vertices {
            v[0] *= 0.05;
            v[1] *= 0.05;
            v[2] *= 0.05;
        }
        m
    }

    /// Load an STL file (binary or ASCII) and return a MeshData with smooth normals.
    #[cfg(feature = "visual")]
    pub fn from_stl(path: &std::path::Path, scale: [f64; 3]) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        let stl = stl_io::read_stl(&mut file)?;

        // STL has per-face vertices (no sharing). We deduplicate via position hash
        // for smooth normals, but for simplicity and correctness we'll use indexed
        // per-face vertices with face normals (flat shading first, smooth later).
        let mut vertices = Vec::with_capacity(stl.faces.len() * 3);
        let mut normals = Vec::with_capacity(stl.faces.len() * 3);
        let mut indices = Vec::with_capacity(stl.faces.len() * 3);

        for face in &stl.faces {
            let n = [face.normal[0], face.normal[1], face.normal[2]];
            let base = vertices.len() as u32;

            for vi in 0..3 {
                let v = &stl.vertices[face.vertices[vi]];
                vertices.push([
                    v[0] * scale[0] as f32,
                    v[1] * scale[1] as f32,
                    v[2] * scale[2] as f32,
                ]);
                normals.push(n);
            }

            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        Ok(Self {
            vertices,
            normals,
            indices,
        })
    }

    /// Generate mesh from URDF geometry shape.
    pub fn from_geometry(shape: &GeometryShape) -> Self {
        match shape {
            GeometryShape::Box { x, y, z } => {
                let mut m = Self::cube();
                for v in &mut m.vertices { v[0] *= *x as f32 * 2.0; v[1] *= *y as f32 * 2.0; v[2] *= *z as f32 * 2.0; }
                m
            }
            GeometryShape::Cylinder { radius, length } => {
                let mut m = Self::cylinder(16);
                for v in &mut m.vertices { v[0] *= *radius as f32 * 2.0; v[1] *= *length as f32; v[2] *= *radius as f32 * 2.0; }
                m
            }
            GeometryShape::Sphere { radius } => {
                let mut m = Self::sphere(16, 12);
                for v in &mut m.vertices { v[0] *= *radius as f32 * 2.0; v[1] *= *radius as f32 * 2.0; v[2] *= *radius as f32 * 2.0; }
                m
            }
            GeometryShape::Mesh { filename, scale } => {
                Self::from_mesh_file(filename, *scale)
            }
        }
    }
}

/// Mesh registry: stores loaded meshes by handle.
#[derive(Default)]
pub struct MeshRegistry {
    meshes: Vec<MeshData>,
}

impl MeshRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn load(&mut self, mesh: MeshData) -> MeshHandle {
        let h = MeshHandle(self.meshes.len());
        self.meshes.push(mesh);
        h
    }
    pub fn get(&self, handle: MeshHandle) -> Option<&MeshData> { self.meshes.get(handle.0) }
    pub fn count(&self) -> usize { self.meshes.len() }
}

// ═══════════════════════════════════════════════════════════════════════════
// Camera
// ═══════════════════════════════════════════════════════════════════════════

/// Camera projection type.
#[derive(Debug, Clone, Copy)]
pub enum Projection {
    Perspective { fov_y: f32, near: f32, far: f32 },
    Orthographic { scale: f32, near: f32, far: f32 },
}

/// Camera for 3D viewing.
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub projection: Projection,
}

impl Camera {
    pub fn perspective(pos: [f32; 3], target: [f32; 3], fov: f32) -> Self {
        Self {
            position: Point3::from(pos),
            target: Point3::from(target),
            up: Vector3::y(),
            projection: Projection::Perspective { fov_y: fov, near: 0.01, far: 100.0 },
        }
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        let dir = self.position - self.target;
        let dist = dir.norm();
        if dist < 1e-10 {
            return; // Camera at target — cannot orbit
        }
        let current_yaw = dir.z.atan2(dir.x);
        let current_pitch = (dir.y / dist).clamp(-1.0, 1.0).asin();
        let new_yaw = current_yaw + delta_yaw;
        let new_pitch = (current_pitch + delta_pitch).clamp(-1.5, 1.5);
        self.position = self.target + Vector3::new(
            dist * new_pitch.cos() * new_yaw.cos(),
            dist * new_pitch.sin(),
            dist * new_pitch.cos() * new_yaw.sin(),
        );
    }

    pub fn zoom(&mut self, factor: f32) {
        let dir = self.target - self.position;
        self.position += dir * factor;
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position, &self.target, &self.up)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Robot Model (URDF → Scene Graph)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a scene graph from a URDF robot model.
pub fn robot_scene_graph(robot: &Robot, meshes: &mut MeshRegistry) -> SceneNode {
    let mut root = SceneNode::new(&robot.name);

    for (_i, link) in robot.links.iter().enumerate() {
        let mut link_node = SceneNode::new(&link.name);

        // Visual geometry
        for (gi, geom) in link.collision_geometry.iter().enumerate() {
            let mesh_data = MeshData::from_geometry(&geom.shape);
            let handle = meshes.load(mesh_data);

            let child = SceneNode::new(&format!("{}_{}", link.name, gi))
                .with_mesh(handle)
                .with_transform(geom.origin.0)
                .with_material(Material::solid(0.6, 0.6, 0.7));
            link_node.add_child(child);
        }

        root.add_child(link_node);
    }

    root
}

/// Update scene graph transforms from FK poses.
pub fn update_robot_transforms(root: &mut SceneNode, link_poses: &[Pose]) {
    for (i, child) in root.children.iter_mut().enumerate() {
        if i < link_poses.len() {
            child.transform = link_poses[i].0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trajectory Player
// ═══════════════════════════════════════════════════════════════════════════

/// Trajectory playback controller.
#[derive(Debug, Clone)]
pub struct TrajectoryPlayer {
    trajectory: Trajectory,
    t: f64,
    speed: f64,
    playing: bool,
    looping: bool,
}

impl TrajectoryPlayer {
    pub fn new(trajectory: Trajectory) -> Self {
        Self { trajectory, t: 0.0, speed: 1.0, playing: false, looping: false }
    }

    pub fn play(&mut self) { self.playing = true; }
    pub fn pause(&mut self) { self.playing = false; }
    pub fn reset(&mut self) { self.t = 0.0; }
    pub fn set_speed(&mut self, speed: f64) { self.speed = speed; }
    pub fn set_looping(&mut self, looping: bool) { self.looping = looping; }
    pub fn is_playing(&self) -> bool { self.playing }
    pub fn progress(&self) -> f64 { self.t }
    pub fn seek(&mut self, t: f64) { self.t = t.clamp(0.0, 1.0); }

    /// Access the underlying trajectory (e.g., for ghost robot sampling).
    pub fn trajectory(&self) -> &Trajectory { &self.trajectory }

    /// Advance playback by dt seconds. Returns current joint values.
    pub fn tick(&mut self, dt: f64) -> JointValues {
        if self.playing {
            self.t += dt * self.speed;
            if self.t > 1.0 {
                if self.looping { self.t %= 1.0; }
                else { self.t = 1.0; self.playing = false; }
            }
        }
        self.trajectory.sample(self.t)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display Modes
// ═══════════════════════════════════════════════════════════════════════════

/// Display settings for the viewer.
#[derive(Debug, Clone)]
pub struct ViewerSettings {
    pub show_robot: bool,
    pub show_collision_geometry: bool,
    pub show_collision_wireframe: bool,
    pub show_trajectory_trail: bool,
    pub show_voxels: bool,
    pub show_point_cloud: bool,
    pub show_grid: bool,
    pub show_axes: bool,
    pub background_color: [f32; 4],
    pub trail_color: [f32; 4],
    pub grid_size: f32,
    pub grid_divisions: usize,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        Self {
            show_robot: true,
            show_collision_geometry: true,
            show_collision_wireframe: false,
            show_trajectory_trail: true,
            show_voxels: false,
            show_point_cloud: false,
            show_grid: true,
            show_axes: true,
            background_color: [0.15, 0.15, 0.18, 1.0],
            trail_color: [0.2, 0.8, 0.2, 0.5],
            grid_size: 2.0,
            grid_divisions: 20,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Render Commands (renderer-agnostic)
// ═══════════════════════════════════════════════════════════════════════════

/// A render command for the GPU renderer.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    DrawMesh { handle: MeshHandle, transform: Matrix4<f32>, material: Material },
    DrawLine { start: [f32; 3], end: [f32; 3], color: [f32; 4] },
    DrawPoint { position: [f32; 3], size: f32, color: [f32; 4] },
    DrawGrid { size: f32, divisions: usize },
    DrawAxes { length: f32 },
}

/// Collect render commands from the scene graph.
pub fn collect_render_commands(
    root: &SceneNode,
    parent_transform: &Matrix4<f32>,
    commands: &mut Vec<RenderCommand>,
) {
    if !root.visible { return; }

    let local: Matrix4<f32> = root.transform.to_homogeneous().cast::<f32>();
    let world = parent_transform * local;

    if let Some(handle) = root.mesh {
        commands.push(RenderCommand::DrawMesh {
            handle,
            transform: world,
            material: root.material.clone(),
        });
    }

    for child in &root.children {
        collect_render_commands(child, &world, commands);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_data_primitives() {
        let cube = MeshData::cube();
        assert_eq!(cube.num_vertices(), 8);
        assert!(cube.num_triangles() > 0);

        let sphere = MeshData::sphere(8, 6);
        assert!(sphere.num_vertices() > 0);
        assert!(sphere.num_triangles() > 0);

        let cyl = MeshData::cylinder(8);
        assert!(cyl.num_vertices() > 0);
    }

    #[test]
    fn mesh_registry() {
        let mut reg = MeshRegistry::new();
        let h1 = reg.load(MeshData::cube());
        let h2 = reg.load(MeshData::sphere(8, 6));
        assert_eq!(reg.count(), 2);
        assert!(reg.get(h1).is_some());
        assert!(reg.get(h2).is_some());
    }

    #[test]
    fn camera_orbit() {
        let mut cam = Camera::perspective([3.0, 2.0, 3.0], [0.0, 0.0, 0.0], 45.0);
        let orig_pos = cam.position;
        cam.orbit(0.1, 0.05);
        assert_ne!(cam.position, orig_pos, "Orbit should move camera");
    }

    #[test]
    fn camera_zoom() {
        let mut cam = Camera::perspective([3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 45.0);
        cam.zoom(0.1);
        assert!(cam.position.x < 3.0, "Zoom should move closer");
    }

    #[test]
    fn scene_node_builder() {
        let node = SceneNode::new("test")
            .with_mesh(MeshHandle(0))
            .with_material(Material::solid(1.0, 0.0, 0.0));
        assert_eq!(node.name, "test");
        assert!(node.mesh.is_some());
    }

    #[test]
    fn robot_scene_graph_from_urdf() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
</robot>"#;

        let robot = Robot::from_urdf_string(urdf).unwrap();
        let mut meshes = MeshRegistry::new();
        let scene = robot_scene_graph(&robot, &mut meshes);

        assert_eq!(scene.name, "test");
        assert_eq!(scene.children.len(), 2); // base + link1
        assert!(meshes.count() >= 2); // at least one mesh per link with geometry
    }

    #[test]
    fn trajectory_player_playback() {
        let mut traj = Trajectory::with_dof(2);
        traj.push_waypoint(&[0.0, 0.0]);
        traj.push_waypoint(&[1.0, 1.0]);
        traj.push_waypoint(&[2.0, 2.0]);

        let mut player = TrajectoryPlayer::new(traj);
        player.play();

        let wp = player.tick(0.5);
        assert!(player.is_playing());
        assert!(player.progress() > 0.0);
    }

    #[test]
    fn trajectory_player_loop() {
        let mut traj = Trajectory::with_dof(1);
        traj.push_waypoint(&[0.0]);
        traj.push_waypoint(&[1.0]);

        let mut player = TrajectoryPlayer::new(traj);
        player.set_looping(true);
        player.play();

        // Advance past end
        for _ in 0..20 {
            player.tick(0.1);
        }
        assert!(player.is_playing(), "Looping player should still be playing");
        assert!(player.progress() < 1.0, "Should have looped: {}", player.progress());
    }

    #[test]
    fn render_commands_from_scene() {
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child").with_mesh(MeshHandle(0)));

        let mut commands = Vec::new();
        collect_render_commands(&root, &Matrix4::identity(), &mut commands);
        assert_eq!(commands.len(), 1); // one DrawMesh for the child
    }

    #[test]
    fn viewer_settings_default() {
        let settings = ViewerSettings::default();
        assert!(settings.show_robot);
        assert!(settings.show_grid);
        assert_eq!(settings.grid_divisions, 20);
    }

    #[test]
    fn material_constructors() {
        let solid = Material::solid(1.0, 0.0, 0.0);
        assert_eq!(solid.color[3], 1.0);
        assert!(!solid.wireframe);

        let wire = Material::wireframe(0.0, 1.0, 0.0);
        assert!(wire.wireframe);

        let trans = Material::transparent(0.0, 0.0, 1.0, 0.5);
        assert_eq!(trans.color[3], 0.5);
    }

    #[test]
    fn mesh_from_geometry() {
        let box_mesh = MeshData::from_geometry(&GeometryShape::Box { x: 0.1, y: 0.2, z: 0.3 });
        assert!(!box_mesh.is_empty());

        let sphere_mesh = MeshData::from_geometry(&GeometryShape::Sphere { radius: 0.05 });
        assert!(!sphere_mesh.is_empty());

        let cyl_mesh = MeshData::from_geometry(&GeometryShape::Cylinder { radius: 0.04, length: 0.3 });
        assert!(!cyl_mesh.is_empty());
    }
}
