//! Web export: generate self-contained HTML+three.js visualization files.
//!
//! No GPU required on the generating machine. The HTML file contains all
//! geometry data inline and renders in any modern browser using three.js.
//!
//! # Usage
//!
//! ```ignore
//! let mut exporter = WebExporter::new();
//! exporter.add_robot(&robot, &joint_poses);
//! exporter.add_trajectory(&trajectory);
//! exporter.add_point_cloud(&points, [0.2, 0.8, 0.2]);
//! exporter.add_collision_box("table", [0.0, 0.0, -0.05], [0.5, 0.5, 0.05]);
//! std::fs::write("scene.html", exporter.to_html())?;
//! // Open scene.html in any browser — interactive 3D with orbit controls
//! ```

use kinetic_core::{Pose, Trajectory};
use kinetic_robot::{GeometryShape, Robot};

/// A mesh to embed in the HTML export.
#[derive(Debug, Clone)]
struct ExportMesh {
    name: String,
    vertices: Vec<[f64; 3]>,
    indices: Vec<[usize; 3]>,
    color: [f32; 3],
    opacity: f32,
    wireframe: bool,
    transform: [f64; 16], // column-major 4x4
}

/// A point cloud to embed.
#[derive(Debug, Clone)]
struct ExportPointCloud {
    name: String,
    points: Vec<[f64; 3]>,
    color: [f32; 3],
    point_size: f32,
}

/// A line strip (trajectory trail, axes, etc.).
#[derive(Debug, Clone)]
struct ExportLineStrip {
    name: String,
    points: Vec<[f64; 3]>,
    color: [f32; 3],
}

/// Web exporter: builds an HTML document with embedded three.js scene.
pub struct WebExporter {
    meshes: Vec<ExportMesh>,
    point_clouds: Vec<ExportPointCloud>,
    line_strips: Vec<ExportLineStrip>,
    title: String,
    background_color: [f32; 3],
    show_grid: bool,
    show_axes: bool,
    camera_position: [f64; 3],
    camera_target: [f64; 3],
}

impl WebExporter {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            point_clouds: Vec::new(),
            line_strips: Vec::new(),
            title: "Kinetic Scene".into(),
            background_color: [0.15, 0.15, 0.18],
            show_grid: true,
            show_axes: true,
            camera_position: [2.0, 1.5, 2.0],
            camera_target: [0.0, 0.0, 0.3],
        }
    }

    /// Set the page title.
    pub fn set_title(&mut self, title: &str) { self.title = title.to_string(); }

    /// Set camera position and target.
    pub fn set_camera(&mut self, position: [f64; 3], target: [f64; 3]) {
        self.camera_position = position;
        self.camera_target = target;
    }

    /// Set background color.
    pub fn set_background(&mut self, r: f32, g: f32, b: f32) {
        self.background_color = [r, g, b];
    }

    /// Toggle grid floor.
    pub fn set_grid(&mut self, show: bool) { self.show_grid = show; }

    /// Toggle axes helper.
    pub fn set_axes(&mut self, show: bool) { self.show_axes = show; }

    /// Add robot links from URDF with FK poses.
    pub fn add_robot(&mut self, robot: &Robot, link_poses: &[Pose]) {
        for (i, link) in robot.links.iter().enumerate() {
            let link_tf = if i < link_poses.len() {
                link_poses[i].0.to_homogeneous()
            } else {
                nalgebra::Matrix4::identity()
            };

            for (gi, geom) in link.collision_geometry.iter().enumerate() {
                let geom_tf = geom.origin.0.to_homogeneous();
                let world_tf = link_tf * geom_tf;

                let (vertices, indices) = geometry_to_mesh(&geom.shape);
                let mut transform = [0.0f64; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        transform[c * 4 + r] = world_tf[(r, c)];
                    }
                }

                self.meshes.push(ExportMesh {
                    name: format!("{}_{}", link.name, gi),
                    vertices,
                    indices,
                    color: link_color(i),
                    opacity: 0.85,
                    wireframe: false,
                    transform,
                });
            }
        }
    }

    /// Add a trajectory as a line strip through end-effector positions.
    ///
    /// `ee_positions`: list of [x, y, z] end-effector positions along the trajectory.
    pub fn add_trajectory_trail(&mut self, ee_positions: &[[f64; 3]]) {
        self.line_strips.push(ExportLineStrip {
            name: "trajectory".into(),
            points: ee_positions.to_vec(),
            color: [0.2, 0.9, 0.2],
        });
    }

    /// Add a trajectory by sampling joint configurations and computing positions.
    ///
    /// Uses the last joint's origin as a rough end-effector proxy.
    pub fn add_trajectory(&mut self, trajectory: &Trajectory) {
        let n = trajectory.len();
        if n == 0 { return; }

        // Sample waypoint positions (use first 3 joints as XYZ proxy if DOF >= 3)
        let mut points = Vec::new();
        for i in 0..n {
            let wp = trajectory.waypoint(i);
            let p = if wp.positions.len() >= 3 {
                [wp.positions[0], wp.positions[1], wp.positions[2]]
            } else if wp.positions.len() >= 1 {
                [wp.positions[0], 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            };
            points.push(p);
        }

        self.line_strips.push(ExportLineStrip {
            name: "trajectory_joints".into(),
            points,
            color: [0.2, 0.8, 0.2],
        });
    }

    /// Add a point cloud.
    pub fn add_point_cloud(&mut self, points: &[[f64; 3]], color: [f32; 3]) {
        self.point_clouds.push(ExportPointCloud {
            name: format!("pointcloud_{}", self.point_clouds.len()),
            points: points.to_vec(),
            color,
            point_size: 3.0,
        });
    }

    /// Add a box collision object.
    pub fn add_collision_box(
        &mut self,
        name: &str,
        center: [f64; 3],
        half_extents: [f64; 3],
        color: [f32; 3],
    ) {
        let (vertices, indices) = make_box_mesh(half_extents);
        let mut transform = identity_transform();
        transform[12] = center[0];
        transform[13] = center[1];
        transform[14] = center[2];

        self.meshes.push(ExportMesh {
            name: name.to_string(),
            vertices, indices,
            color,
            opacity: 0.4,
            wireframe: false,
            transform,
        });
    }

    /// Add a sphere collision object.
    pub fn add_collision_sphere(
        &mut self,
        name: &str,
        center: [f64; 3],
        radius: f64,
        color: [f32; 3],
    ) {
        let (vertices, indices) = make_sphere_mesh(radius, 12, 8);
        let mut transform = identity_transform();
        transform[12] = center[0];
        transform[13] = center[1];
        transform[14] = center[2];

        self.meshes.push(ExportMesh {
            name: name.to_string(),
            vertices, indices,
            color,
            opacity: 0.4,
            wireframe: false,
            transform,
        });
    }

    /// Add a wireframe collision box.
    pub fn add_wireframe_box(
        &mut self,
        name: &str,
        center: [f64; 3],
        half_extents: [f64; 3],
        color: [f32; 3],
    ) {
        let (vertices, indices) = make_box_mesh(half_extents);
        let mut transform = identity_transform();
        transform[12] = center[0];
        transform[13] = center[1];
        transform[14] = center[2];

        self.meshes.push(ExportMesh {
            name: name.to_string(),
            vertices, indices,
            color,
            opacity: 1.0,
            wireframe: true,
            transform,
        });
    }

    /// Add occupied voxels from an octree.
    pub fn add_voxels(&mut self, voxel_centers: &[[f64; 3]], voxel_half_size: f64, color: [f32; 3]) {
        // Render each voxel as a point (efficient for large voxel counts)
        self.point_clouds.push(ExportPointCloud {
            name: "voxels".into(),
            points: voxel_centers.to_vec(),
            color,
            point_size: (voxel_half_size * 500.0).max(2.0).min(10.0) as f32,
        });
    }

    /// Generate the complete self-contained HTML document.
    pub fn to_html(&self) -> String {
        let mut html = String::with_capacity(64 * 1024);

        html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #{bg}; }}
  canvas {{ display: block; }}
  #info {{ position: absolute; top: 10px; left: 10px; color: #ccc; font: 14px monospace;
           background: rgba(0,0,0,0.6); padding: 8px 12px; border-radius: 4px; }}
</style>
</head>
<body>
<div id="info">{title} — orbit: drag | zoom: scroll | pan: right-drag</div>
<script type="importmap">
{{ "imports": {{ "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
                 "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color({bg_r}, {bg_g}, {bg_b});

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set({cam_x}, {cam_y}, {cam_z});

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set({tgt_x}, {tgt_y}, {tgt_z});
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0x404040, 2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(3, 5, 4);
scene.add(dirLight);
"#,
            title = self.title,
            bg = format!("{:02x}{:02x}{:02x}",
                (self.background_color[0] * 255.0) as u8,
                (self.background_color[1] * 255.0) as u8,
                (self.background_color[2] * 255.0) as u8),
            bg_r = self.background_color[0],
            bg_g = self.background_color[1],
            bg_b = self.background_color[2],
            cam_x = self.camera_position[0],
            cam_y = self.camera_position[1],
            cam_z = self.camera_position[2],
            tgt_x = self.camera_target[0],
            tgt_y = self.camera_target[1],
            tgt_z = self.camera_target[2],
        ));

        // Grid and axes
        if self.show_grid {
            html.push_str("scene.add(new THREE.GridHelper(4, 40, 0x444444, 0x333333));\n");
        }
        if self.show_axes {
            html.push_str("scene.add(new THREE.AxesHelper(0.5));\n");
        }

        // Meshes
        for mesh in &self.meshes {
            html.push_str(&format!(
                "{{ const verts = new Float32Array({vertices});\n\
                 const idx = new Uint32Array({indices});\n\
                 const geom = new THREE.BufferGeometry();\n\
                 geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));\n\
                 geom.setIndex(new THREE.BufferAttribute(idx, 1));\n\
                 geom.computeVertexNormals();\n\
                 const mat = new THREE.Mesh{mat_type}({{ color: new THREE.Color({r},{g},{b}), \
                   opacity: {opacity}, transparent: {transparent}, wireframe: {wireframe} }});\n\
                 const mesh = new THREE.Mesh(geom, mat);\n\
                 mesh.applyMatrix4(new THREE.Matrix4().fromArray({transform}));\n\
                 mesh.name = '{name}';\n\
                 scene.add(mesh); }}\n",
                vertices = format_f64_array_flat(&mesh.vertices),
                indices = format_index_array_flat(&mesh.indices),
                mat_type = if mesh.wireframe { "BasicMaterial" } else { "PhongMaterial" },
                r = mesh.color[0], g = mesh.color[1], b = mesh.color[2],
                opacity = mesh.opacity,
                transparent = if mesh.opacity < 1.0 { "true" } else { "false" },
                wireframe = if mesh.wireframe { "true" } else { "false" },
                transform = format_f64_array_1d(&mesh.transform),
                name = mesh.name,
            ));
        }

        // Point clouds
        for pc in &self.point_clouds {
            html.push_str(&format!(
                "{{ const pts = new Float32Array({points});\n\
                 const geom = new THREE.BufferGeometry();\n\
                 geom.setAttribute('position', new THREE.BufferAttribute(pts, 3));\n\
                 const mat = new THREE.PointsMaterial({{ color: new THREE.Color({r},{g},{b}), size: {size} }});\n\
                 const cloud = new THREE.Points(geom, mat);\n\
                 cloud.name = '{name}';\n\
                 scene.add(cloud); }}\n",
                points = format_f64_array_flat(&pc.points),
                r = pc.color[0], g = pc.color[1], b = pc.color[2],
                size = pc.point_size / 100.0,
                name = pc.name,
            ));
        }

        // Line strips
        for ls in &self.line_strips {
            html.push_str(&format!(
                "{{ const pts = new Float32Array({points});\n\
                 const geom = new THREE.BufferGeometry();\n\
                 geom.setAttribute('position', new THREE.BufferAttribute(pts, 3));\n\
                 const mat = new THREE.LineBasicMaterial({{ color: new THREE.Color({r},{g},{b}), linewidth: 2 }});\n\
                 const line = new THREE.Line(geom, mat);\n\
                 line.name = '{name}';\n\
                 scene.add(line); }}\n",
                points = format_f64_array_flat(&ls.points),
                r = ls.color[0], g = ls.color[1], b = ls.color[2],
                name = ls.name,
            ));
        }

        // Render loop + resize
        html.push_str(r#"
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>"#);

        html
    }

    /// Write HTML to a file.
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_html())
    }

    /// Number of meshes in the export.
    pub fn num_meshes(&self) -> usize { self.meshes.len() }
    /// Number of point clouds.
    pub fn num_point_clouds(&self) -> usize { self.point_clouds.len() }
    /// Number of line strips.
    pub fn num_line_strips(&self) -> usize { self.line_strips.len() }
}

// ─── Geometry helpers ────────────────────────────────────────────────────

fn geometry_to_mesh(shape: &GeometryShape) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    match shape {
        GeometryShape::Box { x, y, z } => make_box_mesh([*x, *y, *z]),
        GeometryShape::Sphere { radius } => make_sphere_mesh(*radius, 12, 8),
        GeometryShape::Cylinder { radius, length } => make_cylinder_mesh(*radius, *length, 12),
        GeometryShape::Mesh { filename, scale } => {
            load_mesh_file(filename, *scale)
                .unwrap_or_else(|| make_box_mesh([0.05, 0.05, 0.05]))
        }
    }
}

/// Try to load an STL mesh file and convert to export-compatible vertices/indices.
///
/// Returns `None` if the file cannot be loaded (missing file, unsupported format,
/// or `visual` feature disabled), falling back to a placeholder box.
fn load_mesh_file(filename: &str, scale: [f64; 3]) -> Option<(Vec<[f64; 3]>, Vec<[usize; 3]>)> {
    #[cfg(feature = "visual")]
    {
        let path = std::path::Path::new(filename);
        let mut file = std::fs::File::open(path).ok()?;
        let stl = stl_io::read_stl(&mut file).ok()?;

        let mut vertices = Vec::with_capacity(stl.faces.len() * 3);
        let mut indices = Vec::with_capacity(stl.faces.len());

        for face in &stl.faces {
            let base = vertices.len();
            for vi in 0..3 {
                let v = &stl.vertices[face.vertices[vi]];
                vertices.push([
                    v[0] as f64 * scale[0],
                    v[1] as f64 * scale[1],
                    v[2] as f64 * scale[2],
                ]);
            }
            indices.push([base, base + 1, base + 2]);
        }

        Some((vertices, indices))
    }
    #[cfg(not(feature = "visual"))]
    {
        let _ = (filename, scale);
        None
    }
}

fn make_box_mesh(half: [f64; 3]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (hx, hy, hz) = (half[0], half[1], half[2]);
    let v = vec![
        [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
        [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz],
    ];
    let i = vec![
        [0,2,1],[0,3,2], [4,5,6],[4,6,7], [0,1,5],[0,5,4],
        [2,3,7],[2,7,6], [1,2,6],[1,6,5], [0,4,7],[0,7,3],
    ];
    (v, i)
}

fn make_sphere_mesh(radius: f64, segments: usize, rings: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut verts = Vec::new();
    let mut tris = Vec::new();

    for r in 0..=rings {
        let phi = std::f64::consts::PI * r as f64 / rings as f64;
        for s in 0..=segments {
            let theta = 2.0 * std::f64::consts::PI * s as f64 / segments as f64;
            verts.push([
                radius * phi.sin() * theta.cos(),
                radius * phi.cos(),
                radius * phi.sin() * theta.sin(),
            ]);
        }
    }

    for r in 0..rings {
        for s in 0..segments {
            let a = r * (segments + 1) + s;
            let b = a + segments + 1;
            tris.push([a, b, a + 1]);
            tris.push([a + 1, b, b + 1]);
        }
    }

    (verts, tris)
}

fn make_cylinder_mesh(radius: f64, length: f64, segments: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let half = length / 2.0;
    let mut verts = Vec::new();
    let mut tris = Vec::new();

    for i in 0..=segments {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
        let x = radius * theta.cos();
        let z = radius * theta.sin();
        verts.push([x, -half, z]);
        verts.push([x, half, z]);
    }

    for i in 0..segments {
        let a = i * 2;
        tris.push([a, a + 2, a + 1]);
        tris.push([a + 1, a + 2, a + 3]);
    }

    (verts, tris)
}

fn link_color(index: usize) -> [f32; 3] {
    let hue = (index as f32 * 0.618034) % 1.0;
    let (r, g, b) = hsv_to_rgb(hue, 0.5, 0.8);
    [r, g, b]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32) % 6 {
        0 => (v, t, p), 1 => (q, v, p), 2 => (p, v, t),
        3 => (p, q, v), 4 => (t, p, v), _ => (v, p, q),
    }
}

fn identity_transform() -> [f64; 16] {
    [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]
}

fn format_f64_array_flat(arr: &[[f64; 3]]) -> String {
    let mut s = String::from("[");
    for (i, v) in arr.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{:.6},{:.6},{:.6}", v[0], v[1], v[2]));
    }
    s.push(']');
    s
}

fn format_index_array_flat(arr: &[[usize; 3]]) -> String {
    let mut s = String::from("[");
    for (i, t) in arr.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{},{},{}", t[0], t[1], t[2]));
    }
    s.push(']');
    s
}

fn format_f64_array_1d(arr: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, v) in arr.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{:.6}", v));
    }
    s.push(']');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_exporter_produces_valid_html() {
        let exporter = WebExporter::new();
        let html = exporter.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("three"));
        assert!(html.contains("OrbitControls"));
        assert!(html.contains("animate()"));
    }

    #[test]
    fn exporter_with_box() {
        let mut exporter = WebExporter::new();
        exporter.add_collision_box("table", [0.0, 0.0, -0.05], [0.5, 0.5, 0.05], [0.6, 0.4, 0.2]);

        assert_eq!(exporter.num_meshes(), 1);
        let html = exporter.to_html();
        assert!(html.contains("table"));
        assert!(html.contains("BufferGeometry"));
    }

    #[test]
    fn exporter_with_point_cloud() {
        let mut exporter = WebExporter::new();
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        exporter.add_point_cloud(&points, [0.2, 0.8, 0.2]);

        assert_eq!(exporter.num_point_clouds(), 1);
        let html = exporter.to_html();
        assert!(html.contains("PointsMaterial"));
    }

    #[test]
    fn exporter_with_trajectory() {
        let mut exporter = WebExporter::new();
        let trail = vec![[0.0, 0.0, 0.3], [0.5, 0.0, 0.3], [0.5, 0.5, 0.3]];
        exporter.add_trajectory_trail(&trail);

        assert_eq!(exporter.num_line_strips(), 1);
        let html = exporter.to_html();
        assert!(html.contains("LineBasicMaterial"));
    }

    #[test]
    fn exporter_with_robot() {
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
        let poses = vec![Pose::identity(), Pose::from_xyz(0.0, 0.0, 0.1)];

        let mut exporter = WebExporter::new();
        exporter.add_robot(&robot, &poses);

        assert_eq!(exporter.num_meshes(), 2);
        let html = exporter.to_html();
        assert!(html.contains("base_0"));
        assert!(html.contains("link1_0"));
    }

    #[test]
    fn exporter_with_sphere_and_wireframe() {
        let mut exporter = WebExporter::new();
        exporter.add_collision_sphere("ball", [1.0, 0.0, 0.5], 0.1, [1.0, 0.0, 0.0]);
        exporter.add_wireframe_box("zone", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]);

        assert_eq!(exporter.num_meshes(), 2);
        let html = exporter.to_html();
        assert!(html.contains("wireframe: true"));
    }

    #[test]
    fn exporter_with_voxels() {
        let mut exporter = WebExporter::new();
        let voxels = vec![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]];
        exporter.add_voxels(&voxels, 0.05, [1.0, 0.3, 0.1]);

        assert_eq!(exporter.num_point_clouds(), 1);
    }

    #[test]
    fn exporter_camera_config() {
        let mut exporter = WebExporter::new();
        exporter.set_camera([5.0, 3.0, 5.0], [0.0, 0.0, 1.0]);
        exporter.set_title("My Robot");
        exporter.set_background(0.1, 0.1, 0.12);

        let html = exporter.to_html();
        assert!(html.contains("My Robot"));
        assert!(html.contains("5") && html.contains("My Robot"));
    }

    #[test]
    fn full_scene_export() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="tip">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
</robot>"#;

        let robot = Robot::from_urdf_string(urdf).unwrap();
        let poses = vec![
            Pose::identity(),
            Pose::from_xyz(0.0, 0.0, 0.05),
            Pose::from_xyz(0.0, 0.0, 0.35),
        ];

        let mut exporter = WebExporter::new();
        exporter.set_title("Full Scene Test");
        exporter.add_robot(&robot, &poses);
        exporter.add_collision_box("table", [0.0, 0.5, -0.01], [0.3, 0.3, 0.01], [0.5, 0.3, 0.1]);
        exporter.add_point_cloud(
            &[[0.1, 0.5, 0.05], [0.15, 0.5, 0.08], [0.12, 0.52, 0.06]],
            [0.2, 0.8, 0.2],
        );
        exporter.add_trajectory_trail(&[
            [0.0, 0.0, 0.35], [0.1, 0.1, 0.35], [0.2, 0.3, 0.3], [0.1, 0.5, 0.1],
        ]);

        let html = exporter.to_html();
        assert!(html.len() > 1000, "Full scene HTML should be substantial: {}", html.len());
        assert_eq!(exporter.num_meshes(), 4); // 3 robot links + 1 table
        assert_eq!(exporter.num_point_clouds(), 1);
        assert_eq!(exporter.num_line_strips(), 1);

        // Verify it's valid HTML
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.ends_with("</html>"));
    }
}
