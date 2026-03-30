//! Planning scene and world model for KINETIC.
//!
//! Manages collision objects, attached objects, the Allowed Collision
//! Matrix (ACM), and scene queries (collision check, min distance, contacts).
//!
//! # Perception Pipeline
//!
//! Point clouds and depth images can be ingested as collision obstacles:
//!
//! ```ignore
//! use kinetic_scene::{Scene, PointCloudConfig};
//!
//! scene.add_pointcloud("camera_0", &points, PointCloudConfig::default());
//! scene.update_from_depth(&depth, 640, 480, &intrinsics, &camera_pose);
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kinetic_scene::{Scene, Shape};
//!
//! let mut scene = Scene::new(&robot);
//! scene.add("table", Shape::Cuboid(0.5, 0.3, 0.01), table_pose);
//! scene.attach("bolt", Shape::Cylinder(0.005, 0.03), grasp_tf, "hand_link");
//! let colliding = scene.check_collision(&joint_values)?;
//! ```

pub mod depth;
pub mod octree;
pub mod pointcloud;
pub mod processing;

pub use depth::{CameraIntrinsics, DepthConfig};
pub use octree::{Octree, OctreeConfig};
pub use pointcloud::{OutlierConfig, PointCloudConfig, PointCloudSource, RadiusOutlierConfig};

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::{Isometry3, Vector3};

use kinetic_collision::{AllowedCollisionMatrix, ContactPoint, RobotSphereModel, SpheresSoA};
use kinetic_core::{KineticError, Pose};
use kinetic_kinematics::{forward_kinematics_all, KinematicChain};
use kinetic_robot::Robot;

/// Collision shape types.
#[derive(Debug, Clone)]
pub enum Shape {
    /// Cuboid with half-extents (x, y, z).
    Cuboid(f64, f64, f64),
    /// Cylinder with radius and half-height.
    Cylinder(f64, f64),
    /// Sphere with radius.
    Sphere(f64),
    /// Infinite half-space defined by normal and offset.
    /// Points `p` where `normal . p <= offset` are inside.
    HalfSpace(Vector3<f64>, f64),
}

impl Shape {
    /// Create a cuboid (box) shape from half-extents.
    ///
    /// `half_x`, `half_y`, `half_z` are the half-lengths along each axis.
    /// A 1m x 0.5m x 0.1m box uses `Shape::cuboid(0.5, 0.25, 0.05)`.
    pub fn cuboid(half_x: f64, half_y: f64, half_z: f64) -> Self {
        Shape::Cuboid(half_x, half_y, half_z)
    }

    /// Create a cylinder shape.
    ///
    /// `radius` is the cylinder radius, `half_height` is half the total height.
    pub fn cylinder(radius: f64, half_height: f64) -> Self {
        Shape::Cylinder(radius, half_height)
    }

    /// Create a sphere shape.
    pub fn sphere(radius: f64) -> Self {
        Shape::Sphere(radius)
    }

    /// Create an infinite half-space defined by a normal vector and offset.
    ///
    /// Points `p` where `normal . p <= offset` are inside.
    pub fn half_space(normal: Vector3<f64>, offset: f64) -> Self {
        Shape::HalfSpace(normal, offset)
    }

    /// Generate collision spheres approximating this shape.
    fn to_spheres(&self, pose: &Isometry3<f64>, resolution: f64) -> Vec<(f64, f64, f64, f64)> {
        match self {
            Shape::Sphere(r) => {
                let p = pose.translation;
                vec![(p.x, p.y, p.z, *r)]
            }
            Shape::Cuboid(hx, hy, hz) => cuboid_to_spheres(*hx, *hy, *hz, pose, resolution),
            Shape::Cylinder(radius, half_height) => {
                cylinder_to_spheres(*radius, *half_height, pose, resolution)
            }
            Shape::HalfSpace(normal, offset) => {
                half_space_to_spheres(normal, *offset, pose, resolution)
            }
        }
    }
}

/// A scene object (obstacle) in the world.
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// Object name.
    pub name: String,
    /// Collision shape.
    pub shape: Shape,
    /// Pose in world frame.
    pub pose: Isometry3<f64>,
}

/// An object attached to a robot link (grasped or mounted).
#[derive(Debug, Clone)]
pub struct AttachedObject {
    /// Object name.
    pub name: String,
    /// Collision shape.
    pub shape: Shape,
    /// Transform relative to parent link frame.
    pub grasp_transform: Isometry3<f64>,
    /// Name of the parent link.
    pub parent_link: String,
}

/// The planning scene — world model tracking all collision geometry.
pub struct Scene {
    /// Reference to the robot model.
    robot: Arc<Robot>,
    /// Kinematic chain for FK.
    chain: KinematicChain,
    /// Sphere model for robot self-collision.
    sphere_model: RobotSphereModel,
    /// Scene objects (obstacles).
    objects: HashMap<String, SceneObject>,
    /// Objects attached to robot links.
    attached: HashMap<String, AttachedObject>,
    /// Point cloud sources (sensor data converted to collision spheres).
    pointclouds: HashMap<String, PointCloudSource>,
    /// Octree-based volumetric occupancy maps.
    octrees: HashMap<String, Octree>,
    /// Allowed collision matrix.
    acm: AllowedCollisionMatrix,
    /// Sphere approximation resolution for objects.
    sphere_resolution: f64,
}

impl Scene {
    /// Create a new empty scene for the given robot.
    ///
    /// Auto-detects the kinematic chain from planning groups or URDF tree.
    pub fn new(robot: &Robot) -> kinetic_core::Result<Self> {
        let chain = auto_detect_chain(robot)?;
        let sphere_model = RobotSphereModel::from_robot_default(robot);
        let acm = AllowedCollisionMatrix::from_robot(robot);

        Ok(Scene {
            robot: Arc::new(robot.clone()),
            chain,
            sphere_model,
            objects: HashMap::new(),
            attached: HashMap::new(),
            pointclouds: HashMap::new(),
            octrees: HashMap::new(),
            acm,
            sphere_resolution: 0.02,
        })
    }

    /// Create a scene with a specific kinematic chain.
    pub fn with_chain(robot: &Robot, chain: KinematicChain) -> Self {
        let sphere_model = RobotSphereModel::from_robot_default(robot);
        let acm = AllowedCollisionMatrix::from_robot(robot);

        Scene {
            robot: Arc::new(robot.clone()),
            chain,
            sphere_model,
            objects: HashMap::new(),
            attached: HashMap::new(),
            pointclouds: HashMap::new(),
            octrees: HashMap::new(),
            acm,
            sphere_resolution: 0.02,
        }
    }

    /// Set the sphere approximation resolution for scene objects.
    pub fn set_sphere_resolution(&mut self, resolution: f64) {
        self.sphere_resolution = resolution;
    }

    /// Number of objects in the scene (excluding attached).
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Number of attached objects.
    pub fn num_attached(&self) -> usize {
        self.attached.len()
    }

    /// Get a reference to a scene object by name.
    pub fn get_object(&self, name: &str) -> Option<&SceneObject> {
        self.objects.get(name)
    }

    /// Get a reference to an attached object by name.
    pub fn get_attached(&self, name: &str) -> Option<&AttachedObject> {
        self.attached.get(name)
    }

    /// Get a reference to the ACM.
    pub fn acm(&self) -> &AllowedCollisionMatrix {
        &self.acm
    }

    /// Get a reference to the robot.
    pub fn robot(&self) -> &Robot {
        &self.robot
    }

    /// DOF of the kinematic chain.
    pub fn dof(&self) -> usize {
        self.chain.dof
    }

    // === Object Management ===

    /// Add an object to the scene.
    pub fn add(&mut self, name: &str, shape: Shape, pose: Isometry3<f64>) {
        self.objects.insert(
            name.to_string(),
            SceneObject {
                name: name.to_string(),
                shape,
                pose,
            },
        );
    }

    /// Add a box obstacle at a position (no rotation).
    ///
    /// `half_extents`: `[half_x, half_y, half_z]` — half-lengths along each axis.
    /// `position`: `[x, y, z]` in world frame.
    ///
    /// For rotated obstacles, use [`add()`](Self::add) with a full `Isometry3`.
    pub fn add_box(&mut self, name: &str, half_extents: [f64; 3], position: [f64; 3]) {
        self.add(
            name,
            Shape::Cuboid(half_extents[0], half_extents[1], half_extents[2]),
            Isometry3::translation(position[0], position[1], position[2]),
        );
    }

    /// Add a sphere obstacle at a position.
    pub fn add_sphere(&mut self, name: &str, radius: f64, position: [f64; 3]) {
        self.add(
            name,
            Shape::Sphere(radius),
            Isometry3::translation(position[0], position[1], position[2]),
        );
    }

    /// Add a cylinder obstacle at a position (no rotation, axis along Z).
    ///
    /// For tilted cylinders, use [`add()`](Self::add) with a full `Isometry3`.
    pub fn add_cylinder(
        &mut self,
        name: &str,
        radius: f64,
        half_height: f64,
        position: [f64; 3],
    ) {
        self.add(
            name,
            Shape::Cylinder(radius, half_height),
            Isometry3::translation(position[0], position[1], position[2]),
        );
    }

    /// Remove an object from the scene. Returns the removed object if it existed.
    pub fn remove(&mut self, name: &str) -> Option<SceneObject> {
        self.objects.remove(name)
    }

    /// Remove all objects from the scene (regular, attached, pointclouds, and octrees).
    pub fn clear(&mut self) {
        self.objects.clear();
        self.attached.clear();
        self.pointclouds.clear();
        self.octrees.clear();
    }

    /// Iterate over all scene objects.
    pub fn objects_iter(&self) -> impl Iterator<Item = &SceneObject> {
        self.objects.values()
    }

    /// Iterate over all attached objects.
    pub fn attached_iter(&self) -> impl Iterator<Item = &AttachedObject> {
        self.attached.values()
    }

    /// Update the pose of an existing scene object.
    pub fn update_pose(&mut self, name: &str, pose: Isometry3<f64>) -> bool {
        if let Some(obj) = self.objects.get_mut(name) {
            obj.pose = pose;
            true
        } else {
            false
        }
    }

    // === Attach/Detach ===

    /// Attach an object to a robot link.
    ///
    /// - Removes the object from scene objects if it exists there.
    /// - Adds to attached objects with the given grasp transform.
    /// - Automatically allows collision between the attached object and parent link.
    pub fn attach(
        &mut self,
        name: &str,
        shape: Shape,
        grasp_transform: Isometry3<f64>,
        parent_link: &str,
    ) {
        // Remove from scene objects if it was there
        self.objects.remove(name);

        // Allow collision between attached object and parent link
        self.acm.allow(name, parent_link);

        self.attached.insert(
            name.to_string(),
            AttachedObject {
                name: name.to_string(),
                shape,
                grasp_transform,
                parent_link: parent_link.to_string(),
            },
        );
    }

    /// Detach an object from the robot and place it in the scene.
    ///
    /// - Removes from attached objects.
    /// - Adds back to scene objects at the given pose.
    /// - Removes ACM entry for the object/link pair.
    pub fn detach(&mut self, name: &str, place_pose: Isometry3<f64>) -> bool {
        if let Some(att) = self.attached.remove(name) {
            self.acm.disallow(name, &att.parent_link);

            self.objects.insert(
                name.to_string(),
                SceneObject {
                    name: name.to_string(),
                    shape: att.shape,
                    pose: place_pose,
                },
            );
            true
        } else {
            false
        }
    }

    // === ACM Management ===

    /// Allow collision between two named entities (links or objects).
    pub fn allow_collision(&mut self, a: &str, b: &str) {
        self.acm.allow(a, b);
    }

    /// Disallow collision between two named entities.
    pub fn disallow_collision(&mut self, a: &str, b: &str) {
        self.acm.disallow(a, b);
    }

    // === Point Cloud / Depth Ingestion ===

    /// Add a point cloud source to the scene as collision obstacles.
    ///
    /// Points are processed through the configured pipeline (crop, voxel filter,
    /// floor removal, etc.) and converted to collision spheres.
    ///
    /// If a source with the same name already exists, it is replaced.
    pub fn add_pointcloud(&mut self, name: &str, points: &[[f64; 3]], config: PointCloudConfig) {
        let raw_count = points.len();
        let (processed_pts, spheres) = pointcloud::process_pointcloud(points, &config);

        self.pointclouds.insert(
            name.to_string(),
            PointCloudSource {
                name: name.to_string(),
                raw_count,
                processed_count: processed_pts.len(),
                config,
                spheres,
            },
        );
    }

    /// Update an existing point cloud source with new data.
    ///
    /// Reprocesses the new points using the source's existing config.
    /// Returns `false` if the source doesn't exist (use `add_pointcloud` first).
    pub fn update_pointcloud(&mut self, name: &str, points: &[[f64; 3]]) -> bool {
        if let Some(source) = self.pointclouds.get(name) {
            let config = source.config.clone();
            self.add_pointcloud(name, points, config);
            true
        } else {
            false
        }
    }

    /// Remove a point cloud source from the scene.
    pub fn remove_pointcloud(&mut self, name: &str) -> bool {
        self.pointclouds.remove(name).is_some()
    }

    /// Number of active point cloud sources.
    pub fn num_pointclouds(&self) -> usize {
        self.pointclouds.len()
    }

    /// Get a reference to a point cloud source.
    pub fn get_pointcloud(&self, name: &str) -> Option<&PointCloudSource> {
        self.pointclouds.get(name)
    }

    /// Ingest a depth image as a point cloud collision source.
    ///
    /// Back-projects depth pixels to 3D points using camera intrinsics,
    /// transforms to world frame via `camera_pose`, then processes as
    /// a point cloud with the given config.
    ///
    /// The source name is derived from the camera (you choose the name).
    #[allow(clippy::too_many_arguments)]
    pub fn update_from_depth(
        &mut self,
        name: &str,
        depth_image: &[f32],
        width: usize,
        height: usize,
        intrinsics: &CameraIntrinsics,
        camera_pose: &Isometry3<f64>,
        depth_config: &DepthConfig,
        pc_config: PointCloudConfig,
    ) {
        let points = depth::depth_to_points_world(
            depth_image,
            width,
            height,
            intrinsics,
            camera_pose,
            depth_config,
        );
        self.add_pointcloud(name, &points, pc_config);
    }

    // === Octree Management ===

    /// Add an octree-based volumetric occupancy map to the scene.
    ///
    /// If an octree with the same name already exists, it is replaced.
    pub fn add_octree(&mut self, name: &str, octree: Octree) {
        self.octrees.insert(name.to_string(), octree);
    }

    /// Get a mutable reference to a named octree for incremental updates.
    pub fn get_octree_mut(&mut self, name: &str) -> Option<&mut Octree> {
        self.octrees.get_mut(name)
    }

    /// Get a reference to a named octree.
    pub fn get_octree(&self, name: &str) -> Option<&Octree> {
        self.octrees.get(name)
    }

    /// Remove an octree from the scene.
    pub fn remove_octree(&mut self, name: &str) -> bool {
        self.octrees.remove(name).is_some()
    }

    /// Number of active octrees.
    pub fn num_octrees(&self) -> usize {
        self.octrees.len()
    }

    /// Insert a point cloud into a named octree with sensor-model ray-casting.
    ///
    /// Creates the octree with default config if it doesn't exist.
    /// Returns `false` if the octree didn't exist (was created with defaults).
    pub fn update_octree(
        &mut self,
        name: &str,
        points: &[[f64; 3]],
        sensor_origin: &[f64; 3],
    ) -> bool {
        let existed = self.octrees.contains_key(name);
        let tree = self
            .octrees
            .entry(name.to_string())
            .or_insert_with(|| Octree::new(OctreeConfig::default()));
        tree.insert_pointcloud(points, sensor_origin);
        existed
    }

    // === Collision Queries ===

    /// Build the environment collision spheres from all scene objects, point clouds,
    /// and octrees.
    pub fn build_environment_spheres(&self) -> SpheresSoA {
        let mut spheres = SpheresSoA::new();

        // Shape-based obstacles
        for obj in self.objects.values() {
            let obj_spheres = obj.shape.to_spheres(&obj.pose, self.sphere_resolution);
            for (x, y, z, r) in obj_spheres {
                spheres.push(x, y, z, r, 0);
            }
        }

        // Point cloud obstacles
        for pc in self.pointclouds.values() {
            for i in 0..pc.spheres.len() {
                spheres.push(
                    pc.spheres.x[i],
                    pc.spheres.y[i],
                    pc.spheres.z[i],
                    pc.spheres.radius[i],
                    0,
                );
            }
        }

        // Octree obstacles
        for tree in self.octrees.values() {
            let oct_spheres = tree.to_collision_spheres();
            for i in 0..oct_spheres.len() {
                spheres.push(
                    oct_spheres.x[i],
                    oct_spheres.y[i],
                    oct_spheres.z[i],
                    oct_spheres.radius[i],
                    0,
                );
            }
        }

        spheres
    }

    /// Build collision spheres for attached objects given link poses.
    fn build_attached_spheres(&self, link_poses: &[Pose]) -> SpheresSoA {
        let mut spheres = SpheresSoA::new();
        for att in self.attached.values() {
            if let Ok(link_idx) = self.robot.link_index(&att.parent_link) {
                if link_idx < link_poses.len() {
                    let link_pose = link_poses[link_idx].0;
                    let world_pose = link_pose * att.grasp_transform;
                    let att_spheres = att.shape.to_spheres(&world_pose, self.sphere_resolution);
                    for (x, y, z, r) in att_spheres {
                        spheres.push(x, y, z, r, link_idx);
                    }
                }
            }
        }
        spheres
    }

    /// Compute link poses from joint values via FK.
    fn compute_link_poses(&self, joint_values: &[f64]) -> kinetic_core::Result<Vec<Pose>> {
        forward_kinematics_all(&self.robot, &self.chain, joint_values)
    }

    /// Check if the robot is in collision with any scene objects.
    ///
    /// Checks: robot self-collision, robot vs environment, attached vs environment.
    pub fn check_collision(&self, joint_values: &[f64]) -> kinetic_core::Result<bool> {
        let link_poses = self.compute_link_poses(joint_values)?;

        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);

        let env_spheres = self.build_environment_spheres();

        // Check robot self-collision
        let skip_pairs = self.acm.resolve_to_indices(&self.robot);
        if runtime.self_collision(&skip_pairs) {
            return Ok(true);
        }

        // Check robot vs environment
        if !env_spheres.is_empty() && runtime.collides_with(&env_spheres) {
            return Ok(true);
        }

        // Check attached objects vs environment
        if !self.attached.is_empty() && !env_spheres.is_empty() {
            let att_spheres = self.build_attached_spheres(&link_poses);
            if !att_spheres.is_empty() && att_spheres.any_overlap(&env_spheres) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Compute the minimum distance from the robot to any scene object.
    ///
    /// Returns `f64::INFINITY` if there are no objects.
    pub fn min_distance_to_robot(&self, joint_values: &[f64]) -> kinetic_core::Result<f64> {
        let link_poses = self.compute_link_poses(joint_values)?;

        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);

        let env_spheres = self.build_environment_spheres();

        if env_spheres.is_empty() {
            return Ok(f64::INFINITY);
        }

        if let Some((dist, _, _)) = runtime.world.min_distance(&env_spheres) {
            Ok(dist)
        } else {
            Ok(f64::INFINITY)
        }
    }

    /// Get contact points within a given margin.
    pub fn contact_points(
        &self,
        joint_values: &[f64],
        margin: f64,
    ) -> kinetic_core::Result<Vec<ContactPoint>> {
        let link_poses = self.compute_link_poses(joint_values)?;

        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);

        let env_spheres = self.build_environment_spheres();
        let mut contacts = Vec::new();

        if env_spheres.is_empty() {
            return Ok(contacts);
        }

        let robot_sph = &runtime.world;
        for i in 0..robot_sph.len() {
            for j in 0..env_spheres.len() {
                let dist = robot_sph.signed_distance(i, &env_spheres, j);
                if dist <= margin {
                    let rx = robot_sph.x[i];
                    let ry = robot_sph.y[i];
                    let rz = robot_sph.z[i];
                    let ex = env_spheres.x[j];
                    let ey = env_spheres.y[j];
                    let ez = env_spheres.z[j];

                    let dx = ex - rx;
                    let dy = ey - ry;
                    let dz = ez - rz;
                    let d_len = (dx * dx + dy * dy + dz * dz).sqrt();

                    let (nx, ny, nz) = if d_len > 1e-12 {
                        (dx / d_len, dy / d_len, dz / d_len)
                    } else {
                        (0.0, 0.0, 1.0)
                    };

                    let rr = robot_sph.radius[i];
                    let px = rx + nx * rr;
                    let py = ry + ny * rr;
                    let pz = rz + nz * rr;

                    contacts.push(ContactPoint {
                        point_robot: nalgebra::Point3::new(px, py, pz),
                        point_obstacle: nalgebra::Point3::new(
                            ex - env_spheres.radius[j] * nx,
                            ey - env_spheres.radius[j] * ny,
                            ez - env_spheres.radius[j] * nz,
                        ),
                        distance: dist,
                        link_idx: robot_sph.link_id[i],
                    });
                }
            }
        }

        Ok(contacts)
    }
}

/// Auto-detect kinematic chain from robot (planning groups or tree walk).
fn auto_detect_chain(robot: &Robot) -> kinetic_core::Result<KinematicChain> {
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

// === Shape to sphere approximation helpers ===

fn cuboid_to_spheres(
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

fn cylinder_to_spheres(
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

fn half_space_to_spheres(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// URDF with collision geometry for testing.
    const TEST_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="ee_link">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
</robot>"#;

    fn test_robot() -> Robot {
        Robot::from_urdf_string(TEST_URDF).unwrap()
    }

    #[test]
    fn scene_creation() {
        let robot = test_robot();
        let scene = Scene::new(&robot).unwrap();
        assert_eq!(scene.num_objects(), 0);
        assert_eq!(scene.num_attached(), 0);
        assert_eq!(scene.dof(), 3);
    }

    #[test]
    fn add_remove_objects() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        let pose = Isometry3::translation(1.0, 0.0, 0.5);

        scene.add("table", Shape::Cuboid(0.5, 0.3, 0.02), pose);
        assert_eq!(scene.num_objects(), 1);
        assert!(scene.get_object("table").is_some());

        scene.add("cup", Shape::Cylinder(0.04, 0.06), Isometry3::identity());
        assert_eq!(scene.num_objects(), 2);

        let removed = scene.remove("table");
        assert!(removed.is_some());
        assert_eq!(scene.num_objects(), 1);

        let not_found = scene.remove("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn update_pose() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        let pose1 = Isometry3::translation(1.0, 0.0, 0.0);
        let pose2 = Isometry3::translation(2.0, 0.0, 0.0);

        scene.add("box", Shape::Cuboid(0.1, 0.1, 0.1), pose1);
        assert!(scene.update_pose("box", pose2));
        assert!(!scene.update_pose("nonexistent", pose2));

        let obj = scene.get_object("box").unwrap();
        assert!((obj.pose.translation.x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn attach_detach() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        let pose = Isometry3::translation(1.0, 0.0, 0.5);
        scene.add("bolt", Shape::Cylinder(0.005, 0.03), pose);
        assert_eq!(scene.num_objects(), 1);

        let grasp_tf = Isometry3::translation(0.0, 0.0, 0.05);
        scene.attach("bolt", Shape::Cylinder(0.005, 0.03), grasp_tf, "base_link");

        assert_eq!(scene.num_objects(), 0);
        assert_eq!(scene.num_attached(), 1);
        assert!(scene.get_attached("bolt").is_some());
        assert!(scene.acm().is_allowed("bolt", "base_link"));

        let place_pose = Isometry3::translation(0.5, 0.0, 0.3);
        assert!(scene.detach("bolt", place_pose));

        assert_eq!(scene.num_objects(), 1);
        assert_eq!(scene.num_attached(), 0);
        assert!(!scene.acm().is_allowed("bolt", "base_link"));
    }

    #[test]
    fn acm_management() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        scene.allow_collision("base_link", "table");
        assert!(scene.acm().is_allowed("base_link", "table"));

        scene.disallow_collision("base_link", "table");
        assert!(!scene.acm().is_allowed("base_link", "table"));
    }

    #[test]
    fn shape_sphere_generation() {
        let pose = Isometry3::identity();

        let spheres = Shape::Sphere(0.1).to_spheres(&pose, 0.02);
        assert_eq!(spheres.len(), 1);
        assert!((spheres[0].3 - 0.1).abs() < 1e-10);

        let spheres = Shape::Cuboid(0.1, 0.1, 0.1).to_spheres(&pose, 0.05);
        assert!(spheres.len() > 1);

        let spheres = Shape::Cylinder(0.1, 0.1).to_spheres(&pose, 0.05);
        assert!(spheres.len() > 1);

        let spheres = Shape::HalfSpace(Vector3::z(), 0.0).to_spheres(&pose, 0.1);
        assert!(!spheres.is_empty());
    }

    #[test]
    fn collision_check_no_objects() {
        let robot = test_robot();
        let scene = Scene::new(&robot).unwrap();

        let zeros = vec![0.0; scene.dof()];
        let result = scene.check_collision(&zeros);
        assert!(result.is_ok());
    }

    #[test]
    fn collision_check_with_object() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Place a sphere right at the robot base — should collide
        scene.add(
            "obstacle",
            Shape::Sphere(0.2),
            Isometry3::translation(0.0, 0.0, 0.0),
        );

        let zeros = vec![0.0; scene.dof()];
        let result = scene.check_collision(&zeros);
        assert!(result.is_ok());
        // Very likely colliding since sphere overlaps base
        assert!(
            result.unwrap(),
            "Sphere at origin should collide with robot base"
        );
    }

    #[test]
    fn min_distance_no_objects() {
        let robot = test_robot();
        let scene = Scene::new(&robot).unwrap();

        let zeros = vec![0.0; scene.dof()];
        let dist = scene.min_distance_to_robot(&zeros);
        assert!(dist.is_ok());
        assert!(dist.unwrap() == f64::INFINITY);
    }

    #[test]
    fn min_distance_with_object() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Place object far away
        scene.add(
            "far_box",
            Shape::Sphere(0.1),
            Isometry3::translation(10.0, 0.0, 0.0),
        );

        let zeros = vec![0.0; scene.dof()];
        let dist = scene.min_distance_to_robot(&zeros).unwrap();
        assert!(dist > 0.0, "Far object should have positive distance");
        assert!(dist < f64::INFINITY, "Should have finite distance");
    }

    #[test]
    fn contact_points_no_objects() {
        let robot = test_robot();
        let scene = Scene::new(&robot).unwrap();

        let zeros = vec![0.0; scene.dof()];
        let contacts = scene.contact_points(&zeros, 0.1);
        assert!(contacts.is_ok());
        assert!(contacts.unwrap().is_empty());
    }

    #[test]
    fn detach_nonexistent() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();
        assert!(!scene.detach("nonexistent", Isometry3::identity()));
    }

    // === Point Cloud / Depth Integration Tests ===

    #[test]
    fn pointcloud_add_remove() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        assert_eq!(scene.num_pointclouds(), 0);

        let points: Vec<[f64; 3]> = (0..100)
            .map(|i| [0.5 + i as f64 * 0.001, 0.0, 0.3])
            .collect();

        scene.add_pointcloud("cam0", &points, PointCloudConfig::default());
        assert_eq!(scene.num_pointclouds(), 1);

        let source = scene.get_pointcloud("cam0").unwrap();
        assert_eq!(source.raw_count, 100);
        assert_eq!(source.processed_count, 100);

        assert!(scene.remove_pointcloud("cam0"));
        assert_eq!(scene.num_pointclouds(), 0);
        assert!(!scene.remove_pointcloud("cam0"));
    }

    #[test]
    fn pointcloud_update_replaces_data() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        let points1: Vec<[f64; 3]> = (0..50).map(|i| [i as f64 * 0.01, 0.0, 0.5]).collect();
        let points2: Vec<[f64; 3]> = (0..80).map(|i| [i as f64 * 0.01, 0.0, 0.3]).collect();

        scene.add_pointcloud("cam0", &points1, PointCloudConfig::default());
        assert_eq!(scene.get_pointcloud("cam0").unwrap().raw_count, 50);

        assert!(scene.update_pointcloud("cam0", &points2));
        assert_eq!(scene.get_pointcloud("cam0").unwrap().raw_count, 80);

        // Non-existent source
        assert!(!scene.update_pointcloud("nonexistent", &points2));
    }

    #[test]
    fn pointcloud_collision_detected() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Without pointcloud — no collision (ignoring self-collision)
        let zeros = vec![0.0; scene.dof()];

        // Dense cloud of points right at the robot base
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push([
                    -0.05 + i as f64 * 0.01,
                    -0.05 + j as f64 * 0.01,
                    0.0, // at robot base height
                ]);
            }
        }

        scene.add_pointcloud(
            "obstacle_cloud",
            &points,
            PointCloudConfig {
                sphere_radius: 0.05,
                ..Default::default()
            },
        );

        let result = scene.check_collision(&zeros).unwrap();
        assert!(
            result,
            "Dense point cloud at robot base should cause collision"
        );
    }

    #[test]
    fn pointcloud_far_away_no_collision() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Points far from robot
        let points: Vec<[f64; 3]> = (0..100)
            .map(|i| [10.0 + i as f64 * 0.01, 10.0, 10.0])
            .collect();

        scene.add_pointcloud("far_cloud", &points, PointCloudConfig::default());

        let zeros = vec![0.0; scene.dof()];
        let dist = scene.min_distance_to_robot(&zeros).unwrap();
        assert!(
            dist > 1.0,
            "Far point cloud should have large distance: {}",
            dist
        );
    }

    #[test]
    fn pointcloud_included_in_environment_spheres() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Add a regular object
        scene.add(
            "table",
            Shape::Cuboid(0.5, 0.3, 0.02),
            Isometry3::translation(1.0, 0.0, 0.5),
        );

        let env_before = scene.build_environment_spheres();
        let count_before = env_before.len();

        // Add point cloud
        let points: Vec<[f64; 3]> = (0..50).map(|i| [2.0 + i as f64 * 0.01, 0.0, 0.5]).collect();
        scene.add_pointcloud("cam0", &points, PointCloudConfig::default());

        let env_after = scene.build_environment_spheres();
        assert!(
            env_after.len() > count_before,
            "Environment spheres should include point cloud: {} > {}",
            env_after.len(),
            count_before
        );
    }

    #[test]
    fn depth_image_creates_pointcloud() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        let width = 64;
        let height = 48;
        let mut depth_image = vec![0.0f32; width * height];

        // Put a 1m depth reading at center
        for v in 20..28 {
            for u in 28..36 {
                depth_image[v * width + u] = 1.0;
            }
        }

        let intrinsics = CameraIntrinsics::new(50.0, 50.0, 32.0, 24.0);
        let camera_pose = Isometry3::translation(0.0, 0.0, 0.0);
        let depth_config = DepthConfig::default();
        let pc_config = PointCloudConfig::default();

        scene.update_from_depth(
            "depth_cam",
            &depth_image,
            width,
            height,
            &intrinsics,
            &camera_pose,
            &depth_config,
            pc_config,
        );

        assert_eq!(scene.num_pointclouds(), 1);
        let source = scene.get_pointcloud("depth_cam").unwrap();
        assert!(source.processed_count > 0);
        assert!(!source.spheres.is_empty());
    }

    #[test]
    fn contact_points_with_nearby_object() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Place a sphere near the robot (should produce contact points)
        scene.add(
            "nearby",
            Shape::Sphere(0.15),
            Isometry3::translation(0.0, 0.0, 0.15),
        );

        let zeros = vec![0.0; scene.dof()];
        let contacts = scene.contact_points(&zeros, 0.5).unwrap(); // Large margin
                                                                   // Should find some contact points with generous margin
        assert!(
            !contacts.is_empty(),
            "Should have contact points with nearby object"
        );

        for cp in &contacts {
            assert!(cp.distance.is_finite());
        }
    }

    #[test]
    fn contact_points_far_object() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        scene.add(
            "far",
            Shape::Sphere(0.05),
            Isometry3::translation(50.0, 50.0, 50.0),
        );

        let zeros = vec![0.0; scene.dof()];
        let contacts = scene.contact_points(&zeros, 0.1).unwrap();
        assert!(
            contacts.is_empty(),
            "Far object should have no contacts: got {}",
            contacts.len()
        );
    }

    #[test]
    fn collision_check_with_attached_object() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        // Add obstacle
        scene.add(
            "wall",
            Shape::Cuboid(0.5, 0.5, 0.5),
            Isometry3::translation(0.0, 0.0, 1.0),
        );

        // Attach object to ee_link
        scene.attach(
            "tool",
            Shape::Cylinder(0.1, 0.3),
            Isometry3::identity(),
            "ee_link",
        );

        let zeros = vec![0.0; scene.dof()];
        let result = scene.check_collision(&zeros);
        assert!(result.is_ok());
        // May or may not collide — just verify it doesn't crash with attached objects
    }

    #[test]
    fn min_distance_with_attached() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        scene.add(
            "box",
            Shape::Cuboid(0.1, 0.1, 0.1),
            Isometry3::translation(5.0, 0.0, 0.0),
        );

        scene.attach(
            "gripper_part",
            Shape::Sphere(0.02),
            Isometry3::identity(),
            "ee_link",
        );

        let zeros = vec![0.0; scene.dof()];
        let dist = scene.min_distance_to_robot(&zeros).unwrap();
        assert!(dist > 0.0 && dist.is_finite());
    }

    #[test]
    fn auto_detect_chain_from_urdf() {
        // Test auto_detect_chain with our test robot (has no groups)
        let robot = test_robot();
        let chain = auto_detect_chain(&robot);
        assert!(chain.is_ok(), "auto_detect_chain should succeed");
        let chain = chain.unwrap();
        assert!(chain.dof > 0);
    }

    #[test]
    fn clear_removes_pointclouds() {
        let robot = test_robot();
        let mut scene = Scene::new(&robot).unwrap();

        scene.add("table", Shape::Sphere(0.1), Isometry3::identity());
        let points = [[0.5, 0.0, 0.5]];
        scene.add_pointcloud("cam0", &points, PointCloudConfig::default());

        assert_eq!(scene.num_objects(), 1);
        assert_eq!(scene.num_pointclouds(), 1);

        scene.clear();
        assert_eq!(scene.num_objects(), 0);
        assert_eq!(scene.num_pointclouds(), 0);
    }
}
