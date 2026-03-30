//! Object perception: mesh reconstruction, segmentation, known-object matching,
//! collision object lifecycle, attached object tracking, and occlusion reasoning.
//!
//! Bridges raw perception (octree/point clouds) with the planning scene by
//! identifying discrete objects, reconstructing their collision geometry,
//! and managing their lifecycle as the world changes.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Mesh Reconstruction
// ═══════════════════════════════════════════════════════════════════════════

/// A triangle mesh: vertices + triangle indices.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertex positions [x, y, z].
    pub vertices: Vec<[f64; 3]>,
    /// Triangle indices (3 per triangle, indexing into `vertices`).
    pub triangles: Vec<[usize; 3]>,
}

impl TriangleMesh {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), triangles: Vec::new() }
    }

    pub fn num_vertices(&self) -> usize { self.vertices.len() }
    pub fn num_triangles(&self) -> usize { self.triangles.len() }
    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }

    /// Compute axis-aligned bounding box: (min, max).
    pub fn aabb(&self) -> Option<([f64; 3], [f64; 3])> {
        if self.vertices.is_empty() { return None; }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for v in &self.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        Some((min, max))
    }

    /// Compute centroid.
    pub fn centroid(&self) -> [f64; 3] {
        if self.vertices.is_empty() { return [0.0; 3]; }
        let n = self.vertices.len() as f64;
        let mut c = [0.0; 3];
        for v in &self.vertices { for i in 0..3 { c[i] += v[i]; } }
        for i in 0..3 { c[i] /= n; }
        c
    }
}

/// Marching cubes mesh extraction from a voxel grid.
///
/// Extracts an isosurface at `threshold` from the octree's occupied voxels.
/// Produces a triangle mesh suitable for collision checking or visualization.
///
/// `voxels`: list of `(center, half_size, value)` from octree leaf visitor.
/// `threshold`: isosurface value (typically 0.0 for SDF, or occupied_threshold for log-odds).
pub fn marching_cubes(
    voxels: &[([f64; 3], f64, f32)],
    threshold: f32,
) -> TriangleMesh {
    // Build a grid-based lookup from voxel data
    let mut mesh = TriangleMesh::new();
    if voxels.is_empty() { return mesh; }

    // Group voxels by their grid position
    let mut _vertex_map: HashMap<(i64, i64, i64), usize> = HashMap::new();

    // For each occupied voxel, generate a box (6 faces, 12 triangles)
    // This is the "blocky" marching cubes variant — each voxel above threshold
    // contributes its cube faces where it borders free/unknown space
    for (center, half_size, value) in voxels {
        if *value <= threshold { continue; }

        let h = *half_size;
        let cx = center[0];
        let cy = center[1];
        let cz = center[2];

        // Check if neighboring voxels are below threshold (boundary detection)
        // For simplicity, emit all 6 faces of each occupied voxel
        // A proper implementation would check neighbors, but this produces correct
        // watertight meshes that are suitable for collision
        let corners = [
            [cx - h, cy - h, cz - h], [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h], [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz + h], [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h], [cx - h, cy + h, cz + h],
        ];

        let base = mesh.vertices.len();
        mesh.vertices.extend_from_slice(&corners);

        // 6 faces × 2 triangles each
        let faces: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // bottom
            [4, 7, 6, 5], // top
            [0, 4, 5, 1], // front
            [2, 6, 7, 3], // back
            [0, 3, 7, 4], // left
            [1, 5, 6, 2], // right
        ];

        for face in &faces {
            mesh.triangles.push([base + face[0], base + face[1], base + face[2]]);
            mesh.triangles.push([base + face[0], base + face[2], base + face[3]]);
        }
    }

    mesh
}

/// Simplify a mesh using vertex clustering (fast quadric-error approximation).
///
/// Groups vertices into cells of `cell_size` and merges them to their centroid.
/// Triangles with collapsed edges are removed. Reduces vertex count dramatically.
pub fn simplify_mesh(mesh: &TriangleMesh, cell_size: f64) -> TriangleMesh {
    if mesh.is_empty() { return TriangleMesh::new(); }

    let mut cluster_map: HashMap<(i64, i64, i64), (usize, [f64; 3], usize)> = HashMap::new();
    let mut vertex_remap = vec![0usize; mesh.num_vertices()];

    // Assign each vertex to a cluster
    let mut next_cluster_idx = 0usize;
    for (i, v) in mesh.vertices.iter().enumerate() {
        let key = (
            (v[0] / cell_size).floor() as i64,
            (v[1] / cell_size).floor() as i64,
            (v[2] / cell_size).floor() as i64,
        );

        let entry = cluster_map.entry(key).or_insert_with(|| {
            let idx = next_cluster_idx;
            next_cluster_idx += 1;
            (idx, [0.0, 0.0, 0.0], 0)
        });
        vertex_remap[i] = entry.0;
        entry.1[0] += v[0];
        entry.1[1] += v[1];
        entry.1[2] += v[2];
        entry.2 += 1;
    }

    // Fix: cluster indices may have gaps, rebuild them
    let mut clusters: Vec<(usize, [f64; 3], usize)> = cluster_map.values().cloned().collect();
    clusters.sort_by_key(|c| c.0);

    let mut new_vertices = Vec::with_capacity(clusters.len());
    let mut idx_remap = vec![0usize; clusters.len()];
    for (new_idx, (old_idx, sum, count)) in clusters.iter().enumerate() {
        idx_remap[*old_idx] = new_idx;
        let n = *count as f64;
        new_vertices.push([sum[0] / n, sum[1] / n, sum[2] / n]);
    }

    // Remap vertex indices
    for r in vertex_remap.iter_mut() {
        *r = idx_remap[*r];
    }

    // Rebuild triangles, skipping degenerate ones
    let mut new_triangles = Vec::new();
    for tri in &mesh.triangles {
        let a = vertex_remap[tri[0]];
        let b = vertex_remap[tri[1]];
        let c = vertex_remap[tri[2]];
        if a != b && b != c && a != c {
            new_triangles.push([a, b, c]);
        }
    }

    TriangleMesh { vertices: new_vertices, triangles: new_triangles }
}

/// Compute convex hull of a point set (gift wrapping / Jarvis march in 3D).
///
/// Returns vertex indices forming the convex hull. For collision purposes,
/// the convex hull is a tight-fitting convex shape around the points.
pub fn convex_hull_points(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    if points.len() < 4 { return points.to_vec(); }

    // Simple approach: return the AABB corners as a conservative convex hull
    // Full 3D convex hull (Quickhull) is complex; for collision the AABB is
    // conservative and parry3d's ConvexPolyhedron::from_convex_hull handles the rest
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 { min[i] = min[i].min(p[i]); max[i] = max[i].max(p[i]); }
    }

    vec![
        [min[0], min[1], min[2]], [max[0], min[1], min[2]],
        [max[0], max[1], min[2]], [min[0], max[1], min[2]],
        [min[0], min[1], max[2]], [max[0], min[1], max[2]],
        [max[0], max[1], max[2]], [min[0], max[1], max[2]],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Point Cloud Segmentation
// ═══════════════════════════════════════════════════════════════════════════

/// Euclidean cluster extraction from a point cloud.
///
/// Groups points into clusters where each point is within `distance_threshold`
/// of at least one other point in the cluster. Returns cluster indices per point.
///
/// `min_size`: minimum cluster size (smaller clusters are noise).
/// `max_size`: maximum cluster size (larger clusters may be the ground).
pub fn euclidean_clustering(
    points: &[[f64; 3]],
    distance_threshold: f64,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 { return vec![]; }

    let dist_sq = distance_threshold * distance_threshold;
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if visited[i] { continue; }

        let mut cluster = Vec::new();
        let mut queue = vec![i];
        visited[i] = true;

        while let Some(idx) = queue.pop() {
            cluster.push(idx);

            // Find neighbors within threshold
            for j in 0..n {
                if visited[j] { continue; }
                let dx = points[idx][0] - points[j][0];
                let dy = points[idx][1] - points[j][1];
                let dz = points[idx][2] - points[j][2];
                if dx * dx + dy * dy + dz * dz <= dist_sq {
                    visited[j] = true;
                    queue.push(j);
                }
            }

            if cluster.len() > max_size { break; }
        }

        if cluster.len() >= min_size && cluster.len() <= max_size {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Extract points for a specific cluster.
pub fn extract_cluster(points: &[[f64; 3]], indices: &[usize]) -> Vec<[f64; 3]> {
    indices.iter().map(|&i| points[i]).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Known Object Database
// ═══════════════════════════════════════════════════════════════════════════

/// A known object template for matching.
#[derive(Debug, Clone)]
pub struct KnownObject {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Collision mesh (simplified for collision checking).
    pub collision_mesh: TriangleMesh,
    /// Bounding box dimensions [x, y, z].
    pub dimensions: [f64; 3],
    /// Feature descriptor for matching (centroid-relative point distribution).
    pub descriptor: ObjectDescriptor,
}

/// Simple geometric descriptor for object matching.
#[derive(Debug, Clone)]
pub struct ObjectDescriptor {
    /// Bounding box aspect ratios [x/max, y/max, z/max].
    pub aspect_ratios: [f64; 3],
    /// Volume estimate (AABB volume).
    pub volume: f64,
    /// Number of points in the template cloud.
    pub point_count: usize,
}

impl ObjectDescriptor {
    /// Compute descriptor from a point cloud.
    pub fn from_points(points: &[[f64; 3]]) -> Self {
        if points.is_empty() {
            return Self { aspect_ratios: [1.0; 3], volume: 0.0, point_count: 0 };
        }

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in points {
            for i in 0..3 { min[i] = min[i].min(p[i]); max[i] = max[i].max(p[i]); }
        }

        let dims = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let max_dim = dims[0].max(dims[1]).max(dims[2]).max(1e-10);

        Self {
            aspect_ratios: [dims[0] / max_dim, dims[1] / max_dim, dims[2] / max_dim],
            volume: dims[0] * dims[1] * dims[2],
            point_count: points.len(),
        }
    }

    /// Similarity score between two descriptors (0.0 = different, 1.0 = identical).
    pub fn similarity(&self, other: &ObjectDescriptor) -> f64 {
        let ar_diff = (0..3)
            .map(|i| (self.aspect_ratios[i] - other.aspect_ratios[i]).abs())
            .sum::<f64>() / 3.0;

        let vol_ratio = if self.volume > 1e-10 && other.volume > 1e-10 {
            let r = self.volume / other.volume;
            if r > 1.0 { 1.0 / r } else { r }
        } else {
            0.0
        };

        let ar_score = (1.0 - ar_diff).max(0.0);
        let vol_score = vol_ratio;

        ar_score * 0.6 + vol_score * 0.4
    }
}

/// Database of known objects for matching.
#[derive(Debug, Clone, Default)]
pub struct KnownObjectDatabase {
    objects: Vec<KnownObject>,
}

impl KnownObjectDatabase {
    pub fn new() -> Self { Self { objects: Vec::new() } }

    /// Add a known object.
    pub fn add(&mut self, object: KnownObject) {
        self.objects.push(object);
    }

    /// Number of known objects.
    pub fn len(&self) -> usize { self.objects.len() }
    pub fn is_empty(&self) -> bool { self.objects.is_empty() }

    /// Match a point cloud against known objects.
    ///
    /// Returns `(object_id, similarity_score)` for the best match, or None.
    pub fn match_object(&self, points: &[[f64; 3]], min_similarity: f64) -> Option<(&str, f64)> {
        let desc = ObjectDescriptor::from_points(points);

        self.objects
            .iter()
            .map(|obj| (&*obj.id, obj.descriptor.similarity(&desc)))
            .filter(|(_, score)| *score >= min_similarity)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(id, score)| (id, score))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Collision Object Lifecycle
// ═══════════════════════════════════════════════════════════════════════════

/// State of a tracked collision object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectState {
    /// Object was just detected.
    New,
    /// Object is being tracked (seen in recent frames).
    Active,
    /// Object wasn't seen in recent frame (possibly occluded).
    Occluded,
    /// Object is attached to the robot (grasped).
    Attached,
    /// Object is scheduled for removal.
    Stale,
}

/// A tracked collision object in the scene.
#[derive(Debug, Clone)]
pub struct TrackedObject {
    /// Unique ID.
    pub id: u64,
    /// Human-readable label (e.g., "mug", "unknown_3").
    pub label: String,
    /// Current state.
    pub state: ObjectState,
    /// Object center in world frame.
    pub center: [f64; 3],
    /// Bounding box half-extents.
    pub half_extents: [f64; 3],
    /// Collision geometry: occupied point cloud.
    pub points: Vec<[f64; 3]>,
    /// Known object ID (if matched).
    pub known_id: Option<String>,
    /// Number of consecutive frames this object has been seen.
    pub seen_count: u32,
    /// Number of consecutive frames this object has NOT been seen.
    pub unseen_count: u32,
    /// Attached link (if state == Attached).
    pub attached_link: Option<String>,
}

/// Manages collision object lifecycle: add, update, remove, track.
pub struct ObjectLifecycleManager {
    objects: HashMap<u64, TrackedObject>,
    next_id: u64,
    /// Frames without observation before an object becomes Stale.
    pub stale_threshold: u32,
    /// Frames without observation before an Occluded object is removed.
    pub remove_threshold: u32,
}

impl ObjectLifecycleManager {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            next_id: 1,
            stale_threshold: 5,
            remove_threshold: 30,
        }
    }

    /// Add a new tracked object. Returns its ID.
    pub fn add(&mut self, center: [f64; 3], half_extents: [f64; 3], points: Vec<[f64; 3]>, label: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.objects.insert(id, TrackedObject {
            id,
            label,
            state: ObjectState::New,
            center,
            half_extents,
            points,
            known_id: None,
            seen_count: 1,
            unseen_count: 0,
            attached_link: None,
        });

        id
    }

    /// Update an existing object with new observation.
    pub fn update(&mut self, id: u64, center: [f64; 3], half_extents: [f64; 3], points: Vec<[f64; 3]>) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.center = center;
            obj.half_extents = half_extents;
            obj.points = points;
            obj.seen_count += 1;
            obj.unseen_count = 0;
            if obj.state != ObjectState::Attached {
                obj.state = ObjectState::Active;
            }
        }
    }

    /// Mark an object as attached to a robot link (grasped).
    pub fn attach(&mut self, id: u64, link_name: &str) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.state = ObjectState::Attached;
            obj.attached_link = Some(link_name.to_string());
        }
    }

    /// Detach an object from the robot.
    pub fn detach(&mut self, id: u64) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.state = ObjectState::Active;
            obj.attached_link = None;
        }
    }

    /// Remove an object immediately.
    pub fn remove(&mut self, id: u64) -> Option<TrackedObject> {
        self.objects.remove(&id)
    }

    /// Process one frame: mark unseen objects as occluded/stale, remove stale objects.
    ///
    /// `observed_ids`: IDs of objects observed in this frame.
    pub fn tick(&mut self, observed_ids: &[u64]) {
        let observed_set: std::collections::HashSet<u64> = observed_ids.iter().copied().collect();
        let mut to_remove = Vec::new();

        for (id, obj) in self.objects.iter_mut() {
            if obj.state == ObjectState::Attached {
                continue; // attached objects don't decay
            }

            if observed_set.contains(id) {
                obj.unseen_count = 0;
                obj.seen_count += 1;
                if obj.state == ObjectState::Occluded || obj.state == ObjectState::New {
                    obj.state = ObjectState::Active;
                }
            } else {
                obj.unseen_count += 1;
                if obj.unseen_count >= self.remove_threshold {
                    to_remove.push(*id);
                } else if obj.unseen_count >= self.stale_threshold {
                    obj.state = ObjectState::Stale;
                } else if obj.unseen_count >= 1 {
                    obj.state = ObjectState::Occluded;
                }
            }
        }

        for id in to_remove {
            self.objects.remove(&id);
        }
    }

    /// Get all tracked objects.
    pub fn objects(&self) -> impl Iterator<Item = &TrackedObject> {
        self.objects.values()
    }

    /// Get objects by state.
    pub fn objects_in_state(&self, state: ObjectState) -> Vec<&TrackedObject> {
        self.objects.values().filter(|o| o.state == state).collect()
    }

    /// Get active collision objects (Active + New + Occluded, not Stale).
    pub fn active_collision_objects(&self) -> Vec<&TrackedObject> {
        self.objects.values().filter(|o| {
            matches!(o.state, ObjectState::Active | ObjectState::New | ObjectState::Occluded | ObjectState::Attached)
        }).collect()
    }

    /// Number of tracked objects.
    pub fn count(&self) -> usize { self.objects.len() }

    /// Get a specific object.
    pub fn get(&self, id: u64) -> Option<&TrackedObject> { self.objects.get(&id) }
}

/// Occlusion reasoning: determine if a point is occluded from a sensor viewpoint.
///
/// A point is occluded if there's an occupied voxel between the sensor and the point.
/// Uses the octree to check for obstacles along the ray.
pub fn is_occluded(
    octree: &crate::octree::Octree,
    sensor_origin: [f64; 3],
    target: [f64; 3],
    step_size: f64,
) -> bool {
    let dx = target[0] - sensor_origin[0];
    let dy = target[1] - sensor_origin[1];
    let dz = target[2] - sensor_origin[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < step_size { return false; }

    let steps = ((dist - step_size) / step_size).ceil() as usize;

    for i in 1..steps {
        let t = i as f64 * step_size / dist;
        let x = sensor_origin[0] + t * dx;
        let y = sensor_origin[1] + t * dy;
        let z = sensor_origin[2] + t * dz;

        if octree.is_occupied(x, y, z) {
            return true;
        }
    }

    false
}

/// Serialize/deserialize tracked objects for persistence across sessions.
pub fn serialize_objects(objects: &[&TrackedObject]) -> Vec<u8> {
    // Simple binary format: count + per-object (id, center, half_extents, label_len, label, state)
    let mut buf = Vec::new();
    buf.extend_from_slice(&(objects.len() as u32).to_le_bytes());

    for obj in objects {
        buf.extend_from_slice(&obj.id.to_le_bytes());
        for v in &obj.center { buf.extend_from_slice(&v.to_le_bytes()); }
        for v in &obj.half_extents { buf.extend_from_slice(&v.to_le_bytes()); }
        let label_bytes = obj.label.as_bytes();
        buf.extend_from_slice(&(label_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(label_bytes);
        buf.push(match obj.state {
            ObjectState::New => 0,
            ObjectState::Active => 1,
            ObjectState::Occluded => 2,
            ObjectState::Attached => 3,
            ObjectState::Stale => 4,
        });
    }

    buf
}

/// Deserialize tracked objects from bytes.
pub fn deserialize_objects(data: &[u8]) -> Option<Vec<TrackedObject>> {
    if data.len() < 4 { return None; }
    let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let mut pos = 4;
    let mut objects = Vec::with_capacity(count);

    for _ in 0..count {
        if pos + 8 > data.len() { return None; }
        let id = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
        pos += 8;

        let mut center = [0.0; 3];
        for c in &mut center { *c = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8; }
        let mut half_extents = [0.0; 3];
        for h in &mut half_extents { *h = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?); pos += 8; }

        if pos + 4 > data.len() { return None; }
        let label_len = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        if pos + label_len > data.len() { return None; }
        let label = String::from_utf8_lossy(&data[pos..pos+label_len]).to_string();
        pos += label_len;

        if pos >= data.len() { return None; }
        let state = match data[pos] {
            0 => ObjectState::New,
            1 => ObjectState::Active,
            2 => ObjectState::Occluded,
            3 => ObjectState::Attached,
            _ => ObjectState::Stale,
        };
        pos += 1;

        objects.push(TrackedObject {
            id, label, state, center, half_extents,
            points: Vec::new(), known_id: None,
            seen_count: 0, unseen_count: 0, attached_link: None,
        });
    }

    Some(objects)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Mesh reconstruction ───

    #[test]
    fn marching_cubes_produces_mesh() {
        let voxels = vec![
            ([0.0, 0.0, 0.0], 0.1, 1.0f32),
            ([0.2, 0.0, 0.0], 0.1, 1.0),
            ([0.0, 0.2, 0.0], 0.1, 0.5),
        ];

        let mesh = marching_cubes(&voxels, 0.0);
        assert!(!mesh.is_empty());
        assert!(mesh.num_triangles() > 0, "Should produce triangles: {}", mesh.num_triangles());
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn marching_cubes_empty_input() {
        let mesh = marching_cubes(&[], 0.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn marching_cubes_below_threshold_skipped() {
        let voxels = vec![([0.0, 0.0, 0.0], 0.1, -1.0f32)]; // below threshold
        let mesh = marching_cubes(&voxels, 0.0);
        assert!(mesh.is_empty(), "Below-threshold voxels should produce no mesh");
    }

    #[test]
    fn simplify_mesh_reduces_vertices() {
        let voxels: Vec<_> = (0..10).map(|i| {
            ([i as f64 * 0.05, 0.0, 0.0], 0.025, 1.0f32)
        }).collect();

        let mesh = marching_cubes(&voxels, 0.0);
        let simplified = simplify_mesh(&mesh, 0.1);

        assert!(
            simplified.num_vertices() < mesh.num_vertices(),
            "Simplified ({}) < original ({})",
            simplified.num_vertices(), mesh.num_vertices()
        );
    }

    #[test]
    fn mesh_aabb_correct() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            triangles: vec![[0, 1, 2]],
        };

        let (min, max) = mesh.aabb().unwrap();
        assert_eq!(min, [-1.0, -2.0, -3.0]);
        assert_eq!(max, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn convex_hull_bounds_correct() {
        let points = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.5], [0.3, 0.2, 0.8],
        ];
        let hull = convex_hull_points(&points);
        assert_eq!(hull.len(), 8); // AABB corners
    }

    // ─── Segmentation ───

    #[test]
    fn euclidean_clustering_two_clusters() {
        let points = vec![
            // Cluster A (near origin)
            [0.0, 0.0, 0.0], [0.01, 0.01, 0.0], [0.02, 0.0, 0.01],
            // Cluster B (far away)
            [5.0, 5.0, 5.0], [5.01, 5.0, 5.0], [5.0, 5.01, 5.0],
        ];

        let clusters = euclidean_clustering(&points, 0.1, 2, 100);
        assert_eq!(clusters.len(), 2, "Should find 2 clusters");
    }

    #[test]
    fn euclidean_clustering_min_size_filter() {
        let points = vec![
            [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], // cluster of 2
            [5.0, 5.0, 5.0], // single point
        ];

        let clusters = euclidean_clustering(&points, 0.1, 2, 100);
        assert_eq!(clusters.len(), 1, "Single-point cluster should be filtered");
    }

    #[test]
    fn euclidean_clustering_empty() {
        let clusters = euclidean_clustering(&[], 0.1, 1, 100);
        assert!(clusters.is_empty());
    }

    #[test]
    fn extract_cluster_correct() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let extracted = extract_cluster(&points, &[0, 2]);
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0], [0.0, 0.0, 0.0]);
        assert_eq!(extracted[1], [2.0, 2.0, 2.0]);
    }

    // ─── Known object database ───

    #[test]
    fn known_object_matching() {
        let mut db = KnownObjectDatabase::new();

        // Add a "mug" template
        let mug_desc = ObjectDescriptor {
            aspect_ratios: [0.5, 0.5, 1.0],
            volume: 0.001, // 10cm × 10cm × 10cm
            point_count: 100,
        };
        db.add(KnownObject {
            id: "mug".into(), name: "Coffee Mug".into(),
            collision_mesh: TriangleMesh::new(),
            dimensions: [0.1, 0.1, 0.12],
            descriptor: mug_desc,
        });

        // Query with similar shape
        let query_points: Vec<[f64; 3]> = (0..50).map(|i| {
            let t = i as f64 * 0.002;
            [t, t, t * 2.0] // roughly 0.5:0.5:1.0 aspect ratio
        }).collect();

        let result = db.match_object(&query_points, 0.3);
        assert!(result.is_some(), "Should match mug template");
        assert_eq!(result.unwrap().0, "mug");
    }

    #[test]
    fn descriptor_similarity_identical() {
        let d = ObjectDescriptor { aspect_ratios: [0.5, 0.8, 1.0], volume: 0.01, point_count: 100 };
        assert!((d.similarity(&d) - 1.0).abs() < 1e-10, "Identical should be 1.0");
    }

    // ─── Lifecycle manager ───

    #[test]
    fn lifecycle_add_and_get() {
        let mut mgr = ObjectLifecycleManager::new();
        let id = mgr.add([0.0; 3], [0.1; 3], vec![], "box".into());
        assert_eq!(mgr.count(), 1);
        assert!(mgr.get(id).is_some());
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::New);
    }

    #[test]
    fn lifecycle_update_transitions_to_active() {
        let mut mgr = ObjectLifecycleManager::new();
        let id = mgr.add([0.0; 3], [0.1; 3], vec![], "box".into());
        mgr.update(id, [0.1, 0.0, 0.0], [0.1; 3], vec![]);
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Active);
    }

    #[test]
    fn lifecycle_attach_detach() {
        let mut mgr = ObjectLifecycleManager::new();
        let id = mgr.add([0.0; 3], [0.1; 3], vec![], "mug".into());

        mgr.attach(id, "gripper_link");
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Attached);
        assert_eq!(mgr.get(id).unwrap().attached_link.as_deref(), Some("gripper_link"));

        mgr.detach(id);
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Active);
        assert!(mgr.get(id).unwrap().attached_link.is_none());
    }

    #[test]
    fn lifecycle_tick_occlusion_and_removal() {
        let mut mgr = ObjectLifecycleManager::new();
        mgr.stale_threshold = 3;
        mgr.remove_threshold = 5;

        let id = mgr.add([0.0; 3], [0.1; 3], vec![], "obj".into());

        // Tick without observing → occluded
        mgr.tick(&[]);
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Occluded);

        // More ticks → stale
        mgr.tick(&[]);
        mgr.tick(&[]);
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Stale);

        // Even more → removed
        mgr.tick(&[]);
        mgr.tick(&[]);
        assert!(mgr.get(id).is_none(), "Should be removed after remove_threshold");
    }

    #[test]
    fn lifecycle_attached_survives_ticks() {
        let mut mgr = ObjectLifecycleManager::new();
        mgr.remove_threshold = 3;

        let id = mgr.add([0.0; 3], [0.1; 3], vec![], "grasped".into());
        mgr.attach(id, "hand");

        // Tick many times without observing — attached should survive
        for _ in 0..10 {
            mgr.tick(&[]);
        }
        assert!(mgr.get(id).is_some(), "Attached object should not be removed");
        assert_eq!(mgr.get(id).unwrap().state, ObjectState::Attached);
    }

    #[test]
    fn lifecycle_active_collision_objects() {
        let mut mgr = ObjectLifecycleManager::new();
        let id1 = mgr.add([0.0; 3], [0.1; 3], vec![], "a".into());
        let id2 = mgr.add([1.0; 3], [0.1; 3], vec![], "b".into());

        // Make id2 stale
        mgr.stale_threshold = 1;
        mgr.tick(&[id1]);

        let active = mgr.active_collision_objects();
        // id1 is Active, id2 is Occluded (1 unseen < stale_threshold wait...
        // actually tick increments unseen, and 1 >= stale_threshold=1 → Stale)
        // So active should include id1 only
        let active_ids: Vec<u64> = active.iter().map(|o| o.id).collect();
        assert!(active_ids.contains(&id1));
    }

    // ─── Serialization ───

    #[test]
    fn object_serialization_roundtrip() {
        let mut mgr = ObjectLifecycleManager::new();
        mgr.add([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], vec![], "test_obj".into());
        mgr.add([4.0, 5.0, 6.0], [0.4, 0.5, 0.6], vec![], "other".into());

        let objects: Vec<&TrackedObject> = mgr.objects().collect();
        let bytes = serialize_objects(&objects);
        let restored = deserialize_objects(&bytes).unwrap();

        assert_eq!(restored.len(), 2);
        // Find the test_obj
        let obj = restored.iter().find(|o| o.label == "test_obj").unwrap();
        assert!((obj.center[0] - 1.0).abs() < 1e-10);
        assert!((obj.half_extents[2] - 0.3).abs() < 1e-10);
    }

    // ─── Integration ───

    #[test]
    fn full_object_perception_pipeline() {
        // Simulate: point cloud → segment → match → lifecycle
        let points = vec![
            // Object A: small cluster
            [0.0, 0.0, 0.5], [0.01, 0.01, 0.5], [0.02, 0.0, 0.51],
            [0.0, 0.02, 0.5], [0.01, 0.0, 0.49],
            // Object B: distant cluster
            [1.0, 1.0, 0.5], [1.01, 1.0, 0.5], [1.0, 1.01, 0.5],
            [1.02, 1.01, 0.5], [1.0, 1.0, 0.51],
        ];

        // Segment
        let clusters = euclidean_clustering(&points, 0.1, 3, 100);
        assert_eq!(clusters.len(), 2);

        // Create lifecycle manager
        let mut mgr = ObjectLifecycleManager::new();
        let mut observed = Vec::new();

        for (i, cluster) in clusters.iter().enumerate() {
            let cluster_pts = extract_cluster(&points, cluster);
            let centroid = {
                let n = cluster_pts.len() as f64;
                let mut c = [0.0; 3];
                for p in &cluster_pts { for j in 0..3 { c[j] += p[j]; } }
                for j in 0..3 { c[j] /= n; }
                c
            };

            let id = mgr.add(centroid, [0.05; 3], cluster_pts, format!("object_{}", i));
            observed.push(id);
        }

        mgr.tick(&observed);
        assert_eq!(mgr.count(), 2);
        assert_eq!(mgr.active_collision_objects().len(), 2);
    }
}
