// SDF (Signed Distance Field) builder on GPU
//
// Builds a 3D voxel grid where each cell stores the minimum signed
// distance to the nearest obstacle sphere. Negative = inside obstacle.
//
// Input: obstacle spheres (x, y, z, radius)
// Output: SDF grid (f32 per voxel)

struct SDFParams {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    resolution: f32,
    nx: u32,
    ny: u32,
    nz: u32,
    num_spheres: u32,
}

@group(0) @binding(0) var<uniform> params: SDFParams;
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>; // xyz + radius
@group(0) @binding(2) var<storage, read_write> sdf_grid: array<f32>;

@compute @workgroup_size(64)
fn sdf_build_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_idx = gid.x;
    let total_voxels = params.nx * params.ny * params.nz;
    if (voxel_idx >= total_voxels) {
        return;
    }

    // Decode voxel coordinates
    let iz = voxel_idx / (params.nx * params.ny);
    let rem = voxel_idx % (params.nx * params.ny);
    let iy = rem / params.nx;
    let ix = rem % params.nx;

    // Voxel center in world coordinates
    let vx = params.min_x + (f32(ix) + 0.5) * params.resolution;
    let vy = params.min_y + (f32(iy) + 0.5) * params.resolution;
    let vz = params.min_z + (f32(iz) + 0.5) * params.resolution;

    // Find minimum signed distance to any sphere
    var min_dist: f32 = 1e10;

    for (var i: u32 = 0u; i < params.num_spheres; i = i + 1u) {
        let s = spheres[i];
        let dx = vx - s.x;
        let dy = vy - s.y;
        let dz = vz - s.z;
        let center_dist = sqrt(dx * dx + dy * dy + dz * dz);
        let signed_dist = center_dist - s.w; // radius = s.w

        min_dist = min(min_dist, signed_dist);
    }

    sdf_grid[voxel_idx] = min_dist;
}
