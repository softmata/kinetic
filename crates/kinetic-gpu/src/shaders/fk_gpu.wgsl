// Forward Kinematics on GPU
//
// Computes FK for N parallel trajectory seeds × T timesteps.
// Each invocation handles one (seed, timestep) pair.
//
// The robot is described by DH-like parameters stored in uniform:
//   - joint_origins: 4x4 homogeneous transform per joint (static URDF origin)
//   - joint_axes: normalized axis per joint (x,y,z)
//   - joint_types: 0=revolute, 1=prismatic, 2=continuous, 3=fixed
//   - num_joints: how many joints
//
// Input: joint_values buffer [num_seeds * timesteps * dof]
// Output: link_spheres buffer - world-frame sphere positions [num_seeds * timesteps * num_spheres * 4]

struct FKParams {
    num_seeds: u32,
    timesteps: u32,
    dof: u32,
    num_spheres: u32,
    num_joints: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Joint origin as 4x4 column-major matrix (static URDF transform)
struct JointOrigin {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    col3: vec4<f32>,
}

// Sphere in local link frame
struct LocalSphere {
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    joint_idx: u32, // which joint's child link owns this sphere
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: FKParams;
@group(0) @binding(1) var<storage, read> joint_origins: array<JointOrigin>;
@group(0) @binding(2) var<storage, read> joint_axes: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> joint_types: array<u32>;
@group(0) @binding(4) var<storage, read> joint_values: array<f32>;
@group(0) @binding(5) var<storage, read> local_spheres: array<LocalSphere>;
@group(0) @binding(6) var<storage, read_write> world_spheres: array<vec4<f32>>;

// Multiply two 4x4 column-major matrices
fn mat4_mul(a_c0: vec4<f32>, a_c1: vec4<f32>, a_c2: vec4<f32>, a_c3: vec4<f32>,
            b_c0: vec4<f32>, b_c1: vec4<f32>, b_c2: vec4<f32>, b_c3: vec4<f32>)
    -> array<vec4<f32>, 4>
{
    var result: array<vec4<f32>, 4>;
    // result column j = A * b_cj
    result[0] = a_c0 * b_c0.x + a_c1 * b_c0.y + a_c2 * b_c0.z + a_c3 * b_c0.w;
    result[1] = a_c0 * b_c1.x + a_c1 * b_c1.y + a_c2 * b_c1.z + a_c3 * b_c1.w;
    result[2] = a_c0 * b_c2.x + a_c1 * b_c2.y + a_c2 * b_c2.z + a_c3 * b_c2.w;
    result[3] = a_c0 * b_c3.x + a_c1 * b_c3.y + a_c2 * b_c3.z + a_c3 * b_c3.w;
    return result;
}

// Build rotation matrix from axis-angle
fn axis_angle_to_mat4(axis: vec3<f32>, angle: f32) -> array<vec4<f32>, 4> {
    let c = cos(angle);
    let s = sin(angle);
    let t = 1.0 - c;
    let x = axis.x;
    let y = axis.y;
    let z = axis.z;

    var result: array<vec4<f32>, 4>;
    result[0] = vec4<f32>(t*x*x + c,   t*x*y + s*z, t*x*z - s*y, 0.0);
    result[1] = vec4<f32>(t*x*y - s*z, t*y*y + c,   t*y*z + s*x, 0.0);
    result[2] = vec4<f32>(t*x*z + s*y, t*y*z - s*x, t*z*z + c,   0.0);
    result[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return result;
}

// Build translation matrix
fn translation_mat4(axis: vec3<f32>, dist: f32) -> array<vec4<f32>, 4> {
    var result: array<vec4<f32>, 4>;
    result[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    result[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    result[2] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    result[3] = vec4<f32>(axis * dist, 1.0);
    return result;
}

// Identity matrix
fn identity_mat4() -> array<vec4<f32>, 4> {
    var result: array<vec4<f32>, 4>;
    result[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    result[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    result[2] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    result[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return result;
}

// Transform a point by 4x4 matrix
fn transform_point(m: array<vec4<f32>, 4>, p: vec3<f32>) -> vec3<f32> {
    let h = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
    return h.xyz;
}

@compute @workgroup_size(64)
fn fk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sphere_idx = gid.x;
    if (sphere_idx >= params.num_seeds * params.timesteps * params.num_spheres) {
        return;
    }

    // Decode which seed, timestep, and sphere
    let total_per_seed = params.timesteps * params.num_spheres;
    let seed = sphere_idx / total_per_seed;
    let remainder = sphere_idx % total_per_seed;
    let timestep = remainder / params.num_spheres;
    let local_sphere_idx = remainder % params.num_spheres;

    // Get the local sphere data
    let ls = local_spheres[local_sphere_idx];
    let target_joint = ls.joint_idx;

    // Base offset into joint_values for this seed/timestep
    let jv_base = seed * params.timesteps * params.dof + timestep * params.dof;

    // Chain FK: multiply transforms from joint 0 to target_joint
    var accum = identity_mat4();
    var active_idx: u32 = 0u;

    for (var j: u32 = 0u; j <= target_joint && j < params.num_joints; j = j + 1u) {
        // Apply static joint origin
        let origin = joint_origins[j];
        accum = mat4_mul(accum[0], accum[1], accum[2], accum[3],
                         origin.col0, origin.col1, origin.col2, origin.col3);

        let jtype = joint_types[j];
        if (jtype == 0u || jtype == 2u) {
            // Revolute or continuous: rotate around axis
            let axis = joint_axes[j].xyz;
            let angle = joint_values[jv_base + active_idx];
            let rot = axis_angle_to_mat4(axis, angle);
            accum = mat4_mul(accum[0], accum[1], accum[2], accum[3],
                             rot[0], rot[1], rot[2], rot[3]);
            active_idx = active_idx + 1u;
        } else if (jtype == 1u) {
            // Prismatic: translate along axis
            let axis = joint_axes[j].xyz;
            let dist = joint_values[jv_base + active_idx];
            let trans = translation_mat4(axis, dist);
            accum = mat4_mul(accum[0], accum[1], accum[2], accum[3],
                             trans[0], trans[1], trans[2], trans[3]);
            active_idx = active_idx + 1u;
        }
        // Fixed joints: no motion transform, no joint value consumed
    }

    // Transform local sphere to world frame
    let world_pos = transform_point(accum, vec3<f32>(ls.x, ls.y, ls.z));

    // Write world sphere position + radius
    let out_idx = sphere_idx;
    world_spheres[out_idx] = vec4<f32>(world_pos, ls.radius);
}
