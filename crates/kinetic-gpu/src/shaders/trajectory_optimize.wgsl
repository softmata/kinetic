// Trajectory Optimization Kernel
//
// Each workgroup processes one trajectory seed.
// Computes cost (collision + smoothness + goal + joint limit) and gradient,
// then applies gradient descent step.
//
// The kernel is called iteratively from CPU side.

struct OptParams {
    num_seeds: u32,
    timesteps: u32,
    dof: u32,
    num_spheres: u32,
    collision_weight: f32,
    smoothness_weight: f32,
    goal_weight: f32,
    step_size: f32,
    // SDF params
    sdf_min_x: f32,
    sdf_min_y: f32,
    sdf_min_z: f32,
    sdf_resolution: f32,
    sdf_nx: u32,
    sdf_ny: u32,
    sdf_nz: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: OptParams;
@group(0) @binding(1) var<storage, read_write> trajectories: array<f32>;
  // [num_seeds * timesteps * dof] - joint values
@group(0) @binding(2) var<storage, read> world_spheres: array<vec4<f32>>;
  // [num_seeds * timesteps * num_spheres] - from FK pass
@group(0) @binding(3) var<storage, read> sdf_grid: array<f32>;
@group(0) @binding(4) var<storage, read> goal_joints: array<f32>; // [dof]
@group(0) @binding(5) var<storage, read> joint_lower: array<f32>; // [dof]
@group(0) @binding(6) var<storage, read> joint_upper: array<f32>; // [dof]
@group(0) @binding(7) var<storage, read_write> costs: array<f32>; // [num_seeds]

// Query SDF at a world position (nearest-neighbor for speed in inner loop)
fn query_sdf(pos: vec3<f32>) -> f32 {
    let inv_res = 1.0 / params.sdf_resolution;
    let fx = (pos.x - params.sdf_min_x) * inv_res;
    let fy = (pos.y - params.sdf_min_y) * inv_res;
    let fz = (pos.z - params.sdf_min_z) * inv_res;

    let ix = i32(fx);
    let iy = i32(fy);
    let iz = i32(fz);

    if (ix < 0 || iy < 0 || iz < 0 ||
        u32(ix) >= params.sdf_nx || u32(iy) >= params.sdf_ny || u32(iz) >= params.sdf_nz) {
        return 1e10; // Outside SDF bounds = no collision
    }

    let idx = u32(iz) * params.sdf_nx * params.sdf_ny + u32(iy) * params.sdf_nx + u32(ix);
    return sdf_grid[idx];
}

// Compute SDF gradient via central finite differences
fn sdf_gradient(pos: vec3<f32>) -> vec3<f32> {
    let h = params.sdf_resolution * 0.5;
    let dx = query_sdf(pos + vec3<f32>(h, 0.0, 0.0)) - query_sdf(pos - vec3<f32>(h, 0.0, 0.0));
    let dy = query_sdf(pos + vec3<f32>(0.0, h, 0.0)) - query_sdf(pos - vec3<f32>(0.0, h, 0.0));
    let dz = query_sdf(pos + vec3<f32>(0.0, 0.0, h)) - query_sdf(pos - vec3<f32>(0.0, 0.0, h));
    let inv_2h = 1.0 / (2.0 * h);
    return vec3<f32>(dx, dy, dz) * inv_2h;
}

// Hinge loss: max(0, margin - sdf_value)
fn collision_cost_at(pos: vec3<f32>, radius: f32) -> f32 {
    let sdf_val = query_sdf(pos);
    let penetration = radius - sdf_val; // positive = in collision
    return max(0.0, penetration);
}

// Joint limit penalty: quadratic cost when approaching limits
fn joint_limit_cost(q: f32, lower: f32, upper: f32) -> f32 {
    let margin = (upper - lower) * 0.05; // 5% of range as soft margin
    var cost: f32 = 0.0;
    if (q < lower + margin) {
        let d = lower + margin - q;
        cost = d * d;
    }
    if (q > upper - margin) {
        let d = q - (upper - margin);
        cost = d * d;
    }
    return cost;
}

@compute @workgroup_size(1)
fn optimize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = gid.x;
    if (seed >= params.num_seeds) {
        return;
    }

    let traj_base = seed * params.timesteps * params.dof;
    let sphere_base = seed * params.timesteps * params.num_spheres;

    // === Compute cost for this trajectory ===
    var total_cost: f32 = 0.0;

    // 1. Collision cost: sum of hinge losses at each sphere position
    var collision_cost: f32 = 0.0;
    for (var t: u32 = 0u; t < params.timesteps; t = t + 1u) {
        for (var s: u32 = 0u; s < params.num_spheres; s = s + 1u) {
            let ws = world_spheres[sphere_base + t * params.num_spheres + s];
            collision_cost += collision_cost_at(ws.xyz, ws.w);
        }
    }
    total_cost += params.collision_weight * collision_cost;

    // 2. Smoothness cost: finite-difference jerk (third derivative)
    var smooth_cost: f32 = 0.0;
    if (params.timesteps >= 4u) {
        for (var j: u32 = 0u; j < params.dof; j = j + 1u) {
            for (var t: u32 = 0u; t < params.timesteps - 3u; t = t + 1u) {
                let q0 = trajectories[traj_base + t * params.dof + j];
                let q1 = trajectories[traj_base + (t + 1u) * params.dof + j];
                let q2 = trajectories[traj_base + (t + 2u) * params.dof + j];
                let q3 = trajectories[traj_base + (t + 3u) * params.dof + j];
                let jerk = q3 - 3.0 * q2 + 3.0 * q1 - q0;
                smooth_cost += jerk * jerk;
            }
        }
    }
    total_cost += params.smoothness_weight * smooth_cost;

    // 3. Goal cost: L2 distance of final timestep to goal
    var goal_cost: f32 = 0.0;
    let last_t = params.timesteps - 1u;
    for (var j: u32 = 0u; j < params.dof; j = j + 1u) {
        let q = trajectories[traj_base + last_t * params.dof + j];
        let g = goal_joints[j];
        let diff = q - g;
        goal_cost += diff * diff;
    }
    total_cost += params.goal_weight * goal_cost;

    // 4. Joint limit cost
    var limit_cost: f32 = 0.0;
    for (var t: u32 = 0u; t < params.timesteps; t = t + 1u) {
        for (var j: u32 = 0u; j < params.dof; j = j + 1u) {
            let q = trajectories[traj_base + t * params.dof + j];
            limit_cost += joint_limit_cost(q, joint_lower[j], joint_upper[j]);
        }
    }
    total_cost += 10.0 * limit_cost;

    // Store total cost for this seed
    costs[seed] = total_cost;

    // === Gradient descent on joint values ===
    let eps: f32 = 0.001;

    for (var t: u32 = 1u; t < params.timesteps - 1u; t = t + 1u) {
        for (var j: u32 = 0u; j < params.dof; j = j + 1u) {
            let idx = traj_base + t * params.dof + j;
            let q_curr = trajectories[idx];

            // --- Smoothness gradient (analytical from jerk cost) ---
            var smooth_grad: f32 = 0.0;

            // The jerk cost involves terms: (q[t+3] - 3*q[t+2] + 3*q[t+1] - q[t])^2
            // Gradient of jerk^2 w.r.t. q[t] depends on which jerk terms include q[t].
            // For a given joint j at timestep t, it appears in up to 4 jerk terms:
            //   jerk at t-3: coefficient +1 (q[t] is the q3 term)
            //   jerk at t-2: coefficient -3 (q[t] is the q2 term)
            //   jerk at t-1: coefficient +3 (q[t] is the q1 term)
            //   jerk at t:   coefficient -1 (q[t] is the q0 term)

            if (params.timesteps >= 4u) {
                // jerk starting at t-3 (if valid): d/dq[t] = 2 * jerk * 1
                if (t >= 3u) {
                    let jk = trajectories[idx] - 3.0 * trajectories[idx - params.dof]
                           + 3.0 * trajectories[idx - 2u * params.dof] - trajectories[idx - 3u * params.dof];
                    smooth_grad += 2.0 * jk;
                }
                // jerk starting at t-2 (if valid): d/dq[t] = 2 * jerk * (-3)
                if (t >= 2u && t + 1u < params.timesteps) {
                    let jk = trajectories[idx + params.dof] - 3.0 * trajectories[idx]
                           + 3.0 * trajectories[idx - params.dof] - trajectories[idx - 2u * params.dof];
                    smooth_grad += 2.0 * jk * (-3.0);
                }
                // jerk starting at t-1 (if valid): d/dq[t] = 2 * jerk * 3
                if (t >= 1u && t + 2u < params.timesteps) {
                    let jk = trajectories[idx + 2u * params.dof] - 3.0 * trajectories[idx + params.dof]
                           + 3.0 * trajectories[idx] - trajectories[idx - params.dof];
                    smooth_grad += 2.0 * jk * 3.0;
                }
                // jerk starting at t (if valid): d/dq[t] = 2 * jerk * (-1)
                if (t + 3u < params.timesteps) {
                    let jk = trajectories[idx + 3u * params.dof] - 3.0 * trajectories[idx + 2u * params.dof]
                           + 3.0 * trajectories[idx + params.dof] - trajectories[idx];
                    smooth_grad += 2.0 * jk * (-1.0);
                }
            }
            smooth_grad *= params.smoothness_weight;

            // --- Goal gradient (propagate goal error to nearby timesteps) ---
            var goal_grad: f32 = 0.0;
            // Direct goal gradient only on last timestep, but we propagate
            // a small attraction to the second-to-last timestep
            if (t == params.timesteps - 2u) {
                let q_last = trajectories[traj_base + last_t * params.dof + j];
                goal_grad = 2.0 * params.goal_weight * (q_last - goal_joints[j]) * 0.1;
            }

            // --- Collision gradient via SDF gradient ---
            var coll_grad: f32 = 0.0;
            for (var s: u32 = 0u; s < params.num_spheres; s = s + 1u) {
                let ws = world_spheres[sphere_base + t * params.num_spheres + s];
                let c = collision_cost_at(ws.xyz, ws.w);
                if (c > 0.0) {
                    // Use SDF gradient to push sphere out of collision
                    // The gradient points toward increasing distance (away from obstacles)
                    let grad = sdf_gradient(ws.xyz);
                    let grad_mag = length(grad);
                    if (grad_mag > 1e-6) {
                        // Project SDF gradient onto joint j's effect
                        // Simplified: use the gradient magnitude as a push signal
                        // This is an approximation — full Jacobian-transpose would be better
                        // but too expensive per-joint per-sphere on GPU
                        coll_grad -= params.collision_weight * c * 0.1;
                    } else {
                        // No gradient info — use small random push
                        coll_grad += params.collision_weight * eps;
                    }
                }
            }

            // --- Joint limit gradient ---
            var limit_grad: f32 = 0.0;
            let margin = (joint_upper[j] - joint_lower[j]) * 0.05;
            if (q_curr < joint_lower[j] + margin) {
                limit_grad = -2.0 * 10.0 * (joint_lower[j] + margin - q_curr);
            }
            if (q_curr > joint_upper[j] - margin) {
                limit_grad = 2.0 * 10.0 * (q_curr - (joint_upper[j] - margin));
            }

            let grad = smooth_grad + goal_grad + coll_grad + limit_grad;
            var new_q = q_curr - params.step_size * grad;

            // Hard clamp to joint limits
            new_q = clamp(new_q, joint_lower[j], joint_upper[j]);

            trajectories[idx] = new_q;
        }
    }
}
