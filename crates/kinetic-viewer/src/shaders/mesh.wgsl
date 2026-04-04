// Mesh shader: Blinn-Phong lighting with per-instance transform and color.
//
// Vertex attributes: position (vec3), normal (vec3)
// Instance attributes: model matrix (4x vec4), color (vec4)
// Uniforms: view-projection, view matrix, camera position, lights

struct ViewUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct LightUniforms {
    ambient_color: vec4<f32>,      // rgb + intensity
    directional_dir: vec4<f32>,    // xyz + intensity
    directional_color: vec4<f32>,  // rgb + unused
};

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    let world_pos = model * vec4<f32>(vertex.position, 1.0);

    // Normal matrix: upper-left 3x3 of model (assumes uniform scale)
    let normal_mat = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );
    let world_normal = normalize(normal_mat * vertex.normal);

    var out: VertexOutput;
    out.clip_position = view.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = world_normal;
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(view.camera_pos.xyz - in.world_position);

    // Ambient
    let ambient = light.ambient_color.rgb * light.ambient_color.a;

    // Directional diffuse (Lambertian)
    let L = normalize(-light.directional_dir.xyz);
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = light.directional_color.rgb * light.directional_dir.w * NdotL;

    // Blinn-Phong specular
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);
    let specular_power = 32.0;
    let spec = pow(NdotH, specular_power) * light.directional_dir.w * 0.3;
    let specular = light.directional_color.rgb * spec;

    let lighting = ambient + diffuse + specular;
    let final_color = in.color.rgb * lighting;

    return vec4<f32>(final_color, in.color.a);
}
