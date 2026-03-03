#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct AxonInstance {
    packed_start: u32,
    packed_end: u32,
}

struct MaterialUniforms {
    base_color: vec4<f32>,
    clip_plane: vec4<f32>,
    view_mode: u32,
    _padding: vec3<f32>,
};

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var<storage, read> instances: array<AxonInstance>;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) physical_y: f32,
    @location(2) world_position: vec3<f32>,
}

fn unpack_pos(packed: u32) -> vec3<f32> {
    let x = f32(packed & 0x7FFu) * 25.0;
    let y = f32((packed >> 11u) & 0x7FFu) * 25.0;
    let z = f32((packed >> 22u) & 0x3Fu) * 25.0;
    return vec3<f32>(x, y, z);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance = instances[vertex.instance_idx];

    let start = unpack_pos(instance.packed_start);
    let end = unpack_pos(instance.packed_end);
    let dir = end - start;
    let length_val = length(dir);

    if length_val < 0.001 {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.color = material.base_color;
        out.physical_y = 0.0;
        return out;
    }

    let up = dir / length_val;
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), up);
    if length(right) < 0.001 {
        right = cross(vec3<f32>(1.0, 0.0, 0.0), up);
    }
    right = normalize(right);
    let forward = cross(right, up);

    let scaled_pos = vec3<f32>(vertex.position.x * 0.15, vertex.position.y * length_val, vertex.position.z * 0.15);
    let rotated_pos = right * scaled_pos.x + up * scaled_pos.y + forward * scaled_pos.z;
    let center = start + dir * 0.5;
    let final_pos = rotated_pos + center;

    let world_from_local = get_world_from_local(vertex.instance_idx);
    let world_pos = (world_from_local * vec4<f32>(final_pos, 1.0)).xyz;

    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(final_pos, 1.0));
    out.color = material.base_color;
    out.world_position = world_pos;
    out.physical_y = vertex.position.y * length_val;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if dot(in.world_position, material.clip_plane.xyz) > material.clip_plane.w {
        discard;
    }

    if fract(in.physical_y * 0.2) > 0.5 {
        discard;
    }

    return in.color;
}

