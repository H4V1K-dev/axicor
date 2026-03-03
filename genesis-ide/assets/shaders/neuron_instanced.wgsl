#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct NeuronInstance {
    packed_pos: u32,
    emissive: f32,
    selected: u32,
}

struct MaterialUniforms {
    base_color: vec4<f32>,
    clip_plane: vec4<f32>,
    view_mode: u32,
    _padding: vec3<f32>,
};

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var<storage, read> instances: array<NeuronInstance>;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) emissive: f32,
    @location(2) @interpolate(flat) selected: u32,
    @location(3) world_position: vec3<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    let instance = instances[vertex.instance_idx];
    
    // Распаковка PackedPosition (X: 11, Y: 11, Z: 10)
    // Voxel size = 25.0 um
    let pos_x = f32(instance.packed_pos & 0x7FFu) * 25.0;
    let pos_y = f32((instance.packed_pos >> 11u) & 0x7FFu) * 25.0;
    let pos_z = f32((instance.packed_pos >> 22u) & 0x3FFu) * 25.0;
    
    let world_offset = vec3<f32>(pos_x, pos_y, pos_z);
    
    // Сдвигаем базовую геометрию сферы на распакованные координаты
    var final_pos = vertex.position + world_offset;
    let world_from_local = get_world_from_local(vertex.instance_idx);
    let world_pos = (world_from_local * vec4<f32>(final_pos, 1.0)).xyz;

    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(final_pos, 1.0));
    out.world_position = world_pos;
    out.color = material.base_color;
    out.emissive = instance.emissive;
    out.selected = instance.selected;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if dot(in.world_position, material.clip_plane.xyz) > material.clip_plane.w {
        discard;
    }

    var final_color: vec3<f32>;

    if material.view_mode == 0u {
        // РЕЖИМ 0: Solid (Оригинальная логика)
        final_color = mix(in.color.rgb, vec3<f32>(1.0, 1.0, 1.0), in.emissive);
    } else {
        // РЕЖИМ 1: Activity (Heatmap)
        let cold = vec3<f32>(0.02, 0.05, 0.15);
        let warm = vec3<f32>(0.8, 0.1, 0.0);
        let hot = vec3<f32>(1.0, 0.9, 0.2);

        var heat = mix(cold, warm, smoothstep(0.0, 0.4, in.emissive));
        heat = mix(heat, hot, smoothstep(0.4, 1.0, in.emissive));

        final_color = heat;
    }

    // Подсветка выделения (Cyan) поверх любого режима
    if in.selected > 0u {
        final_color = mix(final_color, vec3<f32>(0.2, 0.8, 1.0), 0.6);
    }

    return vec4<f32>(final_color, 1.0);
}
