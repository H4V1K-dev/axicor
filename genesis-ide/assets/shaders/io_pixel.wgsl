#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct IoPixelInstance {
    position: vec3<f32>,
    state: f32,
}

struct MaterialUniforms {
    base_color: vec4<f32>,
};

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var<storage, read> instances: array<IoPixelInstance>;

struct Vertex {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance = instances[vertex.instance_idx];
    
    // Сдвигаем Quad на позицию инстанса
    let final_pos = vertex.position + instance.position;
    
    out.clip_position = mesh_position_local_to_clip(get_world_from_local(vertex.instance_idx), vec4<f32>(final_pos, 1.0));
    
    // Свечение зависит от состояния
    let glow = mix(vec3<f32>(0.1, 0.1, 0.1), material.base_color.rgb * 2.0, instance.state);
    out.color = vec4<f32>(glow, material.base_color.a);
    
    // Прокидываем локальные координаты для рамки
    out.uv = vertex.position.xy;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = in.color.rgb;
    
    // Рисуем тонкую темную рамку вокруг каждого пикселя для эффекта "сетки"
    let border = 0.45; // Размер Quad'а от -0.5 до 0.5
    if abs(in.uv.x) > border || abs(in.uv.y) > border {
        final_color *= 0.2;
    }
    
    return vec4<f32>(final_color, in.color.a);
}

