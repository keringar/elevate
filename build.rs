use shaderc::ShaderKind;

use std::fs::File;
use std::io::prelude::*;

pub fn main() {
    let out_dir = std::path::PathBuf::from(&std::env::var("OUT_DIR").unwrap());

    let mut compiler = shaderc::Compiler::new().unwrap();

    let vertex_shader = compiler
        .compile_into_spirv(
            include_str!("data/triangle.vert"),
            ShaderKind::Vertex,
            "triangle.vert",
            "main",
            None,
        )
        .unwrap();

    let mut buffer = File::create(out_dir.join("triangle_vert.spirv")).unwrap();
    buffer.write(&vertex_shader.as_binary_u8()).unwrap();

    let fragment_shader = compiler
        .compile_into_spirv(
            include_str!("data/triangle.frag"),
            ShaderKind::Fragment,
            "triangle.frag",
            "main",
            None,
        )
        .unwrap();

    let mut buffer = File::create(out_dir.join("triangle_frag.spirv")).unwrap();
    buffer.write(&fragment_shader.as_binary_u8()).unwrap();
}
