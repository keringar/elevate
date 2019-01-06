use gfx_hal::format::Format;
use gfx_hal::pso::{AttributeDesc, Element, VertexBufferDesc};
use nalgebra::{Vector2, Vector3};

#[repr(C)]
pub struct Vertex {
    pos: Vector2<f32>,
    color: Vector3<f32>,
}

impl Vertex {
    pub fn new(pos: (f32, f32), color: (f32, f32, f32)) -> Vertex {
        Vertex {
            pos: Vector2::new(pos.0, pos.1),
            color: Vector3::new(color.0, color.1, color.2),
        }
    }

    pub fn get_buffer_desc() -> Vec<VertexBufferDesc> {
        // Only one binding at index 0 (binding: 0)
        let description = VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            rate: 0, // 1 if we are using instanced rendering
        };

        vec![description]
    }

    pub fn get_attribute_desc() -> Vec<AttributeDesc> {
        let position = AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rg32Float,
                offset: 0,
            },
        };

        let color = AttributeDesc {
            location: 1,
            binding: 0,
            element: Element {
                format: Format::Rgb32Float,
                offset: std::mem::size_of::<Vector2<f32>>() as u32,
            },
        };

        vec![position, color]
    }
}
