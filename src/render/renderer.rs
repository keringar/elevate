use back;
use gfx_hal::{
    buffer, command, format, image, pso, queue, window, Adapter, Backbuffer, Backend, Capability,
    CommandQueue, Device, Graphics, Instance, PhysicalDevice, QueueFamily, Surface, Swapchain,
    SwapchainConfig,
};

use super::util::*;
use super::vertex::Vertex;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

lazy_static! {
    static ref vertices: Vec<Vertex> = vec![
        Vertex::new((0.0, -0.5), (1.0, 1.0, 1.0)),
        Vertex::new((0.5, 0.5), (0.0, 1.0, 0.0)),
        Vertex::new((-0.5, 0.5), (0.0, 0.0, 1.0)),
    ];
}

pub struct Renderer {
    current_frame: usize,
    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    submission_command_buffers:
        Vec<command::CommandBuffer<back::Backend, Graphics, command::MultiShot, command::Primary>>,
    vertex_buffer_memory: <back::Backend as Backend>::Memory,
    vertex_buffer: <back::Backend as Backend>::Buffer,
    command_pool: gfx_hal::pool::CommandPool<back::Backend, Graphics>,
    swapchain_framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    gfx_pipeline: <back::Backend as Backend>::GraphicsPipeline,
    descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pipeline_layout: <back::Backend as Backend>::PipelineLayout,
    render_pass: <back::Backend as Backend>::RenderPass,
    swapchain_images: Vec<(
        <back::Backend as Backend>::Image,
        <back::Backend as Backend>::ImageView,
    )>,
    swapchain: <back::Backend as Backend>::Swapchain,
    command_queues: Vec<gfx_hal::CommandQueue<back::Backend, Graphics>>,
    device: <back::Backend as gfx_hal::Backend>::Device,
    surface: <back::Backend as gfx_hal::Backend>::Surface,
    adapter: gfx_hal::Adapter<back::Backend>,
}

impl Renderer {
    pub fn new(window: &winit::Window) -> Renderer {
        let instance = gfx_backend_vulkan::Instance::create("elevate", 1);
        let mut adapter = instance.enumerate_adapters().into_iter().next().unwrap();
        let mut surface = instance.create_surface(window);

        print_adapter_info(&adapter);

        let (device, command_queues, queue_type, qf_id) =
            open_compatible_device(&mut adapter, &surface);

        let (swapchain, extent, backbuffer, format) =
            create_swapchain(&adapter, &device, &mut surface, None);

        let swapchain_images = create_image_views(backbuffer, format, &device);

        let render_pass = create_render_pass(&device, Some(format));

        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) =
            create_graphics_pipeline(&device, extent, &render_pass);

        let swapchain_framebuffers =
            create_framebuffers(&device, &render_pass, &swapchain_images, extent);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_objects(&device);

        let mut command_pool = create_command_pool(&device, queue_type, qf_id);

        let (mut vertex_buffer, requirements) = create_vertex_buffer(&device);

        // copy vertices into vertex buffer
        let vertex_buffer_memory = unsafe {
            let memory = device
                .allocate_memory(get_memory_heap(&adapter, requirements), requirements.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut vertex_buffer)
                .unwrap();
            let vertex_buffer_ptr = device.map_memory(&memory, 0..requirements.size).unwrap();
            let vertices_ptr = vertices.as_ptr() as *const u8;
            let vertices_len = vertices.len() * std::mem::size_of::<Vertex>();
            std::ptr::copy_nonoverlapping(vertices_ptr, vertex_buffer_ptr, vertices_len);
            device.unmap_memory(&memory);

            memory
        };

        let submission_command_buffers = create_command_buffers(
            &mut command_pool,
            &render_pass,
            &swapchain_framebuffers,
            extent,
            &gfx_pipeline,
            &vertex_buffer,
        );

        Renderer {
            current_frame: 0,
            in_flight_fences,
            render_finished_semaphores,
            image_available_semaphores,
            submission_command_buffers,
            vertex_buffer_memory,
            vertex_buffer,
            command_pool,
            swapchain_framebuffers,
            gfx_pipeline,
            descriptor_set_layouts,
            pipeline_layout,
            render_pass,
            swapchain_images,
            swapchain,
            command_queues,
            device,
            surface,
            adapter,
        }
    }

    pub fn draw_frame(&mut self) {
        unsafe {
            let image_available_semaphore = &self.image_available_semaphores[self.current_frame];
            let render_finished_semaphore = &self.render_finished_semaphores[self.current_frame];
            let in_flight_fence = &self.in_flight_fences[self.current_frame];

            self.device
                .wait_for_fence(in_flight_fence, std::u64::MAX)
                .unwrap();

            let image_index = match self.swapchain.acquire_image(
                std::u64::MAX,
                window::FrameSync::Semaphore(image_available_semaphore),
            ) {
                Ok(index) => index,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            };

            let i = image_index as usize;
            let submission = queue::Submission {
                command_buffers: &self.submission_command_buffers[i..i + 1],
                wait_semaphores: vec![(
                    image_available_semaphore,
                    pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                )],
                signal_semaphores: vec![render_finished_semaphore],
            };

            self.device.reset_fence(in_flight_fence).unwrap();

            self.command_queues[0].submit(submission, Some(in_flight_fence));

            self.swapchain
                .present(
                    &mut self.command_queues[0],
                    image_index,
                    vec![render_finished_semaphore],
                )
                .expect("Unable to present image to swapchain");

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        // Wait for resources to be idle
        self.device.wait_idle().unwrap();

        // Take out old resources
        let old_swapchain = std::mem::replace(&mut self.swapchain, std::mem::uninitialized());
        let old_swapchain_images = std::mem::replace(&mut self.swapchain_images, Vec::new());
        let old_render_pass = std::mem::replace(&mut self.render_pass, std::mem::uninitialized());
        let old_pipeline_layout =
            std::mem::replace(&mut self.pipeline_layout, std::mem::uninitialized());
        let old_descriptor_set_layouts =
            std::mem::replace(&mut self.descriptor_set_layouts, Vec::new());
        let old_gfx_pipeline = std::mem::replace(&mut self.gfx_pipeline, std::mem::uninitialized());
        let old_swapchain_framebuffers =
            std::mem::replace(&mut self.swapchain_framebuffers, Vec::new());
        let _old_submission_command_buffers =
            std::mem::replace(&mut self.submission_command_buffers, Vec::new());

        // Begin destroying them all
        self.command_pool.reset();
        for framebuffer in old_swapchain_framebuffers {
            self.device.destroy_framebuffer(framebuffer);
        }
        self.device.destroy_graphics_pipeline(old_gfx_pipeline);
        for dsl in old_descriptor_set_layouts {
            self.device.destroy_descriptor_set_layout(dsl);
        }
        self.device.destroy_pipeline_layout(old_pipeline_layout);
        self.device.destroy_render_pass(old_render_pass);
        for (_, image_view) in old_swapchain_images.into_iter() {
            self.device.destroy_image_view(image_view);
        }
        self.device.destroy_swapchain(old_swapchain);

        // Create replacement resources
        let (swapchain, extent, backbuffer, format) =
            create_swapchain(&self.adapter, &self.device, &mut self.surface, None);
        let swapchain_images = create_image_views(backbuffer, format, &self.device);
        let render_pass = create_render_pass(&self.device, Some(format));
        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) =
            create_graphics_pipeline(&self.device, extent, &render_pass);
        let swapchain_framebuffers =
            create_framebuffers(&self.device, &render_pass, &swapchain_images, extent);
        let submission_command_buffers = create_command_buffers(
            &mut self.command_pool,
            &render_pass,
            &swapchain_framebuffers,
            extent,
            &gfx_pipeline,
            &self.vertex_buffer,
        );

        // Replace uninitialized values
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.render_pass = render_pass;
        self.pipeline_layout = pipeline_layout;
        self.descriptor_set_layouts = descriptor_set_layouts;
        self.gfx_pipeline = gfx_pipeline;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.submission_command_buffers = submission_command_buffers;
    }

    pub fn cleanup(self) {
        unsafe {
            self.device.wait_idle().unwrap();

            self.device.destroy_buffer(self.vertex_buffer);

            self.device.free_memory(self.vertex_buffer_memory);

            for fence in self.in_flight_fences {
                self.device.destroy_fence(fence);
            }

            for semaphore in self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore);
            }

            for semaphore in self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore);
            }

            self.device
                .destroy_command_pool(self.command_pool.into_raw());

            for framebuffer in self.swapchain_framebuffers {
                self.device.destroy_framebuffer(framebuffer);
            }

            self.device.destroy_graphics_pipeline(self.gfx_pipeline);

            for dsl in self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(dsl);
            }

            self.device.destroy_pipeline_layout(self.pipeline_layout);

            self.device.destroy_render_pass(self.render_pass);

            for (_, image_view) in self.swapchain_images.into_iter() {
                self.device.destroy_image_view(image_view);
            }

            self.device.destroy_swapchain(self.swapchain);
        }
    }
}
fn open_compatible_device(
    adapter: &mut Adapter<back::Backend>,
    surface: &<back::Backend as Backend>::Surface,
) -> (
    <back::Backend as Backend>::Device,
    Vec<CommandQueue<back::Backend, Graphics>>,
    gfx_hal::queue::QueueType,
    gfx_hal::queue::family::QueueFamilyId,
) {
    let family = adapter
        .queue_families
        .iter()
        .find(|family| {
            Graphics::supported_by(family.queue_type())
                && family.max_queues() > 0
                && surface.supports_queue_family(family)
        })
        .expect("Could not find a queue family supporting graphics.");

    let priorities = vec![1.0; 1];
    let families = [(family, priorities.as_slice())];

    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&families)
            .expect("Could not create device.")
    };

    let mut queue_group = gpu
        .queues
        .take::<Graphics>(family.id())
        .expect("Could not take ownership of relevant queue group.");

    let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

    (gpu.device, command_queues, family.queue_type(), family.id())
}

fn create_swapchain(
    adapter: &Adapter<back::Backend>,
    device: &<back::Backend as Backend>::Device,
    surface: &mut <back::Backend as Backend>::Surface,
    previous_swapchain: Option<<back::Backend as Backend>::Swapchain>,
) -> (
    <back::Backend as Backend>::Swapchain,
    gfx_hal::window::Extent2D,
    Backbuffer<back::Backend>,
    gfx_hal::format::Format,
) {
    let (capabilities, formats, _present_modes, _alphas) =
        surface.compatibility(&adapter.physical_device);

    // Choose Rgba8Srgb if no preferred format is found, otherwise choose the
    // first format that supports SRGB, otherwise just choose the first available
    let format = formats.map_or(gfx_hal::format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == format::ChannelType::Srgb)
            .map(|format| *format)
            .unwrap_or(formats[0])
    });

    let swap_config = SwapchainConfig::from_caps(&capabilities, format, capabilities.extents.end)
        .with_mode(gfx_hal::window::PresentMode::Fifo)
        .with_image_usage(gfx_hal::image::Usage::COLOR_ATTACHMENT);

    let extent = swap_config.extent;

    let (swapchain, backbuffer) = unsafe {
        device
            .create_swapchain(surface, swap_config, previous_swapchain)
            .unwrap()
    };

    (swapchain, extent, backbuffer, format)
}

fn create_image_views(
    backbuffer: Backbuffer<back::Backend>,
    format: gfx_hal::format::Format,
    device: &<back::Backend as Backend>::Device,
) -> Vec<(
    <back::Backend as Backend>::Image,
    <back::Backend as Backend>::ImageView,
)> {
    match backbuffer {
        Backbuffer::Images(images) => images
            .into_iter()
            .map(|image| unsafe {
                let image_view = device
                    .create_image_view(
                        &image,
                        image::ViewKind::D2,
                        format,
                        format::Swizzle::NO,
                        image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    )
                    .expect("Unable to create image view for an image");

                (image, image_view)
            })
            .collect(),
        _ => unimplemented!(),
    }
}

fn create_render_pass(
    device: &<back::Backend as Backend>::Device,
    format: Option<format::Format>,
) -> <back::Backend as Backend>::RenderPass {
    let samples: u8 = 1;

    let ops = gfx_hal::pass::AttachmentOps {
        load: gfx_hal::pass::AttachmentLoadOp::Clear,
        store: gfx_hal::pass::AttachmentStoreOp::Store,
    };

    let stencil_ops = gfx_hal::pass::AttachmentOps::DONT_CARE;

    let layouts = gfx_hal::image::Layout::Undefined..gfx_hal::image::Layout::Present;

    let color_attachment = gfx_hal::pass::Attachment {
        format,
        samples,
        ops,
        stencil_ops,
        layouts,
    };

    let color_attachment_ref: gfx_hal::pass::AttachmentRef =
        (0, gfx_hal::image::Layout::ColorAttachmentOptimal);

    let subpass = gfx_hal::pass::SubpassDesc {
        colors: &[color_attachment_ref],
        depth_stencil: None,
        inputs: &[],
        resolves: &[],
        preserves: &[],
    };

    unsafe {
        device
            .create_render_pass(&[color_attachment], &[subpass], &[])
            .unwrap()
    }
}

fn create_graphics_pipeline(
    device: &<back::Backend as Backend>::Device,
    extent: gfx_hal::window::Extent2D,
    render_pass: &<back::Backend as Backend>::RenderPass,
) -> (
    Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    <back::Backend as Backend>::PipelineLayout,
    <back::Backend as Backend>::GraphicsPipeline,
) {
    let triangle_vert = include_bytes!(concat!(env!("OUT_DIR"), "/triangle_vert.spirv"));
    let triangle_frag = include_bytes!(concat!(env!("OUT_DIR"), "/triangle_frag.spirv"));
    unsafe {
        let vert_shader_module = device.create_shader_module(triangle_vert).unwrap();
        let frag_shader_module = device.create_shader_module(triangle_frag).unwrap();

        let vs_entry = gfx_hal::pso::EntryPoint::<back::Backend> {
            entry: "main",
            module: &vert_shader_module,
            specialization: gfx_hal::pso::Specialization {
                constants: &[],
                data: &[],
            },
        };

        let fs_entry = gfx_hal::pso::EntryPoint::<back::Backend> {
            entry: "main",
            module: &frag_shader_module,
            specialization: gfx_hal::pso::Specialization {
                constants: &[],
                data: &[],
            },
        };

        let shaders = gfx_hal::pso::GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let rasterizer = gfx_hal::pso::Rasterizer {
            polygon_mode: gfx_hal::pso::PolygonMode::Fill,
            cull_face: gfx_hal::pso::Face::BACK,
            front_face: gfx_hal::pso::FrontFace::Clockwise,
            depth_clamping: false,
            depth_bias: None,
            conservative: false,
        };

        let vertex_buffers = Vertex::get_buffer_desc();
        let attributes = Vertex::get_attribute_desc();

        let input_assembler = gfx_hal::pso::InputAssemblerDesc {
            primitive: gfx_hal::Primitive::TriangleList,
            primitive_restart: gfx_hal::pso::PrimitiveRestart::Disabled,
        };

        let blender = {
            let blend_state = gfx_hal::pso::BlendState::On {
                color: gfx_hal::pso::BlendOp::Add {
                    src: gfx_hal::pso::Factor::One,
                    dst: gfx_hal::pso::Factor::Zero,
                },
                alpha: gfx_hal::pso::BlendOp::Add {
                    src: gfx_hal::pso::Factor::One,
                    dst: gfx_hal::pso::Factor::Zero,
                },
            };

            gfx_hal::pso::BlendDesc {
                logic_op: Some(gfx_hal::pso::LogicOp::Copy),
                targets: vec![gfx_hal::pso::ColorBlendDesc(
                    gfx_hal::pso::ColorMask::ALL,
                    blend_state,
                )],
            }
        };

        let depth_stencil = gfx_hal::pso::DepthStencilDesc {
            depth: gfx_hal::pso::DepthTest::Off,
            depth_bounds: false,
            stencil: gfx_hal::pso::StencilTest::Off,
        };

        let multisampling: Option<gfx_hal::pso::Multisampling> = None;

        let baked_states = gfx_hal::pso::BakedStates {
            viewport: Some(gfx_hal::pso::Viewport {
                rect: gfx_hal::pso::Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as i16,
                    h: extent.height as i16,
                },
                depth: (0.0..1.0),
            }),
            scissor: Some(gfx_hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as i16,
                h: extent.height as i16,
            }),
            blend_color: None,
            depth_bounds: None,
        };

        let bindings = Vec::<gfx_hal::pso::DescriptorSetLayoutBinding>::new();
        let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
        let ds_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> = vec![device
            .create_descriptor_set_layout(bindings, immutable_samplers)
            .unwrap()];
        let push_constants = Vec::<(gfx_hal::pso::ShaderStageFlags, std::ops::Range<u32>)>::new();
        let pipeline_layout = device
            .create_pipeline_layout(&ds_layouts, push_constants)
            .unwrap();

        let subpass = gfx_hal::pass::Subpass {
            index: 0,
            main_pass: render_pass,
        };

        let flags = gfx_hal::pso::PipelineCreationFlags::empty();

        let parent = gfx_hal::pso::BasePipeline::None;

        let gfx_pipeline = {
            let desc = gfx_hal::pso::GraphicsPipelineDesc {
                shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler,
                blender,
                depth_stencil,
                multisampling,
                baked_states,
                layout: &pipeline_layout,
                subpass,
                flags,
                parent,
            };

            device
                .create_graphics_pipeline(&desc, None)
                .expect("Unable to create graphics pipeline")
        };

        device.destroy_shader_module(vert_shader_module);
        device.destroy_shader_module(frag_shader_module);

        (ds_layouts, pipeline_layout, gfx_pipeline)
    }
}

fn create_framebuffers(
    device: &<back::Backend as Backend>::Device,
    render_pass: &<back::Backend as Backend>::RenderPass,
    frame_images: &[(
        <back::Backend as Backend>::Image,
        <back::Backend as Backend>::ImageView,
    )],
    extent: gfx_hal::window::Extent2D,
) -> Vec<<back::Backend as Backend>::Framebuffer> {
    let mut swapchain_framebuffers = Vec::new();

    unsafe {
        for (_, image_view) in frame_images.iter() {
            let framebuffer = device
                .create_framebuffer(
                    render_pass,
                    vec![image_view],
                    image::Extent {
                        width: extent.width as _,
                        height: extent.height as _,
                        depth: 1,
                    },
                )
                .unwrap();

            swapchain_framebuffers.push(framebuffer);
        }
    }

    swapchain_framebuffers
}

fn create_command_pool(
    device: &<back::Backend as Backend>::Device,
    queue_type: gfx_hal::queue::QueueType,
    qf_id: gfx_hal::queue::family::QueueFamilyId,
) -> gfx_hal::pool::CommandPool<back::Backend, Graphics> {
    unsafe {
        let raw_command_pool = device
            .create_command_pool(qf_id, gfx_hal::pool::CommandPoolCreateFlags::empty())
            .unwrap();

        assert_eq!(Graphics::supported_by(queue_type), true);
        gfx_hal::pool::CommandPool::new(raw_command_pool)
    }
}

fn create_vertex_buffer(
    device: &<back::Backend as Backend>::Device,
) -> (
    <back::Backend as Backend>::Buffer,
    gfx_hal::memory::Requirements,
) {
    unsafe {
        let size = std::mem::size_of::<Vertex>() * vertices.len();
        let buffer = device
            .create_buffer(size as u64, buffer::Usage::VERTEX)
            .unwrap();

        let requirements = device.get_buffer_requirements(&buffer);

        (buffer, requirements)
    }
}

fn create_command_buffers<'a>(
    command_pool: &'a mut gfx_hal::pool::CommandPool<back::Backend, Graphics>,
    render_pass: &<back::Backend as Backend>::RenderPass,
    framebuffers: &[<back::Backend as Backend>::Framebuffer],
    extent: gfx_hal::window::Extent2D,
    pipeline: &<back::Backend as Backend>::GraphicsPipeline,
    vertex_buffer: &<back::Backend as Backend>::Buffer,
) -> Vec<command::CommandBuffer<back::Backend, Graphics, command::MultiShot, command::Primary>> {
    let mut submission_command_buffers = Vec::new();

    for fb in framebuffers {
        unsafe {
            let mut command_buffer: command::CommandBuffer<
                back::Backend,
                Graphics,
                command::MultiShot,
                command::Primary,
            > = command_pool.acquire_command_buffer();

            command_buffer.begin(true);
            command_buffer.bind_graphics_pipeline(pipeline);

            // begin render pass
            {
                let vertex_buffers = vec![(vertex_buffer, 0)];
                command_buffer.bind_vertex_buffers(0, vertex_buffers);

                let render_area = pso::Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as _,
                    h: extent.height as _,
                };

                // we only have one attachment so we only need one clear value
                let clear_values = vec![command::ClearValue::Color(command::ClearColor::Float([
                    0.0, 0.0, 0.0, 1.0,
                ]))];

                let mut render_pass_inline_encoder = command_buffer.begin_render_pass_inline(
                    render_pass,
                    fb,
                    render_area,
                    clear_values.iter(),
                );

                render_pass_inline_encoder.draw(0..vertices.len() as u32, 0..1);
            }

            command_buffer.finish();

            submission_command_buffers.push(command_buffer);
        }
    }

    submission_command_buffers
}

fn create_sync_objects(
    device: &<back::Backend as Backend>::Device,
) -> (
    Vec<<back::Backend as Backend>::Semaphore>,
    Vec<<back::Backend as Backend>::Semaphore>,
    Vec<<back::Backend as Backend>::Fence>,
) {
    let mut image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore> = Vec::new();
    let mut render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore> = Vec::new();
    let mut in_flight_fences: Vec<<back::Backend as Backend>::Fence> = Vec::new();

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        image_available_semaphores.push(device.create_semaphore().unwrap());
        render_finished_semaphores.push(device.create_semaphore().unwrap());
        in_flight_fences.push(device.create_fence(true).unwrap());
    }

    (
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
    )
}
