use gfx_hal::Backend;

pub fn print_adapter_info<B: Backend>(adapter: &gfx_hal::adapter::Adapter<B>) {
    use gfx_hal::PhysicalDevice;
    use gfx_hal::queue::family::QueueFamily;

    println!("{}\n{:?}", adapter.info.name, adapter.info.device_type);

    println!("\nPhysical Device Properties");
    println!("Memory");
    let memory_properties = &adapter.physical_device.memory_properties();
    for memory_type in &memory_properties.memory_types {
        let size = memory_properties.memory_heaps[memory_type.heap_index];
        println!("\t{:?} - {} GB", memory_type.properties, size / 1073741824); // 2^30 = 1 GB = 1073741824
    }

    println!("\nFeatures");
    let feature_string = format!("{:?}", &adapter.physical_device.features());
    println!("\t{}", feature_string.replace(" | ", "\n\t"));


    let limits = &adapter.physical_device.limits();
    println!("\nLimits");
    println!("\tmax_texture_size: {}", limits.max_texture_size);
    println!("\tmax_texel_elements: {}", limits.max_texel_elements);
    println!("\tmax_patch_size: {}", limits.max_patch_size);
    println!("\tmax_viewports: {}", limits.max_viewports);
    println!("\tmax_compute_group_count: {:?}", limits.max_compute_group_count);
    println!("\tmax_compute_group_size: {:?}", limits.max_compute_group_size);
    println!("\tmax_vertex_input_attributes: {}", limits.max_vertex_input_attributes);
    println!("\tmax_vertex_input_bindings: {}", limits.max_vertex_input_bindings);
    println!("\tmax_vertex_input_attribute_offset: {}", limits.max_vertex_input_attribute_offset);
    println!("\tmax_vertex_input_binding_stride: {}", limits.max_vertex_input_binding_stride);
    println!("\tmax_vertex_output_components: {}", limits.max_vertex_output_components);
    println!("\tmin_buffer_copy_offset_alignment: {}", limits.min_buffer_copy_offset_alignment);
    println!("\tmin_buffer_copy_pitch_alignment: {}", limits.min_buffer_copy_pitch_alignment);
    println!("\tmin_texel_buffer_offset_alignment: {}", limits.min_texel_buffer_offset_alignment);
    println!("\tmin_uniform_buffer_offset_alignment: {}", limits.min_uniform_buffer_offset_alignment);
    println!("\tmin_storage_buffer_offset_alignment: {}", limits.min_storage_buffer_offset_alignment);
    println!("\tframebuffer_color_samples_count: {}", limits.framebuffer_color_samples_count);
    println!("\tframebuffer_depth_samples_count: {}", limits.framebuffer_depth_samples_count);
    println!("\tframebuffer_stencil_samples_count: {}", limits.framebuffer_stencil_samples_count);
    println!("\tmax_color_attachments: {}", limits.max_color_attachments);
    println!("\tnon_coherent_atom_size: {}", limits.non_coherent_atom_size);
    println!("\tmax_sampler_anisotropy: {}", limits.max_sampler_anisotropy);

    println!("\nQueue families");
    for family in &adapter.queue_families {
        println!("\t{:?}. Max queues: {}. Queue Family ID: {}", family.queue_type(), family.max_queues(), family.id().0);
    }
}
