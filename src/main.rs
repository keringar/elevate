use gfx_hal::Instance;

mod util;

fn main() {
    env_logger::init();

    // Platform/backend initialization
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let mut events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new().with_title("elevate".to_string());
    let window = wb.build(&events_loop).unwrap();
    let instance = gfx_backend_vulkan::Instance::create("elevate", 1);
    let surface = instance.create_surface(&window);
    let adapters = instance.enumerate_adapters();

    // Choose a dedicated card, falling back to a integrated one.
    // Eventually expose this to the user so we don't need to pick one
    let mut adapter = match adapters.iter().filter(|a| a.info.device_type == gfx_hal::adapter::DeviceType::DiscreteGpu).next() {
        Some(adapter) => adapter,
        None => match adapters.iter().filter(|a| a.info.device_type == gfx_hal::adapter::DeviceType::IntegratedGpu).next() {
            Some(adapter) => adapter,
            None => adapters.first().expect("No device adapter found"),
        }
    };

    util::print_adapter_info(&adapter);
}
