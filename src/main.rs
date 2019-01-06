extern crate gfx_backend_vulkan as back;
#[macro_use]
extern crate lazy_static;

mod render;
mod window;

fn main() {
    env_logger::init();

    let mut window_state = window::WindowState::new();
    let mut renderer = render::Renderer::new(&window_state.window);

    let mut is_running = true;
    while is_running {
        window_state.events_loop.poll_events(|event| match event {
            winit::Event::WindowEvent { event, .. } => match event {
                winit::WindowEvent::CloseRequested => is_running = false,
                winit::WindowEvent::Resized(_) => unsafe { renderer.recreate_swapchain() },
                _ => (),
            },
            winit::Event::DeviceEvent { .. } => (),
            winit::Event::Awakened => (),
            winit::Event::Suspended(_) => (),
        });

        renderer.draw_frame();
    }

    renderer.cleanup();
}
