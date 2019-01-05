extern crate gfx_backend_vulkan as back;

mod render;
mod util;
mod window;

fn main() {
    env_logger::init();

    let mut window_state = window::WindowState::new();
    let mut renderer = render::Renderer::new(&window_state.window);

    window_state.events_loop.run_forever(|event| match event {
        winit::Event::WindowEvent {
            event: winit::WindowEvent::CloseRequested,
            ..
        } => {
            renderer.wait_until_idle();
            winit::ControlFlow::Break
        }
        _ => {
            renderer.draw_frame();
            winit::ControlFlow::Continue
        }
    });

    unsafe {
        renderer.cleanup();
    }
}
