pub struct WindowState {
    pub events_loop: winit::EventsLoop,
    pub window: winit::Window,
}

impl WindowState {
    pub fn new() -> WindowState {
        // Platform/backend initialization
        std::env::set_var("WINIT_UNIX_BACKEND", "x11");

        let initial_size = winit::dpi::LogicalSize::new(854.0, 480.0);

        let events_loop = winit::EventsLoop::new();
        let wb = winit::WindowBuilder::new()
            .with_title("elevate".to_string())
            .with_min_dimensions(initial_size)
            .with_dimensions(initial_size);

        let window = wb.build(&events_loop).unwrap();

        WindowState {
            events_loop,
            window,
        }
    }
}
