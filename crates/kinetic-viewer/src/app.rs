//! Application: fully integrated kinetic viewer with FK, planning, collision, servo.
//!
//! All kinetic crates are wired in:
//! - Joint sliders → FK → robot display
//! - Plan button → kinetic-planning → trajectory → playback
//! - Collision checking → collision viz overlay
//! - Servo panel → kinetic-reactive twist control

use std::sync::{Arc, Mutex};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use kinetic_collision::{RobotSphereModel, SphereGenConfig};
use kinetic_kinematics::{forward_kinematics_all, KinematicChain};
use kinetic_planning::Planner;
use kinetic_core::Pose;

use crate::collision_viz::{self, CollisionVizConfig, CollisionVizData, SphereViz};
use crate::egui_ui::ViewerUI;
use crate::gizmo;
use crate::gpu_buffers::GpuScene;
use crate::gpu_context::GpuContext;
use crate::interaction::{InteractionManager, PlanningPanel, PlanningStatus};
use crate::perception_viz::{self, PerceptionManager};
use crate::pipeline::{LightUniforms, LineVertex, Pipelines};
use crate::trajectory_viz::{self, TrajectoryVizState};
use crate::wgpu_renderer::InstanceData;
use crate::{
    collect_render_commands, update_robot_transforms, Camera, MeshHandle, MeshRegistry,
    RenderCommand, SceneNode, ViewerSettings,
};
use nalgebra::Matrix4;

/// Configuration for launching the viewer.
pub struct ViewerConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub msaa_samples: u32,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            title: "KINETIC Viewer".into(),
            width: 1280,
            height: 720,
            msaa_samples: 4,
        }
    }
}

/// Launch the viewer window and enter the render loop.
pub fn run_viewer(
    config: ViewerConfig,
    robot: kinetic_robot::Robot,
    camera: Camera,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ViewerApp {
        config,
        robot,
        camera,
        state: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct ViewerApp {
    config: ViewerConfig,
    robot: kinetic_robot::Robot,
    camera: Camera,
    state: Option<GpuState>,
}

/// All viewer state.
struct GpuState {
    gpu: GpuContext,
    pipelines: Pipelines,
    scene: GpuScene,
    settings: ViewerSettings,
    // egui
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // Mouse
    mouse_pressed_left: bool,
    mouse_pressed_right: bool,
    last_cursor: Option<(f64, f64)>,
    egui_wants_pointer: bool,
    egui_wants_keyboard: bool,
    // UI + interaction
    ui: ViewerUI,
    planning_panel: PlanningPanel,
    interaction: InteractionManager,
    trajectory_viz: TrajectoryVizState,
    collision_viz_config: CollisionVizConfig,
    collision_viz_data: CollisionVizData,
    perception: PerceptionManager,
    // Gizmo
    active_gizmo_handle: Option<gizmo::GizmoHandle>,
    prev_ray: Option<gizmo::Ray>,
    // Robot state
    scene_root: SceneNode,
    joint_names: Vec<String>,
    joint_values: Vec<f64>,
    joint_limits: Vec<(f64, f64)>,
    link_poses: Vec<Pose>,
    // Kinetic integration
    chain: KinematicChain,
    planner: Planner,
    sphere_model: RobotSphereModel,
    // Servo (Gap 1)
    servo: Option<kinetic_reactive::Servo>,
    servo_scene: Arc<kinetic_scene::Scene>,
    robot_arc: Arc<kinetic_robot::Robot>,
    // Async planning (Gap 2)
    planning_result: Arc<Mutex<Option<Result<kinetic_planning::PlanningResult, kinetic_core::KineticError>>>>,
    planning_thread_active: bool,
    planning_start_time: std::time::Instant,
    // Timing
    last_frame_time: std::time::Instant,
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));

        let window = match event_loop.create_window(window_attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let gpu = match GpuContext::new(window.clone(), self.config.msaa_samples) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("GPU initialization failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let pipelines = Pipelines::new(&gpu.device, gpu.surface_format, gpu.msaa_samples);
        let mut gpu_scene = GpuScene::new(&gpu.device, &pipelines);

        // --- Build kinetic objects from robot ---
        let chain = match KinematicChain::auto_detect(&self.robot) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to detect kinematic chain: {e}");
                event_loop.exit();
                return;
            }
        };

        let planner = match Planner::new(&self.robot) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to create planner: {e}");
                event_loop.exit();
                return;
            }
        };

        let sphere_model = RobotSphereModel::from_robot(&self.robot, &SphereGenConfig::coarse());

        // Extract joint info from active joint indices
        let joint_names: Vec<String> = self.robot.active_joints.iter()
            .map(|&idx| self.robot.joints[idx].name.clone())
            .collect();
        let joint_values = vec![0.0f64; chain.dof];
        let joint_limits: Vec<(f64, f64)> = self.robot.joint_limits.iter()
            .map(|l| (l.lower, l.upper))
            .collect();

        // Initial FK
        let link_poses = forward_kinematics_all(&self.robot, &chain, &joint_values)
            .unwrap_or_else(|_| vec![Pose::identity(); self.robot.links.len()]);

        // Build scene graph with meshes
        let mut mesh_registry = crate::MeshRegistry::new();
        let mut scene_root = crate::robot_scene_graph(&self.robot, &mut mesh_registry);
        update_robot_transforms(&mut scene_root, &link_poses);
        gpu_scene.upload_registry(&gpu.device, &mesh_registry);

        // egui
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(), egui_ctx.viewport_id(), &window,
            Some(window.scale_factor() as f32), None, None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.surface_format, None, 1, false);

        let robot_arc = Arc::new(self.robot.clone());
        let servo_scene = Arc::new(
            kinetic_scene::Scene::new(&self.robot)
                .unwrap_or_else(|_| kinetic_scene::Scene::with_chain(&self.robot, chain.clone()))
        );

        self.state = Some(GpuState {
            gpu, pipelines, scene: gpu_scene, settings: ViewerSettings::default(),
            egui_ctx, egui_winit, egui_renderer,
            mouse_pressed_left: false, mouse_pressed_right: false,
            last_cursor: None, egui_wants_pointer: false, egui_wants_keyboard: false,
            ui: ViewerUI::default(), planning_panel: PlanningPanel::default(),
            interaction: InteractionManager::new(),
            trajectory_viz: TrajectoryVizState::default(),
            collision_viz_config: CollisionVizConfig::default(),
            collision_viz_data: CollisionVizData::default(),
            perception: PerceptionManager::default(),
            active_gizmo_handle: None, prev_ray: None,
            scene_root, joint_names, joint_values, joint_limits, link_poses,
            chain, planner, sphere_model,
            servo: None,
            servo_scene,
            robot_arc,
            planning_result: Arc::new(Mutex::new(None)),
            planning_thread_active: false,
            planning_start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        let egui_response = state.egui_winit.on_window_event(&state.gpu.window, &event);
        if egui_response.consumed {
            state.gpu.window.request_redraw();
            return;
        }
        state.egui_wants_pointer = state.egui_ctx.wants_pointer_input();
        state.egui_wants_keyboard = state.egui_ctx.wants_keyboard_input();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                state.gpu.window.request_redraw();
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if state.egui_wants_pointer { return; }
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed {
                            state.mouse_pressed_left = true;
                            // Try gizmo hit test on click
                            if let Some((cx, cy)) = state.last_cursor {
                                let ray = gizmo::screen_to_ray(
                                    cx as f32, cy as f32,
                                    state.gpu.width() as f32, state.gpu.height() as f32,
                                    &self.camera, state.gpu.aspect(),
                                );
                                // Test against goal marker gizmo
                                if let Some(marker) = state.interaction.markers.get("goal") {
                                    let pos = marker.position();
                                    let center = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
                                    if let Some(handle) = gizmo::hit_test_gizmo(&ray, center, 0.15, 0.12, 0.03) {
                                        state.active_gizmo_handle = Some(handle);
                                        state.prev_ray = Some(ray);
                                    }
                                }
                            }
                        } else {
                            state.mouse_pressed_left = false;
                            state.active_gizmo_handle = None;
                            state.prev_ray = None;
                        }
                    }
                    MouseButton::Right => state.mouse_pressed_right = pressed,
                    MouseButton::Middle => {
                        // Middle-click: place goal marker at click position on ground plane
                        if pressed {
                            if let Some((cx, cy)) = state.last_cursor {
                                let ray = gizmo::screen_to_ray(
                                    cx as f32, cy as f32,
                                    state.gpu.width() as f32, state.gpu.height() as f32,
                                    &self.camera, state.gpu.aspect(),
                                );
                                // Intersect ray with Y=0 ground plane
                                if ray.direction.y.abs() > 1e-6 {
                                    let t = -ray.origin.y / ray.direction.y;
                                    if t > 0.0 {
                                        let hit = ray.origin + ray.direction * t;
                                        let pose = nalgebra::Isometry3::translation(
                                            hit.x as f64, hit.y as f64, hit.z as f64,
                                        );
                                        state.interaction.set_goal_pose(pose);
                                        state.gpu.window.request_redraw();
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                if !state.egui_wants_pointer {
                    if state.mouse_pressed_left && state.active_gizmo_handle.is_some() {
                        let ray = gizmo::screen_to_ray(
                            new_pos.0 as f32, new_pos.1 as f32,
                            state.gpu.width() as f32, state.gpu.height() as f32,
                            &self.camera, state.gpu.aspect(),
                        );
                        if let (Some(handle), Some(prev)) = (state.active_gizmo_handle, state.prev_ray) {
                            if let Some(marker) = state.interaction.markers.get("goal") {
                                let pos = marker.position();
                                let center = nalgebra::Point3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
                                let axis = handle.axis();
                                let delta = gizmo::drag_along_axis(&ray, &prev, &center, &axis);
                                let offset = nalgebra::Isometry3::translation(
                                    (axis.x * delta) as f64, (axis.y * delta) as f64, (axis.z * delta) as f64,
                                );
                                let current = state.interaction.markers["goal"].pose;
                                state.interaction.move_marker("goal", offset * current);
                            }
                        }
                        state.prev_ray = Some(ray);
                        state.gpu.window.request_redraw();
                    } else if let Some(last) = state.last_cursor {
                        let dx = (new_pos.0 - last.0) as f32;
                        let dy = (new_pos.1 - last.1) as f32;
                        let s = 0.005;
                        if state.mouse_pressed_left {
                            self.camera.orbit(-dx * s, -dy * s);
                            state.gpu.window.request_redraw();
                        }
                        if state.mouse_pressed_right {
                            let right = (self.camera.target - self.camera.position).cross(&self.camera.up).normalize() * dx * s * 0.5;
                            let up = self.camera.up * dy * s * 0.5;
                            self.camera.position -= right; self.camera.target -= right;
                            self.camera.position += up; self.camera.target += up;
                            state.gpu.window.request_redraw();
                        }
                    }
                }
                state.last_cursor = Some(new_pos);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !state.egui_wants_pointer {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.002,
                    };
                    self.camera.zoom(scroll);
                    state.gpu.window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !state.egui_wants_keyboard =>
            {
                use winit::keyboard::{Key, NamedKey};
                match event.logical_key {
                    Key::Character(ref c) => match c.as_str() {
                        "g" => state.settings.show_grid = !state.settings.show_grid,
                        "a" => state.settings.show_axes = !state.settings.show_axes,
                        "t" => state.settings.show_trajectory_trail = !state.settings.show_trajectory_trail,
                        "c" => state.ui.show_collision_debug = !state.ui.show_collision_debug,
                        "v" => state.settings.show_voxels = !state.settings.show_voxels,
                        "w" => state.settings.show_collision_wireframe = !state.settings.show_collision_wireframe,
                        "p" => state.settings.show_point_cloud = !state.settings.show_point_cloud,
                        "r" => self.camera = Camera::perspective([2.0, 1.5, 2.0], [0.0, 0.3, 0.0], 45.0),
                        " " => {
                            if let Some(player) = &mut state.trajectory_viz.player {
                                if player.is_playing() { player.pause(); } else { player.play(); }
                            }
                        }
                        "z" => { state.interaction.undo(); }
                        "f" => {
                            // Focus camera on goal marker if one exists
                            if let Some(marker) = state.interaction.markers.get("goal") {
                                let pos = marker.position();
                                self.camera.target = nalgebra::Point3::new(
                                    pos[0] as f32, pos[1] as f32, pos[2] as f32,
                                );
                            }
                        }
                        _ => {}
                    },
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::F1) => state.ui.show_shortcuts = !state.ui.show_shortcuts,
                    Key::Named(NamedKey::F3) => state.ui.show_stats = !state.ui.show_stats,
                    _ => {}
                }
                state.gpu.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                render_frame(state, &self.camera, &self.robot);
                state.gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(state: &mut GpuState, camera: &Camera, robot: &kinetic_robot::Robot) {
    let now = std::time::Instant::now();
    let dt = now.duration_since(state.last_frame_time).as_secs_f64();
    state.last_frame_time = now;

    // --- Trajectory playback → FK update ---
    if let Some(player) = &mut state.trajectory_viz.player {
        if player.is_playing() {
            let joints = player.tick(dt);
            for (i, v) in joints.as_slice().iter().enumerate() {
                if i < state.joint_values.len() { state.joint_values[i] = *v; }
            }
        }
    }

    // --- Servo: send twist each frame when active (Gap 1) ---
    if let Some(servo) = &mut state.servo {
        let twist_arr = state.interaction.servo.twist;
        let twist = kinetic_core::Twist::from_slice(&twist_arr);
        // Sync servo state from current joint values
        let zeros = vec![0.0; state.joint_values.len()];
        let _ = servo.set_state(&state.joint_values, &zeros);
        match servo.send_twist(&twist) {
            Ok(cmd) => {
                for (i, &pos) in cmd.positions.iter().enumerate() {
                    if i < state.joint_values.len() {
                        state.joint_values[i] = pos;
                    }
                }
                // Update servo overlay status from servo state
                let ss = servo.state();
                state.interaction.servo.collision_distance = ss.min_obstacle_distance;
                state.interaction.servo.near_singularity = ss.is_near_singularity;
                state.interaction.servo.velocity_magnitude = twist.linear_magnitude();
            }
            Err(_) => {
                // Emergency stop or other error — keep current joints
            }
        }
    }

    // --- Poll async planning result (Gap 2) ---
    if state.planning_thread_active {
        // GAP 13: Timeout check — fail if planning exceeds 30 seconds
        if state.planning_start_time.elapsed().as_secs() > 30 {
            state.planning_thread_active = false;
            state.planning_panel.status = PlanningStatus::Failed("Planning timed out".into());
        }
        let mut result_lock = state.planning_result.lock().unwrap();
        if let Some(result) = result_lock.take() {
            state.planning_thread_active = false;
            match result {
                Ok(plan_result) => {
                    state.planning_panel.status = PlanningStatus::Succeeded;
                    state.planning_panel.last_planning_time = Some(plan_result.planning_time);
                    let dof = plan_result.waypoints.first().map_or(0, |w| w.len());
                    let mut traj = kinetic_core::Trajectory::with_dof(dof);
                    for wp in &plan_result.waypoints {
                        traj.push_waypoint(wp);
                    }
                    state.trajectory_viz.set_trajectory(traj, &state.planning_panel.planner_id);
                }
                Err(e) => {
                    state.planning_panel.status = PlanningStatus::Failed(format!("{e}"));
                }
            }
        }
    }

    // --- FK from current joint values ---
    if let Ok(poses) = forward_kinematics_all(robot, &state.chain, &state.joint_values) {
        state.link_poses = poses;
        update_robot_transforms(&mut state.scene_root, &state.link_poses);
    }

    // --- Collision checking (generates viz data for debug overlay) ---
    state.collision_viz_data = build_collision_viz(&state.sphere_model, &state.link_poses);

    // --- Surface ---
    let output = match state.gpu.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            state.gpu.surface.configure(&state.gpu.device, &state.gpu.surface_config);
            return;
        }
        Err(e) => { eprintln!("Surface error: {e}"); return; }
    };
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Uniforms ---
    let aspect = state.gpu.aspect();
    let view_uniforms = state.scene.update_view_uniforms_from_camera(camera, aspect);
    state.scene.update_view_uniforms(&state.gpu.queue, &view_uniforms);
    state.scene.update_light_uniforms(&state.gpu.queue, &LightUniforms::default());

    // --- 3D commands ---
    let mut commands = Vec::new();
    // GAP 4: Gate robot rendering behind show_robot
    if state.settings.show_robot {
        collect_render_commands(&state.scene_root, &Matrix4::identity(), &mut commands);
    }
    if state.settings.show_grid {
        commands.push(RenderCommand::DrawGrid { size: state.settings.grid_size, divisions: state.settings.grid_divisions });
    }
    if state.settings.show_axes {
        commands.push(RenderCommand::DrawAxes { length: 0.5 });
    }

    let (batches, mut line_vertices) = build_draw_data(&commands);

    // Gizmo
    if let Some(marker) = state.interaction.markers.get("goal") {
        let pos = marker.position();
        let c = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
        line_vertices.extend(gizmo::translation_gizmo_lines(c, 0.15));
        line_vertices.extend(gizmo::rotation_gizmo_lines(c, 0.12, 24));
    }
    // Trajectory trails
    if state.settings.show_trajectory_trail { line_vertices.extend(state.trajectory_viz.collect_trail_lines()); }
    // Collision viz — gated behind show_collision_geometry (GAP 5)
    if state.settings.show_collision_geometry {
        line_vertices.extend(collision_viz::collect_collision_lines(&state.collision_viz_data, &state.collision_viz_config));
    }
    // Octree
    if state.settings.show_voxels { line_vertices.extend(state.perception.collect_octree_lines()); }
    // Point clouds (GAP 3)
    if state.settings.show_point_cloud { line_vertices.extend(state.perception.collect_point_cloud_lines()); }

    // Ghost robots: render translucent copies at trajectory start and goal poses.
    if state.trajectory_viz.show_ghosts {
        if let Some(ref player) = state.trajectory_viz.player {
            let ghost_configs: [(f64, [f32; 4]); 2] = [
                (0.0, [0.2, 0.8, 0.2, 0.3]), // start ghost — green transparent
                (1.0, [0.2, 0.4, 1.0, 0.3]), // goal ghost — blue transparent
            ];
            for (ghost_t, ghost_color) in ghost_configs {
                let ghost_joints = player.trajectory().sample(ghost_t);
                if let Ok(ghost_poses) = forward_kinematics_all(robot, &state.chain, ghost_joints.as_slice()) {
                    update_robot_transforms(&mut state.scene_root, &ghost_poses);
                    let mut ghost_cmds = Vec::new();
                    collect_render_commands(&state.scene_root, &Matrix4::identity(), &mut ghost_cmds);
                    for cmd in &mut ghost_cmds {
                        if let RenderCommand::DrawMesh { material, .. } = cmd {
                            material.color = ghost_color;
                        }
                    }
                    commands.extend(ghost_cmds);
                }
            }
            // Restore original link poses and transforms
            update_robot_transforms(&mut state.scene_root, &state.link_poses);
        }
    }

    let line_count = state.scene.write_lines(&state.gpu.device, &state.gpu.queue, &line_vertices);

    // --- egui ---
    let egui_ctx = state.egui_ctx.clone();
    let egui_input = state.egui_winit.take_egui_input(&state.gpu.window);
    let egui_output = egui_ctx.run(egui_input, |ctx| { draw_egui_panels(ctx, state, robot); });
    state.egui_winit.handle_platform_output(&state.gpu.window, egui_output.platform_output);

    let egui_prims = state.egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);
    let screen_desc = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [state.gpu.width(), state.gpu.height()],
        pixels_per_point: egui_output.pixels_per_point,
    };
    for (id, delta) in &egui_output.textures_delta.set {
        state.egui_renderer.update_texture(&state.gpu.device, &state.gpu.queue, *id, delta);
    }

    let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render") });
    let egui_bufs = state.egui_renderer.update_buffers(&state.gpu.device, &state.gpu.queue, &mut encoder, &egui_prims, &screen_desc);

    // --- 3D pass ---
    {
        let color_att = if state.gpu.msaa_samples > 1 {
            wgpu::RenderPassColorAttachment {
                view: &state.gpu.msaa_texture, resolve_target: Some(&view),
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color(&state.settings)), store: wgpu::StoreOp::Store },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: &view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color(&state.settings)), store: wgpu::StoreOp::Store },
            }
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3d"), color_attachments: &[Some(color_att)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &state.gpu.depth_texture,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        for batch in &batches {
            if state.scene.get_mesh(MeshHandle(batch.mesh_id)).is_none() { continue; }
            let n = state.scene.write_instances(&state.gpu.device, &state.gpu.queue, &batch.instances);
            let gm = state.scene.get_mesh(MeshHandle(batch.mesh_id)).unwrap();
            pass.set_pipeline(if batch.wireframe { &state.pipelines.wireframe_pipeline } else { &state.pipelines.mesh_pipeline });
            pass.set_bind_group(0, &state.scene.mesh_bind_group, &[]);
            pass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, state.scene.instance_buffer.slice(..));
            pass.set_index_buffer(gm.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gm.index_count, 0, 0..n);
        }
        if line_count > 0 {
            pass.set_pipeline(&state.pipelines.line_pipeline);
            pass.set_bind_group(0, &state.scene.line_bind_group, &[]);
            pass.set_vertex_buffer(0, state.scene.line_buffer.slice(..));
            pass.draw(0..line_count, 0..1);
        }
    }

    // --- egui pass ---
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })], depth_stencil_attachment: None, ..Default::default()
        });
        let mut sp: wgpu::RenderPass<'static> = pass.forget_lifetime();
        state.egui_renderer.render(&mut sp, &egui_prims, &screen_desc);
    }

    state.gpu.queue.submit(egui_bufs.into_iter().chain(std::iter::once(encoder.finish())));
    output.present();
    for id in &egui_output.textures_delta.free { state.egui_renderer.free_texture(id); }
}

fn draw_egui_panels(ctx: &egui::Context, state: &mut GpuState, robot: &kinetic_robot::Robot) {
    use crate::egui_ui;

    if state.ui.show_planning_panel {
        let actions = egui_ui::planning_panel_ui(ctx, &mut state.planning_panel);
        if actions.plan_clicked && !state.planning_thread_active {
            // --- PLAN: run planner in background thread (Gap 2) ---
            if let Some(goal) = state.interaction.get_goal() {
                state.planning_panel.status = PlanningStatus::Planning;

                // Gap 3: Set planner type from UI selection before planning
                let planner_type = kinetic_planning::PlannerType::from_id(&actions.planner_id);
                state.planner.set_planner_type(planner_type);

                // Clone what the background thread needs
                let start = state.joint_values.clone();
                let result_slot = Arc::clone(&state.planning_result);

                // Build a fresh planner for the thread (Planner is not Send, so reconstruct)
                let robot_clone = robot.clone();
                let planner_id = actions.planner_id.clone();

                // Sync scene objects into a collision scene for the planner
                let scene_objects_snapshot: Vec<(String, crate::interaction::SceneShapeUI, nalgebra::Isometry3<f64>)> =
                    state.interaction.scene_objects.iter().map(|obj| {
                        (obj.name.clone(), obj.shape.clone(), obj.pose)
                    }).collect();

                state.planning_thread_active = true;
                state.planning_start_time = std::time::Instant::now();
                std::thread::spawn(move || {
                    let plan_result = (|| {
                        let pt = kinetic_planning::PlannerType::from_id(&planner_id);
                        let mut planner = kinetic_planning::Planner::new(&robot_clone)?
                            .with_planner_type(pt);

                        // Add scene objects as collision obstacles
                        if !scene_objects_snapshot.is_empty() {
                            let scene = kinetic_scene::Scene::with_chain(
                                &robot_clone,
                                kinetic_kinematics::KinematicChain::auto_detect(&robot_clone)?,
                            );
                            let mut planning_scene = scene;
                            for (name, shape, pose) in &scene_objects_snapshot {
                                let ks_shape = match shape {
                                    crate::interaction::SceneShapeUI::Box { half_extents } =>
                                        kinetic_scene::Shape::Cuboid(half_extents[0], half_extents[1], half_extents[2]),
                                    crate::interaction::SceneShapeUI::Sphere { radius } =>
                                        kinetic_scene::Shape::Sphere(*radius),
                                    crate::interaction::SceneShapeUI::Cylinder { radius, half_height } =>
                                        kinetic_scene::Shape::Cylinder(*radius, *half_height),
                                    crate::interaction::SceneShapeUI::Mesh { .. } =>
                                        kinetic_scene::Shape::Cuboid(0.05, 0.05, 0.05),
                                };
                                planning_scene.add(name, ks_shape, *pose);
                            }
                            planner = planner.with_scene(&planning_scene);
                        }

                        planner.plan(&start, &goal)
                    })();
                    let mut lock = result_slot.lock().unwrap();
                    *lock = Some(plan_result);
                });
            } else {
                state.planning_panel.status = PlanningStatus::Failed("No goal set. Click in 3D to place a goal marker.".into());
            }
        }
    }

    if state.ui.show_joint_sliders && !state.joint_names.is_empty() {
        egui_ui::joint_slider_panel_ui(ctx, &state.joint_names, &mut state.joint_values, &state.joint_limits);
    }

    if state.ui.show_scene_panel {
        egui_ui::scene_object_panel_ui(ctx, &mut state.interaction.scene_objects);
    }
    if state.ui.show_constraints {
        egui_ui::constraint_panel_ui(ctx, &mut state.interaction.constraints);
    }
    if state.ui.show_servo {
        // Gap 1: Wire servo start/stop to kinetic_reactive::Servo
        let actions = egui_ui::servo_panel_ui(ctx, &mut state.interaction.servo);
        if actions.start {
            state.interaction.servo.active = true;
            match kinetic_reactive::Servo::new(
                &state.robot_arc,
                &state.servo_scene,
                kinetic_reactive::ServoConfig::default(),
            ) {
                Ok(mut s) => {
                    let zeros = vec![0.0; state.joint_values.len()];
                    let _ = s.set_state(&state.joint_values, &zeros);
                    state.servo = Some(s);
                }
                Err(e) => {
                    log::warn!("Failed to start servo: {e}");
                    state.interaction.servo.active = false;
                }
            }
        }
        if actions.stop {
            state.interaction.servo.active = false;
            state.servo = None;
            // Reset twist sliders
            state.interaction.servo.twist = [0.0; 6];
        }
    }

    // Gap 4: Gate collision debug panel behind toggle
    if state.ui.show_collision_debug {
        collision_viz::collision_debug_panel_ui(ctx, &mut state.collision_viz_config);
    }
    if !state.perception.point_clouds.is_empty() || !state.perception.octrees.is_empty() {
        perception_viz::perception_panel_ui(ctx, &mut state.perception);
    }
    trajectory_viz::playback_panel_ui(ctx, &mut state.trajectory_viz);

    if state.ui.show_stats { egui_ui::stats_panel_ui(ctx, &state.settings); }
    if state.ui.show_shortcuts { egui_ui::shortcuts_panel_ui(ctx); }
}

/// Build collision viz data from current sphere model and link poses.
fn build_collision_viz(
    sphere_model: &RobotSphereModel,
    link_poses: &[Pose],
) -> CollisionVizData {
    let mut data = CollisionVizData::default();

    // Generate sphere markers for each link
    for (link_idx, &(start, end)) in sphere_model.link_ranges.iter().enumerate() {
        if link_idx >= link_poses.len() { continue; }
        let pose = &link_poses[link_idx];

        for si in start..end {
            let lx = sphere_model.local.x[si];
            let ly = sphere_model.local.y[si];
            let lz = sphere_model.local.z[si];
            let radius = sphere_model.local.radius[si] as f32;

            // Transform to world frame
            let world = pose.0 * nalgebra::Point3::new(lx, ly, lz);

            data.robot_spheres.push(SphereViz {
                center: [world.x as f32, world.y as f32, world.z as f32],
                radius,
                color: [0.2, 0.4, 1.0, 0.3],
            });
        }
    }

    data
}

fn bg_color(settings: &ViewerSettings) -> wgpu::Color {
    wgpu::Color {
        r: settings.background_color[0] as f64,
        g: settings.background_color[1] as f64,
        b: settings.background_color[2] as f64,
        a: 1.0,
    }
}

struct DrawBatch { mesh_id: usize, wireframe: bool, instances: Vec<InstanceData> }

fn build_draw_data(commands: &[RenderCommand]) -> (Vec<DrawBatch>, Vec<LineVertex>) {
    let mut batches: Vec<DrawBatch> = Vec::new();
    let mut lines: Vec<LineVertex> = Vec::new();
    for cmd in commands {
        match cmd {
            RenderCommand::DrawMesh { handle, transform, material } => {
                let inst = InstanceData { model: m2a(transform), color: material.color };
                let wf = material.wireframe;
                let id = handle.0;
                if let Some(b) = batches.iter_mut().find(|b| b.mesh_id == id && b.wireframe == wf) {
                    b.instances.push(inst);
                } else {
                    batches.push(DrawBatch { mesh_id: id, wireframe: wf, instances: vec![inst] });
                }
            }
            RenderCommand::DrawLine { start, end, color } => {
                lines.push(LineVertex { position: *start, color: *color });
                lines.push(LineVertex { position: *end, color: *color });
            }
            RenderCommand::DrawGrid { size, divisions } => {
                let h = *size; let step = (2.0 * h) / *divisions as f32; let c = [0.3, 0.3, 0.3, 0.5];
                for i in 0..=*divisions {
                    let p = -h + step * i as f32;
                    lines.push(LineVertex { position: [p, 0.0, -h], color: c });
                    lines.push(LineVertex { position: [p, 0.0, h], color: c });
                    lines.push(LineVertex { position: [-h, 0.0, p], color: c });
                    lines.push(LineVertex { position: [h, 0.0, p], color: c });
                }
            }
            RenderCommand::DrawAxes { length } => {
                let l = *length;
                lines.push(LineVertex { position: [0.0,0.0,0.0], color: [1.0,0.0,0.0,1.0] });
                lines.push(LineVertex { position: [l,0.0,0.0], color: [1.0,0.0,0.0,1.0] });
                lines.push(LineVertex { position: [0.0,0.0,0.0], color: [0.0,1.0,0.0,1.0] });
                lines.push(LineVertex { position: [0.0,l,0.0], color: [0.0,1.0,0.0,1.0] });
                lines.push(LineVertex { position: [0.0,0.0,0.0], color: [0.0,0.0,1.0,1.0] });
                lines.push(LineVertex { position: [0.0,0.0,l], color: [0.0,0.0,1.0,1.0] });
            }
            RenderCommand::DrawPoint { .. } => {}
        }
    }
    (batches, lines)
}

fn m2a(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [[s[0],s[1],s[2],s[3]], [s[4],s[5],s[6],s[7]], [s[8],s[9],s[10],s[11]], [s[12],s[13],s[14],s[15]]]
}
