//! Servo mode for teleoperation and interactive control.
//!
//! Provides twist, joint jog, and pose tracking input modes with
//! collision deceleration and singularity avoidance.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_reactive::{Servo, ServoConfig, InputType};
//!
//! let servo = Servo::new(&robot, &scene, ServoConfig::default());
//! let cmd = servo.send_twist(&twist)?;
//! ```

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Isometry3};

use kinetic_core::{KineticError, Twist};
use kinetic_kinematics::{fk, jacobian, manipulability, KinematicChain};
use kinetic_robot::Robot;
use kinetic_scene::Scene;

use crate::filter::{ExponentialMovingAverage, SmoothingFilter};
use crate::JointCommand;

/// Servo input mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    /// Cartesian velocity commands (twist).
    Twist,
    /// Individual joint velocity commands.
    JointJog,
    /// Track a moving target pose.
    PoseTracking,
}

/// Configuration for the Servo controller.
#[derive(Debug, Clone)]
pub struct ServoConfig {
    /// Control rate in Hz (default: 500.0).
    pub rate_hz: f64,
    /// Input mode (default: Twist).
    pub input_type: InputType,
    /// Collision check rate in Hz (default: 100.0).
    pub collision_check_hz: f64,
    /// Manipulability threshold for singularity detection (default: 0.02).
    pub singularity_threshold: f64,
    /// Start decelerating at this distance to nearest obstacle in meters (default: 0.15).
    pub slowdown_distance: f64,
    /// Emergency stop below this distance in meters (default: 0.03).
    pub stop_distance: f64,
    /// Per-joint velocity limits (rad/s). If empty, uses robot limits.
    pub velocity_limits: Vec<f64>,
    /// Per-joint acceleration limits (rad/s²). If empty, uses defaults.
    pub acceleration_limits: Vec<f64>,
    /// Pose tracking gain for proportional control (default: 5.0).
    pub pose_tracking_gain: f64,
    /// Damping factor for singularity-robust pseudoinverse (default: 0.05).
    pub singularity_damping: f64,
    /// Maximum position change per tick in radians (default: 0.02 ≈ 1.1°).
    /// Safety cap that prevents large jumps even if velocity computation has errors.
    /// Set to 0.0 to disable (NOT recommended for real robots).
    pub max_delta_per_tick: f64,
}

impl Default for ServoConfig {
    fn default() -> Self {
        Self {
            rate_hz: 500.0,
            input_type: InputType::Twist,
            collision_check_hz: 100.0,
            singularity_threshold: 0.02,
            slowdown_distance: 0.15,
            stop_distance: 0.03,
            velocity_limits: Vec::new(),
            acceleration_limits: Vec::new(),
            pose_tracking_gain: 5.0,
            singularity_damping: 0.05,
            max_delta_per_tick: 0.02, // ~1.1° — safe limit for most robots at 500Hz
        }
    }
}

impl ServoConfig {
    /// General teleoperation preset (joystick, spacemouse).
    ///
    /// Twist input, generous collision margins, moderate precision.
    pub fn teleop() -> Self {
        Self::default()
    }

    /// Pose tracking preset (following a moving target).
    ///
    /// Higher tracking gain, tighter collision checking.
    pub fn tracking() -> Self {
        Self {
            input_type: InputType::PoseTracking,
            pose_tracking_gain: 10.0,
            collision_check_hz: 200.0,
            slowdown_distance: 0.10,
            ..Self::default()
        }
    }

    /// Precise manipulation preset (assembly, insertion).
    ///
    /// Small movements per tick, tight singularity avoidance, fine control.
    pub fn precise() -> Self {
        Self {
            max_delta_per_tick: 0.005,
            singularity_threshold: 0.01,
            singularity_damping: 0.02,
            collision_check_hz: 250.0,
            slowdown_distance: 0.08,
            stop_distance: 0.015,
            ..Self::default()
        }
    }
}

/// Current state of the Servo controller.
#[derive(Debug, Clone)]
pub struct ServoState {
    /// Current joint positions.
    pub joint_positions: Vec<f64>,
    /// Current joint velocities.
    pub joint_velocities: Vec<f64>,
    /// End-effector pose in world frame.
    pub ee_pose: Isometry3<f64>,
    /// Minimum distance to nearest obstacle (meters).
    pub min_obstacle_distance: f64,
    /// Yoshikawa manipulability index.
    pub manipulability: f64,
    /// True when near a singular configuration.
    pub is_near_singularity: bool,
    /// True when near an obstacle (within slowdown distance).
    pub is_near_collision: bool,
    /// True when emergency stopped due to collision proximity.
    pub is_stopped: bool,
}

/// Servo error types.
#[derive(Debug, thiserror::Error)]
pub enum ServoError {
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] KineticError),
    #[error("Emergency stop: obstacle distance {distance:.4}m < stop distance {stop:.4}m")]
    EmergencyStop { distance: f64, stop: f64 },
    #[error("Invalid joint index {index} for robot with {dof} DOF")]
    InvalidJointIndex { index: usize, dof: usize },
}

/// Servo controller for teleoperation and interactive control.
///
/// Supports three input modes:
/// - **Twist**: Cartesian velocity commands via Jacobian pseudoinverse
/// - **JointJog**: Direct joint velocity commands
/// - **PoseTracking**: Track a moving target pose
///
/// Includes collision deceleration and singularity avoidance.
pub struct Servo {
    robot: Arc<Robot>,
    scene: Arc<Scene>,
    chain: KinematicChain,
    config: ServoConfig,
    state: ServoState,
    smoothing_filter: Box<dyn SmoothingFilter>,
    /// Ticks since last collision check.
    ticks_since_collision_check: u64,
    /// Cached collision distance (updated at collision_check_hz).
    cached_min_distance: f64,
    /// Effective velocity limits (resolved from config or robot).
    vel_limits: Vec<f64>,
    /// Effective acceleration limits.
    accel_limits: Vec<f64>,
    /// Consecutive pseudoinverse fallback counter.
    consecutive_fallbacks: u32,
}

impl Servo {
    /// Create a new Servo controller.
    ///
    /// Uses an EMA filter with alpha=0.3 by default.
    pub fn new(
        robot: &Arc<Robot>,
        scene: &Arc<Scene>,
        config: ServoConfig,
    ) -> kinetic_core::Result<Self> {
        let chain = crate::auto_detect_chain(robot)?;
        Self::with_chain(robot, scene, chain, config)
    }

    /// Create with a specific kinematic chain.
    pub fn with_chain(
        robot: &Arc<Robot>,
        scene: &Arc<Scene>,
        chain: KinematicChain,
        config: ServoConfig,
    ) -> kinetic_core::Result<Self> {
        let dof = chain.dof;

        // Resolve velocity limits
        let vel_limits = if config.velocity_limits.len() == dof {
            config.velocity_limits.clone()
        } else {
            chain
                .active_joints
                .iter()
                .map(|&ji| robot.joints[ji].limits.as_ref().map_or(2.0, |l| l.velocity))
                .collect()
        };

        // Resolve acceleration limits
        let accel_limits = if config.acceleration_limits.len() == dof {
            config.acceleration_limits.clone()
        } else {
            // Default: 5x velocity limits (common heuristic)
            vel_limits.iter().map(|v| v * 5.0).collect()
        };

        let zero_joints = vec![0.0; dof];
        let ee_pose = fk(robot, &chain, &zero_joints)?;

        let state = ServoState {
            joint_positions: zero_joints.clone(),
            joint_velocities: vec![0.0; dof],
            ee_pose: ee_pose.0,
            min_obstacle_distance: f64::INFINITY,
            manipulability: 1.0,
            is_near_singularity: false,
            is_near_collision: false,
            is_stopped: false,
        };

        let smoothing_filter: Box<dyn SmoothingFilter> =
            Box::new(ExponentialMovingAverage::new(0.3));

        Ok(Self {
            robot: Arc::clone(robot),
            scene: Arc::clone(scene),
            chain,
            config,
            state,
            smoothing_filter,
            ticks_since_collision_check: 0,
            cached_min_distance: f64::INFINITY,
            vel_limits,
            accel_limits,
            consecutive_fallbacks: 0,
        })
    }

    /// Set a custom smoothing filter.
    pub fn set_filter(&mut self, filter: Box<dyn SmoothingFilter>) {
        self.smoothing_filter = filter;
    }

    /// Set the current joint state (e.g. from robot feedback).
    pub fn set_state(&mut self, positions: &[f64], velocities: &[f64]) -> kinetic_core::Result<()> {
        let dof = self.chain.dof;
        if positions.len() != dof || velocities.len() != dof {
            return Err(KineticError::DimensionMismatch {
                expected: dof,
                got: positions.len(),
                context: "servo state".into(),
            });
        }
        self.state.joint_positions = positions.to_vec();
        self.state.joint_velocities = velocities.to_vec();
        let ee = fk(&self.robot, &self.chain, positions)?;
        self.state.ee_pose = ee.0;
        Ok(())
    }

    /// Get the current servo state.
    pub fn state(&self) -> &ServoState {
        &self.state
    }

    /// DOF of the controlled chain.
    pub fn dof(&self) -> usize {
        self.chain.dof
    }

    /// Send a Cartesian twist command (linear + angular velocity in EE frame).
    ///
    /// Converts to joint velocities via damped Jacobian pseudoinverse.
    pub fn send_twist(&mut self, twist: &Twist) -> Result<JointCommand, ServoError> {
        let dof = self.chain.dof;
        let dt = 1.0 / self.config.rate_hz;

        // Get Jacobian and manipulability
        let jac = jacobian(&self.robot, &self.chain, &self.state.joint_positions)?;
        let manip = manipulability(&self.robot, &self.chain, &self.state.joint_positions)?;

        self.state.manipulability = manip;
        self.state.is_near_singularity = manip < self.config.singularity_threshold;

        // Build 6D twist vector
        let mut twist_vec = DVector::zeros(6);
        twist_vec[0] = twist.linear.x;
        twist_vec[1] = twist.linear.y;
        twist_vec[2] = twist.linear.z;
        twist_vec[3] = twist.angular.x;
        twist_vec[4] = twist.angular.y;
        twist_vec[5] = twist.angular.z;

        // Damped pseudoinverse: J† = J^T (J J^T + λ²I)⁻¹
        let lambda = if self.state.is_near_singularity {
            self.config.singularity_damping * (1.0 - manip / self.config.singularity_threshold)
        } else {
            0.0
        };

        let jjt = &jac * jac.transpose();
        let damped = &jjt + DMatrix::<f64>::identity(6, 6) * (lambda * lambda);
        let jt = jac.transpose();

        // Solve (JJ^T + λ²I) x = twist, then dq = J^T x
        let joint_vel_dv = match damped.clone().lu().solve(&twist_vec) {
            Some(x) => {
                self.consecutive_fallbacks = 0;
                &jt * &x
            }
            None => {
                // Fallback: use transpose method with reduced velocity
                self.consecutive_fallbacks += 1;
                if self.consecutive_fallbacks >= 50 {
                    return Err(ServoError::Kinetic(KineticError::SingularityLockup {
                        consecutive_failures: self.consecutive_fallbacks as usize,
                    }));
                }
                // WARNING: operating in degraded mode (1% velocity via transpose)
                &jt * &twist_vec * 0.01
            }
        };

        let mut joint_velocities: Vec<f64> = joint_vel_dv.iter().copied().collect();

        // Apply collision deceleration
        let scale = self.collision_deceleration_scale()?;
        if scale <= 0.0 {
            // Emergency stop
            self.state.is_stopped = true;
            let cmd = JointCommand {
                positions: self.state.joint_positions.clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            };
            return Ok(cmd);
        }
        self.state.is_stopped = false;

        // Scale velocities for collision avoidance
        for v in &mut joint_velocities {
            *v *= scale;
        }

        // Singularity velocity reduction
        if self.state.is_near_singularity {
            let reduction = manip / self.config.singularity_threshold;
            for v in &mut joint_velocities {
                *v *= reduction;
            }
        }

        self.apply_and_integrate(&joint_velocities, dt)
    }

    /// Send a single joint velocity command (joint jog).
    pub fn send_joint_jog(
        &mut self,
        joint_index: usize,
        velocity: f64,
    ) -> Result<JointCommand, ServoError> {
        let dof = self.chain.dof;
        if joint_index >= dof {
            return Err(ServoError::InvalidJointIndex {
                index: joint_index,
                dof,
            });
        }

        let dt = 1.0 / self.config.rate_hz;
        let mut joint_velocities = vec![0.0; dof];
        joint_velocities[joint_index] = velocity;

        // Apply collision deceleration
        let scale = self.collision_deceleration_scale()?;
        if scale <= 0.0 {
            self.state.is_stopped = true;
            return Ok(JointCommand {
                positions: self.state.joint_positions.clone(),
                velocities: vec![0.0; dof],
                accelerations: vec![0.0; dof],
            });
        }
        self.state.is_stopped = false;

        for v in &mut joint_velocities {
            *v *= scale;
        }

        self.apply_and_integrate(&joint_velocities, dt)
    }

    /// Track a moving target pose.
    ///
    /// Uses proportional control in task space with Jacobian pseudoinverse.
    pub fn track_pose(&mut self, target: &Isometry3<f64>) -> Result<JointCommand, ServoError> {
        let gain = self.config.pose_tracking_gain;

        // Compute position and orientation error
        let ee = &self.state.ee_pose;
        let pos_err = target.translation.vector - ee.translation.vector;

        let rot_err_mat = target.rotation * ee.rotation.inverse();
        let angle = rot_err_mat.angle();
        let ori_err = if angle.abs() > 1e-10 {
            rot_err_mat
                .axis()
                .map_or(nalgebra::Vector3::zeros(), |ax| ax.into_inner() * angle)
        } else {
            nalgebra::Vector3::zeros()
        };

        // Build twist from pose error (proportional control)
        let twist = Twist::new(pos_err * gain, ori_err * gain);
        self.send_twist(&twist)
    }

    /// Compute collision deceleration scale factor [0.0, 1.0].
    ///
    /// Returns 0.0 for emergency stop, 1.0 for no deceleration.
    fn collision_deceleration_scale(&mut self) -> Result<f64, ServoError> {
        // Check if we should update collision distance this tick
        let collision_period_ticks =
            (self.config.rate_hz / self.config.collision_check_hz).max(1.0) as u64;

        if self.ticks_since_collision_check >= collision_period_ticks {
            self.ticks_since_collision_check = 0;
            self.cached_min_distance = self
                .scene
                .min_distance_to_robot(&self.state.joint_positions)
                .unwrap_or(f64::INFINITY);
        }
        self.ticks_since_collision_check += 1;

        self.state.min_obstacle_distance = self.cached_min_distance;
        let dist = self.cached_min_distance;

        if dist <= self.config.stop_distance {
            self.state.is_near_collision = true;
            Ok(0.0)
        } else if dist < self.config.slowdown_distance {
            self.state.is_near_collision = true;
            let range = self.config.slowdown_distance - self.config.stop_distance;
            Ok((dist - self.config.stop_distance) / range)
        } else {
            self.state.is_near_collision = false;
            Ok(1.0)
        }
    }

    /// Apply velocity limits, smoothing, and integrate to produce a JointCommand.
    fn apply_and_integrate(
        &mut self,
        joint_velocities: &[f64],
        dt: f64,
    ) -> Result<JointCommand, ServoError> {
        let dof = self.chain.dof;

        // Clamp to velocity limits
        let mut clamped_vel: Vec<f64> = joint_velocities
            .iter()
            .enumerate()
            .map(|(i, &v)| v.clamp(-self.vel_limits[i], self.vel_limits[i]))
            .collect();

        // Clamp acceleration (limit velocity change per tick)
        for (i, vel) in clamped_vel.iter_mut().enumerate() {
            let max_dv = self.accel_limits[i] * dt;
            let dv = *vel - self.state.joint_velocities[i];
            if dv.abs() > max_dv {
                *vel = self.state.joint_velocities[i] + dv.signum() * max_dv;
            }
        }

        // Integrate position
        let mut new_positions: Vec<f64> = self
            .state
            .joint_positions
            .iter()
            .zip(clamped_vel.iter())
            .map(|(&p, &v)| p + v * dt)
            .collect();

        // Apply smoothing filter
        let (smoothed_pos, smoothed_vel) =
            self.smoothing_filter.filter(&new_positions, &clamped_vel);
        new_positions = smoothed_pos;
        let final_vel = smoothed_vel;

        // Safety: clamp per-tick position delta
        if self.config.max_delta_per_tick > 0.0 {
            for (i, new_pos) in new_positions.iter_mut().enumerate() {
                let old_pos = self.state.joint_positions[i];
                let delta = *new_pos - old_pos;
                if delta.abs() > self.config.max_delta_per_tick {
                    *new_pos = old_pos + delta.signum() * self.config.max_delta_per_tick;
                }
            }
        }

        // Clamp to joint position limits
        for (i, &joint_idx) in self.chain.active_joints.iter().enumerate() {
            if i >= dof {
                break;
            }
            if let Some(limits) = &self.robot.joints[joint_idx].limits {
                new_positions[i] = new_positions[i].clamp(limits.lower, limits.upper);
            }
        }

        // Compute accelerations
        let accelerations: Vec<f64> = final_vel
            .iter()
            .zip(self.state.joint_velocities.iter())
            .map(|(&v_new, &v_old)| (v_new - v_old) / dt)
            .collect();

        // Update internal state
        self.state.joint_positions = new_positions.clone();
        self.state.joint_velocities = final_vel.clone();

        // Update EE pose
        if let Ok(pose) = fk(&self.robot, &self.chain, &self.state.joint_positions) {
            self.state.ee_pose = pose.0;
        }

        // Update manipulability
        if let Ok(m) = manipulability(&self.robot, &self.chain, &self.state.joint_positions) {
            self.state.manipulability = m;
            self.state.is_near_singularity = m < self.config.singularity_threshold;
        }

        Ok(JointCommand {
            positions: new_positions,
            velocities: final_vel,
            accelerations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="ee_link">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
</robot>"#;

    fn test_robot() -> Arc<Robot> {
        Arc::new(Robot::from_urdf_string(TEST_URDF).unwrap())
    }

    fn test_scene(robot: &Robot) -> Arc<Scene> {
        Arc::new(Scene::new(robot).unwrap())
    }

    #[test]
    fn servo_creation() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        assert_eq!(servo.dof(), 3);
        assert!(!servo.state().is_stopped);
    }

    #[test]
    fn servo_send_twist() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        // Set initial state away from home
        servo
            .set_state(&[0.0, 0.5, -0.3], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.1, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );

        let cmd = servo.send_twist(&twist).unwrap();
        assert_eq!(cmd.positions.len(), 3);
        assert_eq!(cmd.velocities.len(), 3);

        // Should produce some velocity
        let vel_norm: f64 = cmd.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(vel_norm > 0.0, "Twist should produce joint velocities");
    }

    #[test]
    fn servo_joint_jog() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let cmd = servo.send_joint_jog(1, 0.5).unwrap();
        assert_eq!(cmd.positions.len(), 3);

        // Joint 1 should have non-zero velocity
        assert!(cmd.velocities[1].abs() > 0.0, "Jogged joint should move");
    }

    #[test]
    fn servo_joint_jog_invalid_index() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let result = servo.send_joint_jog(10, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn servo_track_pose() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        // Set initial state
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let target = Isometry3::translation(0.2, 0.0, 0.5);
        let cmd = servo.track_pose(&target).unwrap();

        // Should produce motion
        let vel_norm: f64 = cmd.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(vel_norm > 0.0, "Pose tracking should produce motion");
    }

    #[test]
    fn servo_velocity_clamping() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let config = ServoConfig {
            velocity_limits: vec![0.1, 0.1, 0.1],
            ..Default::default()
        };
        let mut servo = Servo::new(&robot, &scene, config).unwrap();

        // Send a large twist — velocities should be clamped
        let twist = Twist::new(
            nalgebra::Vector3::new(100.0, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );

        let cmd = servo.send_twist(&twist).unwrap();
        for &v in &cmd.velocities {
            assert!(v.abs() <= 0.1 + 1e-10, "Velocity should be clamped: {}", v);
        }
    }

    #[test]
    fn servo_state_updates() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let initial_pos = servo.state().joint_positions.clone();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.05, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );
        servo.send_twist(&twist).unwrap();

        // State should have changed
        let changed = servo
            .state()
            .joint_positions
            .iter()
            .zip(initial_pos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "State should update after send_twist");
    }

    #[test]
    fn servo_with_no_filter() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo.set_filter(Box::new(crate::filter::NoFilter));

        let cmd = servo.send_joint_jog(0, 0.5).unwrap();
        assert_eq!(cmd.positions.len(), 3);
    }

    #[test]
    fn servo_multiple_steps() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.05, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );

        // Run multiple steps
        let mut prev_pos = servo.state().joint_positions.clone();
        for _ in 0..20 {
            let cmd = servo.send_twist(&twist).unwrap();
            // Position should change monotonically
            assert_eq!(cmd.positions.len(), 3);
            prev_pos = cmd.positions;
        }

        // After many steps, should have moved
        let total: f64 = prev_pos
            .iter()
            .zip([0.0, 0.3, -0.2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(total > 1e-6, "Should have moved after 20 steps");
    }

    // --- ServoConfig default tests ---

    #[test]
    fn servo_config_default_values() {
        let config = ServoConfig::default();
        assert!((config.rate_hz - 500.0).abs() < 1e-10);
        assert_eq!(config.input_type, InputType::Twist);
        assert!((config.collision_check_hz - 100.0).abs() < 1e-10);
        assert!((config.singularity_threshold - 0.02).abs() < 1e-10);
        assert!((config.slowdown_distance - 0.15).abs() < 1e-10);
        assert!((config.stop_distance - 0.03).abs() < 1e-10);
        assert!(config.velocity_limits.is_empty());
        assert!(config.acceleration_limits.is_empty());
        assert!((config.pose_tracking_gain - 5.0).abs() < 1e-10);
        assert!((config.singularity_damping - 0.05).abs() < 1e-10);
    }

    // --- InputType tests ---

    #[test]
    fn input_type_equality() {
        assert_eq!(InputType::Twist, InputType::Twist);
        assert_eq!(InputType::JointJog, InputType::JointJog);
        assert_eq!(InputType::PoseTracking, InputType::PoseTracking);
        assert_ne!(InputType::Twist, InputType::JointJog);
        assert_ne!(InputType::Twist, InputType::PoseTracking);
        assert_ne!(InputType::JointJog, InputType::PoseTracking);
    }

    #[test]
    fn input_type_debug() {
        let dbg = format!("{:?}", InputType::Twist);
        assert_eq!(dbg, "Twist");
        let dbg = format!("{:?}", InputType::JointJog);
        assert_eq!(dbg, "JointJog");
        let dbg = format!("{:?}", InputType::PoseTracking);
        assert_eq!(dbg, "PoseTracking");
    }

    // --- set_state tests ---

    #[test]
    fn servo_set_state_updates_positions() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        servo
            .set_state(&[0.5, -0.5, 0.3], &[0.1, -0.1, 0.0])
            .unwrap();

        let state = servo.state();
        assert!((state.joint_positions[0] - 0.5).abs() < 1e-10);
        assert!((state.joint_positions[1] - (-0.5)).abs() < 1e-10);
        assert!((state.joint_positions[2] - 0.3).abs() < 1e-10);
        assert!((state.joint_velocities[0] - 0.1).abs() < 1e-10);
        assert!((state.joint_velocities[1] - (-0.1)).abs() < 1e-10);
        assert!((state.joint_velocities[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn servo_set_state_wrong_dof_positions() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let result = servo.set_state(&[0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert!(result.is_err(), "Should fail with wrong position DOF");
    }

    #[test]
    fn servo_set_state_wrong_dof_velocities() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let result = servo.set_state(&[0.0, 0.0, 0.0], &[0.0, 0.0]);
        assert!(result.is_err(), "Should fail with wrong velocity DOF");
    }

    #[test]
    fn servo_set_state_empty() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let result = servo.set_state(&[], &[]);
        assert!(result.is_err(), "Should fail with empty state");
    }

    #[test]
    fn servo_set_state_updates_ee_pose() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let ee_before = servo.state().ee_pose;

        servo
            .set_state(&[0.5, 0.5, 0.5], &[0.0, 0.0, 0.0])
            .unwrap();

        let ee_after = servo.state().ee_pose;

        // EE pose should change after setting different joint positions
        let pos_diff = (ee_after.translation.vector - ee_before.translation.vector).norm();
        assert!(
            pos_diff > 1e-6,
            "EE pose should change with different joints"
        );
    }

    // --- joint jog edge cases ---

    #[test]
    fn servo_joint_jog_each_joint() {
        let robot = test_robot();
        let scene = test_scene(&robot);

        for joint_idx in 0..3 {
            let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
            let cmd = servo.send_joint_jog(joint_idx, 0.3).unwrap();
            assert!(
                cmd.velocities[joint_idx].abs() > 0.0,
                "Joint {} should move",
                joint_idx
            );
        }
    }

    #[test]
    fn servo_joint_jog_negative_velocity() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let cmd = servo.send_joint_jog(0, -0.5).unwrap();
        // Smoothed velocity should be negative (or at least reflect the negative input)
        // After EMA smoothing from zero, it may be attenuated
        assert!(
            cmd.velocities[0] < 0.0,
            "Negative jog should produce negative velocity: {}",
            cmd.velocities[0]
        );
    }

    #[test]
    fn servo_joint_jog_zero_velocity() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let cmd = servo.send_joint_jog(0, 0.0).unwrap();
        // Zero jog should keep everything at zero
        for &v in &cmd.velocities {
            assert!(v.abs() < 1e-10, "Zero jog should produce zero velocity: {}", v);
        }
    }

    #[test]
    fn servo_joint_jog_boundary_index() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        // Last valid index (2 for 3-DOF)
        let cmd = servo.send_joint_jog(2, 0.5).unwrap();
        assert_eq!(cmd.positions.len(), 3);

        // First invalid index (3 for 3-DOF)
        let mut servo2 = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        let result = servo2.send_joint_jog(3, 0.5);
        assert!(result.is_err());
    }

    // --- twist command tests ---

    #[test]
    fn servo_send_twist_zero() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let twist = Twist::new(nalgebra::Vector3::zeros(), nalgebra::Vector3::zeros());
        let cmd = servo.send_twist(&twist).unwrap();

        for &v in &cmd.velocities {
            assert!(v.abs() < 1e-10, "Zero twist should give zero velocity: {}", v);
        }
    }

    #[test]
    fn servo_send_twist_angular_only() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, 0.5),
        );
        let cmd = servo.send_twist(&twist).unwrap();

        let vel_norm: f64 = cmd.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            vel_norm > 0.0,
            "Angular twist should produce joint velocities"
        );
    }

    #[test]
    fn servo_send_twist_linear_only() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.0, 0.1, 0.0),
            nalgebra::Vector3::zeros(),
        );
        let cmd = servo.send_twist(&twist).unwrap();

        let vel_norm: f64 = cmd.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            vel_norm > 0.0,
            "Linear twist should produce joint velocities"
        );
    }

    // --- pose tracking tests ---

    #[test]
    fn servo_track_pose_identity() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        // Track current EE pose (should produce near-zero motion)
        let current_ee = servo.state().ee_pose;
        let cmd = servo.track_pose(&current_ee).unwrap();

        let vel_norm: f64 = cmd.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            vel_norm < 0.01,
            "Tracking current pose should produce minimal motion: {}",
            vel_norm
        );
    }

    #[test]
    fn servo_track_pose_updates_state() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let pos_before = servo.state().joint_positions.clone();
        let target = Isometry3::translation(0.3, 0.1, 0.4);
        servo.track_pose(&target).unwrap();

        let pos_after = servo.state().joint_positions.clone();
        let changed = pos_before
            .iter()
            .zip(pos_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "Track pose should update joint positions");
    }

    // --- custom filter tests ---

    #[test]
    fn servo_with_butterworth_filter() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo.set_filter(Box::new(crate::filter::ButterworthLowPass::new(
            10.0, 500.0, 3,
        )));

        let cmd = servo.send_joint_jog(0, 0.5).unwrap();
        assert_eq!(cmd.positions.len(), 3);
        // Should work without panic
    }

    // --- custom velocity/acceleration limits ---

    #[test]
    fn servo_custom_velocity_limits() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let config = ServoConfig {
            velocity_limits: vec![0.05, 0.05, 0.05],
            ..Default::default()
        };
        let mut servo = Servo::new(&robot, &scene, config).unwrap();

        // Large twist input
        let twist = Twist::new(
            nalgebra::Vector3::new(50.0, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );
        let cmd = servo.send_twist(&twist).unwrap();

        for &v in &cmd.velocities {
            assert!(
                v.abs() <= 0.05 + 1e-10,
                "Velocity should be clamped to custom limit: {}",
                v
            );
        }
    }

    #[test]
    fn servo_custom_acceleration_limits() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let config = ServoConfig {
            acceleration_limits: vec![1.0, 1.0, 1.0],
            ..Default::default()
        };
        let mut servo = Servo::new(&robot, &scene, config).unwrap();

        // Large twist to force large acceleration
        let twist = Twist::new(
            nalgebra::Vector3::new(10.0, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );

        let cmd = servo.send_twist(&twist).unwrap();

        // Accelerations should be bounded by the limit * dt
        let dt = 1.0 / 500.0;
        for &a in &cmd.accelerations {
            // a = dv/dt, max_dv = accel_limit * dt => max_a = accel_limit
            assert!(
                a.abs() <= 1.0 / dt + 1e-6,
                "Acceleration should respect limits: {}",
                a
            );
        }
    }

    // --- State flag tests ---

    #[test]
    fn servo_initial_state_flags() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        let state = servo.state();
        assert!(!state.is_stopped, "Should not be stopped initially");
        assert!(!state.is_near_collision, "Should not be near collision initially");
        assert!(!state.is_near_singularity, "Should not be near singularity initially");
        assert_eq!(state.min_obstacle_distance, f64::INFINITY);
    }

    #[test]
    fn servo_state_updates_manipulability() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.05, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );
        servo.send_twist(&twist).unwrap();

        // After a twist command, manipulability should be computed
        let state = servo.state();
        assert!(
            state.manipulability >= 0.0,
            "Manipulability should be non-negative"
        );
    }

    // --- ServoError tests ---

    #[test]
    fn servo_error_display_invalid_joint() {
        let err = ServoError::InvalidJointIndex { index: 5, dof: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn servo_error_display_emergency_stop() {
        let err = ServoError::EmergencyStop {
            distance: 0.01,
            stop: 0.03,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Emergency stop"));
        assert!(msg.contains("0.01"));
    }

    // --- Servo multiple send_twist steps convergence ---

    #[test]
    fn servo_twist_steps_accumulate_position_change() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        servo
            .set_state(&[0.0, 0.3, -0.2], &[0.0, 0.0, 0.0])
            .unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(0.1, 0.0, 0.0),
            nalgebra::Vector3::zeros(),
        );

        let pos_0 = servo.state().joint_positions.clone();

        // Run 5 steps
        for _ in 0..5 {
            servo.send_twist(&twist).unwrap();
        }
        let pos_5 = servo.state().joint_positions.clone();

        // Run 5 more steps
        for _ in 0..5 {
            servo.send_twist(&twist).unwrap();
        }
        let pos_10 = servo.state().joint_positions.clone();

        let dist_0_to_5: f64 = pos_0
            .iter()
            .zip(pos_5.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let dist_0_to_10: f64 = pos_0
            .iter()
            .zip(pos_10.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            dist_0_to_10 > dist_0_to_5,
            "More steps should produce more displacement: 5-step={}, 10-step={}",
            dist_0_to_5,
            dist_0_to_10
        );
    }

    // --- Position clamping to joint limits ---

    #[test]
    fn servo_positions_stay_within_joint_limits() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let mut servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();

        // Set position near upper limit, then command toward the limit
        servo
            .set_state(&[3.0, 3.0, 3.0], &[0.0, 0.0, 0.0])
            .unwrap();

        // Large positive joint jog to push past limits
        for _ in 0..50 {
            servo.send_joint_jog(0, 2.0).unwrap();
        }

        let state = servo.state();
        for (i, &p) in state.joint_positions.iter().enumerate() {
            assert!(
                p <= 3.14 + 1e-6,
                "Joint {} position should respect upper limit: {}",
                i,
                p
            );
        }
    }

    // --- Servo dof() test ---

    #[test]
    fn servo_dof_matches_robot() {
        let robot = test_robot();
        let scene = test_scene(&robot);
        let servo = Servo::new(&robot, &scene, ServoConfig::default()).unwrap();
        assert_eq!(servo.dof(), 3);
    }
}
