//! ServoNode — wraps KINETIC Servo as a message-driven real-time node.

use std::sync::Arc;

use nalgebra::Vector3;

use kinetic_core::Twist;
use kinetic_reactive::servo::{Servo, ServoConfig};
use kinetic_robot::Robot;
use kinetic_scene::Scene;

use crate::messages::{JointCommandMsg, JointJogMsg, ServoStateMsg, TwistMsg};

/// Error from ServoNode operations.
#[derive(Debug, thiserror::Error)]
pub enum ServoNodeError {
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] kinetic_core::KineticError),
    #[error("Servo error: {0}")]
    Servo(#[from] kinetic_reactive::servo::ServoError),
}

/// ServoNode provides real-time servo control at high rates (500Hz+).
///
/// Accepts twist (Cartesian velocity) or joint jog commands and
/// outputs joint position/velocity commands suitable for robot drivers.
pub struct ServoNode {
    robot: Arc<Robot>,
    servo: Servo,
    config: ServoConfig,
}

impl ServoNode {
    /// Create a new ServoNode for the named robot.
    pub fn new(robot_name: &str) -> Result<Self, ServoNodeError> {
        let robot = Arc::new(Robot::from_name(robot_name)?);
        let config = ServoConfig::default();
        let scene = Arc::new(Scene::new(&robot)?);
        let servo = Servo::new(&robot, &scene, config.clone())?;

        Ok(Self {
            robot,
            servo,
            config,
        })
    }

    /// Create with custom servo configuration.
    pub fn with_config(robot_name: &str, config: ServoConfig) -> Result<Self, ServoNodeError> {
        let robot = Arc::new(Robot::from_name(robot_name)?);
        let scene = Arc::new(Scene::new(&robot)?);
        let servo = Servo::new(&robot, &scene, config.clone())?;

        Ok(Self {
            robot,
            servo,
            config,
        })
    }

    /// Handle a twist command and return joint commands.
    pub fn handle_twist(&mut self, twist_msg: &TwistMsg) -> Result<JointCommandMsg, ServoNodeError> {
        let twist = Twist::new(
            Vector3::new(twist_msg.linear[0], twist_msg.linear[1], twist_msg.linear[2]),
            Vector3::new(twist_msg.angular[0], twist_msg.angular[1], twist_msg.angular[2]),
        );

        let result = self.servo.send_twist(&twist)?;

        Ok(JointCommandMsg {
            positions: result.positions,
            velocities: result.velocities,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        })
    }

    /// Handle a joint jog command and return joint commands.
    ///
    /// Jogs each joint by the specified velocity for one servo tick.
    pub fn handle_joint_jog(
        &mut self,
        jog: &JointJogMsg,
    ) -> Result<JointCommandMsg, ServoNodeError> {
        // Jog each joint individually, accumulating the result
        let mut last_cmd = None;
        for (idx, &vel) in jog.velocities.iter().enumerate() {
            if vel.abs() > 1e-10 {
                let cmd = self.servo.send_joint_jog(idx, vel)?;
                last_cmd = Some(cmd);
            }
        }

        let cmd = match last_cmd {
            Some(cmd) => cmd,
            None => {
                // No joints moving — return current state
                let state = self.servo.state();
                kinetic_reactive::JointCommand {
                    positions: state.joint_positions.clone(),
                    velocities: state.joint_velocities.clone(),
                    accelerations: vec![0.0; state.joint_positions.len()],
                }
            }
        };

        Ok(JointCommandMsg {
            positions: cmd.positions,
            velocities: cmd.velocities,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        })
    }

    /// Get current servo state for publishing.
    pub fn get_state(&self) -> ServoStateMsg {
        let state = self.servo.state();
        let t = state.ee_pose.translation.vector;
        let q = state.ee_pose.rotation;

        ServoStateMsg {
            ee_position: [t.x, t.y, t.z],
            ee_orientation: [q.i, q.j, q.k, q.w],
            min_obstacle_distance: state.min_obstacle_distance,
            manipulability: state.manipulability,
            near_singularity: state.is_near_singularity,
            collision_avoidance_active: state.is_near_collision,
        }
    }

    /// Get the servo rate in Hz.
    pub fn rate_hz(&self) -> f64 {
        self.config.rate_hz
    }

    /// Get a reference to the underlying robot.
    pub fn robot(&self) -> &Robot {
        &self.robot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn servo_node_creation() {
        let node = ServoNode::new("ur5e").unwrap();
        assert_eq!(node.robot().name, "ur5e");
        assert!((node.rate_hz() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn servo_node_twist_command() {
        let mut node = ServoNode::new("ur5e").unwrap();

        let twist = TwistMsg {
            linear: [0.0, 0.0, -0.01],
            angular: [0.0, 0.0, 0.0],
        };

        let cmd = node.handle_twist(&twist).unwrap();
        assert_eq!(cmd.positions.len(), 6);
        assert_eq!(cmd.velocities.len(), 6);
        assert!(cmd.timestamp_ns > 0);
    }

    #[test]
    fn servo_node_state() {
        let node = ServoNode::new("ur5e").unwrap();
        let state = node.get_state();
        // Orientation should not be all zeros (at least w should be nonzero)
        assert!(
            state.ee_orientation.iter().any(|&v| v.abs() > 1e-10),
            "Orientation should have nonzero components"
        );
    }
}
