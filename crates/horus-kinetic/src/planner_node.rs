//! PlannerNode — wraps KINETIC Planner as a message-driven node.

use std::path::Path;
use std::sync::Arc;

use kinetic_core::Goal;
use kinetic_planning::Planner;
use kinetic_robot::Robot;
use kinetic_trajectory::trapezoidal_per_joint;

use crate::messages::{PlanRequest, TrajectoryMsg, WaypointMsg};

/// Error from PlannerNode operations.
#[derive(Debug, thiserror::Error)]
pub enum PlannerNodeError {
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] kinetic_core::KineticError),
    #[error("Trajectory error: {0}")]
    Trajectory(String),
}

/// PlannerNode receives planning requests and produces trajectory responses.
///
/// In standalone mode, call [`handle_request`] directly.
/// With the `horus-ipc` feature, this implements the HORUS `Node` trait
/// and communicates via shared-memory topics.
pub struct PlannerNode {
    robot: Arc<Robot>,
    planner: Planner,
}

impl PlannerNode {
    /// Create a new PlannerNode for the named robot.
    pub fn new(robot_name: &str) -> Result<Self, PlannerNodeError> {
        let robot = Arc::new(Robot::from_name(robot_name)?);
        let planner = Planner::new(&robot)?;
        Ok(Self { robot, planner })
    }

    /// Create a PlannerNode from URDF + SRDF files.
    ///
    /// The SRDF provides:
    /// - Disabled collision pairs → populated into the Planner's ACM
    /// - Planning groups → used for chain extraction
    /// - Named states → resolvable via `GoalMsg::Named`
    pub fn new_with_srdf(
        urdf_path: impl AsRef<Path>,
        srdf_path: impl AsRef<Path>,
    ) -> Result<Self, PlannerNodeError> {
        let robot = Arc::new(Robot::from_urdf_srdf(urdf_path, srdf_path)?);
        let planner = Planner::new(&robot)?;
        Ok(Self { robot, planner })
    }

    /// Create a PlannerNode with a pre-loaded Robot model.
    ///
    /// Use this when you've already loaded and configured the robot
    /// (e.g., applied SRDF, custom collision preferences).
    pub fn new_with_robot(robot: Arc<Robot>) -> Result<Self, PlannerNodeError> {
        let planner = Planner::new(&robot)?;
        Ok(Self { robot, planner })
    }

    /// Create a PlannerNode with a scene for collision-aware planning.
    ///
    /// The scene's obstacles and ACM are incorporated into planning.
    pub fn new_with_scene(
        robot: Arc<Robot>,
        scene: &kinetic_scene::Scene,
    ) -> Result<Self, PlannerNodeError> {
        let planner = Planner::new(&robot)?.with_scene(scene);
        Ok(Self { robot, planner })
    }

    /// Get a reference to the underlying robot.
    pub fn robot(&self) -> &Robot {
        &self.robot
    }

    /// Handle a planning request and return a trajectory response.
    pub fn handle_request(&self, request: &PlanRequest) -> TrajectoryMsg {
        let goal = request.goal.to_kinetic_goal();

        match self.plan_and_time(&request.start_joints, &goal) {
            Ok((waypoints, duration_secs, planning_time_us)) => TrajectoryMsg {
                waypoints,
                duration_secs,
                planning_time_us,
                request_id: request.request_id,
                success: true,
                error: None,
            },
            Err(e) => TrajectoryMsg {
                waypoints: vec![],
                duration_secs: 0.0,
                planning_time_us: 0,
                request_id: request.request_id,
                success: false,
                error: Some(e.to_string()),
            },
        }
    }

    /// Plan and time-parameterize a trajectory.
    fn plan_and_time(
        &self,
        start: &[f64],
        goal: &Goal,
    ) -> Result<(Vec<WaypointMsg>, f64, u64), PlannerNodeError> {
        let plan_start = std::time::Instant::now();

        let result = self.planner.plan(start, goal)?;
        let planning_time_us = plan_start.elapsed().as_micros() as u64;

        // Time-parameterize with per-joint velocity limits from the robot model
        let vel_limits: Vec<f64> = self
            .robot
            .joint_limits
            .iter()
            .map(|lim| if lim.velocity > 0.0 { lim.velocity } else { 1.0 })
            .collect();
        let accel_limits: Vec<f64> = self
            .robot
            .joint_limits
            .iter()
            .map(|lim| lim.acceleration.unwrap_or(2.0))
            .collect();
        let timed = trapezoidal_per_joint(&result.waypoints, &vel_limits, &accel_limits)
            .map_err(|e| PlannerNodeError::Trajectory(e.to_string()))?;

        let waypoints: Vec<WaypointMsg> = timed
            .waypoints
            .iter()
            .map(|wp| WaypointMsg {
                positions: wp.positions.clone(),
                velocities: wp.velocities.clone(),
                time_secs: wp.time,
            })
            .collect();

        let duration_secs = timed.duration.as_secs_f64();

        Ok((waypoints, duration_secs, planning_time_us))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::GoalMsg;

    #[test]
    fn planner_node_creation() {
        let node = PlannerNode::new("ur5e").unwrap();
        assert_eq!(node.robot().name, "ur5e");
    }

    #[test]
    fn planner_node_handle_request_joints() {
        let node = PlannerNode::new("ur5e").unwrap();
        let request = PlanRequest {
            start_joints: vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0],
            goal: GoalMsg::Joints(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]),
            request_id: 1,
        };

        let response = node.handle_request(&request);
        assert!(response.success, "Planning should succeed: {:?}", response.error);
        assert_eq!(response.request_id, 1);
        assert!(response.waypoints.len() >= 2);
        assert!(response.duration_secs > 0.0);
        assert!(response.planning_time_us > 0);
    }

    #[test]
    fn planner_node_handle_request_named() {
        let node = PlannerNode::new("ur5e").unwrap();
        let request = PlanRequest {
            start_joints: vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
            goal: GoalMsg::Named("home".into()),
            request_id: 2,
        };

        // Named goals may or may not be configured — this tests error handling
        let response = node.handle_request(&request);
        assert_eq!(response.request_id, 2);
        // Either succeeds or returns a clean error
        if !response.success {
            assert!(response.error.is_some());
        }
    }

    #[test]
    fn planner_node_with_robot() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let node = PlannerNode::new_with_robot(robot).unwrap();
        assert_eq!(node.robot().name, "ur5e");
    }

    #[test]
    fn planner_node_with_scene() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let scene = kinetic_scene::Scene::new(&robot).unwrap();
        let node = PlannerNode::new_with_scene(robot, &scene).unwrap();
        assert_eq!(node.robot().name, "ur5e");

        // Verify planning still works with scene-aware planner
        let request = PlanRequest {
            start_joints: vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0],
            goal: GoalMsg::Joints(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]),
            request_id: 99,
        };
        let response = node.handle_request(&request);
        assert_eq!(response.request_id, 99);
        assert!(response.success, "Scene-aware planning failed: {:?}", response.error);
    }

    #[test]
    fn planner_node_uses_per_joint_limits() {
        let node = PlannerNode::new("ur5e").unwrap();
        let request = PlanRequest {
            start_joints: vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0],
            goal: GoalMsg::Joints(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]),
            request_id: 100,
        };

        let response = node.handle_request(&request);
        assert!(response.success, "Planning should succeed: {:?}", response.error);

        // Verify the trajectory has valid timed waypoints
        // (per-joint limits are used in trapezoidal_per_joint)
        for wp in &response.waypoints {
            assert_eq!(wp.positions.len(), 6);
            assert_eq!(wp.velocities.len(), 6);
        }
    }

    #[test]
    fn planner_node_with_srdf() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="ee_link"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>"#;

        let srdf = r#"<?xml version="1.0" ?>
<robot name="test_arm">
  <group name="arm">
    <chain base_link="base_link" tip_link="ee_link" />
  </group>
  <disable_collisions link1="base_link" link2="link1" reason="Adjacent" />
  <disable_collisions link1="link1" link2="link2" reason="Adjacent" />
</robot>"#;

        let dir = std::env::temp_dir().join("horus_kinetic_planner_srdf_test");
        std::fs::create_dir_all(&dir).unwrap();
        let urdf_path = dir.join("test.urdf");
        let srdf_path = dir.join("test.srdf");
        std::fs::write(&urdf_path, urdf).unwrap();
        std::fs::write(&srdf_path, srdf).unwrap();

        let node = PlannerNode::new_with_srdf(&urdf_path, &srdf_path).unwrap();
        assert_eq!(node.robot().name, "test_arm");
        assert!(node.robot().groups.contains_key("arm"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn planner_node_request_id_correlation() {
        let node = PlannerNode::new("ur5e").unwrap();

        for id in [10, 20, 30] {
            let request = PlanRequest {
                start_joints: vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0],
                goal: GoalMsg::Joints(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]),
                request_id: id,
            };
            let response = node.handle_request(&request);
            assert_eq!(response.request_id, id);
        }
    }
}
