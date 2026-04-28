//! Reactive control for KINETIC.
//!
//! Provides two main subsystems:
//!
//! - **RMP** (Riemannian Motion Policies): Combine multiple objectives
//!   (reaching, avoidance, limits) via metric-weighted averaging.
//! - **Servo**: Teleoperation with twist, joint jog, and pose tracking modes,
//!   including collision deceleration and singularity avoidance.
//!
//! # RMP Algorithm
//!
//! Each policy defines a desired acceleration and Riemannian metric in its
//! task space. Policies are pulled back to joint space via the Jacobian and
//! combined via metric-weighted averaging:
//!
//! ```text
//! a_combined = (Σ Mᵢ)⁻¹ Σ(Mᵢ aᵢ)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kinetic_reactive::{RMP, PolicyType};
//!
//! let mut rmp = RMP::new(&robot);
//! rmp.add(PolicyType::ReachTarget { target_pose, gain: 10.0 });
//! rmp.add(PolicyType::Damping { coefficient: 0.5 });
//! let cmd = rmp.compute(&joints, &velocities, 0.002)?;
//! ```

pub mod filter;
mod policy;
pub mod servo;

pub use filter::{ButterworthLowPass, ExponentialMovingAverage, NoFilter, SmoothingFilter};
pub use servo::{InputType, Servo, ServoConfig, ServoError, ServoState};

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Isometry3};

use kinetic_core::{KineticError, Pose};
use kinetic_kinematics::{fk, jacobian, manipulability, KinematicChain};
use kinetic_robot::Robot;
use kinetic_scene::Scene;

/// Joint-space command output from RMP computation.
#[derive(Debug, Clone)]
pub struct JointCommand {
    /// Desired joint positions.
    pub positions: Vec<f64>,
    /// Desired joint velocities.
    pub velocities: Vec<f64>,
    /// Computed joint accelerations.
    pub accelerations: Vec<f64>,
}

/// Robot state at a given instant (used internally by RMP).
#[derive(Debug, Clone)]
pub(crate) struct RobotState {
    /// Current joint positions.
    pub joint_positions: Vec<f64>,
    /// Current joint velocities.
    pub joint_velocities: Vec<f64>,
    /// End-effector pose in world frame.
    pub ee_pose: Pose,
    /// Jacobian matrix (6 x DOF).
    pub jacobian: DMatrix<f64>,
    /// Yoshikawa manipulability index.
    pub manipulability: f64,
}

/// Policy type variants for reactive control.
#[derive(Clone)]
pub enum PolicyType {
    /// Attract end-effector toward a target pose.
    ReachTarget {
        target_pose: Isometry3<f64>,
        gain: f64,
    },
    /// Repel robot from scene obstacles.
    AvoidObstacles {
        scene: Arc<Scene>,
        influence_distance: f64,
        gain: f64,
    },
    /// Repel self-collision between robot links.
    AvoidSelfCollision { gain: f64 },
    /// Soft repulsion from joint limits.
    JointLimitAvoidance { margin: f64, gain: f64 },
    /// Slow down near singular configurations.
    SingularityAvoidance { threshold: f64, gain: f64 },
    /// Velocity damping for stability.
    Damping { coefficient: f64 },
}

impl std::fmt::Debug for PolicyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyType::ReachTarget { gain, .. } => {
                f.debug_struct("ReachTarget").field("gain", gain).finish()
            }
            PolicyType::AvoidObstacles {
                influence_distance,
                gain,
                ..
            } => f
                .debug_struct("AvoidObstacles")
                .field("influence_distance", influence_distance)
                .field("gain", gain)
                .finish(),
            PolicyType::AvoidSelfCollision { gain } => f
                .debug_struct("AvoidSelfCollision")
                .field("gain", gain)
                .finish(),
            PolicyType::JointLimitAvoidance { margin, gain } => f
                .debug_struct("JointLimitAvoidance")
                .field("margin", margin)
                .field("gain", gain)
                .finish(),
            PolicyType::SingularityAvoidance { threshold, gain } => f
                .debug_struct("SingularityAvoidance")
                .field("threshold", threshold)
                .field("gain", gain)
                .finish(),
            PolicyType::Damping { coefficient } => f
                .debug_struct("Damping")
                .field("coefficient", coefficient)
                .finish(),
        }
    }
}

/// Riemannian Motion Policy combiner.
///
/// Combines multiple policies via metric-weighted averaging in joint space.
pub struct RMP {
    robot: Arc<Robot>,
    chain: KinematicChain,
    policies: Vec<PolicyType>,
}

impl RMP {
    /// Create a new RMP controller for the given robot.
    ///
    /// Auto-detects the kinematic chain from planning groups or URDF tree.
    pub fn new(robot: &Arc<Robot>) -> kinetic_core::Result<Self> {
        let chain = auto_detect_chain(robot)?;
        Ok(RMP {
            robot: Arc::clone(robot),
            chain,
            policies: Vec::new(),
        })
    }

    /// Create with a specific kinematic chain.
    pub fn with_chain(robot: &Arc<Robot>, chain: KinematicChain) -> Self {
        RMP {
            robot: Arc::clone(robot),
            chain,
            policies: Vec::new(),
        }
    }

    /// Add a policy to the RMP controller.
    pub fn add(&mut self, policy: PolicyType) {
        self.policies.push(policy);
    }

    /// Clear all policies.
    pub fn clear(&mut self) {
        self.policies.clear();
    }

    /// Number of policies.
    pub fn num_policies(&self) -> usize {
        self.policies.len()
    }

    /// DOF of the kinematic chain.
    pub fn dof(&self) -> usize {
        self.chain.dof
    }

    /// Compute combined joint command from all policies.
    ///
    /// `current_joints`: current joint positions (length = DOF of chain).
    /// `current_velocities`: current joint velocities (length = DOF of chain).
    /// `dt`: timestep in seconds.
    pub fn compute(
        &self,
        current_joints: &[f64],
        current_velocities: &[f64],
        dt: f64,
    ) -> kinetic_core::Result<JointCommand> {
        let dof = self.chain.dof;

        if current_joints.len() != dof || current_velocities.len() != dof {
            return Err(KineticError::DimensionMismatch {
                expected: dof,
                got: current_joints.len(),
                context: "RMP state".into(),
            });
        }

        // Build robot state
        let ee_pose = fk(&self.robot, &self.chain, current_joints)?;
        let jac = jacobian(&self.robot, &self.chain, current_joints)?;
        let manip = manipulability(&self.robot, &self.chain, current_joints)?;

        let state = RobotState {
            joint_positions: current_joints.to_vec(),
            joint_velocities: current_velocities.to_vec(),
            ee_pose,
            jacobian: jac,
            manipulability: manip,
        };

        // Accumulate metric-weighted acceleration in joint space
        let mut total_metric = DMatrix::zeros(dof, dof);
        let mut total_weighted_accel = DVector::zeros(dof);

        for policy in &self.policies {
            let (accel, metric) = evaluate_policy(policy, &state, &self.robot, &self.chain)?;
            total_metric += &metric;
            total_weighted_accel += metric * accel;
        }

        // Solve: a_combined = total_metric^{-1} * total_weighted_accel
        // Use pseudo-inverse for numerical stability
        let accelerations = if total_metric.norm() > 1e-12 {
            let svd = total_metric.svd(true, true);
            svd.solve(&total_weighted_accel, 1e-8)
                .unwrap_or_else(|_| DVector::zeros(dof))
        } else {
            DVector::zeros(dof)
        };

        // Integrate: velocity += accel * dt, position += velocity * dt
        let mut new_velocities = Vec::with_capacity(dof);
        let mut new_positions = Vec::with_capacity(dof);
        let accel_vec: Vec<f64> = accelerations.iter().copied().collect();

        for i in 0..dof {
            let new_vel = current_velocities[i] + accel_vec[i] * dt;
            let new_pos = current_joints[i] + new_vel * dt;
            new_velocities.push(new_vel);
            new_positions.push(new_pos);
        }

        // Clamp to joint limits and velocity limits
        clamp_to_limits(
            &self.robot,
            &self.chain,
            &mut new_positions,
            &mut new_velocities,
        );

        Ok(JointCommand {
            positions: new_positions,
            velocities: new_velocities,
            accelerations: accel_vec,
        })
    }
}

/// Evaluate a single policy, returning (joint_acceleration, joint_metric).
fn evaluate_policy(
    policy: &PolicyType,
    state: &RobotState,
    robot: &Robot,
    chain: &KinematicChain,
) -> kinetic_core::Result<(DVector<f64>, DMatrix<f64>)> {
    let dof = chain.dof;
    match policy {
        PolicyType::ReachTarget { target_pose, gain } => {
            policy::evaluate_reach_target(state, target_pose, *gain)
        }
        PolicyType::AvoidObstacles {
            scene,
            influence_distance,
            gain,
        } => policy::evaluate_avoid_obstacles(state, scene, *influence_distance, *gain, dof),
        PolicyType::AvoidSelfCollision { gain } => {
            policy::evaluate_avoid_self_collision(state, *gain, dof)
        }
        PolicyType::JointLimitAvoidance { margin, gain } => {
            policy::evaluate_joint_limit_avoidance(state, robot, chain, *margin, *gain)
        }
        PolicyType::SingularityAvoidance { threshold, gain } => {
            policy::evaluate_singularity_avoidance(state, *threshold, *gain, dof)
        }
        PolicyType::Damping { coefficient } => policy::evaluate_damping(state, *coefficient, dof),
    }
}

/// Clamp positions and velocities to joint limits.
fn clamp_to_limits(
    robot: &Robot,
    chain: &KinematicChain,
    positions: &mut [f64],
    velocities: &mut [f64],
) {
    for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
        if i >= positions.len() {
            break;
        }
        let joint = &robot.joints[joint_idx];
        let limits = match &joint.limits {
            Some(l) => l,
            None => continue,
        };
        let lower = limits.lower;
        let upper = limits.upper;
        let vel_limit = limits.velocity;

        positions[i] = positions[i].clamp(lower, upper);
        velocities[i] = velocities[i].clamp(-vel_limit, vel_limit);
    }
}

/// Auto-detect kinematic chain from robot.
fn auto_detect_chain(robot: &Robot) -> kinetic_core::Result<KinematicChain> {
    if let Some((_, group)) = robot.groups.iter().next() {
        return KinematicChain::extract(robot, &group.base_link, &group.tip_link);
    }

    if robot.links.is_empty() {
        return Err(KineticError::NoLinks);
    }

    let root_name = &robot.links[0].name;
    let mut has_child = vec![false; robot.links.len()];
    for joint in &robot.joints {
        has_child[joint.parent_link] = true;
    }

    let mut best_leaf = robot.links.len() - 1;
    let mut best_depth = 0;
    for (i, _) in robot.links.iter().enumerate() {
        if has_child[i] {
            continue;
        }
        let mut depth = 0;
        let mut current = i;
        while current != 0 {
            if let Some(joint_idx) = robot.links[current].parent_joint {
                depth += 1;
                current = robot.joints[joint_idx].parent_link;
            } else {
                break;
            }
        }
        if depth > best_depth {
            best_depth = depth;
            best_leaf = i;
        }
    }

    let tip_name = &robot.links[best_leaf].name;
    KinematicChain::extract(robot, root_name, tip_name)
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

    #[test]
    fn rmp_creation() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();
        assert_eq!(rmp.dof(), 3);
        assert_eq!(rmp.num_policies(), 0);
    }

    #[test]
    fn rmp_add_policies() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        rmp.add(PolicyType::Damping { coefficient: 0.5 });
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.1,
            gain: 15.0,
        });
        assert_eq!(rmp.num_policies(), 2);

        rmp.clear();
        assert_eq!(rmp.num_policies(), 0);
    }

    #[test]
    fn rmp_compute_damping_only() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 1.0 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![1.0, -0.5, 0.3];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Damping should decelerate (accelerations oppose velocities)
        assert!(
            cmd.accelerations[0] < 0.0,
            "Should decelerate positive velocity"
        );
        assert!(
            cmd.accelerations[1] > 0.0,
            "Should decelerate negative velocity"
        );
    }

    #[test]
    fn rmp_compute_reach_target() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        // Target offset from home position to create a position error
        let target = Isometry3::translation(0.2, 0.1, 0.4);
        rmp.add(PolicyType::ReachTarget {
            target_pose: target,
            gain: 10.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 0.5 });

        let joints = vec![0.0, 0.0, 0.0];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Should produce non-zero accelerations toward the target
        let accel_norm: f64 = cmd.accelerations.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(accel_norm > 0.0, "Should produce motion toward target");
    }

    #[test]
    fn rmp_joint_limit_avoidance() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.2,
            gain: 15.0,
        });

        // Joint near upper limit (3.14)
        let joints = vec![3.0, 0.0, 0.0];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Joint 0 is near upper limit — should be pushed back (negative acceleration)
        assert!(
            cmd.accelerations[0] < 0.0,
            "Should push away from upper limit: accel={}",
            cmd.accelerations[0]
        );
    }

    #[test]
    fn rmp_joint_limit_clamping() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 0.1 });

        // Start past joint limits
        let joints = vec![5.0, -5.0, 0.0]; // beyond ±3.14
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Positions should be clamped to [-π, π]
        assert!(
            cmd.positions[0] <= std::f64::consts::PI + 1e-6,
            "Position should be clamped: {}",
            cmd.positions[0]
        );
        assert!(
            cmd.positions[1] >= -std::f64::consts::PI - 1e-6,
            "Position should be clamped: {}",
            cmd.positions[1]
        );
    }

    #[test]
    fn rmp_singularity_avoidance() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::SingularityAvoidance {
            threshold: 0.1,
            gain: 5.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 0.1 });

        let joints = vec![0.0, 0.0, 0.0];
        let velocities = vec![0.5, 0.5, 0.5];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Should compute without error
        assert_eq!(cmd.positions.len(), 3);
        assert_eq!(cmd.velocities.len(), 3);
    }

    #[test]
    fn rmp_wrong_dof() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        let result = rmp.compute(&[0.0, 0.0], &[0.0, 0.0], 0.002);
        assert!(result.is_err(), "Should fail with wrong DOF");
    }

    #[test]
    fn rmp_combined_policies() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        let target = Isometry3::translation(0.2, 0.0, 0.5);
        rmp.add(PolicyType::ReachTarget {
            target_pose: target,
            gain: 10.0,
        });
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.1,
            gain: 15.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 0.5 });

        let joints = vec![0.0, 0.3, -0.2];
        let velocities = vec![0.0, 0.0, 0.0];

        // Simulate a few steps
        let mut pos = joints.clone();
        let mut vel = velocities.clone();

        for _ in 0..10 {
            let cmd = rmp.compute(&pos, &vel, 0.01).unwrap();
            pos = cmd.positions;
            vel = cmd.velocities;
        }

        // Should have moved (not stuck at initial position)
        let moved: f64 = pos.iter().zip(&joints).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            moved > 1e-6,
            "Robot should have moved: total_displacement={}",
            moved
        );
    }

    #[test]
    fn rmp_no_policies_zero_output() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // With no policies, accelerations should be zero
        for &a in &cmd.accelerations {
            assert!(
                a.abs() < 1e-10,
                "No policies should produce zero acceleration: {}",
                a
            );
        }
    }

    // --- PolicyType Debug formatting tests ---

    #[test]
    fn policy_type_debug_reach_target() {
        let policy = PolicyType::ReachTarget {
            target_pose: Isometry3::identity(),
            gain: 10.0,
        };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("ReachTarget"));
        assert!(dbg.contains("10"));
    }

    #[test]
    fn policy_type_debug_avoid_obstacles() {
        let robot = test_robot();
        let scene = Arc::new(kinetic_scene::Scene::new(&robot).unwrap());
        let policy = PolicyType::AvoidObstacles {
            scene,
            influence_distance: 0.3,
            gain: 20.0,
        };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("AvoidObstacles"));
        assert!(dbg.contains("0.3"));
        assert!(dbg.contains("20"));
    }

    #[test]
    fn policy_type_debug_avoid_self_collision() {
        let policy = PolicyType::AvoidSelfCollision { gain: 5.0 };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("AvoidSelfCollision"));
        assert!(dbg.contains("5"));
    }

    #[test]
    fn policy_type_debug_joint_limit_avoidance() {
        let policy = PolicyType::JointLimitAvoidance {
            margin: 0.1,
            gain: 15.0,
        };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("JointLimitAvoidance"));
        assert!(dbg.contains("0.1"));
        assert!(dbg.contains("15"));
    }

    #[test]
    fn policy_type_debug_singularity_avoidance() {
        let policy = PolicyType::SingularityAvoidance {
            threshold: 0.02,
            gain: 5.0,
        };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("SingularityAvoidance"));
        assert!(dbg.contains("0.02"));
        assert!(dbg.contains("5"));
    }

    #[test]
    fn policy_type_debug_damping() {
        let policy = PolicyType::Damping { coefficient: 0.5 };
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("Damping"));
        assert!(dbg.contains("0.5"));
    }

    // --- RMP with individual policy types ---

    #[test]
    fn rmp_avoid_self_collision_damps_velocity() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::AvoidSelfCollision { gain: 10.0 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![1.0, -0.5, 0.3];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Self-collision avoidance uses velocity damping; accelerations should oppose velocities
        assert!(
            cmd.accelerations[0] < 0.0,
            "Should damp positive velocity: {}",
            cmd.accelerations[0]
        );
        assert!(
            cmd.accelerations[1] > 0.0,
            "Should damp negative velocity: {}",
            cmd.accelerations[1]
        );
        assert!(
            cmd.accelerations[2] < 0.0,
            "Should damp positive velocity: {}",
            cmd.accelerations[2]
        );
    }

    #[test]
    fn rmp_singularity_avoidance_zero_accel_away_from_singularity() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        // Very low threshold so we're likely not near singularity at a generic config
        rmp.add(PolicyType::SingularityAvoidance {
            threshold: 1e-10,
            gain: 5.0,
        });

        let joints = vec![0.5, 0.5, 0.5];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // With zero velocity and away from singularity, accelerations should be near zero
        for &a in &cmd.accelerations {
            assert!(
                a.abs() < 1e-6,
                "Away from singularity with zero velocity should give near-zero accel: {}",
                a
            );
        }
    }

    #[test]
    fn rmp_joint_limit_avoidance_near_lower_limit() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.2,
            gain: 15.0,
        });

        // Joint near lower limit (-3.14)
        let joints = vec![-3.0, 0.0, 0.0];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Joint 0 is near lower limit; should be pushed away (positive acceleration)
        assert!(
            cmd.accelerations[0] > 0.0,
            "Should push away from lower limit: accel={}",
            cmd.accelerations[0]
        );
    }

    #[test]
    fn rmp_joint_limit_avoidance_past_limits() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.2,
            gain: 15.0,
        });

        // Joint past upper limit
        let joints = vec![3.2, 0.0, 0.0]; // > 3.14
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Joint 0 is past upper limit; should be pushed back strongly (negative accel)
        assert!(
            cmd.accelerations[0] < 0.0,
            "Should push back from beyond upper limit: accel={}",
            cmd.accelerations[0]
        );
    }

    #[test]
    fn rmp_joint_limit_avoidance_past_lower_limit() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.2,
            gain: 15.0,
        });

        // Joint past lower limit
        let joints = vec![-3.2, 0.0, 0.0]; // < -3.14
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Joint 0 is past lower limit; should be pushed back strongly (positive accel)
        assert!(
            cmd.accelerations[0] > 0.0,
            "Should push back from beyond lower limit: accel={}",
            cmd.accelerations[0]
        );
    }

    #[test]
    fn rmp_damping_zero_velocity_gives_zero_accel() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 1.0 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        for &a in &cmd.accelerations {
            assert!(
                a.abs() < 1e-10,
                "Zero velocity with damping only should give zero accel: {}",
                a
            );
        }
    }

    #[test]
    fn rmp_damping_coefficient_scales_deceleration() {
        let robot = test_robot();

        // Low coefficient
        let mut rmp_low = RMP::new(&robot).unwrap();
        rmp_low.add(PolicyType::Damping { coefficient: 0.1 });

        // High coefficient
        let mut rmp_high = RMP::new(&robot).unwrap();
        rmp_high.add(PolicyType::Damping { coefficient: 10.0 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![1.0, -0.5, 0.3];

        let cmd_low = rmp_low.compute(&joints, &velocities, 0.002).unwrap();
        let cmd_high = rmp_high.compute(&joints, &velocities, 0.002).unwrap();

        // Higher coefficient should produce stronger deceleration
        let accel_norm_low: f64 = cmd_low
            .accelerations
            .iter()
            .map(|a| a * a)
            .sum::<f64>()
            .sqrt();
        let accel_norm_high: f64 = cmd_high
            .accelerations
            .iter()
            .map(|a| a * a)
            .sum::<f64>()
            .sqrt();
        assert!(
            accel_norm_high > accel_norm_low,
            "Higher damping should produce stronger deceleration: high={}, low={}",
            accel_norm_high,
            accel_norm_low
        );
    }

    #[test]
    fn rmp_reach_target_with_angular_error() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        // Target with rotation to create orientation error
        let target = Isometry3::new(
            nalgebra::Vector3::new(0.2, 0.0, 0.5),
            nalgebra::Vector3::new(0.0, 0.5, 0.0), // rotation about Y
        );
        rmp.add(PolicyType::ReachTarget {
            target_pose: target,
            gain: 10.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 0.5 });

        let joints = vec![0.0, 0.0, 0.0];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        let accel_norm: f64 = cmd.accelerations.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            accel_norm > 0.0,
            "Angular error should produce non-zero acceleration"
        );
    }

    // --- RMP wrong DOF edge cases ---

    #[test]
    fn rmp_wrong_dof_velocities_only() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        // Correct positions length, wrong velocities length
        let result = rmp.compute(&[0.0, 0.0, 0.0], &[0.0, 0.0], 0.002);
        assert!(result.is_err(), "Should fail with wrong velocity DOF");
    }

    #[test]
    fn rmp_wrong_dof_positions_only() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        // Wrong positions length, correct velocities length
        let result = rmp.compute(&[0.0, 0.0], &[0.0, 0.0, 0.0], 0.002);
        assert!(result.is_err(), "Should fail with wrong position DOF");
    }

    #[test]
    fn rmp_wrong_dof_too_many() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        let result = rmp.compute(&[0.0; 5], &[0.0; 5], 0.002);
        assert!(result.is_err(), "Should fail with too many joints");
    }

    #[test]
    fn rmp_wrong_dof_empty() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        let result = rmp.compute(&[], &[], 0.002);
        assert!(result.is_err(), "Should fail with empty joints");
    }

    // --- RMP with_chain and utility methods ---

    #[test]
    fn rmp_with_chain_works() {
        let robot = test_robot();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        let rmp = RMP::with_chain(&robot, chain);
        assert_eq!(rmp.dof(), 3);
        assert_eq!(rmp.num_policies(), 0);
    }

    #[test]
    fn rmp_clear_resets_policies() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 0.5 });
        rmp.add(PolicyType::Damping { coefficient: 1.0 });
        rmp.add(PolicyType::Damping { coefficient: 1.5 });
        assert_eq!(rmp.num_policies(), 3);

        rmp.clear();
        assert_eq!(rmp.num_policies(), 0);

        // After clearing, should behave like no policies
        let cmd = rmp
            .compute(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 0.002)
            .unwrap();
        for &a in &cmd.accelerations {
            assert!(a.abs() < 1e-10, "Cleared RMP should give zero accel: {}", a);
        }
    }

    // --- Multi-policy combination tests ---

    #[test]
    fn rmp_combined_reach_and_joint_limits_converges() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        let target = Isometry3::translation(0.1, 0.0, 0.6);
        rmp.add(PolicyType::ReachTarget {
            target_pose: target,
            gain: 10.0,
        });
        rmp.add(PolicyType::JointLimitAvoidance {
            margin: 0.3,
            gain: 20.0,
        });
        rmp.add(PolicyType::SingularityAvoidance {
            threshold: 0.05,
            gain: 3.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 2.0 });

        let mut pos = vec![0.0, 0.0, 0.0];
        let mut vel = vec![0.0, 0.0, 0.0];

        for _ in 0..50 {
            let cmd = rmp.compute(&pos, &vel, 0.01).unwrap();
            pos = cmd.positions;
            vel = cmd.velocities;
        }

        // After 50 steps, all positions should be within joint limits
        for &p in &pos {
            assert!(
                p >= -3.14 - 1e-6 && p <= 3.14 + 1e-6,
                "Position should be within limits: {}",
                p
            );
        }
    }

    #[test]
    fn rmp_combined_damping_and_self_collision() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        rmp.add(PolicyType::AvoidSelfCollision { gain: 5.0 });
        rmp.add(PolicyType::Damping { coefficient: 1.0 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![1.0, -0.5, 0.3];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Both policies damp velocity; acceleration should strongly oppose it
        assert!(
            cmd.accelerations[0] < 0.0,
            "Combined damping should oppose positive velocity"
        );
        assert!(
            cmd.accelerations[1] > 0.0,
            "Combined damping should oppose negative velocity"
        );
    }

    // --- RMP no-policies edge cases ---

    #[test]
    fn rmp_no_policies_with_velocity_preserves_position() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        // Even with velocity, no policies means zero acceleration
        let joints = vec![0.5, -0.5, 0.1];
        let velocities = vec![1.0, -1.0, 0.5];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        for &a in &cmd.accelerations {
            assert!(a.abs() < 1e-10, "Zero-policy accel should be zero: {}", a);
        }

        // Positions should integrate from velocity (velocity*dt since accel is zero)
        for i in 0..3 {
            let expected = joints[i] + velocities[i] * 0.002;
            // May be clamped, but the direction should be right
            assert!(
                (cmd.positions[i] - expected).abs() < 0.1,
                "Position should integrate from velocity"
            );
        }
    }

    #[test]
    fn rmp_no_policies_zero_velocity_unchanged_positions() {
        let robot = test_robot();
        let rmp = RMP::new(&robot).unwrap();

        let joints = vec![0.5, -0.5, 0.1];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();

        // Zero accel + zero velocity = positions should not change
        for (i, (&out, &inp)) in cmd.positions.iter().zip(joints.iter()).enumerate() {
            assert!(
                (out - inp).abs() < 1e-10,
                "Position {} should be unchanged: {} vs {}",
                i,
                out,
                inp
            );
        }
    }

    // --- RMP integration test: multiple timesteps ---

    #[test]
    fn rmp_damping_reduces_velocity_over_time() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 5.0 });

        let mut pos = vec![0.0, 0.5, -0.3];
        let mut vel = vec![1.0, -1.0, 0.5];

        let initial_speed: f64 = vel.iter().map(|v| v * v).sum::<f64>().sqrt();

        for _ in 0..100 {
            let cmd = rmp.compute(&pos, &vel, 0.002).unwrap();
            pos = cmd.positions;
            vel = cmd.velocities;
        }

        let final_speed: f64 = vel.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            final_speed < initial_speed * 0.5,
            "Damping should reduce speed over time: initial={}, final={}",
            initial_speed,
            final_speed
        );
    }

    #[test]
    fn rmp_output_dimensions_match_dof() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::Damping { coefficient: 0.5 });

        let cmd = rmp
            .compute(&[0.0, 0.0, 0.0], &[0.1, -0.1, 0.0], 0.002)
            .unwrap();

        assert_eq!(cmd.positions.len(), 3);
        assert_eq!(cmd.velocities.len(), 3);
        assert_eq!(cmd.accelerations.len(), 3);
    }

    #[test]
    fn rmp_velocity_clamped_to_limits() {
        let robot = test_robot();
        let mut rmp = RMP::new(&robot).unwrap();

        // Very high-gain reach target to produce large velocities
        let target = Isometry3::translation(10.0, 10.0, 10.0);
        rmp.add(PolicyType::ReachTarget {
            target_pose: target,
            gain: 1000.0,
        });

        let joints = vec![0.0, 0.0, 0.0];
        let velocities = vec![0.0, 0.0, 0.0];

        let cmd = rmp.compute(&joints, &velocities, 0.01).unwrap();

        // Velocities should be clamped to robot limits (2.0 rad/s for this URDF)
        for &v in &cmd.velocities {
            assert!(
                v.abs() <= 2.0 + 1e-6,
                "Velocity should be clamped to limit: {}",
                v
            );
        }
    }

    #[test]
    fn rmp_avoid_obstacles_no_scene_objects() {
        // With an empty scene, obstacle avoidance should produce near-zero contribution
        let robot = test_robot();
        let scene = Arc::new(kinetic_scene::Scene::new(&robot).unwrap());

        let mut rmp = RMP::new(&robot).unwrap();
        rmp.add(PolicyType::AvoidObstacles {
            scene,
            influence_distance: 0.3,
            gain: 20.0,
        });
        rmp.add(PolicyType::Damping { coefficient: 0.1 });

        let joints = vec![0.0, 0.5, -0.3];
        let velocities = vec![0.0, 0.0, 0.0];

        // Should compute without error
        let cmd = rmp.compute(&joints, &velocities, 0.002).unwrap();
        assert_eq!(cmd.positions.len(), 3);
    }
}
