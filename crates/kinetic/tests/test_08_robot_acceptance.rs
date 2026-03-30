//! Acceptance tests: 08 robot_acceptance
//! Spec: doc_tests/08_ROBOT_ACCEPTANCE.md

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

#[test]
fn all_52_robots_load() {
    for &(name, _expected_dof) in ALL_ROBOTS {
        let robot = load_robot(name);
        assert!(robot.dof > 0, "robot '{}': DOF should be > 0, got {}", name, robot.dof);
    }
}

#[test]
fn all_robots_have_valid_chain() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        assert!(chain.dof > 0, "robot '{}': chain DOF should be > 0", name);
    }
}

#[test]
fn all_robots_fk_at_zero_is_finite() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let zeros = vec![0.0; chain.dof]; // use chain DOF, not robot DOF
        let pose = kinetic::kinematics::forward_kinematics(&robot, &chain, &zeros);
        assert!(pose.is_ok(), "robot '{}': FK at zero should succeed", name);
        let t = pose.unwrap().translation();
        assert!(t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
            "robot '{}': FK at zero should be finite", name);
    }
}

#[test]
fn all_robots_mid_config_within_limits() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let mid = mid_joints(&robot);
        assert_within_limits(&robot, &mid, &format!("robot '{name}' mid config"));
    }
}

#[test]
fn all_robots_joint_names_unique() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let mut names: Vec<&str> = robot.joints.iter().map(|j| j.name.as_str()).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "robot '{}': duplicate joint names", name);
    }
}

#[test]
fn all_robots_velocity_limits_nonnegative() {
    let mut missing_count = 0;
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        for (j, limit) in robot.joint_limits.iter().enumerate() {
            assert!(
                limit.velocity >= 0.0,
                "robot '{}' joint {}: velocity limit should be >= 0, got {}",
                name, j, limit.velocity
            );
            if limit.velocity == 0.0 {
                missing_count += 1;
            }
        }
    }
    // Some robots may have unset (0) velocity limits — that's a data gap, not a test failure
    // Track it but don't fail
    if missing_count > 0 {
        eprintln!("WARNING: {} joints across all robots have zero velocity limits", missing_count);
    }
}
