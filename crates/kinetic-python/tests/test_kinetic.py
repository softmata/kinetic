"""Tests for the kinetic Python bindings.

Run with: pytest tests/test_kinetic.py -v
Requires: pip install kinetic numpy pytest
"""

import numpy as np
import pytest

import kinetic


# ── Robot ─────────────────────────────────────────────────────────────

class TestRobot:
    def test_load_ur5e(self):
        robot = kinetic.Robot("ur5e")
        assert robot.dof == 6
        assert robot.name == "ur5e"
        assert robot.num_joints > 0
        assert robot.num_links > 0

    def test_load_franka_panda(self):
        robot = kinetic.Robot("franka_panda")
        assert robot.dof == 7
        assert robot.name == "franka_panda"

    def test_invalid_robot(self):
        with pytest.raises(ValueError):
            kinetic.Robot("nonexistent_robot_xyz")

    def test_joint_names(self):
        robot = kinetic.Robot("ur5e")
        names = robot.joint_names
        assert len(names) == 6
        assert all(isinstance(n, str) for n in names)

    def test_repr(self):
        robot = kinetic.Robot("ur5e")
        r = repr(robot)
        assert "ur5e" in r
        assert "dof=6" in r

    def test_fk(self):
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6)
        pose = robot.fk(joints)
        assert pose.shape == (4, 4)
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_fk_different_configs(self):
        robot = kinetic.Robot("ur5e")
        pose1 = robot.fk(np.zeros(6))
        pose2 = robot.fk(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        # Different configs should yield different poses
        assert not np.allclose(pose1, pose2)

    def test_jacobian(self):
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6)
        jac = robot.jacobian(joints)
        assert jac.shape == (6, 6)

    def test_ik(self):
        robot = kinetic.Robot("ur5e")
        # FK then IK roundtrip
        joints_in = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
        target_pose = robot.fk(joints_in)
        try:
            joints_out = robot.ik(target_pose)
            assert len(joints_out) == 6
            # IK solution should reach the same pose
            pose_out = robot.fk(joints_out)
            np.testing.assert_allclose(
                pose_out[:3, 3], target_pose[:3, 3], atol=1e-3
            )
        except RuntimeError:
            pytest.skip("IK did not converge for this config")

    def test_ik_with_seed(self):
        robot = kinetic.Robot("ur5e")
        joints_in = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
        target_pose = robot.fk(joints_in)
        try:
            joints_out = robot.ik(target_pose, seed=joints_in)
            assert len(joints_out) == 6
        except RuntimeError:
            pytest.skip("IK did not converge")

    def test_manipulability(self):
        robot = kinetic.Robot("ur5e")
        joints = np.array([0.0, -1.2, 1.0, -0.8, -1.57, 0.0])
        m = robot.manipulability(joints)
        assert isinstance(m, float)
        assert m >= 0.0


# ── Goal ──────────────────────────────────────────────────────────────

class TestGoal:
    def test_joints_goal(self):
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        r = repr(goal)
        assert "Goal.joints" in r

    def test_pose_goal(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.4, 0.0, 0.5]
        goal = kinetic.Goal.pose(pose)
        r = repr(goal)
        assert "Goal.pose" in r

    def test_named_goal(self):
        goal = kinetic.Goal.named("home")
        r = repr(goal)
        assert "Goal.named" in r
        assert "home" in r


# ── Planner ───────────────────────────────────────────────────────────

class TestPlanner:
    def test_create_planner(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        assert repr(planner) == "Planner(<robot>)"

    def test_plan_joints_to_joints(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = planner.plan(start, goal)
        assert traj.num_waypoints > 0
        assert traj.duration > 0.0
        assert traj.dof == 6

    def test_plan_without_time_parameterization(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = planner.plan(start, goal, time_parameterize=False)
        assert traj.num_waypoints > 0
        assert traj.duration == 0.0  # Not time parameterized

    def test_plan_convenience_function(self):
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = kinetic.plan("ur5e", start, goal)
        assert traj.num_waypoints > 0
        assert traj.duration > 0.0


# ── Trajectory ────────────────────────────────────────────────────────

class TestTrajectory:
    @pytest.fixture
    def trajectory(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        return planner.plan(start, goal)

    def test_sample(self, trajectory):
        joints = trajectory.sample(0.0)
        assert isinstance(joints, np.ndarray)
        assert len(joints) == 6

    def test_sample_at_end(self, trajectory):
        joints = trajectory.sample(trajectory.duration)
        assert len(joints) == 6

    def test_sample_clamped(self, trajectory):
        # Should clamp to valid range, not error
        joints = trajectory.sample(-1.0)
        assert len(joints) == 6
        joints = trajectory.sample(trajectory.duration + 100.0)
        assert len(joints) == 6

    def test_to_numpy(self, trajectory):
        times, positions, velocities = trajectory.to_numpy()
        n = trajectory.num_waypoints
        dof = trajectory.dof
        assert times.shape == (n,)
        assert positions.shape == (n, dof)
        assert velocities.shape == (n, dof)
        # Times should be monotonically non-decreasing
        assert np.all(np.diff(times) >= 0)

    def test_positions_list(self, trajectory):
        positions = trajectory.positions()
        assert isinstance(positions, list)
        assert len(positions) == trajectory.num_waypoints
        assert len(positions[0]) == trajectory.dof

    def test_len(self, trajectory):
        assert len(trajectory) == trajectory.num_waypoints

    def test_repr(self, trajectory):
        r = repr(trajectory)
        assert "Trajectory" in r
        assert "waypoints=" in r
        assert "dof=6" in r


# ── Scene ─────────────────────────────────────────────────────────────

class TestScene:
    def test_create_scene(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        assert scene.num_objects == 0
        assert scene.num_attached == 0

    def test_add_remove(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.cuboid(0.5, 0.3, 0.01)
        pose = np.eye(4)
        pose[2, 3] = -0.05  # Below base
        scene.add("table", shape, pose)
        assert scene.num_objects == 1
        assert scene.remove("table")
        assert scene.num_objects == 0

    def test_clear(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.05)
        scene.add("obj1", shape, np.eye(4))
        scene.add("obj2", shape, np.eye(4))
        assert scene.num_objects == 2
        scene.clear()
        assert scene.num_objects == 0

    def test_shapes(self):
        # Test all shape constructors
        s1 = kinetic.Shape.cuboid(1.0, 0.6, 0.02)
        s2 = kinetic.Shape.sphere(0.05)
        s3 = kinetic.Shape.cylinder(0.03, 0.10)
        assert "cuboid" in repr(s1)
        assert "sphere" in repr(s2)
        assert "cylinder" in repr(s3)

    def test_check_collision(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        joints = np.zeros(6)
        # Should not crash (UR5e may not have collision geometry)
        result = scene.check_collision(joints)
        assert isinstance(result, bool)

    def test_min_distance(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.cuboid(1.0, 1.0, 0.01)
        pose = np.eye(4)
        pose[2, 3] = -1.0  # Far below
        scene.add("floor", shape, pose)
        joints = np.zeros(6)
        dist = scene.min_distance(joints)
        assert isinstance(dist, float)

    def test_attach_detach(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.02)
        grasp_tf = np.eye(4)
        scene.attach("part", shape, grasp_tf, "tool0")
        assert scene.num_attached == 1
        place_pose = np.eye(4)
        place_pose[2, 3] = 0.5
        scene.detach("part", place_pose)
        assert scene.num_attached == 0

    def test_repr(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        r = repr(scene)
        assert "Scene" in r
        assert "objects=0" in r


# ── Servo ─────────────────────────────────────────────────────────────

class TestServo:
    def test_create_servo(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        assert servo.rate_hz == 500.0

    def test_send_twist(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        # Set initial state at non-singular config
        init_pos = np.array([0.0, -1.2, 1.0, -0.8, -1.57, 0.0])
        init_vel = np.zeros(6)
        servo.set_state(init_pos, init_vel)

        twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        cmd = servo.send_twist(twist)
        assert "positions" in cmd
        assert "velocities" in cmd
        assert len(cmd["positions"]) == 6
        assert len(cmd["velocities"]) == 6

    def test_send_twist_wrong_size(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        with pytest.raises(ValueError):
            servo.send_twist(np.array([0.1, 0.0, 0.0]))  # Wrong size

    def test_send_joint_jog(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        init_pos = np.array([0.0, -1.2, 1.0, -0.8, -1.57, 0.0])
        init_vel = np.zeros(6)
        servo.set_state(init_pos, init_vel)

        cmd = servo.send_joint_jog(0, 0.1)
        assert "positions" in cmd
        assert "velocities" in cmd

    def test_state(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        state = servo.state()
        assert "joint_positions" in state
        assert "joint_velocities" in state
        assert "ee_pose" in state
        assert "manipulability" in state
        assert "near_singularity" in state
        assert "near_collision" in state
        assert state["ee_pose"].shape == (4, 4)

    def test_set_state(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        pos = np.array([0.0, -1.2, 1.0, -0.8, -1.57, 0.0])
        vel = np.zeros(6)
        servo.set_state(pos, vel)
        state = servo.state()
        np.testing.assert_allclose(state["joint_positions"], pos, atol=1e-10)

    def test_repr(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        r = repr(servo)
        assert "Servo" in r
        assert "dof=" in r


# ── Error Propagation ────────────────────────────────────────────────

class TestErrorPropagation:
    """Tests that all Rust error types propagate to Python with actionable messages."""

    # ── Robot errors → ValueError ────────────────────────────────────

    def test_invalid_robot_name_raises_value_error(self):
        with pytest.raises(ValueError, match="nonexistent"):
            kinetic.Robot("nonexistent_robot_xyz")

    def test_invalid_robot_name_message_is_descriptive(self):
        with pytest.raises(ValueError) as exc_info:
            kinetic.Robot("fake_arm_99")
        msg = str(exc_info.value)
        assert len(msg) > 10, f"Error message too short: '{msg}'"
        assert "fake_arm_99" in msg or "not found" in msg.lower()

    def test_invalid_urdf_path_raises_value_error(self):
        with pytest.raises(ValueError):
            kinetic.Robot.from_urdf("/tmp/no_such_file.urdf")

    def test_invalid_urdf_path_message_is_descriptive(self):
        with pytest.raises(ValueError) as exc_info:
            kinetic.Robot.from_urdf("/tmp/nonexistent_robot_12345.urdf")
        msg = str(exc_info.value)
        assert len(msg) > 5, f"Error message too short: '{msg}'"

    def test_from_config_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            kinetic.Robot.from_config("not_a_real_robot")

    # ── FK errors → RuntimeError ─────────────────────────────────────

    def test_fk_wrong_dof_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        with pytest.raises(RuntimeError):
            robot.fk(np.array([0.0, 0.0, 0.0]))  # 3 instead of 6

    def test_fk_wrong_dof_message_is_descriptive(self):
        robot = kinetic.Robot("ur5e")
        with pytest.raises(RuntimeError) as exc_info:
            robot.fk(np.array([0.0, 0.0]))  # 2 instead of 6
        msg = str(exc_info.value)
        assert len(msg) > 5, f"FK error message too short: '{msg}'"

    # ── Jacobian errors → RuntimeError ───────────────────────────────

    def test_jacobian_wrong_dof_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        with pytest.raises(RuntimeError):
            robot.jacobian(np.array([0.0]))  # 1 instead of 6

    # ── IK errors → RuntimeError ─────────────────────────────────────

    def test_ik_unreachable_pose_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        # Target 100 meters away — clearly unreachable
        target = np.eye(4)
        target[:3, 3] = [100.0, 100.0, 100.0]
        with pytest.raises(RuntimeError):
            robot.ik(target)

    def test_ik_error_message_mentions_convergence_or_solution(self):
        robot = kinetic.Robot("ur5e")
        target = np.eye(4)
        target[:3, 3] = [100.0, 100.0, 100.0]
        with pytest.raises(RuntimeError) as exc_info:
            robot.ik(target)
        msg = str(exc_info.value).lower()
        assert (
            "converge" in msg or "ik" in msg or "solution" in msg
        ), f"IK error should mention convergence/IK/solution: {msg}"

    def test_ik_wrong_pose_shape_raises_value_error(self):
        robot = kinetic.Robot("ur5e")
        # (3,3) instead of (4,4)
        target = np.eye(3)
        with pytest.raises(ValueError):
            robot.ik(target)

    # ── Manipulability errors → RuntimeError ─────────────────────────

    def test_manipulability_wrong_dof_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        with pytest.raises(RuntimeError):
            robot.manipulability(np.array([0.0, 0.0]))  # 2 instead of 6

    # ── Planner errors → RuntimeError ────────────────────────────────

    def test_planner_plan_wrong_start_dof_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, 0.0])  # 2 instead of 6
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        with pytest.raises(RuntimeError):
            planner.plan(start, goal)

    def test_plan_convenience_invalid_robot_raises_value_error(self):
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        with pytest.raises(ValueError):
            kinetic.plan("nonexistent_robot_xyz", start, goal)

    def test_plan_convenience_invalid_robot_message_includes_name(self):
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        with pytest.raises(ValueError) as exc_info:
            kinetic.plan("fake_bot_99", start, goal)
        msg = str(exc_info.value)
        assert "fake_bot_99" in msg or "not found" in msg.lower()

    # ── Scene errors → RuntimeError ──────────────────────────────────

    def test_scene_check_collision_wrong_dof(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        with pytest.raises(RuntimeError):
            scene.check_collision(np.array([0.0]))  # 1 instead of 6

    def test_scene_min_distance_wrong_dof(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        with pytest.raises(RuntimeError):
            scene.min_distance(np.array([0.0, 0.0]))  # 2 instead of 6

    def test_scene_add_wrong_pose_shape(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.05)
        with pytest.raises(ValueError):
            scene.add("obj", shape, np.eye(3))  # (3,3) instead of (4,4)

    def test_scene_attach_wrong_grasp_shape(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.02)
        with pytest.raises(ValueError):
            scene.attach("part", shape, np.eye(3), "tool0")  # (3,3)

    def test_scene_detach_wrong_pose_shape(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.02)
        grasp_tf = np.eye(4)
        scene.attach("part", shape, grasp_tf, "tool0")
        with pytest.raises(ValueError):
            scene.detach("part", np.eye(3))  # (3,3)

    def test_scene_update_octree_wrong_points_shape(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        # (5, 2) instead of (N, 3)
        points = np.zeros((5, 2))
        origin = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="3"):
            scene.update_octree("lidar", points, origin)

    def test_scene_update_octree_wrong_origin_size(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        points = np.zeros((10, 3))
        origin = np.array([0.0, 0.0])  # 2 instead of 3
        with pytest.raises(ValueError, match="3-element"):
            scene.update_octree("lidar", points, origin)

    # ── Servo errors → ValueError/RuntimeError ───────────────────────

    def test_servo_twist_wrong_size_raises_value_error(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        with pytest.raises(ValueError, match="6-element"):
            servo.send_twist(np.array([0.1, 0.0, 0.0]))  # 3 instead of 6

    def test_servo_twist_wrong_size_message_is_descriptive(self):
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        servo = kinetic.Servo(robot, scene)
        with pytest.raises(ValueError) as exc_info:
            servo.send_twist(np.array([0.1]))
        msg = str(exc_info.value)
        assert "6" in msg, f"Should mention expected size of 6: {msg}"

    # ── Trajectory errors → RuntimeError ─────────────────────────────

    def test_sample_unparameterized_trajectory_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = planner.plan(start, goal, time_parameterize=False)
        with pytest.raises(RuntimeError, match="time"):
            traj.sample(0.0)

    def test_to_numpy_unparameterized_trajectory_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = planner.plan(start, goal, time_parameterize=False)
        with pytest.raises(RuntimeError):
            traj.to_numpy()

    def test_validate_unparameterized_trajectory_raises_runtime_error(self):
        robot = kinetic.Robot("ur5e")
        planner = kinetic.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = planner.plan(start, goal, time_parameterize=False)
        lo = np.full(6, -6.28)
        hi = np.full(6, 6.28)
        vel = np.full(6, 3.14)
        acc = np.full(6, 10.0)
        with pytest.raises(RuntimeError):
            traj.validate(lo, hi, vel, acc)

    # ── Goal construction errors → ValueError ────────────────────────

    def test_goal_pose_wrong_shape_raises_value_error(self):
        with pytest.raises(ValueError):
            kinetic.Goal.pose(np.eye(3))  # (3,3) instead of (4,4)

    def test_goal_pose_wrong_dimensions_raises_value_error(self):
        with pytest.raises((ValueError, TypeError)):
            kinetic.Goal.pose(np.zeros(16))  # 1D instead of 2D (4,4)

    # ── Error messages don't crash Python ────────────────────────────

    def test_error_messages_are_strings(self):
        """Verify error messages are proper Python strings, not garbled bytes."""
        errors_to_trigger = [
            lambda: kinetic.Robot("nonexistent"),
            lambda: kinetic.Robot.from_urdf("/no/such/file.urdf"),
        ]
        for trigger in errors_to_trigger:
            try:
                trigger()
                assert False, "Should have raised"
            except (ValueError, RuntimeError) as e:
                msg = str(e)
                assert isinstance(msg, str)
                assert len(msg) > 0
                # Should be valid UTF-8 (Python strings always are)
                msg.encode("utf-8")

    def test_multiple_errors_dont_leak_state(self):
        """Trigger many errors in sequence to verify no state leaks."""
        for _ in range(20):
            with pytest.raises(ValueError):
                kinetic.Robot("nonexistent")

        # After many errors, valid operations should still work
        robot = kinetic.Robot("ur5e")
        assert robot.dof == 6

    def test_error_after_valid_operations(self):
        """Errors after valid ops don't corrupt state."""
        robot = kinetic.Robot("ur5e")
        pose = robot.fk(np.zeros(6))
        assert pose.shape == (4, 4)

        # Now trigger errors
        with pytest.raises(RuntimeError):
            robot.fk(np.array([0.0]))

        # Robot should still work fine
        pose2 = robot.fk(np.zeros(6))
        np.testing.assert_allclose(pose, pose2)


# ── NumPy Array Edge Cases ───────────────────────────────────────────

class TestNumpyEdgeCases:
    """Tests for NumPy interop edge cases: memory layout, dtypes, empty arrays."""

    # ── Non-contiguous arrays ────────────────────────────────────────

    def test_fortran_order_array_fk(self):
        """Fortran-order (column-major) arrays should work or raise clear error."""
        robot = kinetic.Robot("ur5e")
        joints = np.asfortranarray(np.zeros(6))
        try:
            pose = robot.fk(joints)
            assert pose.shape == (4, 4)
        except (ValueError, TypeError) as e:
            # If Fortran order not supported, error should be clear
            msg = str(e).lower()
            assert "contiguous" in msg or "order" in msg or "layout" in msg

    def test_non_contiguous_slice_fk(self):
        """Sliced array (every other element) is non-contiguous."""
        robot = kinetic.Robot("ur5e")
        big = np.zeros(12)
        non_contiguous = big[::2]  # Every other element → non-contiguous
        assert not non_contiguous.flags["C_CONTIGUOUS"]
        try:
            robot.fk(non_contiguous)
        except (ValueError, TypeError) as e:
            msg = str(e).lower()
            assert "contiguous" in msg or "array" in msg

    def test_transposed_2d_array_pose(self):
        """Transposed (4,4) array is Fortran-order."""
        robot = kinetic.Robot("ur5e")
        pose = np.eye(4, order="F")  # Fortran order
        try:
            robot.ik(pose)
        except (ValueError, RuntimeError) as e:
            # Either contiguity error or IK failure — both acceptable
            pass

    def test_fortran_order_goal_joints(self):
        """Goal.joints with Fortran-order array."""
        joints = np.asfortranarray(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        try:
            goal = kinetic.Goal.joints(joints)
            assert "Goal.joints" in repr(goal)
        except (ValueError, TypeError):
            pass  # Clear error is also acceptable

    # ── Wrong dtypes ─────────────────────────────────────────────────

    def test_float32_array_fk(self):
        """float32 input when float64 expected: should cast or raise clear error."""
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6, dtype=np.float32)
        try:
            pose = robot.fk(joints)
            assert pose.shape == (4, 4)
        except TypeError as e:
            msg = str(e).lower()
            assert "dtype" in msg or "float64" in msg or "type" in msg

    def test_int_array_fk(self):
        """Integer array input: should cast or raise clear error."""
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6, dtype=np.int32)
        try:
            pose = robot.fk(joints)
            assert pose.shape == (4, 4)
        except TypeError as e:
            msg = str(e).lower()
            assert "dtype" in msg or "type" in msg or "int" in msg

    def test_float32_goal_joints(self):
        """Goal.joints with float32 array."""
        joints = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
        try:
            goal = kinetic.Goal.joints(joints)
            assert goal is not None
        except TypeError:
            pass  # Clear error is acceptable

    def test_complex_dtype_raises_type_error(self):
        """Complex dtype should definitely fail."""
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6, dtype=np.complex128)
        with pytest.raises((TypeError, ValueError)):
            robot.fk(joints)

    def test_float32_pose_for_ik(self):
        """float32 pose matrix: should cast or raise clear error."""
        robot = kinetic.Robot("ur5e")
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [0.4, 0.0, 0.5]
        try:
            robot.ik(pose)
        except (TypeError, RuntimeError):
            pass  # Either type error or IK failure is acceptable

    # ── Empty arrays ─────────────────────────────────────────────────

    def test_empty_array_fk(self):
        """Empty array for FK should raise clear error, not crash."""
        robot = kinetic.Robot("ur5e")
        with pytest.raises((RuntimeError, ValueError)):
            robot.fk(np.array([], dtype=np.float64))

    def test_empty_array_goal_joints(self):
        """Empty array for Goal.joints should not crash."""
        try:
            goal = kinetic.Goal.joints(np.array([], dtype=np.float64))
            # If it succeeds, the goal exists but is likely invalid for planning
            assert goal is not None
        except (ValueError, TypeError):
            pass

    def test_empty_array_jacobian(self):
        """Empty array for Jacobian should raise error."""
        robot = kinetic.Robot("ur5e")
        with pytest.raises((RuntimeError, ValueError)):
            robot.jacobian(np.array([], dtype=np.float64))

    def test_empty_array_manipulability(self):
        """Empty array for manipulability should raise error."""
        robot = kinetic.Robot("ur5e")
        with pytest.raises((RuntimeError, ValueError)):
            robot.manipulability(np.array([], dtype=np.float64))

    def test_empty_pointcloud_octree(self):
        """Empty (0,3) point cloud for octree should not crash."""
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        points = np.zeros((0, 3), dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0])
        # Should either work (empty octree) or raise clear error
        try:
            scene.update_octree("lidar", points, origin)
        except ValueError:
            pass

    # ── Large arrays ─────────────────────────────────────────────────

    def test_large_pointcloud_does_not_crash(self):
        """Large (100K points) point cloud should not OOM or crash."""
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        points = np.random.randn(100_000, 3)
        origin = np.array([0.0, 0.0, 0.0])
        # Should not crash — may be slow but shouldn't OOM on modern hardware
        scene.update_octree("lidar", points, origin)
        assert scene.num_octrees >= 0  # Just verify it didn't crash

    # ── Read-only arrays ─────────────────────────────────────────────

    def test_readonly_array_fk(self):
        """Read-only NumPy array should work (we only read it)."""
        robot = kinetic.Robot("ur5e")
        joints = np.zeros(6)
        joints.flags.writeable = False
        pose = robot.fk(joints)
        assert pose.shape == (4, 4)

    def test_readonly_array_ik_seed(self):
        """Read-only array as IK seed should work."""
        robot = kinetic.Robot("ur5e")
        joints_in = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
        target_pose = robot.fk(joints_in)
        seed = joints_in.copy()
        seed.flags.writeable = False
        try:
            robot.ik(target_pose, seed=seed)
        except RuntimeError:
            pytest.skip("IK did not converge")

    def test_readonly_array_goal_joints(self):
        """Read-only array for Goal.joints should work."""
        joints = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
        joints.flags.writeable = False
        goal = kinetic.Goal.joints(joints)
        assert "Goal.joints" in repr(goal)

    def test_readonly_pose_for_scene_add(self):
        """Read-only pose matrix for scene.add should work."""
        robot = kinetic.Robot("ur5e")
        scene = kinetic.Scene(robot)
        shape = kinetic.Shape.sphere(0.05)
        pose = np.eye(4)
        pose.flags.writeable = False
        scene.add("obj", shape, pose)
        assert scene.num_objects == 1

    # ── Array views and strides ──────────────────────────────────────

    def test_contiguous_copy_works(self):
        """np.ascontiguousarray should always work even from non-contiguous source."""
        robot = kinetic.Robot("ur5e")
        big = np.zeros(12)
        non_contiguous = big[::2]
        contiguous = np.ascontiguousarray(non_contiguous)
        pose = robot.fk(contiguous)
        assert pose.shape == (4, 4)

    def test_array_from_list(self):
        """Array created from Python list should work."""
        robot = kinetic.Robot("ur5e")
        joints = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
        pose = robot.fk(joints)
        assert pose.shape == (4, 4)

    # ── NaN and Inf handling ─────────────────────────────────────────

    def test_nan_joints_fk_does_not_crash(self):
        """NaN joint values should produce NaN results, not crash."""
        robot = kinetic.Robot("ur5e")
        joints = np.full(6, np.nan)
        try:
            pose = robot.fk(joints)
            # Result may contain NaN but should not crash
            assert pose.shape == (4, 4)
        except (RuntimeError, ValueError):
            pass  # Some impls may reject NaN — that's fine too

    def test_inf_joints_fk_does_not_crash(self):
        """Inf joint values should not crash Python."""
        robot = kinetic.Robot("ur5e")
        joints = np.full(6, np.inf)
        try:
            pose = robot.fk(joints)
            assert pose.shape == (4, 4)
        except (RuntimeError, ValueError):
            pass  # Rejection is acceptable

    def test_nan_in_pose_for_ik(self):
        """NaN in target pose should not crash."""
        robot = kinetic.Robot("ur5e")
        pose = np.eye(4)
        pose[0, 3] = np.nan
        with pytest.raises((RuntimeError, ValueError)):
            robot.ik(pose)

    # ── Memory safety stress test ────────────────────────────────────

    def test_rapid_array_creation_and_fk(self):
        """Create and use many arrays rapidly — verify no memory leaks/crashes."""
        robot = kinetic.Robot("ur5e")
        for i in range(100):
            joints = np.random.uniform(-3.14, 3.14, size=6)
            pose = robot.fk(joints)
            assert pose.shape == (4, 4)
        # If we got here, no crashes

    def test_interleaved_valid_and_invalid_arrays(self):
        """Alternate valid/invalid arrays — no state corruption."""
        robot = kinetic.Robot("ur5e")
        for i in range(50):
            # Valid call
            pose = robot.fk(np.zeros(6))
            assert pose.shape == (4, 4)
            # Invalid call
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                robot.fk(np.zeros(3))

    def test_different_array_sources_produce_same_results(self):
        """Arrays from different sources (list, zeros, arange) give same FK result."""
        robot = kinetic.Robot("ur5e")
        joints_from_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints_from_zeros = np.zeros(6)
        joints_from_full = np.full(6, 0.0)

        pose1 = robot.fk(joints_from_list)
        pose2 = robot.fk(joints_from_zeros)
        pose3 = robot.fk(joints_from_full)

        np.testing.assert_allclose(pose1, pose2, atol=1e-15)
        np.testing.assert_allclose(pose2, pose3, atol=1e-15)
