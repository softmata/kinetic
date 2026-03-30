"""
Tests for new KINETIC Python APIs added in the execution & runtime roadmap.

Covers: SimExecutor, LogExecutor, FrameTree, Constraint, CartesianConfig,
RMP, Policy, GraspGenerator, GripperType, Task, ik_config, time_parameterize.

Run: cd crates/kinetic-python && maturin develop --release && pytest tests/test_new_apis.py -v
"""

import numpy as np
import pytest


def import_kinetic():
    try:
        import kinetic
        return kinetic
    except ImportError:
        pytest.skip("kinetic Python bindings not built (run: maturin develop)")


# ─── Execution ───


class TestSimExecutor:
    def test_create(self):
        k = import_kinetic()
        ex = k.SimExecutor()
        assert repr(ex) == "SimExecutor()"

    def test_create_custom_rate(self):
        k = import_kinetic()
        ex = k.SimExecutor(rate_hz=100.0)
        assert repr(ex) == "SimExecutor()"

    def test_execute_returns_result(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal)

        ex = k.SimExecutor(rate_hz=100.0)
        result = ex.execute(traj)
        assert result["state"] == "Completed"
        assert result["commands_sent"] > 0
        assert result["actual_duration"] > 0
        assert result["final_positions"].shape == (6,)

    def test_not_time_parameterized_raises(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal, time_parameterize=False)

        ex = k.SimExecutor()
        with pytest.raises(RuntimeError, match="not time-parameterized"):
            ex.execute(traj)


class TestLogExecutor:
    def test_execute_and_commands(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal)

        ex = k.LogExecutor(rate_hz=100.0)
        result = ex.execute(traj)
        assert result["state"] == "Completed"

        cmds = ex.commands()
        assert len(cmds) > 0
        assert "time" in cmds[0]
        assert "positions" in cmds[0]
        assert "velocities" in cmds[0]
        assert cmds[0]["positions"].shape == (6,)

    def test_num_commands(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal)

        ex = k.LogExecutor(rate_hz=50.0)
        ex.execute(traj)
        assert ex.num_commands > 0

    def test_clear(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal)

        ex = k.LogExecutor()
        ex.execute(traj)
        assert ex.num_commands > 0
        ex.clear()
        assert ex.num_commands == 0


# ─── Frame Tree ───


class TestFrameTree:
    def test_create_empty(self):
        k = import_kinetic()
        tree = k.FrameTree()
        assert tree.num_transforms == 0

    def test_set_and_lookup(self):
        k = import_kinetic()
        tree = k.FrameTree()
        pose = np.eye(4)
        pose[0, 3] = 1.5
        tree.set_transform("world", "base", pose, 0.0)
        result = tree.lookup("world", "base")
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result[0, 3], 1.5, atol=1e-10)

    def test_inverse_lookup(self):
        k = import_kinetic()
        tree = k.FrameTree()
        pose = np.eye(4)
        pose[0, 3] = 2.0
        tree.set_transform("A", "B", pose, 0.0)
        result = tree.lookup("B", "A")
        np.testing.assert_allclose(result[0, 3], -2.0, atol=1e-10)

    def test_chain_lookup(self):
        k = import_kinetic()
        tree = k.FrameTree()
        p1 = np.eye(4); p1[0, 3] = 1.0
        p2 = np.eye(4); p2[1, 3] = 2.0
        tree.set_transform("A", "B", p1, 0.0)
        tree.set_transform("B", "C", p2, 0.0)
        result = tree.lookup("A", "C")
        np.testing.assert_allclose(result[0, 3], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 3], 2.0, atol=1e-10)

    def test_static_transform(self):
        k = import_kinetic()
        tree = k.FrameTree()
        tree.set_static("base", "camera", np.eye(4))
        assert tree.has_transform("base", "camera")

    def test_list_frames(self):
        k = import_kinetic()
        tree = k.FrameTree()
        tree.set_transform("X", "Y", np.eye(4), 0.0)
        frames = tree.list_frames()
        assert "X" in frames
        assert "Y" in frames

    def test_update_from_robot(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        tree = k.FrameTree()
        tree.update_from_robot(robot, np.zeros(6), 0.0)
        assert tree.num_transforms > 0

    def test_clear(self):
        k = import_kinetic()
        tree = k.FrameTree()
        tree.set_transform("A", "B", np.eye(4), 0.0)
        assert tree.num_transforms == 1
        tree.clear()
        assert tree.num_transforms == 0

    def test_clear_dynamic(self):
        k = import_kinetic()
        tree = k.FrameTree()
        tree.set_static("A", "B", np.eye(4))
        tree.set_transform("C", "D", np.eye(4), 0.0)
        assert tree.num_transforms == 2
        tree.clear_dynamic()
        assert tree.num_transforms == 1  # only static remains

    def test_no_path_raises(self):
        k = import_kinetic()
        tree = k.FrameTree()
        tree.set_transform("A", "B", np.eye(4), 0.0)
        with pytest.raises(RuntimeError):
            tree.lookup("A", "Z")


# ─── Constraint ───


class TestConstraint:
    def test_orientation(self):
        k = import_kinetic()
        c = k.Constraint.orientation("ee", [0, 0, 1], 0.1)
        assert "orientation" in repr(c).lower() or "Orientation" in repr(c)

    def test_position_bound(self):
        k = import_kinetic()
        c = k.Constraint.position_bound("ee", "z", 0.3, 1.5)
        r = repr(c)
        assert "position" in r.lower() or "Position" in r

    def test_joint(self):
        k = import_kinetic()
        c = k.Constraint.joint(3, -1.0, 1.0)
        r = repr(c)
        assert "3" in r or "joint" in r.lower()

    def test_visibility(self):
        k = import_kinetic()
        c = k.Constraint.visibility("cam", [1, 0, 0.5], 0.5)
        r = repr(c)
        assert "visibility" in r.lower() or "cam" in r

    def test_invalid_axis(self):
        k = import_kinetic()
        with pytest.raises(ValueError):
            k.Constraint.position_bound("ee", "w", 0, 1)


# ─── CartesianConfig ───


class TestCartesianConfig:
    def test_create_default(self):
        k = import_kinetic()
        cfg = k.CartesianConfig()
        assert "CartesianConfig" in repr(cfg)

    def test_create_custom(self):
        k = import_kinetic()
        cfg = k.CartesianConfig(max_step=0.01, jump_threshold=2.0)
        assert "CartesianConfig" in repr(cfg)


# ─── RMP and Policy ───


class TestRMP:
    def test_create(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        rmp = k.RMP(robot)
        assert rmp.dof == 6
        assert rmp.num_policies == 0

    def test_add_policy(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        rmp = k.RMP(robot)
        rmp.add(k.Policy.damping(0.5))
        assert rmp.num_policies == 1
        rmp.add(k.Policy.joint_limit_avoidance(0.1, 15.0))
        assert rmp.num_policies == 2

    def test_clear(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        rmp = k.RMP(robot)
        rmp.add(k.Policy.damping(0.5))
        rmp.clear()
        assert rmp.num_policies == 0

    def test_compute(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        rmp = k.RMP(robot)
        rmp.add(k.Policy.damping(0.5))
        rmp.add(k.Policy.joint_limit_avoidance(0.1, 15.0))

        joints = np.zeros(6)
        vels = np.zeros(6)
        cmd = rmp.compute(joints, vels, dt=0.002)
        assert "positions" in cmd
        assert "velocities" in cmd
        assert "accelerations" in cmd
        assert cmd["positions"].shape == (6,)


class TestPolicy:
    def test_all_types(self):
        k = import_kinetic()
        policies = [
            k.Policy.avoid_self_collision(20.0),
            k.Policy.joint_limit_avoidance(0.1, 15.0),
            k.Policy.singularity_avoidance(0.02, 5.0),
            k.Policy.damping(0.5),
        ]
        for p in policies:
            assert repr(p)  # should not crash


# ─── Grasp ───


class TestGrasp:
    def test_gripper_parallel(self):
        k = import_kinetic()
        g = k.GripperType.parallel(0.08, 0.03)
        assert "parallel" in repr(g).lower()

    def test_gripper_suction(self):
        k = import_kinetic()
        g = k.GripperType.suction(0.02)
        assert "suction" in repr(g).lower()

    def test_generate_grasps(self):
        k = import_kinetic()
        gen = k.GraspGenerator(k.GripperType.parallel(0.08, 0.03))
        grasps = gen.from_shape("cylinder", [0.04, 0.12], np.eye(4), num_candidates=10)
        assert len(grasps) > 0
        g = grasps[0]
        assert g.quality_score >= 0
        assert len(g.approach_vector) == 3
        pose = g.pose()
        assert pose.shape == (4, 4)

    def test_invalid_shape(self):
        k = import_kinetic()
        gen = k.GraspGenerator(k.GripperType.parallel(0.08, 0.03))
        with pytest.raises(ValueError, match="Unknown shape"):
            gen.from_shape("triangle", [1.0], np.eye(4))


# ─── Task ───


class TestTask:
    def test_move_to(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        goal = k.Goal.joints(np.zeros(6))
        task = k.Task.move_to(robot, goal)
        assert "move_to" in repr(task)

    def test_gripper(self):
        k = import_kinetic()
        task = k.Task.gripper(0.08)
        assert "gripper" in repr(task)


# ─── IK Config ───


class TestIKConfig:
    def test_dls_solver(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        pose = robot.fk(joints)
        result = robot.ik_config(pose, solver="dls", seed=joints)
        assert "joints" in result
        assert "converged" in result

    def test_fabrik_solver(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        pose = robot.fk(joints)
        result = robot.ik_config(pose, solver="fabrik", seed=joints)
        assert "joints" in result

    def test_position_only_mode(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        pose = robot.fk(joints)
        result = robot.ik_config(pose, mode="position_only", seed=joints)
        assert "joints" in result

    def test_null_space(self):
        k = import_kinetic()
        robot = k.Robot("franka_panda")
        joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        pose = robot.fk(joints)
        result = robot.ik_config(
            pose, solver="dls", null_space="manipulability", seed=joints
        )
        assert "joints" in result

    def test_invalid_solver(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        pose = robot.fk(np.zeros(6))
        with pytest.raises(ValueError, match="Unknown solver"):
            robot.ik_config(pose, solver="invalid_solver")


# ─── Trajectory Processing ───


class TestTrajectoryProcessing:
    def _make_path_traj(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal, time_parameterize=False)
        return k, traj, robot

    def test_time_parameterize_trapezoidal(self):
        k, traj, robot = self._make_path_traj()
        vel = np.array(robot.velocity_limits)
        acc = np.array(robot.acceleration_limits)
        timed = traj.time_parameterize("trapezoidal", vel, acc)
        assert timed.duration > 0
        assert timed.num_waypoints >= 2

    def test_time_parameterize_totp(self):
        k, traj, robot = self._make_path_traj()
        vel = np.array(robot.velocity_limits)
        acc = np.array(robot.acceleration_limits)
        timed = traj.time_parameterize("totp", vel, acc)
        assert timed.duration > 0

    def test_time_parameterize_invalid(self):
        k, traj, robot = self._make_path_traj()
        vel = np.array(robot.velocity_limits)
        acc = np.array(robot.acceleration_limits)
        with pytest.raises(ValueError, match="Unknown profile"):
            traj.time_parameterize("invalid_profile", vel, acc)

    def test_validate(self):
        k = import_kinetic()
        robot = k.Robot("ur5e")
        planner = k.Planner(robot)
        start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        goal = k.Goal.joints(np.array([0.5, -1.0, 0.5, -1.0, -0.5, 0.1]))
        traj = planner.plan(start, goal)

        lower = np.array([-6.28] * 6)
        upper = np.array([6.28] * 6)
        vel = np.array([3.0] * 6)
        acc = np.array([10.0] * 6)
        violations = traj.validate(lower, upper, vel, acc)
        assert isinstance(violations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
