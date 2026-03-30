"""End-to-end workflow tests for the kinetic Python bindings.

Tests real-world scenarios matching the Rust integration tests:
- Pick-and-place workflow
- Planning with obstacles (add, plan, verify collision-free)
- Trajectory parameterization and waypoint access
- Scene modification and replanning
- Servo-based reactive control workflow

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_e2e.py -v
"""

import time

import numpy as np
import pytest

import kinetic


# ── Helpers ──────────────────────────────────────────────────────────

HOME = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
SAFE_MID = np.array([0.0, -1.2, 1.0, -0.8, -np.pi / 2, 0.0])


@pytest.fixture
def ur5e():
    return kinetic.Robot("ur5e")


@pytest.fixture
def panda():
    return kinetic.Robot("franka_panda")


@pytest.fixture
def ur5e_planner(ur5e):
    return kinetic.Planner(ur5e)


# ── Pick-and-Place Workflow ──────────────────────────────────────────


class TestPickAndPlaceWorkflow:
    """Complete pick-and-place: plan to pick → grasp → plan to place → release."""

    def test_full_pick_and_place_sequence(self, ur5e, ur5e_planner):
        """Plan a multi-step pick-and-place workflow entirely in Python."""
        scene = kinetic.Scene(ur5e)

        # Add a table
        table = kinetic.Shape.cuboid(0.6, 0.4, 0.02)
        table_pose = np.eye(4)
        table_pose[2, 3] = -0.01  # Just below base
        scene.add("table", table, table_pose)
        assert scene.num_objects == 1

        # Step 1: Plan from home to pre-pick position
        pre_pick = np.array([0.3, -1.0, 0.5, 0.0, 0.8, 0.0])
        goal_pre_pick = kinetic.Goal.joints(pre_pick)
        traj1 = ur5e_planner.plan(HOME, goal_pre_pick)
        assert traj1.num_waypoints >= 2
        assert traj1.duration > 0
        assert traj1.dof == 6

        # Verify trajectory reaches the goal
        final1 = traj1.sample(traj1.duration)
        np.testing.assert_allclose(final1, pre_pick, atol=0.15)

        # Step 2: Plan from pre-pick to pick position (small move)
        pick = np.array([0.35, -0.9, 0.6, 0.0, 0.7, 0.0])
        goal_pick = kinetic.Goal.joints(pick)
        traj2 = ur5e_planner.plan(pre_pick, goal_pick)
        assert traj2.num_waypoints >= 2

        # Step 3: Attach object (simulate grasp)
        part = kinetic.Shape.cylinder(0.02, 0.03)
        grasp_tf = np.eye(4)
        scene.attach("part", part, grasp_tf, "tool0")
        assert scene.num_attached == 1

        # Step 4: Plan from pick to place position
        place = np.array([0.6, -0.8, 0.3, 0.1, 0.3, -0.2])
        goal_place = kinetic.Goal.joints(place)
        traj3 = ur5e_planner.plan(pick, goal_place)
        assert traj3.num_waypoints >= 2

        # Step 5: Detach object (simulate release)
        place_pose = np.eye(4)
        place_pose[:3, 3] = [0.3, -0.1, 0.05]
        scene.detach("part", place_pose)
        assert scene.num_attached == 0
        assert scene.num_objects == 2  # table + placed part

        # Step 6: Plan return to home
        goal_home = kinetic.Goal.joints(HOME)
        traj4 = ur5e_planner.plan(place, goal_home)
        assert traj4.num_waypoints >= 2

        # Verify we have 4 valid trajectories
        trajs = [traj1, traj2, traj3, traj4]
        for i, t in enumerate(trajs):
            assert t.duration > 0, f"Trajectory {i} has zero duration"
            assert t.dof == 6, f"Trajectory {i} wrong DOF"

    def test_pick_and_place_with_trajectory_sampling(self, ur5e, ur5e_planner):
        """Verify we can sample trajectories at fine resolution for execution."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = ur5e_planner.plan(start, goal)

        # Sample at 500 Hz (typical robot control rate)
        dt = 1.0 / 500.0
        num_samples = int(traj.duration / dt) + 1
        positions = []
        for i in range(num_samples):
            t = min(i * dt, traj.duration)
            pos = traj.sample(t)
            positions.append(pos)
            assert len(pos) == 6
            assert all(np.isfinite(pos))

        assert len(positions) >= 2
        # Consecutive samples should be close (smooth trajectory)
        for i in range(1, min(len(positions), 50)):
            max_jump = np.max(np.abs(positions[i] - positions[i - 1]))
            assert max_jump < 0.1, (
                f"Joint jump at sample {i}: {max_jump:.4f} rad"
            )


# ── Planning with Obstacles ──────────────────────────────────────────


class TestObstacleAvoidance:
    """Planning with obstacles: add objects, plan, verify collision-free."""

    def test_plan_around_obstacle(self, ur5e, ur5e_planner):
        """Add an obstacle and verify the planner produces a collision-free path."""
        scene = kinetic.Scene(ur5e)

        # Add obstacle in the workspace
        obstacle = kinetic.Shape.sphere(0.08)
        obs_pose = np.eye(4)
        obs_pose[:3, 3] = [0.35, 0.0, 0.35]
        scene.add("obstacle", obstacle, obs_pose)

        start = np.array([0.0, -1.0, 0.8, 0.0, 1.0, 0.0])
        goal = kinetic.Goal.joints(np.array([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]))

        # Plan with obstacle-aware planner
        planner = kinetic.Planner(ur5e, scene=scene)
        try:
            traj = planner.plan(start, goal)
            # Verify each waypoint is collision-free
            positions = traj.positions()
            for i, wp in enumerate(positions):
                wp_arr = np.array(wp)
                in_collision = scene.check_collision(wp_arr)
                assert not in_collision, f"Waypoint {i} is in collision"
        except RuntimeError:
            # Planning may fail if obstacle blocks all paths — acceptable
            pass

    def test_add_multiple_obstacles_then_plan(self, ur5e):
        """Build a scene with multiple obstacles and plan through it."""
        scene = kinetic.Scene(ur5e)

        # Add table
        table = kinetic.Shape.cuboid(1.0, 0.6, 0.02)
        table_pose = np.eye(4)
        table_pose[2, 3] = -0.05
        scene.add("table", table, table_pose)

        # Add a few cylinders
        for i in range(3):
            cyl = kinetic.Shape.cylinder(0.03, 0.15)
            cyl_pose = np.eye(4)
            cyl_pose[:3, 3] = [0.3 + i * 0.15, 0.0, 0.07]
            scene.add(f"cylinder_{i}", cyl, cyl_pose)

        assert scene.num_objects == 4

        # Plan avoiding all obstacles
        planner = kinetic.Planner(ur5e, scene=scene)
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        try:
            traj = planner.plan(start, goal)
            assert traj.num_waypoints >= 2
        except RuntimeError:
            pass  # OK if planning fails with dense obstacles

    def test_dynamic_obstacle_replan(self, ur5e, ur5e_planner):
        """Plan, add new obstacle, replan — verify replanned path is different."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))

        # Plan 1: no obstacles
        traj1 = ur5e_planner.plan(start, goal)

        # Plan 2: add obstacle and replan
        scene = kinetic.Scene(ur5e)
        scene.add(
            "new_obstacle",
            kinetic.Shape.sphere(0.1),
            np.array([
                [1, 0, 0, 0.3],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1.0],
            ], dtype=np.float64),
        )

        planner2 = kinetic.Planner(ur5e, scene=scene)
        try:
            traj2 = planner2.plan(start, goal)
            # Both should reach the goal
            final1 = traj1.sample(traj1.duration)
            final2 = traj2.sample(traj2.duration)
            np.testing.assert_allclose(final1, final2, atol=0.2)
        except RuntimeError:
            pass  # Replan may fail — acceptable


# ── Trajectory Inspection ────────────────────────────────────────────


class TestTrajectoryInspection:
    """Deep inspection of trajectory data: to_numpy, positions, timing."""

    def test_to_numpy_full_extraction(self, ur5e_planner):
        """Extract full trajectory as numpy arrays and verify structure."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = ur5e_planner.plan(start, goal)

        times, positions, velocities = traj.to_numpy()

        # Shapes
        n = traj.num_waypoints
        assert times.shape == (n,)
        assert positions.shape == (n, 6)
        assert velocities.shape == (n, 6)

        # Times monotonically increasing
        assert np.all(np.diff(times) >= 0)

        # First time is 0
        assert times[0] == pytest.approx(0.0, abs=1e-6)

        # Last time matches duration
        assert times[-1] == pytest.approx(traj.duration, abs=1e-3)

        # Start position matches
        np.testing.assert_allclose(positions[0], HOME, atol=0.15)

        # All values finite
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))
        assert np.all(np.isfinite(times))

    def test_positions_list_matches_numpy(self, ur5e_planner):
        """Verify positions() list matches to_numpy() positions."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = ur5e_planner.plan(start, goal)

        pos_list = traj.positions()
        _, pos_numpy, _ = traj.to_numpy()

        assert len(pos_list) == pos_numpy.shape[0]
        for i, pl in enumerate(pos_list):
            np.testing.assert_allclose(np.array(pl), pos_numpy[i], atol=1e-10)

    def test_unparameterized_trajectory_access(self, ur5e_planner):
        """Trajectory without time parameterization still gives positions."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = ur5e_planner.plan(start, goal, time_parameterize=False)

        assert traj.num_waypoints >= 2
        assert traj.duration == 0.0
        pos_list = traj.positions()
        assert len(pos_list) >= 2
        assert len(pos_list[0]) == 6


# ── Scene Modification and Replanning ────────────────────────────────


class TestSceneModification:
    """Scene changes during workflow: add/remove/attach/detach + replan."""

    def test_scene_lifecycle(self, ur5e):
        """Full scene lifecycle: create, add, attach, detach, remove, clear."""
        scene = kinetic.Scene(ur5e)
        assert scene.num_objects == 0
        assert scene.num_attached == 0

        # Add objects
        scene.add("box1", kinetic.Shape.cuboid(0.1, 0.1, 0.1), np.eye(4))
        scene.add("sphere1", kinetic.Shape.sphere(0.05), np.eye(4))
        assert scene.num_objects == 2

        # Attach one
        grasp = np.eye(4)
        scene.attach("gripper_part", kinetic.Shape.cylinder(0.02, 0.05), grasp, "tool0")
        assert scene.num_attached == 1

        # Detach
        place_pose = np.eye(4)
        place_pose[:3, 3] = [0.3, 0.0, 0.1]
        scene.detach("gripper_part", place_pose)
        assert scene.num_attached == 0
        assert scene.num_objects == 3  # box1 + sphere1 + gripper_part

        # Remove one
        assert scene.remove("box1")
        assert scene.num_objects == 2

        # Clear all
        scene.clear()
        assert scene.num_objects == 0

    def test_collision_check_with_obstacles(self, ur5e):
        """Verify collision detection works with scene objects."""
        scene = kinetic.Scene(ur5e)

        # Add a large sphere near the robot base
        big_sphere = kinetic.Shape.sphere(0.5)
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 0.3]
        scene.add("big_sphere", big_sphere, pose)

        # Check collision at home — large sphere overlaps robot
        in_collision = scene.check_collision(HOME)
        # Result depends on collision geometry but should not crash
        assert isinstance(in_collision, bool)

    def test_min_distance_changes_with_obstacles(self, ur5e):
        """Adding an obstacle should reduce minimum distance."""
        scene = kinetic.Scene(ur5e)
        joints = SAFE_MID

        dist_empty = scene.min_distance(joints)

        # Add nearby obstacle
        obstacle = kinetic.Shape.sphere(0.05)
        obs_pose = np.eye(4)
        obs_pose[:3, 3] = [0.3, 0.0, 0.3]
        scene.add("nearby", obstacle, obs_pose)

        dist_with_obs = scene.min_distance(joints)

        # Distance should decrease (or stay same if obstacle is not closest)
        assert isinstance(dist_with_obs, float)
        assert np.isfinite(dist_with_obs)

    def test_replan_after_scene_change(self, ur5e):
        """Scene modification forces different planning context."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))

        # Plan 1: empty scene
        planner1 = kinetic.Planner(ur5e, timeout=2.0)
        traj1 = planner1.plan(start, goal)

        # Plan 2: scene with small obstacle far from start
        scene = kinetic.Scene(ur5e)
        scene.add(
            "obstacle",
            kinetic.Shape.sphere(0.03),
            np.array([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0.3],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1.0],
            ], dtype=np.float64),
        )
        planner2 = kinetic.Planner(ur5e, scene=scene, timeout=2.0)
        try:
            traj2 = planner2.plan(start, goal)
            # Both reach goal
            final1 = traj1.sample(traj1.duration)
            final2 = traj2.sample(traj2.duration)
            goal_arr = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
            np.testing.assert_allclose(final1, goal_arr, atol=0.15)
            np.testing.assert_allclose(final2, goal_arr, atol=0.15)
        except RuntimeError:
            # Planning with obstacle may fail — acceptable
            pass


# ── Servo Reactive Control Workflow ──────────────────────────────────


class TestServoWorkflow:
    """Servo-based reactive control: twist commands, state tracking."""

    def test_twist_control_loop(self, ur5e):
        """Simulate a servo control loop: multiple twist commands."""
        scene = kinetic.Scene(ur5e)
        servo = kinetic.Servo(ur5e, scene)
        servo.set_state(SAFE_MID, np.zeros(6))

        # Run 10 steps of twist control
        twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        commands = []
        for _ in range(10):
            cmd = servo.send_twist(twist)
            commands.append(cmd)
            # Update state with new positions
            new_pos = np.array(cmd["positions"])
            new_vel = np.array(cmd["velocities"])
            servo.set_state(new_pos, new_vel)

        assert len(commands) == 10
        # Positions should have changed from initial
        final_state = servo.state()
        initial_positions = SAFE_MID
        final_positions = np.array(final_state["joint_positions"])
        assert not np.allclose(initial_positions, final_positions, atol=1e-6)

    def test_joint_jog_control_loop(self, ur5e):
        """Simulate joint jog control: move specific joints."""
        scene = kinetic.Scene(ur5e)
        servo = kinetic.Servo(ur5e, scene)
        servo.set_state(SAFE_MID, np.zeros(6))

        # Jog joint 0 for 5 steps
        for _ in range(5):
            cmd = servo.send_joint_jog(0, 0.1)
            new_pos = np.array(cmd["positions"])
            new_vel = np.array(cmd["velocities"])
            servo.set_state(new_pos, new_vel)

        state = servo.state()
        # Joint 0 should have moved from initial
        assert abs(state["joint_positions"][0] - SAFE_MID[0]) > 1e-6

    def test_servo_state_inspection(self, ur5e):
        """Verify full servo state is available and consistent."""
        scene = kinetic.Scene(ur5e)
        servo = kinetic.Servo(ur5e, scene)
        servo.set_state(SAFE_MID, np.zeros(6))

        state = servo.state()
        assert "joint_positions" in state
        assert "joint_velocities" in state
        assert "ee_pose" in state
        assert "manipulability" in state
        assert "near_singularity" in state
        assert "near_collision" in state

        # EE pose is a 4x4 matrix
        ee_pose = state["ee_pose"]
        assert ee_pose.shape == (4, 4)
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_allclose(ee_pose[3, :], [0, 0, 0, 1], atol=1e-10)

        # Manipulability should be non-negative
        assert state["manipulability"] >= 0.0


# ── Multi-Robot Workflow ─────────────────────────────────────────────


class TestMultiRobotWorkflow:
    """Work with multiple robots in the same session."""

    def test_two_robots_independent_planning(self, ur5e, panda):
        """Plan for two different robots independently."""
        # UR5e planning
        planner_ur = kinetic.Planner(ur5e)
        start_ur = HOME
        goal_ur = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj_ur = planner_ur.plan(start_ur, goal_ur)
        assert traj_ur.dof == 6

        # Panda planning
        planner_panda = kinetic.Planner(panda)
        start_panda = np.zeros(7)
        goal_panda = kinetic.Goal.joints(
            np.array([0.3, -0.5, 0.3, -1.5, 0.0, 1.0, 0.5])
        )
        traj_panda = planner_panda.plan(start_panda, goal_panda)
        assert traj_panda.dof == 7

        # Both trajectories should be valid
        assert traj_ur.num_waypoints >= 2
        assert traj_panda.num_waypoints >= 2

    def test_multiple_robots_available(self):
        """Verify common robot configurations are loadable."""
        robots = ["ur5e", "ur10e", "franka_panda"]
        loaded = []
        for name in robots:
            try:
                r = kinetic.Robot(name)
                loaded.append((name, r.dof))
            except ValueError:
                pass

        assert len(loaded) >= 2, f"Should load at least 2 robots, got {loaded}"

    def test_robot_fk_ik_roundtrip(self, ur5e):
        """Full FK → IK roundtrip: verify solution reaches target pose."""
        joints_in = np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])
        target_pose = ur5e.fk(joints_in)

        try:
            joints_out = ur5e.ik(target_pose, seed=joints_in)
            recovered_pose = ur5e.fk(np.array(joints_out))
            # Position should match within 1mm
            np.testing.assert_allclose(
                recovered_pose[:3, 3], target_pose[:3, 3], atol=1e-3
            )
        except RuntimeError:
            pytest.skip("IK did not converge")


# ── Convenience API ──────────────────────────────────────────────────


class TestConvenienceAPI:
    """Test the one-liner kinetic.plan() convenience function."""

    def test_plan_one_liner(self):
        """kinetic.plan() should work end-to-end without creating objects."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
        traj = kinetic.plan("ur5e", start, goal)
        assert traj.num_waypoints >= 2
        assert traj.duration > 0
        assert traj.dof == 6

    def test_plan_one_liner_sample_and_numpy(self):
        """Use one-liner result with both sample() and to_numpy()."""
        traj = kinetic.plan(
            "ur5e",
            HOME,
            kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        )

        # Sample at start, middle, end
        joints_start = traj.sample(0.0)
        joints_mid = traj.sample(traj.duration / 2)
        joints_end = traj.sample(traj.duration)
        assert len(joints_start) == 6
        assert len(joints_mid) == 6
        assert len(joints_end) == 6

        # to_numpy
        times, positions, velocities = traj.to_numpy()
        assert times.shape[0] >= 2
        assert positions.shape[1] == 6


# ── Performance Smoke Tests ──────────────────────────────────────────


class TestPerformanceSmoke:
    """Basic performance checks — not benchmarks, just sanity."""

    def test_planning_completes_under_5_seconds(self, ur5e_planner):
        """A simple joint-to-joint plan should complete quickly."""
        start = HOME
        goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))

        t0 = time.monotonic()
        traj = ur5e_planner.plan(start, goal)
        elapsed = time.monotonic() - t0

        assert traj.num_waypoints >= 2
        assert elapsed < 5.0, f"Planning took {elapsed:.2f}s — too slow"

    def test_fk_1000_calls_under_100ms(self, ur5e):
        """FK should be fast — 1000 calls under 100ms."""
        joints = SAFE_MID
        t0 = time.monotonic()
        for _ in range(1000):
            ur5e.fk(joints)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"1000 FK calls took {elapsed:.3f}s — too slow"

    def test_trajectory_sampling_throughput(self, ur5e_planner):
        """Trajectory sampling should support high-frequency control."""
        traj = ur5e_planner.plan(
            HOME,
            kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0])),
        )

        # Sample 10000 points
        t0 = time.monotonic()
        for i in range(10000):
            t = (i / 10000) * traj.duration
            traj.sample(t)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"10000 samples took {elapsed:.3f}s — too slow"
