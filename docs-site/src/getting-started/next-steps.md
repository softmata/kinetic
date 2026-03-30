# Next Steps

You can now load robots, compute FK, and plan motions. Here's where to go based on what you want to do.

## "I want to..."

### ...avoid obstacles during planning

Add collision objects to a Scene and plan around them.

→ **[Planning with Obstacles](../tutorials/rust/planning-with-obstacles.md)** (Rust) or **[Planning in Python](../tutorials/python/planning.md)**

### ...control a robot in real-time

Use Servo for teleoperation with joystick, spacemouse, or programmatic twist commands at 500 Hz.

→ **[Servo Control](../tutorials/rust/servo-control.md)** (Rust) or **[Servo Control in Python](../tutorials/python/servo-control.md)**

### ...do pick and place

Plan approach, grasp, retreat, transport, and place with task composition.

→ **[Pick and Place](../tutorials/rust/pick-and-place.md)** (Rust) or **[Pick and Place in Python](../tutorials/python/pick-and-place.md)**

### ...choose the right IK solver

10 solvers with different trade-offs. The flowchart tells you which to use.

→ **[IK Solver Selection](../tutorials/rust/ik-solver-selection.md)**

### ...choose the right planner

14 algorithms. RRT-Connect is the default, but GCS is globally optimal and EST handles narrow passages.

→ **[Planner Selection Guide](../guides/planner-selection.md)**

### ...use Python

Full API with numpy integration, type stubs for IDE autocomplete, and matplotlib-friendly trajectory export.

→ **[Python Quickstart](../tutorials/python/quickstart.md)**

### ...understand the algorithms

Learn what FK, IK, RRT, and RMP actually do, with diagrams and examples.

→ **[Core Concepts](../concepts/glossary.md)** — start with the Glossary

### ...deploy to production

Checklist for safety validation, error handling, monitoring, and trajectory verification.

→ **[Production Deployment](../guides/production-deployment.md)**

### ...migrate from MoveIt2

Step-by-step migration with config translation and API mapping.

→ **[From MoveIt2](../migration/from-moveit2.md)**

### ...add my own robot

Load custom URDFs and create kinetic configuration files.

→ **[Custom Robots](../guides/custom-robots.md)**

### ...integrate with HORUS

Use kinetic as PlannerNode, ServoNode, and SceneNode in the HORUS robotics framework.

→ **[HORUS Integration](../guides/horus-integration.md)**

### ...optimize with GPU

Parallel-seed trajectory optimization using wgpu compute shaders.

→ **[GPU Optimization](../tutorials/rust/gpu-optimization.md)**

### ...contribute to kinetic

Add planners, IK solvers, robot configs, or documentation.

→ **[Contributing](../community/contributing.md)** and **[Extending Kinetic](../guides/extending-kinetic.md)**

## Learning Path

If you want to learn kinetic systematically, follow this order:

1. [Core Concepts](../concepts/glossary.md) — understand the fundamentals
2. [FK and IK Tutorial](../tutorials/rust/fk-and-ik.md) — kinematics hands-on
3. [Planning Basics](../tutorials/rust/planning-basics.md) — your first planner
4. [Planning with Obstacles](../tutorials/rust/planning-with-obstacles.md) — scenes and collision
5. [Servo Control](../tutorials/rust/servo-control.md) — real-time control
6. [Pick and Place](../tutorials/rust/pick-and-place.md) — complete workflow
7. [Production Deployment](../guides/production-deployment.md) — deploying on real robots
