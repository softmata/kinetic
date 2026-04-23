# Contributing

How to contribute code, tests, robot configs, and documentation to kinetic.

## Bug Reports

File issues on the GitHub repository with:

1. **Kinetic version** (`cargo pkgid kinetic`)
2. **Rust version** (`rustc --version`)
3. **Robot model** (e.g., `ur5e`, `franka_panda`)
4. **Minimal reproduction** -- smallest code that triggers the bug
5. **Expected vs actual behavior**
6. **Error output** (full `KineticError` Display string)

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Research first** -- read existing code in the area you are changing.
   Understand patterns, conventions, and dependencies before writing code.
3. **Address tech debt** -- if you find issues in the code you are touching,
   fix them in a separate commit before adding new functionality.
4. **Write tests** -- every PR must include tests for new functionality.
   Bug fixes must include a regression test.
5. **Run the full check suite:**

```bash
# Format
cargo fmt --check

# Lint (all warnings are errors)
cargo clippy -- -D warnings

# Tests
cargo test

# Visual tests (if applicable)
xvfb-run cargo test
```

6. **Open the PR** with a clear description of what changed and why
7. **Respond to review** -- address all comments before merge

## Code Style

### Rust Conventions

- **Format:** `cargo fmt` (rustfmt defaults, no custom config)
- **Lint:** `cargo clippy -D warnings` (zero warnings policy)
- **Coverage target:** 85% for new code
- **Error handling:** Return `kinetic_core::Result<T>`, use `thiserror`
- **Documentation:** All `pub` items must have `///` doc comments
- **Tests:** Place unit tests in `#[cfg(test)] mod tests` at the bottom
  of each file. Integration tests go in `tests/`.
- **Naming:** `UpperCamelCase` for types, `snake_case` for functions,
  `SCREAMING_SNAKE` for constants.

### Commit Messages

- Use imperative mood: "Add RRT* planner" not "Added RRT* planner"
- First line under 72 characters
- Reference issues: "Fix #123: IK diverges near singularity"

## Adding Robot Configurations

Robot configs are the easiest way to contribute. Every new robot helps
the entire community.

### Template

Create `robot_configs/<name>/` with:

**kinetic.toml:**
```toml
[robot]
name = "<name>"
urdf = "<name>.urdf"
dof = <n>

[planning_group.arm]
chain = ["<base_link>", "<tip_link>"]

[end_effector.tool]
parent_link = "<tip_link>"
parent_group = "arm"
tcp_xyz = [0.0, 0.0, 0.0]

[ik]
solver = "dls"   # or "opw" for 6-DOF spherical wrist

[named_poses]
home = [0.0, ...]
zero = [0.0, ...]

[collision]
self_collision_pairs = "auto"
padding = 0.01
skip_pairs = []
```

### Required Verification

Before submitting a robot config PR, verify:

1. **FK roundtrip:** `fk(ik(fk(joints))) == fk(joints)` within 1mm
2. **Named poses:** All poses are within joint limits
3. **Planning:** `Planner::new(&robot)` succeeds and can plan at least
   one simple path
4. **IK solver:** The configured solver converges for 95%+ of random
   reachable poses

### Test Template

Include this test in your PR:

```rust
#[test]
fn <name>_fk_ik_roundtrip() {
    let robot = Robot::from_name("<name>").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let home = robot.named_pose("home").unwrap();
    let pose = planner.fk(&home).unwrap();
    let ik_joints = planner.ik(&pose).unwrap();
    let recovered = planner.fk(&ik_joints).unwrap();
    let err = (pose.translation() - recovered.translation()).norm();
    assert!(err < 0.001, "FK/IK error: {err}");
}
```

## Adding Planners

1. Create a new file in `crates/kinetic-planning/src/`
2. Implement the planner (see `rrt.rs` for reference)
3. Add a `PlannerType` variant in `facade.rs`
4. Add dispatch in `Planner::plan_with_config`
5. Write acceptance tests on at least two robot models (UR5e + Panda)
6. Add a benchmark in `benches/`

## Adding IK Solvers

1. Create a new file in `crates/kinetic-kinematics/src/`
2. Implement the solver function returning `Result<IKSolution>`
3. Add an `IKSolver` variant in `ik.rs`
4. Add dispatch in `solve_once`
5. Write FK/IK roundtrip tests
6. Document which robot geometries the solver supports

## Adding Documentation

Documentation lives in `docs-site/src/`. Pages are Markdown.

- **Guides:** Practical how-to content with code examples
- **Reference:** Exhaustive technical details
- **Tutorials:** Step-by-step walkthroughs for specific tasks
- **Migration:** Help users transition from other frameworks

All code examples should compile (or be marked `ignore` with a comment
explaining why). Test code examples with `cargo test --doc`.

## Development Setup

```bash
# Clone
git clone https://github.com/softmata/kinetic.git
cd kinetic

# Build
cargo build

# Test
cargo test

# Benchmarks
cargo bench

# Documentation
cargo doc --open
```

## License

Kinetic is licensed under Apache-2.0. All contributions must be compatible
with this license. By submitting a PR, you agree that your contribution
is licensed under the same terms.
