# Kinetic Production Acceptance Test Specifications

## Purpose

These documents specify **every test** that Kinetic must pass before it can be deployed on
real robots in production environments and research labs. If Kinetic passes all tests described
in these specs, it is considered production-ready.

## For the Implementing Claude

You are reading these specs to **write actual Rust test code**. Here's how:

1. Read `00_CONVENTIONS.md` first — it has import patterns, helpers, and assertion macros
2. Each spec file (`01_` through `11_`) describes a test category
3. Each test has: exact function name, file location, setup, assertions, tolerances
4. Write the tests as integration tests in `crates/kinetic/tests/acceptance/`
5. Tests must compile and run with `cargo test --test acceptance`
6. If a test FAILS, that means Kinetic has a bug that must be fixed before production

## Test Files to Create

```
crates/kinetic/tests/acceptance/
  mod.rs                          -- module root
  test_kinematic_correctness.rs   -- from 01_KINEMATIC_CORRECTNESS.md
  test_joint_limits.rs            -- from 02_JOINT_LIMIT_ENFORCEMENT.md
  test_collision_safety.rs        -- from 03_COLLISION_SAFETY.md
  test_trajectory_safety.rs       -- from 04_TRAJECTORY_SAFETY.md
  test_planner_correctness.rs     -- from 05_PLANNER_CORRECTNESS.md
  test_execution_safety.rs        -- from 06_EXECUTION_SAFETY.md
  test_numerical_robustness.rs    -- from 07_NUMERICAL_ROBUSTNESS.md
  test_robot_acceptance.rs        -- from 08_ROBOT_ACCEPTANCE.md
  test_multi_robot.rs             -- from 09_MULTI_ROBOT.md
  test_regression_ci.rs           -- from 10_REGRESSION_CI.md
  helpers.rs                      -- shared test utilities
```

## Priority Levels

- **P0**: Must pass before ANY real robot deployment. Failure = potential hardware damage or injury
- **P1**: Must pass before researcher/lab deployment. Failure = incorrect results or degraded safety
- **P2**: Must pass before public release. Failure = poor user experience or edge-case bugs

## How to Run

```bash
# Run all acceptance tests
cargo test --test acceptance

# Run a specific category
cargo test --test acceptance kinematic_correctness

# Run with output (see diagnostics)
cargo test --test acceptance -- --nocapture

# Run only P0 tests
cargo test --test acceptance -- --skip p1 --skip p2
```

## Spec Index

| File | Category | Priority | Est. Test Count |
|------|----------|----------|-----------------|
| 01_KINEMATIC_CORRECTNESS.md | FK/IK accuracy | P0 | ~33,000 parameterized |
| 02_JOINT_LIMIT_ENFORCEMENT.md | Joint limits | P0 | ~36,000 parameterized |
| 03_COLLISION_SAFETY.md | Collision detection | P0 | ~63,000 parameterized |
| 04_TRAJECTORY_SAFETY.md | Trajectory validation | P0 | ~5,000 parameterized |
| 05_PLANNER_CORRECTNESS.md | Planning correctness | P1 | ~2,500 parameterized |
| 06_EXECUTION_SAFETY.md | Execution safety | P0 | ~100 |
| 07_NUMERICAL_ROBUSTNESS.md | NaN/Inf/degenerate | P1 | ~12,000 parameterized |
| 08_ROBOT_ACCEPTANCE.md | Per-robot validation | P0 | ~10 per robot |
| 09_MULTI_ROBOT.md | Dual-arm/multi | P1 | ~75 |
| 10_REGRESSION_CI.md | CI gates | P0 | continuous |
| 11_CRITICAL_FIXES.md | Safety fixes needed | P0 | 5 fixes + tests |
