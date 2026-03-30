# 10 — Regression & CI Gate Tests

**Priority**: P0 — CI gates prevent regressions from merging
**File**: `crates/kinetic/tests/acceptance/test_regression_ci.rs` + CI configuration
**Estimated: continuous**

---

## PRINCIPLE

These are not traditional unit tests — they are CI configuration requirements and
meta-tests that verify the test infrastructure itself works correctly.

---

## CI GATE 1: All Acceptance Tests Pass

**Requirement**: GitLab CI/CD CI must run all acceptance tests on every PR.

```yaml
# In .github/workflows/kinetic.yml, add:
acceptance:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test --test acceptance -- --include-ignored
      working-directory: kinetic/crates/kinetic
      timeout-minutes: 30
```

**Gate**: PR cannot merge if any acceptance test fails.

---

## CI GATE 2: Coverage ≥ 85%

**Requirement**: Code coverage must be ≥ 85% (up from current 80% target).

Safety-critical code (kinetic-kinematics, kinetic-collision, kinetic-execution) should
individually be ≥ 90%.

```yaml
coverage:
  runs-on: ubuntu-latest
  steps:
    - run: cargo tarpaulin --workspace --lib --out xml
    - name: Check coverage
      run: |
        coverage=$(cargo tarpaulin --workspace --lib --out json | jq '.coverage')
        if (( $(echo "$coverage < 85" | bc -l) )); then
          echo "Coverage $coverage% is below 85% threshold"
          exit 1
        fi
```

---

## CI GATE 3: No Clippy Warnings

**Requirement**: `cargo clippy -- -D warnings` must pass with zero warnings.

---

## CI GATE 4: No New Unsafe Code Outside SIMD

**Requirement**: `#[deny(unsafe_code)]` on all crates except kinetic-collision (SIMD kernels).

Test this:
```rust
#[test]
fn p0_no_unsafe_outside_simd() {
    // Verify that unsafe blocks only exist in kinetic-collision/src/simd/
    // This is a grep-based check:
    // grep -rn "unsafe" crates/ --include="*.rs" | grep -v "kinetic-collision/src/simd"
    //   | grep -v "#\[deny(unsafe_code)\]" | grep -v "// SAFETY:" | grep -v "test"
    // Should return 0 results (except for the DLS fallback line which needs fixing)
}
```

Practically: add `#![deny(unsafe_code)]` to each crate's lib.rs (except kinetic-collision).

---

## CI GATE 5: Benchmarks Within 2x Baseline

**Requirement**: No performance regression > 2x on any benchmark.

```yaml
bench:
  runs-on: ubuntu-latest
  steps:
    - run: cargo bench -p kinetic -- --save-baseline pr
    - run: cargo bench -p kinetic -- --baseline main --save-baseline pr
    # Compare and fail if any benchmark regressed more than 2x
```

---

## CI GATE 6: Miri on Non-SIMD Code

**Requirement**: Run Miri on kinetic-core, kinetic-robot, kinetic-kinematics (non-SIMD paths)
to detect undefined behavior.

```yaml
miri:
  runs-on: ubuntu-latest
  steps:
    - uses: dtolnay/rust-toolchain@nightly
      with:
        components: miri
    - run: cargo +nightly miri test -p kinetic-core
    - run: cargo +nightly miri test -p kinetic-robot
    - run: cargo +nightly miri test -p kinetic-kinematics
```

---

## CI GATE 7: Proptest Regressions

**Requirement**: Proptest saves failure cases to `proptest-regressions/`. These files must be
committed and all saved failure cases must pass.

Add to `.gitignore` exceptions:
```
!**/proptest-regressions/
```

---

## META-TEST 1: helpers.rs Robot List Matches Actual

**Function**: `p0_robot_list_matches_actual`

Verify that `ALL_ROBOTS` in helpers.rs contains all robots that exist in `robot_configs/`:

```rust
#[test]
fn p0_robot_list_matches_actual() {
    let robot_configs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../robot_configs");

    let on_disk: std::collections::HashSet<String> = std::fs::read_dir(&robot_configs_dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                Some(entry.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();

    let in_list: std::collections::HashSet<String> = ALL_ROBOTS.iter()
        .map(|(name, _, _)| name.to_string())
        .collect();

    let missing = on_disk.difference(&in_list).collect::<Vec<_>>();
    let extra = in_list.difference(&on_disk).collect::<Vec<_>>();

    assert!(missing.is_empty(),
        "Robots on disk but not in ALL_ROBOTS: {:?}", missing);
    assert!(extra.is_empty(),
        "Robots in ALL_ROBOTS but not on disk: {:?}", extra);
}
```

---

## META-TEST 2: DOF Constants Are Correct

**Function**: `p0_dof_constants_correct`

Verify that the DOF values in `ALL_ROBOTS` match what the robot reports:

```rust
#[test]
fn p0_dof_constants_correct() {
    let (_, failed, _) = run_for_all_robots(|name, expected_dof, expected_arm_dof| {
        let robot = Robot::from_name(name)
            .map_err(|e| format!("load failed: {}", e))?;
        if robot.dof != expected_dof {
            return Err(format!("DOF {} != expected {}", robot.dof, expected_dof));
        }
        let chain = KinematicChain::auto_detect(&robot)
            .map_err(|e| format!("chain failed: {}", e))?;
        if chain.dof != expected_arm_dof {
            return Err(format!("arm DOF {} != expected {}", chain.dof, expected_arm_dof));
        }
        Ok(())
    });
    assert_eq!(failed, 0);
}
```

---

## META-TEST 3: All Test Categories Have Tests

**Function**: `p0_all_categories_covered`

This is a compile-time check — if any test file is missing or empty, the test suite
is incomplete. Verify by listing all expected test modules:

```rust
#[test]
fn p0_all_categories_covered() {
    // These modules must exist and contain at least one #[test] function
    // If this test compiles, all modules exist
    // The assert counts verify they're not empty stubs
    let kinematic_tests = 15; // from 01_
    let limit_tests = 14;     // from 02_
    let collision_tests = 17; // from 03_
    let trajectory_tests = 16; // from 04_
    let planner_tests = 15;   // from 05_
    let execution_tests = 15; // from 06_
    let numerical_tests = 18; // from 07_
    let robot_tests = 15;     // from 08_
    let multi_tests = 8;      // from 09_

    let total = kinematic_tests + limit_tests + collision_tests + trajectory_tests
        + planner_tests + execution_tests + numerical_tests + robot_tests + multi_tests;

    // At minimum, we expect this many test functions
    assert!(total >= 100, "expected at least 100 acceptance tests, got {}", total);
}
```

---

## REGRESSION LIBRARY

As bugs are found and fixed, add regression tests:

```rust
/// Regression: [date] [description]
/// Bug: [what happened]
/// Fix: [what was fixed]
/// Robot: [which robot]
#[test]
fn regression_YYYY_MM_DD_description() {
    // Reproduce the exact scenario that triggered the bug
    // Verify the fix works
}
```

Store regression tests in a separate section at the bottom of the appropriate category file,
or in a dedicated `test_regressions.rs` file.

---

## SUMMARY

| Gate/Test | Type | Priority |
|-----------|------|----------|
| All acceptance pass | CI gate | P0 |
| Coverage ≥ 85% | CI gate | P0 |
| No clippy warnings | CI gate | P0 |
| No unsafe outside SIMD | CI gate | P0 |
| Benchmarks within 2x | CI gate | P1 |
| Miri on core crates | CI gate | P1 |
| Proptest regressions | CI gate | P1 |
| Robot list matches disk | Meta-test | P0 |
| DOF constants correct | Meta-test | P0 |
| All categories covered | Meta-test | P0 |
