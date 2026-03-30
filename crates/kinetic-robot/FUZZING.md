# Fuzz Testing for kinetic-robot

Fuzz targets for URDF, SRDF, and TOML config parsing using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer).

## Prerequisites

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
```

## Available Targets

| Target | Input | What it tests |
|--------|-------|---------------|
| `fuzz_urdf` | Arbitrary bytes as URDF XML | `Robot::from_urdf_string()` — XML parsing, link/joint conversion, geometry extraction |
| `fuzz_srdf` | Arbitrary bytes as SRDF XML | `SrdfModel::from_string()` + `apply_to_robot()` — XML parsing, group/collision pair extraction |
| `fuzz_config` | Arbitrary bytes as TOML config | `Robot::from_config_strings()` — TOML parsing, config application to robot model |

## Running

```bash
cd crates/kinetic-robot

# Run a specific target (runs until stopped or crash found)
cargo +nightly fuzz run fuzz_urdf

# Run for a fixed duration (e.g., 10 minutes)
cargo +nightly fuzz run fuzz_urdf -- -max_total_time=600

# Run with specific corpus directory
cargo +nightly fuzz run fuzz_urdf fuzz/corpus/fuzz_urdf

# List all targets
cargo +nightly fuzz list
```

## Investigating Crashes

```bash
# Crashes are saved in fuzz/artifacts/<target>/
# To reproduce a crash:
cargo +nightly fuzz run fuzz_urdf fuzz/artifacts/fuzz_urdf/crash-<hash>

# Minimize a crash input:
cargo +nightly fuzz tmin fuzz_urdf fuzz/artifacts/fuzz_urdf/crash-<hash>
```

## Seed Corpus

Pre-seeded corpus files are in `fuzz/corpus/<target>/`. These give the fuzzer a starting point with structurally valid inputs to mutate from.
