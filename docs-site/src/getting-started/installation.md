# Installation

## Rust

Add kinetic to your project:

```bash
cargo add kinetic
```

Or add it to your `Cargo.toml` manually:

```toml
[dependencies]
kinetic = "0.1"
```

**Requirements:** Rust 1.75 or later. Works on Linux (primary), macOS, and Windows (experimental).

## Python

```bash
pip install kinetic
```

This installs a native extension built with PyO3 — no Rust toolchain needed on your machine.

**Requirements:** Python 3.9 or later. numpy is installed automatically.

## From Source

```bash
git clone https://gitlab.com/softmata/kinetic.git
cd kinetic
cargo build --release
```

For the Python bindings:

```bash
cd crates/kinetic-python
pip install maturin
maturin develop --release
```

## Verify Installation

**Rust:**

```bash
cargo run --example plan_simple -p kinetic
```

You should see output like:

```
=== KINETIC Simple Planning ===
One-liner: 14 waypoints, 237ms, path length 2.145
```

**Python:**

```bash
python -c "import kinetic; r = kinetic.Robot('ur5e'); print(f'{r.name}: {r.dof} DOF')"
```

Expected output:

```
ur5e: 6 DOF
```

## Next

[Hello World →](hello-world.md)
