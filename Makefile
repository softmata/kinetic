.PHONY: test coverage miri miri-full bench clean

# Run all workspace tests
test:
	cargo test --workspace --exclude kinetic-gpu

# Generate code coverage report (requires cargo-tarpaulin)
coverage:
	cargo tarpaulin --config tarpaulin.toml
	@echo "Coverage report: coverage/tarpaulin-report.html"

# Run Miri on unsafe code — fast subset (requires nightly + miri component)
# Runs scalar fallback + SoA tests. Use `make miri-full` for complete suite.
miri:
	cargo +nightly miri test -p kinetic-collision --lib simd::tests::scalar_fallback_correctness
	cargo +nightly miri test -p kinetic-collision --lib soa::tests

# Run Miri on the full SIMD test suite (slow — may take 10+ minutes)
miri-full:
	cargo +nightly miri test -p kinetic-collision --lib simd::tests
	cargo +nightly miri test -p kinetic-collision --lib soa::tests

# Run benchmarks
bench:
	cargo bench --workspace --exclude kinetic-gpu

# Clean build artifacts and coverage
clean:
	cargo clean
	rm -rf coverage/
