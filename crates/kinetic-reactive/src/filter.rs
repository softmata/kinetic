//! Smoothing filters for servo commands.

/// Trait for smoothing joint commands.
pub trait SmoothingFilter: Send + Sync {
    /// Filter a joint command, returning smoothed values.
    fn filter(&mut self, positions: &[f64], velocities: &[f64]) -> (Vec<f64>, Vec<f64>);

    /// Reset filter state.
    fn reset(&mut self);
}

/// Exponential Moving Average filter.
///
/// Smooths commands with: `y[n] = α * x[n] + (1 - α) * y[n-1]`
///
/// Higher α = less smoothing (more responsive).
/// Lower α = more smoothing (smoother but more latency).
pub struct ExponentialMovingAverage {
    alpha: f64,
    prev_positions: Option<Vec<f64>>,
    prev_velocities: Option<Vec<f64>>,
}

impl ExponentialMovingAverage {
    /// Create a new EMA filter with the given alpha (0.0, 1.0].
    ///
    /// `alpha = 1.0` means no smoothing (pass-through).
    /// `alpha = 0.1` means heavy smoothing.
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(0.01, 1.0);
        Self {
            alpha,
            prev_positions: None,
            prev_velocities: None,
        }
    }
}

impl SmoothingFilter for ExponentialMovingAverage {
    fn filter(&mut self, positions: &[f64], velocities: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let smoothed_pos = match &self.prev_positions {
            Some(prev) => positions
                .iter()
                .zip(prev.iter())
                .map(|(&curr, &prev)| self.alpha * curr + (1.0 - self.alpha) * prev)
                .collect(),
            None => positions.to_vec(),
        };

        let smoothed_vel = match &self.prev_velocities {
            Some(prev) => velocities
                .iter()
                .zip(prev.iter())
                .map(|(&curr, &prev)| self.alpha * curr + (1.0 - self.alpha) * prev)
                .collect(),
            None => velocities.to_vec(),
        };

        self.prev_positions = Some(smoothed_pos.clone());
        self.prev_velocities = Some(smoothed_vel.clone());

        (smoothed_pos, smoothed_vel)
    }

    fn reset(&mut self) {
        self.prev_positions = None;
        self.prev_velocities = None;
    }
}

/// Second-order Butterworth low-pass filter (per-joint).
///
/// Provides -12dB/octave rolloff above the cutoff frequency.
pub struct ButterworthLowPass {
    /// Per-joint filter state: [x[n-1], x[n-2], y[n-1], y[n-2]] for positions.
    pos_state: Vec<[f64; 4]>,
    /// Per-joint filter state for velocities.
    vel_state: Vec<[f64; 4]>,
    /// Filter coefficients: [b0, b1, b2, a1, a2].
    coeffs: [f64; 5],
    initialized: bool,
}

impl ButterworthLowPass {
    /// Create a Butterworth low-pass filter.
    ///
    /// `cutoff_hz`: cutoff frequency in Hz (must be < sample_rate / 2).
    /// `sample_rate_hz`: the rate at which `filter` will be called.
    /// `dof`: number of joints.
    pub fn new(cutoff_hz: f64, sample_rate_hz: f64, dof: usize) -> Self {
        // Bilinear transform coefficients for 2nd-order Butterworth
        let wc = std::f64::consts::TAU * cutoff_hz / sample_rate_hz;
        let wc_half = (wc / 2.0).tan();
        let k = wc_half;
        let k2 = k * k;
        let sqrt2 = std::f64::consts::SQRT_2;
        let norm = 1.0 / (1.0 + sqrt2 * k + k2);

        let b0 = k2 * norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - sqrt2 * k + k2) * norm;

        Self {
            pos_state: vec![[0.0; 4]; dof],
            vel_state: vec![[0.0; 4]; dof],
            coeffs: [b0, b1, b2, a1, a2],
            initialized: false,
        }
    }

    fn filter_sample(coeffs: &[f64; 5], state: &mut [f64; 4], x: f64) -> f64 {
        let [b0, b1, b2, a1, a2] = *coeffs;
        // x[n-1] = state[0], x[n-2] = state[1], y[n-1] = state[2], y[n-2] = state[3]
        let y = b0 * x + b1 * state[0] + b2 * state[1] - a1 * state[2] - a2 * state[3];
        state[1] = state[0];
        state[0] = x;
        state[3] = state[2];
        state[2] = y;
        y
    }
}

impl SmoothingFilter for ButterworthLowPass {
    fn filter(&mut self, positions: &[f64], velocities: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if !self.initialized {
            // Initialize state to avoid transient
            for (i, &p) in positions.iter().enumerate() {
                if i < self.pos_state.len() {
                    self.pos_state[i] = [p, p, p, p];
                }
            }
            for (i, &v) in velocities.iter().enumerate() {
                if i < self.vel_state.len() {
                    self.vel_state[i] = [v, v, v, v];
                }
            }
            self.initialized = true;
            return (positions.to_vec(), velocities.to_vec());
        }

        let smoothed_pos: Vec<f64> = positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i < self.pos_state.len() {
                    Self::filter_sample(&self.coeffs, &mut self.pos_state[i], p)
                } else {
                    p
                }
            })
            .collect();

        let smoothed_vel: Vec<f64> = velocities
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < self.vel_state.len() {
                    Self::filter_sample(&self.coeffs, &mut self.vel_state[i], v)
                } else {
                    v
                }
            })
            .collect();

        (smoothed_pos, smoothed_vel)
    }

    fn reset(&mut self) {
        for s in &mut self.pos_state {
            *s = [0.0; 4];
        }
        for s in &mut self.vel_state {
            *s = [0.0; 4];
        }
        self.initialized = false;
    }
}

/// Pass-through filter (no smoothing).
pub struct NoFilter;

impl SmoothingFilter for NoFilter {
    fn filter(&mut self, positions: &[f64], velocities: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (positions.to_vec(), velocities.to_vec())
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_passthrough_at_alpha_one() {
        let mut filter = ExponentialMovingAverage::new(1.0);
        let pos = vec![1.0, 2.0, 3.0];
        let vel = vec![0.1, 0.2, 0.3];
        let (fp, fv) = filter.filter(&pos, &vel);
        assert_eq!(fp, pos);
        assert_eq!(fv, vel);
    }

    #[test]
    fn ema_smooths_step_input() {
        let mut filter = ExponentialMovingAverage::new(0.3);
        // First call initializes
        let (p1, _) = filter.filter(&[0.0], &[0.0]);
        assert_eq!(p1, vec![0.0]);

        // Step to 1.0 — should be smoothed
        let (p2, _) = filter.filter(&[1.0], &[1.0]);
        assert!((p2[0] - 0.3).abs() < 1e-10, "Should be 0.3: {}", p2[0]);

        // Another step at 1.0
        let (p3, _) = filter.filter(&[1.0], &[1.0]);
        // 0.3 * 1.0 + 0.7 * 0.3 = 0.51
        assert!((p3[0] - 0.51).abs() < 1e-10, "Should be 0.51: {}", p3[0]);
    }

    #[test]
    fn ema_reset() {
        let mut filter = ExponentialMovingAverage::new(0.5);
        filter.filter(&[1.0], &[1.0]);
        filter.reset();
        // After reset, next call should be pass-through again
        let (p, _) = filter.filter(&[5.0], &[5.0]);
        assert_eq!(p, vec![5.0]);
    }

    #[test]
    fn butterworth_first_call_passthrough() {
        let mut filter = ButterworthLowPass::new(10.0, 500.0, 3);
        let pos = vec![1.0, 2.0, 3.0];
        let vel = vec![0.1, 0.2, 0.3];
        let (fp, fv) = filter.filter(&pos, &vel);
        assert_eq!(fp, pos);
        assert_eq!(fv, vel);
    }

    #[test]
    fn butterworth_smooths_signal() {
        let mut filter = ButterworthLowPass::new(5.0, 500.0, 1);
        // Initialize
        filter.filter(&[0.0], &[0.0]);

        // Apply step — should be attenuated
        let (p, _) = filter.filter(&[1.0], &[0.0]);
        assert!(p[0] < 1.0 && p[0] > 0.0, "Should smooth step: {}", p[0]);
    }

    #[test]
    fn butterworth_converges_to_dc() {
        let mut filter = ButterworthLowPass::new(50.0, 500.0, 1);
        filter.filter(&[0.0], &[0.0]);

        // Feed constant 1.0 for many samples — should converge to 1.0
        let mut val = 0.0;
        for _ in 0..200 {
            let (p, _) = filter.filter(&[1.0], &[0.0]);
            val = p[0];
        }
        assert!((val - 1.0).abs() < 0.01, "Should converge to DC: {}", val);
    }

    #[test]
    fn no_filter_passthrough() {
        let mut filter = NoFilter;
        let pos = vec![1.0, 2.0];
        let vel = vec![0.5, -0.5];
        let (fp, fv) = filter.filter(&pos, &vel);
        assert_eq!(fp, pos);
        assert_eq!(fv, vel);
    }

    // --- EMA additional tests ---

    #[test]
    fn ema_clamps_alpha_low() {
        // Alpha below 0.01 should be clamped to 0.01
        let mut filter = ExponentialMovingAverage::new(0.0);
        filter.filter(&[0.0], &[0.0]);
        let (p, _) = filter.filter(&[1.0], &[0.0]);
        // With alpha=0.01: 0.01 * 1.0 + 0.99 * 0.0 = 0.01
        assert!(
            (p[0] - 0.01).abs() < 1e-10,
            "Alpha should be clamped to 0.01, got {}",
            p[0]
        );
    }

    #[test]
    fn ema_clamps_alpha_high() {
        // Alpha above 1.0 should be clamped to 1.0 (passthrough)
        let mut filter = ExponentialMovingAverage::new(5.0);
        filter.filter(&[0.0], &[0.0]);
        let (p, _) = filter.filter(&[1.0], &[0.0]);
        assert!(
            (p[0] - 1.0).abs() < 1e-10,
            "Alpha clamped to 1.0 means passthrough, got {}",
            p[0]
        );
    }

    #[test]
    fn ema_converges_to_constant() {
        let mut filter = ExponentialMovingAverage::new(0.2);
        filter.filter(&[0.0], &[0.0]);
        let mut val = 0.0;
        for _ in 0..200 {
            let (p, _) = filter.filter(&[1.0], &[0.0]);
            val = p[0];
        }
        assert!(
            (val - 1.0).abs() < 1e-6,
            "EMA should converge to constant input, got {}",
            val
        );
    }

    #[test]
    fn ema_smooths_velocities() {
        let mut filter = ExponentialMovingAverage::new(0.3);
        filter.filter(&[0.0], &[0.0]);
        let (_, v) = filter.filter(&[0.0], &[1.0]);
        assert!(
            (v[0] - 0.3).abs() < 1e-10,
            "Velocity should be smoothed: {}",
            v[0]
        );
    }

    #[test]
    fn ema_multi_joint() {
        let mut filter = ExponentialMovingAverage::new(0.5);
        let (p, v) = filter.filter(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]);
        // First call is passthrough
        assert_eq!(p, vec![1.0, 2.0, 3.0]);
        assert_eq!(v, vec![0.1, 0.2, 0.3]);

        let (p2, v2) = filter.filter(&[3.0, 4.0, 5.0], &[0.5, 0.6, 0.7]);
        // 0.5 * new + 0.5 * old
        assert!((p2[0] - 2.0).abs() < 1e-10);
        assert!((p2[1] - 3.0).abs() < 1e-10);
        assert!((p2[2] - 4.0).abs() < 1e-10);
        assert!((v2[0] - 0.3).abs() < 1e-10);
        assert!((v2[1] - 0.4).abs() < 1e-10);
        assert!((v2[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ema_reset_clears_state_fully() {
        let mut filter = ExponentialMovingAverage::new(0.3);
        // Build up state
        filter.filter(&[1.0], &[1.0]);
        filter.filter(&[2.0], &[2.0]);
        filter.filter(&[3.0], &[3.0]);
        filter.reset();

        // After reset, should behave as if fresh
        let (p, v) = filter.filter(&[10.0], &[5.0]);
        assert_eq!(p, vec![10.0], "First call after reset is passthrough");
        assert_eq!(v, vec![5.0], "First call after reset is passthrough");
    }

    // --- Butterworth additional tests ---

    #[test]
    fn butterworth_reset_clears_state() {
        let mut filter = ButterworthLowPass::new(10.0, 500.0, 2);
        filter.filter(&[1.0, 2.0], &[0.1, 0.2]);
        filter.filter(&[1.5, 2.5], &[0.15, 0.25]);
        filter.reset();

        // After reset, next call should be passthrough again (re-initialization)
        let (p, v) = filter.filter(&[5.0, 6.0], &[0.5, 0.6]);
        assert_eq!(p, vec![5.0, 6.0]);
        assert_eq!(v, vec![0.5, 0.6]);
    }

    #[test]
    fn butterworth_attenuates_high_frequency() {
        // cutoff 5 Hz, sample rate 500 Hz
        let mut filter = ButterworthLowPass::new(5.0, 500.0, 1);
        filter.filter(&[0.0], &[0.0]);

        // Simulate a high-frequency signal (alternating +1/-1 at 250 Hz)
        let mut max_output = 0.0_f64;
        for i in 0..100 {
            let input = if i % 2 == 0 { 1.0 } else { -1.0 };
            let (p, _) = filter.filter(&[input], &[0.0]);
            max_output = max_output.max(p[0].abs());
        }
        // After initial transient settles, output should be heavily attenuated
        // The absolute max may be large during transient, but let's check the last value
        let (p_last, _) = filter.filter(&[1.0], &[0.0]);
        // At 250 Hz input with 5 Hz cutoff, attenuation is massive
        // Just verify it's much less than 1.0
        assert!(
            p_last[0].abs() < 0.5,
            "High freq should be attenuated, got {}",
            p_last[0]
        );
    }

    #[test]
    fn butterworth_multi_joint_independent() {
        let mut filter = ButterworthLowPass::new(20.0, 500.0, 3);
        // Initialize
        filter.filter(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);

        // Step joint 0 to 1.0, keep others at 0
        let (p, _) = filter.filter(&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert!(p[0] > 0.0, "Joint 0 should respond to step");
        assert!(
            p[1].abs() < 1e-10,
            "Joint 1 should remain near zero: {}",
            p[1]
        );
        assert!(
            p[2].abs() < 1e-10,
            "Joint 2 should remain near zero: {}",
            p[2]
        );
    }

    #[test]
    fn butterworth_handles_extra_joints_beyond_dof() {
        // Filter created for 2 joints but fed 3
        let mut filter = ButterworthLowPass::new(10.0, 500.0, 2);
        filter.filter(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);

        let (p, v) = filter.filter(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]);
        // First two joints should be filtered
        assert!(p[0] < 1.0, "Joint 0 should be smoothed");
        assert!(p[1] < 2.0, "Joint 1 should be smoothed");
        // Third joint should pass through since it's beyond the state vector
        assert!(
            (p[2] - 3.0).abs() < 1e-10,
            "Extra joint should pass through: {}",
            p[2]
        );
        assert!(
            (v[2] - 0.3).abs() < 1e-10,
            "Extra joint velocity should pass through: {}",
            v[2]
        );
    }

    #[test]
    fn butterworth_velocity_smoothing() {
        let mut filter = ButterworthLowPass::new(10.0, 500.0, 1);
        filter.filter(&[0.0], &[0.0]);

        // Step velocity to 1.0
        let (_, v) = filter.filter(&[0.0], &[1.0]);
        assert!(
            v[0] > 0.0 && v[0] < 1.0,
            "Velocity step should be smoothed: {}",
            v[0]
        );
    }

    // --- NoFilter additional tests ---

    #[test]
    fn no_filter_multi_joint() {
        let mut filter = NoFilter;
        let pos = vec![1.0, -2.0, 3.0, -4.0];
        let vel = vec![0.1, -0.2, 0.3, -0.4];
        let (fp, fv) = filter.filter(&pos, &vel);
        assert_eq!(fp, pos);
        assert_eq!(fv, vel);
    }

    #[test]
    fn no_filter_reset_is_noop() {
        let mut filter = NoFilter;
        filter.filter(&[1.0], &[2.0]);
        filter.reset();
        let (p, v) = filter.filter(&[3.0], &[4.0]);
        assert_eq!(p, vec![3.0]);
        assert_eq!(v, vec![4.0]);
    }

    #[test]
    fn no_filter_empty_input() {
        let mut filter = NoFilter;
        let (p, v) = filter.filter(&[], &[]);
        assert!(p.is_empty());
        assert!(v.is_empty());
    }

    #[test]
    fn ema_empty_input() {
        let mut filter = ExponentialMovingAverage::new(0.5);
        let (p, v) = filter.filter(&[], &[]);
        assert!(p.is_empty());
        assert!(v.is_empty());
    }
}
