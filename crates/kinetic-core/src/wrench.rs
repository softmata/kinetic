//! Wrench (6-DOF force/torque) type.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A 6-DOF wrench: force + torque.
///
/// Used for force-torque sensor data and grasp quality evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wrench {
    /// Force (N) in x, y, z.
    pub force: Vector3<f64>,
    /// Torque (Nm) around x, y, z.
    pub torque: Vector3<f64>,
}

impl Wrench {
    /// Create a wrench with both force and torque.
    pub fn new(force: Vector3<f64>, torque: Vector3<f64>) -> Self {
        Self { force, torque }
    }

    /// Create a wrench with force only.
    pub fn force_only(fx: f64, fy: f64, fz: f64) -> Self {
        Self {
            force: Vector3::new(fx, fy, fz),
            torque: Vector3::zeros(),
        }
    }

    /// Create a wrench with torque only.
    pub fn torque_only(tx: f64, ty: f64, tz: f64) -> Self {
        Self {
            force: Vector3::zeros(),
            torque: Vector3::new(tx, ty, tz),
        }
    }

    /// Zero wrench.
    pub fn zero() -> Self {
        Self {
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
        }
    }

    /// Construct from a 6-element slice [fx, fy, fz, tx, ty, tz].
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 6, "Wrench requires at least 6 elements");
        Self {
            force: Vector3::new(s[0], s[1], s[2]),
            torque: Vector3::new(s[3], s[4], s[5]),
        }
    }

    /// Convert to a 6-element array [fx, fy, fz, tx, ty, tz].
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.force.x,
            self.force.y,
            self.force.z,
            self.torque.x,
            self.torque.y,
            self.torque.z,
        ]
    }

    /// Magnitude of the force component.
    pub fn force_magnitude(&self) -> f64 {
        self.force.norm()
    }

    /// Magnitude of the torque component.
    pub fn torque_magnitude(&self) -> f64 {
        self.torque.norm()
    }
}

impl Default for Wrench {
    fn default() -> Self {
        Self::zero()
    }
}

// softmata-core conversions
impl From<softmata_core::geometry::Wrench> for Wrench {
    fn from(w: softmata_core::geometry::Wrench) -> Self {
        Self {
            force: Vector3::new(w.force.x, w.force.y, w.force.z),
            torque: Vector3::new(w.torque.x, w.torque.y, w.torque.z),
        }
    }
}

impl Wrench {
    /// Convert to softmata-core Wrench.
    pub fn to_core(&self) -> softmata_core::geometry::Wrench {
        softmata_core::geometry::Wrench::new(
            softmata_core::geometry::Vector3::new(self.force.x, self.force.y, self.force.z),
            softmata_core::geometry::Vector3::new(self.torque.x, self.torque.y, self.torque.z),
        )
    }
}

impl softmata_core::sensor::WrenchData for Wrench {
    fn force(&self) -> [f64; 3] {
        [self.force.x, self.force.y, self.force.z]
    }

    fn torque(&self) -> [f64; 3] {
        [self.torque.x, self.torque.y, self.torque.z]
    }
}

impl std::ops::Add for Wrench {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            force: self.force + rhs.force,
            torque: self.torque + rhs.torque,
        }
    }
}

impl std::ops::Neg for Wrench {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            force: -self.force,
            torque: -self.torque,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_wrench() {
        let w = Wrench::zero();
        assert_eq!(w.force, Vector3::zeros());
        assert_eq!(w.torque, Vector3::zeros());
    }

    #[test]
    fn force_only() {
        let w = Wrench::force_only(0.0, 0.0, -9.81);
        assert!((w.force_magnitude() - 9.81).abs() < 1e-10);
        assert!((w.torque_magnitude() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn from_slice_roundtrip() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = Wrench::from_slice(&arr);
        assert_eq!(arr, w.to_array());
    }

    #[test]
    fn torque_only() {
        let w = Wrench::torque_only(1.0, 0.0, 0.0);
        assert!((w.torque_magnitude() - 1.0).abs() < 1e-10);
        assert!((w.force_magnitude()).abs() < 1e-10);
    }

    #[test]
    fn new_constructor() {
        let w = Wrench::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        assert!((w.force.x - 1.0).abs() < 1e-10);
        assert!((w.torque.z - 6.0).abs() < 1e-10);
    }

    #[test]
    fn default_is_zero() {
        let w = Wrench::default();
        assert_eq!(w, Wrench::zero());
    }

    #[test]
    fn add_wrenches() {
        let a = Wrench::force_only(1.0, 0.0, 0.0);
        let b = Wrench::torque_only(0.0, 1.0, 0.0);
        let sum = a + b;
        assert!((sum.force.x - 1.0).abs() < 1e-10);
        assert!((sum.torque.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn negate_wrench() {
        let w = Wrench::force_only(5.0, 0.0, 0.0);
        let neg = -w;
        assert!((neg.force.x + 5.0).abs() < 1e-10);
    }

    #[test]
    fn wrench_data_trait() {
        use softmata_core::sensor::WrenchData;
        let w = Wrench::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        let f = w.force();
        let t = w.torque();
        assert_eq!(f, [1.0, 2.0, 3.0]);
        assert_eq!(t, [4.0, 5.0, 6.0]);
    }
}
