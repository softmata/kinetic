//! Twist (6-DOF velocity) type.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A 6-DOF velocity: linear velocity + angular velocity.
///
/// Used for Cartesian velocity commands in servo mode and Jacobian computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Twist {
    /// Linear velocity (m/s) in x, y, z.
    pub linear: Vector3<f64>,
    /// Angular velocity (rad/s) around x, y, z.
    pub angular: Vector3<f64>,
}

impl Twist {
    /// Create a twist with both linear and angular components.
    pub fn new(linear: Vector3<f64>, angular: Vector3<f64>) -> Self {
        Self { linear, angular }
    }

    /// Create a twist with linear velocity only (no rotation).
    pub fn linear(x: f64, y: f64, z: f64) -> Self {
        Self {
            linear: Vector3::new(x, y, z),
            angular: Vector3::zeros(),
        }
    }

    /// Create a twist with angular velocity only (no translation).
    pub fn angular(x: f64, y: f64, z: f64) -> Self {
        Self {
            linear: Vector3::zeros(),
            angular: Vector3::new(x, y, z),
        }
    }

    /// Zero twist (no motion).
    pub fn zero() -> Self {
        Self {
            linear: Vector3::zeros(),
            angular: Vector3::zeros(),
        }
    }

    /// Construct from a 6-element slice [vx, vy, vz, wx, wy, wz].
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 6, "Twist requires at least 6 elements");
        Self {
            linear: Vector3::new(s[0], s[1], s[2]),
            angular: Vector3::new(s[3], s[4], s[5]),
        }
    }

    /// Convert to a 6-element array [vx, vy, vz, wx, wy, wz].
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.linear.x,
            self.linear.y,
            self.linear.z,
            self.angular.x,
            self.angular.y,
            self.angular.z,
        ]
    }

    /// Magnitude of the linear component.
    pub fn linear_magnitude(&self) -> f64 {
        self.linear.norm()
    }

    /// Magnitude of the angular component.
    pub fn angular_magnitude(&self) -> f64 {
        self.angular.norm()
    }

    /// Scale this twist by a scalar factor.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            linear: self.linear * factor,
            angular: self.angular * factor,
        }
    }
}

impl Default for Twist {
    fn default() -> Self {
        Self::zero()
    }
}

// softmata-core conversions
impl From<softmata_core::geometry::Twist3D> for Twist {
    fn from(t: softmata_core::geometry::Twist3D) -> Self {
        Self {
            linear: Vector3::new(t.linear.x, t.linear.y, t.linear.z),
            angular: Vector3::new(t.angular.x, t.angular.y, t.angular.z),
        }
    }
}

impl Twist {
    /// Convert to softmata-core Twist3D.
    pub fn to_core(&self) -> softmata_core::geometry::Twist3D {
        softmata_core::geometry::Twist3D::new(
            softmata_core::geometry::Vector3::new(self.linear.x, self.linear.y, self.linear.z),
            softmata_core::geometry::Vector3::new(self.angular.x, self.angular.y, self.angular.z),
        )
    }
}

impl std::ops::Add for Twist {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular,
        }
    }
}

impl std::ops::Sub for Twist {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular,
        }
    }
}

impl std::ops::Neg for Twist {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            linear: -self.linear,
            angular: -self.angular,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_twist() {
        let t = Twist::zero();
        assert_eq!(t.linear, Vector3::zeros());
        assert_eq!(t.angular, Vector3::zeros());
    }

    #[test]
    fn linear_only() {
        let t = Twist::linear(1.0, 0.0, 0.0);
        assert!((t.linear_magnitude() - 1.0).abs() < 1e-10);
        assert!((t.angular_magnitude() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn from_slice_roundtrip() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Twist::from_slice(&arr);
        let back = t.to_array();
        assert_eq!(arr, back);
    }

    #[test]
    fn scale() {
        let t = Twist::linear(1.0, 2.0, 3.0);
        let scaled = t.scale(2.0);
        assert!((scaled.linear.x - 2.0).abs() < 1e-10);
        assert!((scaled.linear.y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn add_sub() {
        let a = Twist::linear(1.0, 0.0, 0.0);
        let b = Twist::angular(0.0, 1.0, 0.0);
        let sum = a.clone() + b.clone();
        assert!((sum.linear.x - 1.0).abs() < 1e-10);
        assert!((sum.angular.y - 1.0).abs() < 1e-10);

        let diff = sum - b;
        assert!((diff.linear.x - 1.0).abs() < 1e-10);
        assert!((diff.angular.y).abs() < 1e-10);
    }

    #[test]
    fn negate() {
        let t = Twist::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        let neg = -t;
        assert!((neg.linear.x + 1.0).abs() < 1e-10);
        assert!((neg.angular.z + 6.0).abs() < 1e-10);
    }

    #[test]
    fn default_is_zero() {
        let t = Twist::default();
        assert_eq!(t, Twist::zero());
    }

    #[test]
    fn angular_only() {
        let t = Twist::angular(0.0, 0.0, 1.0);
        assert!((t.angular_magnitude() - 1.0).abs() < 1e-10);
        assert!((t.linear_magnitude()).abs() < 1e-10);
    }

    #[test]
    fn new_constructor() {
        let t = Twist::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        assert!((t.linear.x - 1.0).abs() < 1e-10);
        assert!((t.angular.z - 6.0).abs() < 1e-10);
    }
}
