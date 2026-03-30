//! Joint values type — a named vector of joint positions.

use serde::{Deserialize, Serialize};

/// A vector of joint values (positions, velocities, or accelerations).
///
/// Newtype over `Vec<f64>` providing semantic meaning and joint-indexed access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointValues(pub Vec<f64>);

impl JointValues {
    /// Create from a Vec.
    pub fn new(values: Vec<f64>) -> Self {
        Self(values)
    }

    /// Create from a slice.
    pub fn from_slice(s: &[f64]) -> Self {
        Self(s.to_vec())
    }

    /// Create zero-valued joint vector of given DOF.
    pub fn zeros(dof: usize) -> Self {
        Self(vec![0.0; dof])
    }

    /// Number of joints.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get joint value by index.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.0.get(index).copied()
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Get a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }

    /// Euclidean distance to another joint configuration.
    pub fn distance_to(&self, other: &JointValues) -> f64 {
        assert_eq!(self.len(), other.len(), "JointValues DOF mismatch");
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Linear interpolation between two joint configurations.
    pub fn lerp(&self, other: &JointValues, t: f64) -> JointValues {
        assert_eq!(self.len(), other.len(), "JointValues DOF mismatch");
        let values: Vec<f64> = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a + t * (b - a))
            .collect();
        JointValues(values)
    }
}

impl std::ops::Index<usize> for JointValues {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for JointValues {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.0[index]
    }
}

impl std::ops::Deref for JointValues {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        &self.0
    }
}

impl std::ops::DerefMut for JointValues {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

impl From<Vec<f64>> for JointValues {
    fn from(v: Vec<f64>) -> Self {
        Self(v)
    }
}

impl<const N: usize> From<[f64; N]> for JointValues {
    fn from(arr: [f64; N]) -> Self {
        Self(arr.to_vec())
    }
}

impl From<JointValues> for Vec<f64> {
    fn from(jv: JointValues) -> Self {
        jv.0
    }
}

impl AsRef<[f64]> for JointValues {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

impl softmata_core::sensor::JointData for JointValues {
    fn num_joints(&self) -> usize {
        self.0.len()
    }

    fn position_at(&self, index: usize) -> f64 {
        self.0.get(index).copied().unwrap_or(0.0)
    }

    fn velocity_at(&self, _index: usize) -> f64 {
        0.0 // JointValues stores positions only
    }

    fn effort_at(&self, _index: usize) -> f64 {
        0.0 // JointValues stores positions only
    }

    fn joint_name(&self, _index: usize) -> Option<&str> {
        None // JointValues has no joint names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let jv = JointValues::zeros(7);
        assert_eq!(jv.len(), 7);
        assert!((jv[0] - 0.0).abs() < 1e-15);
    }

    #[test]
    fn distance() {
        let a = JointValues::new(vec![0.0, 0.0, 0.0]);
        let b = JointValues::new(vec![3.0, 4.0, 0.0]);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn lerp_midpoint() {
        let a = JointValues::new(vec![0.0, 0.0]);
        let b = JointValues::new(vec![2.0, 4.0]);
        let mid = a.lerp(&b, 0.5);
        assert!((mid[0] - 1.0).abs() < 1e-10);
        assert!((mid[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn lerp_endpoints() {
        let a = JointValues::new(vec![1.0, 2.0]);
        let b = JointValues::new(vec![3.0, 4.0]);
        let at_zero = a.lerp(&b, 0.0);
        let at_one = a.lerp(&b, 1.0);
        assert!((at_zero[0] - 1.0).abs() < 1e-10);
        assert!((at_one[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn from_slice() {
        let jv = JointValues::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(jv.len(), 3);
        assert!((jv[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn is_empty() {
        let empty = JointValues::new(vec![]);
        assert!(empty.is_empty());
        let non_empty = JointValues::new(vec![1.0]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn get_method() {
        let jv = JointValues::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(jv.get(0), Some(1.0));
        assert_eq!(jv.get(2), Some(3.0));
        assert_eq!(jv.get(5), None);
    }

    #[test]
    fn as_slice_and_mut_slice() {
        let mut jv = JointValues::new(vec![1.0, 2.0]);
        assert_eq!(jv.as_slice(), &[1.0, 2.0]);
        jv.as_mut_slice()[0] = 5.0;
        assert!((jv[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn index_mut() {
        let mut jv = JointValues::new(vec![1.0, 2.0, 3.0]);
        jv[1] = 10.0;
        assert!((jv[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn from_vec_conversion() {
        let v = vec![1.0, 2.0, 3.0];
        let jv: JointValues = v.into();
        assert_eq!(jv.len(), 3);
        let back: Vec<f64> = jv.into();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_ref_trait() {
        let jv = JointValues::new(vec![1.0, 2.0]);
        let slice: &[f64] = jv.as_ref();
        assert_eq!(slice, &[1.0, 2.0]);
    }

    #[test]
    fn deref_to_slice() {
        let jv = JointValues::new(vec![1.0, 2.0, 3.0]);
        // Deref allows slice methods
        assert_eq!(jv.iter().count(), 3);
    }

    #[test]
    fn joint_data_trait() {
        use softmata_core::sensor::JointData;
        let jv = JointValues::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(jv.num_joints(), 3);
        assert!((jv.position_at(1) - 2.0).abs() < 1e-10);
        assert!((jv.position_at(99)).abs() < 1e-10); // Out-of-bounds returns 0
        assert!((jv.velocity_at(0)).abs() < 1e-10);
        assert!((jv.effort_at(0)).abs() < 1e-10);
        assert!(jv.joint_name(0).is_none());
    }
}
