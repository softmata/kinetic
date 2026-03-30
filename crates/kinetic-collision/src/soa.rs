//! Structure-of-Arrays (SoA) sphere storage for SIMD-friendly collision.
//!
//! Stores sphere data in separate contiguous arrays (x[], y[], z[], r[])
//! rather than an array-of-structs. This layout enables efficient SIMD
//! vectorization for batch distance computations.

/// SoA storage for bounding spheres.
///
/// Each sphere is defined by a center (x, y, z) and a radius.
/// The `link_id` array maps each sphere back to its owning link index.
///
/// All arrays have the same length.
#[derive(Debug, Clone)]
pub struct SpheresSoA {
    /// X coordinates of sphere centers.
    pub x: Vec<f64>,
    /// Y coordinates of sphere centers.
    pub y: Vec<f64>,
    /// Z coordinates of sphere centers.
    pub z: Vec<f64>,
    /// Sphere radii.
    pub radius: Vec<f64>,
    /// Link index that owns each sphere.
    pub link_id: Vec<usize>,
}

impl SpheresSoA {
    /// Create empty SoA storage.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            radius: Vec::new(),
            link_id: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            radius: Vec::with_capacity(capacity),
            link_id: Vec::with_capacity(capacity),
        }
    }

    /// Number of spheres.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether this storage is empty.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Add a sphere.
    pub fn push(&mut self, x: f64, y: f64, z: f64, radius: f64, link_id: usize) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.radius.push(radius);
        self.link_id.push(link_id);
    }

    /// Clear all spheres (retains allocated memory).
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
        self.radius.clear();
        self.link_id.clear();
    }

    /// Compute squared distance between sphere `i` in `self` and sphere `j` in `other`.
    ///
    /// Returns the squared center-to-center distance minus the sum of radii.
    /// Negative values indicate overlap.
    #[inline]
    pub fn signed_distance_sq(&self, i: usize, other: &SpheresSoA, j: usize) -> f64 {
        let dx = self.x[i] - other.x[j];
        let dy = self.y[i] - other.y[j];
        let dz = self.z[i] - other.z[j];
        let center_dist_sq = dx * dx + dy * dy + dz * dz;
        let radii_sum = self.radius[i] + other.radius[j];
        center_dist_sq - radii_sum * radii_sum
    }

    /// Check if sphere `i` in `self` overlaps sphere `j` in `other`.
    #[inline]
    pub fn overlaps(&self, i: usize, other: &SpheresSoA, j: usize) -> bool {
        self.signed_distance_sq(i, other, j) < 0.0
    }

    /// Check if sphere `i` in `self` overlaps sphere `j` in `other` with a margin.
    ///
    /// Returns true if the gap between the spheres is less than `margin`.
    #[inline]
    pub fn overlaps_with_margin(
        &self,
        i: usize,
        other: &SpheresSoA,
        j: usize,
        margin: f64,
    ) -> bool {
        let dx = self.x[i] - other.x[j];
        let dy = self.y[i] - other.y[j];
        let dz = self.z[i] - other.z[j];
        let center_dist_sq = dx * dx + dy * dy + dz * dz;
        let threshold = self.radius[i] + other.radius[j] + margin;
        center_dist_sq < threshold * threshold
    }

    /// Compute the signed distance between sphere `i` in `self` and sphere `j` in `other`.
    ///
    /// Positive = gap between surfaces, negative = penetration depth.
    #[inline]
    pub fn signed_distance(&self, i: usize, other: &SpheresSoA, j: usize) -> f64 {
        let dx = self.x[i] - other.x[j];
        let dy = self.y[i] - other.y[j];
        let dz = self.z[i] - other.z[j];
        let center_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        center_dist - self.radius[i] - other.radius[j]
    }

    /// Find the minimum signed distance between any sphere in `self` and any in `other`.
    ///
    /// Returns (distance, self_index, other_index).
    /// Negative distance means penetration.
    pub fn min_distance(&self, other: &SpheresSoA) -> Option<(f64, usize, usize)> {
        if self.is_empty() || other.is_empty() {
            return None;
        }

        // First pass: use cheap squared-distance comparison to find closest pair
        let mut best_dist_sq = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..self.len() {
            for j in 0..other.len() {
                let d = self.signed_distance_sq(i, other, j);
                if d < best_dist_sq {
                    best_dist_sq = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Compute exact signed distance for the closest pair
        let dist = self.signed_distance(best_i, other, best_j);
        Some((dist, best_i, best_j))
    }

    /// Check if any sphere in `self` overlaps any sphere in `other`.
    pub fn any_overlap(&self, other: &SpheresSoA) -> bool {
        for i in 0..self.len() {
            for j in 0..other.len() {
                if self.overlaps(i, other, j) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if any sphere in `self` is within `margin` of any in `other`.
    pub fn any_overlap_with_margin(&self, other: &SpheresSoA, margin: f64) -> bool {
        for i in 0..self.len() {
            for j in 0..other.len() {
                if self.overlaps_with_margin(i, other, j, margin) {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for SpheresSoA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_soa() {
        let soa = SpheresSoA::new();
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
    }

    #[test]
    fn push_and_len() {
        let mut soa = SpheresSoA::new();
        soa.push(1.0, 2.0, 3.0, 0.5, 0);
        soa.push(4.0, 5.0, 6.0, 0.3, 1);
        assert_eq!(soa.len(), 2);
        assert!(!soa.is_empty());
        assert_eq!(soa.x[0], 1.0);
        assert_eq!(soa.radius[1], 0.3);
        assert_eq!(soa.link_id[1], 1);
    }

    #[test]
    fn clear_retains_capacity() {
        let mut soa = SpheresSoA::with_capacity(100);
        soa.push(1.0, 2.0, 3.0, 0.5, 0);
        soa.clear();
        assert!(soa.is_empty());
        assert!(soa.x.capacity() >= 100);
    }

    #[test]
    fn overlapping_spheres() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 1.0, 0); // sphere at origin, radius 1

        let mut b = SpheresSoA::new();
        b.push(1.5, 0.0, 0.0, 1.0, 1); // sphere at x=1.5, radius 1

        // Distance between centers = 1.5, sum of radii = 2.0 → overlap
        assert!(a.overlaps(0, &b, 0));
        assert!(a.any_overlap(&b));
    }

    #[test]
    fn non_overlapping_spheres() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);

        let mut b = SpheresSoA::new();
        b.push(2.0, 0.0, 0.0, 0.5, 1);

        // Distance = 2.0, radii sum = 1.0 → no overlap
        assert!(!a.overlaps(0, &b, 0));
        assert!(!a.any_overlap(&b));
    }

    #[test]
    fn margin_check() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);

        let mut b = SpheresSoA::new();
        b.push(1.5, 0.0, 0.0, 0.5, 1);

        // Gap = 1.5 - 0.5 - 0.5 = 0.5
        assert!(!a.overlaps(0, &b, 0)); // no direct overlap
        assert!(a.overlaps_with_margin(0, &b, 0, 0.6)); // within 0.6 margin
        assert!(!a.overlaps_with_margin(0, &b, 0, 0.4)); // not within 0.4
    }

    #[test]
    fn min_distance_overlap() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 1.0, 0);

        let mut b = SpheresSoA::new();
        b.push(1.5, 0.0, 0.0, 1.0, 1);

        let (dist, i, j) = a.min_distance(&b).unwrap();
        assert!(
            dist < 0.0,
            "Expected negative distance for overlap, got {}",
            dist
        );
        assert!((dist - (-0.5)).abs() < 1e-10); // penetration = 1.5 - 2.0 = -0.5
        assert_eq!(i, 0);
        assert_eq!(j, 0);
    }

    #[test]
    fn min_distance_separated() {
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);

        let mut b = SpheresSoA::new();
        b.push(3.0, 0.0, 0.0, 0.5, 1);

        let (dist, _, _) = a.min_distance(&b).unwrap();
        // Center distance = 3.0, gap = 3.0 - 0.5 - 0.5 = 2.0
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn min_distance_empty() {
        let a = SpheresSoA::new();
        let b = SpheresSoA::new();
        assert!(a.min_distance(&b).is_none());
    }

    #[test]
    fn self_overlap_check() {
        // Test checking a set against itself (for self-collision)
        let mut a = SpheresSoA::new();
        a.push(0.0, 0.0, 0.0, 0.5, 0);
        a.push(0.8, 0.0, 0.0, 0.5, 1);
        a.push(5.0, 0.0, 0.0, 0.5, 2);

        // Spheres 0 and 1 overlap (distance=0.8, radii sum=1.0)
        assert!(a.overlaps(0, &a, 1));
        // Spheres 0 and 2 don't overlap
        assert!(!a.overlaps(0, &a, 2));
    }
}
