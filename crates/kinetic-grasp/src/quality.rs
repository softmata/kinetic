//! Force-closure quality estimates and candidate ranking.

use crate::{GraspCandidate, GraspConfig, GraspMetric};

/// Estimate force closure quality for cylinder antipodal grasp.
pub(crate) fn estimate_force_closure_cylinder(radius: f64, max_opening: f64) -> f64 {
    // Quality increases when object fills more of the gripper
    let fill_ratio = (2.0 * radius) / max_opening;
    // Best when fill ratio is ~0.6-0.8 (good contact without being at limits)
    let optimal = 0.7;
    let deviation = (fill_ratio - optimal).abs();
    (1.0 - deviation * 2.0).clamp(0.0, 1.0)
}

/// Estimate force closure quality for box antipodal grasp.
pub(crate) fn estimate_force_closure_box(half_dim: f64, max_opening: f64) -> f64 {
    let fill_ratio = (2.0 * half_dim) / max_opening;
    let optimal = 0.6;
    let deviation = (fill_ratio - optimal).abs();
    (1.0 - deviation * 2.5).clamp(0.0, 1.0)
}

/// Estimate force closure quality for sphere antipodal grasp.
pub(crate) fn estimate_force_closure_sphere(radius: f64, max_opening: f64) -> f64 {
    let fill_ratio = (2.0 * radius) / max_opening;
    // Spheres are harder to grasp — lower base quality
    let optimal = 0.65;
    let deviation = (fill_ratio - optimal).abs();
    (0.8 - deviation * 2.0).clamp(0.0, 1.0)
}

/// Rank candidates by the configured metric.
pub(crate) fn rank_candidates(candidates: &mut [GraspCandidate], config: &GraspConfig) {
    match config.rank_by {
        GraspMetric::ForceClosureQuality => {
            // Already stored in quality field
        }
        GraspMetric::DistanceFromCenterOfMass => {
            // Re-score by proximity to origin (object center)
            for c in candidates.iter_mut() {
                let dist = c.grasp_pose.translation.vector.norm();
                c.quality = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);
            }
        }
        GraspMetric::ApproachAngle => {
            // Re-score by alignment with preferred approach axis
            for c in candidates.iter_mut() {
                let alignment = c.approach_direction.dot(&config.approach_axis).abs();
                c.quality = alignment.clamp(0.0, 1.0);
            }
        }
    }

    // Sort descending by quality
    candidates.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
