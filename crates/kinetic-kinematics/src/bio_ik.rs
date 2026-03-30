//! Bio-IK: evolutionary IK solver for highly redundant kinematic chains.
//!
//! Uses a population-based evolutionary strategy (CMA-ES inspired) to find
//! IK solutions. Particularly effective for:
//! - Redundant robots (7+ DOF) where null-space exploration is needed
//! - Multi-objective IK (position + orientation + secondary goals)
//! - Escaping local minima that trap gradient-based solvers
//!
//! # Algorithm
//!
//! 1. Initialize population of N random joint configurations.
//! 2. Each generation:
//!    a. Evaluate fitness (pose error + secondary objectives).
//!    b. Select best individuals.
//!    c. Mutate: perturb with Gaussian noise, variance adapted per generation.
//!    d. Crossover: blend best individuals.
//!    e. Elitism: keep top K unchanged.
//! 3. Return best individual when converged or timeout.

use kinetic_core::Pose;
use kinetic_robot::Robot;
use rand::Rng;

use crate::forward::forward_kinematics;
use crate::ik::IKSolution;
use crate::KinematicChain;

/// Bio-IK configuration.
#[derive(Debug, Clone)]
pub struct BioIKConfig {
    /// Population size (default: 50).
    pub population_size: usize,
    /// Maximum generations (default: 200).
    pub max_generations: usize,
    /// Initial mutation sigma (default: 0.3 radians).
    pub initial_sigma: f64,
    /// Sigma decay per generation (default: 0.995).
    pub sigma_decay: f64,
    /// Minimum sigma (default: 0.01).
    pub min_sigma: f64,
    /// Elitism: keep top N individuals unchanged (default: 2).
    pub elite_count: usize,
    /// Position error convergence threshold in meters (default: 1e-4).
    pub position_tolerance: f64,
    /// Orientation error convergence threshold in radians (default: 1e-3).
    pub orientation_tolerance: f64,
    /// Weight for joint centering secondary objective (default: 0.01).
    pub joint_centering_weight: f64,
}

impl Default for BioIKConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 200,
            initial_sigma: 0.3,
            sigma_decay: 0.995,
            min_sigma: 0.01,
            elite_count: 2,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            joint_centering_weight: 0.01,
        }
    }
}

/// An individual in the population.
#[derive(Debug, Clone)]
struct Individual {
    joints: Vec<f64>,
    fitness: f64,
}

/// Solve IK using Bio-IK evolutionary algorithm.
pub fn solve_bio_ik(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    seed: &[f64],
    config: &BioIKConfig,
) -> Option<IKSolution> {
    let dof = chain.dof;
    let mut rng = rand::thread_rng();

    // Get joint limits
    let limits: Vec<(f64, f64)> = chain.active_joints.iter().map(|&ji| {
        robot.joints[ji].limits.as_ref()
            .map(|l| (l.lower, l.upper))
            .unwrap_or((-std::f64::consts::PI, std::f64::consts::PI))
    }).collect();

    let joint_centers: Vec<f64> = limits.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
    let joint_ranges: Vec<f64> = limits.iter().map(|(lo, hi)| (hi - lo).max(1e-6)).collect();

    // Initialize population
    let mut population: Vec<Individual> = Vec::with_capacity(config.population_size);

    // Seed individual
    population.push(Individual {
        joints: seed.to_vec(),
        fitness: f64::INFINITY,
    });

    // Random individuals
    for _ in 1..config.population_size {
        let joints: Vec<f64> = (0..dof).map(|j| {
            let (lo, hi) = limits[j];
            let range = hi - lo;
            if range.is_finite() && range < 100.0 {
                rng.gen_range(lo..=hi)
            } else {
                rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
            }
        }).collect();
        population.push(Individual { joints, fitness: f64::INFINITY });
    }

    let mut sigma = config.initial_sigma;
    let mut best_ever = Individual { joints: seed.to_vec(), fitness: f64::INFINITY };

    for _gen in 0..config.max_generations {
        // Evaluate fitness
        for ind in &mut population {
            ind.fitness = evaluate_fitness(
                robot, chain, target, &ind.joints,
                &joint_centers, &joint_ranges, config.joint_centering_weight,
            );
        }

        // Sort by fitness (lower = better)
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Update best
        if population[0].fitness < best_ever.fitness {
            best_ever = population[0].clone();
        }

        // Check convergence
        if let Some(sol) = check_convergence(robot, chain, target, &best_ever.joints, config) {
            return Some(sol);
        }

        // Selection + mutation + crossover
        let mut next_gen = Vec::with_capacity(config.population_size);

        // Elitism
        for i in 0..config.elite_count.min(population.len()) {
            next_gen.push(population[i].clone());
        }

        // Fill rest with mutated/crossed offspring
        while next_gen.len() < config.population_size {
            let parent_idx = rng.gen_range(0..population.len() / 2 + 1); // bias toward better
            let mut child = population[parent_idx].joints.clone();

            // Crossover with another parent (50% chance)
            if rng.gen_bool(0.5) && population.len() > 1 {
                let other_idx = rng.gen_range(0..population.len() / 2 + 1);
                let alpha = rng.gen_range(0.3..0.7);
                for j in 0..dof {
                    child[j] = alpha * child[j] + (1.0 - alpha) * population[other_idx].joints[j];
                }
            }

            // Mutation
            for j in 0..dof {
                child[j] += rng.gen_range(-sigma..sigma);
                child[j] = child[j].clamp(limits[j].0, limits[j].1);
            }

            next_gen.push(Individual { joints: child, fitness: f64::INFINITY });
        }

        population = next_gen;
        sigma = (sigma * config.sigma_decay).max(config.min_sigma);
    }

    // Return best found
    check_convergence(robot, chain, target, &best_ever.joints, config)
        .or_else(|| {
            // Return even if not converged (best effort)
            let pose = forward_kinematics(robot, chain, &best_ever.joints).ok()?;
            let pos_err = (pose.translation() - target.translation()).norm();
            let rot_diff = pose.0.rotation.inverse() * target.0.rotation;
            let orient_err = rot_diff.angle();
            Some(IKSolution {
                joints: best_ever.joints,
                position_error: pos_err,
                orientation_error: orient_err,
                iterations: config.max_generations,
                converged: false,
                mode_used: crate::ik::IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
            })
        })
}

fn evaluate_fitness(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    joints: &[f64],
    joint_centers: &[f64],
    joint_ranges: &[f64],
    centering_weight: f64,
) -> f64 {
    let pose = match forward_kinematics(robot, chain, joints) {
        Ok(p) => p,
        Err(_) => return f64::INFINITY,
    };

    let pos_err = (pose.translation() - target.translation()).norm();
    let rot_diff = pose.0.rotation.inverse() * target.0.rotation;
    let orient_err = rot_diff.angle();

    // Joint centering: penalize deviation from center of range
    let centering: f64 = joints.iter().zip(joint_centers).zip(joint_ranges)
        .map(|((j, c), r)| ((j - c) / r).powi(2))
        .sum::<f64>() / joints.len() as f64;

    pos_err + orient_err * 0.1 + centering_weight * centering
}

fn check_convergence(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    joints: &[f64],
    config: &BioIKConfig,
) -> Option<IKSolution> {
    let pose = forward_kinematics(robot, chain, joints).ok()?;
    let pos_err = (pose.translation() - target.translation()).norm();
    let rot_diff = pose.0.rotation.inverse() * target.0.rotation;
    let orient_err = rot_diff.angle();

    if pos_err < config.position_tolerance && orient_err < config.orientation_tolerance {
        Some(IKSolution {
            joints: joints.to_vec(),
            position_error: pos_err,
            orientation_error: orient_err,
            iterations: 0,
            converged: true,
            mode_used: crate::ik::IKMode::Full6D,
        degraded: false,
        condition_number: f64::INFINITY,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="test3dof">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/><child link="tip"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="10"/>
  </joint>
</robot>"#;

    #[test]
    fn bio_ik_finds_solution() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();

        // Target: tip at a known reachable position
        let seed = [0.5, 0.3, -0.2];
        let target_pose = forward_kinematics(&robot, &chain, &seed).unwrap();

        // Solve from different seed
        let config = BioIKConfig {
            population_size: 30,
            max_generations: 100,
            ..Default::default()
        };

        let result = solve_bio_ik(&robot, &chain, &target_pose, &[0.0; 3], &config);
        assert!(result.is_some(), "Bio-IK should find a solution");

        let sol = result.unwrap();
        assert!(sol.position_error < 0.01, "Position error: {}", sol.position_error);
    }

    #[test]
    fn bio_ik_returns_best_effort() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();

        // Unreachable target (way too far)
        let target = Pose::from_xyz(10.0, 10.0, 10.0);
        let config = BioIKConfig {
            max_generations: 10,
            population_size: 10,
            ..Default::default()
        };

        let result = solve_bio_ik(&robot, &chain, &target, &[0.0; 3], &config);
        assert!(result.is_some(), "Should return best-effort even if not converged");
        assert!(!result.unwrap().converged);
    }

    #[test]
    fn bio_ik_respects_joint_limits() {
        // Intent: all solutions must be within URDF joint limits
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let target = Pose::from_xyz(0.2, 0.1, 0.4);
        let config = BioIKConfig {
            population_size: 40,
            max_generations: 50,
            ..Default::default()
        };
        let result = solve_bio_ik(&robot, &chain, &target, &[0.0; 3], &config);
        if let Some(sol) = result {
            assert!(sol.joints[0] >= -3.14 && sol.joints[0] <= 3.14, "j0={}", sol.joints[0]);
            assert!(sol.joints[1] >= -2.0 && sol.joints[1] <= 2.0, "j1={}", sol.joints[1]);
            assert!(sol.joints[2] >= -2.5 && sol.joints[2] <= 2.5, "j2={}", sol.joints[2]);
        }
    }

    #[test]
    fn bio_ik_config_default() {
        let config = BioIKConfig::default();
        assert_eq!(config.population_size, 50);
        assert_eq!(config.max_generations, 200);
        assert!(config.initial_sigma > 0.0);
        assert!(config.elite_count > 0);
    }

    #[test]
    fn bio_ik_from_known_seed_converges_fast() {
        // Intent: starting from near the solution should converge quickly
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let seed = [0.5, 0.3, -0.2];
        let target = forward_kinematics(&robot, &chain, &seed).unwrap();
        let config = BioIKConfig {
            population_size: 20,
            max_generations: 30,
            ..Default::default()
        };
        // Start from near the solution
        let near_seed = [0.52, 0.28, -0.18];
        let result = solve_bio_ik(&robot, &chain, &target, &near_seed, &config);
        assert!(result.is_some());
        let sol = result.unwrap();
        assert!(sol.position_error < 0.01, "near seed should converge: err={}", sol.position_error);
    }

    #[test]
    fn bio_ik_returns_finite_values() {
        // Intent: no NaN or Inf in solution joints or errors
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "tip").unwrap();
        let target = Pose::from_xyz(0.15, 0.0, 0.45);
        let config = BioIKConfig::default();
        let result = solve_bio_ik(&robot, &chain, &target, &[0.0; 3], &config);
        if let Some(sol) = result {
            for (i, &j) in sol.joints.iter().enumerate() {
                assert!(j.is_finite(), "joint {i} is not finite: {j}");
            }
            assert!(sol.position_error.is_finite());
            assert!(sol.orientation_error.is_finite());
        }
    }
}
