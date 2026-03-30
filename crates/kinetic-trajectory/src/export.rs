//! Export trajectories to JSON and CSV formats.
//!
//! Enables piping trajectory data to external visualization tools,
//! logging systems, or analysis scripts.

use crate::TimedTrajectory;

/// Export a trajectory to JSON format.
///
/// Returns a JSON string with the structure:
/// ```json
/// {
///   "dof": 6,
///   "duration": 2.5,
///   "num_waypoints": 100,
///   "waypoints": [
///     {"time": 0.0, "positions": [...], "velocities": [...], "accelerations": [...]},
///     ...
///   ]
/// }
/// ```
pub fn trajectory_to_json(traj: &TimedTrajectory) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"dof\": {},\n", traj.dof));
    json.push_str(&format!(
        "  \"duration\": {:.6},\n",
        traj.duration.as_secs_f64()
    ));
    json.push_str(&format!("  \"num_waypoints\": {},\n", traj.waypoints.len()));
    json.push_str("  \"waypoints\": [\n");

    for (i, wp) in traj.waypoints.iter().enumerate() {
        json.push_str("    {");
        json.push_str(&format!("\"time\": {:.6}", wp.time));
        json.push_str(&format!(
            ", \"positions\": [{}]",
            wp.positions
                .iter()
                .map(|v| format!("{:.8}", v))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        json.push_str(&format!(
            ", \"velocities\": [{}]",
            wp.velocities
                .iter()
                .map(|v| format!("{:.8}", v))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        json.push_str(&format!(
            ", \"accelerations\": [{}]",
            wp.accelerations
                .iter()
                .map(|v| format!("{:.8}", v))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        json.push('}');
        if i + 1 < traj.waypoints.len() {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n");
    json.push('}');
    json
}

/// Export a trajectory to CSV format.
///
/// Header: `time,j0,j1,...,v0,v1,...,a0,a1,...`
/// One row per waypoint.
pub fn trajectory_to_csv(traj: &TimedTrajectory) -> String {
    let dof = traj.dof;
    let mut csv = String::new();

    // Header
    csv.push_str("time");
    for j in 0..dof {
        csv.push_str(&format!(",j{}", j));
    }
    for j in 0..dof {
        csv.push_str(&format!(",v{}", j));
    }
    for j in 0..dof {
        csv.push_str(&format!(",a{}", j));
    }
    csv.push('\n');

    // Data rows
    for wp in &traj.waypoints {
        csv.push_str(&format!("{:.6}", wp.time));
        for p in &wp.positions {
            csv.push_str(&format!(",{:.8}", p));
        }
        for v in &wp.velocities {
            csv.push_str(&format!(",{:.8}", v));
        }
        for a in &wp.accelerations {
            csv.push_str(&format!(",{:.8}", a));
        }
        csv.push('\n');
    }

    csv
}

/// Export a trajectory to a JSON file.
pub fn trajectory_to_json_file(
    traj: &TimedTrajectory,
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    std::fs::write(path, trajectory_to_json(traj))
}

/// Export a trajectory to a CSV file.
pub fn trajectory_to_csv_file(
    traj: &TimedTrajectory,
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    std::fs::write(path, trajectory_to_csv(traj))
}

/// Import a trajectory from JSON string.
///
/// Parses the format produced by `trajectory_to_json`.
pub fn trajectory_from_json(json: &str) -> Result<TimedTrajectory, String> {
    // Minimal JSON parser using serde_json (available in workspace)
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;

    let dof = value["dof"].as_u64().ok_or("missing 'dof' field")? as usize;
    let duration_secs = value["duration"]
        .as_f64()
        .ok_or("missing 'duration' field")?;

    let wps = value["waypoints"]
        .as_array()
        .ok_or("missing 'waypoints' array")?;

    let mut waypoints = Vec::with_capacity(wps.len());
    for wp_val in wps {
        let time = wp_val["time"]
            .as_f64()
            .ok_or("missing 'time' in waypoint")?;
        let positions = parse_f64_array(&wp_val["positions"])?;
        let velocities = parse_f64_array(&wp_val["velocities"]).unwrap_or_else(|_| vec![0.0; dof]);
        let accelerations =
            parse_f64_array(&wp_val["accelerations"]).unwrap_or_else(|_| vec![0.0; dof]);

        if positions.len() != dof {
            return Err(format!(
                "positions length {} doesn't match dof {}",
                positions.len(),
                dof
            ));
        }

        waypoints.push(crate::TimedWaypoint {
            time,
            positions,
            velocities,
            accelerations,
        });
    }

    Ok(TimedTrajectory {
        duration: std::time::Duration::from_secs_f64(duration_secs),
        dof,
        waypoints,
    })
}

fn parse_f64_array(value: &serde_json::Value) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| "expected array".to_string())?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "expected number".to_string()))
        .collect()
}

/// Import a trajectory from CSV string.
///
/// Parses the format produced by `trajectory_to_csv`.
/// Handles positions-only CSV (infers zero velocities/accelerations).
pub fn trajectory_from_csv(csv: &str) -> Result<TimedTrajectory, String> {
    let mut lines = csv.lines();

    // Parse header to determine DOF
    let header = lines.next().ok_or("empty CSV")?;
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

    // Count position columns (j0, j1, ...)
    let dof = cols.iter().filter(|c| c.starts_with('j')).count();
    if dof == 0 {
        return Err("no joint columns found (expected j0, j1, ...)".into());
    }

    let has_velocities = cols.iter().any(|c| c.starts_with('v'));
    let has_accelerations = cols.iter().any(|c| c.starts_with('a'));

    let mut waypoints = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let vals: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("CSV parse error: {e}"))?;

        if vals.is_empty() {
            continue;
        }

        let time = vals[0];
        let positions = vals[1..1 + dof].to_vec();
        let velocities = if has_velocities && vals.len() >= 1 + 2 * dof {
            vals[1 + dof..1 + 2 * dof].to_vec()
        } else {
            vec![0.0; dof]
        };
        let accelerations = if has_accelerations && vals.len() >= 1 + 3 * dof {
            vals[1 + 2 * dof..1 + 3 * dof].to_vec()
        } else {
            vec![0.0; dof]
        };

        waypoints.push(crate::TimedWaypoint {
            time,
            positions,
            velocities,
            accelerations,
        });
    }

    if waypoints.is_empty() {
        return Err("no waypoints found in CSV".into());
    }

    let duration = waypoints.last().unwrap().time;

    Ok(TimedTrajectory {
        duration: std::time::Duration::from_secs_f64(duration),
        dof,
        waypoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimedWaypoint;
    use std::time::Duration;

    fn sample_traj() -> TimedTrajectory {
        TimedTrajectory {
            duration: Duration::from_secs_f64(0.2),
            dof: 3,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0, 0.1, 0.2],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.1,
                    positions: vec![0.1, 0.2, 0.3],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.2,
                    positions: vec![0.2, 0.3, 0.4],
                    velocities: vec![0.0, 0.0, 0.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn json_roundtrip() {
        let traj = sample_traj();
        let json = trajectory_to_json(&traj);
        let parsed = trajectory_from_json(&json).unwrap();
        assert_eq!(parsed.dof, 3);
        assert_eq!(parsed.waypoints.len(), 3);
        assert!((parsed.waypoints[1].positions[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn csv_roundtrip() {
        let traj = sample_traj();
        let csv = trajectory_to_csv(&traj);
        let parsed = trajectory_from_csv(&csv).unwrap();
        assert_eq!(parsed.dof, 3);
        assert_eq!(parsed.waypoints.len(), 3);
        assert!((parsed.waypoints[2].positions[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn json_valid_format() {
        let traj = sample_traj();
        let json = trajectory_to_json(&traj);
        // Should be valid JSON
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn csv_header_correct() {
        let traj = sample_traj();
        let csv = trajectory_to_csv(&traj);
        let first_line = csv.lines().next().unwrap();
        assert!(first_line.starts_with("time,j0,j1,j2,v0,v1,v2,a0,a1,a2"));
    }

    #[test]
    fn empty_trajectory_json() {
        let traj = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 3,
            waypoints: vec![],
        };
        let json = trajectory_to_json(&traj);
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn csv_positions_only_import() {
        let csv = "time,j0,j1\n0.0,0.0,0.0\n0.5,1.0,1.0\n";
        let traj = trajectory_from_csv(csv).unwrap();
        assert_eq!(traj.dof, 2);
        assert_eq!(traj.waypoints.len(), 2);
        assert_eq!(traj.waypoints[0].velocities, vec![0.0, 0.0]);
    }

    #[test]
    fn json_invalid_format() {
        let result = trajectory_from_json("not json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON parse error"));
    }

    #[test]
    fn json_missing_dof() {
        let result = trajectory_from_json(r#"{"waypoints": []}"#);
        assert!(result.is_err());
    }

    #[test]
    fn json_missing_waypoints() {
        let result = trajectory_from_json(r#"{"dof": 3, "duration": 1.0}"#);
        assert!(result.is_err());
    }

    #[test]
    fn json_wrong_dof_in_waypoint() {
        let json = r#"{"dof": 3, "duration": 0.1, "waypoints": [
            {"time": 0.0, "positions": [0.0, 0.0]}
        ]}"#;
        let result = trajectory_from_json(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("doesn't match dof"));
    }

    #[test]
    fn csv_empty_string() {
        let result = trajectory_from_csv("");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty CSV"));
    }

    #[test]
    fn csv_header_only_no_data() {
        let result = trajectory_from_csv("time,j0,j1\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no waypoints"));
    }

    #[test]
    fn csv_no_joint_columns() {
        let result = trajectory_from_csv("time,foo,bar\n0.0,1.0,2.0\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no joint columns"));
    }

    #[test]
    fn csv_non_numeric_value() {
        let result = trajectory_from_csv("time,j0\n0.0,abc\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CSV parse error"));
    }

    #[test]
    fn csv_with_blank_lines() {
        let csv = "time,j0,j1\n0.0,0.0,0.0\n\n0.5,1.0,1.0\n\n";
        let traj = trajectory_from_csv(csv).unwrap();
        assert_eq!(traj.waypoints.len(), 2);
    }

    #[test]
    fn csv_with_velocities_and_accelerations() {
        let csv = "time,j0,j1,v0,v1,a0,a1\n0.0,0.0,0.0,1.0,1.0,0.0,0.0\n0.5,0.5,0.5,0.0,0.0,-2.0,-2.0\n";
        let traj = trajectory_from_csv(csv).unwrap();
        assert_eq!(traj.dof, 2);
        assert!((traj.waypoints[0].velocities[0] - 1.0).abs() < 1e-6);
        assert!((traj.waypoints[1].accelerations[0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn json_single_waypoint() {
        let traj = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 2,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: vec![1.0, 2.0],
                velocities: vec![0.0, 0.0],
                accelerations: vec![0.0, 0.0],
            }],
        };
        let json = trajectory_to_json(&traj);
        let parsed = trajectory_from_json(&json).unwrap();
        assert_eq!(parsed.waypoints.len(), 1);
        assert!((parsed.waypoints[0].positions[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn csv_duration_from_last_waypoint() {
        let csv = "time,j0\n0.0,0.0\n1.5,1.0\n3.0,2.0\n";
        let traj = trajectory_from_csv(csv).unwrap();
        assert!((traj.duration.as_secs_f64() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn json_waypoints_without_velocities() {
        let json = r#"{"dof": 2, "duration": 0.5, "waypoints": [
            {"time": 0.0, "positions": [0.0, 0.0]},
            {"time": 0.5, "positions": [1.0, 1.0]}
        ]}"#;
        let traj = trajectory_from_json(json).unwrap();
        assert_eq!(traj.waypoints[0].velocities, vec![0.0, 0.0]);
    }

    #[test]
    fn trajectory_to_csv_file_and_back() {
        let traj = sample_traj();
        let path = std::env::temp_dir().join("kinetic_test_traj.csv");
        trajectory_to_csv_file(&traj, &path).unwrap();
        let csv = std::fs::read_to_string(&path).unwrap();
        let reimported = trajectory_from_csv(&csv).unwrap();
        assert_eq!(reimported.dof, traj.dof);
        assert_eq!(reimported.waypoints.len(), traj.waypoints.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn trajectory_to_json_file_and_back() {
        let traj = sample_traj();
        let path = std::env::temp_dir().join("kinetic_test_traj.json");
        trajectory_to_json_file(&traj, &path).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        let reimported = trajectory_from_json(&json).unwrap();
        assert_eq!(reimported.dof, traj.dof);
        std::fs::remove_file(&path).ok();
    }
}
