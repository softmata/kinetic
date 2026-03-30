#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(toml_str) = std::str::from_utf8(data) {
        // Fuzz the TOML config parser with a minimal URDF.
        let urdf = r#"<?xml version="1.0"?>
<robot name="fuzz_bot">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="ee_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>
</robot>"#;
        // Must not panic — errors are fine.
        let _ = kinetic_robot::Robot::from_config_strings(toml_str, urdf);
    }
});
