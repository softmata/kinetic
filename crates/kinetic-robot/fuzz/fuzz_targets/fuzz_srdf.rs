#![no_main]

use libfuzzer_sys::fuzz_target;
use kinetic_robot::srdf::SrdfModel;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Parse SRDF — must not panic on any input.
        if let Ok(srdf) = SrdfModel::from_string(s) {
            // If parsing succeeds, also test applying to a minimal robot.
            let urdf = r#"<?xml version="1.0"?>
<robot name="test">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
</robot>"#;
            if let Ok(mut robot) = kinetic_robot::Robot::from_urdf_string(urdf) {
                let _ = srdf.apply_to_robot(&mut robot);
            }
        }
    }
});
