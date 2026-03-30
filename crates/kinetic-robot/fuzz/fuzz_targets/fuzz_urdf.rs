#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Must not panic on any input — errors are fine.
        let _ = kinetic_robot::Robot::from_urdf_string(s);
    }
});
