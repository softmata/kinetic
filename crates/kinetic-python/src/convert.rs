//! Conversion utilities between numpy arrays and nalgebra types.

use nalgebra::{Isometry3, Matrix4, Translation3, UnitQuaternion};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kinetic_core::Pose;

/// Convert a numpy (4,4) SE3 homogeneous matrix to nalgebra Isometry3<f64>.
pub fn numpy_4x4_to_isometry(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Isometry3<f64>> {
    let shape = arr.shape();
    if shape != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected (4,4) array, got ({},{})",
            shape[0], shape[1]
        )));
    }

    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    // nalgebra Matrix4 is column-major, numpy is row-major
    // Read elements row by row
    let m = Matrix4::new(
        slice[0], slice[1], slice[2], slice[3],
        slice[4], slice[5], slice[6], slice[7],
        slice[8], slice[9], slice[10], slice[11],
        slice[12], slice[13], slice[14], slice[15],
    );

    // Extract rotation and translation
    let rotation = m.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Translation3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);

    let rot = nalgebra::Rotation3::from_matrix_unchecked(rotation);
    let quat = UnitQuaternion::from_rotation_matrix(&rot);

    Ok(Isometry3::from_parts(translation, quat))
}

/// Convert nalgebra Isometry3<f64> to a numpy (4,4) array.
pub fn isometry_to_numpy_4x4<'py>(
    py: Python<'py>,
    iso: &Isometry3<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let m = iso.to_homogeneous();
    // m is column-major, numpy expects row-major
    let mut data = [0.0f64; 16];
    for r in 0..4 {
        for c in 0..4 {
            data[r * 4 + c] = m[(r, c)];
        }
    }
    // Create (4,4) array
    let flat = PyArray1::from_slice(py, &data);
    flat.reshape([4, 4]).expect("reshape to (4,4) should succeed")
}

/// Convert nalgebra Pose to a numpy (4,4) array.
pub fn pose_to_numpy_4x4<'py>(py: Python<'py>, pose: &Pose) -> Bound<'py, PyArray2<f64>> {
    isometry_to_numpy_4x4(py, &pose.0)
}

/// Convert numpy 1D array to Vec<f64>.
pub fn numpy_to_vec(arr: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;
    Ok(slice.to_vec())
}

/// Convert Vec<f64> to numpy 1D array.
pub fn vec_to_numpy<'py>(py: Python<'py>, data: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, data)
}
