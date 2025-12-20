//! ONNX export and import functionality using tract

#[cfg(feature = "onnx")]
use tract_onnx::prelude::*;
use crate::Result;
use std::path::Path;

#[cfg(feature = "onnx")]
pub struct OnnxModelExporter;

#[cfg(feature = "onnx")]
impl OnnxModelExporter {
    /// Export Candle model to ONNX format
    pub fn export_to_onnx(
        model_path: &Path,
        input_shape: &[usize],
    ) -> Result<()> {
        // Create ONNX model using tract
        let mut model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Set input facts
        model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Optimize model
        let model = model.into_optimized()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Convert to runnable model
        let _runnable = model.into_runnable()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(())
    }

    /// Load ONNX model
    pub fn load_onnx_model(model_path: &Path) -> Result<TypedModel> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .into_optimized()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .into_typed()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        Ok(model)
    }

    /// Run inference with ONNX model
    pub fn run_inference(
        model: &TypedModel,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<Vec<f32>> {
        use tract_core::prelude::*;

        // Create input tensor
        let input_tensor = tract_ndarray::Array::from_shape_vec(
            input_shape.to_vec(),
            input.to_vec(),
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let input_tensor = input_tensor.into_tensor();

        // Run inference
        let result = model.run(tvec!(input_tensor.into()))
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Extract output
        let output = result[0]
            .to_array_view::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(output.iter().copied().collect())
    }
}
#[cfg(not(feature = "onnx"))]
pub struct OnnxModelExporter;
#[cfg(not(feature = "onnx"))]
impl OnnxModelExporter {
    pub fn export_to_onnx(_model_path: &Path, _input_shape: &[usize]) -> Result<()> {
        Err(crate::ExtractionError::ModelError(
            "ONNX support not enabled. Compile with --features onnx".to_string()
        ))
    }
}
