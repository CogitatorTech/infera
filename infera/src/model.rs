// Defines the internal representation of a model and the global model store.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

/// Type alias for a Tract `SimplePlan`.
/// This represents a runnable, optimized ONNX model execution plan.
#[cfg(feature = "tract")]
pub(crate) type OnnxModelPlan =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Represents a loaded ONNX model, holding its execution plan and metadata.
#[cfg(feature = "tract")]
pub(crate) struct OnnxModel {
    /// The compiled, runnable model plan from the Tract engine.
    pub model: OnnxModelPlan,
    /// The shape of the model's input tensor. Dynamic dimensions are represented by -1.
    pub input_shape: Vec<i64>,
    /// The shape of the model's output tensor. Dynamic dimensions are represented by -1.
    pub output_shape: Vec<i64>,
    /// The user-defined name for the model.
    pub name: String,
}

/// A placeholder struct for when the "tract" feature is not enabled.
#[cfg(not(feature = "tract"))]
pub(crate) struct OnnxModel {
    /// The user-defined name for the model.
    pub name: String,
}

/// A global, thread-safe store for all loaded ONNX models.
///
/// This is a `Lazy` static, meaning it is initialized on first access.
/// It uses a `RwLock` to allow multiple concurrent reads and exclusive writes,
/// mapping model names (strings) to their `OnnxModel` representations.
pub(crate) static MODELS: Lazy<RwLock<HashMap<String, OnnxModel>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
