// src/model.rs
// Defines the internal representation of a model and the global model store.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

#[cfg(feature = "tract")]
pub(crate) struct OnnxModel {
    pub model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    pub name: String,
}

#[cfg(not(feature = "tract"))]
pub(crate) struct OnnxModel {
    pub name: String,
}

pub(crate) static MODELS: Lazy<RwLock<HashMap<String, OnnxModel>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
