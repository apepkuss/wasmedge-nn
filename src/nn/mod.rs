pub mod backend;
pub mod ctx;

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    data: Vec<u8>,
    dtype: Dtype,
    shape: Vec<u32>,
}
impl Tensor {
    pub fn new(dtype: Dtype, shape: impl AsRef<[u32]>, data: impl AsRef<[u8]>) -> Self {
        Self {
            data: data.as_ref().to_vec(),
            dtype,
            shape: shape.as_ref().to_vec(),
        }
    }

    pub fn as_wasinn_tensor(&self) -> wasi_nn::Tensor<'_> {
        wasi_nn::Tensor {
            dimensions: self.shape.as_slice(),
            r#type: self.dtype.into(),
            data: self.data.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphEncoding {
    Openvino,
}
impl From<GraphEncoding> for wasi_nn::GraphEncoding {
    fn from(encoding: GraphEncoding) -> Self {
        match encoding {
            GraphEncoding::Openvino => wasi_nn::GRAPH_ENCODING_OPENVINO,
        }
    }
}
impl fmt::Display for GraphEncoding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GraphEncoding::Openvino => write!(f, "openvino"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    CPU,
    GPU,
}
impl From<ExecutionTarget> for wasi_nn::ExecutionTarget {
    fn from(target: ExecutionTarget) -> Self {
        match target {
            ExecutionTarget::CPU => wasi_nn::EXECUTION_TARGET_CPU,
            ExecutionTarget::GPU => wasi_nn::EXECUTION_TARGET_GPU,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
}
impl Dtype {
    /// Returns the number of bytes occupied by the dtype.
    pub fn bytes(&self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::F32 => 4,
        }
    }
}
impl From<Dtype> for wasi_nn::TensorType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F32 => wasi_nn::TENSOR_TYPE_F32,
            Dtype::U8 => wasi_nn::TENSOR_TYPE_U8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dimension {
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub dtype: Dtype,
}
impl Dimension {
    pub fn new(height: u32, width: u32, channels: u32, dtype: Dtype) -> Self {
        Self {
            height,
            width,
            channels,
            dtype,
        }
    }
}
