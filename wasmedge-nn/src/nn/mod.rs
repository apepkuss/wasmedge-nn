pub mod backend;
pub mod ctx;

use std::fmt;

pub type Tensor<'a> = wasi_nn::Tensor<'a>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphEncoding {
    Openvino,
    Pytorch,
}
impl From<GraphEncoding> for wasi_nn::GraphEncoding {
    fn from(encoding: GraphEncoding) -> Self {
        match encoding {
            GraphEncoding::Openvino => wasi_nn::GRAPH_ENCODING_OPENVINO,
            GraphEncoding::Pytorch => wasi_nn::GRAPH_ENCODING_PYTORCH,
        }
    }
}
impl fmt::Display for GraphEncoding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GraphEncoding::Openvino => write!(f, "openvino"),
            GraphEncoding::Pytorch => write!(f, "pytorch"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    CPU,
    GPU,
    TPU,
}
impl From<ExecutionTarget> for wasi_nn::ExecutionTarget {
    fn from(target: ExecutionTarget) -> Self {
        match target {
            ExecutionTarget::CPU => wasi_nn::EXECUTION_TARGET_CPU,
            ExecutionTarget::GPU => wasi_nn::EXECUTION_TARGET_GPU,
            ExecutionTarget::TPU => wasi_nn::EXECUTION_TARGET_TPU,
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
