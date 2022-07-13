pub mod backend;
pub mod ctx;

use std::fmt;

pub type Tensor<'a> = wasi_nn::Tensor<'a>;

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
impl From<Dtype> for wasi_nn::TensorType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F32 => wasi_nn::TENSOR_TYPE_F32,
            Dtype::U8 => wasi_nn::TENSOR_TYPE_U8,
        }
    }
}
