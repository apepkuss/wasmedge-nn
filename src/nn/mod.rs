pub mod backend;
pub mod ctx;

use std::fmt;

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

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    pub shape: Vec<u32>,
    pub dtype: Dtype,
    pub data: &'a [u8],
}
impl<'a> Tensor<'a> {
    pub fn as_wasinn_tensor(&'a self) -> wasi_nn::Tensor<'a> {
        wasi_nn::Tensor {
            dimensions: self.shape.as_slice(),
            r#type: self.dtype.into(),
            data: self.data,
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

// pub struct NTensor<'a> {
//     /// Describe the size of the tensor (e.g. 2x2x2x2 -> [2, 2, 2, 2]). To represent a tensor containing a single value,
//     /// use `[1]` for the tensor dimensions.
//     pub dimensions: TensorDimensions<'a>,
//     pub r#type: TensorType,
//     /// Contains the tensor data.
//     pub data: TensorData<'a>,
// }
