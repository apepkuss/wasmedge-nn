pub mod openvino;

use std::path::Path;
use thiserror::Error;

/// A [Backend] contains the necessary state to load [BackendGraph]s.
pub(crate) trait Backend {
    fn name(&self) -> &str;
    fn load(
        &mut self,
        architecure: impl AsRef<Path>,
        weights: impl AsRef<Path>,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError>;
    fn load_from_bytes(
        &mut self,
        architecure: impl AsRef<[u8]>,
        weights: impl AsRef<[u8]>,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError>;
}

/// A [BackendGraph] can create [BackendExecutionContext]s; this is the backing
/// implementation for a [crate::witx::types::Graph].
pub(crate) trait BackendGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError>;
}

/// A [BackendExecutionContext] performs the actual inference; this is the
/// backing implementation for a [crate::witx::types::GraphExecutionContext].
pub(crate) trait BackendExecutionContext {
    fn set_input(&mut self, index: u32, tensor: wasi_nn::Tensor) -> Result<(), BackendError>;
    fn compute(&mut self) -> Result<(), BackendError>;
    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError>;
}

/// Errors returned by a backend; [BackendError::BackendAccess] is a catch-all
/// for failures interacting with the ML library.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Failed while loading model file: {0}")]
    InvalidPath(#[from] std::io::Error),
    #[error("Failed while loading model: {0}")]
    ModelLoad(String),
    #[error("Failed while setting input tensor: {0}")]
    SetInput(String),
    #[error("Failed while performing the inference: {0}")]
    Compute(String),
    #[error("Failed while getting output bytes: {0}")]
    GetOutput(String),
    // #[error("Failed while accessing backend")]
    // BackendAccess(#[from] anyhow::Error),
    // #[error("Failed while accessing guest module")]
    // GuestAccess(#[from] GuestError),
    #[error("The backend expects {0} buffers, passed {1}")]
    InvalidNumberOfBuilders(u32, u32),
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(usize),
}

#[derive(Debug, Clone, Copy)]
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

// #[repr(C)]
// #[derive(Copy, Clone, Debug)]
// pub struct Tensor<'a> {
//     pub shape: TensorShape,
//     pub dtype: TensorType,
//     /// Contains the tensor data.
//     pub data: &'a [u8],
// }
// impl<'a> Tensor<'a> {
//     fn as_wasinn_tenor(&'a self) -> wasi_nn::Tensor<'a> {
//         wasi_nn::Tensor {
//             dimensions: self.shape.as_slice(),
//             r#type: self.dtype.into(),
//             data: self.data,
//         }
//     }
// }
// impl From<Tensor<'_>> for wasi_nn::Tensor<'_> {
//     fn from(tensor: Tensor) -> Self {
//         wasi_nn::Tensor {
//             dimensions: [tensor.shape.batch, tensor.shape.channels],
//             r#type: tensor.dtype.into(),
//             data: tensor.data,
//         }
//     }
// }

// #[derive(Debug, Clone, Copy)]
// pub struct TensorShape {
//     pub batch: u32,
//     pub height: u32,
//     pub width: u32,
//     pub channels: u32,
// }
// impl TensorShape {
//     pub fn new(batch: u32, height: u32, width: u32, channels: u32) -> Self {
//         Self {
//             batch,
//             height,
//             width,
//             channels,
//         }
//     }
// }

// #[derive(Debug, Copy, Clone, PartialEq, Eq)]
// pub enum TensorType {
//     U8,
//     F32,
// }
// impl TensorType {
//     /// Returns the number of bytes occupied by the dtype.
//     pub fn bytes(&self) -> usize {
//         match self {
//             TensorType::U8 => 1,
//             TensorType::F32 => 4,
//         }
//     }
// }
// impl From<TensorType> for wasi_nn::TensorType {
//     fn from(dtype: TensorType) -> Self {
//         match dtype {
//             TensorType::U8 => wasi_nn::TENSOR_TYPE_U8,
//             TensorType::F32 => wasi_nn::TENSOR_TYPE_F32,
//         }
//     }
// }
