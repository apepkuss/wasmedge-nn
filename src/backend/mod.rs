pub mod openvino;

use thiserror::Error;

/// A [Backend] contains the necessary state to load [BackendGraph]s.
pub(crate) trait Backend {
    fn name(&self) -> &str;
    // fn load(
    //     &mut self,
    //     architecure: &str,
    //     weights: &str,
    //     target: ExecutionTarget,
    // ) -> Result<Box<dyn BackendGraph>, BackendError>;
    fn load(
        &mut self,
        architecure: &[u8],
        weights: &[u8],
        target: wasi_nn::ExecutionTarget,
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
