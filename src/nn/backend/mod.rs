pub(crate) mod openvino;

use crate::error::BackendError;

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
