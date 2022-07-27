pub(crate) mod openvino;
pub(crate) mod torch;

use crate::{
    error::BackendError,
    nn::{ExecutionTarget, Tensor},
};

/// A [Backend] contains the necessary state to load [BackendGraph]s.
pub(crate) trait Backend {
    fn name(&self) -> &str;
    fn load(
        &mut self,
        architecure: Option<&[u8]>,
        weights: &[u8],
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
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), BackendError>;
    fn compute(&mut self) -> Result<(), BackendError>;
    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError>;
}
