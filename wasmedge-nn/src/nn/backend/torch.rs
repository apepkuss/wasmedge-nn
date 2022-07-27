use super::*;
use crate::nn::Tensor;

#[derive(Debug, Default)]
pub(crate) struct TorchBackend {}
impl Backend for TorchBackend {
    fn name(&self) -> &'static str {
        "pytorch"
    }

    fn load(
        &mut self,
        architecure: Option<&[u8]>,
        weights: &[u8],
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        unimplemented!()
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchGraph {
    graph: wasmedge_wasi_nn::Graph,
}
impl BackendGraph for TorchGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        unimplemented!()
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchExecutionContext {
    ctx: wasmedge_wasi_nn::GraphExecutionContext,
}
impl BackendExecutionContext for TorchExecutionContext {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), BackendError> {
        unimplemented!()
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        unimplemented!()
    }

    fn get_output(&mut self, index: u32, out_buffer: &mut [u8]) -> Result<u32, BackendError> {
        unimplemented!()
    }
}
