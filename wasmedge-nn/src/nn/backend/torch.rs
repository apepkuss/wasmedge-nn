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
        _arch: Option<&[u8]>,
        weights: &[u8],
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        let graph = unsafe {
            wasmedge_wasi_nn::load(
                &[weights],
                wasmedge_wasi_nn::GRAPH_ENCODING_TORCH,
                target.into(),
            )
            .map_err(|e| BackendError::ModelLoad(e.to_string()))?
        };

        Ok(Box::new(TorchGraph { graph }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchGraph {
    graph: wasmedge_wasi_nn::Graph,
}
impl BackendGraph for TorchGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        let ctx = unsafe {
            wasmedge_wasi_nn::init_execution_context(self.graph)
                .expect("failed to create execution context")
        };
        println!("Created wasi-nn execution context with ID: {}", ctx);

        Ok(Box::new(TorchExecutionContext { ctx }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchExecutionContext {
    ctx: wasmedge_wasi_nn::GraphExecutionContext,
}
impl BackendExecutionContext for TorchExecutionContext {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), BackendError> {
        unsafe {
            wasmedge_wasi_nn::set_input(self.ctx, index, tensor)
                .map_err(|e| BackendError::SetInput(e.to_string()))
        }
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        unsafe {
            wasmedge_wasi_nn::compute(self.ctx).map_err(|e| BackendError::Compute(e.to_string()))
        }
    }

    fn get_output(&mut self, index: u32, out_buffer: &mut [u8]) -> Result<u32, BackendError> {
        unsafe {
            wasmedge_wasi_nn::get_output(
                self.ctx,
                index,
                out_buffer.as_mut_ptr(),
                out_buffer.len() as u32,
            )
            .map_err(|e| BackendError::GetOutput(e.to_string()))
        }
    }
}
