use super::*;
use crate::nn::Tensor;
use wasi_nn as nn;

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
            nn::load(&[weights], nn::GRAPH_ENCODING_PYTORCH, target.into())
                .map_err(|e| BackendError::ModelLoad(e.to_string()))?
        };

        Ok(Box::new(TorchGraph { graph }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchGraph {
    graph: nn::Graph,
}
impl BackendGraph for TorchGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        let ctx = unsafe {
            nn::init_execution_context(self.graph).expect("failed to create execution context")
        };
        println!("Created wasi-nn execution context with ID: {}", ctx);

        Ok(Box::new(TorchExecutionContext { ctx }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct TorchExecutionContext {
    ctx: nn::GraphExecutionContext,
}
impl BackendExecutionContext for TorchExecutionContext {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), BackendError> {
        unsafe {
            nn::set_input(self.ctx, index, tensor)
                .map_err(|e| BackendError::SetInput(e.to_string()))
        }
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        unsafe { nn::compute(self.ctx).map_err(|e| BackendError::Compute(e.to_string())) }
    }

    fn get_output(&mut self, index: u32, out_buffer: &mut [u8]) -> Result<u32, BackendError> {
        unsafe {
            nn::get_output(
                self.ctx,
                index,
                out_buffer.as_mut_ptr(),
                out_buffer.len() as u32,
            )
            .map_err(|e| BackendError::GetOutput(e.to_string()))
        }
    }
}
