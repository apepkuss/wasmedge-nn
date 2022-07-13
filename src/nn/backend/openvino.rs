use super::*;

#[derive(Debug, Default)]
pub(crate) struct OpenvinoBackend {}
impl Backend for OpenvinoBackend {
    fn name(&self) -> &str {
        "openvino"
    }

    fn load(
        &mut self,
        xml_bytes: &[u8],
        weights: &[u8],
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        let graph = unsafe {
            wasi_nn::load(
                &[xml_bytes, weights],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                target.into(),
            )
            .map_err(|e| BackendError::ModelLoad(e.to_string()))?
        };

        Ok(Box::new(OpenvinoGraph { graph }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct OpenvinoGraph {
    graph: wasi_nn::Graph,
}
impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        let ctx = unsafe {
            wasi_nn::init_execution_context(self.graph).expect("failed to create execution context")
        };
        println!("Created wasi-nn execution context with ID: {}", ctx);

        Ok(Box::new(OpenvinoExecutionContext { ctx }))
    }
}

#[derive(Default, Debug)]
pub(crate) struct OpenvinoExecutionContext {
    ctx: wasi_nn::GraphExecutionContext,
}
impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), BackendError> {
        unsafe {
            wasi_nn::set_input(self.ctx, index, tensor.as_wasinn_tensor())
                .map_err(|e| BackendError::SetInput(e.to_string()))
        }
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        unsafe { wasi_nn::compute(self.ctx).map_err(|e| BackendError::Compute(e.to_string())) }
    }

    fn get_output(&mut self, index: u32, out_buffer: &mut [u8]) -> Result<u32, BackendError> {
        unsafe {
            wasi_nn::get_output(
                self.ctx,
                index,
                out_buffer.as_mut_ptr(),
                out_buffer.len() as u32,
            )
            .map_err(|e| BackendError::GetOutput(e.to_string()))
        }
    }
}
