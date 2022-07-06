use super::*;
use std::{fs, path::Path};

#[derive(Debug, Default)]
pub(crate) struct OpenvinoBackend {
    core: String, // Option<openvino::Core>,
}
impl Backend for OpenvinoBackend {
    fn name(&self) -> &str {
        "openvino"
    }

    fn load(
        &mut self,
        xml_file: impl AsRef<Path>,
        bin_file: impl AsRef<Path>,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        let xml =
            fs::read_to_string(xml_file.as_ref()).map_err(|e| BackendError::InvalidPath(e))?;
        let xml_bytes = xml.into_bytes();
        println!("Load graph XML, size in bytes: {}", xml_bytes.len());

        let weights = fs::read(bin_file.as_ref()).map_err(|e| BackendError::InvalidPath(e))?;
        println!("Load graph weights, size in bytes: {}", weights.len());

        self.load_from_bytes(xml_bytes.as_slice(), weights.as_slice(), target)
    }

    fn load_from_bytes(
        &mut self,
        xml_bytes: impl AsRef<[u8]>,
        weights: impl AsRef<[u8]>,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        let graph = unsafe {
            wasi_nn::load(
                &[xml_bytes.as_ref(), weights.as_ref()],
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
    fn set_input(&mut self, index: u32, tensor: wasi_nn::Tensor) -> Result<(), BackendError> {
        unsafe {
            wasi_nn::set_input(self.ctx, index, tensor)
                .map_err(|e| BackendError::SetInput(e.to_string()))
        }
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        unsafe { wasi_nn::compute(self.ctx).map_err(|e| BackendError::Compute(e.to_string())) }
    }

    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        unsafe {
            wasi_nn::get_output(
                self.ctx,
                index,
                destination.as_mut_ptr(),
                destination.len() as u32,
            )
            .map_err(|e| BackendError::GetOutput(e.to_string()))
        }
    }
}
