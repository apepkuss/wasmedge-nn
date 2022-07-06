use crate::{Dimension, Dtype};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct Network {
    ctx: u32,
    backend: Backend,
    target: Target,
}
impl Network {
    pub fn init_openvino(
        xml_file: impl AsRef<Path>,
        bin_file: impl AsRef<Path>,
        target: Target,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let xml = fs::read_to_string(xml_file.as_ref())?;
        let xml_bytes = xml.into_bytes();
        println!("Load graph XML, size in bytes: {}", xml_bytes.len());

        let weights = fs::read(bin_file.as_ref())?;
        println!("Load graph weights, size in bytes: {}", weights.len());

        let graph = unsafe {
            let graph = wasi_nn::load(
                &[xml_bytes.as_ref(), weights.as_ref()],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                target.into(),
            )
            .expect("failed to load openvino model");
            graph
        };

        let ctx = unsafe {
            wasi_nn::init_execution_context(graph).expect("failed to create execution context")
        };
        println!("Created wasi-nn execution context with ID: {}", ctx);

        Ok(Self {
            ctx,
            backend: Backend::OpenVINO,
            target,
        })
    }

    pub fn infer(
        &self,
        input_tensor: wasi_nn::Tensor,
        out_dim: Dimension,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // set the input tensor
        unsafe {
            wasi_nn::set_input(self.ctx, 0, input_tensor).unwrap();
        }

        // Execute the inference.
        unsafe {
            wasi_nn::compute(self.ctx).unwrap();
        }
        println!("Executed graph inference");

        // extract the output bytes
        let len = match out_dim.dtype {
            Dtype::U8 => out_dim.height * out_dim.width * out_dim.channels,
            Dtype::F32 => (out_dim.height * out_dim.width * out_dim.channels) * 4,
        };
        let mut output_buffer: Vec<u8> = Vec::with_capacity(len as usize);
        unsafe {
            wasi_nn::get_output(
                self.ctx,
                0,
                &mut output_buffer[..] as *mut [u8] as *mut u8,
                len.try_into().unwrap(),
            )
            .unwrap();
            output_buffer.set_len(len as usize);
        };
        println!("bytes_written: {:?}", len);

        Ok(output_buffer)
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn target(&self) -> Target {
        self.target
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Target {
    CPU,
    GPU,
}
impl From<Target> for wasi_nn::ExecutionTarget {
    fn from(target: Target) -> Self {
        match target {
            Target::CPU => wasi_nn::EXECUTION_TARGET_CPU,
            Target::GPU => wasi_nn::EXECUTION_TARGET_GPU,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    OpenVINO,
}
impl From<Backend> for wasi_nn::GraphEncoding {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::OpenVINO => wasi_nn::GRAPH_ENCODING_OPENVINO,
        }
    }
}
