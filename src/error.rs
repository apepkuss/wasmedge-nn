use thiserror::Error;

/// Possible errors while interacting with [WasiNnCtx].
#[derive(Debug, Error)]
pub enum WasiNnError {
    #[error("Failed while loading model file: {0}")]
    InvalidPath(#[from] std::io::Error),
    #[error("backend error")]
    BackendError(#[from] BackendError),
    #[error("usage error")]
    UsageError(#[from] UsageError),
}

#[derive(Debug, Error)]
pub enum UsageError {
    #[error("Invalid context; has the load function been called?")]
    InvalidContext,
    #[error("Only OpenVINO's IR is currently supported, passed encoding: {0:?}")]
    InvalidEncoding(wasi_nn::GraphEncoding),
    #[error("OpenVINO expects only two buffers (i.e. [ir, weights]), passed: {0}")]
    InvalidNumberOfBuilders(u32),
    #[error("Invalid graph handle; has it been loaded?")]
    InvalidGraphHandle,
    #[error("Invalid execution context handle; has it been initialized?")]
    InvalidExecutionContextHandle,
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(u32),
}

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
    #[error("The backend expects {0} buffers, passed {1}")]
    InvalidNumberOfBuilders(u32, u32),
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(usize),
}

#[derive(Debug, Error)]
pub enum CvError {
    #[error("Failed while")]
    ProcessImage(#[from] image::ImageError),
}
