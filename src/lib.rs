pub mod cv;
pub mod error;
pub mod nn;

#[derive(Debug, Clone, Copy)]
pub struct Dimension {
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub dtype: Dtype,
}
impl Dimension {
    pub fn new(height: u32, width: u32, channels: u32, dtype: Dtype) -> Self {
        Self {
            height,
            width,
            channels,
            dtype,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Dtype {
    U8,
    F32,
}
impl Dtype {
    /// Returns the number of bytes occupied by the dtype.
    pub fn bytes(&self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::F32 => 4,
        }
    }
}
