use image::{self, imageops, Bgr, DynamicImage, ImageBuffer};

pub fn imread(
    filename: impl AsRef<std::path::Path>,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    Ok(image::open(filename.as_ref())?)
}

pub fn resize(src: DynamicImage, height: u32, width: u32) -> DynamicImage {
    src.resize_exact(width, height, imageops::Triangle)
}

/// Converts RGB-8 image to BGR-8 image.
pub fn to_bgr(src: DynamicImage) -> ImageBuffer<Bgr<u8>, Vec<u8>> {
    src.to_bgr8()
}
