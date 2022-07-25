use crate::{error::CvError, nn::Dtype};
use image::{self, io::Reader, DynamicImage};
use std::path::Path;

type CvResult<T> = Result<T, CvError>;

pub fn image_to_tensor(
    path: impl AsRef<Path>,
    nheight: u32,
    nwidth: u32,
    dtype: Dtype,
) -> CvResult<Vec<u8>> {
    let pixels = Reader::open(path.as_ref())?.decode()?;
    let dyn_img: DynamicImage = pixels.resize_exact(nwidth, nheight, image::imageops::Triangle);
    let bgr_img = dyn_img.to_bgr8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];

    // Create an array to hold the f32 value of those pixels
    let bytes_required = raw_u8_arr.len() * dtype.bytes();
    let mut u8_arr: Vec<u8> = vec![0; bytes_required];

    for i in 0..raw_u8_arr.len() {
        // Read the number as a f32 and break it into u8 bytes
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let u8_bytes = u8_f32.to_ne_bytes();

        for j in 0..dtype.bytes() {
            u8_arr[(i * dtype.bytes()) + j] = u8_bytes[j];
        }
    }

    Ok(u8_arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_to_tensor() {
        let image_file = "/Volumes/Dev/secondstate/me/WasmEdge-WASINN-examples/openvino-road-segmentation-adas/rust/image/empty_road_mapillary.jpg";
        let result = image_to_tensor(image_file, 512, 896, Dtype::F32);
        assert!(result.is_ok());
        let bytes = result.unwrap();

        let result = std::fs::write("test_image_to_tensor.tensor", bytes.as_slice());
        assert!(result.is_ok());
    }
}
