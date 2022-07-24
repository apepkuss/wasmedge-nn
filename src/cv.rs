use crate::{error::CvError, Dtype};
use image::{self, io::Reader, DynamicImage};
use std::path::Path;

type CvResult<T> = Result<T, CvError>;

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
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

    let u8_f32_arr: Vec<u8> = match dtype {
        Dtype::F32 => {
            // Create an array to hold the f32 value of those pixels
            let bytes_required = raw_u8_arr.len() * 4;
            let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

            for i in 0..raw_u8_arr.len() {
                // Read the number as a f32 and break it into u8 bytes
                let u8_f32: f32 = raw_u8_arr[i] as f32;
                let u8_bytes = u8_f32.to_ne_bytes();

                for j in 0..4 {
                    u8_f32_arr[(i * 4) + j] = u8_bytes[j];
                }
            }

            u8_f32_arr
        }
        Dtype::U8 => raw_u8_arr.to_vec(),
    };

    Ok(u8_f32_arr)
}
