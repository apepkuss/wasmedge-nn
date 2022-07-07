use crate::error::CvError;
use image::{self, imageops, DynamicImage, Pixel, Rgb32FImage, RgbImage};

type CvResult<T> = Result<T, CvError>;

pub fn imread(filename: impl AsRef<std::path::Path>) -> CvResult<DynamicImage> {
    Ok(image::open(filename.as_ref())?)
}

/// Resize this image and returns a new image. Does not preserve aspect ratio.
///
/// `nwidth` and `nheight` are the new image's dimensions.
///
pub fn resize(src: DynamicImage, nheight: u32, nwidth: u32) -> DynamicImage {
    src.resize_exact(nwidth, nheight, imageops::Triangle)
}

/// Convert RGB_F32 image to BGR_32F image. The RGB_F32 image will be consumed.
pub fn to_bgr32f_bytes(src: &mut Rgb32FImage) -> &[u8] {
    // convert rgb to bgr
    let pixels = src.pixels_mut();
    pixels.for_each(|p| {
        let channel = p.channels_mut();
        channel.swap(0, 2);
    });

    let data = src.as_raw();
    let slice: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    slice
}

/// Convert RGB_U8 image to BGR_U8 image. The RGB_8U image will be consumed.
pub fn to_bgr8u_bytes(src: &mut RgbImage) -> &[u8] {
    // convert rgb to bgr
    let pixels = src.pixels_mut();
    pixels.for_each(|p| {
        let channel = p.channels_mut();
        channel.swap(0, 2);
    });

    let data = src.as_raw();
    let slice: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    slice
}

pub fn to_tensor<T>(h: usize, w: usize, c: usize, data: Vec<T>) -> ndarray::Array3<T> {
    ndarray::Array::from_shape_vec((h, w, c), data as Vec<T>).unwrap()
}

#[cfg(test)]
mod tests {
    use image::GenericImageView;

    use super::*;

    #[test]
    fn test_rgb8_to_bgr8() {
        let src = image::open("/Users/sam/workspace/rust/image-demo/ferris.png").unwrap();
        dbg!("dims: {:?}", src.dimensions());

        let dst = resize(src, 224, 224);
        assert_eq!(dst.dimensions(), (224, 224));

        let mut rgb8_image = dst.into_rgb8();
        println!("number of elements: {}", rgb8_image.len());

        let dst = to_bgr8u_bytes(&mut rgb8_image);
        assert_eq!(dst.len(), 224 * 224 * 3);

        let tensor = to_tensor(224, 224, 3, Vec::from(dst));
        assert_eq!(tensor.shape(), [224, 224, 3]);
    }

    #[test]
    fn test_rgb8_to_bgr32f() {
        let src = image::open("/Users/sam/workspace/rust/image-demo/ferris.png").unwrap();
        dbg!("dims: {:?}", src.dimensions());

        let dst = resize(src, 224, 224);
        assert_eq!(dst.dimensions(), (224, 224));

        let mut rgb32f_image = dst.into_rgb32f();
        println!("number of elements: {}", rgb32f_image.len());

        let dst = to_bgr32f_bytes(&mut rgb32f_image);
        assert_eq!(dst.len(), 224 * 224 * 3 * 4);
    }

    #[test]
    fn test_tensor_u8() {
        let src = image::open("/Users/sam/workspace/rust/image-demo/ferris.png").unwrap();
        dbg!("dims: {:?}", src.dimensions());

        let dst = resize(src, 224, 224);
        assert_eq!(dst.dimensions(), (224, 224));

        let rgb8_image = dst.into_rgb8();
        println!("number of elements: {}", rgb8_image.len());

        let data = rgb8_image.to_vec();

        let tensor = to_tensor(224, 224, 3, data);
        assert_eq!(tensor.shape(), [224, 224, 3]);

        let new_tensor = tensor.insert_axis(ndarray::Axis(0));
        assert_eq!(new_tensor.shape(), [1, 224, 224, 3]);
    }

    #[test]
    fn test_tensor_f32() {
        let src = image::open("/Users/sam/workspace/rust/image-demo/ferris.png").unwrap();
        dbg!("dims: {:?}", src.dimensions());

        let dst = resize(src, 224, 224);
        assert_eq!(dst.dimensions(), (224, 224));

        let rgb32f_image = dst.into_rgb32f();
        println!("number of elements: {}", rgb32f_image.len());

        let data = rgb32f_image.into_vec();
        let tensor = to_tensor(224, 224, 3, data);
        assert_eq!(tensor.shape(), [224, 224, 3]);

        let new_tensor = tensor.insert_axis(ndarray::Axis(0));
        assert_eq!(new_tensor.shape(), [1, 224, 224, 3]);
    }
}
