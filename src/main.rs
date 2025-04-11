use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, ImageBuffer, Rgb};
use log::{error, info};
use usls::{Annotator, COCO_CLASS_NAMES_91, Options, ResizeMode, models::RFDETR};
use video_rs::decode::Decoder;

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// The path to the input video file
    #[arg(long)]
    video_path: PathBuf,

    /// The path to the model file
    #[arg(long)]
    model_path: PathBuf,

    /// The device to use for inference
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn rgb_slice_to_dynamic_image(rgb_slice: &[u8], width: u32, height: u32) -> Option<DynamicImage> {
    if rgb_slice.len() as u32 != width * height * 3 {
        error!("Slice length does not match expected width * height * 3.");
        return None;
    }

    let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, rgb_slice.to_vec())?;

    Some(DynamicImage::ImageRgb8(img_buf))
}

fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    video_rs::init().unwrap();

    let dtype = "auto";

    // options
    let options = Options::default()
        .with_model_name("rfdetr")
        .with_model_file(args.model_path.as_path().to_str().unwrap())
        .with_model_dtype(dtype.try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .with_batch_size(1)
        .with_model_ixx(0, 2, 560.into())
        .with_model_ixx(0, 3, 560.into())
        .with_resize_mode(ResizeMode::FitAdaptive)
        .with_normalize(true)
        .with_image_mean(&[0.485, 0.456, 0.406])
        .with_image_std(&[0.229, 0.224, 0.225])
        .with_class_confs(&[0.4])
        .with_class_names(&COCO_CLASS_NAMES_91)
        .with_model_num_dry_run(3)
        .commit()?;

    let mut decoder = Decoder::new(args.video_path).expect("failed to create decoder");

    let (width, height) = decoder.size();
    let fps = decoder.frame_rate();

    info!("Video size: {} x {}", width, height);
    info!("Video FPS: {}", fps);

    let mut model = RFDETR::new(options)?;

    for (idx, frame) in decoder.decode_iter().enumerate() {
        // Only process 1 frame every every second
        if idx % fps as usize != 0 {
            continue;
        }

        if let Ok((_, frame)) = frame {
            let rgb = frame.as_slice().unwrap();
            let dynamic_image = rgb_slice_to_dynamic_image(rgb, width, height);

            if dynamic_image.is_none() {
                error!("Failed to convert RGB slice to DynamicImage.");
                continue;
            }

            let xs = [dynamic_image.unwrap()];

            let start_time = std::time::Instant::now();
            let ys = model.forward(&xs)?;
            let elapsed = start_time.elapsed();

            info!("[Inference]: Elapsed time: {:?}", elapsed);

            // extract bboxes
            for y in ys.iter() {
                if let Some(bboxes) = y.bboxes() {
                    info!("[Bboxes]: Found {} objects", bboxes.len());
                    for (i, bbox) in bboxes.iter().enumerate() {
                        info!("{}: {:?}", i, bbox)
                    }
                }
            }

            // annotate
            let annotator = Annotator::default()
                .with_bboxes_thickness(3)
                .with_saveout(model.spec());
            annotator.annotate(&xs, &ys);
        } else {
            break;
        }
    }

    Ok(())
}
