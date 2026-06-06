use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

fn compute_pixel(x: f32, y: f32) -> [u8; 4] {
    [
        (x * 255.0f32) as u8,
        (x * 255.0f32) as u8,
        (x * 255.0f32) as u8,
        255u8
    ]
}

struct Sink {
    width: usize,
    height: usize,
    data: Vec<u8>
}

impl Sink {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4]
        }
    }

    fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 4]) {
        for i in 0..4 {
            self.data[(y * self.width + x) * 4 + i] = color[i];
        }
    }
}

fn write_image(sink: Sink, path: &Path) {
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, sink.width as u32, sink.height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));     // 1.0 / 2.2, unscaled, but rounded
    let source_chromaticities = png::SourceChromaticities::new(     // Using unscaled instantiation here
        (0.31270, 0.32900),
        (0.64000, 0.33000),
        (0.30000, 0.60000),
        (0.15000, 0.06000)
    );
    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&sink.data).unwrap();
}

fn main() {
    let mut sink = Sink::new(256, 256);

    for x in 0..sink.width {
        for y in 0..sink.height {
            let rx = x as f32 / sink.width as f32;
            let ry = y as f32 / sink.height as f32;
            let color = compute_pixel(rx, ry);
            sink.set_pixel(x, y, color);
        }
    }

    let path = Path::new("out.png");
    write_image(sink, path);
}
