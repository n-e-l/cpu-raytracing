use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use glam::Vec3;

struct Sphere {
    radius: f32,
    pos: Vec3
}

struct World {
   spheres: Vec<Sphere>
}

struct Ray {
    origin: Vec3,
    dir: Vec3
}

impl World {
    fn new() -> Self {
        Self {
            spheres: vec![Sphere { radius: 0.1, pos: Vec3::new(0.0, 0.0, 0.0) }]
        }
    }

    fn sdf(&self, pos: Vec3) -> Option<(f32, &Sphere)> {
        let mut result = None;
        let mut min = f32::MAX;
        for sphere in &self.spheres {
            let d = (sphere.pos - pos).length() - sphere.radius;
            if d <= min {
                min = d;
                result = Some((d, sphere));
            }
        }
        result
    }

    fn hit(&self, ray: Ray) -> Option<(f32, &Sphere)> {

        if self.sdf(ray.origin).is_none() {
            return None;
        }

        let mut dist = 0.0;
        for i in 0..100 {
            let (t, sphere) = self.sdf(ray.origin + ray.dir * dist).unwrap();
            dist += t;
            if t < 0.0001 {
                return Some((dist, sphere));
            }

            if t > 1000.0 {
                break;
            }
        }

        None
    }
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

    fn set_pixel_u8(&mut self, x: usize, y: usize, color: [u8; 4]) {
        for i in 0..4 {
            self.data[(y * self.width + x) * 4 + i] = color[i];
        }
    }

    fn set_pixel(&mut self, x: usize, y: usize, color: Vec3) {
        for i in 0..3 {
            self.data[(y * self.width + x) * 4 + i] = (color[i] * 255f32) as u8;
        }
        // Alpha channel
        self.data[(y * self.width + x) * 4 + 3] = 255u8;
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

    let mut world = World::new();

    for i in 0..10 {
        world.spheres.push(Sphere {
            radius: 0.3,
            pos: Vec3::new(
                rand::random::<f32>() * 2.0 - 1.0,
                rand::random::<f32>() * 2.0 - 1.0,
                rand::random::<f32>() * 2.0 - 1.0,
            )
        });
    }

    for x in 0..sink.width {
        for y in 0..sink.height {
            let rx = x as f32 / sink.width as f32;
            let ry = y as f32 / sink.height as f32;
            let spread = 2.5;
            let ray = Ray {
                origin: Vec3::new(0.0, 0.0, -2.0),
                dir: Vec3::new(rx * 2.0 - 1.0, ry * 2.0 - 1.0, spread).normalize()
            };

            let color = if let Some((dist, _)) = world.hit(ray) {
                1.0 - (Vec3::new(dist, dist, dist) * 0.3)
            } else {
                Vec3::new(0f32, 0f32, 0f32)
            };

            sink.set_pixel(x, y, color);
        }
    }

    let path = Path::new("out.png");
    write_image(sink, path);
}
