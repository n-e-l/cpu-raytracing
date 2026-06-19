use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use glam::Vec3;
use parry3d::query::PointQuery;

struct Material {
    color: Vec3,
    reflective: bool
}

trait SdfObject {
    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)>;
    fn normal(&self, pos: Vec3) -> Vec3;
    fn material(&self) -> &Material;
}

struct Floor {
    y: f32,
    material: Material
}

impl SdfObject for Floor {
    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)> {
        if pos.y <= self.y {
            Some((pos.y - self.y, self))
        } else {
            None
        }
    }

    fn normal(&self, pos: Vec3) -> Vec3 {
        Vec3::new(0.0, 1.0, 0.0)
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct Sphere {
    radius: f32,
    pos: Vec3,
    material: Material
}

impl SdfObject for Sphere {
    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)> {
        let d = (pos - self.pos).length() - self.radius;
        if d <= 0.0 {
            Some((d, self))
        } else {
            None
        }
    }

    fn normal(&self, hit: Vec3) -> Vec3 {
        (hit - self.pos).normalize()
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct Triangle {
    g_triangle: parry3d::shape::Triangle,
    material: Material
}

impl Triangle {
    fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self {
            g_triangle: parry3d::shape::Triangle::new(
                a, b, c
            ),
            material: Material {
                color: Vec3::new(1.0, 0.0, 0.0),
                reflective: false
            }
        }
    }
}

impl SdfObject for Triangle {
    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)> {
        Some((self.g_triangle.distance_to_local_point(pos, true), self))
    }

    fn normal(&self, pos: Vec3) -> Vec3 {
        self.g_triangle.normal().unwrap()
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct Docecahedron {
    scale: f32,
    tris: Vec<Sphere>
}

impl Docecahedron {
    fn new() -> Self {
        let phi = (1.0f32 + f32::sqrt(5.0f32)) / 2.0f32;
        let vertices = vec![
            // Orange vertices
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            // Green vertices
            Vec3::new(0.0, -phi, -1.0 / phi),
            Vec3::new(0.0, -phi, 1.0 / phi),
            Vec3::new(0.0, phi, -1.0 / phi),
            Vec3::new(0.0, phi, 1.0 / phi),
            // Blue vertices
            Vec3::new(-1.0 / phi, 0.0, -phi),
            Vec3::new(1.0 / phi, 0.0, -phi),
            Vec3::new(-1.0 / phi, 0.0, phi),
            Vec3::new(1.0 / phi, 0.0, phi),
            // Red vertices
            Vec3::new(-phi, -1.0 / phi, 0.0),
            Vec3::new(-phi, 1.0 / phi, 0.0),
            Vec3::new(phi, -1.0 / phi, 0.0),
            Vec3::new(phi, 1.0 / phi, 0.0),
        ];

        let mut tris = vertices.iter().map(|v| {
            Sphere {
                radius: 0.1,
                pos: *v,
                material: Material {
                    color: Vec3::new(1.0, 1.0, 0.0),
                    reflective: false,
                },
            }
        }).collect();
        Self {
            scale: 1.0f32,
            tris
        }
    }
}

impl SdfObject for Docecahedron {
    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)> {
        let mut result: Option<(f32, &dyn SdfObject)> = None;
        for tri in &self.tris {
            if let Some(sub) = tri.sdf(pos) {
                if result.is_none() || result.unwrap().0 < sub.0 {
                    result = Some(sub);
                }
            }
        }
        result
    }

    fn normal(&self, pos: Vec3) -> Vec3 {
        todo!()
    }

    fn material(&self) -> &Material {
        todo!()
    }
}

struct World {
    objects: Vec<Box<dyn SdfObject>>,
}

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    dir: Vec3
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

struct Hit<'a> {
    distance: f32,
    point: Vec3,
    object: &'a dyn SdfObject,
    normal: Vec3
}

impl World {
    fn new() -> Self {
        Self {
            objects: vec![
                Box::new(Floor {
                y: -0.2,
                material: Material {
                    color: Vec3::new(0.5, 0.5, 0.5),
                    reflective: false
                }
            })],
        }
    }

    fn sdf(&self, pos: Vec3) -> Option<(f32, &dyn SdfObject)> {
        let mut result: Option<(f32, &dyn SdfObject)> = None;
        let mut min = f32::MAX;

        for o in &self.objects {
            if let Some((d, t)) = o.sdf(pos) {
                if d <= min {
                    min = d;
                    result = Some((d, t));
                }
            }
        }

        result
    }

    fn hit(&self, ray: Ray) -> Option<Hit> {

        if self.sdf(ray.origin).is_none() {
            return None;
        }

        let mut dist = 0.0;
        for i in 0..300 {
            let (t, object) = self.sdf(ray.origin + ray.dir * dist).unwrap();

            dist += t;
            if t < 0.0001 {
                let point = ray.origin + ray.dir * dist;
                return Some(Hit {
                    distance: dist,
                    point,
                    object: object,
                    normal: object.normal(point)
                });
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
    let mut sink = Sink::new(512, 512);

    let mut world = World::new();

    world.objects.push(Box::new(Docecahedron::new()));
    // for i in 0..15 {
    //     world.objects.push(Box::new(Sphere {
    //         radius: rand::random::<f32>() * 0.3,
    //         pos: Vec3::new(
    //             rand::random::<f32>() * 2.0 - 1.0,
    //             rand::random::<f32>() * 1.0,
    //             rand::random::<f32>() * 2.0 - 1.0,
    //         ) * 0.7,
    //         material: Material {
    //             color: Vec3::new(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()),
    //             reflective: rand::random::<f32>() > 0.5
    //         }
    //     }));
    // }

    world.objects.push(Box::new(Triangle::new(
        Vec3::new(-0.5, -0.1, 0.0),
        Vec3::new(0.0, 0.6, 0.0),
        Vec3::new(0.5, -0.1, -0.4),
    )));

    let light = Vec3::new(1.0, -3.4, 4.5).normalize();

    for x in 0..sink.width {
        for y in 0..sink.height {
            let rx = x as f32 / sink.width as f32;
            let ry = y as f32 / sink.height as f32;
            let spread = 2.5;
            let ray_dir = Vec3::new(rx * 2.0 - 1.0, -ry * 2.0 + 1.0, spread).normalize();
            let mut ray = Ray {
                origin: Vec3::new(0.0, 0.0, -2.0),
                dir: ray_dir
            };

            let mut final_hit = None;
            while let Some(hit) = world.hit(ray) {
                // Exit on first non reflective object
                if !hit.object.material().reflective {
                    final_hit = Some(hit);
                    break;
                }

                // Calculate the reflection
                let reflect = ray.dir - hit.normal * 2.0 * hit.normal.dot(ray.dir);
                ray = Ray {
                    origin: hit.point + reflect * 0.001,
                    dir: reflect
                };
            }

            if let Some(mut hit) = final_hit {

                let mut color = hit.object.material().color;
                // let mut color = 1.0 - (Vec3::new(hit.distance, hit.distance, hit.distance) * 0.3) * hit.object.material().color;

                let dot = hit.normal.dot(-light);
                color *= dot;

                let shadowed = world.hit(Ray {
                    origin: hit.point - light * 0.001,
                    dir: -light
                }).is_some();

                if shadowed {
                    color *= 0.5;
                }

                // object.normal(ray.at(dist)) * 0.5 + 0.5
                sink.set_pixel(x, y, color);
            } else {
                let color = Vec3::new(0f32, 0f32, 0f32);
                sink.set_pixel(x, y, color);
            };
        }
    }

    let path = Path::new("out.png");
    write_image(sink, path);
}
