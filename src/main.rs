use std::cmp::min;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use glam::Vec3;
use parry3d::math::{Pose, Rot3};
use parry3d::query::{PointQuery, Ray, RayCast};
use parry3d::shape::Triangle;
use rayon::iter::IntoParallelIterator;

struct Material {
    color: Vec3,
    reflective: bool
}

trait CastObject: Sync {
    fn hit(&self, ray: &Ray) -> Option<Hit>;
    fn material(&self) -> &Material;
}

struct Floor {
    y: f32,
    material: Material
}

struct Sphere {
    radius: f32,
    pos: Vec3,
    material: Material
}

impl CastObject for Sphere {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let p = Pose {
            rotation: Rot3::IDENTITY,
            translation: self.pos,
            padding: 0
        };
        if let Some(hit) = parry3d::shape::Ball::new(self.radius).cast_ray_and_get_normal(
            &p,
            ray,
            9999.0f32,
            true
        ) {
            return Some(Hit {
                distance: hit.time_of_impact,
                object: self,
                normal: hit.normal
            })
        }

        None
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct PentSlice {
    material: Material,
    tris: Vec<Triangle>
}

impl PentSlice {
    fn new(material: Material) -> Self {
        let scale = 0.3f32;
        let phi = (1.0f32 + f32::sqrt(5.0f32)) / 2.0f32;
        let vertices = vec![
            // Orange vertices
            scale * Vec3::new(0.0, phi, -1.0 / phi),
            scale * Vec3::new(0.0, phi, 1.0 / phi),
            scale * Vec3::new(1.0, 1.0, 1.0),
            scale * Vec3::new(phi, 1.0 / phi, 0.0),
            scale * Vec3::new(1.0, 1.0, -1.0),
        ];

        let tris = vec![
            Triangle::new(
                Vec3::ZERO,
                vertices[0],
                vertices[1],
            ),
            Triangle::new(
                Vec3::ZERO,
                vertices[1],
                vertices[2],
            ),
            Triangle::new(
                Vec3::ZERO,
                vertices[2],
                vertices[3],
            ),
            Triangle::new(
                Vec3::ZERO,
                vertices[3],
                vertices[4],
            ),
            Triangle::new(
                Vec3::ZERO,
                vertices[4],
                vertices[0],
            ),
        ];

        Self {
            material,
            tris
        }
    }
}

impl CastObject for PentSlice {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let mut min_hit: Option<Hit> = None;

        self.tris.iter().for_each(|o| {
            if let Some(hit) = o.cast_local_ray_and_get_normal(
                ray,
                9999.0f32,
                true
            ) {
                if min_hit.is_none() || hit.time_of_impact < min_hit.as_ref().unwrap().distance {
                    min_hit = Some(Hit {
                        distance: hit.time_of_impact,
                        object: self,
                        normal: hit.normal
                    });
                }
            }
        });

        min_hit
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct Docecahedron {
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

        let scale = 0.3f32;
        let mut tris = vertices.iter().map(|v| {
            Sphere {
                radius: 1.0 / phi * scale,
                pos: *v * scale,
                material: Material {
                    color: Vec3::new(1.0, 1.0, 0.0),
                    reflective: false,
                },
            }
        }).collect();
        Self {
            tris
        }
    }
}

impl CastObject for Docecahedron {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let mut min_hit: Option<Hit> = None;

        self.tris.iter().for_each(|o| {
            if let Some(hit) = o.hit(&ray) {
                if min_hit.is_none() || hit.distance < min_hit.as_ref().unwrap().distance {
                    min_hit = Some(hit);
                }
            }
        });

        min_hit
    }

    fn material(&self) -> &Material {
        self.material()
    }
}

struct World {
    objects: Vec<Box<dyn CastObject>>,
}

struct Hit<'a> {
    distance: f32,
    object: &'a dyn CastObject,
    normal: Vec3
}

impl World {
    fn new() -> Self {
        Self {
            objects: vec![
            //     Box::new(Floor {
            //     y: -0.2,
            //     material: Material {
            //         color: Vec3::new(0.5, 0.5, 0.5),
            //         reflective: false
            //     }
            // })
            ],
        }
    }

    fn hit(&self, ray: Ray) -> Option<Hit> {

        let mut min_hit: Option<Hit> = None;

        self.objects.iter().for_each(|o| {
            if let Some(hit) = o.hit(&ray) {
                if min_hit.is_none() || hit.distance < min_hit.as_ref().unwrap().distance {
                    min_hit = Some(hit);
                }
            }
        });

        min_hit
    }
}

struct Sink {
    width: usize,
    height: usize,
    data: Vec<Vec3>
}

impl Sink {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![Vec3::ZERO; width * height]
        }
    }

    fn get_mut_vec(&mut self) -> Vec<(usize, usize, &mut Vec3)> {
        self.data.iter_mut()
            .enumerate().map(|(i, data)| {
            (i / self.width, i % self.width, data)
        }).collect()
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

    // Convert Vec3 to u8
    let u8_data = sink.data.iter().map(|pixel| {
        vec![
            (pixel.x * 255f32) as u8,
            (pixel.y * 255f32) as u8,
            (pixel.z * 255f32) as u8,
            255u8
        ]
    }).flatten().collect::<Vec<u8>>();

    writer.write_image_data(&u8_data).unwrap();
}

fn main() {
    let size = 512;
    let mut sink = Sink::new(size, size);

    let mut world = World::new();

    world.objects.push(Box::new(PentSlice::new(
        Material {
            color: Vec3::new(0.0, 1.0, 1.0),
            reflective: false
        }
    )));
    // world.objects.push(Box::new(Docecahedron::new()));
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

    // world.objects.push(Box::new(Triangle::new(
    //     Vec3::new(-0.5, -0.1, 0.0),
    //     Vec3::new(0.0, 0.6, 0.0),
    //     Vec3::new(0.5, -0.1, -0.4),
    // )));

    let light_dir = Vec3::new(1.0, -3.4, 4.5).normalize();

    sink.get_mut_vec()
        // .into_par_iter()
        .into_iter()
        .for_each(|(x, y, data)| {
            let rx = x as f32 / size as f32;
            let ry = y as f32 / size as f32;
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
                let point = ray.point_at(hit.distance);
                ray = Ray {
                    origin: point + reflect * 0.001,
                    dir: reflect
                };
            }

            if let Some(mut hit) = final_hit {

                let mut color = hit.object.material().color;
                // let mut color = 1.0 - (Vec3::new(hit.distance, hit.distance, hit.distance) * 0.3) * hit.object.material().color;

                let mut light = hit.normal.dot(-light_dir);
                let point = ray.point_at(hit.distance);

                let shadowed = world.hit(Ray {
                    origin: point - light_dir * 0.001,
                    dir: -light_dir
                }).is_some();

                light = f32::max(0.1f32, light);

                if shadowed {
                    // light = f32::min(0.1, light);
                }

                color *= light;

                // object.normal(ray.at(dist)) * 0.5 + 0.5
                *data = color;
            } else {
                let color = Vec3::new(0f32, 0f32, 0f32);
                *data = color;
            };
        }
    );

    let path = Path::new("out.png");
    write_image(sink, path);
}
