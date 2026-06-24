use std::cmp::min;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Mul;
use std::path::Path;
use glam::Vec3;
use parry3d::math::{Pose, Rot3};
use parry3d::query::{intersection_test, PointQuery, Ray, RayCast};
use parry3d::shape;
use parry3d::shape::{CompositeShape, Compound, ConvexPolyhedron, Shape, SharedShape, Triangle};
use petgraph::data::Build;
use petgraph::Graph;
use rand::{random, random_range};
use rayon::iter::IntoParallelIterator;
use crate::PentFace::{Pent, Tri};

#[derive(Copy, Clone)]
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

#[derive(Clone)]
struct PentSlice {
    pose: Pose,
    vertices: Vec<Vec3>,
    material: Material,
    pent: ConvexPolyhedron,
    compound: Compound,
    tris: Vec<Triangle>,
}

enum PentFace {
    Tri(usize),
    Pent
}

impl PentSlice {
    fn new(material: Material) -> Self {

        let scale = 0.3f32;
        let phi = (1.0f32 + f32::sqrt(5.0f32)) / 2.0f32;
        let mut vertices = vec![
            Vec3::ZERO,
            scale * Vec3::new(0.0, phi, -1.0 / phi),
            scale * Vec3::new(0.0, phi, 1.0 / phi),
            scale * Vec3::new(1.0, 1.0, 1.0),
            scale * Vec3::new(phi, 1.0 / phi, 0.0),
            scale * Vec3::new(1.0, 1.0, -1.0),
        ];

        // Calculate the center pos
        let mut center = Vec3::ZERO;
        vertices.iter().for_each(|v| {
            center += v;
        });
        center = center / vertices.len() as f32;

        // Move the vertices around the center pos
        vertices.iter_mut().for_each(|mut v| {
            *v -= center;
        });

        let tris = vec![
            Triangle::new(
                vertices[0],
                vertices[1],
                vertices[2],
            ),
            Triangle::new(
                vertices[0],
                vertices[2],
                vertices[3],
            ),
            Triangle::new(
                vertices[0],
                vertices[3],
                vertices[4],
            ),
            Triangle::new(
                vertices[0],
                vertices[4],
                vertices[5],
            ),
            Triangle::new(
                vertices[0],
                vertices[5],
                vertices[1],
            ),
        ];

        let pent_verts = vec![
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[4],
            vertices[5],
        ];
        let pent = shape::ConvexPolyhedron::from_convex_hull(
            &pent_verts
        ).unwrap();

        let compound = shape::Compound::new(
            vec![
                (Pose::identity(), SharedShape::new(tris[0])),
                (Pose::identity(), SharedShape::new(tris[1])),
                (Pose::identity(), SharedShape::new(tris[2])),
                (Pose::identity(), SharedShape::new(tris[3])),
                (Pose::identity(), SharedShape::new(tris[4])),
                (Pose::identity(), SharedShape::new(pent.clone())),
            ]
        );

        Self {
            vertices,
            material,
            compound,
            pose: Pose::from_translation(center),
            pent,
            tris
        }
    }

    pub fn flip(&self, face: PentFace) -> PentSlice {
        let mut vertices = self.vertices.clone();

        // Apply center to the vertices
        vertices.iter_mut().for_each(|mut v| {
            *v += self.pose.translation;
        });

        match face {
            PentFace::Tri(face_index) => {
                let face = &self.tris[face_index];

                for v in &mut vertices {
                    let mut is_face = false;
                    for fv in face.vertices() {
                        if fv == v {
                            is_face = true;
                            break;
                        }
                    }

                    if !is_face {
                        let pose = Pose {
                            rotation: Rot3::IDENTITY,
                            translation: face.a + self.pose.translation,
                            padding: 0
                        };

                        // Invert the vertex along the face
                        let space = parry3d::shape::HalfSpace::new(face.robust_normal());
                        let projected = space.project_point(
                            &pose,
                            v.clone(),
                            false
                        ).point;
                        let dist = space.distance_to_point(
                            &pose,
                            v.clone(),
                            false
                        );
                        *v = projected - dist * face.normal().unwrap();
                    }
                }
            }
            PentFace::Pent => {
                let face = Triangle::new(
                    self.vertices[1] + self.pose.translation,
                    self.vertices[3] + self.pose.translation,
                    self.vertices[4] + self.pose.translation,
                );

                for v in &mut vertices {
                    let mut is_face = false;
                    for fv in face.vertices() {
                        if fv == v {
                            is_face = true;
                            break;
                        }
                    }

                    if !is_face {
                        let pose = Pose {
                            rotation: Rot3::IDENTITY,
                            translation: face.a,
                            padding: 0
                        };

                        // Invert the vertex along the face
                        let space = parry3d::shape::HalfSpace::new(face.robust_normal());
                        let projected = space.project_point(
                            &pose,
                            v.clone(),
                            false
                        ).point;
                        let dist = space.distance_to_point(
                            &pose,
                            v.clone(),
                            false
                        );
                        *v = projected - dist * face.normal().unwrap();
                    }
                }
            }
        }

        // Calculate the center pos
        let mut center = Vec3::ZERO;
        vertices.iter().for_each(|v| {
            center += v;
        });
        center = center / vertices.len() as f32;

        // Move the vertices around the center pos
        vertices.iter_mut().for_each(|mut v| {
            *v -= center;
        });

        let tris = vec![
            Triangle::new(
                vertices[0],
                vertices[2],
                vertices[1],
            ),
            Triangle::new(
                vertices[0],
                vertices[3],
                vertices[2],
            ),
            Triangle::new(
                vertices[0],
                vertices[4],
                vertices[3],
            ),
            Triangle::new(
                vertices[0],
                vertices[5],
                vertices[4],
            ),
            Triangle::new(
                vertices[0],
                vertices[1],
                vertices[5],
            ),
        ];

        let pent_verts = vec![
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[4],
            vertices[5],
        ];
        let pent = shape::ConvexPolyhedron::from_convex_hull(
            &pent_verts
        ).unwrap();

        let compound = shape::Compound::new(
            vec![
                (Pose::identity(), SharedShape::new(tris[0])),
                (Pose::identity(), SharedShape::new(tris[1])),
                (Pose::identity(), SharedShape::new(tris[2])),
                (Pose::identity(), SharedShape::new(tris[3])),
                (Pose::identity(), SharedShape::new(tris[4])),
                (Pose::identity(), SharedShape::new(pent.clone())),
            ]
        );

        PentSlice {
            pose: Pose::from_translation(center),
            vertices,
            tris,
            pent,
            compound,
            material: self.material
        }
    }
}

impl CastObject for PentSlice {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let mut min_hit: Option<Hit> = None;

        let scale = Vec3::new(0.8, 0.8, 0.8);
        self.tris.iter().for_each(|o| {
            if let Some(hit) = o.scaled(scale).cast_ray_and_get_normal(
                &self.pose,
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

        if let Some(hit) = self.pent.clone().scaled(scale).unwrap().cast_ray_and_get_normal(
            &self.pose,
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

        // let radius = 0.02f32;
        // self.vertices.iter().for_each(|o| {
        //     let p = Pose {
        //         rotation: Rot3::IDENTITY,
        //         translation: *o,
        //         padding: 0
        //     };
        //     if let Some(hit) = parry3d::shape::Ball::new(radius).cast_ray_and_get_normal(
        //         &p,
        //         ray,
        //         9999.0f32,
        //         true
        //     ) {
        //         if min_hit.is_none() || hit.time_of_impact < min_hit.as_ref().unwrap().distance {
        //             min_hit = Some(Hit {
        //                 distance: hit.time_of_impact,
        //                 object: self,
        //                 normal: hit.normal
        //             });
        //         }
        //     }
        // });

        min_hit
    }

    fn material(&self) -> &Material {
        &self.material
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
            (i % self.width, i / self.width, data)
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
    let mut graph = Graph::<PentSlice, ()>::new();

    let size = 1024;
    let mut sink = Sink::new(size, size);

    let mut world = World::new();

    let mut slice = PentSlice::new(
        Material {
            color: Vec3::new(0.0, 1.0, 1.0),
            reflective: false
        }
    );
    world.objects.push(Box::new(slice.clone()));
    let mut last_index = graph.add_node(slice.clone());

    let scale = Vec3::new(0.9, 0.9, 0.9);
    for i in 0..62 {

        let dir = if i % 8 == 0 {
            PentFace::Pent
            // PentFace::Tri(0)
        } else {
            PentFace::Tri((i + 1) % 5)
        };

        slice = slice.flip(dir);
        // slice.material.color = Vec3::new(random(), random(), random());
        slice.material.color = Vec3::new(1.0, 0.0, 0.0);

        // Check if there's any intersection
        let collides = graph.raw_nodes().iter().any(|n| {
            let collision_slice = &n.weight.compound.clone().scale_dyn(scale, 1).unwrap();
            if intersection_test(&slice.pose, &slice.compound, &n.weight.pose, collision_slice.as_ref()).unwrap() {
                return true;
            }
            false
        });

        if collides {
            continue
        }

        // Add the pent for raycasting
        world.objects.push(Box::new(slice.clone()));
    }

    println!("Finished computing mesh");

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

    let light_dir = Vec3::new(4.0, -0.4, 1.8).normalize();

    sink.get_mut_vec()
        .into_par_iter()
        // .into_iter()
        .for_each(|(x, y, data)| {
            let rx = x as f32 / size as f32;
            let ry = y as f32 / size as f32;
            let spread = 2.5;
            let ray_dir = Vec3::new(rx * 2.0 - 1.0, -ry * 2.0 + 1.0, spread).normalize();
            let mut ray = Ray {
                origin: Vec3::new(0.0, 2.33, -7.4),
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

                light = f32::max(0.4f32, light);

                if shadowed {
                    // light = f32::min(0.2, light);
                    light *= 0.5;
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
