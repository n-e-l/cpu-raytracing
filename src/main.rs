use std::fmt::Write;
use indicatif::ParallelProgressIterator;
use std::cmp::min;
use std::fs;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Mul;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use glam::Vec3;
use indicatif::ProgressIterator;
use parry3d::math::{Pose, Rot3};
use parry3d::query::{intersection_test, PointQuery, Ray, RayCast};
use parry3d::shape;
use parry3d::shape::{CompositeShape, Compound, ConvexPolyhedron, FeatureId, Shape, SharedShape, TriMesh, Triangle};
use petgraph::data::Build;
use petgraph::Graph;
use petgraph::graph::{GraphIndex, NodeIndex};
use rand::{random, random_bool, random_range};
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
    mesh: TriMesh,
    scaled_mesh: TriMesh,
    tris: Vec<Triangle>,
}

#[derive(Clone, Copy)]
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

        let indices: Vec<[u32; 3]> = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
        ];

        let mesh = TriMesh::new(
            vertices.clone(),
            indices
        ).unwrap();

        let scale = Vec3::new(0.9, 0.9, 0.9);
        let scaled_mesh = mesh.clone().scaled(scale);

        Self {
            vertices,
            material,
            mesh,
            scaled_mesh,
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

        let indices: Vec<[u32; 3]> = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
        ];

        let mesh = TriMesh::new(
            vertices.clone(),
            indices
        ).unwrap();

        let scale = Vec3::new(0.8, 0.8, 0.8);
        let scaled_mesh = mesh.clone().scaled(scale);

        PentSlice {
            pose: Pose::from_translation(center),
            vertices,
            tris,
            mesh,
            scaled_mesh,
            pent,
            material: self.material,
        }
    }
}

impl CastObject for PentSlice {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let mut min_hit: Option<Hit> = None;

        if let Some(hit) = self.scaled_mesh.cast_ray_and_get_normal(
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

        min_hit
    }

    fn material(&self) -> &Material {
        &self.material
    }
}

struct Hit<'a> {
    distance: f32,
    object: &'a dyn CastObject,
    normal: Vec3
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

fn build_uber(slices: &[PentSlice]) -> (TriMesh, Vec<Material>) {
    let mut verts: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut materials: Vec<Material> = Vec::new();

    for s in slices {
        let base = verts.len() as u32;
        // pose is translation-only here, so bake it directly
        for v in s.scaled_mesh.vertices() {
            verts.push(*v + s.pose.translation);
        }
        for tri in s.scaled_mesh.indices() {
            indices.push([tri[0] + base, tri[1] + base, tri[2] + base]);
            materials.push(s.material); // one entry per triangle
        }
    }
    (TriMesh::new(verts, indices).unwrap(), materials)
}

fn bake_uber_vcolor(mesh: &TriMesh, materials: &[Material]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let src_verts = mesh.vertices();
    let src_indices = mesh.indices();
    assert_eq!(src_indices.len(), materials.len(), "one material per triangle");

    let mut verts = Vec::with_capacity(src_indices.len() * 3);
    let mut cols = Vec::with_capacity(src_indices.len() * 3);
    let mut idx = Vec::with_capacity(src_indices.len());

    for (tri, mat) in src_indices.iter().zip(materials) {
        let base = verts.len() as u32;
        let c = mat.color;           // the baked color
        for &vi in tri {
            let p = src_verts[vi as usize];
            verts.push([p.x, p.y, p.z]);
            cols.push(c.into());
        }
        idx.push([base, base + 1, base + 2]);
    }
    (verts, cols, idx)
}

fn write_obj_vcolor(
    path: impl AsRef<Path>,
    vertices: &[[f32; 3]],
    colors: &[[f32; 3]],     // one per vertex, same length as vertices
    indices: &[[u32; 3]],
) -> std::io::Result<()> {
    let mut s = String::new();
    for (v, c) in vertices.iter().zip(colors) {
        writeln!(s, "v {} {} {} {} {} {}", v[0], v[1], v[2], c[0], c[1], c[2]).unwrap();
    }
    for f in indices {
        writeln!(s, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1).unwrap();
    }
    fs::write(path, s)
}

fn trimesh_to_obj(path: impl AsRef<Path>, mesh: &TriMesh, materials: &Vec<Material>) -> std::io::Result<()> {
    let verts: Vec<[f32; 3]> = mesh.vertices().iter().map(|p| [p.x, p.y, p.z]).collect();
    let idx: Vec<[u32; 3]> = mesh.indices().iter().map(|i| [i[0], i[1], i[2]]).collect();
    let (verts, cols, idx) = bake_uber_vcolor(mesh, &materials);
    write_obj_vcolor(path, &verts, &cols, &idx)
}

struct World {
    graph: Graph::<(Option<PentSlice>), PentFace>,
    slices: Vec<PentSlice>,
    root: NodeIndex
}

impl World {
    fn new() -> Self {
        let mut graph = Graph::<(Option<PentSlice>), PentFace>::new();

        let mut slice = PentSlice::new(
            Material {
                color: Vec3::new(0.0, 1.0, 1.0),
                reflective: false
            }
        );
        let root = graph.add_node(Some(slice.clone()));

        Self {
            graph,
            slices: vec![slice],
            root
        }
    }

    /// Tries to add a slice. Returns none if there was no physical space to add it
    fn add_slice(&mut self, index: NodeIndex, face: PentFace, slice: PentSlice) -> Option<NodeIndex> {

        // Check if there's any intersection
        // let collides = self.graph.raw_nodes().iter().any(|n| {
        //     let pent = &n.weight;
        //     if let Some(pent) = pent {
        //         if intersection_test(&slice.pose, &slice.mesh, &pent.pose, &pent.scaled_mesh).unwrap() {
        //             return true;
        //         }
        //     }
        //     false
        // });
        //
        // if collides {
        //     return None
        // }

        self.slices.push(slice.clone());
        let index = self.graph.add_node(Some(slice));

        Some(index)
    }

    fn get_slice(&self, i: usize) -> &PentSlice {
        &self.slices[i]
    }
}

fn main() {

    let size = 1024;
    let mut sink = Sink::new(size, size);

    let mut world = World::new();

    let mut index = world.root;
    let mut slice = world.slices.get(0).unwrap().clone();

    let count = 34492;
    for i in 0..count {

        let mut color = Vec3::new(1.0, 0.0, 0.0);
        let dir = if i % 90 == 0 {
            color = Vec3::new(1.0, 0.0, 0.0);
            PentFace::Tri(2)
        } else if i % 7 == 0 {
            color = Vec3::new(0.0, 0.0, 1.0);
            PentFace::Pent
        } else {
            let t = i as f32 / count as f32;
            color = Vec3::new(1.0, 0.0, 0.0) * t + Vec3::new(1.0, 1.0, 1.0) * (1.0 - t);
            PentFace::Tri(i % 5)
        };

        slice = slice.flip(dir).clone();
        // slice.material.color = Vec3::new(random(), random(), random());
        slice.material.color = color;

        world.add_slice(index, dir, slice.clone());
    }

    println!("Finished computing mesh");

    // Generate world mesh to optimize ray traversal
    let (uber, materials) = build_uber(&world.slices);

    // Export
    trimesh_to_obj("mesh.obj", &uber, &materials);

    let light_dir = Vec3::new(2.0, -0.4, 2.8).normalize();
    let total = size * size; // or sink.get_mut_vec().len()

    let target = Vec3::new(-1.0, 0.4, -3.8);           // center of the scene
    let radius = 12.5;
    let height = 1.93;
    let angle: f32 = -1.7;              // azimuth in radians — this is what you sweep

    let eye = target + Vec3::new(angle.sin() * radius, height, -angle.cos() * radius);

    // look-at basis
    let world_up = Vec3::Y;
    let forward = (target - eye).normalize();
    let right   = forward.cross(world_up).normalize();
    let up      = right.cross(forward);
    let spread = 2.5;

    let start_time = Instant::now();
    sink.get_mut_vec()
        .into_par_iter()
        .progress_count(total as u64)
        // .into_iter()
        .for_each(|(x, y, data)| {
            let rx = x as f32 / size as f32;
            let ry = y as f32 / size as f32;

            let px = rx * 2.0 - 1.0;
            let py = -ry * 2.0 + 1.0;
            let ray_dir = (right * px + up * py + forward * spread).normalize();
            let mut ray = Ray { origin: eye, dir: ray_dir };

            if let Some(hit) = uber.cast_ray_and_get_normal(&Pose::identity(), &ray, 9999.0, true) {
                let face = match hit.feature {
                    FeatureId::Face(i) => i,
                    _ => 0,
                };
                let tri_id = face as usize % materials.len(); // see caveat below
                let mat = materials[tri_id];

                let mut color = mat.color;
                // let mut color = 1.0 - (Vec3::new(hit.distance, hit.distance, hit.distance) * 0.3) * hit.object.material().color;

                let mut light = hit.normal.dot(-light_dir);
                let point = ray.point_at(hit.time_of_impact);

                // Cast shadow
                let shadow_ray = Ray {
                    origin: point - light_dir * 0.001,
                    dir: -light_dir
                };
                let shadowed = uber.cast_ray(&Pose::identity(), &shadow_ray, 9999.0, true).is_some();

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

    let duration = Instant::now().duration_since(start_time).as_millis() as f32 / 1000.0;
    println!("Took {duration} seconds");

    let path = Path::new("out.png");
    write_image(sink, path);
}
