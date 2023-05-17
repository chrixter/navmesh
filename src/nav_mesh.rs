use crate::{
    primitives::{self, NavVec3, SpadePoint},
    Error, NavConnection, NavResult, Scalar, ZERO_TRESHOLD,
};
use petgraph::{
    algo::{astar, tarjan_scc},
    graph::NodeIndex,
    visit::EdgeRef,
    Graph, Undirected,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spade::{rtree::RTree, BoundingRect, PointN, SpatialObject};
#[cfg(not(feature = "scalar64"))]
use std::f32::MAX as SCALAR_MAX;
#[cfg(feature = "scalar64")]
use std::f64::MAX as SCALAR_MAX;
use std::{
    collections::HashMap,
    ops::{Div, Mul},
};
use typid::ID;

#[cfg(feature = "parallel")]
macro_rules! iter {
    ($v:expr) => {
        $v.par_iter()
    };
}
#[cfg(not(feature = "parallel"))]
macro_rules! iter {
    ($v:expr) => {
        $v.iter()
    };
}
#[cfg(feature = "parallel")]
macro_rules! into_iter {
    ($v:expr) => {
        $v.into_par_iter()
    };
}
#[cfg(not(feature = "parallel"))]
macro_rules! into_iter {
    ($v:expr) => {
        $v.into_iter()
    };
}

/// Nav mash identifier.
pub type NavMeshID<T> = ID<NavMesh<T>>;

/// Nav mesh triangle description - lists used vertices indices.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct NavTriangle {
    pub first: u32,
    pub second: u32,
    pub third: u32,
}

impl From<(u32, u32, u32)> for NavTriangle {
    fn from(value: (u32, u32, u32)) -> Self {
        Self {
            first: value.0,
            second: value.1,
            third: value.2,
        }
    }
}

impl From<[u32; 3]> for NavTriangle {
    fn from(value: [u32; 3]) -> Self {
        Self {
            first: value[0],
            second: value[1],
            third: value[2],
        }
    }
}

/// Nav mesh area descriptor. Nav mesh area holds information about specific nav mesh triangle.
#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct NavArea<T: NavVec3> {
    /// Triangle index.
    pub triangle: u32,
    /// Area size (triangle area value).
    pub size: Scalar,
    /// Traverse cost factor. Big values tells that this area is hard to traverse, smaller tells
    /// the opposite.
    pub cost: Scalar,
    /// Triangle center point.
    pub center: T,
    /// Radius of sphere that contains this triangle.
    pub radius: Scalar,
    /// Squared version of `radius`.
    pub radius_sqr: Scalar,
}

impl<T: NavVec3> NavArea<T> {
    /// Calculate triangle area value.
    ///
    /// # Arguments
    /// * `a` - first vertice point.
    /// * `b` - second vertice point.
    /// * `c` - thirs vertice point.
    #[inline]
    pub fn calculate_area(a: T, b: T, c: T) -> Scalar {
        let ab = b - a;
        let ac = c - a;
        ab.cross(ac).magnitude() * 0.5
    }

    /// Calculate triangle center point.
    ///
    /// # Arguments
    /// * `a` - first vertice point.
    /// * `b` - second vertice point.
    /// * `c` - thirs vertice point.
    #[inline]
    pub fn calculate_center(a: T, b: T, c: T) -> T {
        let v = a + b + c;
        NavVec3::new(v.x() / 3.0, v.y() / 3.0, v.z() / 3.0)
    }
}

#[derive(Debug, Clone)]
pub struct NavSpatialObject<T: NavVec3> {
    pub index: usize,
    pub a: T,
    pub b: T,
    pub c: T,
    normal: T,
    dab: T,
    dbc: T,
    dca: T,
}

impl<T: NavVec3> NavSpatialObject<T> {
    pub fn new(index: usize, a: T, b: T, c: T) -> Self {
        let ab = b - a;
        let bc = c - b;
        let ca = a - c;
        let normal = (a - b).cross(a - c).normalize();
        let dab = normal.cross(ab);
        let dbc = normal.cross(bc);
        let dca = normal.cross(ca);
        Self {
            index,
            a,
            b,
            c,
            normal,
            dab,
            dbc,
            dca,
        }
    }

    #[inline]
    pub fn normal(&self) -> T {
        self.normal
    }

    pub fn closest_point(&self, point: T) -> T {
        let pab = point.project(self.a, self.b);
        let pbc = point.project(self.b, self.c);
        let pca = point.project(self.c, self.a);
        if pca > 1.0 && pab < 0.0 {
            return self.a;
        } else if pab > 1.0 && pbc < 0.0 {
            return self.b;
        } else if pbc > 1.0 && pca < 0.0 {
            return self.c;
        } else if (0.0..=1.0).contains(&pab) && !point.is_above_plane(self.a, self.dab) {
            return T::unproject(self.a, self.b, pab);
        } else if (0.0..=1.0).contains(&pbc) && !point.is_above_plane(self.b, self.dbc) {
            return T::unproject(self.b, self.c, pbc);
        } else if (0.0..=1.0).contains(&pca) && !point.is_above_plane(self.c, self.dca) {
            return T::unproject(self.c, self.a, pca);
        }
        point.project_on_plane(self.a, self.normal)
    }
}

impl<T> SpatialObject for NavSpatialObject<T>
where
    T: NavVec3,
    SpadePoint<T>: PointN<Scalar = Scalar>,
{
    type Point = SpadePoint<T>;

    fn mbr(&self) -> BoundingRect<Self::Point> {
        let min = T::new(
            self.a.x().min(*self.b.x()).min(*self.c.x()),
            self.a.y().min(*self.b.y()).min(*self.c.y()),
            self.a.z().min(*self.b.z()).min(*self.c.z()),
        );
        let max = T::new(
            self.a.x().max(*self.b.x()).max(*self.c.x()),
            self.a.y().max(*self.b.y()).max(*self.c.y()),
            self.a.z().max(*self.b.z()).max(*self.c.z()),
        );
        BoundingRect::from_corners(&SpadePoint(min), &SpadePoint(max))
    }

    fn distance2(&self, point: &Self::Point) -> Scalar {
        (point.0 - self.closest_point(point.0)).sqr_magnitude()
    }
}

/// Quality of querying a point on nav mesh.
#[derive(Debug, Clone, Copy)]
pub enum NavQuery {
    /// Best quality, totally accurate.
    Accuracy,
    /// Medium quality, finds point in closest triangle.
    Closest,
    /// Low quality, finds first triangle in range of query.
    ClosestFirst,
}

/// Quality of finding path.
#[derive(Debug, Clone, Copy)]
pub enum NavPathMode {
    /// Best quality, finds shortest path.
    Accuracy,
    /// Medium quality, finds shortest path througs triangles midpoints.
    MidPoints,
}

/// Nav mesh object used to find shortest path between two points.
#[derive(Debug, Default, Clone)]
pub struct NavMesh<T>
where
    T: NavVec3,
    NavSpatialObject<T>: SpatialObject<Point = SpadePoint<T>>,
{
    id: NavMeshID<T>,
    vertices: Vec<T>,
    triangles: Vec<NavTriangle>,
    areas: Vec<NavArea<T>>,
    // {triangle connection: (distance sqr, vertex connection)}
    connections: HashMap<NavConnection, (Scalar, NavConnection)>,
    graph: Graph<(), Scalar, Undirected>,
    nodes: Vec<NodeIndex>,
    nodes_map: HashMap<NodeIndex, usize>,
    rtree: RTree<NavSpatialObject<T>>,
    spatials: Vec<NavSpatialObject<T>>,
    origin: T,
}

impl<T> NavMesh<T>
where
    T: NavVec3 + PartialEq + Mul<Scalar, Output = T> + Div<Scalar, Output = T>,
    NavSpatialObject<T>: SpatialObject<Point = SpadePoint<T>>,
    SpadePoint<T>: PointN<Scalar = Scalar>,
{
    /// Create new nav mesh object from vertices and triangles.
    ///
    /// # Arguments
    /// * `vertices` - list of vertices points.
    /// * `triangles` - list of vertices indices that produces triangles.
    ///
    /// # Returns
    /// `Ok` with nav mesh object or `Err` with `Error::TriangleVerticeIndexOutOfBounds` if input
    /// data is invalid.
    ///
    /// # Example
    /// ```
    /// use navmesh::*;
    ///
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0].into(), // 0
    ///     [1.0, 0.0, 0.0].into(), // 1
    ///     [2.0, 0.0, 1.0].into(), // 2
    ///     [0.0, 1.0, 0.0].into(), // 3
    ///     [1.0, 1.0, 0.0].into(), // 4
    ///     [2.0, 1.0, 1.0].into(), // 5
    /// ];
    /// let triangles = vec![
    ///     (0, 1, 4).into(), // 0
    ///     (4, 3, 0).into(), // 1
    ///     (1, 2, 5).into(), // 2
    ///     (5, 4, 1).into(), // 3
    /// ];
    ///
    /// let mesh = NavMesh::<NVec3>::new(vertices, triangles).unwrap();
    /// ```
    pub fn new(vertices: Vec<T>, triangles: Vec<NavTriangle>) -> NavResult<Self> {
        let origin =
            vertices.iter().cloned().fold(T::zero(), |a, v| a + v) / vertices.len() as Scalar;

        let areas = iter!(triangles)
            .enumerate()
            .map(|(i, triangle)| {
                if triangle.first >= vertices.len() as u32 {
                    return Err(Error::TriangleVerticeIndexOutOfBounds(
                        i as u32,
                        0,
                        triangle.first,
                    ));
                }
                if triangle.second >= vertices.len() as u32 {
                    return Err(Error::TriangleVerticeIndexOutOfBounds(
                        i as u32,
                        1,
                        triangle.second,
                    ));
                }
                if triangle.third >= vertices.len() as u32 {
                    return Err(Error::TriangleVerticeIndexOutOfBounds(
                        i as u32,
                        2,
                        triangle.third,
                    ));
                }
                let first = vertices[triangle.first as usize];
                let second = vertices[triangle.second as usize];
                let third = vertices[triangle.third as usize];
                let center = NavArea::calculate_center(first, second, third);
                let radius = (first - center)
                    .magnitude()
                    .max((second - center).magnitude())
                    .max((third - center).magnitude());
                Ok(NavArea {
                    triangle: i as u32,
                    size: NavArea::calculate_area(first, second, third),
                    cost: 1.0,
                    center,
                    radius,
                    radius_sqr: radius * radius,
                })
            })
            .collect::<NavResult<Vec<_>>>()?;

        // {edge: [triangle index]}
        let mut edges = HashMap::<NavConnection, Vec<usize>>::with_capacity(triangles.len() * 3);
        for (index, triangle) in triangles.iter().enumerate() {
            let edge_a = NavConnection(triangle.first, triangle.second);
            let edge_b = NavConnection(triangle.second, triangle.third);
            let edge_c = NavConnection(triangle.third, triangle.first);
            if let Some(tris) = edges.get_mut(&edge_a) {
                tris.push(index);
            } else {
                edges.insert(edge_a, vec![index]);
            }
            if let Some(tris) = edges.get_mut(&edge_b) {
                tris.push(index);
            } else {
                edges.insert(edge_b, vec![index]);
            }
            if let Some(tris) = edges.get_mut(&edge_c) {
                tris.push(index);
            } else {
                edges.insert(edge_c, vec![index]);
            }
        }

        let connections = into_iter!(iter!(edges)
            .flat_map(|(verts, tris)| {
                let mut result = HashMap::with_capacity(tris.len() * tris.len());
                for a in tris {
                    for b in tris {
                        if a != b {
                            result.insert(NavConnection(*a as u32, *b as u32), *verts);
                        }
                    }
                }
                result
            })
            .collect::<HashMap<_, _>>())
        .map(|(tri_conn, vert_conn)| {
            let a = areas[tri_conn.0 as usize].center;
            let b = areas[tri_conn.1 as usize].center;
            let weight = (b - a).sqr_magnitude();
            (tri_conn, (weight, vert_conn))
        })
        .collect::<HashMap<_, _>>();

        let mut graph = Graph::<(), Scalar, Undirected>::new_undirected();
        let nodes = (0..triangles.len())
            .map(|_| graph.add_node(()))
            .collect::<Vec<_>>();
        graph.extend_with_edges(
            iter!(connections)
                .map(|(conn, (w, _))| (nodes[conn.0 as usize], nodes[conn.1 as usize], w))
                .collect::<Vec<_>>(),
        );
        let nodes_map = iter!(nodes).enumerate().map(|(i, n)| (*n, i)).collect();

        let spatials = iter!(triangles)
            .enumerate()
            .map(|(index, triangle)| {
                NavSpatialObject::new(
                    index,
                    vertices[triangle.first as usize],
                    vertices[triangle.second as usize],
                    vertices[triangle.third as usize],
                )
            })
            .collect::<Vec<_>>();

        let mut rtree = RTree::new();
        for spatial in &spatials {
            rtree.insert(spatial.clone());
        }

        Ok(Self {
            id: ID::new(),
            vertices,
            triangles,
            areas,
            connections,
            graph,
            nodes,
            nodes_map,
            rtree,
            spatials,
            origin,
        })
    }

    pub fn thicken(&self, value: Scalar) -> NavResult<Self> {
        let shifted = iter!(self.vertices)
            .enumerate()
            .map(|(i, v)| {
                let (mut n, c) = self
                    .triangles
                    .iter()
                    .enumerate()
                    .filter_map(|(j, t)| {
                        if t.first == i as u32 || t.second == i as u32 || t.third == i as u32 {
                            Some(self.spatials[j].normal)
                        } else {
                            None
                        }
                    })
                    .fold((T::zero(), 0), |a, v| (a.0 + v, a.1 + 1));
                if c > 1 {
                    n = n / c as Scalar;
                }
                *v + n.normalize() * value
            })
            .collect::<Vec<_>>();
        Self::new(shifted, self.triangles.clone())
    }

    pub fn scale(&self, value: T, origin: Option<T>) -> NavResult<Self> {
        let origin = origin.unwrap_or(self.origin);
        let vertices = iter!(self.vertices)
            .map(|v| (*v - origin).mul_elem(value) + origin)
            .collect::<Vec<_>>();
        Self::new(vertices, self.triangles.clone())
    }

    /// Nav mesh identifier.
    #[inline]
    pub fn id(&self) -> NavMeshID<T> {
        self.id
    }

    /// Nav mesh origin point.
    #[inline]
    pub fn origin(&self) -> T {
        self.origin
    }

    /// Reference to list of nav mesh vertices points.
    #[inline]
    pub fn vertices(&self) -> &[T] {
        &self.vertices
    }

    /// Reference to list of nav mesh triangles.
    #[inline]
    pub fn triangles(&self) -> &[NavTriangle] {
        &self.triangles
    }

    /// Reference to list of nav mesh area descriptors.
    #[inline]
    pub fn areas(&self) -> &[NavArea<T>] {
        &self.areas
    }

    /// Set area cost by triangle index.
    ///
    /// # Arguments
    /// * `index` - triangle index.
    /// * `cost` - cost factor.
    ///
    /// # Returns
    /// Old area cost value.
    #[inline]
    pub fn set_area_cost(&mut self, index: usize, cost: Scalar) -> Scalar {
        let area = &mut self.areas[index];
        let old = area.cost;
        let cost = cost.max(0.0);
        area.cost = cost;
        old
    }

    /// Find closest point on nav mesh.
    ///
    /// # Arguments
    /// * `point` - query point.
    /// * `query` - query quality.
    ///
    /// # Returns
    /// `Some` with point on nav mesh if found or `None` otherwise.
    pub fn closest_point(&self, point: T, query: NavQuery) -> Option<T> {
        self.find_closest_triangle(point, query)
            .map(|triangle| self.spatials[triangle].closest_point(point))
    }

    /// Find shortest path on nav mesh between two points.
    ///
    /// # Arguments
    /// * `from` - query point from.
    /// * `to` - query point to.
    /// * `query` - query quality.
    /// * `mode` - path finding quality.
    ///
    /// # Returns
    /// `Some` with path points on nav mesh if found or `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use navmesh::*;
    ///
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0].into(), // 0
    ///     [1.0, 0.0, 0.0].into(), // 1
    ///     [2.0, 0.0, 1.0].into(), // 2
    ///     [0.0, 1.0, 0.0].into(), // 3
    ///     [1.0, 1.0, 0.0].into(), // 4
    ///     [2.0, 1.0, 1.0].into(), // 5
    /// ];
    /// let triangles = vec![
    ///     (0, 1, 4).into(), // 0
    ///     (4, 3, 0).into(), // 1
    ///     (1, 2, 5).into(), // 2
    ///     (5, 4, 1).into(), // 3
    /// ];
    ///
    /// let mesh = NavMesh::<NVec3>::new(vertices, triangles).unwrap();
    ///
    /// let path = mesh
    ///     .find_path(
    ///         [0.0, 1.0, 0.0].into(),
    ///         [1.5, 0.25, 0.5].into(),
    ///         NavQuery::Accuracy,
    ///         NavPathMode::MidPoints,
    ///     )
    ///     .unwrap();
    /// assert_eq!(
    ///     path.into_iter()
    ///         .map(|v| (
    ///             (v.x * 10.0) as i32,
    ///             (v.y * 10.0) as i32,
    ///             (v.z * 10.0) as i32,
    ///         ))
    ///         .collect::<Vec<_>>(),
    ///     vec![(0, 10, 0), (10, 5, 0), (15, 2, 5),]
    /// );
    /// ```
    pub fn find_path(&self, from: T, to: T, query: NavQuery, mode: NavPathMode) -> Option<Vec<T>> {
        self.find_path_custom(from, to, query, mode, |_, _, _| true)
    }

    /// Find shortest path on nav mesh between two points, providing custom filtering function.
    ///
    /// # Arguments
    /// * `from` - query point from.
    /// * `to` - query point to.
    /// * `query` - query quality.
    /// * `mode` - path finding quality.
    /// * `filter` - closure that gives you a connection distance squared, first triangle index
    ///   and second triangle index.
    ///
    /// # Returns
    /// `Some` with path points on nav mesh if found or `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use navmesh::*;
    ///
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0].into(), // 0
    ///     [1.0, 0.0, 0.0].into(), // 1
    ///     [2.0, 0.0, 1.0].into(), // 2
    ///     [0.0, 1.0, 0.0].into(), // 3
    ///     [1.0, 1.0, 0.0].into(), // 4
    ///     [2.0, 1.0, 1.0].into(), // 5
    /// ];
    /// let triangles = vec![
    ///     (0, 1, 4).into(), // 0
    ///     (4, 3, 0).into(), // 1
    ///     (1, 2, 5).into(), // 2
    ///     (5, 4, 1).into(), // 3
    /// ];
    ///
    /// let mesh = NavMesh::<NVec3>::new(vertices, triangles).unwrap();
    ///
    /// let path = mesh
    ///     .find_path_custom(
    ///         [0.0, 1.0, 0.0].into(),
    ///         [1.5, 0.25, 0.5].into(),
    ///         NavQuery::Accuracy,
    ///         NavPathMode::MidPoints,
    ///         |_dist_sqr, _first_idx, _second_idx| true,
    ///     )
    ///     .unwrap();
    /// assert_eq!(
    ///     path.into_iter()
    ///         .map(|v| (
    ///             (v.x * 10.0) as i32,
    ///             (v.y * 10.0) as i32,
    ///             (v.z * 10.0) as i32,
    ///         ))
    ///         .collect::<Vec<_>>(),
    ///     vec![(0, 10, 0), (10, 5, 0), (15, 2, 5),]
    /// );
    /// ```
    pub fn find_path_custom<F>(
        &self,
        from: T,
        to: T,
        query: NavQuery,
        mode: NavPathMode,
        filter: F,
    ) -> Option<Vec<T>>
    where
        F: FnMut(Scalar, usize, usize) -> bool,
    {
        if from.coincides(to) {
            return None;
        }
        let start = self.find_closest_triangle(from, query)?;
        let end = self.find_closest_triangle(to, query)?;
        let from = self.spatials[start].closest_point(from);
        let to = self.spatials[end].closest_point(to);
        let (triangles, _) = self.find_path_triangles_custom(start, end, filter)?;
        if triangles.is_empty() {
            return None;
        } else if triangles.len() == 1 {
            return Some(vec![from, to]);
        }
        match mode {
            NavPathMode::Accuracy => Some(self.find_path_accuracy(from, to, &triangles)),
            NavPathMode::MidPoints => Some(self.find_path_midpoints(from, to, &triangles)),
        }
    }

    fn find_path_accuracy(&self, from: T, to: T, triangles: &[usize]) -> Vec<T> {
        #[derive(Debug)]
        enum Node<U> {
            Point(U),
            // (a, b, normal)
            LevelChange(U, U, U),
        }

        // TODO: reduce allocations.
        if triangles.len() == 2 {
            let NavConnection(a, b) =
                self.connections[&NavConnection(triangles[0] as u32, triangles[1] as u32)].1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let n = self.spatials[triangles[0]].normal();
            let m = self.spatials[triangles[1]].normal();
            if !Self::is_line_between_points(from, to, a, b, n) {
                let da = (from - a).sqr_magnitude();
                let db = (from - b).sqr_magnitude();
                let point = if da < db { a } else { b };
                return vec![from, point, to];
            } else if n.dot(m) < 1.0 - ZERO_TRESHOLD {
                let n = (b - a).normalize().cross(n);
                if let Some(point) = Self::raycast_line(from, to, a, b, n) {
                    return vec![from, point, to];
                }
            }
            return vec![from, to];
        }
        let mut start = from;
        let mut last_normal = self.spatials[triangles[0]].normal();
        let mut nodes = Vec::with_capacity(triangles.len() - 1);
        for triplets in triangles.windows(3) {
            let NavConnection(a, b) =
                self.connections[&NavConnection(triplets[0] as u32, triplets[1] as u32)].1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let NavConnection(c, d) =
                self.connections[&NavConnection(triplets[1] as u32, triplets[2] as u32)].1;
            let c = self.vertices[c as usize];
            let d = self.vertices[d as usize];
            let normal = self.spatials[triplets[1]].normal();
            let old_last_normal = last_normal;
            last_normal = normal;
            if !Self::is_line_between_points(start, c, a, b, normal)
                || !Self::is_line_between_points(start, d, a, b, normal)
            {
                let da = (start - a).sqr_magnitude();
                let db = (start - b).sqr_magnitude();
                start = if da < db { a } else { b };
                nodes.push(Node::Point(start));
            } else if old_last_normal.dot(normal) < 1.0 - ZERO_TRESHOLD {
                let normal = self.spatials[triplets[0]].normal();
                let normal = (b - a).normalize().cross(normal);
                nodes.push(Node::LevelChange(a, b, normal));
            }
        }
        {
            let NavConnection(a, b) = self.connections[&NavConnection(
                triangles[triangles.len() - 2] as u32,
                triangles[triangles.len() - 1] as u32,
            )]
                .1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let n = self.spatials[triangles[triangles.len() - 2]].normal();
            let m = self.spatials[triangles[triangles.len() - 1]].normal();
            if !Self::is_line_between_points(start, to, a, b, n) {
                let da = (start - a).sqr_magnitude();
                let db = (start - b).sqr_magnitude();
                let point = if da < db { a } else { b };
                nodes.push(Node::Point(point));
            } else if n.dot(m) < 1.0 - ZERO_TRESHOLD {
                let n = (b - a).normalize().cross(n);
                nodes.push(Node::LevelChange(a, b, n));
            }
        }

        let mut points = Vec::with_capacity(nodes.len() + 2);
        points.push(from);
        let mut point = from;
        for i in 0..nodes.len() {
            match nodes[i] {
                Node::Point(p) => {
                    point = p;
                    points.push(p);
                }
                Node::LevelChange(a, b, n) => {
                    let next = nodes
                        .iter()
                        .skip(i + 1)
                        .find_map(|n| match n {
                            Node::Point(p) => Some(*p),
                            _ => None,
                        })
                        .unwrap_or(to);
                    if let Some(p) = Self::raycast_line(point, next, a, b, n) {
                        points.push(p);
                    }
                }
            }
        }
        points.push(to);
        points.dedup();
        points
    }

    fn find_path_midpoints(&self, from: T, to: T, triangles: &[usize]) -> Vec<T> {
        if triangles.len() == 2 {
            let NavConnection(a, b) =
                self.connections[&NavConnection(triangles[0] as u32, triangles[1] as u32)].1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let n = self.spatials[triangles[0]].normal();
            let m = self.spatials[triangles[1]].normal();
            if n.dot(m) < 1.0 - ZERO_TRESHOLD || !Self::is_line_between_points(from, to, a, b, n) {
                return vec![from, (a + b) * 0.5, to];
            } else {
                return vec![from, to];
            }
        }
        let mut start = from;
        let mut last_normal = self.spatials[triangles[0]].normal();
        let mut points = Vec::with_capacity(triangles.len() + 1);
        points.push(from);
        for triplets in triangles.windows(3) {
            let NavConnection(a, b) =
                self.connections[&NavConnection(triplets[0] as u32, triplets[1] as u32)].1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let point = (a + b) * 0.5;
            let normal = self.spatials[triplets[1]].normal();
            let old_last_normal = last_normal;
            last_normal = normal;
            if old_last_normal.dot(normal) < 1.0 - ZERO_TRESHOLD {
                start = point;
                points.push(start);
            } else {
                let NavConnection(c, d) =
                    self.connections[&NavConnection(triplets[1] as u32, triplets[2] as u32)].1;
                let c = self.vertices[c as usize];
                let d = self.vertices[d as usize];
                let end = (c + d) * 0.5;
                if !Self::is_line_between_points(start, end, a, b, normal) {
                    start = point;
                    points.push(start);
                }
            }
        }
        {
            let NavConnection(a, b) = self.connections[&NavConnection(
                triangles[triangles.len() - 2] as u32,
                triangles[triangles.len() - 1] as u32,
            )]
                .1;
            let a = self.vertices[a as usize];
            let b = self.vertices[b as usize];
            let n = self.spatials[triangles[triangles.len() - 2]].normal();
            let m = self.spatials[triangles[triangles.len() - 1]].normal();
            if n.dot(m) < 1.0 - ZERO_TRESHOLD || !Self::is_line_between_points(start, to, a, b, n) {
                points.push((a + b) * 0.5);
            }
        }
        points.push(to);
        points.dedup();
        points
    }

    /// Find shortest path on nav mesh between two points.
    ///
    /// # Arguments
    /// * `from` - query point from.
    /// * `to` - query point to.
    /// * `query` - query quality.
    /// * `mode` - path finding quality.
    ///
    /// # Returns
    /// `Some` with path points on nav mesh and path length if found or `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use navmesh::*;
    ///
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0].into(), // 0
    ///     [1.0, 0.0, 0.0].into(), // 1
    ///     [2.0, 0.0, 1.0].into(), // 2
    ///     [0.0, 1.0, 0.0].into(), // 3
    ///     [1.0, 1.0, 0.0].into(), // 4
    ///     [2.0, 1.0, 1.0].into(), // 5
    /// ];
    /// let triangles = vec![
    ///     (0, 1, 4).into(), // 0
    ///     (4, 3, 0).into(), // 1
    ///     (1, 2, 5).into(), // 2
    ///     (5, 4, 1).into(), // 3
    /// ];
    ///
    /// let mesh = NavMesh::<NVec3>::new(vertices, triangles).unwrap();
    ///
    /// let path = mesh.find_path_triangles(1, 2).unwrap().0;
    /// assert_eq!(path, vec![1, 0, 3, 2]);
    /// ```
    #[inline]
    pub fn find_path_triangles(&self, from: usize, to: usize) -> Option<(Vec<usize>, Scalar)> {
        self.find_path_triangles_custom(from, to, |_, _, _| true)
    }

    /// Find shortest path on nav mesh between two points, providing custom filtering function.
    ///
    /// # Arguments
    /// * `from` - query point from.
    /// * `to` - query point to.
    /// * `query` - query quality.
    /// * `mode` - path finding quality.
    /// * `filter` - closure that gives you a connection distance squared, first triangle index
    ///   and second triangle index.
    ///
    /// # Returns
    /// `Some` with path points on nav mesh and path length if found or `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use navmesh::*;
    ///
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0].into(), // 0
    ///     [1.0, 0.0, 0.0].into(), // 1
    ///     [2.0, 0.0, 1.0].into(), // 2
    ///     [0.0, 1.0, 0.0].into(), // 3
    ///     [1.0, 1.0, 0.0].into(), // 4
    ///     [2.0, 1.0, 1.0].into(), // 5
    /// ];
    /// let triangles = vec![
    ///     (0, 1, 4).into(), // 0
    ///     (4, 3, 0).into(), // 1
    ///     (1, 2, 5).into(), // 2
    ///     (5, 4, 1).into(), // 3
    /// ];
    ///
    /// let mesh = NavMesh::<NVec3>::new(vertices, triangles).unwrap();
    ///
    /// let path = mesh.find_path_triangles_custom(
    ///     1,
    ///     2,
    ///     |_dist_sqr, _first_idx, _second_idx| true
    /// ).unwrap().0;
    /// assert_eq!(path, vec![1, 0, 3, 2]);
    /// ```
    #[inline]
    pub fn find_path_triangles_custom<F>(
        &self,
        from: usize,
        to: usize,
        mut filter: F,
    ) -> Option<(Vec<usize>, Scalar)>
    where
        F: FnMut(Scalar, usize, usize) -> bool,
    {
        let to = self.nodes[to];
        astar(
            &self.graph,
            self.nodes[from],
            |n| n == to,
            |e| {
                let a = self.nodes_map[&e.source()];
                let b = self.nodes_map[&e.target()];
                let w = *e.weight();
                if filter(w, a, b) {
                    let a = self.areas[a].cost;
                    let b = self.areas[b].cost;
                    w * a * b
                } else {
                    SCALAR_MAX
                }
            },
            |_| 0.0,
        )
        .map(|(c, v)| (iter!(v).map(|v| self.nodes_map[v]).collect(), c))
    }

    pub fn find_triangle_islands(&self) -> Vec<Vec<usize>> {
        tarjan_scc(&self.graph)
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .filter_map(|n| self.nodes_map.get(&n).copied())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .collect()
    }

    /// Find closest triangle on nav mesh closest to given point.
    ///
    /// # Arguments
    /// * `point` - query point.
    /// * `query` - query quality.
    ///
    /// # Returns
    /// `Some` with nav mesh triangle index if found or `None` otherwise.
    pub fn find_closest_triangle(&self, point: T, query: NavQuery) -> Option<usize> {
        match query {
            NavQuery::Accuracy => self
                .rtree
                .nearest_neighbor(&SpadePoint(point))
                .map(|t| t.index),
            NavQuery::ClosestFirst => self
                .rtree
                .close_neighbor(&SpadePoint(point))
                .map(|t| t.index),
            NavQuery::Closest => self
                .rtree
                .nearest_neighbors(&SpadePoint(point))
                .into_iter()
                .map(|o| (o.distance2(&SpadePoint(point)), o))
                .fold(None, |a: Option<(Scalar, &NavSpatialObject<T>)>, i| {
                    if let Some(a) = a {
                        if i.0 < a.0 {
                            Some(i)
                        } else {
                            Some(a)
                        }
                    } else {
                        Some(i)
                    }
                })
                .map(|(_, t)| t.index),
        }
    }

    /// Find target point on nav mesh path.
    ///
    /// # Arguments
    /// * `path` - path points.
    /// * `point` - source point.
    /// * `offset` - target point offset from the source on path.
    ///
    /// # Returns
    /// `Some` with point and distance from path start point if found or `None` otherwise.
    pub fn path_target_point(path: &[T], point: T, offset: Scalar) -> Option<(T, Scalar)> {
        let s = Self::project_on_path(path, point, offset);
        Some((Self::point_on_path(path, s)?, s))
    }

    /// Project point on nav mesh path.
    ///
    /// # Arguments
    /// * `path` - path points.
    /// * `point` - source point.
    /// * `offset` - target point offset from the source on path.
    ///
    /// # Returns
    /// Distance from path start point.
    pub fn project_on_path(path: &[T], point: T, offset: Scalar) -> Scalar {
        let p = match path.len() {
            0 | 1 => 0.0,
            2 => Self::project_on_line(path[0], path[1], point),
            _ => {
                path.windows(2)
                    .scan(0.0, |state, pair| {
                        let dist = *state;
                        *state += (pair[1] - pair[0]).magnitude();
                        Some((dist, pair))
                    })
                    .map(|(dist, pair)| {
                        let (p, s) = Self::point_on_line(pair[0], pair[1], point);
                        (dist + s, (p - point).sqr_magnitude())
                    })
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            }
        };
        (p + offset).max(0.0).min(Self::path_length(path))
    }

    /// Find point on nav mesh path at given distance.
    ///
    /// # Arguments
    /// * `path` - path points.
    /// * `s` - Distance from path start point.
    ///
    /// # Returns
    /// `Some` with point on path ot `None` otherwise.
    pub fn point_on_path(path: &[T], mut s: Scalar) -> Option<T> {
        match path.len() {
            0 | 1 => None,
            2 => Some(T::unproject(path[0], path[1], s / Self::path_length(path))),
            _ => {
                for pair in path.windows(2) {
                    let d = (pair[1] - pair[0]).magnitude();
                    if s <= d {
                        return Some(T::unproject(pair[0], pair[1], s / d));
                    }
                    s -= d;
                }
                None
            }
        }
    }

    /// Calculate path length.
    ///
    /// # Arguments
    /// * `path` - path points.
    ///
    /// # Returns
    /// Path length.
    pub fn path_length(path: &[T]) -> Scalar {
        match path.len() {
            0 | 1 => 0.0,
            2 => (path[1] - path[0]).magnitude(),
            _ => path
                .windows(2)
                .fold(0.0, |a, pair| a + (pair[1] - pair[0]).magnitude()),
        }
    }

    fn project_on_line(from: T, to: T, point: T) -> Scalar {
        let d = (to - from).magnitude();
        let p = point.project(from, to);
        d * p
    }

    fn point_on_line(from: T, to: T, point: T) -> (T, Scalar) {
        let d = (to - from).magnitude();
        let p = point.project(from, to);
        if p <= 0.0 {
            (from, 0.0)
        } else if p >= 1.0 {
            (to, d)
        } else {
            (T::unproject(from, to, p), p * d)
        }
    }

    fn raycast_line(from: T, to: T, a: T, b: T, normal: T) -> Option<T> {
        let p = Self::raycast_plane(from, to, a, normal)?;
        let t = p.project(a, b).max(0.0).min(1.0);
        Some(T::unproject(a, b, t))
    }

    fn raycast_plane(from: T, to: T, origin: T, normal: T) -> Option<T> {
        let dir = (to - from).normalize();
        let denom = normal.dot(dir);
        if denom.abs() > ZERO_TRESHOLD {
            let t = (origin - from).dot(normal) / denom;
            if t >= 0.0 && t <= (to - from).magnitude() {
                return Some(from + dir * t);
            }
        }
        None
    }

    fn is_line_between_points(from: T, to: T, a: T, b: T, normal: T) -> bool {
        let n = (to - from).cross(normal);
        let sa = Self::side(n.dot(a - from));
        let sb = Self::side(n.dot(b - from));
        sa != sb
    }

    fn side(v: Scalar) -> i8 {
        if v.abs() < ZERO_TRESHOLD {
            0
        } else {
            v.signum() as i8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(any(feature = "glam", feature = "cgmath", feature = "nalgebra")))]
    type Vec3Type = crate::primitives::NVec3;

    #[cfg(feature = "glam")]
    type Vec3Type = glam::Vec3;

    #[cfg(feature = "cgmath")]
    type Vec3Type = cgmath::Vector3<Scalar>;

    #[cfg(feature = "nalgebra")]
    type Vec3Type = nalgebra::Vector3<Scalar>;

    #[test]
    fn test_raycast() {
        assert_eq!(
            NavMesh::<Vec3Type>::raycast_plane(
                [-1.0, -1.0, -1.0].into(),
                [1.0, 1.0, 1.0].into(),
                [0.0, 0.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
            )
            .unwrap(),
            Vec3Type::new(0.0, 0.0, 0.0),
        );
        assert_eq!(
            NavMesh::<Vec3Type>::raycast_plane(
                [0.0, 0.0, 0.0].into(),
                [2.0, 1.0, 1.0].into(),
                [1.0, 0.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
            )
            .unwrap(),
            Vec3Type::new(1.0, 0.5, 0.5),
        );
        assert_eq!(
            NavMesh::<Vec3Type>::raycast_line(
                [-1.0, -1.0, 1.0].into(),
                [1.0, 1.0, 1.0].into(),
                [1.0, -1.0, 0.0].into(),
                [-1.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
            )
            .unwrap(),
            Vec3Type::new(0.0, 0.0, 0.0),
        );
        assert_eq!(
            NavMesh::<Vec3Type>::raycast_line(
                [0.0, 0.0, 0.0].into(),
                [2.0, 1.0, 1.0].into(),
                [1.0, 0.0, 0.0].into(),
                [1.0, 1.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
            )
            .unwrap(),
            Vec3Type::new(1.0, 0.5, 0.0),
        );
    }

    #[test]
    fn test_line_between_points() {
        assert_eq!(
            true,
            NavMesh::<Vec3Type>::is_line_between_points(
                [0.0, -1.0, 0.0].into(),
                [0.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
            ),
        );
        assert_eq!(
            false,
            NavMesh::<Vec3Type>::is_line_between_points(
                [-2.0, -1.0, 0.0].into(),
                [-2.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
            ),
        );
        assert_eq!(
            false,
            NavMesh::<Vec3Type>::is_line_between_points(
                [2.0, -1.0, 0.0].into(),
                [2.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
            ),
        );
        assert_eq!(
            true,
            NavMesh::<Vec3Type>::is_line_between_points(
                [-1.0, -1.0, 0.0].into(),
                [-1.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
            ),
        );
        assert_eq!(
            true,
            NavMesh::<Vec3Type>::is_line_between_points(
                [1.0, -1.0, 0.0].into(),
                [1.0, 1.0, 0.0].into(),
                [-1.0, 0.0, 0.0].into(),
                [1.0, 0.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
            ),
        );
    }
}
