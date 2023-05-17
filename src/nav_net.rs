use crate::{
    primitives::{NavVec3, SpadePoint},
    Error, NavConnection, NavResult, Scalar,
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

#[derive(Debug, Clone)]
pub struct NavSpatialConnection<T> {
    pub connection: NavConnection,
    pub index: usize,
    pub a: T,
    pub b: T,
}

impl<T> NavSpatialConnection<T> {
    pub fn new(connection: NavConnection, index: usize, a: T, b: T) -> Self {
        Self {
            connection,
            index,
            a,
            b,
        }
    }

    pub fn closest_point(&self, point: T) -> T
    where
        T: NavVec3,
    {
        let t = point.project(self.a, self.b);
        NavVec3::unproject(self.a, self.b, t)
    }
}

impl<T> SpatialObject for NavSpatialConnection<T>
where
    T: NavVec3,
    SpadePoint<T>: PointN<Scalar = Scalar>,
{
    type Point = SpadePoint<T>;

    fn mbr(&self) -> BoundingRect<Self::Point> {
        let min = T::new(
            self.a.x().min(*self.b.x()),
            self.a.y().min(*self.b.y()),
            self.a.z().min(*self.b.z()),
        );
        let max = T::new(
            self.a.x().max(*self.b.x()),
            self.a.y().max(*self.b.y()),
            self.a.z().max(*self.b.z()),
        );
        BoundingRect::from_corners(&SpadePoint(min), &SpadePoint(max))
    }

    fn distance2(&self, point: &Self::Point) -> Scalar {
        (point.0 - self.closest_point(point.0)).sqr_magnitude()
    }
}

/// Nav net identifier.
pub type NavNetID<T> = ID<NavNet<T>>;

#[derive(Debug, Default, Clone)]
pub struct NavNet<T>
where
    //T: NavVec3,
    SpadePoint<T>: PointN<Scalar = Scalar>,
    NavSpatialConnection<T>: SpatialObject,
{
    id: NavNetID<T>,
    vertices: Vec<T>,
    connections: Vec<NavConnection>,
    distances: Vec<Scalar>,
    costs: Vec<Scalar>,
    graph: Graph<(), Scalar, Undirected>,
    nodes: Vec<NodeIndex>,
    nodes_map: HashMap<NodeIndex, usize>,
    rtree: RTree<NavSpatialConnection<T>>,
    spatials: Vec<NavSpatialConnection<T>>,
    origin: T,
}

impl<T> NavNet<T>
where
    T: NavVec3 + Mul<T, Output = T> + Div<T, Output = T> + Div<Scalar, Output = T>,
    SpadePoint<T>: PointN<Scalar = Scalar>,
{
    pub fn new(vertices: Vec<T>, connections: Vec<NavConnection>) -> NavResult<Self> {
        let origin =
            vertices.iter().cloned().fold(T::zero(), |a, v| a + v) / (vertices.len() as Scalar);

        let distances = iter!(connections)
            .enumerate()
            .map(|(i, c)| {
                if c.0 as usize >= vertices.len() {
                    return Err(Error::ConnectionVerticeIndexOutOfBounds(i as u32, 0, c.0));
                }
                if c.1 as usize >= vertices.len() {
                    return Err(Error::ConnectionVerticeIndexOutOfBounds(i as u32, 1, c.1));
                }
                let a = vertices[c.0 as usize];
                let b = vertices[c.1 as usize];
                Ok((b - a).sqr_magnitude())
            })
            .collect::<NavResult<Vec<_>>>()?;

        let costs = vec![1.0; vertices.len()];

        let mut graph = Graph::<(), Scalar, Undirected>::new_undirected();
        let nodes = (0..vertices.len())
            .map(|_| graph.add_node(()))
            .collect::<Vec<_>>();
        graph.extend_with_edges(
            iter!(connections)
                .enumerate()
                .map(|(i, conn)| (nodes[conn.0 as usize], nodes[conn.1 as usize], distances[i]))
                .collect::<Vec<_>>(),
        );
        let nodes_map = iter!(nodes).enumerate().map(|(i, n)| (*n, i)).collect();

        let spatials = iter!(connections)
            .enumerate()
            .map(|(i, connection)| {
                NavSpatialConnection::new(
                    *connection,
                    i,
                    vertices[connection.0 as usize],
                    vertices[connection.1 as usize],
                )
            })
            .collect::<Vec<_>>();

        let mut rtree = RTree::new();
        for spatial in &spatials {
            rtree.insert(spatial.clone());
        }

        Ok(Self {
            id: ID::default(),
            vertices,
            connections,
            distances,
            costs,
            graph,
            nodes,
            nodes_map,
            rtree,
            spatials,
            origin,
        })
    }

    pub fn scale(&self, value: T, origin: Option<T>) -> NavResult<Self> {
        let origin = origin.unwrap_or(self.origin);
        let vertices = iter!(self.vertices)
            .map(|v| (*v - origin) * value + origin)
            .collect::<Vec<_>>();
        Self::new(vertices, self.connections.clone())
    }

    #[inline]
    pub fn id(&self) -> NavNetID<T> {
        self.id
    }

    #[inline]
    pub fn origin(&self) -> T {
        self.origin
    }

    #[inline]
    pub fn vertices(&self) -> &[T] {
        &self.vertices
    }

    #[inline]
    pub fn connections(&self) -> &[NavConnection] {
        &self.connections
    }

    #[inline]
    pub fn distances(&self) -> &[Scalar] {
        &self.distances
    }

    #[inline]
    pub fn vertices_costs(&self) -> &[Scalar] {
        &self.costs
    }

    #[inline]
    pub fn set_vertice_cost(&mut self, index: usize, cost: Scalar) -> Option<Scalar> {
        let c = self.costs.get_mut(index)?;
        let old = *c;
        *c = cost.max(0.0);
        Some(old)
    }

    pub fn closest_point(&self, point: T) -> Option<T> {
        let index = self.find_closest_connection(point)?;
        Some(self.spatials[index].closest_point(point))
    }

    pub fn find_closest_connection(&self, point: T) -> Option<usize> {
        self.rtree
            .nearest_neighbor(&SpadePoint(point))
            .map(|c| c.index)
    }

    pub fn find_path(&self, from: T, to: T) -> Option<Vec<T>> {
        self.find_path_custom(from, to, |_, _, _| true)
    }

    // filter params: connection distance sqr, first vertex index, second vertex index.
    pub fn find_path_custom<F>(&self, from: T, to: T, mut filter: F) -> Option<Vec<T>>
    where
        F: FnMut(Scalar, usize, usize) -> bool,
    {
        let start_index = self.find_closest_connection(from)?;
        let end_index = self.find_closest_connection(to)?;
        let start_connection = self.connections[start_index];
        let end_connection = self.connections[end_index];
        let start_point = self.spatials[start_index].closest_point(from);
        let end_point = self.spatials[end_index].closest_point(to);
        if start_index == end_index {
            return Some(vec![start_point, end_point]);
        } else if start_point.coincides(end_point) {
            return Some(vec![start_point]);
        }
        let start_vertice = {
            let a = self.vertices[start_connection.0 as usize];
            let b = self.vertices[start_connection.1 as usize];
            if (a - start_point).sqr_magnitude() < (b - start_point).sqr_magnitude() {
                start_connection.0 as usize
            } else {
                start_connection.1 as usize
            }
        };
        let end_vertice = {
            let a = self.vertices[end_connection.0 as usize];
            let b = self.vertices[end_connection.1 as usize];
            if (a - end_point).sqr_magnitude() < (b - end_point).sqr_magnitude() {
                end_connection.0 as usize
            } else {
                end_connection.1 as usize
            }
        };
        let start_node = *self.nodes.get(start_vertice)?;
        let end_node = *self.nodes.get(end_vertice)?;
        let nodes = astar(
            &self.graph,
            start_node,
            |n| n == end_node,
            |e| {
                let a = self.nodes_map[&e.source()];
                let b = self.nodes_map[&e.target()];
                let w = *e.weight();
                if filter(w, a, b) {
                    let a = self.costs[a];
                    let b = self.costs[b];
                    w * a * b
                } else {
                    SCALAR_MAX
                }
            },
            |_| 0.0,
        )?
        .1;
        let mut points = nodes
            .into_iter()
            .map(|n| self.vertices[self.nodes_map[&n]])
            .collect::<Vec<_>>();
        if points.len() > 2 {
            {
                let mut iter = points.iter();
                let a = *iter.next()?;
                let b = *iter.next()?;
                let t = start_point.project(a, b);
                if (0.0..=1.0).contains(&t) {
                    points[0] = start_point;
                } else {
                    points.insert(0, start_point);
                }
            }
            {
                let mut iter = points.iter().rev();
                let a = *iter.next()?;
                let b = *iter.next()?;
                let t = end_point.project(a, b);
                if (0.0..=1.0).contains(&t) {
                    *points.last_mut()? = end_point;
                } else {
                    points.push(end_point);
                }
            }
        }
        Some(points)
    }

    pub fn find_islands(&self) -> Vec<Vec<T>> {
        tarjan_scc(&self.graph)
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .filter_map(|n| self.nodes_map.get(&n).map(|i| self.vertices[*i]))
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .collect()
    }
}
