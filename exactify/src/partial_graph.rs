use ndarray::ArrayD;
use rug::Integer;
use std::collections::HashMap;
use std::ops::Mul;

use crate::tensor::tensor_contract;

/// A partial graph tensor structure that tracks vertices and their degrees,
/// along with a multi-dimensional tensor for homomorphism counting.
///
/// The const generic parameter `D` represents the maximum degree (regularity) of G.
/// The const generic parameter `V` represents the total number of vertices in G.
#[derive(Clone, Debug)]
pub struct PartialGraphTensor<const D: usize, const V: usize> {
    /// Maps vertex id to its current degree in the partial graph.
    /// Only active vertices (degree 1 to D-1) are tracked.
    pub degree: HashMap<usize, usize>,

    /// Maps vertex id to tensor axis index.
    /// If vertex 3 corresponds to axis 1 of the tensor, this contains 3 -> 1.
    pub vertex_to_axis: HashMap<usize, usize>,

    /// Multi-dimensional tensor storing homomorphism counts.
    /// Has one axis per active vertex, each of size c (the graphon size).
    pub tensor: ArrayD<Integer>,
}

impl<const D: usize, const V: usize> PartialGraphTensor<D, V> {
    /// Creates a new PartialGraphTensor from degree map, vertex-to-axis map, and tensor.
    pub fn new(
        degree: HashMap<usize, usize>,
        vertex_to_axis: HashMap<usize, usize>,
        tensor: ArrayD<Integer>,
    ) -> Self {
        debug_assert_eq!(
            degree.len(),
            vertex_to_axis.len(),
            "degree and vertex_to_axis must have the same keys"
        );
        debug_assert_eq!(
            degree.len(),
            tensor.ndim(),
            "tensor rank must equal number of active vertices"
        );
        Self {
            degree,
            vertex_to_axis,
            tensor,
        }
    }

    /// Creates a PartialGraphTensor representing a single edge between vertices u and v.
    /// The tensor is the graphon matrix A.
    pub fn from_edge(u: usize, v: usize, graphon: ArrayD<Integer>) -> Self {
        assert!(u < V && v < V, "Vertex indices must be less than V");
        assert_ne!(u, v, "Self-loops are not supported");
        assert_eq!(graphon.ndim(), 2, "Graphon must be a 2D matrix");
        assert_eq!(
            graphon.shape()[0],
            graphon.shape()[1],
            "Graphon must be square"
        );

        let mut degree = HashMap::new();
        let mut vertex_to_axis = HashMap::new();

        // Use consistent ordering: smaller vertex gets axis 0
        let (first, second) = if u < v { (u, v) } else { (v, u) };

        degree.insert(first, 1);
        degree.insert(second, 1);
        vertex_to_axis.insert(first, 0);
        vertex_to_axis.insert(second, 1);

        // If u > v, we need to transpose the graphon to match axis ordering
        let tensor = if u < v {
            graphon
        } else {
            graphon.t().to_owned().into_dyn()
        };

        Self {
            degree,
            vertex_to_axis,
            tensor,
        }
    }

    /// Returns the vertices currently active in this partial graph.
    pub fn vertices(&self) -> Vec<usize> {
        let mut verts: Vec<usize> = self.degree.keys().copied().collect();
        verts.sort();
        verts
    }

    /// Returns the number of active vertices.
    pub fn vertex_count(&self) -> usize {
        self.degree.len()
    }

    /// Returns the maximum degree allowed for this tensor.
    pub const fn max_degree(&self) -> usize {
        D
    }

    /// Returns the total number of vertices in the full graph G.
    pub const fn total_vertices(&self) -> usize {
        V
    }

    /// Checks if this represents a complete graph (all vertices are dead/saturated).
    /// Returns the scalar homomorphism count if complete.
    pub fn as_scalar(&self) -> Option<Integer> {
        if self.degree.is_empty() && self.tensor.ndim() == 0 {
            Some(self.tensor[[]].clone())
        } else {
            None
        }
    }
}

/// Implements multiplication for PartialGraphTensor.
/// Multiplying two PartialGraphTensors:
/// 1. Unions their edge sets (adds degrees for common vertices)
/// 2. Contracts (marginalizes) over vertices that reach degree D
impl<const D: usize, const V: usize> Mul for PartialGraphTensor<D, V> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        multiply_partial_graphs(&self, &other)
    }
}

/// Implements multiplication by reference for PartialGraphTensor.
impl<const D: usize, const V: usize> Mul for &PartialGraphTensor<D, V> {
    type Output = PartialGraphTensor<D, V>;

    fn mul(self, other: Self) -> PartialGraphTensor<D, V> {
        multiply_partial_graphs(self, other)
    }
}

/// Helper function to multiply two PartialGraphTensors.
fn multiply_partial_graphs<const D: usize, const V: usize>(
    a: &PartialGraphTensor<D, V>,
    b: &PartialGraphTensor<D, V>,
) -> PartialGraphTensor<D, V> {
    // Step 1: Compute merged degrees
    let mut merged_degree: HashMap<usize, usize> = a.degree.clone();
    for (&vertex, &deg) in &b.degree {
        *merged_degree.entry(vertex).or_insert(0) += deg;
    }

    // Step 2: Identify common vertices and classify them
    let mut batch_axes: Vec<(usize, usize)> = Vec::new(); // Common vertices staying active
    let mut contract_axes: Vec<(usize, usize)> = Vec::new(); // Common vertices becoming dead

    for vertex in a.degree.keys() {
        if let Some(&b_axis) = b.vertex_to_axis.get(vertex) {
            let a_axis = a.vertex_to_axis[vertex];
            let new_deg = merged_degree[vertex];

            if new_deg >= D {
                // This vertex becomes dead - contract (marginalize)
                contract_axes.push((a_axis, b_axis));
            } else {
                // This vertex stays active - batch
                batch_axes.push((a_axis, b_axis));
            }
        }
    }

    // Step 3: Perform tensor contraction
    let result_tensor = tensor_contract(&a.tensor, &b.tensor, &batch_axes, &contract_axes);

    // Step 4: Build the new degree and vertex_to_axis maps
    // Result tensor shape: [a_output_dims..., b_output_dims..., batch_dims...]
    // where a_output = axes in a not in batch or contract
    // b_output = axes in b not in batch or contract
    // batch = batch axes

    let a_batch_axes: Vec<usize> = batch_axes.iter().map(|(ai, _)| *ai).collect();
    let a_contract_axes: Vec<usize> = contract_axes.iter().map(|(ai, _)| *ai).collect();
    let b_batch_axes: Vec<usize> = batch_axes.iter().map(|(_, bi)| *bi).collect();
    let b_contract_axes: Vec<usize> = contract_axes.iter().map(|(_, bi)| *bi).collect();

    // Find vertices that are output-only in a (not common)
    let a_output_vertices: Vec<usize> = a
        .vertex_to_axis
        .iter()
        .filter(|(_, axis)| !a_batch_axes.contains(axis) && !a_contract_axes.contains(axis))
        .map(|(&v, _)| v)
        .collect();

    // Find vertices that are output-only in b (not common)
    let b_output_vertices: Vec<usize> = b
        .vertex_to_axis
        .iter()
        .filter(|(_, axis)| !b_batch_axes.contains(axis) && !b_contract_axes.contains(axis))
        .map(|(&v, _)| v)
        .collect();

    // Find vertices that are batched (common and staying active)
    let batch_vertices: Vec<usize> = a
        .vertex_to_axis
        .iter()
        .filter(|(_, axis)| a_batch_axes.contains(axis))
        .map(|(&v, _)| v)
        .collect();

    // Sort for consistent ordering
    let mut a_output_sorted: Vec<(usize, usize)> = a_output_vertices
        .iter()
        .map(|&v| (a.vertex_to_axis[&v], v))
        .collect();
    a_output_sorted.sort_by_key(|&(axis, _)| axis);

    let mut b_output_sorted: Vec<(usize, usize)> = b_output_vertices
        .iter()
        .map(|&v| (b.vertex_to_axis[&v], v))
        .collect();
    b_output_sorted.sort_by_key(|&(axis, _)| axis);

    let mut batch_sorted: Vec<(usize, usize)> = batch_vertices
        .iter()
        .map(|&v| (a.vertex_to_axis[&v], v))
        .collect();
    batch_sorted.sort_by_key(|&(axis, _)| axis);

    // Build new vertex_to_axis mapping
    // Result order: [a_output..., b_output..., batch...]
    let mut new_vertex_to_axis: HashMap<usize, usize> = HashMap::new();
    let mut current_axis = 0;

    for (_, vertex) in &a_output_sorted {
        new_vertex_to_axis.insert(*vertex, current_axis);
        current_axis += 1;
    }

    for (_, vertex) in &b_output_sorted {
        new_vertex_to_axis.insert(*vertex, current_axis);
        current_axis += 1;
    }

    for (_, vertex) in &batch_sorted {
        new_vertex_to_axis.insert(*vertex, current_axis);
        current_axis += 1;
    }

    // Build new degree map (only for active vertices, degree < D)
    let new_degree: HashMap<usize, usize> = merged_degree
        .into_iter()
        .filter(|(_, deg)| *deg < D)
        .collect();

    PartialGraphTensor {
        degree: new_degree,
        vertex_to_axis: new_vertex_to_axis,
        tensor: result_tensor,
    }
}

/// Creates a scalar PartialGraphTensor (no active vertices).
pub fn scalar<const D: usize, const V: usize>(value: Integer) -> PartialGraphTensor<D, V> {
    PartialGraphTensor {
        degree: HashMap::new(),
        vertex_to_axis: HashMap::new(),
        tensor: ArrayD::from_elem(vec![], value),
    }
}

/// Creates an identity PartialGraphTensor for multiplication.
/// This is a scalar with value 1.
pub fn identity<const D: usize, const V: usize>() -> PartialGraphTensor<D, V> {
    scalar(Integer::from(1))
}
