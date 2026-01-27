use homcount::partial_graph::PartialGraphTensor;
use ndarray::{ArrayD, IxDyn};
use rug::Integer;

/// Helper to create a graphon matrix from a 2D vector of i64
fn graphon_from_vec(data: Vec<Vec<i64>>) -> ArrayD<Integer> {
    let c = data.len();
    let flat: Vec<Integer> = data.into_iter().flatten().map(Integer::from).collect();
    ArrayD::from_shape_vec(IxDyn(&[c, c]), flat).unwrap()
}

#[test]
fn test_from_edge_creates_correct_structure() {
    let graphon = graphon_from_vec(vec![vec![1, 2, 3], vec![2, 4, 5], vec![3, 5, 6]]);

    let edge: PartialGraphTensor<3, 10> = PartialGraphTensor::from_edge(0, 1, graphon);

    // Check that both vertices are tracked with degree 1
    assert_eq!(edge.degree.len(), 2);
    assert_eq!(edge.degree[&0], 1);
    assert_eq!(edge.degree[&1], 1);

    // Check vertex to axis mapping
    assert_eq!(edge.vertex_to_axis.len(), 2);
    assert_eq!(edge.vertex_to_axis[&0], 0);
    assert_eq!(edge.vertex_to_axis[&1], 1);

    // Tensor should be 3x3
    assert_eq!(edge.tensor.shape(), &[3, 3]);
}

#[test]
fn test_vertices_method() {
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);

    let edge: PartialGraphTensor<2, 5> = PartialGraphTensor::from_edge(3, 1, graphon);

    let verts = edge.vertices();
    assert_eq!(verts, vec![1, 3]); // Sorted order
}

#[test]
fn test_multiply_disjoint_edges() {
    // Two edges with no common vertices
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);

    let edge01: PartialGraphTensor<2, 4> = PartialGraphTensor::from_edge(0, 1, graphon.clone());
    let edge23: PartialGraphTensor<2, 4> = PartialGraphTensor::from_edge(2, 3, graphon);

    let result = &edge01 * &edge23;

    // Should have 4 active vertices
    assert_eq!(result.vertex_count(), 4);
    assert_eq!(result.degree[&0], 1);
    assert_eq!(result.degree[&1], 1);
    assert_eq!(result.degree[&2], 1);
    assert_eq!(result.degree[&3], 1);

    // Tensor should be 2x2x2x2
    assert_eq!(result.tensor.ndim(), 4);
}

#[test]
fn test_multiply_with_common_vertex_stays_active() {
    // Two edges sharing vertex 1, but vertex 1 has max degree 3
    // So it stays active after multiplication
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);

    let edge01: PartialGraphTensor<3, 3> = PartialGraphTensor::from_edge(0, 1, graphon.clone());
    let edge12: PartialGraphTensor<3, 3> = PartialGraphTensor::from_edge(1, 2, graphon);

    let result = &edge01 * &edge12;

    // Vertex 1 has degree 2, still < 3, so stays active
    assert_eq!(result.vertex_count(), 3);
    assert_eq!(result.degree[&0], 1);
    assert_eq!(result.degree[&1], 2);
    assert_eq!(result.degree[&2], 1);

    // Tensor should be 3D (one axis per active vertex)
    assert_eq!(result.tensor.ndim(), 3);
}

#[test]
fn test_multiply_with_vertex_becoming_dead() {
    // In a 2-regular graph, when a vertex reaches degree 2 it becomes dead
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);

    let edge01: PartialGraphTensor<2, 3> = PartialGraphTensor::from_edge(0, 1, graphon.clone());
    let edge12: PartialGraphTensor<2, 3> = PartialGraphTensor::from_edge(1, 2, graphon);

    let result = &edge01 * &edge12;

    // Vertex 1 reaches degree 2, becomes dead and is marginalized
    assert_eq!(result.vertex_count(), 2);
    assert!(result.degree.contains_key(&0));
    assert!(result.degree.contains_key(&2));
    assert!(!result.degree.contains_key(&1)); // Vertex 1 is dead

    // Tensor should be 2D (vertices 0 and 2)
    assert_eq!(result.tensor.ndim(), 2);
}

/// Count triangles using matrix trace method: tr(A^3) / 6
/// (dividing by 6 because each triangle is counted 6 times: 3 starting vertices × 2 directions)
fn count_triangles_via_trace(graphon: &ArrayD<Integer>) -> Integer {
    let c = graphon.shape()[0];

    // Compute A^2
    let mut a2 = ArrayD::from_elem(IxDyn(&[c, c]), Integer::from(0));
    for i in 0..c {
        for j in 0..c {
            let mut sum = Integer::new();
            for k in 0..c {
                sum += &graphon[[i, k]] * &graphon[[k, j]];
            }
            a2[[i, j]] = sum;
        }
    }

    // Compute A^3
    let mut a3 = ArrayD::from_elem(IxDyn(&[c, c]), Integer::from(0));
    for i in 0..c {
        for j in 0..c {
            let mut sum = Integer::new();
            for k in 0..c {
                sum += &a2[[i, k]] * &graphon[[k, j]];
            }
            a3[[i, j]] = sum;
        }
    }

    // Compute trace
    let mut trace = Integer::new();
    for i in 0..c {
        trace += &a3[[i, i]];
    }

    trace
}

/// Count triangles using PartialGraphTensor multiplication
/// Triangle has vertices 0, 1, 2 with edges (0,1), (1,2), (0,2)
/// In a 2-regular graph, all three vertices become dead after including all edges
fn count_triangles_via_partial_graph(graphon: ArrayD<Integer>) -> Integer {
    // Create edges of a triangle
    let edge01: PartialGraphTensor<2, 3> = PartialGraphTensor::from_edge(0, 1, graphon.clone());
    let edge12: PartialGraphTensor<2, 3> = PartialGraphTensor::from_edge(1, 2, graphon.clone());
    let edge02: PartialGraphTensor<2, 3> = PartialGraphTensor::from_edge(0, 2, graphon);

    // Multiply all three edges
    let result = &(&edge01 * &edge12) * &edge02;

    // All vertices should be dead (degree 2 each)
    assert_eq!(result.vertex_count(), 0, "All vertices should be saturated");
    assert_eq!(result.tensor.ndim(), 0, "Result should be a scalar");

    result.as_scalar().expect("Result should be a scalar")
}

#[test]
fn test_triangle_count_small_graphon() {
    // Simple 2x2 graphon
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 3]]);

    let trace_result = count_triangles_via_trace(&graphon);
    let partial_graph_result = count_triangles_via_partial_graph(graphon);

    assert_eq!(
        trace_result, partial_graph_result,
        "Triangle counts must match: trace={}, partial_graph={}",
        trace_result, partial_graph_result
    );
}

#[test]
fn test_triangle_count_5x5_graphon() {
    // 5x5 symmetric "random" graphon
    let graphon = graphon_from_vec(vec![
        vec![2, 3, 1, 4, 2],
        vec![3, 1, 5, 2, 3],
        vec![1, 5, 3, 1, 4],
        vec![4, 2, 1, 2, 5],
        vec![2, 3, 4, 5, 1],
    ]);

    let trace_result = count_triangles_via_trace(&graphon);
    let partial_graph_result = count_triangles_via_partial_graph(graphon);

    println!("5x5 graphon triangle count via trace: {}", trace_result);
    println!(
        "5x5 graphon triangle count via partial graph: {}",
        partial_graph_result
    );

    assert_eq!(
        trace_result, partial_graph_result,
        "Triangle counts must match: trace={}, partial_graph={}",
        trace_result, partial_graph_result
    );
}

#[test]
fn test_triangle_count_identity_graphon() {
    // Identity matrix - should give trace = c (only diagonal contributes)
    let graphon = graphon_from_vec(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);

    let trace_result = count_triangles_via_trace(&graphon);
    let partial_graph_result = count_triangles_via_partial_graph(graphon);

    // tr(I^3) = tr(I) = 3
    assert_eq!(trace_result, Integer::from(3));
    assert_eq!(partial_graph_result, Integer::from(3));
}

#[test]
fn test_scalar_creation() {
    let s: PartialGraphTensor<2, 3> = homcount::partial_graph::scalar(Integer::from(42));
    assert_eq!(s.vertex_count(), 0);
    assert_eq!(s.as_scalar(), Some(Integer::from(42)));
}

#[test]
fn test_as_scalar_returns_none_for_non_scalar() {
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);
    let edge: PartialGraphTensor<3, 2> = PartialGraphTensor::from_edge(0, 1, graphon);
    assert!(edge.as_scalar().is_none());
}

#[test]
fn test_max_degree_and_total_vertices() {
    let graphon = graphon_from_vec(vec![vec![1, 2], vec![2, 1]]);
    let edge: PartialGraphTensor<5, 100> = PartialGraphTensor::from_edge(0, 1, graphon);

    assert_eq!(edge.max_degree(), 5);
    assert_eq!(edge.total_vertices(), 100);
}
