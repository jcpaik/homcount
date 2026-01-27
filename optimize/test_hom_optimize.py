"""
Tests for the homomorphism optimization module.

Run with: python test_hom_optimize.py
"""

import sys
import traceback

import numpy as np
import torch
from hom_optimize import (
    DESARGUES_EDGES,
    DESARGUES_EINSUM,
    HEAWOOD_EDGES,
    HEAWOOD_EINSUM,
    build_einsum_string,
    hom_desargues,
    hom_heawood,
    optimize_graphon,
)


def test_heawood_graph_properties():
    """Heawood graph: 14 vertices, 21 edges, 3-regular."""
    assert len(HEAWOOD_EDGES) == 21

    # Check all vertices are in range 0-13
    vertices = set()
    for u, v in HEAWOOD_EDGES:
        vertices.add(u)
        vertices.add(v)
    assert vertices == set(range(14))

    # Check 3-regular
    degree = {}
    for u, v in HEAWOOD_EDGES:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    for v in range(14):
        assert degree[v] == 3, f"Vertex {v} has degree {degree[v]}, expected 3"


def test_desargues_graph_properties():
    """Desargues graph: 20 vertices, 30 edges, 3-regular."""
    assert len(DESARGUES_EDGES) == 30

    # Check all vertices are in range 0-19
    vertices = set()
    for u, v in DESARGUES_EDGES:
        vertices.add(u)
        vertices.add(v)
    assert vertices == set(range(20))

    # Check 3-regular
    degree = {}
    for u, v in DESARGUES_EDGES:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    for v in range(20):
        assert degree[v] == 3, f"Vertex {v} has degree {degree[v]}, expected 3"


def test_build_einsum_string_triangle():
    """Test einsum string for triangle graph."""
    # Triangle: vertices 0, 1, 2 with edges (0,1), (1,2), (0,2)
    triangle_edges = [(0, 1), (1, 2), (0, 2)]
    einsum_str = build_einsum_string(triangle_edges, 3)
    assert einsum_str == "ab,bc,ac->"


def test_build_einsum_string_square():
    """Test einsum string for square (4-cycle) graph."""
    # Square: vertices 0, 1, 2, 3 with edges forming a cycle
    square_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    einsum_str = build_einsum_string(square_edges, 4)
    assert einsum_str == "ab,bc,cd,da->"


def test_heawood_einsum_format():
    """Test that Heawood einsum string has correct format."""
    # Should have 21 subscript pairs separated by commas, ending with ->
    parts = HEAWOOD_EINSUM.split("->")
    assert len(parts) == 2
    assert parts[1] == ""  # Output is scalar
    subscripts = parts[0].split(",")
    assert len(subscripts) == 21  # 21 edges


def test_desargues_einsum_format():
    """Test that Desargues einsum string has correct format."""
    # Should have 30 subscript pairs separated by commas, ending with ->
    parts = DESARGUES_EINSUM.split("->")
    assert len(parts) == 2
    assert parts[1] == ""  # Output is scalar
    subscripts = parts[0].split(",")
    assert len(subscripts) == 30  # 30 edges


def test_triangle_count_via_einsum():
    """Test that einsum triangle count matches trace(A^3)."""
    G = torch.tensor(
        [
            [2.0, 3.0, 1.0, 4.0, 2.0],
            [3.0, 1.0, 5.0, 2.0, 3.0],
            [1.0, 5.0, 3.0, 1.0, 4.0],
            [4.0, 2.0, 1.0, 2.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 1.0],
        ],
        dtype=torch.float64,
    )

    # Compute via trace(A^3)
    trace_result = torch.trace(torch.matmul(torch.matmul(G, G), G))

    # Compute via einsum: sum_{i,j,k} G[i,j] * G[j,k] * G[k,i]
    einsum_result = torch.einsum("ab,bc,ca->", G, G, G)

    assert torch.isclose(trace_result, einsum_result)


def test_hom_heawood_returns_scalar():
    """Test that hom_heawood returns a scalar."""
    G = torch.abs(torch.randn(5, 5, dtype=torch.float64))
    G = (G + G.t()) / 2

    result = hom_heawood(G)

    assert result.dim() == 0  # Scalar


def test_hom_desargues_returns_scalar():
    """Test that hom_desargues returns a scalar."""
    G = torch.abs(torch.randn(5, 5, dtype=torch.float64))
    G = (G + G.t()) / 2

    result = hom_desargues(G)

    assert result.dim() == 0  # Scalar


def test_hom_heawood_nonnegative_input():
    """Test hom_heawood with nonnegative symmetric matrix."""
    G = torch.abs(torch.randn(4, 4, dtype=torch.float64))
    G = (G + G.t()) / 2

    result = hom_heawood(G)

    # With nonnegative input, result should be nonnegative
    assert result.item() >= 0


def test_hom_desargues_nonnegative_input():
    """Test hom_desargues with nonnegative symmetric matrix."""
    G = torch.abs(torch.randn(4, 4, dtype=torch.float64))
    G = (G + G.t()) / 2

    result = hom_desargues(G)

    # With nonnegative input, result should be nonnegative
    assert result.item() >= 0


def test_hom_heawood_gradient_exists():
    """Test that gradients flow through hom_heawood."""
    G = torch.abs(torch.randn(4, 4, dtype=torch.float64))
    G.requires_grad = True
    G_sym = (G + G.t()) / 2

    result = hom_heawood(G_sym)
    result.backward()

    assert G.grad is not None
    assert G.grad.shape == (4, 4)


def test_hom_desargues_gradient_exists():
    """Test that gradients flow through hom_desargues."""
    G = torch.abs(torch.randn(4, 4, dtype=torch.float64))
    G.requires_grad = True
    G_sym = (G + G.t()) / 2

    result = hom_desargues(G_sym)
    result.backward()

    assert G.grad is not None
    assert G.grad.shape == (4, 4)


def test_hom_identity_matrix():
    """Test homomorphism counts on identity matrix."""
    # For identity matrix, hom(H, I) counts the number of proper vertex colorings
    # where adjacent vertices get different colors (since I[i,j] = 0 for i != j)
    # For Heawood (bipartite), this should be positive
    G = torch.eye(3, dtype=torch.float64)

    x = hom_heawood(G)
    y = hom_desargues(G)

    # Both should be computable (not NaN or Inf)
    assert not torch.isnan(x)
    assert not torch.isnan(y)
    assert not torch.isinf(x)
    assert not torch.isinf(y)


def test_hom_ones_matrix():
    """Test homomorphism counts on all-ones matrix."""
    c = 3
    G = torch.ones(c, c, dtype=torch.float64)

    x = hom_heawood(G)
    y = hom_desargues(G)

    # For all-ones matrix, hom(H, J) = c^|V(H)|
    # Heawood has 14 vertices, Desargues has 20 vertices
    expected_heawood = c**14
    expected_desargues = c**20

    assert torch.isclose(x, torch.tensor(float(expected_heawood), dtype=torch.float64))
    assert torch.isclose(
        y, torch.tensor(float(expected_desargues), dtype=torch.float64)
    )


def test_optimization_runs():
    """Test that optimization runs without error."""
    np.random.seed(42)
    G_init = np.abs(np.random.randn(4, 4))
    G_init = (G_init + G_init.T) / 2

    # Run just a few iterations
    optimized_G, loss_history = optimize_graphon(
        G_init, num_iterations=5, learning_rate=0.01, verbose=False
    )

    assert len(loss_history) == 5
    assert optimized_G.shape == (4, 4)
    # Check symmetry
    assert np.allclose(optimized_G, optimized_G.T)
    # Check nonnegativity
    assert np.all(optimized_G >= 0)


def test_optimization_changes_loss():
    """Test that optimization actually changes the loss."""
    np.random.seed(123)
    G_init = np.abs(np.random.randn(4, 4))
    G_init = (G_init + G_init.T) / 2

    _, loss_history = optimize_graphon(
        G_init, num_iterations=10, learning_rate=0.01, verbose=False
    )

    # Loss should change (optimizer is working)
    assert not all(loss_history[i] == loss_history[0] for i in range(len(loss_history)))


def run_all_tests():
    """Run all tests and report results."""
    test_functions = [
        test_heawood_graph_properties,
        test_desargues_graph_properties,
        test_build_einsum_string_triangle,
        test_build_einsum_string_square,
        test_heawood_einsum_format,
        test_desargues_einsum_format,
        test_triangle_count_via_einsum,
        test_hom_heawood_returns_scalar,
        test_hom_desargues_returns_scalar,
        test_hom_heawood_nonnegative_input,
        test_hom_desargues_nonnegative_input,
        test_hom_heawood_gradient_exists,
        test_hom_desargues_gradient_exists,
        test_hom_identity_matrix,
        test_hom_ones_matrix,
        test_optimization_runs,
        test_optimization_changes_loss,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_func in test_functions:
        try:
            test_func()
            print(f"  PASS: {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_func.__name__}")
            errors.append((test_func.__name__, str(e), traceback.format_exc()))
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_func.__name__}")
            errors.append((test_func.__name__, str(e), traceback.format_exc()))
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print(f"\nFailures and Errors:")
        for name, msg, tb in errors:
            print(f"\n{name}:")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    print("Running homomorphism optimization tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
