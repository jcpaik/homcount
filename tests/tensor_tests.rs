use homcount::tensor::{batched_matmul, tensor_contract};
use ndarray::prelude::*;
use rug::Integer;

#[test]
fn test_batched_matmul_basic() {
    // Test case: 2 batches, 3x4 @ 4x5 = 3x5
    let a = Array3::from_elem((2, 3, 4), Integer::from(1));
    let b = Array3::from_elem((2, 4, 5), Integer::from(2));

    let result = batched_matmul(&a.view(), &b.view());

    assert_eq!(result.shape(), &[2, 3, 5]);
    // Each element should be 4 * 1 * 2 = 8 (sum of 4 products)
    assert_eq!(result[[0, 0, 0]], Integer::from(8));
    assert_eq!(result[[1, 2, 4]], Integer::from(8));
}

#[test]
fn test_batched_matmul_single_batch() {
    // Test case: 1 batch, 2x3 @ 3x2 = 2x2
    let a = Array3::from_elem((1, 2, 3), Integer::from(3));
    let b = Array3::from_elem((1, 3, 2), Integer::from(4));

    let result = batched_matmul(&a.view(), &b.view());

    assert_eq!(result.shape(), &[1, 2, 2]);
    // Each element should be 3 * 3 * 4 = 36
    assert_eq!(result[[0, 0, 0]], Integer::from(36));
}

#[test]
fn test_batched_matmul_different_values() {
    // Test with specific values to verify correctness
    // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let mut a = Array3::from_elem((1, 2, 2), Integer::from(0));
    a[[0, 0, 0]] = Integer::from(1);
    a[[0, 0, 1]] = Integer::from(2);
    a[[0, 1, 0]] = Integer::from(3);
    a[[0, 1, 1]] = Integer::from(4);

    let mut b = Array3::from_elem((1, 2, 2), Integer::from(0));
    b[[0, 0, 0]] = Integer::from(5);
    b[[0, 0, 1]] = Integer::from(6);
    b[[0, 1, 0]] = Integer::from(7);
    b[[0, 1, 1]] = Integer::from(8);

    let result = batched_matmul(&a.view(), &b.view());

    assert_eq!(result[[0, 0, 0]], Integer::from(19)); // 1*5 + 2*7
    assert_eq!(result[[0, 0, 1]], Integer::from(22)); // 1*6 + 2*8
    assert_eq!(result[[0, 1, 0]], Integer::from(43)); // 3*5 + 4*7
    assert_eq!(result[[0, 1, 1]], Integer::from(50)); // 3*6 + 4*8
}

#[test]
fn test_tensor_contract_basic() {
    // Test the original use case:
    // A: 3 x 4 x 5 x 6 x 9
    // B: 9 x 7 x 8 x 3 x 5
    // batch_axes: [(4, 0)] - size 9
    // contract_axes: [(0, 3), (2, 4)] - sizes 3 and 5
    // Result: 4 x 6 x 7 x 8 x 9
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4, 5, 6, 9]), Integer::from(1));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[9, 7, 8, 3, 5]), Integer::from(2));

    let batch_axes = vec![(4, 0)];
    let contract_axes = vec![(0, 3), (2, 4)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    assert_eq!(result.shape(), &[4, 6, 7, 8, 9]);
    // Each element: sum over 3*5=15 elements of 1*2 = 30
    assert_eq!(result[[0, 0, 0, 0, 0]], Integer::from(30));
    assert_eq!(result[[3, 5, 6, 7, 8]], Integer::from(30));
}

#[test]
fn test_tensor_contract_no_batch() {
    // Pure contraction without batch dimensions (like standard matrix multiplication)
    // A: 3 x 4
    // B: 4 x 5
    // contract_axes: [(1, 0)] - size 4
    // Result: 3 x 5
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4]), Integer::from(2));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[4, 5]), Integer::from(3));

    let batch_axes: Vec<(usize, usize)> = vec![];
    let contract_axes = vec![(1, 0)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    assert_eq!(result.shape(), &[3, 5]);
    // Each element: sum over 4 elements of 2*3 = 24
    assert_eq!(result[[0, 0]], Integer::from(24));
    assert_eq!(result[[2, 4]], Integer::from(24));
}

#[test]
fn test_tensor_contract_multiple_batch() {
    // Multiple batch dimensions
    // A: 2 x 3 x 4 x 5
    // B: 2 x 3 x 5 x 6
    // batch_axes: [(0, 0), (1, 1)] - sizes 2 and 3
    // contract_axes: [(3, 2)] - size 5
    // Result: 4 x 6 x 2 x 3
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 3, 4, 5]), Integer::from(1));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 3, 5, 6]), Integer::from(2));

    let batch_axes = vec![(0, 0), (1, 1)];
    let contract_axes = vec![(3, 2)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    assert_eq!(result.shape(), &[4, 6, 2, 3]);
    // Each element: sum over 5 elements of 1*2 = 10
    assert_eq!(result[[0, 0, 0, 0]], Integer::from(10));
    assert_eq!(result[[3, 5, 1, 2]], Integer::from(10));
}

#[test]
fn test_tensor_contract_multiple_contract() {
    // Multiple contraction dimensions
    // A: 2 x 3 x 4
    // B: 3 x 4 x 5
    // batch_axes: []
    // contract_axes: [(1, 0), (2, 1)] - sizes 3 and 4
    // Result: 2 x 5
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 3, 4]), Integer::from(1));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4, 5]), Integer::from(2));

    let batch_axes: Vec<(usize, usize)> = vec![];
    let contract_axes = vec![(1, 0), (2, 1)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    assert_eq!(result.shape(), &[2, 5]);
    // Each element: sum over 3*4=12 elements of 1*2 = 24
    assert_eq!(result[[0, 0]], Integer::from(24));
    assert_eq!(result[[1, 4]], Integer::from(24));
}

#[test]
fn test_tensor_contract_scalar_result() {
    // Contract everything to get a scalar (wrapped in 1x1x... tensor)
    // A: 2 x 3
    // B: 2 x 3
    // batch_axes: []
    // contract_axes: [(0, 0), (1, 1)]
    // Result: scalar (empty shape handled as 1-element array)
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 3]), Integer::from(2));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 3]), Integer::from(3));

    let batch_axes: Vec<(usize, usize)> = vec![];
    let contract_axes = vec![(0, 0), (1, 1)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    // Result should be a scalar (0-dimensional or 1-element)
    assert_eq!(result.len(), 1);
    // Sum over 2*3=6 elements of 2*3 = 36
    assert_eq!(result.iter().next().unwrap(), &Integer::from(36));
}

#[test]
fn test_tensor_contract_large_integers() {
    // Test with values that would overflow standard integer types
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[2, 100]), Integer::from(i64::MAX));
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[100, 2]), Integer::from(i64::MAX));

    let batch_axes: Vec<(usize, usize)> = vec![];
    let contract_axes = vec![(1, 0)];

    let result = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    assert_eq!(result.shape(), &[2, 2]);
    // Each element: 100 * (i64::MAX)^2
    let expected = Integer::from(i64::MAX) * Integer::from(i64::MAX) * Integer::from(100);
    assert_eq!(result[[0, 0]], expected);
}
