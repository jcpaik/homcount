#![recursion_limit = "1024"]

use ndarray::Zip;
use ndarray::prelude::*;
use rug::Integer;

/// Performs batched matrix multiplication of two 3D tensors.
///
/// Given tensor A of shape (B x N x M) and tensor B of shape (B x M x K),
/// computes the batched matrix product resulting in a tensor of shape (B x N x K).
///
/// For each batch index b:
///   result[b, i, j] = sum over m of A[b, i, m] * B[b, m, j]
fn batched_matmul(
    a: &ArrayView3<Integer>,
    b: &ArrayView3<Integer>,
) -> Array3<Integer> {
    let batch_size = a.shape()[0];
    let n = a.shape()[1];
    let m = a.shape()[2];
    let k = b.shape()[2];

    assert_eq!(b.shape()[0], batch_size, "Batch dimensions must match");
    assert_eq!(b.shape()[1], m, "Inner dimensions must match: A is B x N x M, B must be B x M x K");

    // Prepare output container: B x N x K
    let result = Array3::from_elem((batch_size, n, k), Integer::from(0));

    // Flatten batch and row dimensions for parallel iteration: (B * N) rows total
    let a_flat = a.to_shape((batch_size * n, m)).unwrap();
    let mut result_flat = result.into_shape_with_order((batch_size * n, k)).unwrap();

    Zip::indexed(result_flat.rows_mut())
        .and(a_flat.rows())
        .par_for_each(|idx, mut res_row, a_row| {
            // Determine which batch this row belongs to
            let batch_idx = idx / n;

            // Manual matrix multiplication: compute each element of result row
            for (col_idx, res_elem) in res_row.indexed_iter_mut() {
                let mut sum = Integer::new();

                // Compute dot product: sum of a_row[m] * b[batch_idx, m, col_idx]
                for (m_idx, a_val) in a_row.indexed_iter() {
                    let b_val = &b[[batch_idx, m_idx, col_idx]];
                    sum += a_val * b_val;
                }

                *res_elem = sum;
            }
        });

    // Reshape back to 3D
    result_flat.into_shape_with_order((batch_size, n, k)).unwrap()
}

/// Performs tensor contraction with batched and contracted axes.
///
/// # Arguments
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `batch_axes` - Pairs of axis indices (a_axis, b_axis) that are batched over (multiplied element-wise, not summed)
/// * `contract_axes` - Pairs of axis indices (a_axis, b_axis) that are contracted (summed over, like matrix multiplication)
///
/// # Returns
/// Result tensor with shape: [a_output_dims..., b_output_dims..., batch_dims...]
/// where output dims are the axes not in batch or contract.
fn tensor_contract(
    a: &ArrayD<Integer>,
    b: &ArrayD<Integer>,
    batch_axes: &[(usize, usize)],
    contract_axes: &[(usize, usize)],
) -> ArrayD<Integer> {
    // Extract axis indices for A and B
    let a_batch_axes: Vec<usize> = batch_axes.iter().map(|(ai, _)| *ai).collect();
    let a_contract_axes: Vec<usize> = contract_axes.iter().map(|(ai, _)| *ai).collect();

    let b_batch_axes: Vec<usize> = batch_axes.iter().map(|(_, bi)| *bi).collect();
    let b_contract_axes: Vec<usize> = contract_axes.iter().map(|(_, bi)| *bi).collect();

    // Find output axes (axes not in batch or contract)
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    let a_output_axes: Vec<usize> = (0..a_ndim)
        .filter(|i| !a_batch_axes.contains(i) && !a_contract_axes.contains(i))
        .collect();

    let b_output_axes: Vec<usize> = (0..b_ndim)
        .filter(|i| !b_batch_axes.contains(i) && !b_contract_axes.contains(i))
        .collect();

    // Compute dimension sizes
    let batch_size: usize = a_batch_axes.iter().map(|&i| a.shape()[i]).product::<usize>().max(1);
    let a_output_size: usize = a_output_axes.iter().map(|&i| a.shape()[i]).product::<usize>().max(1);
    let contract_size: usize = a_contract_axes.iter().map(|&i| a.shape()[i]).product::<usize>().max(1);
    let b_output_size: usize = b_output_axes.iter().map(|&i| b.shape()[i]).product::<usize>().max(1);

    // Build permutation for A: [batch_axes, output_axes, contract_axes]
    let a_perm: Vec<usize> = a_batch_axes
        .iter()
        .chain(a_output_axes.iter())
        .chain(a_contract_axes.iter())
        .copied()
        .collect();

    // Build permutation for B: [batch_axes, contract_axes, output_axes]
    let b_perm: Vec<usize> = b_batch_axes
        .iter()
        .chain(b_contract_axes.iter())
        .chain(b_output_axes.iter())
        .copied()
        .collect();

    // Permute and reshape A to 3D: (batch, a_output, contract)
    let a_permuted = a.view().permuted_axes(IxDyn(&a_perm));
    let a_std = a_permuted.as_standard_layout();
    let a_3d = a_std
        .to_shape((batch_size, a_output_size, contract_size))
        .unwrap();

    // Permute and reshape B to 3D: (batch, contract, b_output)
    let b_permuted = b.view().permuted_axes(IxDyn(&b_perm));
    let b_std = b_permuted.as_standard_layout();
    let b_3d = b_std
        .to_shape((batch_size, contract_size, b_output_size))
        .unwrap();

    // Convert to Array3 views for batched_matmul
    let a_view: ArrayView3<Integer> = a_3d.view().into_dimensionality().unwrap();
    let b_view: ArrayView3<Integer> = b_3d.view().into_dimensionality().unwrap();

    // Batched matrix multiplication: (batch, a_output, contract) x (batch, contract, b_output) -> (batch, a_output, b_output)
    let result_3d = batched_matmul(&a_view, &b_view);

    // Build expanded shape: [batch_dims..., a_output_dims..., b_output_dims...]
    let mut expanded_shape: Vec<usize> = Vec::new();
    for &i in &a_batch_axes {
        expanded_shape.push(a.shape()[i]);
    }
    for &i in &a_output_axes {
        expanded_shape.push(a.shape()[i]);
    }
    for &i in &b_output_axes {
        expanded_shape.push(b.shape()[i]);
    }

    // Reshape result to expanded dimensions
    let result_expanded = result_3d
        .into_shape_with_order(IxDyn(&expanded_shape))
        .unwrap();

    // Permute to final order: [a_output_dims, b_output_dims, batch_dims]
    let n_batch = a_batch_axes.len();
    let n_a_out = a_output_axes.len();
    let n_b_out = b_output_axes.len();

    // Build final permutation
    let mut final_perm: Vec<usize> = Vec::new();
    // a_output_dims first
    for i in 0..n_a_out {
        final_perm.push(n_batch + i);
    }
    // b_output_dims next
    for i in 0..n_b_out {
        final_perm.push(n_batch + n_a_out + i);
    }
    // batch_dims last
    for i in 0..n_batch {
        final_perm.push(i);
    }

    result_expanded
        .permuted_axes(IxDyn(&final_perm))
        .into_owned()
}

fn main() {
    // Initialize Tensors with dummy data
    // Tensor A: 3 x 4 x 5 x 6 x 9
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4, 5, 6, 9]), Integer::from(1));

    // Tensor B: 9 x 7 x 8 x 3 x 5
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[9, 7, 8, 3, 5]), Integer::from(2));

    // Operation: result[i4, i6, i7, i8, k] = sum over (i3, i5) of A[i3, i4, i5, i6, k] * B[k, i7, i8, i3, i5]
    //
    // batch_axes: [(4, 0)] - axis 4 of A (size 9) batched with axis 0 of B (size 9)
    // contract_axes: [(0, 3), (2, 4)] - axis 0 of A (size 3) contracted with axis 3 of B (size 3),
    //                                   axis 2 of A (size 5) contracted with axis 4 of B (size 5)
    let batch_axes = vec![(4, 0)];
    let contract_axes = vec![(0, 3), (2, 4)];

    let final_tensor = tensor_contract(&a, &b, &batch_axes, &contract_axes);

    println!("Success!");
    println!("Final Shape: {:?}", final_tensor.shape());

    // Verification: (3*5) elements summed. 1 * 2 = 2 per element.
    // Sum should be 15 * 2 = 30 (same for each position in the 9-dimension)
    println!("First element value: {}", final_tensor[[0, 0, 0, 0, 0]]);
    println!("Element at [0,0,0,0,8] value: {}", final_tensor[[0, 0, 0, 0, 8]]);
}
