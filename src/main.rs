#![recursion_limit = "1024"]

use ndarray::Zip;
use ndarray::prelude::*;
use rug::Integer;

fn main() {
    // 1. Initialize Tensors with dummy data using ArrayD (dynamic dimensions)
    // Tensor A: 3 x 4 x 5 x 6 x 9
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4, 5, 6, 9]), Integer::from(1));

    // Tensor B: 9 x 7 x 8 x 3 x 5
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[9, 7, 8, 3, 5]), Integer::from(2));

    // Operation: result[i4, i6, i7, i8, k] = sum over (i3, i5) of A[i3, i4, i5, i6, k] * B[k, i7, i8, i3, i5]
    // The 9-dimension is multiplied along but NOT aggregated (batched matrix multiplication).

    // 2. Prepare Tensor A (Left side of multiplication)
    // Current indices: 0=3, 1=4, 2=5, 3=6, 4=9
    // Want: 9 x 4 x 6 x 3 x 5 (batch, output_A, contract)
    // Permutation: [4, 1, 3, 0, 2]
    let a_permuted = a.permuted_axes(IxDyn(&[4, 1, 3, 0, 2]));
    let a_std = a_permuted.as_standard_layout();
    // Reshape to 3D: 9 x (4*6) x (3*5) = 9 x 24 x 15
    let a_batched = a_std.to_shape((9, 24, 15)).unwrap();

    // 3. Prepare Tensor B (Right side of multiplication)
    // Current indices: 0=9, 1=7, 2=8, 3=3, 4=5
    // Want: 9 x 3 x 5 x 7 x 8 (batch, contract, output_B)
    // Permutation: [0, 3, 4, 1, 2]
    let b_permuted = b.permuted_axes(IxDyn(&[0, 3, 4, 1, 2]));
    let b_std = b_permuted.as_standard_layout();
    // Reshape to 3D: 9 x (3*5) x (7*8) = 9 x 15 x 56
    let b_batched = b_std.to_shape((9, 15, 56)).unwrap();

    // 4. Prepare Output Container
    // Result of batched (9 x 24 x 15) * (9 x 15 x 56) is (9 x 24 x 56)
    let result_batched = Array3::from_elem((9, 24, 56), Integer::from(0));

    // 5. Perform Parallel Batched Matrix Multiplication
    // Zip over (batch, row) pairs by zipping over the 2D array of rows across all batches
    // Flatten batch and row dimensions for parallel iteration: (9*24) rows total
    let a_flat = a_batched.to_shape((9 * 24, 15)).unwrap();

    // Reshape result_batched in place using into_shape_with_order
    let mut result_flat = result_batched.into_shape_with_order((9 * 24, 56)).unwrap();

    Zip::indexed(result_flat.rows_mut())
        .and(a_flat.rows())
        .par_for_each(|idx, mut res_row, a_row| {
            // Determine which batch this row belongs to
            let batch_idx = idx / 24;

            // Manual matrix multiplication: compute each element of result row
            for (col_idx, res_elem) in res_row.indexed_iter_mut() {
                let mut sum = Integer::new();

                // Compute dot product: sum of a_row[k] * b_batched[batch_idx, k, col_idx]
                for (k, a_val) in a_row.indexed_iter() {
                    let b_val = &b_batched[[batch_idx, k, col_idx]];
                    sum += a_val * b_val;
                }

                *res_elem = sum;
            }
        });

    // Reshape back to 3D for further processing
    let result_batched = result_flat.into_shape_with_order((9, 24, 56)).unwrap();

    // 6. Reshape result: 9 x 24 x 56 -> 9 x 4 x 6 x 7 x 8
    let result_5d = result_batched.to_shape((9, 4, 6, 7, 8)).unwrap();

    // 7. Permute to final shape: 4 x 6 x 7 x 8 x 9
    // Current indices: 0=9, 1=4, 2=6, 3=7, 4=8
    // Want: 4 x 6 x 7 x 8 x 9
    // Permutation: [1, 2, 3, 4, 0]
    let final_tensor = result_5d.permuted_axes([1, 2, 3, 4, 0]);

    println!("Success!");
    println!("Final Shape: {:?}", final_tensor.shape());

    // Verification: (3*5) elements summed. 1 * 2 = 2 per element.
    // Sum should be 15 * 2 = 30 (same for each position in the 9-dimension)
    println!("First element value: {}", final_tensor[[0, 0, 0, 0, 0]]);
    println!("Element at [0,0,0,0,8] value: {}", final_tensor[[0, 0, 0, 0, 8]]);
}
