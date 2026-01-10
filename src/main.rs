#![recursion_limit = "1024"]

use ndarray::Zip;
use ndarray::prelude::*;
use rug::Integer;

fn main() {
    // 1. Initialize Tensors with dummy data using ArrayD (dynamic dimensions)
    // Tensor A: 3 x 4 x 5 x 6
    let a: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[3, 4, 5, 6]), Integer::from(1));

    // Tensor B: 7 x 8 x 3 x 5
    let b: ArrayD<Integer> = ArrayD::from_elem(IxDyn(&[7, 8, 3, 5]), Integer::from(2));

    // 2. Prepare Tensor A (Left side of multiplication)
    // We want Output dims (4, 6) first, then Contracting dims (3, 5)
    // Current indices: 0=3, 1=4, 2=5, 3=6.
    // Permutation order: 1, 3, 0, 2
    let a_permuted = a.permuted_axes(IxDyn(&[1, 3, 0, 2]));

    // Make it contiguous in memory to allow reshaping without copying later if possible,
    // though reshaping usually requires standard layout.
    let a_std = a_permuted.as_standard_layout();

    // Reshape to Matrix: (4*6) x (3*5) -> 24 x 15
    let a_matrix = a_std.to_shape((24, 15)).unwrap();

    // 3. Prepare Tensor B (Right side of multiplication)
    // We want Contracting dims (3, 5) first, then Output dims (7, 8)
    // Current indices: 0=7, 1=8, 2=3, 3=5.
    // Permutation order: 2, 3, 0, 1
    let b_permuted = b.permuted_axes(IxDyn(&[2, 3, 0, 1]));
    let b_std = b_permuted.as_standard_layout();

    // Reshape to Matrix: (3*5) x (7*8) -> 15 x 56
    let b_matrix = b_std.to_shape((15, 56)).unwrap();

    // 4. Prepare Output Container
    // Result of (24 x 15) * (15 x 56) is (24 x 56)
    let mut result_matrix = Array2::from_elem((24, 56), Integer::from(0));

    // 5. Perform Parallel Multiplication (Manual implementation)
    // Instead of using dot product, implement matrix multiplication manually
    Zip::from(result_matrix.rows_mut())
        .and(a_matrix.rows())
        .par_for_each(|mut res_row, a_row| {
            // Manual matrix multiplication: compute each element of result row
            for (col_idx, res_elem) in res_row.indexed_iter_mut() {
                let mut sum = Integer::new(); // Start with zero

                // Compute dot product manually: sum of a_row[k] * b_matrix[k][col_idx]
                for (k, a_val) in a_row.indexed_iter() {
                    let b_val = &b_matrix[[k, col_idx]];
                    sum += a_val * b_val;
                }

                *res_elem = sum;
            }
        });

    // 6. Reshape to Target Dimensions: 4 x 6 x 7 x 8
    let final_tensor = result_matrix.to_shape(IxDyn(&[4, 6, 7, 8])).unwrap();

    println!("Success!");
    println!("Final Shape: {:?}", final_tensor.shape());

    // Verification: (3*5) elements summed. 1 * 2 = 2 per element.
    // Sum should be 15 * 2 = 30
    println!("First element value: {}", final_tensor[[0, 0, 0, 0]]);
}
