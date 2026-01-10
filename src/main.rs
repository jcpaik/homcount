use ndarray::prelude::*;
use ndarray::{Array, Array4};
use rug::Integer;

fn main() {
    // 1. Initialize Tensors with dummy data
    // Tensor A: 3 x 4 x 5 x 6
    let a: Array4<f64> = Array::from_elem((3, 4, 5, 6), 1.0);

    // Tensor B: 7 x 8 x 3 x 5
    let b: Array4<f64> = Array::from_elem((7, 8, 3, 5), 2.0);

    // 2. Prepare Tensor A (Left side of multiplication)
    // We want Output dims (4, 6) first, then Contracting dims (3, 5)
    // Current indices: 0=3, 1=4, 2=5, 3=6.
    // Permutation order: 1, 3, 0, 2
    let a_permuted = a.permuted_axes([1, 3, 0, 2]);

    // Make it contiguous in memory to allow reshaping without copying later if possible,
    // though reshaping usually requires standard layout.
    let a_std = a_permuted.as_standard_layout();

    // Reshape to Matrix: (4*6) x (3*5) -> 24 x 15
    let a_matrix = a_std.into_shape((24, 15)).unwrap();

    // 3. Prepare Tensor B (Right side of multiplication)
    // We want Contracting dims (3, 5) first, then Output dims (7, 8)
    // Current indices: 0=7, 1=8, 2=3, 3=5.
    // Permutation order: 2, 3, 0, 1
    let b_permuted = b.permuted_axes([2, 3, 0, 1]);
    let b_std = b_permuted.as_standard_layout();

    // Reshape to Matrix: (3*5) x (7*8) -> 15 x 56
    let b_matrix = b_std.into_shape((15, 56)).unwrap();

    // 4. Prepare Output Container
    // Result of (24 x 15) * (15 x 56) is (24 x 56)
    let mut result_matrix = Array2::<f64>::zeros((24, 56));

    // 5. Perform Parallel Multiplication
    // We use Zip with par_for_each to iterate over rows of the output and A simultaneously.
    // This splits the work across threads based on the rows of the result.

    // Note: b_matrix must be explicitly captured or shared.
    // Since we are reading b_matrix, we can share the view.

    Zip::from(result_matrix.rows_mut()) // Iterate mutable rows of result
        .and(a_matrix.rows()) // Iterate rows of A
        .par_for_each(|mut res_row, a_row| {
            // Calculate dot product for this row: (1 x 15) * (15 x 56) -> (1 x 56)
            // ndarray's dot method handles the vector-matrix multiplication
            let computed_row = a_row.dot(&b_matrix);

            // Assign result to the output row
            res_row.assign(&computed_row);
        });

    // 6. Reshape to Target Dimensions: 4 x 6 x 7 x 8
    let final_tensor = result_matrix.into_shape((4, 6, 7, 8)).unwrap();

    println!("Success!");
    println!("Final Shape: {:?}", final_tensor.shape());

    // Verification: (3*5) elements summed. 1.0 * 2.0 = 2.0 per element.
    // Sum should be 15 * 2.0 = 30.0
    println!("First element value: {}", final_tensor[[0, 0, 0, 0]]);
}
