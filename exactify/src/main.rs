#![recursion_limit = "1024"]

mod tensor;

use ndarray::prelude::*;
use rug::Integer;
use tensor::tensor_contract;

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
