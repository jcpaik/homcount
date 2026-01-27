"""
Homomorphism counting and optimization using PyTorch.

This module computes homomorphism counts from the Heawood and Desargues graphs
to a target graphon (represented as a matrix), and optimizes the graphon to
minimize x^10 - y^7 where x = hom(Heawood, G) and y = hom(Desargues, G).

Uses opt_einsum for efficient homomorphism computation with optimized contraction paths.
Supports GPU acceleration via CUDA.
"""

from typing import List, Tuple

import numpy as np
import opt_einsum as oe
import torch
import torch.optim as optim


def build_einsum_string(edges: List[Tuple[int, int]], num_vertices: int) -> str:
    """
    Build an einsum string for computing graph homomorphisms.

    For a graph with vertices 0..n-1 and edges E, the einsum computes:
    sum_{v0,v1,...,v_{n-1}} prod_{(u,v) in E} G[v_u, v_v]

    Each vertex is assigned a letter (0='a', 1='b', etc.)
    Each edge (u,v) contributes a subscript pair to the einsum.
    """

    def vertex_to_letter(v: int) -> str:
        return chr(ord("a") + v)

    subscripts = []
    for u, v in edges:
        subscripts.append(vertex_to_letter(u) + vertex_to_letter(v))

    return ",".join(subscripts) + "->"


# Heawood graph: 14 vertices (a-n), 21 edges, 3-regular, girth 6
HEAWOOD_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (0, 5),
    (0, 13),
    (1, 2),
    (1, 10),
    (2, 3),
    (2, 7),
    (3, 4),
    (3, 12),
    (4, 5),
    (4, 9),
    (5, 6),
    (6, 7),
    (6, 11),
    (7, 8),
    (8, 9),
    (8, 13),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
]
HEAWOOD_EINSUM = build_einsum_string(HEAWOOD_EDGES, 14)
# "ab,af,an,bc,bk,cd,ch,de,dm,ef,ej,fg,gh,gl,hi,ij,in,jk,kl,lm,mn->"

# Desargues graph: 20 vertices (a-t), 30 edges, 3-regular, girth 5
DESARGUES_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (0, 5),
    (0, 19),
    (1, 2),
    (1, 16),
    (2, 3),
    (2, 11),
    (3, 4),
    (3, 8),
    (4, 5),
    (4, 15),
    (5, 6),
    (6, 7),
    (6, 12),
    (7, 8),
    (7, 19),
    (8, 9),
    (9, 10),
    (9, 14),
    (10, 11),
    (10, 17),
    (11, 12),
    (12, 13),
    (13, 14),
    (13, 18),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]
DESARGUES_EINSUM = build_einsum_string(DESARGUES_EDGES, 20)
# "ab,af,at,bc,bq,cd,cl,de,di,ef,ep,fg,gh,gm,hi,ht,ij,jk,jo,kl,kr,lm,mn,no,ns,op,pq,qr,rs,st->"


def hom_heawood(G: torch.Tensor) -> torch.Tensor:
    """Computes hom(Heawood, G) using opt_einsum."""
    # Create a list of 21 copies of G for the 21 edges
    tensors = [G] * len(HEAWOOD_EDGES)
    return oe.contract(HEAWOOD_EINSUM, *tensors, backend="torch")


def hom_desargues(G: torch.Tensor) -> torch.Tensor:
    """Computes hom(Desargues, G) using opt_einsum."""
    # Create a list of 30 copies of G for the 30 edges
    tensors = [G] * len(DESARGUES_EDGES)
    return oe.contract(DESARGUES_EINSUM, *tensors, backend="torch")


def downscale_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downscale a matrix by averaging blocks.

    Args:
        matrix: Square matrix of size (n, n) where n is divisible by target_size.
        target_size: Target size for the downscaled matrix.

    Returns:
        Downscaled matrix of size (target_size, target_size).
    """
    n = matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    assert n % target_size == 0, (
        f"Matrix size {n} must be divisible by target_size {target_size}"
    )

    block_size = n // target_size
    downscaled = np.zeros((target_size, target_size))

    for i in range(target_size):
        for j in range(target_size):
            block = matrix[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            downscaled[i, j] = np.mean(block)

    return downscaled


def optimize_graphon(
    initial_G: np.ndarray,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    verbose: bool = True,
    print_every: int = 100,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[float]]:
    """
    Optimizes the graphon G to minimize x^10 - y^7 where
    x = hom(Heawood, G) and y = hom(Desargues, G).

    G is parameterized as P^2 (element-wise) to enforce nonnegativity.

    Args:
        initial_G: Initial c x c numpy matrix of doubles (nonnegative).
        num_iterations: Number of optimization iterations.
        learning_rate: Learning rate for optimizer.
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop').
        momentum: Momentum for SGD/RMSprop.
        weight_decay: Weight decay (L2 regularization).
        verbose: Whether to print progress.
        print_every: Print progress every N iterations.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Tuple of (optimized_G, loss_history).
    """
    assert initial_G.ndim == 2 and initial_G.shape[0] == initial_G.shape[1], (
        f"Expected square matrix, got {initial_G.shape}"
    )

    # Parameterize G = P^2 (element-wise) to enforce nonnegativity
    # Initialize P as sqrt of absolute values of initial_G
    P_init = np.sqrt(np.abs(initial_G))
    P = torch.tensor(P_init, dtype=torch.float64, device=device, requires_grad=True)

    # Create optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam([P], lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            [P], lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(
            [P], lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    loss_history = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Compute G = P^2 (element-wise) to enforce nonnegativity
        G = P * P
        # Symmetrize G
        G_sym = (G + G.t()) / 2

        # Compute homomorphism counts
        x = hom_heawood(G_sym)
        y = hom_desargues(G_sym)

        # Compute loss: x^(10/7) - y
        loss = x.pow(10 / 7) - y

        # Backward pass
        loss.backward()

        # Update P
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if verbose and (
            iteration % print_every == 0 or iteration == num_iterations - 1
        ):
            print(
                f"Iteration {iteration}: loss = {loss_value:.6e}, "
                f"hom_heawood = {x.item():.6e}, hom_desargues = {y.item():.6e}"
            )

    # Return the optimized graphon as numpy array (G = P^2, symmetrized)
    P_final = P.detach().cpu().numpy()
    G_final = P_final * P_final
    G_final = (G_final + G_final.T) / 2

    return G_final, loss_history


def main():
    """Example usage of the optimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize graphon for homomorphism counts"
    )
    parser.add_argument(
        "--size", type=int, default=4, help="Target size of the graphon matrix (c x c)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input numpy matrix file (.npy). Must be square with size divisible by --size",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "rmsprop"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD/RMSprop"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay (L2 penalty)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use (cuda will auto-fallback to cpu if unavailable)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for optimized G"
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="Scale factor for initial random matrix",
    )
    args = parser.parse_args()

    # Detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    c = args.size

    # Load or create initial graphon
    if args.input:
        print(f"Loading matrix from {args.input}...")
        G_loaded = np.load(args.input)

        # Validate
        assert G_loaded.ndim == 2, f"Loaded matrix must be 2D, got {G_loaded.ndim}D"
        assert G_loaded.shape[0] == G_loaded.shape[1], (
            f"Loaded matrix must be square, got {G_loaded.shape}"
        )

        n = G_loaded.shape[0]
        assert n % c == 0, (
            f"Loaded matrix size {n} must be divisible by target size {c}"
        )

        if n == c:
            print(f"Matrix is already {c}x{c}, using as-is")
            G_init = G_loaded
        else:
            print(f"Downscaling {n}x{n} matrix to {c}x{c} by block averaging...")
            G_init = downscale_matrix(G_loaded, c)

        # Ensure nonnegative and symmetric
        G_init = np.abs(G_init)
        G_init = (G_init + G_init.T) / 2

        # Normalize mean to 1/c
        current_mean = G_init.mean()
        target_mean = 1.0 / c
        if current_mean > 0:
            G_init = G_init * (target_mean / current_mean)

        print(
            f"Loaded and processed matrix: min={G_init.min():.4f}, max={G_init.max():.4f}, mean={G_init.mean():.4f}"
        )
    else:
        print(f"Initializing random {c}x{c} graphon...")
        G_init = np.abs(np.random.randn(c, c)) * args.init_scale
        G_init = (G_init + G_init.T) / 2  # Symmetrize

        # Normalize mean to 1/c
        current_mean = G_init.mean()
        target_mean = 1.0 / c
        if current_mean > 0:
            G_init = G_init * (target_mean / current_mean)

    print(f"Running optimization for {args.iterations} iterations...")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    print(f"Heawood einsum: {HEAWOOD_EINSUM}")
    print(f"Desargues einsum: {DESARGUES_EINSUM}")

    optimized_G, loss_history = optimize_graphon(
        G_init,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=device,
        verbose=True,
        print_every=100,
    )

    print(f"\nFinal loss: {loss_history[-1]:.6e}")
    if loss_history[-1] < 0:
        print("*** NEGATIVE LOSS ACHIEVED! ***")

    if args.output:
        np.save(args.output, optimized_G)
        print(f"Saved optimized graphon to {args.output}")

    return optimized_G, loss_history


if __name__ == "__main__":
    main()
