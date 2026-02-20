"""
Section IV experiment world generation (paper Experimental setup).

Paper Section IV:
  - |A| = |C| = 100
  - Singular A2Cs: generated with Dirichlet hyperparameter pair (0.1, 0.1)
  - pA, pC: uniform at communication start
  - Reproducibility: optional fixed seed
"""

import numpy as np
from typing import Tuple, Optional, List

__all__ = [
    "generate_experiment_world",
    "generate_experiment_worlds",
]

_EPS = 1e-12


def generate_experiment_world(
    n_actions: int = 100,
    n_concepts: int = 100,
    dirichlet_alpha: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one world according to paper Section IV experiment conditions.

    - P_Xc|A(TRUE|a;t): for each action a, sample row vector from Dirichlet(α, ..., α).
      Paper "Dirichlet with hyperparameter pair (0.1, 0.1)": row-wise Dirichlet(0.1, ..., 0.1)
      for relevance probability over |C| concepts.
    - pA, pC: uniform (prior at communication start).

    Parameters
    ----------
    n_actions : int
        |A| (default 100, same as paper).
    n_concepts : int
        |C| (default 100, same as paper).
    dirichlet_alpha : float
        Dirichlet concentration. Paper (0.1, 0.1) → 0.1 per dimension (default 0.1).
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    P_Xc_given_A : np.ndarray, shape (|A|, |C|)
        P_Xc_given_A[a, c] = P(X_c=TRUE|a;t). Rows sum = 1 (Dirichlet sample).
    pA : np.ndarray, shape (|A|,)
        Uniform prior, sum=1.
    pC : np.ndarray, shape (|C|,)
        Uniform prior, sum=1.
    """
    rng = np.random.default_rng(seed)
    alpha_vec = np.full(n_concepts, dirichlet_alpha)
    # Row-wise Dirichlet(0.1, ..., 0.1): per action a, concept-wise relevance probability vector
    P_Xc_given_A = np.stack(
        [rng.dirichlet(alpha_vec) for _ in range(n_actions)],
        axis=0,
    )
    P_Xc_given_A = np.clip(P_Xc_given_A, _EPS, 1.0 - _EPS)
    # Keep row sums 1 (Dirichlet already sums to 1)
    P_Xc_given_A = P_Xc_given_A / P_Xc_given_A.sum(axis=1, keepdims=True)

    pA = np.ones(n_actions) / n_actions
    pC = np.ones(n_concepts) / n_concepts
    return P_Xc_given_A, pA, pC


def generate_experiment_worlds(
    n_worlds: int,
    n_actions: int = 100,
    n_concepts: int = 100,
    dirichlet_alpha: float = 0.1,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate multiple world samples (reproducibility: fixed seed yields same order).

    Parameters
    ----------
    n_worlds : int
        Number of worlds to generate.
    n_actions, n_concepts, dirichlet_alpha
        Same as generate_experiment_world.
    seed : int, optional
        If fixed, same n_worlds worlds generated in same order each time.

    Returns
    -------
    list of (P_Xc_given_A, pA, pC)
        Length n_worlds.
    """
    rng = np.random.default_rng(seed)
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(n_worlds):
        # Use different seed per world (fixed seed => same order reproducible)
        s = int(rng.integers(0, 2**31))
        out.append(
            generate_experiment_world(
                n_actions=n_actions,
                n_concepts=n_concepts,
                dirichlet_alpha=dirichlet_alpha,
                seed=int(s),
            )
        )
    return out


if __name__ == "__main__":
    # Demo: generate 1 + 3 worlds, verify shape and sums
    P, pA, pC = generate_experiment_world(seed=42)
    print("Section IV experiment world x1 (seed=42)")
    print("  P_Xc_given_A shape:", P.shape, "row sums (first 3):", np.round(P.sum(axis=1)[:3], 6))
    print("  pA sum:", pA.sum(), "pC sum:", pC.sum())

    worlds = generate_experiment_worlds(3, seed=0)
    print("\n3 worlds generated (seed=0):", len(worlds))
    for i, (Pw, pAw, pCw) in enumerate(worlds):
        print("  world", i, "P shape", Pw.shape, "pA sum", pAw.sum())
