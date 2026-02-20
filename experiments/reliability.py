"""
Section IV Step 9: System 1 vs System 2 setup and reliability γ computation.

Paper Section IV:
  - System 1: extract only concepts with p_Xc|A(TRUE|a;t) ≥ 0.9 for SR, send all at once.
  - Listener: infer intended action a from received concept(s) via C2A (argmax p(a|received concepts)).
  - Reliability γ = (number of correct listener matches) / (total communication rounds). Monte Carlo over intended actions a.
  - System 2: select K concepts via Algorithm 1, then same listener inference and γ computation.
"""

import numpy as np
from typing import Tuple, List, Optional

__all__ = [
    "extract_concepts_system1",
    "listener_infer_system1",
    "run_round_system1",
    "run_round_system2",
    "compute_reliability_system1",
    "compute_reliability_system2",
    "compute_reliability_system2_by_rounds_exact",
]

_EPS = 1e-12

# World = (P_Xc_given_A, pA, pC)
World = Tuple[np.ndarray, np.ndarray, np.ndarray]


def extract_concepts_system1(
    P_Xc_given_A: np.ndarray,
    a_hat: int,
    threshold: float = 0.9,
) -> List[int]:
    """
    System 1 concept extraction (paper Section IV).
    For given action a, extract only concepts with p_Xc|A(TRUE|a;t) ≥ threshold for SR.
    Paper: "all concepts c such that P_{X_c|A}(TRUE|a;t) >= 0.9 are extracted".

    Parameters
    ----------
    P_Xc_given_A : (|A|, |C|)
    a_hat : intended action index
    threshold : extraction criterion (default 0.9)

    Returns
    -------
    List of extracted concept indices ("extracted concepts" sent at once).
    """
    P_Xc_given_A = np.asarray(P_Xc_given_A, dtype=float)
    row = P_Xc_given_A[a_hat, :].ravel()
    return [int(c) for c in np.where(row >= threshold)[0]]


def listener_infer_system1(
    P_Xc_given_A: np.ndarray,
    pA: np.ndarray,
    received_concepts: List[int],
) -> int:
    """
    Listener: infer intended action a from received symbols (concepts) via C2A.
    p(a | received) ∝ pA(a) * Π_{c ∈ received} p_C|A(c|a), then argmax.

    Parameters
    ----------
    P_Xc_given_A : (|A|, |C|)
    pA : (|A|,) prior
    received_concepts : list of received concept indices (empty list => prior only)

    Returns
    -------
    Inferred action index (argmax).
    """
    from system1 import a2c_from_P_Xc_given_A

    pA = np.asarray(pA, dtype=float).ravel()
    if not received_concepts:
        return int(np.argmax(pA))
    pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)  # (|A|, |C|)
    # posterior[a] ∝ pA[a] * Π_{c in received} pC_A[a, c]
    log_post = np.log(np.maximum(pA, _EPS))
    for c in received_concepts:
        log_post += np.log(np.maximum(pC_A[:, c], _EPS))
    return int(np.argmax(log_post))


def run_round_system1(
    world: World,
    a_hat: int,
    threshold: float = 0.9,
) -> bool:
    """
    System 1 one communication round: intended a_hat → send extracted concepts → listener infers → True if match.

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    a_hat : speaker intended action
    threshold : System 1 extraction criterion (0.9)

    Returns
    -------
    correct : True if listener inference matches a_hat.
    """
    P_Xc_given_A, pA, pC = world
    extracted = extract_concepts_system1(P_Xc_given_A, a_hat, threshold)
    a_inferred = listener_infer_system1(P_Xc_given_A, pA, extracted)
    return a_inferred == a_hat


def run_round_system2(
    world: World,
    a_hat: int,
    K: int,
    lam: float = 0.5,
    alpha: float = 1.5,
    beta: float = 1.5,
    max_inner_iter: int = 300,
    tol: float = 1e-9,
    init_pC_A: Optional[np.ndarray] = None,
    init_pA_C: Optional[np.ndarray] = None,
) -> bool:
    """
    System 2 one communication round: Algorithm 1 selects K concepts → listener infers argmax from final pA_final → True if match.
    (pA_final from Algorithm 1 is already listener posterior after K concepts; use argmax directly.)

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    a_hat : speaker intended action
    K : number of concepts to select
    lam, alpha, beta, max_inner_iter, tol : System 2 / Algorithm 1 parameters
    init_pC_A, init_pA_C : optional, for Fig 8 perturbed/quantized A2C, C2A initialization

    Returns
    -------
    correct : True if listener inference matches a_hat.
    """
    from system2 import algorithm1_listener_action_path

    P_Xc_given_A, pA, pC = world
    inferred_actions = algorithm1_listener_action_path(
        P_Xc_given_A=P_Xc_given_A,
        pA_init=pA,
        pC_init=pC,
        a_hat=a_hat,
        K_max=K,
        lam=lam, alpha=alpha, beta=beta,
        max_inner_iter=max_inner_iter, tol=tol,
        init_pC_A=init_pC_A, init_pA_C=init_pA_C,
    )
    # Use listener inference from last step of K-step cumulative update path.
    a_inferred = int(inferred_actions[K - 1])
    return a_inferred == a_hat


def compute_reliability_system1(
    world: World,
    n_rounds: int,
    seed: Optional[int] = None,
    threshold: float = 0.9,
) -> float:
    """
    System 1 reliability γ = (number of correct listener matches) / (total communication rounds).
    Monte Carlo sampling over intended actions a (according to pA).

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    n_rounds : number of communication rounds
    seed : reproducibility
    threshold : extraction criterion (0.9)

    Returns
    -------
    gamma : reliability, 0~1.
    """
    P_Xc_given_A, pA, pC = world
    n_actions = P_Xc_given_A.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_actions, size=n_rounds, replace=True, p=pA)
    correct = 0
    for a_hat in indices:
        if run_round_system1(world, int(a_hat), threshold):
            correct += 1
    return correct / n_rounds if n_rounds else 0.0


def compute_reliability_system2(
    world: World,
    K: int,
    n_rounds: int,
    seed: Optional[int] = None,
    lam: float = 0.5,
    alpha: float = 1.5,
    beta: float = 1.5,
    max_inner_iter: int = 300,
    tol: float = 1e-9,
    init_pC_A: Optional[np.ndarray] = None,
    init_pA_C: Optional[np.ndarray] = None,
) -> float:
    """
    System 2 reliability γ = (number of correct listener matches) / (total communication rounds).
    Monte Carlo over intended a; each round Algorithm 1 selects K concepts then listener infers.

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    K : number of concepts (Algorithm 1)
    n_rounds : number of communication rounds
    seed : reproducibility
    lam, alpha, beta, max_inner_iter, tol : System 2 parameters
    init_pC_A, init_pA_C : optional, Fig 8 perturbed/quantized A2C, C2A

    Returns
    -------
    gamma : reliability, 0~1.
    """
    P_Xc_given_A, pA, pC = world
    n_actions = P_Xc_given_A.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_actions, size=n_rounds, replace=True, p=pA)
    correct = 0
    for a_hat in indices:
        if run_round_system2(
            world, int(a_hat), K,
            lam=lam, alpha=alpha, beta=beta,
            max_inner_iter=max_inner_iter, tol=tol,
            init_pC_A=init_pC_A, init_pA_C=init_pA_C,
        ):
            correct += 1
    return correct / n_rounds if n_rounds else 0.0


def compute_reliability_system2_by_rounds_exact(
    world: World,
    K_max: int,
    lam: float = 0.5,
    alpha: float = 1.5,
    beta: float = 1.5,
    max_inner_iter: int = 300,
    tol: float = 1e-9,
    init_pC_A: Optional[np.ndarray] = None,
    init_pA_C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    System 2 reliability γ(K) for K=1..K_max computed exactly over action prior pA.

    Instead of Monte Carlo, for each intended action a compute Algorithm 1 path once and
    γ(K)=sum_a pA(a) * 1[infer_K(a)=a].

    Returns
    -------
    gamma_by_k : np.ndarray, shape (K_max,)
        reliability at K=1..K_max.
    """
    from system2 import algorithm1_listener_action_path

    P_Xc_given_A, pA, pC = world
    pA = np.asarray(pA, dtype=float).ravel()
    pA = pA / np.maximum(pA.sum(), _EPS)
    n_actions = P_Xc_given_A.shape[0]

    gamma_by_k = np.zeros(K_max, dtype=float)
    for a_hat in range(n_actions):
        inferred = algorithm1_listener_action_path(
            P_Xc_given_A=P_Xc_given_A,
            pA_init=pA,
            pC_init=pC,
            a_hat=int(a_hat),
            K_max=K_max,
            lam=lam,
            alpha=alpha,
            beta=beta,
            max_inner_iter=max_inner_iter,
            tol=tol,
            init_pC_A=init_pC_A,
            init_pA_C=init_pA_C,
        )
        gamma_by_k += pA[a_hat] * (inferred == int(a_hat)).astype(float)
    return gamma_by_k


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.world import generate_experiment_world

    P, pA, pC = generate_experiment_world(n_actions=20, n_concepts=20, seed=42)
    world = (P, pA, pC)

    print("Step 9: System 1 vs System 2 reliability (small world |A|=|C|=20, few rounds)")
    thresh = 0.9
    n_rounds = 50
    gamma1 = compute_reliability_system1(world, n_rounds, seed=0, threshold=thresh)
    print("  System 1 (threshold={}): gamma = {:.4f} (n_rounds={})".format(thresh, gamma1, n_rounds))

    K = 3
    gamma2 = compute_reliability_system2(world, K, n_rounds, seed=0)
    print("  System 2 (K={}): gamma = {:.4f} (n_rounds={})".format(K, gamma2, n_rounds))
