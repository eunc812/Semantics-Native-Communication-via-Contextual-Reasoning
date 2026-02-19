"""
Section IV 실험용 월드 생성 (논문 Experimental setup).

논문 Section IV:
  - |A| = |C| = 100
  - Singular A2Cs: Dirichlet with hyperparameter pair (0.1, 0.1)로 생성
  - pA, pC: 통신 시작 시 균등(uniform)
  - 재현성: seed 고정 옵션
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
    논문 Section IV 실험 조건에 따른 월드 1개 생성.

    - P_Xc|A(TRUE|a;t): 각 action a별로 행 벡터를 Dirichlet(α, ..., α)로 샘플링.
      논문 "Dirichlet with hyperparameter pair (0.1, 0.1)" 해석: 각 concept가
      relevant할 확률 분포를 행별 Dirichlet(0.1, ..., 0.1)로 생성 (|C|개 차원).
    - pA, pC: 균등 (통신 시작 시 prior).

    Parameters
    ----------
    n_actions : int
        |A| (기본 100, 논문과 동일).
    n_concepts : int
        |C| (기본 100, 논문과 동일).
    dirichlet_alpha : float
        Dirichlet concentration. 논문 (0.1, 0.1) → 각 차원에 0.1 (기본 0.1).
    seed : int, optional
        재현성을 위한 시드.

    Returns
    -------
    P_Xc_given_A : np.ndarray, shape (|A|, |C|)
        P_Xc_given_A[a, c] = P(X_c=TRUE|a;t). 행별 합 = 1 (Dirichlet 샘플).
    pA : np.ndarray, shape (|A|,)
        균등 prior, sum=1.
    pC : np.ndarray, shape (|C|,)
        균등 prior, sum=1.
    """
    rng = np.random.default_rng(seed)
    alpha_vec = np.full(n_concepts, dirichlet_alpha)
    # 행별 Dirichlet(0.1, ..., 0.1): 각 action a에 대해 concept별 relevance 확률 벡터
    P_Xc_given_A = np.stack(
        [rng.dirichlet(alpha_vec) for _ in range(n_actions)],
        axis=0,
    )
    P_Xc_given_A = np.clip(P_Xc_given_A, _EPS, 1.0 - _EPS)
    # 행 합이 1이 되도록 유지 (Dirichlet은 이미 합=1)
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
    여러 월드 샘플 생성 (재현성: seed 고정 시 동일 순서로 생성).

    Parameters
    ----------
    n_worlds : int
        생성할 월드 개수.
    n_actions, n_concepts, dirichlet_alpha
        generate_experiment_world와 동일.
    seed : int, optional
        고정 시 동일한 n_worlds 개의 월드가 매번 같은 순서로 생성됨.

    Returns
    -------
    list of (P_Xc_given_A, pA, pC)
        길이 n_worlds.
    """
    rng = np.random.default_rng(seed)
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(n_worlds):
        # 월드마다 서로 다른 시드 사용 (seed 고정 시 동일 순서로 재현)
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
    # 데모: 월드 1개 + 3개 생성, shape 및 합 검증
    P, pA, pC = generate_experiment_world(seed=42)
    print("Section IV 실험용 월드 1개 (seed=42)")
    print("  P_Xc_given_A shape:", P.shape, "row sums (first 3):", np.round(P.sum(axis=1)[:3], 6))
    print("  pA sum:", pA.sum(), "pC sum:", pC.sum())

    worlds = generate_experiment_worlds(3, seed=0)
    print("\n월드 3개 생성 (seed=0):", len(worlds))
    for i, (Pw, pAw, pCw) in enumerate(worlds):
        print("  world", i, "P shape", Pw.shape, "pA sum", pAw.sum())
