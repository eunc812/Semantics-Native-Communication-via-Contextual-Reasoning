"""
Section IV Step 9: System 1 vs System 2 기준 설정 및 reliability γ 계산.

논문 Section IV:
  - System 1: p_Xc|A(TRUE|a;t) ≥ 0.9인 concept만 추출해 SR로 사용, 전체를 한 번에 전송.
  - Listener: 수신한 개념(들)로 C2A로 의도 a 추론 (argmax p(a|수신 개념)).
  - Reliability γ = (listener가 맞춘 횟수) / (총 통신 라운드). Monte Carlo로 여러 의도 a 샘플링.
  - System 2: Algorithm 1로 K개 개념 선택 후 동일 방식으로 listener 추론 및 γ 계산.
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
    System 1 개념 추출 (논문 Section IV 기준).
    주어진 action a에 대해 p_Xc|A(TRUE|a;t) ≥ threshold 인 concept만 추출해 SR로 사용.
    논문: "all concepts c such that P_{X_c|A}(TRUE|a;t) >= 0.9 are extracted".

    Parameters
    ----------
    P_Xc_given_A : (|A|, |C|)
    a_hat : 의도된 행동 인덱스
    threshold : 추출 기준 (기본 0.9)

    Returns
    -------
    추출된 개념 인덱스 리스트 (한 번에 전송하는 "extracted concepts").
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
    Listener: 수신한 symbol(개념)들로 C2A로 의도 a 추론.
    p(a | received) ∝ pA(a) * Π_{c ∈ received} p_C|A(c|a), then argmax.

    Parameters
    ----------
    P_Xc_given_A : (|A|, |C|)
    pA : (|A|,) prior
    received_concepts : 수신한 개념 인덱스 리스트 (빈 리스트면 prior만 사용)

    Returns
    -------
    추론된 행동 인덱스 (argmax).
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
    System 1 통신 1라운드: 의도 a_hat → 추출 개념 전송 → listener 추론 → 맞으면 True.

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    a_hat : 화자 의도 행동
    threshold : System 1 추출 기준 (0.9)

    Returns
    -------
    correct : listener 추론이 a_hat과 일치하면 True.
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
    System 2 통신 1라운드: Algorithm 1로 K개 개념 선택 → listener는 최종 pA_final로 argmax 추론 → 맞으면 True.
    (Algorithm 1이 반환하는 pA_final이 이미 K개 개념 수신 후의 listener 사후이므로 그대로 argmax 사용.)

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    a_hat : 화자 의도 행동
    K : 선택할 개념 개수
    lam, alpha, beta, max_inner_iter, tol : System 2 / Algorithm 1 파라미터
    init_pC_A, init_pA_C : optional, Fig 8 perturbed/quantized A2C, C2A 초기화용

    Returns
    -------
    correct : listener 추론이 a_hat과 일치하면 True.
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
    # 통신 K-step 누적 업데이트 경로에서 마지막 step의 listener 추론을 사용.
    a_inferred = int(inferred_actions[K - 1])
    return a_inferred == a_hat


def compute_reliability_system1(
    world: World,
    n_rounds: int,
    seed: Optional[int] = None,
    threshold: float = 0.9,
) -> float:
    """
    System 1 reliability γ = (listener 맞춘 횟수) / (총 통신 라운드).
    여러 의도 a를 Monte Carlo 샘플링 (pA에 따라).

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    n_rounds : 통신 라운드 수
    seed : 재현성
    threshold : 추출 기준 (0.9)

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
    System 2 reliability γ = (listener 맞춘 횟수) / (총 통신 라운드).
    Monte Carlo로 의도 a 샘플링, 각 라운드에서 Algorithm 1로 K개 개념 선택 후 listener 추론.

    Parameters
    ----------
    world : (P_Xc_given_A, pA, pC)
    K : 개념 개수 (Algorithm 1)
    n_rounds : 통신 라운드 수
    seed : 재현성
    lam, alpha, beta, max_inner_iter, tol : System 2 파라미터
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
    System 2 reliability γ(K) for K=1..K_max를 행동 prior pA에 대해 정확 계산.

    Monte Carlo 샘플링 대신, 각 의도 행동 a에 대해 Algorithm 1 경로를 한 번만 구해
    γ(K)=sum_a pA(a) * 1[infer_K(a)=a] 를 계산한다.

    Returns
    -------
    gamma_by_k : np.ndarray, shape (K_max,)
        K=1..K_max에서의 reliability.
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

    print("Step 9: System 1 vs System 2 reliability (작은 월드 |A|=|C|=20, 소량 라운드)")
    thresh = 0.9
    n_rounds = 50
    gamma1 = compute_reliability_system1(world, n_rounds, seed=0, threshold=thresh)
    print("  System 1 (threshold={}): gamma = {:.4f} (n_rounds={})".format(thresh, gamma1, n_rounds))

    K = 3
    gamma2 = compute_reliability_system2(world, K, n_rounds, seed=0)
    print("  System 2 (K={}): gamma = {:.4f} (n_rounds={})".format(K, gamma2, n_rounds))
