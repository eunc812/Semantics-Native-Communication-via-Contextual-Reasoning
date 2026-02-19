"""
System 2 SNC - Individual Context (6)(7), G (8), Theorem 2 (10)-(13), Corollary 2 (18)(19).

  식 (6)(7): S, L individual context.
  식 (8): G 목적함수. (10)-(13): M1→S→M2→L 교대 최소화.
  식 (18)(19): 수렴한 M1, M2에서 rA2C, rC2A 추출 (Self-SNC).
"""

import numpy as np
from typing import Tuple, Optional, List

__all__ = [
    "individual_context_speaker",
    "individual_context_listener",
    "build_individual_contexts",
    "objective_G",
    "alternate_M1_S_M2_L",
    "minimize_G_alternating",
    "extract_rA2C_from_M1",
    "extract_rC2A_from_M2",
    "run_self_snc",
    "algorithm1_select_K_concepts",
    "algorithm1_listener_action_path",
    "compute_p_C_k_list",
    "system2_bitlength_bounds",
]

_EPS = 1e-12


def individual_context_speaker(
    pC_A: np.ndarray,
    pA: np.ndarray,
) -> np.ndarray:
    """
    논문 식 (6): Speaker의 individual context S(a,c;t).
    S(a,c;t) = p_C|A(c|a;t) * p_A(a) / Z,  Z = sum_{a',c'} p_C|A(c'|a';t) * p_A(a').

    rA2C와 행동 prior의 곱을 (a,c) 전체에 대해 정규화한 joint 분포.

    Parameters
    ----------
    pC_A : array, shape (|A|, |C|)
        rA2C (또는 초기화 시 System 1 A2C). pC_A[a,c] = p(C=c|A=a;t).
    pA : array, shape (|A|,)
        행동 prior p_A(a), sum = 1.

    Returns
    -------
    S : array, shape (|A|, |C|)
        S[a,c] = S(a,c;t). 전체 합 = 1.
    """
    pC_A = np.asarray(pC_A, dtype=float)
    pA = np.asarray(pA, dtype=float).ravel()
    assert pC_A.ndim == 2 and pA.shape[0] == pC_A.shape[0]
    joint = pC_A * pA[:, None]   # (|A|, |C|)
    z = np.maximum(joint.sum(), _EPS)
    return joint / z


def individual_context_listener(
    pA_C: np.ndarray,
    pC: np.ndarray,
) -> np.ndarray:
    """
    논문 식 (7): Listener의 individual context L(a,c;t).
    L(a,c;t) = p_A|C(a|c;t) * p_C(c) / Z,  Z = sum_{a',c'} p_A|C(a'|c';t) * p_C(c').

    rC2A와 개념 prior의 곱을 (a,c) 전체에 대해 정규화한 joint 분포.

    Parameters
    ----------
    pA_C : array, shape (|C|, |A|)
        rC2A (또는 초기화 시 System 1 C2A). pA_C[c,a] = p(A=a|C=c;t).
    pC : array, shape (|C|,)
        개념 prior p_C(c), sum = 1.

    Returns
    -------
    L : array, shape (|A|, |C|)
        L[a,c] = L(a,c;t). 전체 합 = 1.
    """
    pA_C = np.asarray(pA_C, dtype=float)
    pC = np.asarray(pC, dtype=float).ravel()
    assert pA_C.ndim == 2 and pA_C.shape[0] == pC.shape[0]
    # L[a,c] ∝ pA|C(a|c)*pC(c) = pA_C[c,a]*pC[c]. pA_C is (|C|,|A|), so (pA_C.T)[a,c]=pA_C[c,a].
    joint = pA_C.T * pC[None, :]   # (|A|,|C|), joint[a,c] = pA_C[c,a]*pC[c]
    z = np.maximum(joint.sum(), _EPS)
    return joint / z


def build_individual_contexts(
    pC_A: np.ndarray,
    pA: np.ndarray,
    pA_C: np.ndarray,
    pC: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    식 (6)(7) 한 번에: Speaker context S, Listener context L.
    System 1의 A2C, C2A와 prior pA, pC를 넣으면 초기 S, L 생성.

    Returns
    -------
    S : array, shape (|A|, |C|)
    L : array, shape (|A|, |C|)
    """
    S = individual_context_speaker(pC_A, pA)
    L = individual_context_listener(pA_C, pC)
    return S, L


# =============================================================================
# 식 (8) 목적함수 G, 식 (10)-(13) 교대 최소화 (Theorem 2)
# =============================================================================
#
# G = λ [H(S,M) - H(S)/α] + (1-λ) [H(L,M) - H(L)/β]
# H(S)=-Σ S log S, H(L)=-Σ L log L, H(S,M)=-Σ S log M, H(L,M)=-Σ L log M.
# 교대 순서: M1 (10) → S (11) → M2 (12) → L (13) → 반복.
#


def objective_G(
    S: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
) -> float:
    """
    논문 식 (8): G = λ [H(S,M) - H(S)/α] + (1-λ) [H(L,M) - H(L)/β].
    S, L, M은 (|A|,|C|) joint 분포, 전체 합=1.
    """
    S = np.asarray(S, dtype=float)
    L = np.asarray(L, dtype=float)
    M = np.asarray(M, dtype=float)
    M_safe = np.clip(M, _EPS, 1.0)
    S_safe = np.clip(S, _EPS, 1.0)
    L_safe = np.clip(L, _EPS, 1.0)
    H_S = -np.sum(S * np.log(S_safe))
    H_L = -np.sum(L * np.log(L_safe))
    H_SM = -np.sum(S * np.log(M_safe))
    H_LM = -np.sum(L * np.log(M_safe))
    return lam * (H_SM - H_S / alpha) + (1.0 - lam) * (H_LM - H_L / beta)


def alternate_M1_S_M2_L(
    S_prev: np.ndarray,
    L_prev: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Theorem 2 한 라운드: 식 (10)→(11)→(12)→(13).
    M1[r] = λ S[r-1] + (1-λ) L[r-1],
    S[r](a,c) = M1[r](a,c)^α / sum_{c'} M1[r](a,c')^α  (행별 정규화),
    M2[r] = λ S[r] + (1-λ) L[r-1],
    L[r](a,c) = M2[r](a,c)^β / sum_{a'} M2[r](a',c)^β  (열별 정규화).
    반환: (S_new, L_new, M1, M2).
    """
    S_prev = np.asarray(S_prev, dtype=float)
    L_prev = np.asarray(L_prev, dtype=float)
    M1 = lam * S_prev + (1.0 - lam) * L_prev
    M1 = np.maximum(M1, _EPS)
    raw1 = M1 ** alpha
    S_new = raw1 / np.maximum(raw1.sum(axis=1, keepdims=True), _EPS)
    M2 = lam * S_new + (1.0 - lam) * L_prev
    M2 = np.maximum(M2, _EPS)
    raw2 = M2 ** beta
    L_new = raw2 / np.maximum(raw2.sum(axis=0, keepdims=True), _EPS)
    return S_new, L_new, M1, M2


def minimize_G_alternating(
    S0: np.ndarray,
    L0: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    max_iter: int = 500,
    tol: float = 1e-9,
    history_scale: str = "raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], np.ndarray, np.ndarray]:
    """
    식 (8)을 식 (10)-(13) 교대 반복으로 최소화. 수렴 시 S*, L*, M* 반환.
    M* = lim M1 = lim M2 (Theorem 2). G(S[r-1], L[r-1]) ≥ G(S[r], L[r]) (식 14).
    history_scale: "raw" = G 그대로 기록. "joint" = S,L,M을 합 1로 정규화한 뒤 G 기록 (논문 Fig 4(a) 스케일).
    반환: (S, L, M, G_history, M1_last, M2_last). (18)(19) 추출 시 M1_last, M2_last 사용.
    """
    S = np.asarray(S0, dtype=float).copy()
    L = np.asarray(L0, dtype=float).copy()
    assert 0 < lam < 1 and alpha >= 0.9 and beta >= 0.9  # 논문 Fig 5: α, β ∈ [0.9, 2]
    history: List[float] = []
    M = None
    M1_last, M2_last = None, None
    g_prev_raw = None
    for r in range(1, max_iter + 1):
        S, L, M1, M2 = alternate_M1_S_M2_L(S, L, lam, alpha, beta)
        M1_last, M2_last = M1, M2
        M = M2
        g = objective_G(S, L, M, lam, alpha, beta)
        if history_scale == "joint":
            s, l, m = max(S.sum(), _EPS), max(L.sum(), _EPS), max(M.sum(), _EPS)
            g_record = objective_G(S / s, L / l, M / m, lam, alpha, beta)
        else:
            g_record = g
        history.append(g_record)
        if g_prev_raw is not None and abs(g - g_prev_raw) < tol:
            break
        g_prev_raw = g
    return S, L, M, history, M1_last, M2_last


# =============================================================================
# Corollary 2: 식 (18)(19) — M1, M2에서 rA2C, rC2A 추출 (Self-SNC)
# =============================================================================
#
# (18) p^[r]_C|A(c|a) = M1[r](a,c)^α / sum_{c'} M1[r](a,c')^α
# (19) p^[r]_A|C(a|c) = M2[r](a,c)^β / sum_{a'} M2[r](a',c)^β
#


def extract_rA2C_from_M1(M1: np.ndarray, alpha: float) -> np.ndarray:
    """
    논문 식 (18): M1에서 rA2C 추출.
    p^[r]_C|A(c|a) = M1(a,c)^α / sum_{c'} M1(a,c')^α. 행별 정규화.
    반환: (|A|, |C|), 행별 합 = 1.
    """
    M1 = np.maximum(np.asarray(M1, dtype=float), 1e-20)
    raw = M1 ** alpha
    row_sum = raw.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-20)
    return raw / row_sum


def extract_rC2A_from_M2(M2: np.ndarray, beta: float) -> np.ndarray:
    """
    논문 식 (19): M2에서 rC2A 추출.
    p^[r]_A|C(a|c) = M2(a,c)^β / sum_{a'} M2(a',c)^β. 열별 정규화 후 (|C|,|A|).
    반환: (|C|, |A|), pA_C[c,a], 행별 합 = 1.
    """
    M2 = np.maximum(np.asarray(M2, dtype=float), 1e-20)
    raw = M2 ** beta
    col_sum = raw.sum(axis=0, keepdims=True)
    col_sum = np.maximum(col_sum, 1e-20)
    pA_C = (raw / col_sum).T   # (|C|, |A|)
    return pA_C


def run_self_snc(
    S0: np.ndarray,
    L0: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    (10)-(13) self iteration 후 수렴한 S, L, M과 (18)(19)로 rA2C, rC2A 추출.
    반환: (S, L, M, rA2C, rC2A, G_history).
    """
    S, L, M, history, M1, M2 = minimize_G_alternating(
        S0, L0, lam, alpha, beta, max_iter=max_iter, tol=tol
    )
    rA2C = extract_rA2C_from_M1(M1, alpha)
    rC2A = extract_rC2A_from_M2(M2, beta)
    return S, L, M, rA2C, rC2A, history


# =============================================================================
# Algorithm 1: Selecting K Meaningful Concepts for System 2 SNC (논문 Section III-C)
# =============================================================================
#
# 입력: A, C, p_X|A(·|a;t), p_A|X(·|x;t'), p_A, p_C, α, β, λ.
# 출력: c_1, ..., c_K.
# 각 k에서: (6)(7)으로 S0,L0 → self-SNC 수렴 → M에서 p_C|A, p_A|C → c_k = argmax_c p_C|A(c|â)
# → C_k = C_{k-1} \ c_k, p_A ← p_A|C(·|c_k), p_C 재정규화·c_k=0.
#


def algorithm1_select_K_concepts(
    P_Xc_given_A: np.ndarray,
    pA_init: np.ndarray,
    pC_init: np.ndarray,
    a_hat: int,
    K: int,
    lam: float,
    alpha: float,
    beta: float,
    max_inner_iter: int = 500,
    tol: float = 1e-9,
    init_pC_A: Optional[np.ndarray] = None,
    init_pA_C: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Algorithm 1: 의도된 행동 â에 대해 K개의 의미 있는 개념을 탐욕적으로 선택.

    Parameters
    ----------
    P_Xc_given_A : array, shape (|A|, |C|)
        p_{X_c|A}(TRUE|a;t). System 1의 build_P_Xc_given_A 출력.
    pA_init : array, shape (|A|,)
        초기 행동 사전 p_A(a), sum=1.
    pC_init : array, shape (|C|,)
        초기 개념 사전 p_C(c), sum=1.
    a_hat : int
        화자의 의도된 행동 â ∈ A (인덱스).
    K : int
        선택할 개념 개수.
    lam, alpha, beta : float
        System 2 파라미터 (0 < λ < 1, α≥1, β≥1).
    max_inner_iter, tol
        self-SNC 내부 반복 횟수·수렴 허용오차.
    init_pC_A, init_pA_C : array, optional
        Fig 8 등: perturbed/quantized A2C, C2A로 S0, L0 초기화 시 사용. 둘 다 주어지면 P_Xc_given_A 대신 이걸로 (6)(7) 구함.

    Returns
    -------
    selected : list of int
        선택된 개념 인덱스 [c_1, ..., c_K].
    pA_final : array, shape (|A|,)
        마지막 업데이트된 p_A (p_A|C(·|c_K)).
    pC_final : array, shape (|C|,)
        마지막 업데이트된 p_C (c_1..c_K 제외·재정규화, 선택된 개념은 0).
    rA2C_per_k : list of (|A|,|C|) arrays
        각 k에서 수렴한 rA2C (선택 직전).
    rC2A_per_k : list of (|C|,|A|) arrays
        각 k에서 수렴한 rC2A (선택 직전).
    """
    from system1 import a2c_from_P_Xc_given_A, c2a_from_a2c_and_prior

    n_actions, n_concepts = P_Xc_given_A.shape
    assert 0 <= a_hat < n_actions and K >= 1 and K <= n_concepts
    pA = np.asarray(pA_init, dtype=float).ravel().copy()
    pC = np.asarray(pC_init, dtype=float).ravel().copy()
    pA /= pA.sum()
    pC /= pC.sum()
    remaining = np.ones(n_concepts, dtype=bool)
    selected: List[int] = []
    rA2C_per_k: List[np.ndarray] = []
    rC2A_per_k: List[np.ndarray] = []

    for k in range(K):
        # (6)(7) 초기화: 현재 pA, pC로 System 1 A2C/C2A → S0, L0 (또는 Fig 8용 init_pC_A, init_pA_C)
        if init_pC_A is not None and init_pA_C is not None:
            pC_A_s1 = np.asarray(init_pC_A, dtype=float).copy()
            pA_C_s1 = np.asarray(init_pA_C, dtype=float).copy()
        else:
            pC_A_s1 = a2c_from_P_Xc_given_A(P_Xc_given_A)
            pA_C_s1 = c2a_from_a2c_and_prior(pC_A_s1, pA)
        S0, L0 = build_individual_contexts(pC_A_s1, pA, pA_C_s1, pC)
        # Self-SNC 수렴 → rA2C, rC2A (Corollary 2)
        S, L, M, rA2C, rC2A, _ = run_self_snc(
            S0, L0, lam, alpha, beta,
            max_iter=max_inner_iter, tol=tol,
        )
        rA2C_per_k.append(rA2C)
        rC2A_per_k.append(rC2A)
        # Algorithm 1 Step 9–10:
        # Corollary 2 (18)(19)로 얻은 rA2C/rC2A를 라운드 k 의사결정에 직접 사용.
        p_C_A = rA2C                                   # (|A|, |C|)
        p_A_C = rC2A                                   # (|C|, |A|)
        # Step 11: c_k = argmax_{c ∈ C_{k-1}} p_C|A(c|â)
        score_ah = p_C_A[a_hat, :].copy()
        score_ah[~remaining] = -np.inf
        c_k = int(np.argmax(score_ah))
        selected.append(c_k)
        remaining[c_k] = False
        # Step 13: p_A(a) ← p_A|C(a|c_k; t)
        pA = p_A_C[c_k, :].ravel().copy()
        # Step 14: p_C(c_k)=0, ∀c ∈ C_k 에 대해 p_C 재정규화
        pC[c_k] = 0.0
        if remaining.sum() > 0:
            pC[remaining] /= pC[remaining].sum()

    return selected, pA, pC, rA2C_per_k, rC2A_per_k


def algorithm1_listener_action_path(
    P_Xc_given_A: np.ndarray,
    pA_init: np.ndarray,
    pC_init: np.ndarray,
    a_hat: int,
    K_max: int,
    lam: float,
    alpha: float,
    beta: float,
    max_inner_iter: int = 500,
    tol: float = 1e-9,
    init_pC_A: Optional[np.ndarray] = None,
    init_pA_C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Algorithm 1을 K=1..K_max까지 한 번에 수행하며 각 라운드의 listener 추론 행동을 반환.

    Fig 6처럼 "communication rounds = K" 곡선을 계산할 때,
    K별로 Algorithm 1을 다시 시작하는 중복을 피하기 위한 함수.

    Returns
    -------
    inferred_actions : array, shape (K_max,)
        inferred_actions[k-1] = K=k 라운드까지 진행했을 때 listener argmax 행동 인덱스.
    """
    from system1 import a2c_from_P_Xc_given_A, c2a_from_a2c_and_prior

    n_actions, n_concepts = P_Xc_given_A.shape
    assert 0 <= a_hat < n_actions and 1 <= K_max <= n_concepts

    pA = np.asarray(pA_init, dtype=float).ravel().copy()
    pC = np.asarray(pC_init, dtype=float).ravel().copy()
    pA /= np.maximum(pA.sum(), _EPS)
    pC /= np.maximum(pC.sum(), _EPS)

    remaining = np.ones(n_concepts, dtype=bool)
    inferred_actions = np.zeros(K_max, dtype=int)

    for k in range(K_max):
        if init_pC_A is not None and init_pA_C is not None:
            pC_A_s1 = np.asarray(init_pC_A, dtype=float).copy()
            pA_C_s1 = np.asarray(init_pA_C, dtype=float).copy()
        else:
            pC_A_s1 = a2c_from_P_Xc_given_A(P_Xc_given_A)
            pA_C_s1 = c2a_from_a2c_and_prior(pC_A_s1, pA)

        S0, L0 = build_individual_contexts(pC_A_s1, pA, pA_C_s1, pC)
        _, _, _, rA2C, rC2A, _ = run_self_snc(
            S0, L0, lam, alpha, beta,
            max_iter=max_inner_iter, tol=tol,
        )

        # K-step 누적 경로는 각 step의 rA2C/rC2A(Corollary 2)로 업데이트한다.
        p_C_A = rA2C                              # (|A|, |C|)
        p_A_C = rC2A                              # (|C|, |A|)

        score_ah = p_C_A[a_hat, :].copy()
        score_ah[~remaining] = -np.inf
        c_k = int(np.argmax(score_ah))

        pA = p_A_C[c_k, :].ravel().copy()
        inferred_actions[k] = int(np.argmax(pA))

        remaining[c_k] = False
        pC[c_k] = 0.0
        if remaining.any():
            pC[remaining] /= np.maximum(pC[remaining].sum(), _EPS)

    return inferred_actions


# =============================================================================
# Corollary 3: Bit-Length of SR in System 2 SNC (논문 식 21, 22, 23)
# =============================================================================
#
# 식 (23): p_C^k(c; t) = sum_a p_{C|A}^k(c|a; t) p_{A|C}^{k-1}(a|c_{k-1}; t)
# 식 (21) 하한: L_S2(t) >= - sum_{k=1}^K sum_c p_C^k(c) log2(p_C^k(c)) = sum_k H(p_C^k)
# 식 (22) 상한: Shannon 부호화에서 기대 길이 < H+1 관례에 맞춰,
#            상한 = sum_k sum_c p_C^k(c) * ceil(-log2(p_C^k(c))) (정수 부호 길이 기대값).
# → system1 Theorem 1(식4·5)과 동일: 하한=엔트로피, 상한=ceil(-log2(p)) 기대값.
#


def compute_p_C_k_list(
    selected: List[int],
    rA2C_per_k: List[np.ndarray],
    rC2A_per_k: List[np.ndarray],
    pA_init: np.ndarray,
) -> List[np.ndarray]:
    """
    논문 식 (23): k=1..K에 대해 p_C^k(c; t) 계산.

    p_C^k(c) = sum_a p_{C|A}^k(c|a) p_{A|C}^{k-1}(a|c_{k-1}).
    k=1일 때 p_{A|C}^0(a|c_0)는 행동 사전 p_A(a)로 둠.

    Parameters
    ----------
    selected : list of int, length K
        Algorithm 1이 반환한 선택 개념 인덱스 [c_1, ..., c_K].
    rA2C_per_k : list of (|A|,|C|) arrays
        k번째 step의 rA2C (p_{C|A}^k).
    rC2A_per_k : list of (|C|,|A|) arrays
        k번째 step의 rC2A (p_{A|C}^k). p_A|C(·|c_{k-1}) = rC2A_per_k[k-1][c_{k-1}, :].
    pA_init : array, shape (|A|,)
        초기 행동 사전.

    Returns
    -------
    p_C_k_list : list of (|C|,) arrays
        [p_C^1, p_C^2, ..., p_C^K]. 각각 sum=1.
    """
    K = len(selected)
    assert len(rA2C_per_k) == K and len(rC2A_per_k) == K
    pA_init = np.asarray(pA_init, dtype=float).ravel()
    p_C_k_list: List[np.ndarray] = []
    for k in range(K):
        rA2C_k = rA2C_per_k[k]  # (|A|, |C|)
        if k == 0:
            p_action = pA_init
        else:
            c_prev = selected[k - 1]
            p_action = rC2A_per_k[k - 1][c_prev, :].ravel()  # p_A|C(·|c_{k-1})
        # p_C^k(c) = sum_a p_{C|A}^k(c|a) p_action(a) = (rA2C_k.T @ p_action)
        p_C_k = (rA2C_k.T @ p_action).ravel()
        p_C_k = np.maximum(p_C_k, 0.0)
        z = p_C_k.sum()
        if z > _EPS:
            p_C_k = p_C_k / z
        p_C_k_list.append(p_C_k)
    return p_C_k_list


def system2_bitlength_bounds(
    p_C_k_list: List[np.ndarray],
) -> Tuple[float, float]:
    """
    논문 Corollary 3 식 (21)(22): System 2 SR 기대 비트길이 하한·상한.

    식 (21) 하한: L_S2 >= - sum_k sum_c p_C^k(c) log2(p_C^k(c)) = sum_k H(p_C^k).
    식 (22) 상한: Shannon 정수 부호 길이 관례에 따라
                 L_S2 <= sum_k sum_c p_C^k(c) * ceil(-log2(p_C^k(c)))
                 (Theorem 1 식5·system1과 동일한 상한 정의.)

    Returns
    -------
    L_lower, L_upper : float
        비트 단위 기대 길이 하한·상한. L_lower <= L_upper (일반적으로 상한이 더 큼).
    """
    L_lower = 0.0
    L_upper = 0.0
    for p_C_k in p_C_k_list:
        p_C_k = np.asarray(p_C_k, dtype=float).ravel()
        p_pos = np.maximum(p_C_k, 0.0)
        # 하한: 엔트로피 H(p). p=0이면 0*log(0)=0
        mask = p_pos > _EPS
        if np.any(mask):
            L_lower += -np.sum(p_pos[mask] * np.log2(p_pos[mask]))
        # 상한: 기대 부호 길이 E[ceil(-log2(p))] (정수 길이 Shannon 부호)
        p_clip = np.maximum(p_pos, _EPS)
        L_upper += float(np.sum(p_pos * np.ceil(-np.log2(p_clip))))
    return L_lower, L_upper


if __name__ == "__main__":
    from system1 import (
        build_P_Xc_given_A,
        a2c_from_P_Xc_given_A,
        c2a_from_a2c_and_prior,
    )
    n_actions, n_concepts = 3, 3
    P_fixed = np.array([
        [0.9, 0.1, 0.0],
        [0.9, 0.9, 0.0],
        [0.9, 0.9, 0.9],
    ], dtype=float)
    P_Xc_given_A = build_P_Xc_given_A(n_actions, n_concepts, fixed_values=P_fixed)
    pA = np.ones(n_actions) / n_actions
    pC = np.ones(n_concepts) / n_concepts
    pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)
    pA_C = c2a_from_a2c_and_prior(pC_A, pA)
    S0, L0 = build_individual_contexts(pC_A, pA, pA_C, pC)
    print("System 2 Individual Context (식 6, 7)")
    print("S0 sum =", S0.sum(), ", L0 sum =", L0.sum())

    lam, alpha, beta = 0.5, 1.5, 1.5
    M0 = lam * S0 + (1.0 - lam) * L0
    g0 = objective_G(S0, L0, M0, lam, alpha, beta)
    print("식 (8) G(S0, L0, M0) =", g0)

    S, L, M, history, M1, M2 = minimize_G_alternating(S0, L0, lam, alpha, beta, max_iter=200, tol=1e-9)
    print("Theorem 2 (10)-(13) 교대 최소화 후:")
    print("  반복 수:", len(history))
    print("  G 초기 -> 최종:", history[0], "->", history[-1])
    print("  S*, L*, M* sum:", S.sum(), L.sum(), M.sum())

    rA2C = extract_rA2C_from_M1(M1, alpha)
    rC2A = extract_rC2A_from_M2(M2, beta)
    print("Corollary 2 (18)(19) rA2C, rC2A 추출: rA2C 행합", rA2C.sum(axis=1)[:2], "... rC2A 행합", rC2A.sum(axis=1)[:2], "...")

    import sys
    if "--alg1" in sys.argv:
        print("Algorithm 1: K meaningful concepts (same P, a_hat=0, K=2)")
        a_hat = 0
        K = 2
        selected, pA_fin, pC_fin, rA2C_list, rC2A_list = algorithm1_select_K_concepts(
            P_Xc_given_A, pA, pC, a_hat, K, lam, alpha, beta, max_inner_iter=200, tol=1e-9
        )
        print("  선택된 개념 인덱스:", selected)
        print("  Final p_A (given c_K):", np.round(pA_fin, 4))
        print("  Final p_C (selected=0, rest renormalized):", np.round(pC_fin, 4))
    elif "--bitlength" in sys.argv:
        print("Corollary 3: System 2 bit-length (Eq. 21, 22, 23)")
        a_hat = 0
        K = 2
        selected, pA_fin, pC_fin, rA2C_list, rC2A_list = algorithm1_select_K_concepts(
            P_Xc_given_A, pA, pC, a_hat, K, lam, alpha, beta, max_inner_iter=200, tol=1e-9
        )
        p_C_k_list = compute_p_C_k_list(selected, rA2C_list, rC2A_list, pA)
        L_lo, L_up = system2_bitlength_bounds(p_C_k_list)
        print("  Selected concepts:", selected)
        print("  L_lower (Eq.21), L_upper (Eq.22):", round(L_lo, 6), round(L_up, 6))
    else:
        print("Options: --alg1, --bitlength")
