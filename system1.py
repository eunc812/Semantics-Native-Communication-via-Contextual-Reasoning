"""
System 1 SNC - 통합 모듈 (논문 Section II 순서)

  (1) Action-Concept Relevance: p_X|A(x|a;t) = Π_c p_{X_c|A}(x_c|a;t)
  (2) A2C: p_{C|A}(c|a;t) = p_{X_c|A}(TRUE|a;t) / sum_{c'} p_{X_c'|A}(TRUE|a;t)
  (3) C2A: p_{A|C}(a|c;t) = Bayes(A2C, pA)
  (4) Concept-Symbol: C2S s:C→S, S2C s⁻¹:S→C, SR = C2S(관련 개념) → Shannon 부호화(비트 길이) 입력
  (5) Theorem 1: System 1 SNC에서 SR의 기대 비트 길이 L_S1 하한/상한
"""

import numpy as np
from typing import Union, Optional, List, Tuple

_EPS = 1e-12

__all__ = [
    "build_P_Xc_given_A",
    "p_X_given_A_single",
    "p_X_given_A_batch",
    "a2c_from_P_Xc_given_A",
    "c2a_from_a2c_and_prior",
    "build_action_concept_model",
    "build_c2s_s2c",
    "concepts_to_symbols",
    "symbols_to_concepts",
    "sr_symbol_indices_for_action",
    "sr_symbol_indices_from_concepts",
    "p_xc_true",
    "expected_sr_bitlength_bounds",
    "example_system1_full",
]


# =============================================================================
# Part 1. Action-Concept Relevance (논문 식 (1))
# =============================================================================
#
# P_Xc_given_A: shape (|A|, |C|), P_Xc_given_A[a,c] = P(X_c=TRUE|a;t)
#


def build_P_Xc_given_A(
    n_actions: int,
    n_concepts: int,
    rng: Optional[np.random.Generator] = None,
    fixed_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    식 (1)의 빌딩 블록 p_{X_c|A}(x_c|a;t)를 행렬로 구성.
    반환: P_Xc_given_A, shape (n_actions, n_concepts), P_Xc_given_A[a,c] = P(X_c=TRUE|a;t).
    """
    if rng is None:
        rng = np.random.default_rng()
    if fixed_values is not None:
        assert fixed_values.shape == (n_actions, n_concepts)
        return np.clip(fixed_values.astype(float), 1e-9, 1.0 - 1e-9)
    P = rng.uniform(0.1, 0.9, size=(n_actions, n_concepts))
    return P


def p_X_given_A_single(
    x: np.ndarray,
    a: int,
    P_Xc_given_A: np.ndarray,
) -> float:
    """
    식 (1): p_X|A(x|a;t) = Π_{c∈C} p_{X_c|A}(x_c|a;t).
    x: (n_concepts,) 이진 벡터, 1=관련 0=비관련.
    """
    x = np.asarray(x, dtype=int)
    assert x.ndim == 1 and x.shape[0] == P_Xc_given_A.shape[1]
    assert np.all((x == 0) | (x == 1)) and 0 <= a < P_Xc_given_A.shape[0]
    p_true = P_Xc_given_A[a, :]
    p_per = np.where(x == 1, p_true, 1.0 - p_true)
    return float(np.prod(p_per))


def p_X_given_A_batch(
    a: int,
    P_Xc_given_A: np.ndarray,
    X_binary: np.ndarray,
) -> np.ndarray:
    """여러 x에 대해 p_X|A(x|a;t) 일괄 계산. X_binary: (n_samples, n_concepts)."""
    p_true = P_Xc_given_A[a, :]
    p_per = np.where(X_binary == 1, p_true, 1.0 - p_true)
    return np.prod(p_per, axis=1)


# =============================================================================
# Part 2. Action-Concept Model (논문 식 (2), (3))
# =============================================================================


def a2c_from_P_Xc_given_A(P_Xc_given_A: np.ndarray) -> np.ndarray:
    """
    식 (2) A2C: p_{C|A}(c|a;t) = p_{X_c|A}(TRUE|a;t) / sum_{c'} p_{X_c'|A}(TRUE|a;t).
    반환: pC_A, shape (|A|, |C|), 행별 합 = 1.
    """
    P = np.asarray(P_Xc_given_A, dtype=float)
    row_sum = np.maximum(P.sum(axis=1, keepdims=True), _EPS)
    return P / row_sum


def c2a_from_a2c_and_prior(pC_A: np.ndarray, pA: np.ndarray) -> np.ndarray:
    """
    식 (3) C2A: p_{A|C}(a|c;t) = p_{C|A}(c|a;t)p_A(a) / sum_{a'} ...
    반환: pA_C, shape (|C|, |A|), 행별 합 = 1.
    """
    pC_A = np.asarray(pC_A, dtype=float)
    pA = np.asarray(pA, dtype=float).ravel()
    joint = pC_A * pA[:, None]
    denom = np.maximum(joint.sum(axis=0), _EPS)
    return (joint / denom).T


def build_action_concept_model(
    P_Xc_given_A: np.ndarray,
    pA: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """식 (2)(3) 한 번에: A2C → C2A. 반환 (pC_A, pA_C)."""
    pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)
    pA_C = c2a_from_a2c_and_prior(pC_A, pA)
    return pC_A, pA_C


# =============================================================================
# Part 3. Concept-Symbol Model (C2S / S2C, SR)
# =============================================================================
#
# C2S: s:C→S, S2C: s⁻¹:S→C. 논문 가정: 일대일, 결정론, |C|=|S|.
# SR = 의도 a → 관련 개념 추출 → C2S → 심볼 인덱스 (Shannon 부호화 소스).
#


def build_c2s_s2c(
    n_concepts: int,
    identity: bool = True,
    perm: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    C2S/S2C 매핑. identity=True면 s(c)=c.
    반환: (c2s_map, s2c_map), 각 shape (|C|,) = (|S|,).
    """
    if identity:
        arr = np.arange(n_concepts, dtype=np.intp)
        return arr.copy(), arr.copy()
    if perm is not None:
        perm = np.asarray(perm, dtype=np.intp).ravel()
        assert perm.shape[0] == n_concepts and np.all(np.sort(perm) == np.arange(n_concepts))
    else:
        rng = rng or np.random.default_rng()
        perm = rng.permutation(n_concepts)
    c2s_map = perm.copy()
    s2c_map = np.empty(n_concepts, dtype=np.intp)
    s2c_map[c2s_map] = np.arange(n_concepts)
    return c2s_map, s2c_map


def concepts_to_symbols(
    concept_indices: Union[np.ndarray, List[int]],
    c2s_map: np.ndarray,
) -> np.ndarray:
    """개념 인덱스 → 심볼 인덱스 (C2S). SR로 Shannon 부호화에 넘어감."""
    c = np.atleast_1d(np.asarray(concept_indices, dtype=np.intp))
    return c2s_map[c]


def symbols_to_concepts(
    symbol_indices: Union[np.ndarray, List[int]],
    s2c_map: np.ndarray,
) -> np.ndarray:
    """심볼 인덱스 → 개념 인덱스 (S2C). 수신 측 C2A 입력용."""
    s = np.atleast_1d(np.asarray(symbol_indices, dtype=np.intp))
    return s2c_map[s]


def sr_symbol_indices_for_action(
    a: int,
    P_Xc_given_A: np.ndarray,
    c2s_map: np.ndarray,
    *,
    threshold: float = 0.9,
) -> np.ndarray:
    """
    논문 Section IV: 행동 a에 대해 P(X_c=TRUE|a) >= threshold 인 개념 추출 → C2S → SR(심볼 인덱스).
    이 SR이 비트 길이(Theorem 1)의 소스 입력.
    """
    P = np.asarray(P_Xc_given_A, dtype=float)
    assert 0 <= a < P.shape[0] and c2s_map.shape[0] == P.shape[1]
    relevant = np.where(P[a, :] >= threshold)[0]
    return concepts_to_symbols(relevant, c2s_map)


def sr_symbol_indices_from_concepts(
    concept_indices: Union[np.ndarray, List[int]],
    c2s_map: np.ndarray,
) -> np.ndarray:
    """이미 정해진 개념 인덱스(예: A2C 샘플)를 SR 심볼로 변환."""
    return concepts_to_symbols(concept_indices, c2s_map)


# =============================================================================
# Part 4. Shannon Coding under System 1 — SR 비트 길이 (Theorem 1)
# =============================================================================
#
# pXc(TRUE) = sum_a pXc|A(TRUE|a) * pA(a): 개념 c가 행동에 관련될 확률.
# 식 (4) 하한: L_S1 >= - sum_c pXc(TRUE) * log2( pXc(TRUE) / sum_c' pXc'(TRUE) )
# 식 (5) 상한: L_S1 <= sum_c pXc(TRUE) * ceil( -log2( pXc(TRUE) / sum_c' pXc'(TRUE) ) )
#


def p_xc_true(P_Xc_given_A: np.ndarray, pA: np.ndarray) -> np.ndarray:
    """
    pXc(TRUE) = sum_a pXc|A(TRUE|a) * pA(a).
    개념 c가 (prior pA 하에) 행동에 관련될 주변 확률.
    반환: shape (|C|,).
    """
    P = np.asarray(P_Xc_given_A, dtype=float)
    pA = np.asarray(pA, dtype=float).ravel()
    assert P.shape[0] == pA.shape[0]
    return (P.T @ pA).ravel()


def expected_sr_bitlength_bounds(
    P_Xc_given_A: np.ndarray,
    pA: np.ndarray,
) -> Tuple[float, float]:
    """
    Theorem 1: System 1 SNC에서 SR의 기대 비트 길이 L_S1 하한·상한.
    하한(식 4): 엔트로피 하한. 상한(식 5): ceil(-log2(fc)) 부호 길이의 기대값.
    반환: (lower_bound, upper_bound) in bits.
    """
    p_xc = p_xc_true(P_Xc_given_A, pA)
    total = p_xc.sum()
    if total <= 0:
        return 0.0, 0.0
    fc = np.clip(p_xc / total, 1e-12, 1.0)
    # 하한: - sum_c pXc(TRUE) * log2(fc). 0*log(0)=0 처리.
    mask = p_xc > 0
    lower = -np.sum(p_xc[mask] * np.log2(fc[mask]))
    # 상한: sum_c pXc(TRUE) * ceil(-log2(fc))
    upper = float(np.sum(p_xc * np.ceil(-np.log2(fc))))
    return lower, upper


# =============================================================================
# 예시: System 1 전체를 순서대로 한 번에
# =============================================================================

def example_system1_full():
    """
    (1) Action-Concept Relevance → (2)(3) A2C/C2A → (4) C2S/S2C, SR 생성.
    """
    n_actions, n_concepts = 3, 3
    P_fixed = np.array([
        [0.9, 0.1, 0.0],
        [0.9, 0.9, 0.0],
        [0.9, 0.9, 0.9],
    ], dtype=float)

    # (1) P_Xc_given_A
    P_Xc_given_A = build_P_Xc_given_A(n_actions, n_concepts, fixed_values=P_fixed)
    print("=" * 60)
    print("(1) Action-Concept Relevance  P_Xc_given_A[a,c] = P(X_c=TRUE|a)")
    print("    shape (|A|,|C|):")
    print(P_Xc_given_A)

    # (2) A2C
    pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)
    print()
    print("(2) A2C  pC_A[a,c] = p(C=c|A=a), 행별 합=1:")
    print(pC_A)

    # (3) C2A
    pA = np.ones(n_actions) / n_actions
    pA_C = c2a_from_a2c_and_prior(pC_A, pA)
    print()
    print("(3) C2A  pA_C[c,a] = p(A=a|C=c), 행별 합=1:")
    print(pA_C)

    # (4) C2S/S2C, SR
    c2s_map, s2c_map = build_c2s_s2c(n_concepts, identity=True)
    print()
    print("(4) Concept-Symbol  identity: c2s_map=%s, s2c_map=%s" % (c2s_map.tolist(), s2c_map.tolist()))
    print("    행동별 SR (threshold=0.9):")
    for a in range(n_actions):
        sr = sr_symbol_indices_for_action(a, P_Xc_given_A, c2s_map, threshold=0.9)
        print("      a=%d -> SR 심볼 인덱스 %s" % (a, sr.tolist()))

    # (5) Theorem 1: SR 기대 비트 길이
    p_xc = p_xc_true(P_Xc_given_A, pA)
    lower, upper = expected_sr_bitlength_bounds(P_Xc_given_A, pA)
    print()
    print("(5) Shannon Coding (Theorem 1)  SR 기대 비트 길이 L_S1")
    print("    pXc(TRUE) = sum_a pXc|A(TRUE|a)*pA(a):", np.round(p_xc, 4).tolist())
    print("    하한(식4) <= L_S1 <= 상한(식5):  %.4f <= L_S1 <= %.4f (bits)" % (lower, upper))
    return P_Xc_given_A, pC_A, pA_C, c2s_map, s2c_map


if __name__ == "__main__":
    example_system1_full()
