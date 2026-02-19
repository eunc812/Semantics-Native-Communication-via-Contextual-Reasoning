"""
Fig 8 재현 (논문 Section IV): Robustness to Asynchronous Contextual Reasoning in System 2 SNC.

- x축: Perturbation ε (0 ~ 0.05), y축: Reliability γ (0 ~ 1).
- 두 곡선:
    (1) A2C and C2A without Quantization (blue, dash-dot, triangle)
    (2) A2C and C2A with Quantization    (yellow/gold, solid, circle)
- 비동기 모델: speaker와 listener가 서로 다른 P_Xc_given_A로부터 A2C/C2A를 도출하여
  각각 self-SNC를 독립 초기화. 논문: "speaker and listener having different A2C and C2A
  when initializing their self-SNC procedure."
- Perturbation: P_Xc_given_A 각 성분에 uniform[-ε,+ε] 추가 후 [0,1] clip.
- Quantization: P_Xc_given_A를 소수점 둘째 자리에서 반올림(np.round(P, 1)) 후
  A2C/C2A 도출. Beta(0.1,0.1) 분포에서 대부분의 값이 0 또는 1 근처이므로
  ε ≤ 0.05 범위의 perturbation이 동일 양자화 bin에 유지되어 robustness 확보.
- 논문 설정: |A|=|C|=100, λ=0.5, α=1.1, β=1.5, K=3 communication rounds.
"""

import numpy as np
from typing import Optional, List, Tuple

__all__ = ["plot_fig8"]

World = Tuple[np.ndarray, np.ndarray, np.ndarray]
_EPS = 1e-12


def _generate_beta_world(
    n_actions: int = 100,
    n_concepts: int = 100,
    seed: Optional[int] = 42,
) -> World:
    """논문 hyperparameter pair (0.1, 0.1) → Beta(0.1, 0.1) 월드."""
    rng = np.random.default_rng(seed)
    P = rng.beta(0.1, 0.1, size=(n_actions, n_concepts))
    P = np.clip(P, _EPS, 1.0 - _EPS)
    pA = np.ones(n_actions, dtype=float) / n_actions
    pC = np.ones(n_concepts, dtype=float) / n_concepts
    return P, pA, pC


def _perturb_P(P: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    if epsilon <= 0.0:
        return P.copy()
    return np.clip(P + rng.uniform(-epsilon, epsilon, size=P.shape), _EPS, 1.0 - _EPS)


def _quantize_P(P: np.ndarray, decimals: int = 1) -> np.ndarray:
    return np.clip(np.round(P, decimals), _EPS, 1.0 - _EPS)


def _async_algorithm1_infer(
    P_spk: np.ndarray,
    P_lst: np.ndarray,
    pA_init: np.ndarray,
    pC_init: np.ndarray,
    a_hat: int,
    K: int,
    lam: float,
    alpha: float,
    beta: float,
    max_iter: int,
    tol: float = 1e-9,
) -> int:
    """
    비동기 Algorithm 1: speaker/listener가 서로 다른 P로부터 A2C/C2A를 도출,
    K 라운드 통신 후 listener 추론 action 반환.
    """
    from system1 import a2c_from_P_Xc_given_A, c2a_from_a2c_and_prior
    from system2 import build_individual_contexts, run_self_snc

    n_actions, n_concepts = P_spk.shape
    pA = pA_init.copy()
    pC = pC_init.copy()
    remaining = np.ones(n_concepts, dtype=bool)

    A2C_s = a2c_from_P_Xc_given_A(P_spk)
    A2C_l = a2c_from_P_Xc_given_A(P_lst)

    for k in range(K):
        C2A_s = c2a_from_a2c_and_prior(A2C_s, pA)
        C2A_l = c2a_from_a2c_and_prior(A2C_l, pA)

        S0s, L0s = build_individual_contexts(A2C_s, pA, C2A_s, pC)
        _, _, _, rA2C_s, _, _ = run_self_snc(S0s, L0s, lam, alpha, beta, max_iter=max_iter, tol=tol)

        S0l, L0l = build_individual_contexts(A2C_l, pA, C2A_l, pC)
        _, _, _, _, rC2A_l, _ = run_self_snc(S0l, L0l, lam, alpha, beta, max_iter=max_iter, tol=tol)

        score = rA2C_s[a_hat, :].copy()
        score[~remaining] = -np.inf
        c_k = int(np.argmax(score))

        pA = rC2A_l[c_k, :].ravel().copy()
        pA = np.maximum(pA, _EPS)
        pA /= pA.sum()

        remaining[c_k] = False
        pC[c_k] = 0.0
        if remaining.any():
            pC[remaining] /= max(pC[remaining].sum(), _EPS)

    return int(np.argmax(pA))


def _run_trial_pair(
    P_true: np.ndarray,
    pA: np.ndarray,
    pC: np.ndarray,
    epsilon: float,
    K: int,
    n_sample_actions: int,
    rng: np.random.Generator,
    lam: float,
    alpha: float,
    beta: float,
    max_iter: int,
) -> Tuple[float, float]:
    """
    단일 trial: 동일 perturbation/action 샘플로 (no-quant, quant) gamma 쌍 반환.
    """
    n_actions = P_true.shape[0]
    P_lst_raw = _perturb_P(P_true, epsilon, rng)
    sampled = rng.choice(n_actions, size=n_sample_actions, replace=True, p=pA)

    P_spk_nq = P_true.copy()
    P_lst_nq = P_lst_raw.copy()
    P_spk_q = _quantize_P(P_true)
    P_lst_q = _quantize_P(P_lst_raw)

    correct_nq, correct_q = 0, 0
    for a_hat in sampled:
        a_hat = int(a_hat)
        a_nq = _async_algorithm1_infer(
            P_spk_nq, P_lst_nq, pA, pC, a_hat, K,
            lam, alpha, beta, max_iter,
        )
        if a_nq == a_hat:
            correct_nq += 1
        a_q = _async_algorithm1_infer(
            P_spk_q, P_lst_q, pA, pC, a_hat, K,
            lam, alpha, beta, max_iter,
        )
        if a_q == a_hat:
            correct_q += 1

    return correct_nq / n_sample_actions, correct_q / n_sample_actions


def compute_fig8_data(
    world: World,
    epsilons: List[float],
    K: int = 3,
    n_trials: int = 10,
    n_sample_actions: int = 30,
    lam: float = 0.5,
    alpha: float = 1.1,
    beta: float = 1.5,
    max_iter: int = 100,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fig 8 데이터 계산.
    Returns (gamma_no_quant, gamma_with_quant) 각각 shape (len(epsilons),).
    """
    P_true, pA, pC = world
    rng = np.random.default_rng(seed)
    n_eps = len(epsilons)
    gamma_no_q = np.zeros(n_eps)
    gamma_q = np.zeros(n_eps)

    for ie, eps in enumerate(epsilons):
        sum_nq, sum_q = 0.0, 0.0
        for t in range(n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**31))
            g_nq, g_q = _run_trial_pair(
                P_true, pA, pC, eps, K, n_sample_actions,
                trial_rng, lam, alpha, beta, max_iter,
            )
            sum_nq += g_nq
            sum_q += g_q
        gamma_no_q[ie] = sum_nq / n_trials
        gamma_q[ie] = sum_q / n_trials
        print(
            "  eps={:.4f}  no_quant={:.4f}  quant={:.4f}".format(
                eps, gamma_no_q[ie], gamma_q[ie],
            )
        )

    return gamma_no_q, gamma_q


def plot_fig8(
    world: Optional[World] = None,
    n_actions: int = 100,
    n_concepts: int = 100,
    lam: float = 0.5,
    alpha: float = 1.1,
    beta: float = 1.5,
    K: int = 3,
    max_iter: int = 100,
    epsilons: Optional[List[float]] = None,
    n_trials: int = 10,
    n_sample_actions: int = 30,
    world_seed: Optional[int] = 42,
    seed: Optional[int] = 0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Fig 8: Reliability γ of System 2 SNC with different initializations
    versus perturbed communication context.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Fig 8을 그리려면 matplotlib이 필요합니다: pip install matplotlib")

    if world is None:
        world = _generate_beta_world(n_actions, n_concepts, seed=world_seed)
    if epsilons is None:
        epsilons = [
            0.0, 0.005, 0.01, 0.015, 0.02, 0.025,
            0.03, 0.035, 0.04, 0.045, 0.05,
        ]

    print("Fig 8 데이터 계산 중 (K={}, n_trials={}, n_sample={}, max_iter={})...".format(
        K, n_trials, n_sample_actions, max_iter,
    ))
    gamma_no_q, gamma_q = compute_fig8_data(
        world=world, epsilons=epsilons, K=K,
        n_trials=n_trials, n_sample_actions=n_sample_actions,
        lam=lam, alpha=alpha, beta=beta, max_iter=max_iter, seed=seed,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # fig7 팔레트: gold=#E5A800, deep blue-violet=#2F13B8
    color_q = "#E5A800"
    color_nq = "#2F13B8"

    ax.plot(
        epsilons, gamma_q,
        color=color_q, linestyle="-", marker="o", markersize=7,
        markerfacecolor="none", markeredgewidth=1.5, linewidth=1.6,
        label="A2C and C2A with Quantization",
    )
    ax.plot(
        epsilons, gamma_no_q,
        color=color_nq, linestyle="-.", marker="^", markersize=7,
        markerfacecolor="none", markeredgewidth=1.5, linewidth=1.6,
        label="A2C and C2A without Quantization",
    )

    ax.set_xlabel(r"Perturbation $\epsilon$", fontsize=12)
    ax.set_ylabel(r"Reliability $\gamma$", fontsize=12)
    ax.set_xlim(-0.001, 0.051)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig. 8", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    out = Path(__file__).parent / "fig8.png"
    print("Fig 8: Reliability γ vs Perturbation ε (async self-SNC, ±quantization).")
    print("  K=3 comm rounds, max_iter=100 self-SNC, n_trials=10, n_sample=30.")
    plot_fig8(
        n_actions=100,
        n_concepts=100,
        lam=0.5,
        alpha=1.1,
        beta=1.5,
        K=3,
        max_iter=100,
        n_trials=10,
        n_sample_actions=30,
        world_seed=42,
        seed=0,
        save_path=str(out),
        show=True,
    )
