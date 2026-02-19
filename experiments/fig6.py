"""
Fig 6 재현 (논문 Section IV): Reliability γ versus communication rounds in System 2 SNC.

- x축: Communication Rounds (0~5), y축: Reliability γ (0~1).
- 서브플롯 3개 (가로): (a) r=20, (b) r=100, (c) r=200.
- 각 서브플롯에서 α,β = 1.1, 1.5, 2.0 세 곡선.
- 논문 설정: |A|=|C|=100, λ=0.5, pA/pC uniform.
- x축 라운드 m은 "누적 통신 라운드 수"로 해석하여, 각 점은 K=m일 때의 γ를 계산.
- 기본값은 (빠르고 안정적인) action-exact reliability + Beta(0.1, 0.1) 월드 생성 사용.
"""

import numpy as np
from typing import Optional, List, Tuple

__all__ = ["plot_fig6"]

World = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _generate_paper_like_world(
    n_actions: int,
    n_concepts: int,
    seed: Optional[int],
    use_beta_pair: bool,
) -> World:
    """
    Fig 6용 월드 생성.
    - use_beta_pair=True: 각 (a,c)를 Beta(0.1,0.1)로 샘플 (논문의 "hyperparameter pair" 해석).
    - use_beta_pair=False: 기존 world.generate_experiment_world(Dirichlet row) 사용.
    """
    if use_beta_pair:
        rng = np.random.default_rng(seed)
        P_Xc_given_A = rng.beta(0.1, 0.1, size=(n_actions, n_concepts))
        P_Xc_given_A = np.clip(P_Xc_given_A, 1e-12, 1.0 - 1e-12)
        pA = np.ones(n_actions, dtype=float) / float(n_actions)
        pC = np.ones(n_concepts, dtype=float) / float(n_concepts)
        return P_Xc_given_A, pA, pC

    from experiments.world import generate_experiment_world
    return generate_experiment_world(n_actions=n_actions, n_concepts=n_concepts, seed=seed)


def _reliability_vs_rounds(
    world: World,
    n_rounds_max: int,
    r: int,
    alpha: float,
    beta: float,
    lam: float = 0.5,
    use_exact: bool = True,
    n_rounds_reliability: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    통신 라운드 m=1..n_rounds_max 각각에 대해,
    System 2에서 K=m으로 설정한 reliability γ를 계산.
    """
    from experiments.reliability import (
        compute_reliability_system2,
        compute_reliability_system2_by_rounds_exact,
    )

    if use_exact:
        return compute_reliability_system2_by_rounds_exact(
            world=world,
            K_max=n_rounds_max,
            lam=lam,
            alpha=alpha,
            beta=beta,
            max_inner_iter=r,
        )

    gamma = np.zeros(n_rounds_max, dtype=float)
    for m in range(1, n_rounds_max + 1):
        gamma[m - 1] = compute_reliability_system2(
            world=world,
            K=m,
            n_rounds=n_rounds_reliability,
            seed=None if seed is None else int(seed + 1000 * r + m),
            lam=lam,
            alpha=alpha,
            beta=beta,
            max_inner_iter=r,
        )
    return gamma


def plot_fig6(
    world: Optional[World] = None,
    n_actions: int = 100,
    n_concepts: int = 100,
    lam: float = 0.5,
    n_rounds_max: int = 5,
    r_values: Optional[List[int]] = None,
    alpha_beta_values: Optional[List[Tuple[float, float]]] = None,
    n_rounds_reliability: int = 100,
    use_exact_reliability: bool = True,
    use_beta_pair_world: bool = True,
    seed: Optional[int] = None,
    world_seed: Optional[int] = 42,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Fig 6: Communication Rounds (0~n_rounds_max)에 따른 Reliability γ.
    r=20, 100, 200 각각 서브플롯, α,β=1.1/1.5/2.0 곡선.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Fig 6를 그리려면 matplotlib이 필요합니다: pip install matplotlib")

    if world is None:
        P_Xc_given_A, pA, pC = _generate_paper_like_world(
            n_actions=n_actions,
            n_concepts=n_concepts,
            seed=world_seed,
            use_beta_pair=use_beta_pair_world,
        )
        world = (P_Xc_given_A, pA, pC)
    if r_values is None:
        r_values = [20, 100, 200]
    if alpha_beta_values is None:
        alpha_beta_values = [(1.1, 1.1), (1.5, 1.5), (2.0, 2.0)]

    P_Xc_given_A, _, _ = world
    n_actions_w, n_concepts_w = P_Xc_given_A.shape

    # 스타일: α,β=1.1 골드(fig7과 동일), 1.5 시안 점선 사각, 2.0 파랑 대시-점 삼각
    styles = [
        {"color": "#E5A800", "linestyle": "-", "marker": "o", "markersize": 6},
        {"color": "cyan", "linestyle": ":", "marker": "s", "markersize": 5},
        {"color": "C0", "linestyle": "-.", "marker": "^", "markersize": 6},
    ]

    n_panels = len(r_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]
    rounds_axis = np.arange(0, n_rounds_max + 1, dtype=int)  # 0, 1, 2, 3, 4, 5

    for ax_idx, r in enumerate(r_values):
        ax = axes[ax_idx]
        for ab_idx, (alpha, beta) in enumerate(alpha_beta_values):
            gamma_vs_rounds = _reliability_vs_rounds(
                world=world,
                n_rounds_max=n_rounds_max,
                r=r,
                alpha=alpha,
                beta=beta,
                lam=lam,
                use_exact=use_exact_reliability,
                n_rounds_reliability=n_rounds_reliability,
                seed=seed,
            )
            # (0,0) + (1, γ_1), (2, γ_2), ... , (5, γ_5)
            x_plot = np.arange(1, n_rounds_max + 1, dtype=int)
            y_plot = gamma_vs_rounds
            ax.plot(
                [0] + list(x_plot),
                [0.0] + list(y_plot),
                **styles[ab_idx],
                label=r"$\alpha,\ \beta$ = {}".format(alpha),
            )
        ax.set_xlabel("Communication Rounds")
        if ax_idx == 0:
            ax.set_ylabel("Reliability " + r"$\gamma$")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.2, n_rounds_max + 0.2)
        ax.set_xticks(rounds_axis)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        panel_label = chr(ord("a") + ax_idx)
        ax.text(0.5, -0.32, "({}) r={}".format(panel_label, r), transform=ax.transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle("Fig. 6", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0.00, 0.12, 1.00, 0.93])
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

    out = Path(__file__).parent / "fig6.png"
    print("Fig 6: Reliability γ vs Communication Rounds. |A|=|C|=100, λ=0.5, r=20/100/200.")
    plot_fig6(
        n_actions=100,
        n_concepts=100,
        lam=0.5,
        n_rounds_max=5,
        n_rounds_reliability=100,
        use_exact_reliability=True,
        use_beta_pair_world=True,
        world_seed=42,
        seed=0,
        save_path=str(out),
        show=False,
    )
