"""
Fig 6 reproduction (paper Section IV): Reliability γ versus communication rounds in System 2 SNC.

- x-axis: Communication Rounds (0~5), y-axis: Reliability γ (0~1).
- Three subplots (horizontal): (a) r=20, (b) r=100, (c) r=200.
- In each subplot, three curves for α,β = 1.1, 1.5, 2.0.
- Paper setup: |A|=|C|=100, λ=0.5, pA/pC uniform.
- x-axis round m is "cumulative communication rounds"; each point is γ when K=m.
- Default: (fast and stable) action-exact reliability + Beta(0.1, 0.1) world generation.
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
    World generation for Fig 6.
    - use_beta_pair=True: sample each (a,c) from Beta(0.1,0.1) (paper's "hyperparameter pair" interpretation).
    - use_beta_pair=False: use existing world.generate_experiment_world(Dirichlet row).
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
    For each communication round m=1..n_rounds_max,
    compute reliability γ with System 2 set to K=m.
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
    Fig 6: Reliability γ vs Communication Rounds (0~n_rounds_max).
    Subplots for r=20, 100, 200; curves for α,β=1.1/1.5/2.0.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot Fig 6: pip install matplotlib")

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

    # Styles: α,β=1.1 gold (same as fig7), 1.5 cyan dotted square, 2.0 blue dash-dot triangle
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
