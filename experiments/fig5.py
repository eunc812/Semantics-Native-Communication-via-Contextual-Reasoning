"""
Fig 5 reproduction (paper Section IV): Reliability γ with respect to α and β in System 2 SNC.

- 3D surface: Reliability γ vs α, β (0.9~2), self-SNC iteration depth r = 10, 20, 100.
- Paper Fig 5 trend: semantic reliability at self-SNC fixed point (r↑ ⇒ γ↑, peak at suitable α·β).
- use_closed_form_gamma=True: closed-form γ based on rA2C·rC2A.
- use_closed_form_gamma=False: γ estimated via Monte Carlo (Algorithm 1 + n_rounds).
"""

import numpy as np
from typing import Optional, List, Tuple

__all__ = ["plot_fig5"]

World = Tuple[np.ndarray, np.ndarray, np.ndarray]
_EPS = 1e-12


def _gamma_semantic_closed_form(
    rA2C: np.ndarray,
    rC2A: np.ndarray,
    pA: np.ndarray,
) -> float:
    """
    Semantic reliability (closed form) defined by rA2C, rC2A and prior pA after self-SNC convergence.
    Definition aligned with paper Fig 5 trend. Same as Colab gamma_semantic_closed_form.
    """
    pA = np.asarray(pA, dtype=np.float64).ravel()
    pA = pA / np.maximum(pA.sum(), _EPS)
    # rC2A: (|C|, |A|), rA2C: (|A|, |C|). For each concept c, listener optimal action a*(c) = argmax_a rC2A[c,a]
    best_a_for_c = np.argmax(rC2A, axis=1)  # (|C|,)
    return float(np.sum(pA[best_a_for_c] * rA2C[best_a_for_c, np.arange(rA2C.shape[1])]))


def _compute_gamma_grid_closed_form(
    world: World,
    S0: np.ndarray,
    L0: np.ndarray,
    pA: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    r: int,
    lam: float,
) -> np.ndarray:
    """Closed-form γ after self-SNC convergence at each (α, β) grid point. shape (len(betas), len(alphas))."""
    from system2 import run_self_snc

    ny, nx = len(betas), len(alphas)
    gamma_grid = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            _, _, _, rA2C, rC2A, _ = run_self_snc(
                S0, L0, lam, float(alphas[j]), float(betas[i]),
                max_iter=r, tol=1e-9,
            )
            gamma_grid[i, j] = _gamma_semantic_closed_form(rA2C, rC2A, pA)
    return gamma_grid


def _compute_gamma_grid(
    world: World,
    alphas: np.ndarray,
    betas: np.ndarray,
    r: int,
    K: int,
    n_rounds: int,
    lam: float,
    seed: Optional[int],
) -> np.ndarray:
    """Monte Carlo γ at each (α, β) grid point. Returns shape (len(betas), len(alphas))."""
    from experiments.reliability import compute_reliability_system2

    ny, nx = len(betas), len(alphas)
    gamma_grid = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            gamma_grid[i, j] = compute_reliability_system2(
                world,
                K,
                n_rounds,
                seed=seed,
                lam=lam,
                alpha=float(alphas[j]),
                beta=float(betas[i]),
                max_inner_iter=r,
            )
    return gamma_grid


def plot_fig5(
    world: Optional[World] = None,
    n_actions: int = 100,
    n_concepts: int = 100,
    lam: float = 0.5,
    K: int = 10,
    n_rounds: int = 200,
    r_values: Optional[List[int]] = None,
    alpha_beta_min: float = 0.9,
    alpha_beta_max: float = 2.0,
    n_grid: int = 21,
    seed: Optional[int] = None,
    world_seed: Optional[int] = 42,
    save_path: Optional[str] = None,
    use_closed_form_gamma: bool = True,
) -> None:
    """
    Fig 5: Reliability γ 3D surface vs α, β (r = 10, 20, 100).

    use_closed_form_gamma=True (default): closed-form γ from self-SNC fixed point rA2C·rC2A → matches paper trend.
    use_closed_form_gamma=False: Monte Carlo (Algorithm 1 + n_rounds) γ.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError("matplotlib is required to plot Fig 5: pip install matplotlib")

    from experiments.world import generate_experiment_world
    from system1 import a2c_from_P_Xc_given_A, c2a_from_a2c_and_prior
    from system2 import build_individual_contexts

    if world is None:
        P_Xc_given_A, pA, pC = generate_experiment_world(
            n_actions=n_actions, n_concepts=n_concepts, seed=world_seed
        )
        world = (P_Xc_given_A, pA, pC)
    if r_values is None:
        r_values = [10, 20, 100]

    P_Xc_given_A, pA, pC = world
    S0, L0 = None, None
    if use_closed_form_gamma:
        pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)
        pA_C = c2a_from_a2c_and_prior(pC_A, pA)
        S0, L0 = build_individual_contexts(pC_A, pA, pA_C, pC)

    alphas = np.linspace(alpha_beta_min, alpha_beta_max, n_grid)
    betas = np.linspace(alpha_beta_min, alpha_beta_max, n_grid)
    Alpha, Beta = np.meshgrid(alphas, betas)
    # Paper-like colormap: low reliability=blue, high reliability=yellow-orange.
    # Dense color stops near top (0.8~1.0) to increase contrast.
    paper_like_cmap = LinearSegmentedColormap.from_list(
        "fig5_paper_like",
        [
            (0.00, "#2438ff"),  # deep blue
            (0.35, "#00a7ff"),  # blue-cyan
            (0.62, "#27c39f"),  # green-cyan
            (0.82, "#f3e54d"),  # yellow
            (1.00, "#f3a323"),  # yellow-orange
        ],
        N=256,
    )
    # Emphasize resolution in high-reliability region (contrast around peak)
    color_norm = PowerNorm(gamma=2.0, vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(14, 4.8))
    for idx, r in enumerate(r_values):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        if use_closed_form_gamma:
            gamma_grid = _compute_gamma_grid_closed_form(
                world, S0, L0, pA, alphas, betas, r=r, lam=lam,
            )
        else:
            gamma_grid = _compute_gamma_grid(
                world, alphas, betas, r=r, K=K, n_rounds=n_rounds,
                lam=lam, seed=seed,
            )
        surf = ax.plot_surface(
            Alpha, Beta, gamma_grid,
            cmap=paper_like_cmap,
            norm=color_norm,
            edgecolor="none",
            antialiased=True,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.set_zlabel("Reliability " + r"$\gamma$")
        ax.set_xlim(alpha_beta_min, alpha_beta_max)
        ax.set_ylim(alpha_beta_min, alpha_beta_max)
        ax.set_zlim(0.0, 1.0)
        # Same as paper: α, β ticks 1, 1.5, 2 / γ ticks 0, 0.5, 1
        ax.set_xticks([1.0, 1.5, 2.0])
        ax.set_yticks([1.0, 1.5, 2.0])
        ax.set_zticks([0.0, 0.5, 1.0])
        # View/axis orientation similar to paper figure
        ax.view_init(elev=26, azim=-60)
        ax.set_box_aspect((1.3, 1.0, 0.9))

    # Simplified title: "Fig. 5" at top
    fig.suptitle("Fig. 5", fontsize=16, y=0.98)
    # r labels at bottom in order (a), (b), (c)
    labels = ["(a) r = {}".format(r_values[0]), "(b) r = {}".format(r_values[1]), "(c) r = {}".format(r_values[2])]
    x_pos = [0.17, 0.50, 0.83]
    for x, lbl in zip(x_pos, labels):
        fig.text(x, 0.03, lbl, ha="center", va="bottom", fontsize=12)

    plt.tight_layout(rect=[0.00, 0.10, 1.00, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    out = Path(__file__).parent / "fig5.png"
    # Default: closed-form γ (self-SNC fixed point) → matches paper trend. use_closed_form_gamma=False for Monte Carlo.
    print("Fig 5: Reliability γ vs α, β (r=10, 20, 100). γ = semantic (self-SNC fixed point).")
    plot_fig5(
        n_actions=100,
        n_concepts=100,
        lam=0.5,
        K=5,
        n_rounds=80,
        n_grid=21,
        alpha_beta_min=0.9,
        alpha_beta_max=2.0,
        world_seed=42,
        seed=0,
        save_path=str(out),
        use_closed_form_gamma=True,
    )
