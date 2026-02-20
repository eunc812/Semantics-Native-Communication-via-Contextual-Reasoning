"""
Fig 4 reproduction (paper Section IV).

- Fig 4(a): G convergence curve. |A|=|C|=100, λ=0.5, α=β=1.1/1.5/2.
- Fig 4(b): stationary rA2C heatmap. |E|=|A|=10, |C|=20, β=1.5, α=1.1/1.5/2.
  "empirical distribution of chosen meaningful concept", 0=black/1=light beige.
Uses only existing system2 functions.
"""

import numpy as np
from typing import Optional, List, Tuple

__all__ = ["plot_fig4", "plot_fig4a", "plot_fig4b", "verify_fig4b_concentration"]

# Fig 4(b) fixed seed for reproduction (strong alpha=2.0 concentration in raw)
FIG4_FIXED_SEED = 10820
FIG4A_FIXED_SEED = 42

# Fig 4(b) reproduction parameters
FIG4B_DIRICHLET_ALPHA = 0.08
FIG4B_MAX_ITER = 3600
FIG4B_TOL = 1e-9


def _build_fig4b_world(
    n_entities: int,
    n_concepts: int,
    rng: np.random.Generator,
    dirichlet_alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build S0, L0 for 4(b). Returns (S0, L0, pC_A, pA)."""
    from system1 import a2c_from_P_Xc_given_A, c2a_from_a2c_and_prior
    from system2 import build_individual_contexts

    alpha_vec = np.full(n_concepts, dirichlet_alpha)
    P = np.stack([rng.dirichlet(alpha_vec) for _ in range(n_entities)], axis=0)
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    pA = np.ones(n_entities) / n_entities
    pC = np.ones(n_concepts) / n_concepts
    pC_A = a2c_from_P_Xc_given_A(P)
    pA_C = c2a_from_a2c_and_prior(pC_A, pA)
    S0, L0 = build_individual_contexts(pC_A, pA, pA_C, pC)
    return S0, L0, pC_A, pA


def _entity_row_order_for_fig4b(rA2C_alpha2: np.ndarray) -> np.ndarray:
    """
    Row ordering for readability of trend while keeping original probability values.
    At α=2.0: dominant concept (argmax) ascending, within same group row-max descending.
    """
    dominant = np.argmax(rA2C_alpha2, axis=1)
    confidence = np.max(rA2C_alpha2, axis=1)
    return np.lexsort((-confidence, dominant))


def verify_fig4b_concentration(
    rA2C_list: List[np.ndarray],
    alphas: List[float],
    min_mean_row_max: float = 0.45,
    min_rows_dominant: int = 6,
    dominant_thresh: float = 0.5,
    min_increase: float = 0.03,
    min_last_mean_row_max: Optional[float] = None,
    min_rows_confirmed_at_last: Optional[int] = None,
    confirmed_thresh: float = 0.58,
) -> Tuple[bool, str]:
    """
    Verify that rows concentrate on one concept as α increases.
    - min_last_mean_row_max: lower bound on mean(row_max) at max α.
    - min_rows_confirmed_at_last: at α=2.0, lower bound on number of rows with row_max >= confirmed_thresh (entity-concept confirmed).
    Returns: (pass/fail, summary string).
    """
    if len(rA2C_list) != len(alphas):
        return False, "len(rA2C_list) != len(alphas)"
    row_maxes = [np.max(r, axis=1) for r in rA2C_list]
    mean_max = [np.mean(rm) for rm in row_maxes]
    if not all(mean_max[i] <= mean_max[i + 1] + 1e-6 for i in range(len(mean_max) - 1)):
        return False, "mean(row_max) not increasing with alpha: " + str(mean_max)
    if mean_max[-1] - mean_max[0] < min_increase:
        return False, "mean(row_max) spread too small [{:.3f}..{:.3f}] need >= {:.2f}".format(
            mean_max[0], mean_max[-1], min_increase
        )
    if min_last_mean_row_max is not None and mean_max[-1] < min_last_mean_row_max:
        return False, "mean(row_max) at max alpha = {:.3f} < {:.2f}".format(
            mean_max[-1], min_last_mean_row_max
        )
    if min_rows_confirmed_at_last is not None:
        n_confirmed = int(np.sum(row_maxes[-1] >= confirmed_thresh))
        if n_confirmed < min_rows_confirmed_at_last:
            return False, "at alpha=2.0 rows with max>={:.2f} = {} < {}".format(
                confirmed_thresh, n_confirmed, min_rows_confirmed_at_last
            )
    mid_idx = 1 if len(alphas) == 3 else 0
    r_mid = rA2C_list[mid_idx]
    row_max_mid = np.max(r_mid, axis=1)
    n_dominant = int(np.sum(row_max_mid >= dominant_thresh))
    if np.mean(row_max_mid) < min_mean_row_max:
        return False, "mean(row_max) at alpha=1.5 = {:.3f} < {:.2f}".format(
            float(np.mean(row_max_mid)), min_mean_row_max
        )
    if n_dominant < min_rows_dominant:
        return False, "rows with max>=0.5 at alpha=1.5 = {} < {}".format(
            n_dominant, min_rows_dominant
        )
    n_conf = int(np.sum(row_maxes[-1] >= confirmed_thresh)) if min_rows_confirmed_at_last is not None else 0
    return True, "ok mean(row_max)=[{}] rows_dom={} confirmed@2.0={}".format(
        ", ".join("{:.3f}".format(m) for m in mean_max), n_dominant, n_conf
    )


def plot_fig4(
    n_actions_4a: int = 100,
    n_concepts_4a: int = 100,
    n_entities_4b: int = 10,
    n_concepts_4b: int = 20,
    lam: float = 0.5,
    max_iter: int = 150,
    save_path: Optional[str] = None,
) -> None:
    """
    Paper Fig 4 in one window: (a) left G convergence curve, (b) right rA2C heatmap.
    Title "Fig. 4.", (a), (b) below each block.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError("matplotlib is required to plot Fig 4: pip install matplotlib")

    from system1 import (
        build_P_Xc_given_A,
        a2c_from_P_Xc_given_A,
        c2a_from_a2c_and_prior,
    )
    from system2 import (
        build_individual_contexts,
        minimize_G_alternating,
        objective_G,
        run_self_snc,
    )

    alphas_betas = [(1.1, 1.1), (1.5, 1.5), (2.0, 2.0)]
    alphas_4b = [1.1, 1.5, 2.0]
    beta_4b = 1.5
    fixed_seed = FIG4_FIXED_SEED
    max_iter_4b = FIG4B_MAX_ITER
    tol_4b = FIG4B_TOL
    dirichlet_4b = FIG4B_DIRICHLET_ALPHA
    rng_4a = np.random.default_rng(FIG4A_FIXED_SEED)
    rng_4b = np.random.default_rng(fixed_seed)

    # (a) single fixed experiment world
    P4a = build_P_Xc_given_A(n_actions_4a, n_concepts_4a, rng=rng_4a)
    pA4a = np.ones(n_actions_4a) / n_actions_4a
    pC4a = np.ones(n_concepts_4a) / n_concepts_4a
    pC_A_4a = a2c_from_P_Xc_given_A(P4a)
    pA_C_4a = c2a_from_a2c_and_prior(pC_A_4a, pA4a)
    S0_4a, L0_4a = build_individual_contexts(pC_A_4a, pA4a, pA_C_4a, pC4a)

    # (b) single fixed experiment world (raw rA2C, no post-processing)
    S0_4b, L0_4b, _, _ = _build_fig4b_world(
        n_entities_4b, n_concepts_4b, rng_4b, dirichlet_alpha=dirichlet_4b
    )
    rA2C_list = []
    for a in alphas_4b:
        _, _, _, rA2C, _, _ = run_self_snc(
            S0_4b, L0_4b, lam, a, beta_4b, max_iter=max_iter_4b, tol=tol_4b
        )
        rA2C_list.append(rA2C)
    row_order = _entity_row_order_for_fig4b(rA2C_list[-1])
    rA2C_list = [r[row_order, :] for r in rA2C_list]
    row_maxes = [np.max(r, axis=1) for r in rA2C_list]
    mean_row_max = [float(np.mean(rm)) for rm in row_maxes]
    spread = mean_row_max[-1] - mean_row_max[0]
    n_conf55 = int(np.sum(row_maxes[-1] >= 0.55))
    n_conf99 = int(np.sum(row_maxes[-1] >= 0.99))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    ax_4a = [axes[i, 0] for i in range(3)]
    ax_4b = [axes[i, 1] for i in range(3)]

    # (a) G convergence
    for idx, (alpha, beta) in enumerate(alphas_betas):
        _, _, _, history, _, _ = minimize_G_alternating(
            S0_4a, L0_4a, lam, alpha, beta, max_iter=max_iter, tol=1e-9,
            history_scale="joint",
        )
        M0 = lam * S0_4a + (1.0 - lam) * L0_4a
        s0, l0, m0 = max(S0_4a.sum(), 1e-12), max(L0_4a.sum(), 1e-12), max(M0.sum(), 1e-12)
        g0 = objective_G(S0_4a / s0, L0_4a / l0, M0 / m0, lam, alpha, beta)
        history = [g0] + list(history)
        if len(history) < max_iter + 1:
            history = history + [history[-1]] * (max_iter + 1 - len(history))
        steps = np.arange(0, max_iter + 1, dtype=int)
        ax_4a[idx].plot(steps, history[: max_iter + 1], color="C0", solid_capstyle="round", solid_joinstyle="round")
        ax_4a[idx].set_ylabel("Objective (G)")
        ax_4a[idx].set_title(r"$\alpha$, $\beta$ = {}".format(alpha))
        ax_4a[idx].grid(True, alpha=0.3)
        ax_4a[idx].set_xlim(0, max_iter)
    ax_4a[2].set_xlabel("Iteration steps")

    # (b) heatmap — paper colors, raw rA2C as-is.
    cmap_4b = LinearSegmentedColormap.from_list(
        "fig4b_paper", ["#000000", "#5C4033", "#C4A484", "#F5DEB3"], N=256
    )
    im = None
    for idx, alpha in enumerate(alphas_4b):
        rA2C = rA2C_list[idx]
        im = ax_4b[idx].imshow(rA2C, aspect="auto", cmap=cmap_4b, vmin=0.0, vmax=1.0, origin="lower")
        ax_4b[idx].set_ylabel("Entities")
        ax_4b[idx].set_title(r"$\alpha$ = {}, $\beta$ = 1.5".format(alpha))
        ax_4b[idx].set_xticks(np.linspace(0, n_concepts_4b - 1, 5).astype(int))
        ax_4b[idx].set_xticklabels([1, 5, 10, 15, 20])
        ax_4b[idx].set_yticks(np.linspace(0, n_entities_4b - 1, 5).astype(int))
        ax_4b[idx].set_yticklabels([1, 3, 5, 7, 10])
    ax_4b[2].set_xlabel("Concepts")

    fig.suptitle("Fig. 4", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 0.88, 0.98])
    # (a), (b) labels — below each block
    fig.text(0.28, 0.01, "(a)", ha="center", fontsize=12)
    fig.text(0.72, 0.01, "(b)", ha="center", fontsize=12)
    if im is not None:
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cax, label="Probability")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(
        "Fig4 trend seed(4b)={}: mean(row_max)=[{}], spread={:.4f}, rows@a2>=0.55={}/{}, rows@a2>=0.99={}/{}".format(
            fixed_seed,
            ", ".join(f"{m:.4f}" for m in mean_row_max),
            spread,
            n_conf55,
            n_entities_4b,
            n_conf99,
            n_entities_4b,
        )
    )
    plt.show()


def plot_fig4a(
    n_actions: int = 100,
    n_concepts: int = 100,
    lam: float = 0.5,
    max_iter: int = 150,
    alphas_betas: Optional[List[Tuple[float, float]]] = None,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Fig 4(a): G convergence curve. Subplots for α=β=1.1, 1.5, 2.
    x-axis: Iteration steps (1 ~ max_iter), y-axis: Objective G.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot Fig 4(a): pip install matplotlib")

    from system1 import (
        build_P_Xc_given_A,
        a2c_from_P_Xc_given_A,
        c2a_from_a2c_and_prior,
    )
    from system2 import build_individual_contexts, minimize_G_alternating, objective_G

    if alphas_betas is None:
        alphas_betas = [(1.1, 1.1), (1.5, 1.5), (2.0, 2.0)]

    rng = np.random.default_rng(seed)
    P_Xc_given_A = build_P_Xc_given_A(n_actions, n_concepts, rng=rng)
    pA = np.ones(n_actions) / n_actions
    pC = np.ones(n_concepts) / n_concepts
    pC_A = a2c_from_P_Xc_given_A(P_Xc_given_A)
    pA_C = c2a_from_a2c_and_prior(pC_A, pA)
    S0, L0 = build_individual_contexts(pC_A, pA, pA_C, pC)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for idx, (alpha, beta) in enumerate(alphas_betas):
        _, _, _, history, _, _ = minimize_G_alternating(
            S0, L0, lam, alpha, beta, max_iter=max_iter, tol=1e-9,
            history_scale="joint",
        )
        # step 0 = initial G (paper form). Normalize S0,L0,M0 as joint then G
        M0 = lam * S0 + (1.0 - lam) * L0
        s0, l0, m0 = max(S0.sum(), 1e-12), max(L0.sum(), 1e-12), max(M0.sum(), 1e-12)
        g0 = objective_G(S0 / s0, L0 / l0, M0 / m0, lam, alpha, beta)
        history = [g0] + list(history)
        if len(history) < max_iter + 1:
            history = history + [history[-1]] * (max_iter + 1 - len(history))
        steps = np.arange(0, max_iter + 1, dtype=int)
        axes[idx].plot(
            steps,
            history[: max_iter + 1],
            color="C0",
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axes[idx].set_ylabel("Objective (G)")
        axes[idx].set_title(r"$\alpha$, $\beta$ = {}".format(alpha))
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, max_iter)
    axes[-1].set_xlabel("Iteration steps")
    fig.suptitle("Fig. 4(a)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fig4b(
    n_entities: int = 10,
    n_concepts: int = 20,
    lam: float = 0.5,
    beta: float = 1.5,
    alphas: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Fig 4(b): stationary rA2C heatmap (empirical distribution of chosen meaningful concept).
    α=1.1, 1.5, 2.0 (β=1.5 fixed), |E|=10, |C|=20.
    Colors: 0=black, 1=light beige. Colorbar placed on right so it does not overlap figure.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError("matplotlib is required to plot Fig 4(b): pip install matplotlib")

    from system2 import run_self_snc

    if alphas is None:
        alphas = [1.1, 1.5, 2.0]

    # Single experiment with fixed seed, no post-processing; trend clearly visible.
    fixed_seed = FIG4_FIXED_SEED
    max_iter_4b = FIG4B_MAX_ITER
    tol_4b = FIG4B_TOL
    dirichlet_4b = FIG4B_DIRICHLET_ALPHA
    rng = np.random.default_rng(fixed_seed)
    S0, L0, _, _ = _build_fig4b_world(
        n_entities, n_concepts, rng, dirichlet_alpha=dirichlet_4b
    )
    best_rA2C_list = []
    for alpha in alphas:
        _, _, _, rA2C, _, _ = run_self_snc(
            S0, L0, lam, alpha, beta, max_iter=max_iter_4b, tol=tol_4b
        )
        best_rA2C_list.append(rA2C)
    row_order = _entity_row_order_for_fig4b(best_rA2C_list[-1])
    best_rA2C_list = [r[row_order, :] for r in best_rA2C_list]
    row_maxes = [np.max(r, axis=1) for r in best_rA2C_list]
    mean_row_max = [float(np.mean(rm)) for rm in row_maxes]
    spread = mean_row_max[-1] - mean_row_max[0]
    n_conf55 = int(np.sum(row_maxes[-1] >= 0.55))
    n_conf99 = int(np.sum(row_maxes[-1] >= 0.99))
    print(
        "Fig4(b) trend seed={}: mean(row_max)=[{}], spread={:.4f}, rows@a2>=0.55={}/{}, rows@a2>=0.99={}/{}".format(
            fixed_seed,
            ", ".join(f"{m:.4f}" for m in mean_row_max),
            spread,
            n_conf55,
            n_entities,
            n_conf99,
            n_entities,
        )
    )

    cmap = LinearSegmentedColormap.from_list(
        "fig4b", ["#000000", "#5C4033", "#C4A484", "#F5DEB3"], N=256
    )
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True, sharey=True)
    im = None
    for idx, alpha in enumerate(alphas):
        rA2C = best_rA2C_list[idx]
        im = axes[idx].imshow(
            rA2C,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )
        axes[idx].set_ylabel("Entities")
        axes[idx].set_xlabel("Concepts" if idx == 2 else "")
        axes[idx].set_title(r"$\alpha$ = {}, $\beta$ = 1.5".format(alpha))
        axes[idx].set_xticks(np.linspace(0, n_concepts - 1, 5).astype(int))
        axes[idx].set_xticklabels([1, 5, 10, 15, 20])
        axes[idx].set_yticks(np.linspace(0, n_entities - 1, 5).astype(int))
        axes[idx].set_yticklabels([1, 3, 5, 7, 10])

    fig.suptitle("Fig.4. (b)")
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    if im is not None:
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Probability")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    base = Path(__file__).parent
    print("Fig 4: (a) G convergence, (b) rA2C heatmap")
    plot_fig4(save_path=str(base / "fig4.png"))
