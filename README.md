# Semantics-Native Communication via Contextual Reasoning (SNC)
+ Implementation and experiment code for *[Semantics-Native Communication via Contextual Reasoning](https://ieeexplore.ieee.org/document/10054510)* 
+ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A8h6a8rotjgW3andTgensdn252n_MNPC?usp=sharing)
## Architecture
![Architecture](results/architecture.jpg)

## Structure

- **`system1.py`** — system 1 SNC: action–concept relevance, A2C/C2A, concept–symbol mapping, Theorem 1 (SR bit-length bounds)
- **`system2.py`** — system 2 SNC: individual context (S, L), objective G, self-SNC, Algorithm 1
- **`experiments/`**
  - `world.py` — world generation for section IV experiments (|A|=|C|=100, Dirichlet)
  - `reliability.py` — system 1 vs system 2 reliability γ
  - `fig4.py`–`fig8.py` — scripts to reproduce paper Figures 4–8
- **`resulsts/`** - architecture, output figures from the experiment

## Results


| Figure | Description |
|--------|-------------|
| [Fig 4](results/fig4.png) | G convergence and stationary rA2C heatmap |
| [Fig 5](results/fig5.png) | Reliability γ vs α, β (3D surface; r = 10, 20, 100) |
| [Fig 6](results/fig6.png) | Reliability γ vs communication rounds |
| [Fig 7](results/fig7.png) | SR length achieving γ = 1 (System 1 vs System 2) |
| [Fig 8](results/fig8.png) | Robustness to asynchronous context (γ vs perturbation ε) |


Preview:

![Figure 4](results/fig4.png)
![Figure 5](results/fig5.png)
![Figure 6](results/fig6.png)
![Figure 7](results/fig7.png)
![Figure 8](results/fig8.png)
