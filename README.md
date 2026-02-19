
# SNC — Semantics-Native Communication via Contextual Reasoning
Implementation and experiment code for **Semantics-Native Communication via Contextual Reasoning**.  
Includes semantics-based communication models (System 1 / System 2) using action – concept relations and related experiments.

## Structure

- **`system1.py`** — System 1 SNC: Action–Concept Relevance, A2C/C2A, Concept–Symbol mapping, Theorem 1 (SR bit-length bounds)
- **`system2.py`** — System 2 SNC: Individual context (S, L), objective G, Self-SNC, Algorithm 1
- **`experiments/`**
  - `world.py` — World generation for Section IV experiments (|A|=|C|=100, Dirichlet, etc.)
  - `reliability.py` — System 1 vs System 2 reliability γ (Monte Carlo)
  - `fig4.py`–`fig8.py` — Scripts to reproduce paper Figures 4–8

## Requirements

```
numpy
matplotlib
```

## Usage

- Import `system1` and `system2` from the project root.
- experiments/plots: run scripts under `experiments/` (e.g. `python experiments/fig4.py`).

## Paper

*[Semantics-Native Communication via Contextual Reasoning](https://ieeexplore.ieee.org/document/10054510)* 
