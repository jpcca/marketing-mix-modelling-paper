# Hill Mixture MMM Paper

## Overview

This folder contains the research paper documenting the Bayesian Hill Mixture Model for Marketing Mix Modeling implemented in this repository.

## Goal

Standard Marketing Mix Models assume all consumers respond identically to marketing spend. This paper proposes a **mixture of Hill saturation functions** to capture heterogeneous response patterns across latent consumer segments, addressing:

- **Segment heterogeneity**: Different consumer groups (heavy vs. light buyers, loyalists vs. switchers) exhibit different saturation curves
- **Model misspecification**: Aggregate response curves mask segment-specific behaviors
- **Identifiability**: Label-invariant diagnostics and post-hoc relabeling make mixture posteriors interpretable

## Key Contributions

1. **Bayesian mixture framework** combining Hill saturation with latent segment membership
2. **Automatic prior scaling** from training data for diverse scales
3. **Sequential holdout evaluation** that preserves adstock state and absolute time index
4. **NumPyro/JAX implementation** for practical Bayesian inference

## Relationship to Code

| Paper Section | Implementation |
|---------------|----------------|
| Model Specification (§2) | `../src/hill_mixture_mmm/models.py` |
| Data Generation (§3) | `../src/hill_mixture_mmm/data.py` |
| Inference and diagnostics | `../src/hill_mixture_mmm/inference.py` |
| Benchmarking | `../scripts/run_benchmark.py` |

## Files

| File | Description |
|------|-------------|
| `main.tex` | Canonical manuscript source |
| `hill_mixture_mmm_paper.md` | Legacy Markdown draft retained for reference |
| `references.bib` | BibTeX bibliography |
| `main.pdf` | Compiled output (when built locally) |

## Compilation

### Requirements

- [Pandoc](https://pandoc.org/installing.html) (2.11+)
- XeLaTeX (via TeX Live, MacTeX, or MiKTeX)

### Build PDF

    latexmk -pdf main.tex
