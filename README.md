# Hill Mixture MMM Paper

## Overview

This folder contains the preprint manuscript on a Bayesian **predictive mixture** of Hill saturation functions for Marketing Mix Modeling, implemented in the parent repository. The canonical text is `main.tex`.

## Scope

The paper is a preprint that reports what we tested and what we did not. The empirical scope is deliberately narrow:

- All experiments (synthetic and real-data) use a **single aggregated spend proxy** `x_t = sum_c x_{c,t}` rather than channel-level mixtures. The mixture is over saturation curves on that one-dimensional aggregate, not within each channel.
- The model is a **predictive mixture over the observation likelihood**, not a model of latent consumer segments. The aggregated daily resolution of standard MMM cannot identify individual-level structure, and we do not claim it does.
- The real-data evaluation covers **three organisations from three verticals** of one e-commerce dataset (Conjura). This is enough to expose the synthetic-versus-real diagnostic gap but is not a cross-organisation generalisation study.

The Discussion section is explicitly structured as **What worked / What did not transfer / What we did not test**, and the Scope and Limitations section consolidates the empirical and interpretive scope choices.

## What is in the manuscript

1. A controlled synthetic benchmark (3 DGPs × 3 models × 5 seeds = 45 fits) where the DGP is itself a Hill mixture.
2. A component-resolvability sweep (9 profiles × 2 mixture models × 5 seeds = 90 fits) characterising recovery as a function of true cosine separation between component curves.
3. A real-data benchmark on three Conjura organisations (3 orgs × 3 models × 3 seeds = 27 fits), paired with a re-use of the resolvability axis on the real posteriors.

## Relationship to Code

| Paper section | Implementation |
|---|---|
| Model & diagnostic toolkit (§3) | `../src/hill_mixture_mmm/models.py` |
| Data generation (§5.1, Appendix C) | `../src/hill_mixture_mmm/data.py` |
| Inference and diagnostics (§3.2, §4) | `../src/hill_mixture_mmm/inference.py` |
| Synthetic recovery benchmark (§5.1) | `../scripts/run_benchmark.py` |
| Component resolvability (§5.2) | `../scripts/run_component_resolvability_sweep.py` |
| Real-data benchmark (§6) | `../scripts/run_real_data_validation.py`, `../scripts/summarize_real_benchmark.py` |
| Posterior cosine separation (§6.3) | `../scripts/compute_posterior_separation.py` |
| Paper figures | `../src/hill_mixture_mmm/paper_figures.py`, `../scripts/build_real_paper_figures.py` |

## Files

| File | Description |
|---|---|
| `main.tex` | Canonical manuscript source (preprint) |
| `main.pdf` | Compiled PDF (rebuilt locally) |
| `references.bib` | BibTeX bibliography |
| `figures/` | Per-fit JSON summaries and rendered figures |

## Compilation

Requirements: TeX Live or MacTeX with `pdflatex` and `bibtex`.

    latexmk -pdf main.tex

or equivalently

    pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
