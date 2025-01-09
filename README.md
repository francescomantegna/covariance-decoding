# covariance-decoding

<p align="justify"> This repository contains the source code for "Covariance-based decoding reveals a category-specific functional connectivity network for imagined visual objects" (Mantegna, Olivetti, Schwedhelm, Baldauf). </p>

# Abstract

<p align="justify"> The coordination of different brain regions is required for the visual imagery of complex objects (e.g., faces and places). Short-range connectivity within sensory areas is necessary to construct the mental image. Long-range connectivity between control and sensory areas is necessary to re-instantiate and maintain the mental image. While dynamic changes in functional connectivity are expected during visual imagery, it is unclear whether a category-specific network exists in which the strength and the spatial destination of the connections vary depending on the imagery target. In this magnetoencephalography study, we used a minimally constrained experimental paradigm wherein imagery categories were prompted using visual word cues only, and we decoded face versus place imagery based on their underlying functional connectivity patterns as estimated from the spatial covariance across brain regions. A subnetwork analysis further disentangled the contribution of different connections. The results show that face and place imagery can be decoded from both short-range and long-range connections. Overall, the results show that imagined object categories can be distinguished based on functional connectivity patterns observed in a category-specific network. Notably, functional connectivity estimates rely on purely endogenous brain signals suggesting that an external reference is not necessary to elicit such category-specific network dynamics. </p>

# Description

* <p align="justify"> "simulation_plot.py" and "manifold_plot.py" generate Figure 1. This figure represents a comparison between time-domain and covariance-based decoding when simulated signals are temporally misaligned across trials. The input features for time-domain and covariance-based decoding are vectorized. The distribution of cosine similarity within- and between- condition vectors shows that in most of the trials within-condition vectors are more similar than between-condition vectors when using covariance-based decoding. In contrast, within- and between- condition vectors are equally similar, and therefore indistinguishable, when using time-domain decoding. </p>

# Data & Code Availability

<p align="justify"> This repository contains only a portion of the full dataset, and the provided code is meant to replicate the main figures only. For access to the full dataset, please contact the author (Francesco Mantegna, fmantegna93@gmail.com). </p>

# DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14536754.svg)](https://doi.org/10.5281/zenodo.14536754)
