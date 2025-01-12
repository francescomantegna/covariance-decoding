# covariance-decoding

<p align="justify"> This repository contains the source code for "Covariance-based decoding reveals a category-specific functional connectivity network for imagined visual objects" (Mantegna, Olivetti, Schwedhelm, Baldauf). </p>

# Abstract

<p align="justify"> The coordination of different brain regions is required for the visual imagery of complex objects (e.g., faces and places). Short-range connectivity within sensory areas is necessary to construct the mental image. Long-range connectivity between control and sensory areas is necessary to re-instantiate and maintain the mental image. While dynamic changes in functional connectivity are expected during visual imagery, it is unclear whether a category-specific network exists in which the strength and the spatial destination of the connections vary depending on the imagery target. In this magnetoencephalography study, we used a minimally constrained experimental paradigm wherein imagery categories were prompted using visual word cues only, and we decoded face versus place imagery based on their underlying functional connectivity patterns as estimated from the spatial covariance across brain regions. A subnetwork analysis further disentangled the contribution of different connections. The results show that face and place imagery can be decoded from both short-range and long-range connections. Overall, the results show that imagined object categories can be distinguished based on functional connectivity patterns observed in a category-specific network. Notably, functional connectivity estimates rely on purely endogenous brain signals suggesting that an external reference is not necessary to elicit such category-specific network dynamics. </p>

# Description

Two sample datasets were uploaded in the "data/" folder. Each dataset consists of a baseline time window, 7 data time windows, a spatial covariance matrix in source space and a class label array. The datasets are preprocessed and standardized. In addition, an array which contains information about the cortical parcellation was uploaded including atlas label names and positions.

* <p align="justify"> "simulation_plot.py" and "manifold_plot.py" illustrate the decoding pipeline on simulated data, similar to Figure 1. This figure represents a comparison between time-domain and covariance-based decoding when simulated signals are temporally misaligned across trials. The input features for time-domain and covariance-based decoding are vectorized. The distribution of cosine similarity within- and between- condition vectors shows that in most of the trials within-condition vectors are more similar than between-condition vectors when using covariance-based decoding. In contrast, within- and between- condition vectors are equally similar, and therefore indistinguishable, when using time-domain decoding. </p>

* <p align="justify"> "timedomain_plot.py" and "covariance_plot.py" generate single-subject decoding time courses for each decoding method, similar to Figure 2. Time-domain decoding uses multivariate MEG signals as input features. Covariance-based decoding uses spatial covariance matrices across MEG signals represented onto a Riemannian manifold and projected to a vector in an homomorphic Euclidean tangent space as input features. </p>

* <p align="justify"> "importance_plot.py" illustrates the most informative connections for face (red lines) vs. place (blue lines) imagery decoding on a glass brain plot and a connectivity matrix for each hemisphere, similar to Figure 6. The cortical parcellation was derived from the Glasser et al. atlas. A theory-drive selection of visual (blue), parietal (green), inferior temporal (purple), and inferior frontal (orange) Regions of Interest (ROIs) is represented. </p>

# Data & Code Availability

<p align="justify"> This repository contains only a portion of the full dataset, and the provided code is meant to replicate the main figures only. For access to the full dataset, please contact the corresponding author (Francesco Mantegna, fmantegna93@gmail.com). </p>

# Citation

<p align="justify"> The paper is currently under review, if you use the code in this repository please cite the preprint: Mantegna, F., Olivetti, E., Schwedhelm, P., & Baldauf, D. (2022). Covariance-based Decoding Reveals Content-specific Feature Integration and Top-down Processing for Imagined Faces versus Places. BioRxiv, 2022-09. (link: https://www.biorxiv.org/content/10.1101/2022.09.26.509536v2.abstract) </p>

# DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14536754.svg)](https://doi.org/10.5281/zenodo.14536754)
