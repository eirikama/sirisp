# Solver for InfraRed Inverse Scattering Problems (SIRISP)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/paper-Nature%20Comm.%20Chem.%202022-red)](https://doi.org/10.1038/s42004-022-00792-3)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42004--022--00792--3-orange)](https://doi.org/10.1038/s42004-022-00792-3)


<img border="0" align="Center" src="/doc/model_illustration.png" alt="Model Illustration" width=100%/>


A deep learning framework for solving the **infrared inverse scattering problem**: recovering the true 3D molecular absorption distribution of biological cells from distorted IR spectroscopic measurements. Published in *Nature Communications Chemistry* (2022).

---

## Overview

Infrared spectroscopy of biological cells is distorted by Mie scattering — a physical effect that masks the underlying molecular absorption signal and complicates biochemical analysis. SIRISP addresses this by training a neural network to invert the scattering process, enabling high-fidelity recovery of the molecular absorption spectrum from raw measurements.

**Key capabilities:**
- Reconstructs 3D molecular absorption maps from IR spectroscopic data
- Trained on physics-based simulated spectra for robust generalisation
- Applicable to single cells and tissue samples in FTIR microscopy
- Produces results consistent with those reported in peer-reviewed research

---

## Paper

> Magnussen E. A., Zimmermann B., Blazhko U., Dzurendova S., Dupuy-Galet B., Byrtusova D.,
> Muthreich F., Tafintseva V., Liland K. H., Tøndel K., Shapaval V., Kohler A.
> **"Deep Learning-enabled Inference of 3D Molecular Absorption Distribution of Biological Cells from IR Spectroscopic Data"**
> *Nature Communications Chemistry* **5**, 174 (2022).
> https://doi.org/10.1038/s42004-022-00792-3

---


