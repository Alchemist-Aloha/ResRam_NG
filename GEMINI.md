# Project Understanding: Bodipy FSRS Analysis

This project implements the theoretical framework described in the provided publications to benchmark DFT exchange-correlation functionals (XCFs) against experimental **Femtosecond Stimulated Raman Spectroscopy (FSRS)** data of Bodipy.

## Core Components and Mapping to Theory

1.  **Theoretical Model**: The code in `resram_core.py` implements the **Independent Mode Displaced Harmonic Oscillator (IMDHO)** formalism.
    *   **Absorption/Fluorescence**: Calculated using the time-correlator $A(t)$ (Equation S6 in SI) and the Brownian oscillator line shape function $g(t)$ (Equation S9).
    *   **Resonance Raman Cross Sections**: Implemented in `cross_sections(obj)` using the first-order approximation (Equation S5).

2.  **Fitting Procedure**: The `FSRSanalysis_v2.ipynb` notebook uses the `lmfit` library to minimize the residual between experimental and calculated spectra. It optimizes:
    *   **Dimensionless displacements ($\Delta_k$)** for each vibrational mode (loaded from `freqs.dat` and `deltas.dat`).
    *   **Global Parameters** (from `inp.txt`): `gamma` ($\Gamma$, homogeneous broadening), `theta` ($\theta$, inhomogeneous broadening), `E0` (vertical transition energy), and `transition_length` ($M$).

3.  **Data Structures**:
    *   `abs_exp.dat`: Experimental absorption data.
    *   `profs_exp.dat`: Experimental Raman excitation profiles.
    *   `best_fit/`: Contains optimized parameters and resulting spectra from successful fits.

## Purpose

The project effectively maps the multidimensional excited-state potential energy surface (PES) by determining the displacements that best reproduce the vibronic features of the experimental spectra. This allows for a direct comparison with TD-DFT predictions as discussed in the main paper, helping identify which exchange-correlation functionals best describe the electronic excited state of Bodipy.
