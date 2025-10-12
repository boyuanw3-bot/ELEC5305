Project Overview

This project builds a minimal-dependency, fully reproducible audio scene analysis pipeline that:

Achieves perfect reconstruction (PR) of STFT/ISTFT (WOLA, sqrt-Hann, 50% overlap)

Computes 2D modulation spectra, custom 40-band Mel energy, and lightweight spectral features (spectral centroid/bandwidth/rolloff/energy/RMS/ZCR)

Demonstrates spectral-flux/onset detection and simple tempo (BPM) estimation

Implements LPC envelope modeling with noise-excited resynthesis for interpretability

The pipeline serves as an interpretable baseline for teaching/research and a feature front-end for downstream supervised tasks (classification/retrieval/segmentation).

Current progress

PR-STFT/ISTFT (WOLA, sqrt-Hann, 50% overlap) implemented and numerically verified

2D modulation spectrum, 40-band Mel, and various lightweight features implemented

Spectral-flux and onset detection demo completed

LPC (p≈12–20) envelope and noise-excited resynthesis demo completed

Generated figures: magnitude spectrogram, modulation spectrum, Mel plots, feature curves, plus resynthesized audio examples

2) Background & Problem Statement

In instructional and exploratory settings, users face two common issues:

Heavy toolboxes → black-box behavior, harder alignment/explanation

Fragmented code → inconsistent parameters/axes/units across modules, poor reproducibility

Objective. Provide a clear, lightweight, reproducible, and interpretable baseline that verifies PR conditions, aligns features across views, and closes the loop with resynthesis, offering a stable front-end for later supervised/visualization tasks.

3) Research Objectives (Aims & Questions)

O1. Build a PR-STFT/ISTFT pipeline and report numerical PR (SNR/reconstruction error).

O2. Implement 2D modulation spectra, Mel energy, and lightweight features; compare music / ambient / direct scenes.

O3. Provide a lightweight onset/tempo demo and cross-validate with energy, spectral flux, and modulation energy.

O4. Use LPC envelope + noise-excited resynthesis to connect analysis and synthesis for better interpretability.

O5. Deliver reproducible code, a unified plotting interface, and a tutorial-style README/Pages.

4) Methodology
4.1 Signal Processing Core

PR-STFT/ISTFT (WOLA).

Window: sqrt-Hann, hop H = M/2 (50% overlap), FFT length N (typ. 2048), sample rate fs = 48 kHz

PR verification: reconstruction SNR, error curves, and boundary handling (frame stitching/zero-padding/OLA)

2D Modulation Spectrum.

2D FFT over |STFT| or log-magnitude; explicit axes for temporal modulation (Hz) and spectral modulation (cycles/kHz or cycles/bin)

Unified axis labels/units across views to avoid misalignment

Custom 40-Band Mel Energy.

Triangular filterbank on 0–fs/2; log (dB) visualizations

Semantic alignment with linear spectrogram and modulation spectrum

Lightweight Features.

RMS, ZCR, spectral centroid/bandwidth/rolloff, spectral flux

Time-series plots with key peak annotations

Onset / Tempo (brief).

Peak picking on spectral flux for tempo/BPM estimation

Cross-check vs low-temporal-modulation energy (≈0–10 Hz) in modulation spectra

LPC Envelope & Resynthesis.

Levinson–Durbin, p ≈ 12–20; compare to average magnitude spectrum/smoothed envelope

Noise-excited (or quasi-periodic/pulse) resynthesis to examine envelope-driven timbre

4.2 Engineering & Reproducibility

Unified I/O presets for fs / M / H / N

Unified coordinates/units on all plots (time/frequency/modulation freq)

Modular plotting helpers: plot_spectrogram(), plot_modulation(), plot_mel(), plot_features()

PR-Check unit test: one-click SNR & error validation

One-click script: raw .wav → all figures & audio → write to results/

5) Data & Experiments

Data. Self-recorded/owned short clips at 48 kHz, covering music / ambient / direct / speech conditions

Default parameters.

fs = 48_000
M  = 1024
H  = 512
N  = 2048


Metrics.

PR: SNR (dB), reconstruction error curves

Onset/Tempo: peak alignment on example segments, BPM estimates

Runtime: latency/duration comparisons across Light / Standard / High-Precision presets

6) Preliminary Achievements

See results/figures/ and results/audio/ (linked on the Project Pages).

 PR verified at M=1024, H=512, N=2048, fs=48 kHz (SNR near numerical precision; error ≈ 0)

 2D modulation spectra separate scenes: low temporal modulation (≈0–10 Hz) dominates; music vs ambient/direct show distinct textures/energies

 Mel / linear spectrum / features agree: spectral-centroid peaks align with spectral-flux peaks in time

 LPC envelope & resynthesis: around p≈14 captures broadband energy shape; noise-excited resynthesis exposes timbre differences across scenes

Suggestion: include ≥3 core plots (spectrogram, modulation spectrum, Mel/features) and ≥1 resynthesized audio in public folders so staff can view/listen immediately.

7) Expected Outcomes

A clean, reproducible codebase (MATLAB first; optional Python parity) with one-click scripts to regenerate all figures and audio

Tables/metrics: PR-SNR; onset/tempo examples; preset vs runtime comparisons

Documentation/Tutorial: README + GitHub Pages explaining algorithms, parameter choices, results, and reproduction steps

8) Repo & Pages

Repository: https://github.com/boyuanw3-bot/ELEC5305

GitHub Pages: https://boyuanw3-bot.github.io/ELEC5305

Demo anchor: add a Results section at the top of README that links to figures/audio on Pages

9) Milestones & Timeline

W6–W7
Refactor STFT/ISTFT + PR-Check; abstract plotting API; prepare a small demo set; publish v0 README + initial figures

W8–W9
Finalize 2D modulation, Mel, features; unify axes/colormaps; publish v1 results (multi-scene comparisons + resynth audio)

W10–W11
(Optional) Python parity; stronger onset/tempo demos (clear BPM examples); runtime vs preset table

W12–W13
Polish docs & Pages; upload short demo video; finalize report and code freeze

10) Risks & Mitigations

R1: Modulation axes/units confusion.
Mitigation: document formulas & axes clearly; provide a compute_mod_axes() helper used by all plotting functions.

R2: LPC overfitting/stability.
Mitigation: keep p in 12–20; check pole radii; mild regularization if needed.

R3: Runtime/memory trade-offs.
Mitigation: offer Light/Standard/High-Precision presets; include a runtime comparison table; optional downsampling for reproduction.

R4: Limited interpretability.
Mitigation: cross-validate across views (spectrogram ↔ Mel ↔ modulation ↔ features ↔ onset ↔ LPC resynthesis) with concise textual notes.

11) Evaluation

PR-SNR (dB) and reconstruction error

Feature consistency (temporal alignment of centroid/flux/energy peaks)

Visualization quality (consistent axes/units/colormaps)

Reproducibility (one-click script regenerates all artifacts on a fresh machine)

Documentation clarity (README/Pages as a teaching resource)
