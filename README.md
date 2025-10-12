# ELEC5305
project proposal 2
1. Project Overview 
This project aims to build a pipeline for audio scene analysis with minimal dependencies and full reproducibility: achieving perfect reconstruction (PR) of STFT/ISTFT, two-dimensional modulation spectra, custom Mel energy, lightweight spectral features (such as spectral centroid/bandwidth/rolloff/energy/RMS/ZCR), fluctuation/onset and simple rhythm estimation, as well as envelope modeling based on LPC and noise excitation resynthesis without relying on large toolboxes.
This pipeline can serve as an "interpretable" baseline in teaching/research and also as a feature front-end for subsequent supervised tasks (classification/retrieval/segmentation). 
Current progress: 
The PR-STFT/ISTFT (WOLA, sqrt-Hann, 50% overlap) has been implemented and verified numerically. 
2D modulation spectrum, 40-band Mel energy and a variety of lightweight features have been implemented. 
The demonstration of spectral flux and onset detection has been achieved. 
An example of LPC (p ≈ 12–20 adjustable) envelope and noise excitation re-synthesis has been achieved. 
The comparison graphs have been generated: amplitude spectrum, modulation spectrum, Mel, feature curve, resynthesized audio, etc. 
2. Background and Problem Statement 
Teachers and beginners often encounter a dilemma: 
The toolbox is too heavy (there are too many black boxes, making it difficult to "align" and "explain"). 
Fragmentation of code (inconsistent parameters, coordinate systems, and units in each link, making it difficult to reproduce). 
Objective: To provide a clear-structured, lightweight-dependent, reproducible and interpretable audio analysis baseline, which can verify the PR conditions, feature consistency and reproducibility (re-synthesis) within the same framework, and offer stable input for subsequent supervised learning and visualization. 
3. Research Objectives (Aims & Research Questions) 
O1. Build a PR-STFT/ISTFT pipeline and provide numerical-level PR verification (SNR/reconstruction error). 
O2. Implement 2D modulation spectra, Mel energy, and lightweight features, and compare the differences in various scenarios (such as music / ambient / direct). 
O3. Build a lightweight demonstration for onset/tempo detection and cross-validate it with information such as energy, spectral flux, and modulation energy. 
O4. By LPC envelope + noise excitation resynthesis, the "analysis-synthesis" connection is established, enhancing the model's interpretability. 
O5. Output reproducible code, a unified plotting interface, and a README/tutorial page. 
4. Methodology
4.1 Signal Processing Core 
PR-STFT/ISTFT (WOLA): 
Window function: sqrt-Hann; Hop: H = M/2 (50% overlap); FFT length N (typically 2048); Sampling rate fs = 48 kHz; 
Provide PR verification: Reconstruct SNR, error curve, and boundary processing description (frame connection/zero padding/overlap-add). 
Two-dimensional modulation spectrum: 
Perform a 2D FFT on |STFT| or the logarithmic amplitude, and provide an interpretation of the coordinates as time modulation (Hz) and frequency modulation (cycles/kHz or cycles/bin). 
Unify the labeling and units of the frequency axis and the time axis (important, to avoid "misalignment"). 
Mel Energy (Custom 40-band): 
Self-built triangular filter bank (0–fs/2), log visualization (dB); 
Align the observations of the linear spectrum and modulation spectrum semantically. 
Lightweight features: 
RMS, ZCR, spectral centroid/bandwidth/rolloff, spectral flux; 
Draw the characteristic curve over time and mark the key peaks. 
Onset / Tempo (Brief):

Peak detection based on spectral flux to estimate tempo/rate (BPM); 
The relationship with the low-frequency (0–10 Hz) energy of the modulation spectrum was verified. 
LPC Envelope and Resynthesis: 
Solved by Levinson-Durbin, p ≈ 12–20; 
Compare with the average amplitude spectrum / smoothed envelope; 
Synthesize coarse-grained reproductions using noise excitation (or quasi-periodic/pulse) to examine the explanatory power of the envelope on "timbre". 
4.2 Engineering and Reproduction Agreement 
Unified I/O: Default preset for agreed-upon sampling rate / window length / frame shift / FFT length; 
Unified coordinate system: Each graph automatically labels the horizontal and vertical axes along with their units (time/frequency/modulation frequency, etc.); 
Modular plotting: plot_spectrogram() / plot_modulation() / plot_mel() / plot_features(); 
PR-Check unit test: One-click verification of SNR and reconstruction error; 
From the original wav → Generate all graphs and audio → Output to results/. 
5. Data and Experiments 
Data: Self-collected / owned samples (48 kHz), covering various scenarios (music / ambient / direct / speech, etc.); 
Parameter suggestions: M=1024, H=512, N=2048, fs=48 kHz (Three presets of "Lightweight/Standard/High Precision" can be provided in the README). 
Measurement: 
PR: SNR (dB), Reconstruction Error Curve; 
Onset/Tempo: Peak alignment of the sample paragraph and estimated BPM; 
Runtime: Record a comparison table of latency/duration for typical audio under three preset settings. 
6. Preliminary Achievements (What’s Achieved So Far) 
The following content should be accompanied by illustrations (it is recommended to place them in results/figures/ and insert them in README/GitHub Pages). 
PR verification passed: Under the settings of M=1024, H=512, N=2048, fs=48k, the restored SNR reached the level of numerical precision, and the reconstruction error was close to 0. 
2D modulation spectrum stability distinguishes scenarios: low-time modulation (≈0–10 Hz) energy is dominant, and music vs ambient/direct presents different textures and energy distributions; 
Mel/Linear Spectrum/Consistency of Features: There is a consistent relationship between the peak moments of spectral centroid and spectral flux. 
LPC Envelope and Synthesis: An envelope with p ≈ 14 can better describe the broadband energy shape; noise excitation resynthesis can reveal the timbre differences between scenes. 
Please place at least 3 core graphs (spectrogram, modulation spectrum, Mel/feature) and 1 resynthesized audio segment in the public directory of the repository so that the teacher can directly open and listen to/view them. 
7. Expected Outcomes 
A clean and reproducible codebase (MATLAB preferred; optional Python version available), including scripts to generate all figures and audio files with one click. 
Indicators and tables: PR-SNR, onset/tempo sample, parameter preset vs. runtime comparison table; 
Document/Tutorial: README and GitHub Pages (Algorithm Description, Reason for Parameter Selection, Result Presentation, Replication Experiment Script). 
8. Repository and Page (GitHub Links) 
GitHub repository: https://github.com/yourID/elec5305-project-xxxxxx (Please replace) 
GitHub Pages (Project Homepage): https://yourID.github.io/elec5305-project-xxxxxx (Please replace) 
Demo entry: Place an "Results Display" anchor at the top of the README, linking to the image/audio section of Pages. 
9. Milestones and Timeline Planning (Timeline) 
W6–W7: 
Reconstruct STFT/ISTFT and PR-Check; Abstract the drawing API; 
Prepare a small sample dataset; submit the first version of the README and preliminary result graphs. 
W8–W9: 
Improve two-dimensional modulation spectrum, Mel, and lightweight features; unify coordinates/color scales; 
Submit the second version of the results (multi-scenario comparison diagrams + synthesized audio). 
W10–W11: 
Optional: Python version parity; 
Enhanced start/rhythm examples (clearer BPM examples); 
Compare the runtime table with the preset. 
W12–W13: 
Polish the documents and pages; 
Submit the final code and a short demonstration video; complete the final report. 
10. Risks & Mitigations 
(R1) Confusion between modulation spectrum coordinates and units: 
Countermeasures: Clearly write out the formulas and axis units; provide the compute_mod_axes() utility function, and uniformly call it for plotting. 
(R2) Overfitting or instability of LPC: 
Countermeasures: Set the stable range at 12–20; check the radius of the critical point; add mild regularization if necessary. 
(R3) Runtime/Memory Trade-off: 
Countermeasures: Provide "lightweight/standard/high-precision" presets; attach a runtime comparison table; optional downsampling for reproduction. 
(R4) Insufficient explainability: 
Countermeasure: Cross-verification from multiple perspectives (Spectral - Mel - Modulation Spectral - Feature - Onset - LPC Synthesis) with accompanying textual explanations. 
11. Evaluation Method (Evaluation) 
PR-SNR (dB), reconstruction error; 
Feature consistency (whether the spectral centroid, spectral flux, and energy peak moments are aligned); 
Visualization quality (uniformity of coordinates, units, and color scales); 
Reproducibility (whether a one-click script can fully generate images and audio in a new environment); 
Readability of the documentation (clarity and educational value of README/Pages).
