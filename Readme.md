# ELEC5305 Project: Noise-Robust Speech Recognition

**Student:** Boyuan Wang 
**Student ID:** 530775632
**Course:** ELEC5305 - Acoustics and Signal Processing  
**University of Sydney, 2025**

---

## Quick Start for Instructors

### 1. Install Dependencies 

```bash
pip install -r requirements.txt
```

### 2. Run Demo 

```bash
python demo_quickstart.py
```

### 3. View Results

Results will be saved to:
- `results/baseline_summary.txt` - Human-readable summary
- `results/baseline_results.pkl` - Detailed results
- `models/baseline_gmm.pkl` - Trained model
The result image is in outputs
The audio demo is in data/audio_demos
---

## Research Question

**Can ICA blind source separation combined with Gammatone cochlear features significantly improve speech recognition accuracy in noisy environments?**

---

## Project Overview

This project investigates biologically-inspired signal processing methods for noise-robust speech recognition:

- **Baseline:** MFCC + GMM (traditional approach)
- **Proposed:** ICA + Gammatone + MFCC fusion
- **Dataset:** Synthetic digit recognition (0-9)
- **Evaluation:** Multiple SNR levels and noise types

### Key Technologies

1. **MFCC Features** - Mel-Frequency Cepstral Coefficients (Davis & Mermelstein, 1980)
2. **ICA** - Independent Component Analysis for blind source separation (Hyvärinen & Oja, 2000)
3. **Gammatone Filterbank** - Cochlear model (Patterson et al., 1988)
4. **GMM Classifier** - Gaussian Mixture Model (Reynolds & Rose, 1995)

---

## Project Structure

```
ELEC5305_Project/
│
├── demo_quickstart.py          # Quick demonstration script
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
│
├── src/
│   ├── config.py              # Configuration parameters
│   ├── audio_utils.py         # Audio processing utilities 
│   ├── ica.py                 # ICA implementation 
│   ├── gammatone.py           # Gammatone filterbank 
│   ├── evaluate_all.py        # Full evaluation script 
│           ...
├── data/                      # Generated data (created on first run)
│   ├── clean_speech/          # Synthetic clean speech
│   ├── noisy_speech/          # Noisy test data
│   └── noise/                 # Noise files
│
├── models/                    # Trained models (created on first run)
│   └── baseline_gmm.pkl       # Baseline GMM classifier
│
└── results/                   # Experimental results (created on first run)
    ├── baseline_results.pkl   # Detailed results
    ├── baseline_summary.txt   # Human-readable summary
   
```

---

## Expected Results

### Baseline System (MFCC + GMM)

**Clean Speech:**
- Accuracy: 95-100%
- Test samples: 30 (30% of 100 total)

**Noisy Speech (SNR = 10 dB):**
- Accuracy: 85-92%

### Performance Metrics

The system outputs:
- Overall accuracy
- Per-digit accuracy
- Confusion matrix
- Classification report

---

## Methodology

### 1. Data Generation

```
Synthetic Speech Generation
    ↓
10 digits × 10 samples = 100 total samples
    ↓
70% train / 30% test split
```

### 2. Feature Extraction

```
Audio Signal (16 kHz, 1 sec)
    ↓
MFCC Extraction (13 coefficients)
    ↓
Add Δ and ΔΔ (velocity and acceleration)
    ↓
39-dimensional feature vector
```

### 3. Classification

```
Features → GMM Training (one per digit)
    ↓
Test Sample → Compute likelihood for each GMM
    ↓
Predict digit with maximum likelihood
```

---

## Running Options

### Option 1: Quick Demo (Recommended)

```bash
python demo_quickstart.py
```

Runs baseline MFCC system in 2-3 minutes.

### Option 2: Test Installation First

```bash
python test_installation.py
```

Verifies all dependencies are installed correctly.

### Option 3: View Configuration

```bash
python src/config.py
```

Displays all experimental parameters with explanations.

---

## Dependencies

### Core Libraries
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `librosa` - Audio processing
- `soundfile` - Audio I/O
- `scikit-learn` - Machine learning (GMM)
- `pandas` - Data analysis

### Installation

```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install numpy scipy librosa soundfile scikit-learn pandas matplotlib
```

---


### Need Help?

See `GUIDANCE.md` for:
- Detailed installation instructions
- Step-by-step running guide
- Comprehensive troubleshooting
- System requirements

---

## Research Context

### Literature Foundation

This implementation follows established methodologies:

1. **MFCC Feature Extraction**
   - Davis & Mermelstein (1980)
   - Standard in speech recognition

2. **GMM Classification**
   - Reynolds & Rose (1995)
   - Effective for small vocabulary tasks

3. **ICA Blind Source Separation**
   - Hyvärinen & Oja (2000)
   - FastICA algorithm

4. **Gammatone Filterbank**
   - Patterson et al. (1988)
   - Models human auditory system

5. **Evaluation Protocol**
   - Hirsch & Pearce (2000)
   - Standard noise robustness testing

### Contribution

This project:
- ✓ Implements baseline MFCC system
- ✓ Provides framework for ICA integration
- ✓ Enables Gammatone feature extraction
- ✓ Demonstrates systematic evaluation
- ✓ Fully documented and reproducible

---

## Academic Integrity

This project represents original work completed for ELEC5305. All referenced methodologies are properly cited in code comments and documentation.

### Key References

1. Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." IEEE TASSP.

2. Hyvärinen, A., & Oja, E. (2000). "Independent component analysis: Algorithms and applications." Neural Networks.

3. Patterson, R. D., et al. (1988). "An efficient auditory filterbank based on the gammatone function." APU Report.

4. Reynolds, D. A., & Rose, R. C. (1995). "Robust text-independent speaker identification using Gaussian mixture speaker models." IEEE TASSP.

5. Hirsch, H. G., & Pearce, D. (2000). "The Aurora experimental framework for the performance evaluation of speech recognition systems under noisy conditions." ISCA ITRW ASR.

(Full bibliography available in code documentation)

---

## Code Quality

- ✓ Fully commented with English documentation
- ✓ Follows academic research standards
- ✓ Reproducible with fixed random seeds
- ✓ Modular and extensible design
- ✓ Comprehensive error handling

---

F