
## Running Order

### 1. Quick Demo

python demo_quickstart.py

**Output**: Quick demonstration of the speech recognition system with sample audio

### 2. Baseline System

python utils/baseline_mfcc.py

**Output**: Baseline MFCC-based speech recognition results and accuracy metrics

### 3. Feature Extraction

#### ICA Features

python utils/ica.py

**Output**: ICA-extracted features from audio signals

#### Gammatone Features

python utils/gammatone.py

**Output**: Gammatone filterbank features

### 4. Model Training & Evaluation

#### Comprehensive Evaluation

python utils/comprehensive_evaluation.py

**Output**: Full evaluation results comparing different feature extraction methods

#### Multi-SNR Evaluation

python utils/multi_snr_evaluation.py

**Output**: System performance under different signal-to-noise ratio conditions

#### Challenging Evaluation

python utils/challenging_evaluation.py

**Output**: Results on challenging test cases (noisy environments, accents, etc.)

### 5. Analysis & Optimization

#### Feature Analysis

python utils/feature_analysis.py

**Output**: Visualization and analysis of extracted features

#### Detailed Analysis

python utils/detailed_analysis.py

**Output**: Detailed performance breakdown by phoneme, word, or utterance

#### Hyperparameter Optimization

python utils/hyperparameter_optimization.py

**Output**: Optimal hyperparameters and performance curves

#### Noise Robustness Analysis
```bash
python utils/noise_robustness_analysis.py
```
**Output**: System robustness metrics under various noise conditions

### 6. Visualization
```bash
python utils/visualize.py
```
**Output**: Plots and figures showing feature distributions, confusion matrices, and performance comparisons

### 7. Evaluate All
```bash
python utils/evaluate_all.py
```
**Output**: Complete evaluation report with all metrics across all configurations
