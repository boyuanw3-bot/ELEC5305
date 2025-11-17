## Result：
pictures are in the outputs 

Audio is in data and you can see the demo in data/audio demos
## Running Order

### 1. Quick Demo

python demo_quickstart.py

**Output**: Quick demonstration of the speech recognition system with sample audio

### 2. Baseline System

python src/baseline_mfcc.py

**Output**: Baseline MFCC-based speech recognition results and accuracy metrics

### 3. Feature Extraction

#### ICA Features

python src/ica.py

**Output**: ICA-extracted features from audio signals

#### Gammatone Features

python src/gammatone.py

**Output**: Gammatone filterbank features

### 4. Model Training & Evaluation

#### Comprehensive Evaluation

python src/comprehensive_evaluation.py

**Output**: Full evaluation results comparing different feature extraction methods

#### Multi-SNR Evaluation

python src/multi_snr_evaluation.py

**Output**: System performance under different signal-to-noise ratio conditions

#### Challenging Evaluation

python src/challenging_evaluation.py

**Output**: Results on challenging test cases (noisy environments, accents, etc.)

### 5. Analysis & Optimization

#### Feature Analysis

python src/feature_analysis.py

**Output**: Visualization and analysis of extracted features

#### Detailed Analysis

python src/detailed_analysis.py

**Output**: Detailed performance breakdown by phoneme, word, or utterance

#### Hyperparameter Optimization

python src/hyperparameter_optimization.py

**Output**: Optimal hyperparameters and performance curves

#### Noise Robustness Analysis

python src/noise_robustness_analysis.py

**Output**: System robustness metrics under various noise conditions

### 6. Visualization

python src/visualize.py

**Output**: Plots and figures showing feature distributions, confusion matrices, and performance comparisons

### 7. Evaluate All

python src/evaluate_all.py

**Output**: Complete evaluation report with all metrics across all configurations

### 8.Challenging Evaluation
bashpython utils/challenging_evaluation.py

Purpose: Test system performance on difficult scenarios
**Output**:
Performance on low SNR (noisy) conditions

Results with different speaker accents

Recognition accuracy on overlapping speech

Handling of background noise and reverberations

Challenging phoneme confusion analysis
