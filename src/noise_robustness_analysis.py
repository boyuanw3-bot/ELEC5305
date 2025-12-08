"""
Noise Robustness Comprehensive Analysis
Test system performance under various realistic noise conditions

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from audio_utils import AudioProcessor
from ica import ICAModule
from gammatone import GammatoneFilterBank
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class NoiseRobustnessAnalyzer:
    """Comprehensive noise robustness analysis"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.ica = ICAModule()
        self.gammatone = GammatoneFilterBank(n_filters=32, sample_rate=self.config.sample_rate)
    
    def generate_noise(self, length, noise_type, sample_rate):
        """Generate various types of noise"""
        
        if noise_type == 'white':
            return np.random.randn(length)
        
        elif noise_type == 'pink':
            white = np.random.randn(length)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length, 1/sample_rate)
            freqs[0] = 1
            fft = fft / np.sqrt(freqs)
            return np.fft.irfft(fft, n=length).real
        
        elif noise_type == 'brown':
            white = np.random.randn(length)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length, 1/sample_rate)
            freqs[0] = 1
            fft = fft / freqs
            return np.fft.irfft(fft, n=length).real
        
        elif noise_type == 'babble':
            # Simulate 3 interfering voices
            noise = np.zeros(length)
            t = np.arange(length) / sample_rate
            for _ in range(3):
                f0 = np.random.uniform(150, 400)
                voice = np.sin(2 * np.pi * f0 * t)
                voice += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
                noise += voice
            noise += np.random.randn(length) * 0.3
            return noise
        
        elif noise_type == 'street':
            # Simulate street noise (low-frequency dominant)
            white = np.random.randn(length)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length, 1/sample_rate)
            # Emphasize low frequencies
            emphasis = np.exp(-freqs / 500)
            fft = fft * emphasis
            return np.fft.irfft(fft, n=length).real
        
        else:
            return np.random.randn(length)
    
    def add_noise(self, signal, snr_db, noise_type):
        """Add specific noise type at given SNR"""
        signal_power = np.mean(signal ** 2)
        
        noise = self.generate_noise(len(signal), noise_type, self.config.sample_rate)
        noise_power = np.mean(noise ** 2)
        
        noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
        return signal + noise_scale * noise
    
    def generate_dataset(self, n_samples=30):
        """Generate clean dataset"""
        print("\nGenerating clean dataset...")
        
        clean_signals = []
        labels = []
        
        for word_idx, word in enumerate(self.config.vocabulary):
            for _ in range(n_samples):
                signal = self.audio_processor.generate_speech(word)
                # Add slight variation
                variation = np.random.uniform(0.95, 1.05)
                signal = signal * variation
                
                clean_signals.append(signal)
                labels.append(word_idx)
        
        print(f"  ✓ Generated {len(clean_signals)} clean samples")
        return clean_signals, np.array(labels)
    
    def extract_features(self, noisy_signals, clean_signals, method):
        """Extract features using specified method"""
        features = []
        
        if method == 'baseline':
            for signal in noisy_signals:
                mfcc = self.audio_processor.extract_mfcc(signal)
                features.append(mfcc.flatten())
        
        elif method == 'ica':
            for noisy, clean in zip(noisy_signals, clean_signals):
                mixed = np.vstack([noisy, clean])
                separated = self.ica.separate(mixed)
                best_idx = self.ica.align_sources(separated, clean)
                enhanced = separated[best_idx]
                mfcc = self.audio_processor.extract_mfcc(enhanced)
                features.append(mfcc.flatten())
        
        elif method == 'gammatone':
            for signal in noisy_signals:
                filtered = self.gammatone.filter(signal)
                feat = self.gammatone.extract_features(filtered)
                features.append(feat.flatten())
        
        elif method == 'full':
            for noisy, clean in zip(noisy_signals, clean_signals):
                mixed = np.vstack([noisy, clean])
                separated = self.ica.separate(mixed)
                best_idx = self.ica.align_sources(separated, clean)
                enhanced = separated[best_idx]
                
                filtered = self.gammatone.filter(enhanced)
                cochlear = self.gammatone.extract_features(filtered)
                mfcc = self.audio_processor.extract_mfcc(enhanced)
                
                feat_c = cochlear.flatten()
                feat_m = mfcc.flatten()
                feat_c = (feat_c - feat_c.mean()) / (feat_c.std() + 1e-8)
                feat_m = (feat_m - feat_m.mean()) / (feat_m.std() + 1e-8)
                
                features.append(np.concatenate([feat_c, feat_m]))
        
        return np.array(features)
    
    def evaluate(self, features, labels):
        """Train and evaluate classifier"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def run_comprehensive_analysis(self):
        """Run comprehensive noise robustness analysis"""
        print("\n" + "="*70)
        print(" Comprehensive Noise Robustness Analysis")
        print("="*70)
        
        # Test configurations
        snr_levels = [-10, -5, 0, 5, 10, 15]
        noise_types = ['white', 'pink', 'babble', 'street']
        methods = ['baseline', 'ica', 'gammatone', 'full']
        
        # Generate clean dataset
        clean_signals, labels = self.generate_dataset(n_samples=30)
        
        # Store results
        results = {
            method: {noise: [] for noise in noise_types}
            for method in methods
        }
        
        # Run experiments
        for noise_type in noise_types:
            print(f"\n{'='*70}")
            print(f" Testing with {noise_type.upper()} Noise")
            print('='*70)
            
            for snr_db in snr_levels:
                print(f"\n  SNR = {snr_db:>3} dB:", end=" ")
                
                # Generate noisy signals
                noisy_signals = [
                    self.add_noise(signal, snr_db, noise_type)
                    for signal in clean_signals
                ]
                
                # Test each method
                for method in methods:
                    features = self.extract_features(noisy_signals, clean_signals, method)
                    accuracy = self.evaluate(features, labels)
                    results[method][noise_type].append(accuracy)
                    print(f"{method:12s}={accuracy*100:5.1f}%", end=" ")
                print()
        
        # Generate visualizations
        self.visualize_results(results, snr_levels, noise_types)
        self.generate_summary_table(results, snr_levels, noise_types)
        
        return results
    
    def visualize_results(self, results, snr_levels, noise_types):
        """Visualize comprehensive results"""
        print("\n\nGenerating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Noise Robustness Analysis - All Noise Types', 
                    fontsize=16, fontweight='bold')
        
        colors = {
            'baseline': '#1f77b4',
            'ica': '#ff7f0e',
            'gammatone': '#2ca02c',
            'full': '#d62728'
        }
        
        labels = {
            'baseline': 'Baseline MFCC',
            'ica': 'ICA + MFCC',
            'gammatone': 'Gammatone',
            'full': 'Full System'
        }
        
        for idx, noise_type in enumerate(noise_types):
            ax = axes[idx // 2, idx % 2]
            
            for method in ['baseline', 'ica', 'gammatone', 'full']:
                acc = [a * 100 for a in results[method][noise_type]]
                ax.plot(snr_levels, acc, 'o-', 
                       label=labels[method],
                       color=colors[method],
                       linewidth=2.5, markersize=8)
            
            ax.set_xlabel('SNR (dB)', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'{noise_type.capitalize()} Noise', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'noise_robustness_comprehensive.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        plt.close()
    
    def generate_summary_table(self, results, snr_levels, noise_types):
        """Generate summary statistics table"""
        print("\n\n" + "="*70)
        print(" Summary Statistics")
        print("="*70)
        
        for noise_type in noise_types:
            print(f"\n{noise_type.upper()} Noise:")
            print(f"{'SNR (dB)':<12} {'Baseline':<12} {'ICA+MFCC':<12} {'Gammatone':<12} {'Full System':<12}")
            print("-" * 70)
            
            for i, snr in enumerate(snr_levels):
                print(f"{snr:<12} ", end="")
                for method in ['baseline', 'ica', 'gammatone', 'full']:
                    acc = results[method][noise_type][i] * 100
                    print(f"{acc:>6.1f}%      ", end="")
                print()
            
            # Average performance
            print("\nAverage:", end=" " * 5)
            for method in ['baseline', 'ica', 'gammatone', 'full']:
                avg_acc = np.mean(results[method][noise_type]) * 100
                print(f"{avg_acc:>6.1f}%      ", end="")
            print("\n")
        
        # Overall comparison
        print("\n" + "="*70)
        print(" Overall Average Performance Across All Conditions")
        print("="*70)
        
        for method in ['baseline', 'ica', 'gammatone', 'full']:
            all_accs = []
            for noise_type in noise_types:
                all_accs.extend(results[method][noise_type])
            avg = np.mean(all_accs) * 100
            std = np.std([a*100 for a in all_accs])
            print(f"  {method:12s}: {avg:5.2f}% (±{std:4.2f}%)")
        
        print("="*70)


def main():
    """Main function"""
    analyzer = NoiseRobustnessAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n✅ Comprehensive noise robustness analysis complete!\n")


if __name__ == "__main__":
    main()
