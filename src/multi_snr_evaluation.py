"""
Multi-SNR Evaluation Script
Evaluate system performance across different noise levels

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class MultiSNREvaluator:
    """Multi-SNR level evaluator"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.ica_module = ICAModule()
        self.gammatone = GammatoneFilterBank(
            n_filters=32,
            sample_rate=self.config.sample_rate
        )
        
        # Store results
        self.results = {
            'snr_levels': [],
            'baseline': [],
            'ica_enhanced': [],
            'cochlear': [],
            'full_system': []
        }
    
    def generate_test_data(self, n_samples=20):
        """Generate test dataset"""
        print(f"\nGenerating test dataset...")
        print(f"  Samples per word: {n_samples}")
        
        clean_signals = []
        labels = []
        
        for word_idx, word in enumerate(self.config.vocabulary):
            for i in range(n_samples):
                signal = self.audio_processor.generate_speech(word)
                clean_signals.append(signal)
                labels.append(word_idx)
        
        print(f"  ✓ Generated {len(clean_signals)} samples")
        
        return clean_signals, np.array(labels)
    
    def extract_features_baseline(self, signals):
        """Extract baseline MFCC features"""
        features = []
        for signal in signals:
            mfcc = self.audio_processor.extract_mfcc(signal)
            feat = mfcc.flatten()
            features.append(feat)
        return np.array(features)
    
    def extract_features_ica(self, noisy_signals, clean_signals):
        """Extract ICA-enhanced MFCC features"""
        features = []
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            mixed_signals = np.vstack([noisy_signal, clean_signal])
            separated = self.ica_module.separate(mixed_signals)
            best_idx = self.ica_module.align_sources(separated, clean_signal)
            enhanced_signal = separated[best_idx]
            
            mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
            feat = mfcc.flatten()
            features.append(feat)
        return np.array(features)
    
    def extract_features_cochlear(self, signals):
        """Extract Gammatone cochlear features"""
        features = []
        for signal in signals:
            filtered = self.gammatone.filter(signal)
            cochlear_feat = self.gammatone.extract_features(filtered)
            feat = cochlear_feat.flatten()
            features.append(feat)
        return np.array(features)
    
    def extract_features_full(self, noisy_signals, clean_signals):
        """Extract full system features (ICA + Gammatone + MFCC)"""
        features = []
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            # ICA separation
            mixed_signals = np.vstack([noisy_signal, clean_signal])
            separated = self.ica_module.separate(mixed_signals)
            best_idx = self.ica_module.align_sources(separated, clean_signal)
            enhanced_signal = separated[best_idx]
            
            # Gammatone features
            filtered = self.gammatone.filter(enhanced_signal)
            cochlear_feat = self.gammatone.extract_features(filtered)
            
            # MFCC features
            mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
            
            # Feature fusion
            feat_cochlear = cochlear_feat.flatten()
            feat_mfcc = mfcc.flatten()
            
            # Normalize and concatenate
            feat_cochlear = (feat_cochlear - feat_cochlear.mean()) / (feat_cochlear.std() + 1e-8)
            feat_mfcc = (feat_mfcc - feat_mfcc.mean()) / (feat_mfcc.std() + 1e-8)
            
            feat = np.concatenate([feat_cochlear, feat_mfcc])
            features.append(feat)
        
        return np.array(features)
    
    def evaluate_single_snr(self, features, labels, config_name, snr_db):
        """Evaluate single configuration at specific SNR"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def run_multi_snr_evaluation(self, snr_levels=[-5, 0, 5, 10, 15, 20], n_samples=20):
        """
        Run evaluation across multiple SNR levels
        
        Args:
            snr_levels: List of SNR levels to test
            n_samples: Samples per word
        """
        print("\n" + "="*70)
        print(" Multi-SNR Evaluation")
        print("="*70)
        
        # Generate clean signals
        clean_signals, labels = self.generate_test_data(n_samples=n_samples)
        
        self.results['snr_levels'] = snr_levels
        
        for snr_db in snr_levels:
            print(f"\n{'='*70}")
            print(f" Testing at SNR = {snr_db} dB")
            print('='*70)
            
            # Add noise
            noisy_signals = [
                self.audio_processor.add_noise(signal, snr_db)
                for signal in clean_signals
            ]
            
            # Configuration 1: Baseline
            print(f"\n  Config 1: Baseline MFCC...")
            features_baseline = self.extract_features_baseline(noisy_signals)
            acc_baseline = self.evaluate_single_snr(features_baseline, labels, 'baseline', snr_db)
            self.results['baseline'].append(acc_baseline)
            print(f"    Accuracy: {acc_baseline*100:.2f}%")
            
            # Configuration 2: ICA + MFCC
            print(f"\n  Config 2: ICA + MFCC...")
            features_ica = self.extract_features_ica(noisy_signals, clean_signals)
            acc_ica = self.evaluate_single_snr(features_ica, labels, 'ica_enhanced', snr_db)
            self.results['ica_enhanced'].append(acc_ica)
            print(f"    Accuracy: {acc_ica*100:.2f}%")
            
            # Configuration 3: Gammatone
            print(f"\n  Config 3: Gammatone Cochlear...")
            features_cochlear = self.extract_features_cochlear(noisy_signals)
            acc_cochlear = self.evaluate_single_snr(features_cochlear, labels, 'cochlear', snr_db)
            self.results['cochlear'].append(acc_cochlear)
            print(f"    Accuracy: {acc_cochlear*100:.2f}%")
            
            # Configuration 4: Full System
            print(f"\n  Config 4: Full System...")
            features_full = self.extract_features_full(noisy_signals, clean_signals)
            acc_full = self.evaluate_single_snr(features_full, labels, 'full_system', snr_db)
            self.results['full_system'].append(acc_full)
            print(f"    Accuracy: {acc_full*100:.2f}%")
        
        # Plot results
        self.plot_results()
        
        # Print summary
        self.print_summary()
    
    def plot_results(self):
        """Plot accuracy vs SNR curves"""
        print("\n\nGenerating performance curves...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        snr_levels = self.results['snr_levels']
        
        ax.plot(snr_levels, self.results['baseline'], 'o-', 
                label='Baseline MFCC', linewidth=2, markersize=8)
        ax.plot(snr_levels, self.results['ica_enhanced'], 's-', 
                label='ICA + MFCC', linewidth=2, markersize=8)
        ax.plot(snr_levels, self.results['cochlear'], '^-', 
                label='Gammatone Cochlear', linewidth=2, markersize=8)
        ax.plot(snr_levels, self.results['full_system'], 'd-', 
                label='Full System', linewidth=2, markersize=8)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Speech Recognition Accuracy vs Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Save figure
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'multi_snr_performance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        
        plt.close()
    
    def print_summary(self):
        """Print results summary"""
        print("\n\n" + "="*70)
        print(" Results Summary")
        print("="*70)
        
        print(f"\n{'SNR (dB)':<12} {'Baseline':<12} {'ICA+MFCC':<12} {'Cochlear':<12} {'Full System':<12}")
        print("-" * 70)
        
        for i, snr in enumerate(self.results['snr_levels']):
            print(f"{snr:<12} {self.results['baseline'][i]*100:>6.2f}%     "
                  f"{self.results['ica_enhanced'][i]*100:>6.2f}%     "
                  f"{self.results['cochlear'][i]*100:>6.2f}%     "
                  f"{self.results['full_system'][i]*100:>6.2f}%")
        
        # Calculate average improvement
        avg_baseline = np.mean(self.results['baseline'])
        avg_full = np.mean(self.results['full_system'])
        improvement = ((avg_full - avg_baseline) / avg_baseline) * 100
        
        print("\n" + "="*70)
        print(f"Average Improvement (Full System vs Baseline): {improvement:+.2f}%")
        print(f"  Average Baseline Accuracy: {avg_baseline*100:.2f}%")
        print(f"  Average Full System Accuracy: {avg_full*100:.2f}%")
        print("="*70)


def main():
    """Main function"""
    print("\n" + "="*70)
    print(" ELEC5305 Project - Multi-SNR Evaluation")
    print("="*70)
    
    evaluator = MultiSNREvaluator()
    
    # Run evaluation
    evaluator.run_multi_snr_evaluation(
        snr_levels=[-5, 0, 5, 10, 15, 20],
        n_samples=20
    )
    
    print("\n✅ Multi-SNR evaluation complete!\n")


if __name__ == "__main__":
    main()
