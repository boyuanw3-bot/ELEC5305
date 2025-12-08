"""
Detailed Results Analysis Script
Generate confusion matrices, error analysis, and detailed performance metrics

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')


class DetailedAnalyzer:
    """Detailed results analyzer"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.ica_module = ICAModule()
        self.gammatone = GammatoneFilterBank(
            n_filters=32,
            sample_rate=self.config.sample_rate
        )
        
        self.results = {}
    
    def generate_test_data(self, n_samples=30, snr_db=5):
        """Generate test dataset"""
        print(f"\nGenerating test dataset (SNR={snr_db} dB)...")
        
        clean_signals = []
        noisy_signals = []
        labels = []
        
        for word_idx, word in enumerate(self.config.vocabulary):
            for i in range(n_samples):
                signal = self.audio_processor.generate_speech(word)
                clean_signals.append(signal)
                labels.append(word_idx)
                
                noisy_signal = self.audio_processor.add_noise(signal, snr_db)
                noisy_signals.append(noisy_signal)
        
        print(f"  ✓ Generated {len(clean_signals)} samples")
        
        return clean_signals, noisy_signals, np.array(labels)
    
    def extract_full_system_features(self, noisy_signals, clean_signals):
        """Extract full system features"""
        features = []
        
        for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
            # ICA
            mixed_signals = np.vstack([noisy_signal, clean_signal])
            separated = self.ica_module.separate(mixed_signals)
            best_idx = self.ica_module.align_sources(separated, clean_signal)
            enhanced_signal = separated[best_idx]
            
            # Gammatone
            filtered = self.gammatone.filter(enhanced_signal)
            cochlear_feat = self.gammatone.extract_features(filtered)
            
            # MFCC
            mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
            
            # Fusion
            feat_cochlear = cochlear_feat.flatten()
            feat_mfcc = mfcc.flatten()
            
            feat_cochlear = (feat_cochlear - feat_cochlear.mean()) / (feat_cochlear.std() + 1e-8)
            feat_mfcc = (feat_mfcc - feat_mfcc.mean()) / (feat_mfcc.std() + 1e-8)
            
            feat = np.concatenate([feat_cochlear, feat_mfcc])
            features.append(feat)
        
        return np.array(features)
    
    def train_and_evaluate(self, features, labels):
        """Train classifier and return predictions"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        return y_test, y_pred, X_test, X_test_scaled
    
    def generate_confusion_matrix(self, y_test, y_pred):
        """Generate and plot confusion matrix"""
        print("\n\nGenerating confusion matrix...")
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.config.vocabulary,
                   yticklabels=self.config.vocabulary,
                   cbar_kws={'label': 'Count'},
                   ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Full System', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'confusion_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        
        plt.close()
        
        return cm
    
    def analyze_errors(self, y_test, y_pred):
        """Analyze classification errors"""
        print("\n\nError Analysis:")
        print("="*70)
        
        # Find misclassified samples
        errors = np.where(y_test != y_pred)[0]
        
        if len(errors) == 0:
            print("\n✓ No errors! Perfect classification.")
            return
        
        print(f"\nTotal errors: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.2f}%)")
        
        # Analyze error patterns
        error_pairs = {}
        for idx in errors:
            true_label = self.config.vocabulary[y_test[idx]]
            pred_label = self.config.vocabulary[y_pred[idx]]
            pair = (true_label, pred_label)
            
            if pair not in error_pairs:
                error_pairs[pair] = 0
            error_pairs[pair] += 1
        
        # Sort by frequency
        sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost common error patterns:")
        print(f"{'True Label':<15} {'Predicted As':<15} {'Count':<10}")
        print("-" * 45)
        for (true_label, pred_label), count in sorted_errors[:10]:
            print(f"{true_label:<15} {pred_label:<15} {count:<10}")
    
    def generate_per_class_metrics(self, y_test, y_pred):
        """Generate per-class performance metrics"""
        print("\n\nPer-Class Performance Metrics:")
        print("="*70)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(self.config.vocabulary))
        )
        
        # Create dataframe-like display
        print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for i, word in enumerate(self.config.vocabulary):
            print(f"{word:<12} {precision[i]:>6.2%}       {recall[i]:>6.2%}      "
                  f"{f1[i]:>6.2%}       {support[i]:>6}")
        
        # Plot per-class metrics
        self.plot_per_class_metrics(precision, recall, f1)
    
    def plot_per_class_metrics(self, precision, recall, f1):
        """Plot per-class metrics bar chart"""
        print("\n\nGenerating per-class metrics chart...")
        
        x = np.arange(len(self.config.vocabulary))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Word Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.config.vocabulary, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'per_class_metrics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        
        plt.close()
    
    def generate_classification_report(self, y_test, y_pred):
        """Generate detailed classification report"""
        print("\n\nDetailed Classification Report:")
        print("="*70)
        
        report = classification_report(
            y_test, y_pred, 
            target_names=self.config.vocabulary,
            digits=4
        )
        
        print(report)
        
        # Save to file
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'classification_report.txt'
        
        with open(output_file, 'w') as f:
            f.write("ELEC5305 Project - Classification Report\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"\n✓ Report saved to: {output_file}")
    
    def run_detailed_analysis(self, n_samples=30, snr_db=5):
        """
        Run complete detailed analysis
        
        Args:
            n_samples: Samples per word
            snr_db: Signal-to-noise ratio
        """
        print("\n" + "="*70)
        print(" ELEC5305 Project - Detailed Results Analysis")
        print("="*70)
        
        # Generate data
        clean_signals, noisy_signals, labels = self.generate_test_data(
            n_samples=n_samples, snr_db=snr_db
        )
        
        # Extract features
        print("\n\nExtracting full system features...")
        features = self.extract_full_system_features(noisy_signals, clean_signals)
        print(f"  Feature shape: {features.shape}")
        
        # Train and evaluate
        print("\n\nTraining classifier...")
        y_test, y_pred, X_test, X_test_scaled = self.train_and_evaluate(features, labels)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  ✓ Test accuracy: {accuracy*100:.2f}%")
        
        # Generate analyses
        cm = self.generate_confusion_matrix(y_test, y_pred)
        self.analyze_errors(y_test, y_pred)
        self.generate_per_class_metrics(y_test, y_pred)
        self.generate_classification_report(y_test, y_pred)
        
        print("\n\n" + "="*70)
        print(" Analysis Summary")
        print("="*70)
        print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
        print(f"✓ Total Test Samples: {len(y_test)}")
        print(f"✓ Number of Classes: {len(self.config.vocabulary)}")
        print(f"✓ Feature Dimension: {features.shape[1]}")
        print("\n✓ Generated outputs:")
        print("  - confusion_matrix.png")
        print("  - per_class_metrics.png")
        print("  - classification_report.txt")
        print("="*70)


def main():
    """Main function"""
    analyzer = DetailedAnalyzer()
    
    # Run analysis
    analyzer.run_detailed_analysis(
        n_samples=30,
        snr_db=5
    )
    
    print("\n✅ Detailed analysis complete!\n")


if __name__ == "__main__":
    main()
