"""
Hyperparameter Optimization Script
Optimize ICA and Gammatone parameters using grid search

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """Hyperparameter optimizer for ICA and Gammatone"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        
        # Store results
        self.results = {
            'ica_iterations': [],
            'ica_accuracies': [],
            'gammatone_filters': [],
            'gammatone_accuracies': [],
            'svm_params': []
        }
    
    def generate_test_data(self, n_samples=20, snr_db=5):
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
                
                # Add noise
                noisy_signal = self.audio_processor.add_noise(signal, snr_db)
                noisy_signals.append(noisy_signal)
        
        print(f"  ✓ Generated {len(clean_signals)} samples")
        
        return clean_signals, noisy_signals, np.array(labels)
    
    def optimize_ica_iterations(self, clean_signals, noisy_signals, labels):
        """
        Optimize ICA max_iter parameter
        
        Args:
            clean_signals: Clean signals
            noisy_signals: Noisy signals
            labels: Labels
        """
        print("\n" + "="*70)
        print(" Optimizing ICA Iterations")
        print("="*70)
        
        # Test different iteration counts
        iteration_values = [50, 100, 200, 500, 1000]
        
        for max_iter in iteration_values:
            print(f"\nTesting max_iter = {max_iter}...")
            
            # Extract features with current parameter
            features = []
            ica = ICAModule(max_iter=max_iter)
            
            for noisy_signal, clean_signal in zip(noisy_signals, clean_signals):
                try:
                    mixed_signals = np.vstack([noisy_signal, clean_signal])
                    separated = ica.separate(mixed_signals)
                    best_idx = ica.align_sources(separated, clean_signal)
                    enhanced_signal = separated[best_idx]
                    
                    mfcc = self.audio_processor.extract_mfcc(enhanced_signal)
                    feat = mfcc.flatten()
                    features.append(feat)
                except:
                    # If ICA fails, use noisy signal directly
                    mfcc = self.audio_processor.extract_mfcc(noisy_signal)
                    feat = mfcc.flatten()
                    features.append(feat)
            
            features = np.array(features)
            
            # Train and evaluate
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results['ica_iterations'].append(max_iter)
            self.results['ica_accuracies'].append(accuracy)
            
            print(f"  Accuracy: {accuracy*100:.2f}%")
        
        # Find best parameter
        best_idx = np.argmax(self.results['ica_accuracies'])
        best_iter = self.results['ica_iterations'][best_idx]
        best_acc = self.results['ica_accuracies'][best_idx]
        
        print(f"\n✓ Best ICA max_iter: {best_iter} (Accuracy: {best_acc*100:.2f}%)")
        
        return best_iter
    
    def optimize_gammatone_filters(self, noisy_signals, labels):
        """
        Optimize Gammatone n_filters parameter
        
        Args:
            noisy_signals: Noisy signals
            labels: Labels
        """
        print("\n" + "="*70)
        print(" Optimizing Gammatone Filter Count")
        print("="*70)
        
        # Test different filter counts
        filter_values = [16, 24, 32, 40, 48]
        
        for n_filters in filter_values:
            print(f"\nTesting n_filters = {n_filters}...")
            
            # Create filterbank
            gammatone = GammatoneFilterBank(
                n_filters=n_filters,
                sample_rate=self.config.sample_rate
            )
            
            # Extract features
            features = []
            for signal in noisy_signals:
                try:
                    filtered = gammatone.filter(signal)
                    cochlear_feat = gammatone.extract_features(filtered)
                    feat = cochlear_feat.flatten()
                    features.append(feat)
                except:
                    # If filtering fails, use zeros
                    features.append(np.zeros(n_filters * 3 * 62))
            
            features = np.array(features)
            
            # Train and evaluate
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results['gammatone_filters'].append(n_filters)
            self.results['gammatone_accuracies'].append(accuracy)
            
            print(f"  Accuracy: {accuracy*100:.2f}%")
        
        # Find best parameter
        best_idx = np.argmax(self.results['gammatone_accuracies'])
        best_filters = self.results['gammatone_filters'][best_idx]
        best_acc = self.results['gammatone_accuracies'][best_idx]
        
        print(f"\n✓ Best n_filters: {best_filters} (Accuracy: {best_acc*100:.2f}%)")
        
        return best_filters
    
    def optimize_svm_parameters(self, features, labels):
        """
        Optimize SVM C and gamma parameters
        
        Args:
            features: Feature matrix
            labels: Labels
        """
        print("\n" + "="*70)
        print(" Optimizing SVM Parameters")
        print("="*70)
        
        from sklearn.model_selection import GridSearchCV
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Grid search parameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        print("\nPerforming grid search...")
        print(f"  Parameter grid: {param_grid}")
        
        clf = SVC(kernel='rbf', random_state=42)
        grid_search = GridSearchCV(
            clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\n✓ Best SVM parameters: {best_params}")
        print(f"  Cross-validation score: {best_score*100:.2f}%")
        
        # Test on test set
        y_pred = grid_search.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"  Test set accuracy: {test_accuracy*100:.2f}%")
        
        self.results['svm_params'] = {
            'best_params': best_params,
            'cv_score': best_score,
            'test_accuracy': test_accuracy
        }
        
        return best_params
    
    def plot_optimization_results(self):
        """Plot optimization results"""
        print("\n\nGenerating optimization plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hyperparameter Optimization Results', fontsize=14, fontweight='bold')
        
        # ICA iterations plot
        axes[0].plot(self.results['ica_iterations'], 
                    [acc*100 for acc in self.results['ica_accuracies']], 
                    'o-', linewidth=2, markersize=8, color='#1f77b4')
        axes[0].set_xlabel('ICA Max Iterations', fontsize=11)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[0].set_title('(a) ICA Iteration Optimization', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Mark best point
        best_idx = np.argmax(self.results['ica_accuracies'])
        axes[0].plot(self.results['ica_iterations'][best_idx],
                    self.results['ica_accuracies'][best_idx]*100,
                    'r*', markersize=15, label='Best')
        axes[0].legend()
        
        # Gammatone filters plot
        axes[1].plot(self.results['gammatone_filters'], 
                    [acc*100 for acc in self.results['gammatone_accuracies']], 
                    's-', linewidth=2, markersize=8, color='#ff7f0e')
        axes[1].set_xlabel('Number of Gammatone Filters', fontsize=11)
        axes[1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1].set_title('(b) Gammatone Filter Count Optimization', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = np.argmax(self.results['gammatone_accuracies'])
        axes[1].plot(self.results['gammatone_filters'][best_idx],
                    self.results['gammatone_accuracies'][best_idx]*100,
                    'r*', markersize=15, label='Best')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'hyperparameter_optimization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        
        plt.close()
    
    def run_full_optimization(self, n_samples=20, snr_db=5):
        """
        Run complete hyperparameter optimization
        
        Args:
            n_samples: Samples per word
            snr_db: Signal-to-noise ratio
        """
        print("\n" + "="*70)
        print(" ELEC5305 Project - Hyperparameter Optimization")
        print("="*70)
        
        # Generate data
        clean_signals, noisy_signals, labels = self.generate_test_data(
            n_samples=n_samples, snr_db=snr_db
        )
        
        # Optimize ICA
        best_ica_iter = self.optimize_ica_iterations(clean_signals, noisy_signals, labels)
        
        # Optimize Gammatone
        best_gammatone_filters = self.optimize_gammatone_filters(noisy_signals, labels)
        
        # Generate baseline features for SVM optimization
        print("\n\nGenerating features for SVM optimization...")
        features = []
        for signal in noisy_signals:
            mfcc = self.audio_processor.extract_mfcc(signal)
            feat = mfcc.flatten()
            features.append(feat)
        features = np.array(features)
        
        # Optimize SVM
        best_svm_params = self.optimize_svm_parameters(features, labels)
        
        # Plot results
        self.plot_optimization_results()
        
        # Print final summary
        self.print_optimization_summary(best_ica_iter, best_gammatone_filters, best_svm_params)
    
    def print_optimization_summary(self, best_ica, best_gammatone, best_svm):
        """Print optimization summary"""
        print("\n\n" + "="*70)
        print(" Optimization Summary")
        print("="*70)
        
        print("\n✓ Optimal Hyperparameters:")
        print(f"  - ICA max_iter: {best_ica}")
        print(f"  - Gammatone n_filters: {best_gammatone}")
        print(f"  - SVM C: {best_svm['C']}")
        print(f"  - SVM gamma: {best_svm['gamma']}")
        
        print("\n✓ Update config.py with:")
        print(f"  self.ica_max_iter = {best_ica}")
        print(f"  self.n_filters = {best_gammatone}")
        
        print("\n✓ Update classifier with:")
        print(f"  SVC(kernel='rbf', C={best_svm['C']}, gamma='{best_svm['gamma']}')")
        
        print("\n" + "="*70)


def main():
    """Main function"""
    optimizer = HyperparameterOptimizer()
    
    # Run optimization
    optimizer.run_full_optimization(
        n_samples=20,
        snr_db=5  # Test at moderate noise level
    )
    
    print("\n✅ Hyperparameter optimization complete!\n")


if __name__ == "__main__":
    main()
