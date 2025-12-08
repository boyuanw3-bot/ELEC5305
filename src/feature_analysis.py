"""
Feature Visualization and Analysis
Analyze and visualize extracted features from different methods

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from audio_utils import AudioProcessor
from ica import ICAModule
from gammatone import GammatoneFilterBank
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """Advanced feature analysis and visualization"""
    
    def __init__(self):
        self.config = Config()
        self.audio_processor = AudioProcessor(self.config)
        self.ica_module = ICAModule()
        self.gammatone = GammatoneFilterBank(n_filters=32, sample_rate=self.config.sample_rate)
    
    def generate_data(self, n_samples=50, snr_db=5):
        """Generate test data"""
        print(f"\nGenerating data for feature analysis...")
        print(f"  Samples per class: {n_samples}")
        print(f"  SNR: {snr_db} dB")
        
        clean_signals = []
        noisy_signals = []
        labels = []
        word_names = []
        
        for word_idx, word in enumerate(self.config.vocabulary):
            for i in range(n_samples):
                signal = self.audio_processor.generate_speech(word)
                clean_signals.append(signal)
                labels.append(word_idx)
                word_names.append(word)
                
                noisy = self.audio_processor.add_noise(signal, snr_db)
                noisy_signals.append(noisy)
        
        print(f"  ✓ Generated {len(clean_signals)} samples")
        return clean_signals, noisy_signals, np.array(labels), word_names
    
    def extract_all_features(self, clean_signals, noisy_signals):
        """Extract features using all methods"""
        print("\nExtracting features from all methods...")
        
        # Baseline MFCC
        features_baseline = []
        for signal in noisy_signals:
            mfcc = self.audio_processor.extract_mfcc(signal)
            features_baseline.append(mfcc.flatten())
        features_baseline = np.array(features_baseline)
        print(f"  ✓ Baseline MFCC: {features_baseline.shape}")
        
        # ICA + MFCC
        features_ica = []
        for noisy, clean in zip(noisy_signals, clean_signals):
            mixed = np.vstack([noisy, clean])
            separated = self.ica_module.separate(mixed)
            best_idx = self.ica_module.align_sources(separated, clean)
            enhanced = separated[best_idx]
            mfcc = self.audio_processor.extract_mfcc(enhanced)
            features_ica.append(mfcc.flatten())
        features_ica = np.array(features_ica)
        print(f"  ✓ ICA + MFCC: {features_ica.shape}")
        
        # Gammatone
        features_gammatone = []
        for signal in noisy_signals:
            filtered = self.gammatone.filter(signal)
            feat = self.gammatone.extract_features(filtered)
            features_gammatone.append(feat.flatten())
        features_gammatone = np.array(features_gammatone)
        print(f"  ✓ Gammatone: {features_gammatone.shape}")
        
        # Full system
        features_full = []
        for noisy, clean in zip(noisy_signals, clean_signals):
            mixed = np.vstack([noisy, clean])
            separated = self.ica_module.separate(mixed)
            best_idx = self.ica_module.align_sources(separated, clean)
            enhanced = separated[best_idx]
            
            filtered = self.gammatone.filter(enhanced)
            cochlear_feat = self.gammatone.extract_features(filtered)
            mfcc = self.audio_processor.extract_mfcc(enhanced)
            
            feat_c = cochlear_feat.flatten()
            feat_m = mfcc.flatten()
            feat_c = (feat_c - feat_c.mean()) / (feat_c.std() + 1e-8)
            feat_m = (feat_m - feat_m.mean()) / (feat_m.std() + 1e-8)
            
            features_full.append(np.concatenate([feat_c, feat_m]))
        features_full = np.array(features_full)
        print(f"  ✓ Full System: {features_full.shape}")
        
        return {
            'baseline': features_baseline,
            'ica': features_ica,
            'gammatone': features_gammatone,
            'full': features_full
        }
    
    def visualize_feature_space_pca(self, features_dict, labels):
        """Visualize feature space using PCA"""
        print("\nGenerating PCA visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Feature Space Visualization (PCA)', fontsize=14, fontweight='bold')
        
        method_names = {
            'baseline': 'Baseline MFCC',
            'ica': 'ICA + MFCC',
            'gammatone': 'Gammatone Cochlear',
            'full': 'Full System'
        }
        
        for idx, (method, features) in enumerate(features_dict.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Apply PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # Plot
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                               c=labels, cmap='tab10', alpha=0.6, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
            ax.set_title(f'{method_names[method]}', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            if idx == 3:
                legend = ax.legend(*scatter.legend_elements(),
                                 title="Digits", loc='center left',
                                 bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'feature_space_pca.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        plt.close()
    
    def visualize_feature_space_tsne(self, features_dict, labels):
        """Visualize feature space using t-SNE"""
        print("\nGenerating t-SNE visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Feature Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
        
        method_names = {
            'baseline': 'Baseline MFCC',
            'ica': 'ICA + MFCC',
            'gammatone': 'Gammatone Cochlear',
            'full': 'Full System'
        }
        
        for idx, (method, features) in enumerate(features_dict.items()):
            print(f"  Processing {method}...")
            ax = axes[idx // 2, idx % 2]
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)
            
            # Plot
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                               c=labels, cmap='tab10', alpha=0.6, s=50)
            
            ax.set_xlabel('t-SNE 1', fontsize=10)
            ax.set_ylabel('t-SNE 2', fontsize=10)
            ax.set_title(f'{method_names[method]}', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if idx == 3:
                legend = ax.legend(*scatter.legend_elements(),
                                 title="Digits", loc='center left',
                                 bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'feature_space_tsne.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        plt.close()
    
    def analyze_feature_separability(self, features_dict, labels):
        """Analyze class separability using various metrics"""
        print("\nAnalyzing feature separability...")
        
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import pdist, squareform
        
        results = {}
        
        for method, features in features_dict.items():
            # Silhouette score (higher is better, range: -1 to 1)
            silhouette = silhouette_score(features, labels)
            
            # Within-class variance
            within_var = []
            for label in np.unique(labels):
                class_features = features[labels == label]
                within_var.append(np.var(class_features))
            avg_within_var = np.mean(within_var)
            
            # Between-class distance
            class_centers = []
            for label in np.unique(labels):
                class_features = features[labels == label]
                class_centers.append(np.mean(class_features, axis=0))
            class_centers = np.array(class_centers)
            between_dist = np.mean(pdist(class_centers))
            
            results[method] = {
                'silhouette': silhouette,
                'within_variance': avg_within_var,
                'between_distance': between_dist
            }
            
            print(f"\n  {method}:")
            print(f"    Silhouette Score: {silhouette:.4f}")
            print(f"    Within-class Variance: {avg_within_var:.4f}")
            print(f"    Between-class Distance: {between_dist:.4f}")
        
        # Plot comparison
        self.plot_separability_comparison(results)
        
        return results
    
    def plot_separability_comparison(self, results):
        """Plot separability metrics comparison"""
        print("\nGenerating separability comparison plot...")
        
        methods = list(results.keys())
        method_names = {
            'baseline': 'Baseline',
            'ica': 'ICA+MFCC',
            'gammatone': 'Gammatone',
            'full': 'Full System'
        }
        
        silhouettes = [results[m]['silhouette'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar([method_names[m] for m in methods], silhouettes,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Feature Space Separability Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_file = output_dir / 'feature_separability.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_file}")
        plt.close()
    
    def run_complete_analysis(self, n_samples=50, snr_db=5):
        """Run complete feature analysis"""
        print("\n" + "="*70)
        print(" Feature Analysis and Visualization")
        print("="*70)
        
        # Generate data
        clean, noisy, labels, word_names = self.generate_data(n_samples, snr_db)
        
        # Extract features
        features_dict = self.extract_all_features(clean, noisy)
        
        # Visualizations
        self.visualize_feature_space_pca(features_dict, labels)
        self.visualize_feature_space_tsne(features_dict, labels)
        
        # Separability analysis
        sep_results = self.analyze_feature_separability(features_dict, labels)
        
        print("\n" + "="*70)
        print(" Feature Analysis Complete")
        print("="*70)
        print("\n✓ Generated outputs:")
        print("  - feature_space_pca.png")
        print("  - feature_space_tsne.png")
        print("  - feature_separability.png")
        print("="*70)


def main():
    """Main function"""
    analyzer = FeatureAnalyzer()
    analyzer.run_complete_analysis(n_samples=50, snr_db=5)
    print("\n✅ Feature analysis complete!\n")


if __name__ == "__main__":
    main()
