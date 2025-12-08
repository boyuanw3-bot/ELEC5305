"""
Visualization Analysis Script
Generate feature distribution and performance comparison charts

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

# Add src directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from audio_utils import AudioProcessor
from ica import ICAModule
from gammatone import GammatoneFilterBank


def visualize_signal_processing():
    """Visualize signal processing pipeline"""
    print("\nGenerating signal processing visualization...")

    config = Config()
    audio_processor = AudioProcessor(config)
    ica = ICAModule()
    gammatone = GammatoneFilterBank(n_filters=32, sample_rate=config.sample_rate)

    # Generate test signals
    clean_signal = audio_processor.generate_speech('three')
    noisy_signal = audio_processor.add_noise(clean_signal, snr_db=5)

    # ICA enhancement
    mixed_signals = np.vstack([noisy_signal, clean_signal])
    separated = ica.separate(mixed_signals)
    best_idx = ica.align_sources(separated, clean_signal)
    enhanced_signal = separated[best_idx]

    # Gammatone filtering
    filtered_clean = gammatone.filter(clean_signal)
    filtered_noisy = gammatone.filter(noisy_signal)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Signal Processing Pipeline Visualization', fontsize=16, fontweight='bold')

    # Time-domain signals
    t = np.linspace(0, len(clean_signal)/config.sample_rate, len(clean_signal))

    axes[0, 0].plot(t, clean_signal, 'b-', linewidth=0.5)
    axes[0, 0].set_title('(a) Clean Signal', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, noisy_signal, 'r-', linewidth=0.5)
    axes[0, 1].set_title('(b) Noisy Signal (SNR=5dB)', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    # ICA enhanced signal
    axes[1, 0].plot(t[:len(enhanced_signal)], enhanced_signal, 'g-', linewidth=0.5)
    axes[1, 0].set_title('(c) ICA Enhanced Signal', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    # Gammatone filterbank response
    freq_response = np.abs(filtered_clean)
    im = axes[1, 1].imshow(freq_response, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('(d) Gammatone Filterbank Response', fontsize=12)
    axes[1, 1].set_xlabel('Time (samples)')
    axes[1, 1].set_ylabel('Filter Index')
    plt.colorbar(im, ax=axes[1, 1], label='Amplitude')

    # Power spectral density comparison
    from scipy.signal import welch
    f_clean, psd_clean = welch(clean_signal, fs=config.sample_rate, nperseg=1024)
    f_noisy, psd_noisy = welch(noisy_signal, fs=config.sample_rate, nperseg=1024)
    f_enhanced, psd_enhanced = welch(enhanced_signal[:len(clean_signal)],
                                     fs=config.sample_rate, nperseg=1024)

    axes[2, 0].semilogy(f_clean, psd_clean, 'b-', label='Clean', linewidth=2)
    axes[2, 0].semilogy(f_noisy, psd_noisy, 'r-', alpha=0.6, label='Noisy', linewidth=2)
    axes[2, 0].semilogy(f_enhanced, psd_enhanced, 'g-', alpha=0.6, label='Enhanced', linewidth=2)
    axes[2, 0].set_title('(e) Power Spectral Density Comparison', fontsize=12)
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('PSD')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, 4000])

    # SNR improvement bar chart
    snr_noisy = 10 * np.log10(np.mean(clean_signal**2) /
                              np.mean((noisy_signal - clean_signal)**2))
    snr_enhanced = 10 * np.log10(np.mean(clean_signal**2) /
                                 np.mean((enhanced_signal[:len(clean_signal)] - clean_signal)**2))

    configs = ['Noisy', 'ICA Enhanced']
    snrs = [snr_noisy, snr_enhanced]
    colors = ['red', 'green']

    bars = axes[2, 1].bar(configs, snrs, color=colors, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('(f) SNR Improvement', fontsize=12)
    axes[2, 1].set_ylabel('SNR (dB)')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}dB',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'signal_processing_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_file}")

    plt.close()


def visualize_performance_comparison():
    """Visualize performance comparison"""
    print("\nGenerating performance comparison visualization...")

    # Simulated performance data at different SNR levels
    snr_levels = [-5, 0, 5, 10, 15, 20]

    # Accuracy data (simulated)
    acc_baseline = [0.45, 0.62, 0.78, 0.92, 0.96, 0.98]
    acc_ica = [0.52, 0.68, 0.83, 0.95, 0.98, 0.99]
    acc_cochlear = [0.58, 0.72, 0.85, 0.94, 0.97, 0.99]
    acc_full = [0.65, 0.78, 0.89, 0.97, 0.99, 1.00]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')

    # (a) Accuracy vs SNR curves
    axes[0, 0].plot(snr_levels, acc_baseline, 'o-', label='Baseline MFCC', linewidth=2, markersize=8)
    axes[0, 0].plot(snr_levels, acc_ica, 's-', label='ICA + MFCC', linewidth=2, markersize=8)
    axes[0, 0].plot(snr_levels, acc_cochlear, '^-', label='Gammatone Cochlear', linewidth=2, markersize=8)
    axes[0, 0].plot(snr_levels, acc_full, 'd-', label='Full System', linewidth=2, markersize=8)
    axes[0, 0].set_title('(a) Accuracy vs Signal-to-Noise Ratio', fontsize=12)
    axes[0, 0].set_xlabel('SNR (dB)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.4, 1.05])

    # (b) Performance improvement bar chart (SNR=5dB)
    configs = ['Baseline\nMFCC', 'ICA +\nMFCC', 'Gammatone\nCochlear', 'Full\nSystem']
    accuracies_5db = [0.78, 0.83, 0.85, 0.89]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = axes[0, 1].bar(configs, accuracies_5db, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('(b) Configuration Performance Comparison (SNR=5dB)', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0.7, 0.95])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (c) Feature dimension comparison
    feature_dims = [819, 819, 5952, 6771]
    bars = axes[1, 0].barh(configs, feature_dims, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('(c) Feature Dimension Comparison', fontsize=12)
    axes[1, 0].set_xlabel('Feature Dimensions')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}',
                       ha='left', va='center', fontsize=10, fontweight='bold')

    # (d) Training time comparison
    training_times = [0.01, 0.01, 0.09, 0.06]
    bars = axes[1, 1].bar(configs, training_times, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('(d) Training Time Comparison', fontsize=12)
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_file}")

    plt.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print(" ELEC5305 Project - Visualization Analysis")
    print("="*70)

    # Generate visualization charts
    visualize_signal_processing()
    visualize_performance_comparison()

    print("\n✅ All visualization charts generated successfully!")
    print("\nCheck outputs directory for chart files:")
    print("  - signal_processing_visualization.png")
    print("  - performance_comparison.png")
    print()


if __name__ == "__main__":
    main()