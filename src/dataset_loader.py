"""
Real Dataset Loader Module
Load and preprocess real speech datasets (e.g., Common Voice, LibriSpeech)

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import librosa
import os
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class RealDatasetLoader:
    """Real speech dataset loader"""
    
    def __init__(self, dataset_path: str, sample_rate: int = 16000):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to dataset directory
            sample_rate: Target sample rate
        """
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.audio_files = []
        self.labels = []
        
    def load_dataset(self, max_samples_per_class: int = 50) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load audio files from dataset
        
        Args:
            max_samples_per_class: Maximum samples per class
            
        Returns:
            signals: List of audio signals
            labels: List of corresponding labels
        """
        print(f"\nLoading dataset from: {self.dataset_path}")
        
        signals = []
        labels = []
        
        # Assuming directory structure: dataset_path/class_name/*.wav
        class_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
        
        if not class_dirs:
            print("  Warning: No class directories found")
            return signals, labels
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            audio_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
            
            # Limit samples per class
            audio_files = audio_files[:max_samples_per_class]
            
            print(f"  Class {class_idx} ({class_name}): {len(audio_files)} files")
            
            for audio_file in audio_files:
                try:
                    # Load audio
                    signal, sr = librosa.load(audio_file, sr=self.sample_rate)
                    
                    # Ensure minimum length (1 second)
                    if len(signal) < self.sample_rate:
                        signal = np.pad(signal, (0, self.sample_rate - len(signal)))
                    
                    # Truncate to fixed length (1 second)
                    signal = signal[:self.sample_rate]
                    
                    # Normalize
                    signal = signal / (np.max(np.abs(signal)) + 1e-8)
                    
                    signals.append(signal)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"    Error loading {audio_file.name}: {e}")
                    continue
        
        print(f"\n  Total samples loaded: {len(signals)}")
        print(f"  Number of classes: {len(class_dirs)}")
        
        return signals, labels
    
    def add_noise_to_dataset(self, signals: List[np.ndarray], 
                            snr_db: float) -> List[np.ndarray]:
        """
        Add white Gaussian noise to all signals
        
        Args:
            signals: List of clean signals
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            noisy_signals: List of noisy signals
        """
        print(f"\nAdding noise (SNR={snr_db} dB)...")
        
        noisy_signals = []
        
        for signal in signals:
            # Calculate signal power
            signal_power = np.mean(signal ** 2)
            
            # Generate noise
            noise = np.random.randn(len(signal))
            noise_power = np.mean(noise ** 2)
            
            # Calculate noise scale
            noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
            
            # Add noise
            noisy_signal = signal + noise_scale * noise
            noisy_signals.append(noisy_signal)
        
        return noisy_signals
    
    def save_preprocessed_dataset(self, signals: List[np.ndarray], 
                                  labels: List[int],
                                  output_path: str):
        """
        Save preprocessed dataset
        
        Args:
            signals: List of audio signals
            labels: List of labels
            output_path: Output file path (.npz)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_path, signals=signals, labels=labels)
        print(f"\n✓ Dataset saved to: {output_path}")
    
    def load_preprocessed_dataset(self, input_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load preprocessed dataset
        
        Args:
            input_path: Input file path (.npz)
            
        Returns:
            signals: List of audio signals
            labels: List of labels
        """
        data = np.load(input_path, allow_pickle=True)
        signals = data['signals'].tolist()
        labels = data['labels'].tolist()
        
        print(f"\n✓ Dataset loaded from: {input_path}")
        print(f"  Samples: {len(signals)}")
        print(f"  Classes: {len(set(labels))}")
        
        return signals, labels


def create_synthetic_dataset_structure(output_dir: str, 
                                       vocabulary: List[str],
                                       samples_per_word: int = 30,
                                       sample_rate: int = 16000):
    """
    Create synthetic dataset with directory structure for testing
    
    Args:
        output_dir: Output directory path
        vocabulary: List of words
        samples_per_word: Number of samples per word
        sample_rate: Sample rate
    """
    from audio_utils import AudioProcessor
    from config import Config
    
    print("\nCreating synthetic dataset structure...")
    
    output_path = Path(output_dir)
    config = Config()
    audio_processor = AudioProcessor(config)
    
    for word in vocabulary:
        word_dir = output_path / word
        word_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_word):
            # Generate signal
            signal = audio_processor.generate_speech(word)
            
            # Add slight variation
            variation = np.random.randn(len(signal)) * 0.05
            signal = signal + variation
            
            # Save as wav file
            import soundfile as sf
            filename = word_dir / f"{word}_{i:03d}.wav"
            sf.write(filename, signal, sample_rate)
        
        print(f"  ✓ Created {samples_per_word} samples for '{word}'")
    
    print(f"\n✓ Synthetic dataset created at: {output_path}")


# Test module
if __name__ == "__main__":
    print("=" * 70)
    print(" Real Dataset Loader - Test")
    print("=" * 70)
    
    # Example: Create synthetic dataset for testing
    output_dir = "../data/synthetic_dataset"
    vocabulary = ['zero', 'one', 'two', 'three', 'four']
    
    print("\n1. Creating synthetic test dataset...")
    try:
        create_synthetic_dataset_structure(
            output_dir=output_dir,
            vocabulary=vocabulary,
            samples_per_word=10
        )
    except ImportError:
        print("  Note: soundfile not installed. Skipping dataset creation.")
        print("  Install with: pip install soundfile")
    
    # Example: Load dataset
    print("\n2. Loading dataset...")
    loader = RealDatasetLoader(dataset_path=output_dir, sample_rate=16000)
    
    if Path(output_dir).exists():
        signals, labels = loader.load_dataset(max_samples_per_class=10)
        
        if signals:
            print("\n3. Dataset statistics:")
            print(f"  Number of samples: {len(signals)}")
            print(f"  Number of classes: {len(set(labels))}")
            print(f"  Signal length: {len(signals[0])} samples")
            print(f"  Sample rate: {loader.sample_rate} Hz")
            
            # Add noise
            noisy_signals = loader.add_noise_to_dataset(signals, snr_db=10)
            print(f"  Noisy signals generated: {len(noisy_signals)}")
            
            print("\n✅ Dataset loader test successful!")
    else:
        print(f"  Dataset path not found: {output_dir}")
        print("  Please create a dataset first or update the path.")
    
    print("\n" + "=" * 70)
