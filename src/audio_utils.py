"""
Audio Processing Utility Module
Provides functions for speech signal generation, loading, noise adding,
and MFCC feature extraction.

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
import librosa


class AudioProcessor:
    """
    AudioProcessor
    Handles speech generation, feature extraction, and noise simulation.
    """

    def __init__(self, config):
        """
        Initialize the AudioProcessor.

        Args:
            config: A configuration object containing:
                - sample_rate
                - word_to_freq (mapping words → base frequencies)
                - duration
                - n_mfcc, n_fft, hop_length, n_mels
        """
        self.config = config
        self.sample_rate = config.sample_rate

    def generate_speech(self, word):
        """
        Generate synthetic speech using additive sinusoidal components
        + ADSR amplitude envelope.

        Args:
            word (str): Target word (“zero”~“nine”)

        Returns:
            np.ndarray: Generated speech signal
        """

        if word not in self.config.word_to_freq:
            raise ValueError(f"Unknown word: {word}")

        # Base frequency for the word
        f0 = self.config.word_to_freq[word]

        # Time axis
        duration = self.config.duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # Construct harmonic-rich synthetic speech
        signal = np.sin(2 * np.pi * f0 * t)              # Fundamental
        signal += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)   # 2nd overtone
        signal += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)   # 3rd overtone

        # ADSR Envelope Design
        attack = int(0.05 * self.sample_rate)    # 50 ms
        decay = int(0.1 * self.sample_rate)      # 100 ms
        sustain = len(t) - attack - decay - int(0.1 * self.sample_rate)
        release = int(0.1 * self.sample_rate)    # 100 ms

        envelope = np.concatenate([
            np.linspace(0, 1, attack),           # Attack
            np.linspace(1, 0.7, decay),          # Decay
            np.ones(sustain) * 0.7,              # Sustain
            np.linspace(0.7, 0, release)         # Release
        ])

        signal = signal * envelope

        # Normalize to [-1, 1]
        signal = signal / np.max(np.abs(signal))

        return signal

    def extract_mfcc(self, signal):
        """
        Compute MFCC features from audio.

        Args:
            signal (np.ndarray): Input audio

        Returns:
            np.ndarray: (n_mfcc × frames) MFCC feature matrix
        """

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )

        return mfcc

    def add_noise(self, signal, snr_db):
        """
        Add white noise to achieve a target SNR.

        Args:
            signal (np.ndarray): Clean audio
            snr_db (float): Desired signal-to-noise ratio (dB)

        Returns:
            np.ndarray: Noisy audio
        """

        # Signal power
        signal_power = np.mean(signal ** 2)

        # Generate white Gaussian noise
        noise = np.random.randn(len(signal))
        noise_power = np.mean(noise ** 2)

        # Scale noise to match target SNR
        noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)

        noisy_signal = signal + noise_scale * noise

        return noisy_signal


if __name__ == "__main__":
    print("=" * 70)
    print(" Testing Audio Processing Module")
    print("=" * 70)

    from config import Config

    config = Config()
    processor = AudioProcessor(config)

    print("\n1. Generating synthetic speech...")
    word = "three"
    signal = processor.generate_speech(word)
    print(f"  Word: {word}")
    print(f"  Signal length: {len(signal)} samples")
    print(f"  Signal range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")

    print("\n2. Extracting MFCC features...")
    mfcc = processor.extract_mfcc(signal)
    print(f"  MFCC shape: {mfcc.shape}")
    print(f"  Number of coefficients: {mfcc.shape[0]}")
    print(f"  Number of frames: {mfcc.shape[1]}")

    print("\n3. Adding noise...")
    snr_db = 10
    noisy_signal = processor.add_noise(signal, snr_db)
    print(f"  SNR (target): {snr_db} dB")
    print(f"  Noisy signal length: {len(noisy_signal)}")

    # Compute actual achieved SNR
    noise = noisy_signal - signal
    actual_snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    print(f"  Actual SNR: {actual_snr:.2f} dB")

    print("\n✅ Audio processing module test completed successfully!")
    print("=" * 70)
