"""
Gammatone Cochlear Filterbank Module
Implements a biologically inspired auditory filterbank based on Gammatone filters.

Author: ELEC5305 Course Project
Date: 2024
"""

import numpy as np
from scipy.signal import gammatone, lfilter


class GammatoneFilterBank:
    """Gammatone cochlear filterbank"""

    def __init__(self, n_filters=32, sample_rate=16000, freq_min=50.0, freq_max=7500.0):
        """
        Initialize the Gammatone filterbank.

        Args:
            n_filters (int): Number of filters.
            sample_rate (int): Sampling rate (Hz).
            freq_min (float): Minimum center frequency (Hz).
            freq_max (float): Maximum center frequency (Hz).
        """
        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Generate center frequencies on the ERB scale
        self.center_freqs = self._erb_space(freq_min, freq_max, n_filters)

        # Ensure all center frequencies are below the Nyquist frequency
        nyquist = sample_rate / 2
        self.center_freqs = np.clip(self.center_freqs, freq_min, nyquist * 0.99)

        # Initialize filter coefficients
        self.filters = []
        for fc in self.center_freqs:
            b, a = gammatone(fc, 'fir', fs=sample_rate)
            self.filters.append((b, a))

    def _erb_space(self, freq_min, freq_max, n_filters):
        """
        Generate evenly spaced frequencies on the ERB scale.

        Args:
            freq_min (float): Minimum frequency (Hz)
            freq_max (float): Maximum frequency (Hz)
            n_filters (int): Number of filters

        Returns:
            np.ndarray: Array of center frequencies
        """

        def freq_to_erb(freq):
            return 24.7 * (4.37 * freq / 1000 + 1)

        def erb_to_freq(erb):
            return (erb / 24.7 - 1) * 1000 / 4.37

        erb_min = freq_to_erb(freq_min)
        erb_max = freq_to_erb(freq_max)
        erb_points = np.linspace(erb_min, erb_max, n_filters)

        freqs = np.array([erb_to_freq(e) for e in erb_points])
        return freqs

    def filter(self, signal):
        """
        Apply the Gammatone filterbank to an input signal.

        Args:
            signal (np.ndarray): Input signal (n_samples,)

        Returns:
            np.ndarray: Filtered output of shape (n_filters, n_samples)
        """
        filtered_signals = []
        for b, a in self.filters:
            filtered = lfilter(b, a, signal)
            filtered_signals.append(filtered)

        return np.array(filtered_signals)

    def extract_features(self, filtered_signals, n_frames=None, hop_length=256):
        """
        Extract cochlear-inspired features from the filtered outputs.

        Args:
            filtered_signals (np.ndarray): Filter outputs (n_filters, n_samples)
            n_frames (int): Number of frames (None = auto)
            hop_length (int): Hop length for framing

        Returns:
            np.ndarray: Feature matrix (n_frames, n_features)
                        containing Static + Delta + Delta-Delta
        """
        n_filters, n_samples = filtered_signals.shape

        if n_frames is None:
            n_frames = (n_samples - hop_length) // hop_length + 1

        envelopes = np.abs(filtered_signals)

        # Extract static energy features
        features_static = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + hop_length
            if end > n_samples:
                break

            frame_features = np.mean(envelopes[:, start:end], axis=1)
            features_static.append(frame_features)

        features_static = np.array(features_static)

        # Compute delta features (1st derivative)
        features_delta = np.zeros_like(features_static)
        features_delta[1:-1] = (features_static[2:] - features_static[:-2]) / 2
        features_delta[0] = features_static[1] - features_static[0]
        features_delta[-1] = features_static[-1] - features_static[-2]

        # Compute delta-delta (2nd derivative)
        features_delta_delta = np.zeros_like(features_delta)
        features_delta_delta[1:-1] = (features_delta[2:] - features_delta[:-2]) / 2
        features_delta_delta[0] = features_delta[1] - features_delta[0]
        features_delta_delta[-1] = features_delta[-1] - features_delta[-2]

        features = np.hstack([
            features_static,
            features_delta,
            features_delta_delta
        ])

        return features


def test_gammatone_module():
    """Test the Gammatone filterbank module"""
    print("=" * 70)
    print(" Testing Gammatone Cochlear Filterbank Module")
    print("=" * 70)

    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    print("\n1. Creating test signal...")
    signal = (np.sin(2 * np.pi * 200 * t) +
              np.sin(2 * np.pi * 500 * t) +
              np.sin(2 * np.pi * 1000 * t) +
              np.sin(2 * np.pi * 2000 * t))

    print(f"  Signal length: {len(signal)} samples")
    print("  Included frequencies: 200 Hz, 500 Hz, 1000 Hz, 2000 Hz")

    print("\n2. Creating Gammatone filterbank...")
    gammatone_bank = GammatoneFilterBank()

    print(f"  Number of filters: {gammatone_bank.n_filters}")
    print(f"  Center frequency range: {gammatone_bank.center_freqs[0]:.1f}–{gammatone_bank.center_freqs[-1]:.1f} Hz")

    print("\n3. Applying filterbank...")
    filtered = gammatone_bank.filter(signal)
    print(f"  Filtered output shape: {filtered.shape}")

    print("\n4. Extracting features...")
    features = gammatone_bank.extract_features(filtered)
    print(f"  Feature matrix shape: {features.shape}")

    print("\n5. Feature statistics...")
    print(f"  Mean: {np.mean(features):.6f}")
    print(f"  Std: {np.std(features):.6f}")

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("\n  ⚠ Feature matrix contains NaN or Inf values")
    else:
        print("\n  ✓ Feature quality check passed (No NaN/Inf)")

    print("\n✓ Gammatone module test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_gammatone_module()
