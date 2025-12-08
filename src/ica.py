"""
ICA module for blind source separation.

Implements a small wrapper around sklearn's FastICA plus
some simple utilities for generating a demo mixture and
checking that separation is working.
"""

import numpy as np
from sklearn.decomposition import FastICA


class ICAModule:
    """
    Wrapper class for FastICA blind source separation.
    """

    def __init__(self, n_components=None, max_iter=1000, tol=1e-4, random_state=42):
        """
        Parameters
        ----------
        n_components : int or None
            Number of independent components to estimate.
            If None, defaults to the number of input channels.
        max_iter : int
            Maximum number of iterations for the FastICA algorithm.
        tol : float
            Convergence tolerance.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self._ica = None

    def fit(self, mixed_signals):
        """
        Fit the ICA model on mixed signals.

        Parameters
        ----------
        mixed_signals : np.ndarray, shape (n_channels, n_samples)
            Observed mixtures (each row is a channel / microphone).

        Returns
        -------
        self : ICAModule
            Fitted instance.
        """
        mixed_signals = np.asarray(mixed_signals)
        if mixed_signals.ndim != 2:
            raise ValueError("mixed_signals must have shape (n_channels, n_samples)")

        n_channels, _ = mixed_signals.shape
        n_components = self.n_components or n_channels

        # sklearn expects shape (n_samples, n_features)
        X = mixed_signals.T

        self._ica = FastICA(
            n_components=n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._ica.fit(X)
        return self

    def separate(self, mixed_signals):
        """
        Run ICA on the mixed signals and return separated sources.

        Parameters
        ----------
        mixed_signals : np.ndarray, shape (n_channels, n_samples)
            Observed mixtures.

        Returns
        -------
        separated : np.ndarray, shape (n_components, n_samples)
            Estimated source signals (each row is one source).
        """
        mixed_signals = np.asarray(mixed_signals)
        if mixed_signals.ndim != 2:
            raise ValueError("mixed_signals must have shape (n_channels, n_samples)")

        if self._ica is None:
            # Fit on the provided mixture if not already fitted
            self.fit(mixed_signals)

        X = mixed_signals.T  # (n_samples, n_channels)
        S = self._ica.transform(X)  # (n_samples, n_components)
        separated = S.T  # (n_components, n_samples)
        return separated

    @staticmethod
    def align_sources(estimated, references):
        """
        Align estimated sources to reference sources using correlation.

        This is mainly for demo/visualization purposes: ICA recovers
        sources up to scaling and permutation. Here we reorder and flip
        signs so that each estimated signal best matches a reference
        signal in terms of absolute correlation.

        Parameters
        ----------
        estimated : np.ndarray, shape (n_sources_est, n_samples)
            Estimated source signals.
        references : np.ndarray, shape (n_sources_ref, n_samples)
            Ground-truth reference sources.

        Returns
        -------
        aligned : np.ndarray, shape (n_sources_ref, n_samples)
            Estimated sources reordered and sign-corrected to match
            the reference ordering as closely as possible.
        perm : list[int]
            The permutation used to align estimated -> reference.
        signs : list[int]
            The sign (+1 or -1) applied to each estimated source.
        """
        estimated = np.asarray(estimated)
        references = np.asarray(references)

        if estimated.shape[1] != references.shape[1]:
            raise ValueError("estimated and references must have the same number of samples")

        n_est, n_samples = estimated.shape
        n_ref, _ = references.shape

        if n_est < n_ref:
            raise ValueError("estimated has fewer sources than references")

        # Normalise to zero mean, unit variance for correlation
        est_norm = (estimated - estimated.mean(axis=1, keepdims=True)) / (
            estimated.std(axis=1, keepdims=True) + 1e-10
        )
        ref_norm = (references - references.mean(axis=1, keepdims=True)) / (
            references.std(axis=1, keepdims=True) + 1e-10
        )

        # Correlation matrix: (n_ref, n_est)
        corr = ref_norm @ est_norm.T / (n_samples - 1)

        perm = []
        signs = []
        used_est = set()

        # Greedy matching: for each reference, pick the best remaining estimated source
        for i in range(n_ref):
            # For ref i, find best estimated index not used yet
            best_j = None
            best_val = -np.inf
            for j in range(n_est):
                if j in used_est:
                    continue
                val = abs(corr[i, j])
                if val > best_val:
                    best_val = val
                    best_j = j

            if best_j is None:
                raise RuntimeError("Matching failed; not enough estimated sources")

            used_est.add(best_j)
            perm.append(best_j)
            signs.append(1 if corr[i, best_j] >= 0 else -1)

        aligned = np.zeros_like(references)
        for i, (j, s) in enumerate(zip(perm, signs)):
            aligned[i] = s * estimated[j]

        return aligned, perm, signs


def _make_demo_signals(duration=1.0, sample_rate=16000):
    """
    Generate two synthetic demo sources for testing ICA.

    Returns
    -------
    s1, s2 : np.ndarray
        Two clean source signals (each shape (n_samples,)).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Source 1: a simple chirp-like sinusoid
    s1 = np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))

    # Source 2: a square-wave-like signal using sign of a sinusoid
    s2 = np.sign(np.sin(2 * np.pi * 110 * t))
    s2 = librosa.effects.preemphasis(s2) if 'librosa' in globals() else s2  # optional pre-emphasis

    # Normalise
    s1 /= np.max(np.abs(s1)) + 1e-10
    s2 /= np.max(np.abs(s2)) + 1e-10

    return s1, s2


def test_ica_module():
    """
    Simple self-test for the ICAModule.

    This function:
      1. Generates two demo sources.
      2. Mixes them with a 2x2 mixing matrix.
      3. Runs ICA to separate them.
      4. Aligns estimated sources to references.
      5. Computes a simple SNR improvement metric (no mir_eval).
    """
    print("=" * 70)
    print(" Testing ICA module")
    print("=" * 70)

    # 1. Generate demo sources
    sr = 16000
    s1, s2 = _make_demo_signals(duration=1.0, sample_rate=sr)
    sources = np.vstack([s1, s2])
    print("\n1. Generated demo sources:")
    print(f"   Number of sources: {sources.shape[0]}")
    print(f"   Number of samples: {sources.shape[1]}")

    # 2. Mix with a 2x2 matrix
    A = np.array([[0.8, 0.6],
                  [0.4, 0.9]])
    mixed = A @ sources
    print("\n2. Created synthetic mixture:")
    print(f"   Mixing matrix A:\n{A}")
    print(f"   Mixed shape: {mixed.shape} (channels x samples)")

    # 3. Run ICA
    print("\n3. Running FastICA blind separation...")
    ica = ICAModule(n_components=2, max_iter=1000, tol=1e-4, random_state=42)
    separated = ica.separate(mixed)
    print(f"   Separated shape: {separated.shape} (sources x samples)")

    # 4. Align sources to references using correlation
    print("\n4. Aligning estimated sources to references...")
    aligned, perm, signs = ICAModule.align_sources(separated, sources)
    print(f"   Permutation used: {perm}")
    print(f"   Signs used:       {signs}")

    # Correlation before/after alignment
    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    corr_before = [[corr(separated[i], sources[j]) for j in range(2)] for i in range(2)]
    corr_after = [corr(aligned[i], sources[i]) for i in range(2)]

    print("\n   Correlations before alignment (rows=est, cols=ref):")
    for i in range(2):
        print(f"     est {i}: {corr_before[i][0]:>+6.3f}, {corr_before[i][1]:>+6.3f}")

    print("   Correlations after alignment (est -> matching ref):")
    for i in range(2):
        print(f"     source {i}: {corr_after[i]:>+6.3f}")

    # 5. Simple SNR improvement (no mir_eval)
    print("\n5. Evaluating simple SNR improvement (no mir_eval)...")

    def snr(ref, est, noise):
        """Compute SNR = 10 log10(P_ref / P_noise)."""
        p_ref = np.mean(ref ** 2)
        p_noise = np.mean(noise ** 2) + 1e-10
        return 10 * np.log10(p_ref / p_noise)

    # For source 1, compare channel-1 mixture vs aligned estimate
    mix_ch1 = mixed[0]
    mix_noise_1 = mix_ch1 - s1
    sep_noise_1 = aligned[0] - s1

    snr_mixed_1 = snr(s1, mix_ch1, mix_noise_1)
    snr_sep_1 = snr(s1, aligned[0], sep_noise_1)

    print(f"   Source 1 SNR (mixed):    {snr_mixed_1:>6.2f} dB")
    print(f"   Source 1 SNR (separated):{snr_sep_1:>6.2f} dB")
    print(f"   ΔSNR (improvement):      {snr_sep_1 - snr_mixed_1:>6.2f} dB")

    print("\n✅ ICA module self-test completed.")
    print("=" * 70)


if __name__ == "__main__":
    test_ica_module()
