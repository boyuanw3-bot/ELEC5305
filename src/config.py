


class Config:
    """
    Centralized configuration for all experimental parameters.

    This class encapsulates all hyperparameters, paths, and settings used
    throughout the project, ensuring reproducibility and easy parameter tuning.

    Design Philosophy:
        - Single source of truth for all parameters
        - Literature-backed default values
        - Clear documentation for each parameter
        - Easy to modify for ablation studies

    Usage:
        >>> config = Config()
        >>> sr = config.sample_rate
        >>> n_mfcc = config.n_mfcc
    """

    def __init__(self):
        # ====================================================================
        # Audio Parameters
        # Following standard speech processing conventions
        # ====================================================================

        self.sample_rate = 16000  # Sampling rate in Hz
        # Rationale: 16 kHz is standard for speech (sufficient for 8 kHz bandwidth)
        # Reference: Telecommunication standards (ITU-T G.711)

        self.duration = 1.0  # Signal duration in seconds
        # Rationale: 1 second is sufficient for isolated digit recognition
        # Longer duration would capture more temporal context but increase computational cost

        # ====================================================================
        # MFCC (Mel-Frequency Cepstral Coefficients) Parameters
        # Baseline feature extraction method
        # Reference: Davis & Mermelstein (1980)
        # ====================================================================

        self.n_mfcc = 13  # Number of MFCC coefficients
        # Rationale: 13 is standard in speech recognition
        # Captures essential spectral envelope information
        # Final feature dimension: 13 (static) + 13 (Δ) + 13 (ΔΔ) = 39

        self.n_fft = 512  # FFT window size (samples)
        # Rationale: Power of 2 for efficient FFT computation
        # At 16 kHz, this gives ~32 ms window (512/16000)
        # Balances time-frequency resolution

        self.hop_length = 256  # Frame shift (samples)
        # Rationale: 50% overlap (256/512) is typical
        # At 16 kHz, this gives ~16 ms frame shift
        # Provides good temporal resolution

        self.n_mels = 40  # Number of Mel filterbanks
        # Rationale: 40 Mel filters is standard practice
        # Provides sufficient frequency resolution for speech
        # Reference: HTK toolkit defaults

        # ====================================================================
        # ICA (Independent Component Analysis) Parameters
        # For blind source separation
        # Reference: Hyvärinen & Oja (2000) - FastICA algorithm
        # ====================================================================

        self.ica_max_iter = 1000  # Maximum iterations for ICA convergence
        # Rationale: Sufficient for most cases; FastICA typically converges < 100 iterations
        # Prevents infinite loops in edge cases

        self.ica_tol = 1e-4  # Convergence tolerance
        # Rationale: Balance between accuracy and speed
        # Smaller values (e.g., 1e-6) give more precise separation but take longer
        # 1e-4 is a good practical compromise

        self.ica_n_components = 2  # Number of independent components to extract
        # Rationale: For our application - separate speech from noise
        # In theory: n_components ≤ n_observations (microphones)

        self.ica_fun = 'cube'  # Nonlinearity function for ICA
        # Options: 'cube' or 'exp'
        # Rationale: 'cube' is robust and works well for speech
        # 'exp' can be faster but may be less stable

        # ====================================================================
        # Gammatone Filterbank Parameters
        # Models human auditory system frequency selectivity
        # Reference: Patterson et al. (1988), Glasberg & Moore (1990)
        # ====================================================================

        self.n_filters = 32  # Number of Gammatone filters
        # Rationale: 32 channels provide good frequency resolution
        # Compromise between detail (64 filters) and efficiency (16 filters)
        # Similar to cochlear implants (12-22 channels)

        self.freq_min = 50.0  # Minimum center frequency (Hz)
        # Rationale: Below 50 Hz is mostly noise, not speech
        # Human speech fundamental frequency typically 80-300 Hz

        self.freq_max = 7500.0  # Maximum center frequency (Hz)
        # Rationale: Must be < Nyquist frequency (8000 Hz for 16 kHz sampling)
        # Speech information mostly below 8 kHz
        # 7500 Hz provides safety margin

        self.gammatone_order = 4  # Filter order
        # Rationale: Order 4 is standard for Gammatone filters
        # Matches biological auditory filter characteristics
        # Reference: Patterson et al. (1988)

        # ====================================================================
        # GMM (Gaussian Mixture Model) Classifier Parameters
        # Traditional generative classifier for speech recognition
        # Reference: Reynolds & Rose (1995)
        # ====================================================================

        self.gmm_n_components = 8  # Number of Gaussian components per class
        # Rationale: 8 components can model complex feature distributions
        # Fewer (e.g., 4) may underfit; more (e.g., 16) may overfit

        self.gmm_covariance_type = 'diag'  # Covariance matrix type
        # Options: 'full', 'tied', 'diag', 'spherical'
        # Rationale: 'diag' assumes feature independence
        # Good balance between model complexity and robustness
        # Prevents overfitting with limited training data

        self.gmm_max_iter = 100  # Maximum EM iterations
        # Rationale: GMM typically converges in < 50 iterations
        # 100 provides safety margin

        self.gmm_tol = 1e-3  # EM convergence tolerance
        # Rationale: 1e-3 is sufficient for practical convergence

        # ====================================================================
        # Vocabulary and Label Mapping
        # ====================================================================

        self.vocabulary = [
            'zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine'
        ]
        # Rationale: Isolated digit recognition (0-9)
        # Standard small-vocabulary task for speech recognition research

        self.num_classes = len(self.vocabulary)  # Number of classes (10)

        # Label to index mapping
        self.label_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_label = {idx: word for idx, word in enumerate(self.vocabulary)}

        # ====================================================================
        # Frequency Mapping for Synthetic Speech Generation
        # ====================================================================

        self.word_to_freq = {
            'zero': 200,   # 200 Hz fundamental frequency
            'one': 300,    # Each digit has distinct F0
            'two': 400,    # Spacing varies for better discrimination
            'three': 500,  # Range: 200-1100 Hz
            'four': 600,
            'five': 700,
            'six': 800,
            'seven': 900,
            'eight': 1000,
            'nine': 1100
        }
        # Rationale: Distinct fundamental frequencies allow easy discrimination
        # Simulates natural variation in human speech
        # Note: In production, use real recordings instead of synthesis

        # ====================================================================
        # Experimental Design Parameters
        # Following Hirsch & Pearce (2000) evaluation framework
        # ====================================================================

        self.snr_levels = [-5, 0, 5, 10, 15, 20]  # Signal-to-Noise Ratios (dB)
        # Rationale: Covers range from very noisy (-5 dB) to clean (20 dB)
        # -5 dB: Very challenging, barely intelligible
        # 20 dB: Nearly clean, easy recognition

        self.noise_types = ['white', 'pink', 'babble', 'street']
        # Rationale: Tests robustness across different noise characteristics
        # - white: Uniform spectrum, theoretical baseline
        # - pink: 1/f spectrum, more natural
        # - babble: Multi-speaker interference, most challenging
        # - street: Traffic noise, low-frequency dominated

        # ====================================================================
        # Training and Evaluation Parameters
        # ====================================================================

        self.test_size = 0.3  # Fraction of data for testing
        # Rationale: 70/30 train-test split is standard
        # Balances training data size and reliable evaluation

        self.random_seed = 42  # Random seed for reproducibility
        # Rationale: Fixed seed ensures reproducible results
        # Important for scientific validity

        self.n_samples_per_class = 10  # Samples per digit class
        # Rationale: 10 samples × 10 classes = 100 total samples
        # Small dataset for quick experimentation
        # Production systems would use 100-1000 samples per class

    # ========================================================================
    # Helper Methods for Derived Parameters
    # ========================================================================

    def get_frame_duration_ms(self):
        """
        Calculate frame duration in milliseconds.

        Returns:
            float: Frame duration in ms
        """
        return (self.n_fft / self.sample_rate) * 1000

    def get_hop_duration_ms(self):
        """
        Calculate hop (frame shift) duration in milliseconds.

        Returns:
            float: Hop duration in ms
        """
        return (self.hop_length / self.sample_rate) * 1000

    def get_num_frames(self):
        """
        Calculate number of frames for given audio duration.

        Returns:
            int: Number of frames
        """
        n_samples = int(self.duration * self.sample_rate)
        n_frames = (n_samples - self.n_fft) // self.hop_length + 1
        return n_frames

    def get_nyquist_frequency(self):
        """
        Calculate Nyquist frequency.

        Returns:
            float: Nyquist frequency (Hz)
        """
        return self.sample_rate / 2.0

    def print_config(self):
        """
        Print all configuration parameters in a formatted way.

        Useful for:
        - Documentation
        - Debugging
        - Verifying parameter values
        """
        print("=" * 80)
        print(" ELEC5305 Research Project - Configuration")
        print("=" * 80)

        print("\n🎯 Research Question:")
        print("  Can ICA + Gammatone improve speech recognition in noisy environments?")

        print("\n📊 Audio Parameters:")
        print(f"  Sample Rate:       {self.sample_rate:,} Hz")
        print(f"  Signal Duration:   {self.duration} seconds")
        print(f"  Nyquist Frequency: {self.get_nyquist_frequency():,} Hz")

        print("\n🎵 MFCC Configuration (Baseline Method):")
        print(f"  Number of Coefficients: {self.n_mfcc}")
        print(f"  FFT Window Size:        {self.n_fft} samples ({self.get_frame_duration_ms():.1f} ms)")
        print(f"  Hop Length:             {self.hop_length} samples ({self.get_hop_duration_ms():.1f} ms)")
        print(f"  Mel Filterbanks:        {self.n_mels}")
        print(f"  Total Feature Dim:      39 (13 static + 13 Δ + 13 ΔΔ)")

        print("\n🔊 Gammatone Filterbank Configuration:")
        print(f"  Number of Filters:  {self.n_filters}")
        print(f"  Frequency Range:    {self.freq_min} - {self.freq_max} Hz")
        print(f"  Filter Order:       {self.gammatone_order}")

        print("\n🎭 ICA Configuration:")
        print(f"  Number of Components: {self.ica_n_components}")
        print(f"  Max Iterations:       {self.ica_max_iter}")
        print(f"  Tolerance:            {self.ica_tol}")
        print(f"  Nonlinearity:         {self.ica_fun}")

        print("\n🤖 GMM Classifier Configuration:")
        print(f"  Components per Class: {self.gmm_n_components}")
        print(f"  Covariance Type:      {self.gmm_covariance_type}")

        print("\n📚 Dataset Configuration:")
        print(f"  Vocabulary:        {', '.join(self.vocabulary)}")
        print(f"  Number of Classes: {self.num_classes}")
        print(f"  Samples per Class: {self.n_samples_per_class}")

        print("\n🔬 Experimental Design:")
        print(f"  SNR Levels:   {self.snr_levels} dB")
        print(f"  Noise Types:  {', '.join(self.noise_types)}")

        print("\n📖 Key References:")
        print("  - MFCC: Davis & Mermelstein (1980)")
        print("  - Gammatone: Patterson et al. (1988)")
        print("  - ICA: Hyvärinen & Oja (2000)")
        print("  - GMM: Reynolds & Rose (1995)")

        print("\n" + "=" * 80)


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate the configuration module.
    
    This section executes when the module is run directly (not imported).
    Useful for:
    - Verifying configuration values
    - Debugging parameter settings
    - Generating documentation
    """

    print("=" * 80)
    print(" Configuration Module Test")
    print("=" * 80)

    # Create configuration instance
    config = Config()

    # Display all parameters
    config.print_config()

    # Test derived parameter calculations
    print("\n🧮 Derived Parameters Test:")
    print(f"  Frame Duration:    {config.get_frame_duration_ms():.2f} ms")
    print(f"  Hop Duration:      {config.get_hop_duration_ms():.2f} ms")
    print(f"  Frames per Signal: {config.get_num_frames()}")
    print(f"  Nyquist Frequency: {config.get_nyquist_frequency()} Hz")

    # Test vocabulary mapping
    print("\n📝 Vocabulary Mapping Test:")
    for word in config.vocabulary[:3]:  # Show first 3 examples
        idx = config.label_to_idx[word]
        freq = config.word_to_freq[word]
        print(f"  '{word}' → index={idx}, F0={freq} Hz")
    print(f"  ... and {len(config.vocabulary) - 3} more")

    # Validate parameter consistency
    print("\n✅ Parameter Validation:")

    # Check Nyquist constraint
    if config.freq_max < config.get_nyquist_frequency():
        print(f"  ✓ Gammatone max freq ({config.freq_max} Hz) < Nyquist ({config.get_nyquist_frequency()} Hz)")
    else:
        print(f"  ✗ WARNING: Gammatone max freq exceeds Nyquist frequency!")

    # Check FFT size is power of 2
    import math
    if math.log2(config.n_fft).is_integer():
        print(f"  ✓ FFT size ({config.n_fft}) is power of 2")
    else:
        print(f"  ⚠ FFT size ({config.n_fft}) is not power of 2 (may be slower)")

    # Check hop length <= FFT size
    if config.hop_length <= config.n_fft:
        overlap_percent = (1 - config.hop_length / config.n_fft) * 100
        print(f"  ✓ Hop length valid, {overlap_percent:.0f}% overlap")
    else:
        print(f"  ✗ WARNING: Hop length > FFT size (frames won't overlap)")

    # Check vocabulary consistency
    if len(config.vocabulary) == config.num_classes:
        print(f"  ✓ Vocabulary size matches num_classes ({config.num_classes})")
    else:
        print(f"  ✗ ERROR: Vocabulary size mismatch!")

    print("\n" + "=" * 80)
    print(" ✅ Configuration Module Test Complete")
    print("=" * 80)
    print()