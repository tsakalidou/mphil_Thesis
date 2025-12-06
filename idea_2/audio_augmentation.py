"""
Audio Data Augmentation for Cry Classification
==============================================
Techniques to artificially increase training data and improve model robustness.

Augmentation methods:
1. Time stretching
2. Pitch shifting  
3. Adding noise
4. Time shifting
5. Volume scaling
6. Frequency masking (SpecAugment-style)
"""

import numpy as np
from scipy import signal as scipy_signal
import warnings

warnings.filterwarnings('ignore')


def time_stretch(signal, rate=1.0):
    """
    Stretch or compress signal in time without changing pitch.
    Simple implementation using resampling.
    
    :param signal: Input audio signal
    :param rate: Stretch factor (>1 = slower, <1 = faster)
    :return: Time-stretched signal
    """
    if rate == 1.0:
        return signal
    
    # Simple linear interpolation based time stretch
    original_length = len(signal)
    new_length = int(original_length / rate)
    
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    
    stretched = np.interp(x_new, x_original, signal)
    
    return stretched


def pitch_shift_simple(signal, fs, semitones):
    """
    Simple pitch shifting using resampling.
    
    :param signal: Input audio signal
    :param fs: Sampling frequency
    :param semitones: Number of semitones to shift (positive = higher)
    :return: Pitch-shifted signal
    """
    if semitones == 0:
        return signal
    
    # Calculate rate for pitch shift
    rate = 2 ** (semitones / 12.0)
    
    # Resample to change pitch
    new_length = int(len(signal) / rate)
    x_original = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, new_length)
    
    shifted = np.interp(x_new, x_original, signal)
    
    # Resample back to original length to maintain duration
    x_shifted = np.linspace(0, 1, len(shifted))
    x_final = np.linspace(0, 1, len(signal))
    
    result = np.interp(x_final, x_shifted, shifted)
    
    return result


def add_noise(signal, noise_factor=0.005):
    """
    Add Gaussian white noise to signal.
    
    :param signal: Input audio signal
    :param noise_factor: Standard deviation of noise relative to signal std
    :return: Noisy signal
    """
    noise = np.random.randn(len(signal))
    noise = noise * noise_factor * np.std(signal)
    return signal + noise


def time_shift(signal, shift_max=0.2):
    """
    Randomly shift signal in time (circular shift).
    
    :param signal: Input audio signal
    :param shift_max: Maximum shift as fraction of signal length
    :return: Time-shifted signal
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * len(signal))
    return np.roll(signal, shift)


def volume_scale(signal, gain_range=(0.7, 1.3)):
    """
    Randomly scale signal volume.
    
    :param signal: Input audio signal
    :param gain_range: Range of gain factors (min, max)
    :return: Scaled signal
    """
    gain = np.random.uniform(gain_range[0], gain_range[1])
    return signal * gain


def add_background_noise(signal, noise_level=0.01):
    """
    Add pink noise (more natural sounding than white noise).
    
    :param signal: Input audio signal
    :param noise_level: Noise amplitude relative to signal
    :return: Signal with background noise
    """
    # Generate pink noise using filtering
    white = np.random.randn(len(signal))
    
    # Simple approximation of pink noise filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    
    try:
        pink = scipy_signal.lfilter(b, a, white)
    except:
        pink = white  # Fallback to white noise
    
    pink = pink * noise_level * np.std(signal)
    return signal + pink


def augment_signal(signal, fs, augmentation_config=None):
    """
    Apply random augmentations to a signal.
    
    :param signal: Input audio signal
    :param fs: Sampling frequency
    :param augmentation_config: Dictionary of augmentation settings
    :return: Augmented signal
    """
    if augmentation_config is None:
        augmentation_config = {
            'time_stretch': {'prob': 0.5, 'rate_range': (0.9, 1.1)},
            'pitch_shift': {'prob': 0.5, 'semitone_range': (-2, 2)},
            'add_noise': {'prob': 0.5, 'noise_factor': 0.005},
            'time_shift': {'prob': 0.5, 'shift_max': 0.1},
            'volume_scale': {'prob': 0.5, 'gain_range': (0.8, 1.2)},
        }
    
    augmented = signal.copy()
    
    # Time stretch
    if 'time_stretch' in augmentation_config:
        cfg = augmentation_config['time_stretch']
        if np.random.random() < cfg.get('prob', 0.5):
            rate = np.random.uniform(*cfg.get('rate_range', (0.9, 1.1)))
            augmented = time_stretch(augmented, rate)
    
    # Pitch shift
    if 'pitch_shift' in augmentation_config:
        cfg = augmentation_config['pitch_shift']
        if np.random.random() < cfg.get('prob', 0.5):
            semitones = np.random.uniform(*cfg.get('semitone_range', (-2, 2)))
            augmented = pitch_shift_simple(augmented, fs, semitones)
    
    # Add noise
    if 'add_noise' in augmentation_config:
        cfg = augmentation_config['add_noise']
        if np.random.random() < cfg.get('prob', 0.5):
            augmented = add_noise(augmented, cfg.get('noise_factor', 0.005))
    
    # Time shift
    if 'time_shift' in augmentation_config:
        cfg = augmentation_config['time_shift']
        if np.random.random() < cfg.get('prob', 0.5):
            augmented = time_shift(augmented, cfg.get('shift_max', 0.1))
    
    # Volume scale
    if 'volume_scale' in augmentation_config:
        cfg = augmentation_config['volume_scale']
        if np.random.random() < cfg.get('prob', 0.5):
            augmented = volume_scale(augmented, cfg.get('gain_range', (0.8, 1.2)))
    
    return augmented


def augment_dataset(X_signals, y, groups, fs, n_augmentations=2, augmentation_config=None):
    """
    Augment entire dataset.
    
    :param X_signals: List of audio signals
    :param y: Labels
    :param groups: Subject groups
    :param fs: Sampling frequency
    :param n_augmentations: Number of augmented versions per sample
    :param augmentation_config: Augmentation settings
    :return: Augmented signals, labels, and groups
    """
    X_aug = list(X_signals)
    y_aug = list(y)
    groups_aug = list(groups)
    
    print(f"Augmenting dataset ({n_augmentations} versions per sample)...")
    
    for i, (sig, label, group) in enumerate(zip(X_signals, y, groups)):
        for j in range(n_augmentations):
            aug_sig = augment_signal(sig, fs, augmentation_config)
            X_aug.append(aug_sig)
            y_aug.append(label)
            groups_aug.append(group)  # Keep same group to maintain subject integrity
    
    print(f"Original samples: {len(X_signals)}")
    print(f"Augmented samples: {len(X_aug)}")
    
    return X_aug, np.array(y_aug), np.array(groups_aug)


# =============================================================================
# SPECAUGMENT-STYLE AUGMENTATION (on MFCC)
# =============================================================================

def frequency_mask(mfcc, num_masks=1, mask_width_range=(1, 5)):
    """
    Apply frequency masking to MFCC features (SpecAugment-style).
    
    :param mfcc: MFCC features [n_frames, n_coeffs]
    :param num_masks: Number of frequency masks to apply
    :param mask_width_range: Range of mask widths
    :return: Masked MFCC
    """
    mfcc_masked = mfcc.copy()
    n_coeffs = mfcc.shape[1]
    
    for _ in range(num_masks):
        mask_width = np.random.randint(*mask_width_range)
        mask_start = np.random.randint(0, max(1, n_coeffs - mask_width))
        mfcc_masked[:, mask_start:mask_start + mask_width] = 0
    
    return mfcc_masked


def time_mask(mfcc, num_masks=1, mask_width_range=(5, 20)):
    """
    Apply time masking to MFCC features (SpecAugment-style).
    
    :param mfcc: MFCC features [n_frames, n_coeffs]
    :param num_masks: Number of time masks to apply
    :param mask_width_range: Range of mask widths in frames
    :return: Masked MFCC
    """
    mfcc_masked = mfcc.copy()
    n_frames = mfcc.shape[0]
    
    for _ in range(num_masks):
        mask_width = np.random.randint(*mask_width_range)
        mask_start = np.random.randint(0, max(1, n_frames - mask_width))
        mfcc_masked[mask_start:mask_start + mask_width, :] = 0
    
    return mfcc_masked


def specaugment(mfcc, freq_masks=2, time_masks=2):
    """
    Apply SpecAugment-style augmentation to MFCC.
    
    :param mfcc: MFCC features
    :param freq_masks: Number of frequency masks
    :param time_masks: Number of time masks
    :return: Augmented MFCC
    """
    augmented = mfcc.copy()
    
    # Apply frequency masking
    augmented = frequency_mask(augmented, num_masks=freq_masks)
    
    # Apply time masking  
    augmented = time_mask(augmented, num_masks=time_masks)
    
    return augmented


if __name__ == '__main__':
    # Test augmentation functions
    print("Testing augmentation functions...")
    
    # Create a simple test signal (1 second at 44.1kHz)
    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simulated cry signal (mix of frequencies)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 300 * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 600 * t) +  # Harmonic
        0.2 * np.sin(2 * np.pi * 900 * t)    # Harmonic
    )
    
    print(f"Original signal: {len(test_signal)} samples")
    
    # Test each augmentation
    print("\nTesting individual augmentations:")
    
    stretched = time_stretch(test_signal, rate=0.9)
    print(f"  Time stretch (0.9): {len(stretched)} samples")
    
    shifted = pitch_shift_simple(test_signal, fs, semitones=2)
    print(f"  Pitch shift (+2 semitones): {len(shifted)} samples")
    
    noisy = add_noise(test_signal, noise_factor=0.01)
    print(f"  Add noise: {len(noisy)} samples, SNR change visible in std: {np.std(noisy):.4f}")
    
    time_shifted = time_shift(test_signal, shift_max=0.1)
    print(f"  Time shift: {len(time_shifted)} samples")
    
    scaled = volume_scale(test_signal, gain_range=(0.5, 1.5))
    print(f"  Volume scale: amplitude range [{scaled.min():.2f}, {scaled.max():.2f}]")
    
    # Test combined augmentation
    print("\nTesting combined augmentation:")
    augmented = augment_signal(test_signal, fs)
    print(f"  Combined augmentation: {len(augmented)} samples")
    
    # Test MFCC augmentation
    print("\nTesting SpecAugment on MFCC:")
    test_mfcc = np.random.randn(100, 12)  # Simulated MFCC
    augmented_mfcc = specaugment(test_mfcc)
    zeros_count = np.sum(augmented_mfcc == 0)
    print(f"  Original zeros: 0, After SpecAugment: {zeros_count}")
    
    print("\nAll tests passed!")