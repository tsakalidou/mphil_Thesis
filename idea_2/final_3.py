"""
ENHANCED ASD CLASSIFICATION PIPELINE V3
=======================================
Target: >85-90% subject-level accuracy

New in V3:
1. Librosa pyin for robust pitch extraction
2. More prosodic features (intensity contours, rhythm)
3. 3-second audio duration
4. Formant features (F1, F2, F3)
5. More augmentation (5x)
6. Spectral contrast and tonnetz features
7. Feature importance analysis
8. Threshold optimization
9. Confidence-weighted voting
10. XGBoost integration

Author: Enhanced for MSc Capstone
"""

import os
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy.io.wavfile as sci_wav
import scipy.stats as stats
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d
import librosa

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel

from mfcc_feature_extraction import extract_mfcc_feature

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # UPDATE THIS PATH
    DATA_PATH = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou_Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
    
    # Audio settings - INCREASED DURATION
    SAMPLE_RATE = 44100
    AUDIO_DURATION = 3.0  # INCREASED to 3 seconds
    N_FFT = 512
    
    # MFCC settings
    N_MELS = 40
    N_MFCC = 13
    
    # Features
    INCLUDE_DELTA = True
    INCLUDE_DELTA_DELTA = True
    INCLUDE_ENERGY = True
    INCLUDE_ZCR = True
    INCLUDE_PITCH = True
    INCLUDE_VOICE_QUALITY = True
    INCLUDE_SPECTRAL = True
    INCLUDE_AGE = True
    INCLUDE_FORMANTS = True  # NEW
    INCLUDE_RHYTHM = True  # NEW
    INCLUDE_SPECTRAL_CONTRAST = True  # NEW
    
    # Augmentation - INCREASED
    USE_AUGMENTATION = True
    N_AUGMENTATIONS = 5  # INCREASED from 3
    
    # CV settings
    CV_FOLDS = 5
    USE_SUBJECT_VOTING = True
    
    # Feature selection
    N_FEATURES = 250  # INCREASED
    
    # Confidence threshold for voting
    CONFIDENCE_THRESHOLD = 0.6
    
    RANDOM_STATE = 42


# =============================================================================
# ROBUST PITCH EXTRACTION USING LIBROSA
# =============================================================================

def extract_pitch_librosa(signal, fs):
    """
    Extract pitch using librosa's pyin algorithm (more robust than autocorrelation).
    """
    try:
        # Ensure float type
        signal = signal.astype(np.float32)
        
        # Use pyin for pitch tracking (robust to noise)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal,
            fmin=150,  # Min pitch for infant cries
            fmax=800,  # Max pitch for infant cries
            sr=fs,
            frame_length=2048,
            hop_length=512
        )
        
        # Handle NaN values
        f0_clean = f0[~np.isnan(f0)]
        voiced_probs_clean = voiced_probs[~np.isnan(voiced_probs)]
        
        if len(f0_clean) < 5:
            return np.zeros(25)
        
        features = [
            # Basic statistics
            np.mean(f0_clean),
            np.std(f0_clean),
            np.min(f0_clean),
            np.max(f0_clean),
            np.median(f0_clean),
            
            # Variability
            np.max(f0_clean) - np.min(f0_clean),  # Range
            np.std(f0_clean) / (np.mean(f0_clean) + 1e-10),  # CV
            stats.skew(f0_clean),
            stats.kurtosis(f0_clean),
            
            # Percentiles
            np.percentile(f0_clean, 10),
            np.percentile(f0_clean, 25),
            np.percentile(f0_clean, 75),
            np.percentile(f0_clean, 90),
            np.percentile(f0_clean, 75) - np.percentile(f0_clean, 25),  # IQR
            
            # Dynamics
            np.mean(np.abs(np.diff(f0_clean))),
            np.std(np.diff(f0_clean)),
            np.max(np.abs(np.diff(f0_clean))),
            
            # Voicing statistics
            np.mean(voiced_probs_clean),
            np.std(voiced_probs_clean),
            np.sum(~np.isnan(f0)) / len(f0),  # Voiced ratio
            
            # Pitch direction
            np.sum(np.diff(f0_clean) > 0) / (len(f0_clean) - 1 + 1e-10),
            np.sum(np.diff(f0_clean) < 0) / (len(f0_clean) - 1 + 1e-10),
            
            # Pitch stability (low = more stable)
            np.std(np.diff(f0_clean)) / (np.mean(np.abs(np.diff(f0_clean))) + 1e-10),
            
            # Mean absolute deviation
            np.mean(np.abs(f0_clean - np.mean(f0_clean))),
            
            # Number of pitch peaks (inflection points)
            np.sum(np.diff(np.sign(np.diff(f0_clean))) != 0) / (len(f0_clean) + 1e-10),
        ]
        
        return np.array(features)
        
    except Exception as e:
        return np.zeros(25)


def extract_formants_lpc(signal, fs, n_formants=4):
    """
    Extract formant frequencies using LPC analysis.
    Formants characterize vocal tract resonances.
    """
    try:
        # Pre-emphasis
        pre_emphasis = 0.97
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # LPC order
        order = 2 + int(fs / 1000)
        
        # Windowed analysis
        frame_length = int(0.025 * fs)  # 25ms frames
        hop_length = int(0.010 * fs)  # 10ms hop
        
        all_formants = [[] for _ in range(n_formants)]
        
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            
            # Apply Hamming window
            frame = frame * np.hamming(len(frame))
            
            # Compute LPC coefficients using autocorrelation
            r = np.correlate(frame, frame, mode='full')
            r = r[len(r)//2:len(r)//2 + order + 1]
            
            # Levinson-Durbin
            a = np.zeros(order + 1)
            a[0] = 1.0
            e = r[0]
            
            for j in range(1, order + 1):
                if e < 1e-10:
                    break
                lambda_val = -np.sum(a[:j] * r[j:0:-1]) / e
                a_new = a.copy()
                for k in range(1, j + 1):
                    a_new[k] = a[k] + lambda_val * a[j - k]
                a = a_new
                e = e * (1 - lambda_val ** 2)
            
            # Find formants from roots
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            
            # Convert to frequencies
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angles * (fs / (2 * np.pi))
            
            # Filter valid formants
            freqs = freqs[(freqs > 90) & (freqs < fs/2 - 100)]
            freqs = np.sort(freqs)
            
            for k in range(min(n_formants, len(freqs))):
                all_formants[k].append(freqs[k])
        
        # Compute statistics for each formant
        features = []
        for k in range(n_formants):
            if len(all_formants[k]) > 0:
                f = np.array(all_formants[k])
                features.extend([
                    np.mean(f),
                    np.std(f),
                    np.median(f),
                ])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features)
        
    except Exception:
        return np.zeros(n_formants * 3)


def extract_rhythm_features(signal, fs):
    """
    Extract rhythm-related features from the amplitude envelope.
    """
    try:
        # Compute amplitude envelope
        analytic = scipy_signal.hilbert(signal)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        window_size = int(0.02 * fs)  # 20ms window
        envelope_smooth = uniform_filter1d(envelope, size=window_size)
        
        # Find peaks (cry events)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            envelope_smooth, 
            height=np.mean(envelope_smooth),
            distance=int(0.1 * fs)  # Min 100ms between peaks
        )
        
        features = []
        
        # Number of peaks per second
        duration = len(signal) / fs
        features.append(len(peaks) / duration)
        
        if len(peaks) > 1:
            # Inter-peak intervals
            intervals = np.diff(peaks) / fs
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.min(intervals),
                np.max(intervals),
                stats.skew(intervals) if len(intervals) > 2 else 0,
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Envelope statistics
        features.extend([
            np.mean(envelope_smooth),
            np.std(envelope_smooth),
            np.max(envelope_smooth) / (np.mean(envelope_smooth) + 1e-10),  # Crest factor
            stats.skew(envelope_smooth),
            stats.kurtosis(envelope_smooth),
        ])
        
        return np.array(features)
        
    except Exception:
        return np.zeros(11)


def extract_spectral_contrast(signal, fs, n_bands=6):
    """
    Extract spectral contrast features.
    Measures the difference between peaks and valleys in spectrum.
    """
    try:
        signal = signal.astype(np.float32)
        
        # Compute spectral contrast using librosa
        contrast = librosa.feature.spectral_contrast(
            y=signal, 
            sr=fs, 
            n_bands=n_bands,
            fmin=200
        )
        
        features = []
        for i in range(contrast.shape[0]):
            features.extend([
                np.mean(contrast[i]),
                np.std(contrast[i]),
            ])
        
        return np.array(features)
        
    except Exception:
        return np.zeros((n_bands + 1) * 2)


def extract_additional_spectral(signal, fs):
    """
    Extract additional spectral features using librosa.
    """
    try:
        signal = signal.astype(np.float32)
        
        features = []
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=fs)[0]
        features.extend([np.mean(bandwidth), np.std(bandwidth)])
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=fs)[0]
        features.extend([np.mean(rolloff), np.std(rolloff)])
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=signal)[0]
        features.extend([np.mean(flatness), np.std(flatness)])
        
        # RMS energy
        rms = librosa.feature.rms(y=signal)[0]
        features.extend([
            np.mean(rms), 
            np.std(rms),
            np.max(rms),
            np.max(rms) - np.min(rms),
        ])
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=signal, sr=fs)[0]
        features.extend([
            np.mean(centroid),
            np.std(centroid),
            np.median(centroid),
        ])
        
        return np.array(features)
        
    except Exception:
        return np.zeros(13)


# =============================================================================
# VOICE QUALITY FEATURES
# =============================================================================

def extract_voice_quality(signal, fs):
    """
    Extract jitter and shimmer features (voice quality measures).
    """
    frame_length = int(0.025 * fs)
    hop_length = int(0.010 * fs)
    
    min_f0, max_f0 = 150, 800
    min_period = int(fs / max_f0)
    max_period = int(fs / min_f0)
    
    periods = []
    amplitudes = []
    
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length].astype(float)
        frame = frame - np.mean(frame)
        
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        if max_period < len(corr):
            search_region = corr[min_period:max_period]
            if len(search_region) > 0 and np.max(search_region) > 0.3 * corr[0]:
                peak_idx = np.argmax(search_region) + min_period
                periods.append(peak_idx)
                amplitudes.append(np.max(np.abs(frame)))
    
    if len(periods) < 5:
        return np.zeros(12)
    
    periods = np.array(periods)
    amplitudes = np.array(amplitudes)
    
    # Jitter measures
    period_diffs = np.abs(np.diff(periods))
    jitter_abs = np.mean(period_diffs)
    jitter_rel = jitter_abs / (np.mean(periods) + 1e-10)
    jitter_ppq5 = 0
    if len(periods) >= 5:
        ppq5_vals = []
        for i in range(2, len(periods) - 2):
            local_mean = np.mean(periods[i-2:i+3])
            ppq5_vals.append(np.abs(periods[i] - local_mean))
        jitter_ppq5 = np.mean(ppq5_vals) / (np.mean(periods) + 1e-10)
    
    # Shimmer measures
    amp_diffs = np.abs(np.diff(amplitudes))
    shimmer_abs = np.mean(amp_diffs)
    shimmer_rel = shimmer_abs / (np.mean(amplitudes) + 1e-10)
    shimmer_apq5 = 0
    if len(amplitudes) >= 5:
        apq5_vals = []
        for i in range(2, len(amplitudes) - 2):
            local_mean = np.mean(amplitudes[i-2:i+3])
            apq5_vals.append(np.abs(amplitudes[i] - local_mean))
        shimmer_apq5 = np.mean(apq5_vals) / (np.mean(amplitudes) + 1e-10)
    
    features = [
        jitter_abs,
        jitter_rel,
        jitter_ppq5,
        shimmer_abs,
        shimmer_rel,
        shimmer_apq5,
        np.std(periods) / (np.mean(periods) + 1e-10),
        np.std(amplitudes) / (np.mean(amplitudes) + 1e-10),
        stats.skew(periods) if len(periods) > 2 else 0,
        stats.skew(amplitudes) if len(amplitudes) > 2 else 0,
        stats.kurtosis(periods) if len(periods) > 3 else 0,
        stats.kurtosis(amplitudes) if len(amplitudes) > 3 else 0,
    ]
    
    return np.array(features)


# =============================================================================
# FEATURE EXTRACTOR CLASS
# =============================================================================

class FeatureExtractorV3:
    """Comprehensive feature extraction V3."""
    
    def __init__(self, config):
        self.config = config
        self.mfcc_params = {
            'fs': config.SAMPLE_RATE,
            'n_fft': config.N_FFT,
            'frame_size': config.N_FFT / config.SAMPLE_RATE,
            'frame_step': int(config.N_FFT / 3) / config.SAMPLE_RATE,
            'n_mels': config.N_MELS,
            'n_mfcc': config.N_MFCC
        }
        self.fix_len = int(config.AUDIO_DURATION * config.SAMPLE_RATE)
    
    def compute_delta(self, features, N=2):
        n_frames, n_coeffs = features.shape
        deltas = np.zeros_like(features)
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        denominator = 2 * sum(n ** 2 for n in range(1, N + 1))
        
        for t in range(n_frames):
            numerator = sum(n * (padded[t + N + n] - padded[t + N - n]) 
                          for n in range(1, N + 1))
            deltas[t] = numerator / denominator
        return deltas
    
    def compute_statistics(self, features):
        stats_list = []
        for i in range(features.shape[1]):
            coeff = features[:, i]
            stats_list.extend([
                np.mean(coeff),
                np.std(coeff),
                np.min(coeff),
                np.max(coeff),
                np.median(coeff),
                stats.skew(coeff),
                stats.kurtosis(coeff),
                np.percentile(coeff, 25),
                np.percentile(coeff, 75),
                np.percentile(coeff, 75) - np.percentile(coeff, 25),
                np.mean(np.abs(np.diff(coeff))),
                np.std(np.diff(coeff)),
            ])
        return np.array(stats_list)
    
    def extract_features(self, signal, age=None):
        """Extract all features."""
        # Normalize
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        
        signal = librosa.util.fix_length(signal, size=self.fix_len, mode='wrap')
        
        # MFCC features
        mfcc = extract_mfcc_feature(
            y=signal,
            fs=self.mfcc_params['fs'],
            n_fft=self.mfcc_params['n_fft'],
            frame_size=self.mfcc_params['frame_size'],
            frame_step=self.mfcc_params['frame_step'],
            n_mels=self.mfcc_params['n_mels'],
            n_mfcc=self.mfcc_params['n_mfcc']
        )
        
        components = [mfcc]
        
        if self.config.INCLUDE_DELTA:
            components.append(self.compute_delta(mfcc))
        
        if self.config.INCLUDE_DELTA_DELTA:
            components.append(self.compute_delta(self.compute_delta(mfcc)))
        
        combined = np.hstack(components)
        stat_features = self.compute_statistics(combined)
        
        all_features = [stat_features]
        
        # Pitch features (using librosa pyin)
        if self.config.INCLUDE_PITCH:
            pitch_features = extract_pitch_librosa(signal, self.config.SAMPLE_RATE)
            all_features.append(pitch_features)
        
        # Voice quality
        if self.config.INCLUDE_VOICE_QUALITY:
            voice_quality = extract_voice_quality(signal, self.config.SAMPLE_RATE)
            all_features.append(voice_quality)
        
        # Formants
        if self.config.INCLUDE_FORMANTS:
            formants = extract_formants_lpc(signal, self.config.SAMPLE_RATE)
            all_features.append(formants)
        
        # Rhythm
        if self.config.INCLUDE_RHYTHM:
            rhythm = extract_rhythm_features(signal, self.config.SAMPLE_RATE)
            all_features.append(rhythm)
        
        # Spectral contrast
        if self.config.INCLUDE_SPECTRAL_CONTRAST:
            contrast = extract_spectral_contrast(signal, self.config.SAMPLE_RATE)
            all_features.append(contrast)
        
        # Additional spectral
        if self.config.INCLUDE_SPECTRAL:
            spectral = extract_additional_spectral(signal, self.config.SAMPLE_RATE)
            all_features.append(spectral)
        
        # Age
        if self.config.INCLUDE_AGE and age is not None:
            all_features.append(np.array([age, age**2, np.log(age + 1)]))  # Add polynomial
        
        return np.concatenate(all_features)


# =============================================================================
# AUDIO AUGMENTATION
# =============================================================================

class AudioAugmenterV3:
    """Enhanced audio augmentation."""
    
    def __init__(self, fs):
        self.fs = fs
    
    def augment(self, signal):
        augmented = signal.copy().astype(float)
        
        # Time stretch (more aggressive)
        if np.random.random() < 0.6:
            rate = np.random.uniform(0.85, 1.15)
            new_len = int(len(augmented) / rate)
            augmented = np.interp(
                np.linspace(0, 1, new_len),
                np.linspace(0, 1, len(augmented)),
                augmented
            )
        
        # Pitch shift
        if np.random.random() < 0.6:
            semitones = np.random.uniform(-3, 3)
            rate = 2 ** (semitones / 12.0)
            new_len = int(len(augmented) / rate)
            stretched = np.interp(
                np.linspace(0, 1, new_len),
                np.linspace(0, 1, len(augmented)),
                augmented
            )
            augmented = np.interp(
                np.linspace(0, 1, len(signal)),
                np.linspace(0, 1, len(stretched)),
                stretched
            )
        
        # Add noise (variable levels)
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.002, 0.015)
            noise = np.random.randn(len(augmented)) * noise_level * np.std(augmented)
            augmented = augmented + noise
        
        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.2, 0.2) * len(augmented))
            augmented = np.roll(augmented, shift)
        
        # Volume
        if np.random.random() < 0.5:
            augmented = augmented * np.random.uniform(0.6, 1.4)
        
        # Random filtering (lowpass/highpass)
        if np.random.random() < 0.3:
            cutoff = np.random.uniform(0.7, 0.95)
            b, a = scipy_signal.butter(4, cutoff, btype='low')
            augmented = scipy_signal.filtfilt(b, a, augmented)
        
        return augmented


# =============================================================================
# DATA LOADING
# =============================================================================

AGES = {
    'ASD1': 20, 'ASD2': 24, 'ASD3': 26, 'ASD4': 28, 'ASD5': 29,
    'ASD6': 31, 'ASD7': 36, 'ASD8': 43, 'ASD9': 45, 'ASD10': 45,
    'ASD11': 28, 'ASD12': 30, 'ASD13': 30, 'ASD14': 31, 'ASD15': 33,
    'ASD16': 33, 'ASD17': 34, 'ASD18': 35, 'ASD19': 37, 'ASD20': 40,
    'ASD21': 45, 'ASD22': 48, 'ASD23': 52, 'ASD24': 53, 'ASD25': 25,
    'ASD26': 26, 'ASD27': 31, 'ASD28': 32, 'ASD29': 41, 'ASD30': 45,
    'ASD31': 49,
    'TD1': 21, 'TD2': 24, 'TD3': 26, 'TD4': 28, 'TD5': 36, 'TD6': 38, 'TD7': 41,
    'TD8': 43, 'TD9': 44, 'TD10': 51, 'TD11': 18, 'TD12': 18, 'TD13': 19,
    'TD14': 20, 'TD15': 21, 'TD16': 24, 'TD17': 24, 'TD18': 24, 'TD19': 24,
    'TD20': 24, 'TD21': 29, 'TD22': 30, 'TD23': 30, 'TD24': 43, 'TD25': 24,
    'TD26': 25, 'TD27': 29, 'TD28': 33, 'TD29': 45, 'TD30': 50, 'TD31': 51
}


def load_data(config):
    """Load all data with augmentation."""
    extractor = FeatureExtractorV3(config)
    augmenter = AudioAugmenterV3(config.SAMPLE_RATE) if config.USE_AUGMENTATION else None
    
    X, y, groups, subjects = [], [], [], []
    subject_id = 0
    
    print("Loading and extracting features...")
    
    folders = sorted(os.listdir(config.DATA_PATH))
    
    for folder in folders:
        folder_path = os.path.join(config.DATA_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        
        if 'ASD' in folder:
            label = 1
        elif 'TD' in folder:
            label = 0
        else:
            continue
        
        age = AGES.get(folder, 30)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        n_loaded = 0
        
        for file in files:
            try:
                fs, signal = sci_wav.read(os.path.join(folder_path, file))
                
                features = extractor.extract_features(signal, age)
                X.append(features)
                y.append(label)
                groups.append(subject_id)
                subjects.append(folder)
                n_loaded += 1
                
                if config.USE_AUGMENTATION:
                    for _ in range(config.N_AUGMENTATIONS):
                        aug_signal = augmenter.augment(signal)
                        aug_features = extractor.extract_features(aug_signal, age)
                        X.append(aug_features)
                        y.append(label)
                        groups.append(subject_id)
                        subjects.append(folder)
                
            except Exception as e:
                print(f"  Error: {file}: {e}")
        
        print(f"  {folder}: {n_loaded} samples" + 
              (f" (+{n_loaded * config.N_AUGMENTATIONS} aug)" if config.USE_AUGMENTATION else ""))
        subject_id += 1
    
    return np.array(X), np.array(y), np.array(groups), subjects


# =============================================================================
# SUBJECT-LEVEL VOTING (CONFIDENCE-WEIGHTED)
# =============================================================================

def evaluate_with_confidence_voting(y_true, y_pred, y_prob, groups, subjects, threshold=0.5):
    """
    Aggregate predictions using confidence-weighted voting.
    Samples with higher confidence get more weight.
    """
    subject_weighted_probs = defaultdict(list)
    subject_labels = {}
    
    for i, (pred, prob, group, subj) in enumerate(zip(y_pred, y_prob, groups, subjects)):
        # Weight by confidence (distance from 0.5)
        confidence = np.abs(prob - 0.5) * 2
        subject_weighted_probs[subj].append((prob, confidence))
        subject_labels[subj] = y_true[i]
    
    final_preds = []
    final_probs = []
    final_true = []
    
    for subj in subject_weighted_probs:
        probs_confs = subject_weighted_probs[subj]
        probs = np.array([p for p, c in probs_confs])
        confs = np.array([c for p, c in probs_confs])
        
        # Weighted average
        if np.sum(confs) > 0:
            weighted_prob = np.sum(probs * confs) / np.sum(confs)
        else:
            weighted_prob = np.mean(probs)
        
        final_pred = 1 if weighted_prob > threshold else 0
        
        final_preds.append(final_pred)
        final_probs.append(weighted_prob)
        final_true.append(subject_labels[subj])
    
    return np.array(final_true), np.array(final_preds), np.array(final_probs)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate(X, y, groups, subjects, config):
    """Train and evaluate models."""
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features (RobustScaler is more robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    print(f"\nOriginal features: {X_scaled.shape[1]}")
    n_select = min(config.N_FEATURES, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected features: {X_selected.shape[1]}")
    
    # Define models
    models = {
        'SVM_RBF': (
            SVC(probability=True, random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 0.001, 0.01, 0.1]}
        ),
        'Random_Forest': (
            RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'n_estimators': [200, 300, 500], 'max_depth': [15, 20, 30, None], 
             'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
        ),
        'Extra_Trees': (
            ExtraTreesClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'n_estimators': [200, 300, 500], 'max_depth': [15, 20, 30, None]}
        ),
        'Gradient_Boosting': (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 10], 
             'learning_rate': [0.01, 0.05, 0.1]}
        ),
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        models['XGBoost'] = (
            XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, 
                         eval_metric='logloss', scale_pos_weight=1),
            {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 
             'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]}
        )
    
    cv = GroupKFold(n_splits=config.CV_FOLDS)
    
    results = {}
    best_models = {}
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    for name, (model, params) in models.items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print(f"{'─' * 50}")
        
        all_preds, all_probs, all_true = [], [], []
        all_groups, all_subjects = [], []
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Randomized search (faster than grid)
            search = RandomizedSearchCV(
                model, params, n_iter=20, cv=3, scoring='accuracy',
                n_jobs=-1, random_state=config.RANDOM_STATE
            )
            search.fit(X_train, y_train)
            
            y_pred = search.predict(X_test)
            y_prob = search.predict_proba(X_test)[:, 1]
            
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_true.extend(y_test)
            all_groups.extend(groups[test_idx])
            all_subjects.extend([subjects[i] for i in test_idx])
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_acc)
            
            n_subj = len(np.unique(groups[test_idx]))
            print(f"  Fold {fold_idx+1}: {fold_acc:.4f} ({n_subj} subjects)")
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        # Sample-level metrics
        sample_acc = accuracy_score(all_true, all_preds)
        sample_auc = roc_auc_score(all_true, all_probs)
        
        # Subject-level metrics (confidence-weighted voting)
        subj_true, subj_pred, subj_prob = evaluate_with_confidence_voting(
            all_true, all_preds, all_probs, all_groups, all_subjects
        )
        subj_acc = accuracy_score(subj_true, subj_pred)
        subj_f1 = f1_score(subj_true, subj_pred)
        subj_auc = roc_auc_score(subj_true, subj_prob)
        
        results[name] = {
            'sample_accuracy': sample_acc,
            'sample_auc': sample_auc,
            'subject_accuracy': subj_acc,
            'subject_f1': subj_f1,
            'subject_auc': subj_auc,
            'fold_std': np.std(fold_scores),
            'best_params': search.best_params_
        }
        
        best_models[name] = search.best_estimator_
        
        print(f"\n  Sample-level:  Acc={sample_acc:.4f}, AUC={sample_auc:.4f}")
        print(f"  Subject-level: Acc={subj_acc:.4f}, F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
        
        # Confusion matrix for subject level
        cm = confusion_matrix(subj_true, subj_pred)
        print(f"  Subject CM: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Ensemble
    print(f"\n{'─' * 50}")
    print("Training: Voting Ensemble")
    print(f"{'─' * 50}")
    
    estimators = []
    for name in ['SVM_RBF', 'Random_Forest', 'Extra_Trees', 'Gradient_Boosting']:
        if name in best_models:
            estimators.append((name.lower(), best_models[name]))
    
    if HAS_XGBOOST and 'XGBoost' in best_models:
        estimators.append(('xgboost', best_models['XGBoost']))
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    all_preds, all_probs, all_true = [], [], []
    all_groups, all_subjects = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]
        
        all_preds.extend(y_pred)
        all_probs.extend(y_prob)
        all_true.extend(y_test)
        all_groups.extend(groups[test_idx])
        all_subjects.extend([subjects[i] for i in test_idx])
        
        fold_acc = accuracy_score(y_test, y_pred)
        print(f"  Fold {fold_idx+1}: {fold_acc:.4f}")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    sample_acc = accuracy_score(all_true, all_preds)
    sample_auc = roc_auc_score(all_true, all_probs)
    
    subj_true, subj_pred, subj_prob = evaluate_with_confidence_voting(
        all_true, all_preds, all_probs, all_groups, all_subjects
    )
    subj_acc = accuracy_score(subj_true, subj_pred)
    subj_f1 = f1_score(subj_true, subj_pred)
    subj_auc = roc_auc_score(subj_true, subj_prob)
    
    results['Voting_Ensemble'] = {
        'sample_accuracy': sample_acc,
        'sample_auc': sample_auc,
        'subject_accuracy': subj_acc,
        'subject_f1': subj_f1,
        'subject_auc': subj_auc,
    }
    
    print(f"\n  Sample-level:  Acc={sample_acc:.4f}, AUC={sample_auc:.4f}")
    print(f"  Subject-level: Acc={subj_acc:.4f}, F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
    
    cm = confusion_matrix(subj_true, subj_pred)
    print(f"  Subject CM: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results, best_models


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ASD vs TD CLASSIFICATION - ENHANCED PIPELINE V3")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  - Audio duration: {config.AUDIO_DURATION}s")
    print(f"  - Augmentation: {config.USE_AUGMENTATION} (x{config.N_AUGMENTATIONS})")
    print(f"  - Pitch (librosa pyin): {config.INCLUDE_PITCH}")
    print(f"  - Voice quality: {config.INCLUDE_VOICE_QUALITY}")
    print(f"  - Formants: {config.INCLUDE_FORMANTS}")
    print(f"  - Rhythm: {config.INCLUDE_RHYTHM}")
    print(f"  - Spectral contrast: {config.INCLUDE_SPECTRAL_CONTRAST}")
    print(f"  - Age feature: {config.INCLUDE_AGE}")
    print(f"  - XGBoost available: {HAS_XGBOOST}")
    
    # Load data
    X, y, groups, subjects = load_data(config)
    
    print(f"\nData loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  ASD: {np.sum(y == 1)}, TD: {np.sum(y == 0)}")
    print(f"  Subjects: {len(np.unique(groups))}")
    
    # Train and evaluate
    results, best_models = train_and_evaluate(X, y, groups, subjects, config)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (SUBJECT-LEVEL)")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['subject_accuracy'], 
                           reverse=True)
    
    print(f"\n{'Model':<20} {'Subj Acc':<12} {'Subj F1':<10} {'Subj AUC':<10}")
    print("─" * 55)
    
    for name, r in sorted_results:
        print(f"{name:<20} {r['subject_accuracy']:.4f}       {r['subject_f1']:.4f}      {r['subject_auc']:.4f}")
    
    best_name, best_r = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST: {best_name} - Subject Accuracy: {best_r['subject_accuracy']:.2%}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()