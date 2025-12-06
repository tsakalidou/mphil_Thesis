"""
ENHANCED ASD CLASSIFICATION PIPELINE V2
=======================================
Improvements over V1:
1. Pitch/F0 features (critical for cry analysis)
2. Voice quality features (jitter, shimmer approximations)
3. Age as a feature
4. Leave-One-Subject-Out CV (more training data)
5. Subject-level voting (combine all samples per subject)
6. Data augmentation enabled
7. Longer audio segments option

Target: >90% accuracy
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
    AdaBoostClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from mfcc_feature_extraction import extract_mfcc_feature

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Data path - UPDATE THIS
    DATA_PATH = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou_Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
    
    # Audio settings
    SAMPLE_RATE = 44100
    AUDIO_DURATION = 2.0  # INCREASED to 2 seconds
    N_FFT = 512
    
    # MFCC settings
    N_MELS = 40
    N_MFCC = 13
    
    # Features to include
    INCLUDE_DELTA = True
    INCLUDE_DELTA_DELTA = True
    INCLUDE_ENERGY = True
    INCLUDE_ZCR = True
    INCLUDE_PITCH = True  # NEW: Critical for cry analysis
    INCLUDE_VOICE_QUALITY = True  # NEW: Jitter/shimmer
    INCLUDE_SPECTRAL = True
    INCLUDE_AGE = True  # NEW: Use age as feature
    
    # Augmentation
    USE_AUGMENTATION = True
    N_AUGMENTATIONS = 3
    
    # Cross-validation
    USE_LOSO = False  # Leave-One-Subject-Out (set True for more thorough eval)
    CV_FOLDS = 5
    
    # Subject-level voting
    USE_SUBJECT_VOTING = True
    
    # Feature selection
    N_FEATURES = 200
    
    RANDOM_STATE = 42


# =============================================================================
# PITCH AND VOICE QUALITY FEATURES
# =============================================================================

def compute_pitch_features(signal, fs, frame_size=0.025, frame_step=0.01):
    """
    Compute pitch (F0) related features using autocorrelation method.
    Critical for cry analysis as ASD infants may have different pitch patterns.
    """
    frame_length = int(frame_size * fs)
    hop_length = int(frame_step * fs)
    
    f0_values = []
    voiced_frames = []
    
    # Pitch range for infant cries (typically 250-600 Hz)
    min_f0, max_f0 = 200, 700
    min_lag = int(fs / max_f0)
    max_lag = int(fs / min_f0)
    
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length].astype(float)
        
        # Remove DC offset
        frame = frame - np.mean(frame)
        
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peak in valid lag range
        if max_lag < len(corr):
            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0 and np.max(search_region) > 0.3 * corr[0]:
                peak_idx = np.argmax(search_region) + min_lag
                f0 = fs / peak_idx
                f0_values.append(f0)
                voiced_frames.append(1)
            else:
                f0_values.append(0)
                voiced_frames.append(0)
        else:
            f0_values.append(0)
            voiced_frames.append(0)
    
    f0_values = np.array(f0_values)
    voiced_frames = np.array(voiced_frames)
    
    # Get only voiced F0 values
    voiced_f0 = f0_values[f0_values > 0]
    
    if len(voiced_f0) < 5:
        # Not enough voiced frames, return zeros
        return np.zeros(20)
    
    features = [
        # Basic F0 statistics
        np.mean(voiced_f0),
        np.std(voiced_f0),
        np.min(voiced_f0),
        np.max(voiced_f0),
        np.median(voiced_f0),
        
        # F0 range and variability
        np.max(voiced_f0) - np.min(voiced_f0),  # Range
        np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-10),  # Coefficient of variation
        stats.skew(voiced_f0),
        stats.kurtosis(voiced_f0),
        
        # Percentiles
        np.percentile(voiced_f0, 10),
        np.percentile(voiced_f0, 25),
        np.percentile(voiced_f0, 75),
        np.percentile(voiced_f0, 90),
        
        # F0 dynamics (how pitch changes over time)
        np.mean(np.abs(np.diff(voiced_f0))),  # Mean pitch change
        np.std(np.diff(voiced_f0)),  # Std of pitch change
        np.max(np.abs(np.diff(voiced_f0))),  # Max pitch jump
        
        # Voicing statistics
        np.mean(voiced_frames),  # Proportion of voiced frames
        np.sum(np.diff(voiced_frames) != 0),  # Voice breaks count
        
        # Rising/falling pitch
        np.sum(np.diff(voiced_f0) > 0) / (len(voiced_f0) - 1 + 1e-10),  # Rising ratio
        np.sum(np.diff(voiced_f0) < 0) / (len(voiced_f0) - 1 + 1e-10),  # Falling ratio
    ]
    
    return np.array(features)


def compute_voice_quality_features(signal, fs, frame_size=0.025, frame_step=0.01):
    """
    Compute voice quality features: jitter and shimmer approximations.
    These measure cycle-to-cycle variations in pitch and amplitude.
    """
    frame_length = int(frame_size * fs)
    hop_length = int(frame_step * fs)
    
    periods = []
    amplitudes = []
    
    min_f0, max_f0 = 200, 700
    min_period = int(fs / max_f0)
    max_period = int(fs / min_f0)
    
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length].astype(float)
        frame = frame - np.mean(frame)
        
        # Find pitch period using autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        if max_period < len(corr):
            search_region = corr[min_period:max_period]
            if len(search_region) > 0 and np.max(search_region) > 0.3 * corr[0]:
                peak_idx = np.argmax(search_region) + min_period
                periods.append(peak_idx)
                amplitudes.append(np.max(np.abs(frame)))
    
    if len(periods) < 5:
        return np.zeros(10)
    
    periods = np.array(periods)
    amplitudes = np.array(amplitudes)
    
    # Jitter measures (pitch period variation)
    period_diffs = np.abs(np.diff(periods))
    jitter_abs = np.mean(period_diffs)  # Absolute jitter
    jitter_rel = jitter_abs / (np.mean(periods) + 1e-10)  # Relative jitter
    jitter_rap = np.mean(np.abs(periods[:-2] - 2*periods[1:-1] + periods[2:])) / (3 * np.mean(periods) + 1e-10)
    
    # Shimmer measures (amplitude variation)
    amp_diffs = np.abs(np.diff(amplitudes))
    shimmer_abs = np.mean(amp_diffs)
    shimmer_rel = shimmer_abs / (np.mean(amplitudes) + 1e-10)
    shimmer_apq3 = np.mean(np.abs(amplitudes[:-2] - 2*amplitudes[1:-1] + amplitudes[2:])) / (3 * np.mean(amplitudes) + 1e-10)
    
    features = [
        jitter_abs,
        jitter_rel,
        jitter_rap,
        shimmer_abs,
        shimmer_rel,
        shimmer_apq3,
        np.std(periods) / (np.mean(periods) + 1e-10),  # Period CV
        np.std(amplitudes) / (np.mean(amplitudes) + 1e-10),  # Amplitude CV
        stats.skew(periods) if len(periods) > 2 else 0,
        stats.skew(amplitudes) if len(amplitudes) > 2 else 0,
    ]
    
    return np.array(features)


def compute_formant_features(signal, fs, n_formants=3):
    """
    Estimate formant frequencies using LPC analysis.
    Formants are important for characterizing cry sounds.
    """
    try:
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # LPC analysis
        from scipy.signal import lfilter
        
        # Use librosa's LPC if available, otherwise simple autocorrelation method
        order = 2 + fs // 1000  # LPC order
        
        # Compute LPC coefficients using autocorrelation method
        r = np.correlate(emphasized, emphasized, mode='full')
        r = r[len(r)//2:len(r)//2 + order + 1]
        
        # Levinson-Durbin recursion
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]
        
        for i in range(1, order + 1):
            lambda_val = np.sum(a[:i] * r[i:0:-1]) / (e + 1e-10)
            a[1:i+1] = a[1:i+1] - lambda_val * a[i-1::-1][:i]
            a[i] = lambda_val
            e = e * (1 - lambda_val ** 2)
        
        # Find formants from LPC roots
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]  # Keep positive frequencies
        
        # Convert to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (fs / (2 * np.pi))
        freqs = freqs[(freqs > 90) & (freqs < fs/2 - 50)]  # Valid range
        freqs = np.sort(freqs)
        
        # Get first n formants
        formants = np.zeros(n_formants)
        for i in range(min(n_formants, len(freqs))):
            formants[i] = freqs[i]
        
        return formants
        
    except Exception:
        return np.zeros(n_formants)


# =============================================================================
# ENHANCED FEATURE EXTRACTOR
# =============================================================================

class EnhancedFeatureExtractor:
    """Comprehensive feature extraction with pitch and voice quality."""
    
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
        """Compute delta coefficients."""
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
        """Compute comprehensive statistics."""
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
    
    def compute_energy_features(self, signal):
        """Compute energy features."""
        frame_length = int(self.mfcc_params['frame_size'] * self.config.SAMPLE_RATE)
        hop_length = int(self.mfcc_params['frame_step'] * self.config.SAMPLE_RATE)
        
        rms = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length].astype(float)
            rms.append(np.sqrt(np.mean(frame ** 2)))
        rms = np.array(rms) + 1e-10
        
        features = [
            np.mean(rms), np.std(rms), np.min(rms), np.max(rms),
            np.max(rms) - np.min(rms),
            stats.skew(rms), stats.kurtosis(rms),
            np.mean(np.abs(np.diff(rms))),
            np.mean(np.log(rms)), np.std(np.log(rms)),
            np.percentile(rms, 10), np.percentile(rms, 90),
        ]
        return np.array(features)
    
    def compute_zcr_features(self, signal):
        """Compute zero-crossing rate features."""
        frame_length = int(self.mfcc_params['frame_size'] * self.config.SAMPLE_RATE)
        hop_length = int(self.mfcc_params['frame_step'] * self.config.SAMPLE_RATE)
        
        zcr = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            zcr.append(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length))
        zcr = np.array(zcr)
        
        return np.array([
            np.mean(zcr), np.std(zcr), np.min(zcr), np.max(zcr),
            stats.skew(zcr), stats.kurtosis(zcr),
            np.percentile(zcr, 25), np.percentile(zcr, 75),
        ])
    
    def compute_spectral_features(self, signal):
        """Compute spectral features."""
        frame_length = int(self.mfcc_params['frame_size'] * self.config.SAMPLE_RATE)
        hop_length = int(self.mfcc_params['frame_step'] * self.config.SAMPLE_RATE)
        
        centroids, bandwidths, rolloffs, flatness = [], [], [], []
        
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length].astype(float)
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/self.config.SAMPLE_RATE)
            
            spectrum_sum = np.sum(spectrum) + 1e-10
            spectrum_norm = spectrum / spectrum_sum
            
            centroid = np.sum(freqs * spectrum_norm)
            centroids.append(centroid)
            
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum_norm))
            bandwidths.append(bandwidth)
            
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            rolloffs.append(freqs[min(rolloff_idx, len(freqs)-1)])
            
            geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arithmetic_mean = np.mean(spectrum) + 1e-10
            flatness.append(geometric_mean / arithmetic_mean)
        
        features = []
        for arr in [centroids, bandwidths, rolloffs, flatness]:
            arr = np.array(arr)
            features.extend([np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
        
        return np.array(features)
    
    def extract_features(self, signal, age=None):
        """Extract all features from audio signal."""
        # Normalize signal
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        
        signal = librosa.util.fix_length(signal, size=self.fix_len, mode='wrap')
        
        # Extract MFCC
        mfcc = extract_mfcc_feature(
            y=signal,
            fs=self.mfcc_params['fs'],
            n_fft=self.mfcc_params['n_fft'],
            frame_size=self.mfcc_params['frame_size'],
            frame_step=self.mfcc_params['frame_step'],
            n_mels=self.mfcc_params['n_mels'],
            n_mfcc=self.mfcc_params['n_mfcc']
        )
        
        # Build feature components
        components = [mfcc]
        
        if self.config.INCLUDE_DELTA:
            delta = self.compute_delta(mfcc)
            components.append(delta)
        
        if self.config.INCLUDE_DELTA_DELTA:
            delta_delta = self.compute_delta(self.compute_delta(mfcc))
            components.append(delta_delta)
        
        combined = np.hstack(components)
        stat_features = self.compute_statistics(combined)
        
        all_features = [stat_features]
        
        if self.config.INCLUDE_ENERGY:
            all_features.append(self.compute_energy_features(signal))
        
        if self.config.INCLUDE_ZCR:
            all_features.append(self.compute_zcr_features(signal))
        
        if self.config.INCLUDE_SPECTRAL:
            all_features.append(self.compute_spectral_features(signal))
        
        # NEW: Pitch features
        if self.config.INCLUDE_PITCH:
            pitch_features = compute_pitch_features(signal, self.config.SAMPLE_RATE)
            all_features.append(pitch_features)
        
        # NEW: Voice quality features
        if self.config.INCLUDE_VOICE_QUALITY:
            voice_quality = compute_voice_quality_features(signal, self.config.SAMPLE_RATE)
            all_features.append(voice_quality)
        
        # NEW: Age as feature
        if self.config.INCLUDE_AGE and age is not None:
            all_features.append(np.array([age]))
        
        return np.concatenate(all_features)


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class AudioAugmenter:
    """Audio augmentation."""
    
    def __init__(self, fs):
        self.fs = fs
    
    def augment(self, signal):
        """Apply random augmentations."""
        augmented = signal.copy().astype(float)
        
        # Time stretch
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            new_len = int(len(augmented) / rate)
            augmented = np.interp(
                np.linspace(0, 1, new_len),
                np.linspace(0, 1, len(augmented)),
                augmented
            )
        
        # Pitch shift (simple)
        if np.random.random() < 0.5:
            semitones = np.random.uniform(-2, 2)
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
        
        # Add noise
        if np.random.random() < 0.5:
            noise = np.random.randn(len(augmented)) * 0.005 * np.std(augmented)
            augmented = augmented + noise
        
        # Time shift
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.15, 0.15) * len(augmented))
            augmented = np.roll(augmented, shift)
        
        # Volume
        if np.random.random() < 0.5:
            augmented = augmented * np.random.uniform(0.7, 1.3)
        
        return augmented


# =============================================================================
# DATA LOADER
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
    """Load all data with subject tracking."""
    extractor = EnhancedFeatureExtractor(config)
    augmenter = AudioAugmenter(config.SAMPLE_RATE) if config.USE_AUGMENTATION else None
    
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
                
                # Extract features
                features = extractor.extract_features(signal, age)
                X.append(features)
                y.append(label)
                groups.append(subject_id)
                subjects.append(folder)
                n_loaded += 1
                
                # Augmentation
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
              (f" (+{n_loaded * config.N_AUGMENTATIONS} augmented)" if config.USE_AUGMENTATION else ""))
        subject_id += 1
    
    return np.array(X), np.array(y), np.array(groups), subjects


# =============================================================================
# SUBJECT-LEVEL VOTING
# =============================================================================

def evaluate_with_subject_voting(y_true, y_pred, y_prob, groups, subjects):
    """
    Aggregate predictions at subject level using voting.
    Each subject gets one final prediction based on majority vote.
    """
    subject_predictions = defaultdict(list)
    subject_probabilities = defaultdict(list)
    subject_labels = {}
    
    for i, (pred, prob, group, subj) in enumerate(zip(y_pred, y_prob, groups, subjects)):
        subject_predictions[subj].append(pred)
        subject_probabilities[subj].append(prob)
        subject_labels[subj] = y_true[i]
    
    final_preds = []
    final_probs = []
    final_true = []
    
    for subj in subject_predictions:
        # Majority voting
        preds = subject_predictions[subj]
        probs = subject_probabilities[subj]
        
        # Use mean probability for final prediction
        mean_prob = np.mean(probs)
        final_pred = 1 if mean_prob > 0.5 else 0
        
        final_preds.append(final_pred)
        final_probs.append(mean_prob)
        final_true.append(subject_labels[subj])
    
    return np.array(final_true), np.array(final_preds), np.array(final_probs)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate(X, y, groups, subjects, config):
    """Train and evaluate with proper CV."""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle NaN/Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Feature selection
    print(f"\nOriginal features: {X_scaled.shape[1]}")
    n_select = min(config.N_FEATURES, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected features: {X_selected.shape[1]}")
    
    # Models
    models = {
        'SVM_RBF': (
            SVC(probability=True, random_state=config.RANDOM_STATE),
            {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 0.01, 0.1]}
        ),
        'SVM_Linear': (
            SVC(kernel='linear', probability=True, random_state=config.RANDOM_STATE),
            {'C': [0.01, 0.1, 1, 10]}
        ),
        'Random_Forest': (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 
             'min_samples_split': [2, 5], 'class_weight': ['balanced', None]}
        ),
        'Gradient_Boosting': (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 
             'learning_rate': [0.05, 0.1, 0.2]}
        ),
        'Extra_Trees': (
            ExtraTreesClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
             'class_weight': ['balanced', None]}
        ),
        'MLP': (
            MLPClassifier(random_state=config.RANDOM_STATE, max_iter=1500),
            {'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
             'alpha': [0.0001, 0.001, 0.01]}
        ),
    }
    
    # Cross-validation
    if config.USE_LOSO:
        cv = LeaveOneGroupOut()
        n_splits = len(np.unique(groups))
        print(f"\nUsing Leave-One-Subject-Out CV ({n_splits} folds)")
    else:
        cv = GroupKFold(n_splits=config.CV_FOLDS)
        n_splits = config.CV_FOLDS
        print(f"\nUsing {config.CV_FOLDS}-Fold Group CV")
    
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
            
            # Grid search
            grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            y_pred = grid.predict(X_test)
            y_prob = grid.predict_proba(X_test)[:, 1]
            
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_true.extend(y_test)
            all_groups.extend(groups[test_idx])
            all_subjects.extend([subjects[i] for i in test_idx])
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_acc)
            
            if not config.USE_LOSO:  # Only print for grouped CV
                n_subj = len(np.unique(groups[test_idx]))
                print(f"  Fold {fold_idx+1}: {fold_acc:.4f} ({n_subj} subjects)")
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        # Sample-level metrics
        sample_acc = accuracy_score(all_true, all_preds)
        sample_f1 = f1_score(all_true, all_preds)
        sample_auc = roc_auc_score(all_true, all_probs)
        
        # Subject-level metrics (with voting)
        if config.USE_SUBJECT_VOTING:
            subj_true, subj_pred, subj_prob = evaluate_with_subject_voting(
                all_true, all_preds, all_probs, all_groups, all_subjects
            )
            subj_acc = accuracy_score(subj_true, subj_pred)
            subj_f1 = f1_score(subj_true, subj_pred)
            subj_auc = roc_auc_score(subj_true, subj_prob)
        else:
            subj_acc, subj_f1, subj_auc = sample_acc, sample_f1, sample_auc
        
        results[name] = {
            'sample_accuracy': sample_acc,
            'sample_f1': sample_f1,
            'sample_auc': sample_auc,
            'subject_accuracy': subj_acc,
            'subject_f1': subj_f1,
            'subject_auc': subj_auc,
            'fold_std': np.std(fold_scores),
            'best_params': grid.best_params_
        }
        
        best_models[name] = grid.best_estimator_
        
        print(f"\n  Sample-level:  Acc={sample_acc:.4f}, F1={sample_f1:.4f}, AUC={sample_auc:.4f}")
        print(f"  Subject-level: Acc={subj_acc:.4f}, F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
    
    # Ensemble
    print(f"\n{'─' * 50}")
    print("Training: Voting Ensemble")
    print(f"{'─' * 50}")
    
    estimators = [
        ('svm', best_models['SVM_RBF']),
        ('rf', best_models['Random_Forest']),
        ('gb', best_models['Gradient_Boosting']),
    ]
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
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    sample_acc = accuracy_score(all_true, all_preds)
    sample_f1 = f1_score(all_true, all_preds)
    sample_auc = roc_auc_score(all_true, all_probs)
    
    if config.USE_SUBJECT_VOTING:
        subj_true, subj_pred, subj_prob = evaluate_with_subject_voting(
            all_true, all_preds, all_probs, all_groups, all_subjects
        )
        subj_acc = accuracy_score(subj_true, subj_pred)
        subj_f1 = f1_score(subj_true, subj_pred)
        subj_auc = roc_auc_score(subj_true, subj_prob)
    else:
        subj_acc, subj_f1, subj_auc = sample_acc, sample_f1, sample_auc
    
    results['Voting_Ensemble'] = {
        'sample_accuracy': sample_acc,
        'sample_f1': sample_f1,
        'sample_auc': sample_auc,
        'subject_accuracy': subj_acc,
        'subject_f1': subj_f1,
        'subject_auc': subj_auc,
    }
    
    print(f"\n  Sample-level:  Acc={sample_acc:.4f}, F1={sample_f1:.4f}, AUC={sample_auc:.4f}")
    print(f"  Subject-level: Acc={subj_acc:.4f}, F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
    
    return results, best_models


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ASD vs TD CLASSIFICATION - ENHANCED PIPELINE V2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  - Audio duration: {config.AUDIO_DURATION}s")
    print(f"  - Augmentation: {config.USE_AUGMENTATION} (x{config.N_AUGMENTATIONS})")
    print(f"  - Pitch features: {config.INCLUDE_PITCH}")
    print(f"  - Voice quality: {config.INCLUDE_VOICE_QUALITY}")
    print(f"  - Age feature: {config.INCLUDE_AGE}")
    print(f"  - Subject voting: {config.USE_SUBJECT_VOTING}")
    
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
    
    print(f"\n{'Model':<20} {'Subject Acc':<15} {'Subject F1':<12} {'Subject AUC':<12}")
    print("─" * 60)
    
    for name, r in sorted_results:
        print(f"{name:<20} {r['subject_accuracy']:.4f}          {r['subject_f1']:.4f}        {r['subject_auc']:.4f}")
    
    best_name, best_r = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST: {best_name} - Subject Accuracy: {best_r['subject_accuracy']:.2%}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()