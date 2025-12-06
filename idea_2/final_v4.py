"""
FIXED ASD CLASSIFICATION PIPELINE V4
====================================
CRITICAL FIX: Proper Stratified Group K-Fold

The problem was extreme class imbalance across folds:
- Fold 4 had 10 ASD vs 3 TD subjects (90% ASD!)
- Fold 3 had 17 ASD vs 55 TD samples

This version uses StratifiedGroupKFold to ensure:
1. Subjects stay together (no data leakage)
2. Class balance is maintained across folds
3. Shuffling prevents alphabetical clustering

Author: Enhanced for MSc Capstone
"""

import os
import warnings
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import scipy.io.wavfile as sci_wav
import scipy.stats as stats
from scipy import signal as scipy_signal
import librosa

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import (
    StratifiedGroupKFold,  # KEY FIX!
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from mfcc_feature_extraction import extract_mfcc_feature

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # UPDATE THIS PATH
    DATA_PATH = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou_Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
    
    # Audio settings
    SAMPLE_RATE = 44100
    AUDIO_DURATION = 2.0  # seconds
    N_FFT = 512
    
    # MFCC settings
    N_MELS = 40
    N_MFCC = 13
    
    # Features
    INCLUDE_DELTA = True
    INCLUDE_DELTA_DELTA = True
    INCLUDE_PITCH = True
    INCLUDE_VOICE_QUALITY = True
    INCLUDE_SPECTRAL = True
    INCLUDE_AGE = True
    
    # Augmentation
    USE_AUGMENTATION = True
    N_AUGMENTATIONS = 4
    
    # CV settings - KEY FIX
    CV_FOLDS = 5
    SHUFFLE_FOLDS = True  # Shuffle to break alphabetical order
    
    # Feature selection
    N_FEATURES = 200
    
    RANDOM_STATE = 42


# =============================================================================
# SUBJECT AGES
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


# =============================================================================
# FEATURE EXTRACTION (same as before)
# =============================================================================

def extract_pitch_features(signal, fs):
    """Extract pitch features using librosa pyin."""
    try:
        signal = signal.astype(np.float32)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal, fmin=150, fmax=800, sr=fs,
            frame_length=2048, hop_length=512
        )
        
        f0_clean = f0[~np.isnan(f0)]
        voiced_probs_clean = voiced_probs[~np.isnan(voiced_probs)]
        
        if len(f0_clean) < 5:
            return np.zeros(20)
        
        features = [
            np.mean(f0_clean), np.std(f0_clean), np.min(f0_clean), np.max(f0_clean),
            np.median(f0_clean), np.max(f0_clean) - np.min(f0_clean),
            np.std(f0_clean) / (np.mean(f0_clean) + 1e-10),
            stats.skew(f0_clean), stats.kurtosis(f0_clean),
            np.percentile(f0_clean, 10), np.percentile(f0_clean, 25),
            np.percentile(f0_clean, 75), np.percentile(f0_clean, 90),
            np.mean(np.abs(np.diff(f0_clean))), np.std(np.diff(f0_clean)),
            np.max(np.abs(np.diff(f0_clean))),
            np.mean(voiced_probs_clean), np.std(voiced_probs_clean),
            np.sum(~np.isnan(f0)) / len(f0),
            np.sum(np.diff(np.sign(np.diff(f0_clean))) != 0) / (len(f0_clean) + 1e-10),
        ]
        return np.array(features)
    except:
        return np.zeros(20)


def extract_voice_quality(signal, fs):
    """Extract jitter and shimmer."""
    frame_length = int(0.025 * fs)
    hop_length = int(0.010 * fs)
    
    min_period = int(fs / 800)
    max_period = int(fs / 150)
    
    periods, amplitudes = [], []
    
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length].astype(float)
        frame = frame - np.mean(frame)
        
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        if max_period < len(corr):
            search = corr[min_period:max_period]
            if len(search) > 0 and np.max(search) > 0.3 * corr[0]:
                peak_idx = np.argmax(search) + min_period
                periods.append(peak_idx)
                amplitudes.append(np.max(np.abs(frame)))
    
    if len(periods) < 5:
        return np.zeros(10)
    
    periods = np.array(periods)
    amplitudes = np.array(amplitudes)
    
    jitter_abs = np.mean(np.abs(np.diff(periods)))
    jitter_rel = jitter_abs / (np.mean(periods) + 1e-10)
    shimmer_abs = np.mean(np.abs(np.diff(amplitudes)))
    shimmer_rel = shimmer_abs / (np.mean(amplitudes) + 1e-10)
    
    return np.array([
        jitter_abs, jitter_rel,
        shimmer_abs, shimmer_rel,
        np.std(periods) / (np.mean(periods) + 1e-10),
        np.std(amplitudes) / (np.mean(amplitudes) + 1e-10),
        stats.skew(periods) if len(periods) > 2 else 0,
        stats.skew(amplitudes) if len(amplitudes) > 2 else 0,
        stats.kurtosis(periods) if len(periods) > 3 else 0,
        stats.kurtosis(amplitudes) if len(amplitudes) > 3 else 0,
    ])


def extract_spectral_features(signal, fs):
    """Extract spectral features."""
    try:
        signal = signal.astype(np.float32)
        
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=fs)[0]
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=fs)[0]
        flatness = librosa.feature.spectral_flatness(y=signal)[0]
        rms = librosa.feature.rms(y=signal)[0]
        centroid = librosa.feature.spectral_centroid(y=signal, sr=fs)[0]
        contrast = librosa.feature.spectral_contrast(y=signal, sr=fs, n_bands=6, fmin=200)
        
        features = [
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flatness), np.std(flatness),
            np.mean(rms), np.std(rms), np.max(rms),
            np.mean(centroid), np.std(centroid),
        ]
        
        for i in range(contrast.shape[0]):
            features.extend([np.mean(contrast[i]), np.std(contrast[i])])
        
        return np.array(features)
    except:
        return np.zeros(25)


class FeatureExtractor:
    """Feature extraction class."""
    
    def __init__(self, config):
        self.config = config
        self.fix_len = int(config.AUDIO_DURATION * config.SAMPLE_RATE)
        self.mfcc_params = {
            'fs': config.SAMPLE_RATE,
            'n_fft': config.N_FFT,
            'frame_size': config.N_FFT / config.SAMPLE_RATE,
            'frame_step': int(config.N_FFT / 3) / config.SAMPLE_RATE,
            'n_mels': config.N_MELS,
            'n_mfcc': config.N_MFCC
        }
    
    def compute_delta(self, features, N=2):
        n_frames, n_coeffs = features.shape
        deltas = np.zeros_like(features)
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        denom = 2 * sum(n ** 2 for n in range(1, N + 1))
        
        for t in range(n_frames):
            num = sum(n * (padded[t + N + n] - padded[t + N - n]) for n in range(1, N + 1))
            deltas[t] = num / denom
        return deltas
    
    def compute_statistics(self, features):
        stats_list = []
        for i in range(features.shape[1]):
            coeff = features[:, i]
            stats_list.extend([
                np.mean(coeff), np.std(coeff), np.min(coeff), np.max(coeff),
                np.median(coeff), stats.skew(coeff), stats.kurtosis(coeff),
                np.percentile(coeff, 25), np.percentile(coeff, 75),
                np.mean(np.abs(np.diff(coeff))),
            ])
        return np.array(stats_list)
    
    def extract(self, signal, age=None):
        """Extract all features."""
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        
        signal = librosa.util.fix_length(signal, size=self.fix_len, mode='wrap')
        
        # MFCC
        mfcc = extract_mfcc_feature(
            y=signal, fs=self.mfcc_params['fs'],
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
        all_features = [self.compute_statistics(combined)]
        
        if self.config.INCLUDE_PITCH:
            all_features.append(extract_pitch_features(signal, self.config.SAMPLE_RATE))
        
        if self.config.INCLUDE_VOICE_QUALITY:
            all_features.append(extract_voice_quality(signal, self.config.SAMPLE_RATE))
        
        if self.config.INCLUDE_SPECTRAL:
            all_features.append(extract_spectral_features(signal, self.config.SAMPLE_RATE))
        
        if self.config.INCLUDE_AGE and age is not None:
            all_features.append(np.array([age, age**2 / 1000, np.log(age + 1)]))
        
        return np.concatenate(all_features)


# =============================================================================
# AUGMENTATION
# =============================================================================

class Augmenter:
    def __init__(self, fs):
        self.fs = fs
    
    def augment(self, signal):
        aug = signal.copy().astype(float)
        
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            new_len = int(len(aug) / rate)
            aug = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, len(aug)), aug)
        
        if np.random.random() < 0.5:
            semitones = np.random.uniform(-2, 2)
            rate = 2 ** (semitones / 12.0)
            new_len = int(len(aug) / rate)
            stretched = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, len(aug)), aug)
            aug = np.interp(np.linspace(0, 1, len(signal)), np.linspace(0, 1, len(stretched)), stretched)
        
        if np.random.random() < 0.5:
            aug = aug + np.random.randn(len(aug)) * 0.005 * np.std(aug)
        
        if np.random.random() < 0.5:
            aug = np.roll(aug, int(np.random.uniform(-0.15, 0.15) * len(aug)))
        
        if np.random.random() < 0.5:
            aug = aug * np.random.uniform(0.7, 1.3)
        
        return aug


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config):
    """Load data with augmentation."""
    extractor = FeatureExtractor(config)
    augmenter = Augmenter(config.SAMPLE_RATE) if config.USE_AUGMENTATION else None
    
    X, y, groups, subjects = [], [], [], []
    subject_id = 0
    
    print("Loading data...")
    
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
                
                features = extractor.extract(signal, age)
                X.append(features)
                y.append(label)
                groups.append(subject_id)
                subjects.append(folder)
                n_loaded += 1
                
                if config.USE_AUGMENTATION:
                    for _ in range(config.N_AUGMENTATIONS):
                        aug_features = extractor.extract(augmenter.augment(signal), age)
                        X.append(aug_features)
                        y.append(label)
                        groups.append(subject_id)
                        subjects.append(folder)
            except Exception as e:
                print(f"  Error: {file}: {e}")
        
        print(f"  {folder}: {n_loaded}" + (f" (+{n_loaded * config.N_AUGMENTATIONS} aug)" if config.USE_AUGMENTATION else ""))
        subject_id += 1
    
    return np.array(X), np.array(y), np.array(groups), subjects


# =============================================================================
# SUBJECT-LEVEL VOTING
# =============================================================================

def subject_level_voting(y_true, y_pred, y_prob, groups, subjects):
    """Aggregate predictions at subject level."""
    subject_probs = defaultdict(list)
    subject_labels = {}
    
    for i, (prob, group, subj) in enumerate(zip(y_prob, groups, subjects)):
        subject_probs[subj].append(prob)
        subject_labels[subj] = y_true[i]
    
    final_true, final_pred, final_prob = [], [], []
    
    for subj in subject_probs:
        avg_prob = np.mean(subject_probs[subj])
        final_prob.append(avg_prob)
        final_pred.append(1 if avg_prob > 0.5 else 0)
        final_true.append(subject_labels[subj])
    
    return np.array(final_true), np.array(final_pred), np.array(final_prob)


# =============================================================================
# VERIFY FOLD BALANCE (for debugging)
# =============================================================================

def verify_fold_balance(y, groups, subjects, cv):
    """Print fold composition to verify balance."""
    print("\n📊 FOLD BALANCE VERIFICATION:")
    print("-" * 60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        test_subjects = set([subjects[i] for i in test_idx])
        test_y = y[test_idx]
        
        asd_subj = len([s for s in test_subjects if 'ASD' in s])
        td_subj = len([s for s in test_subjects if 'TD' in s])
        asd_samples = sum(test_y)
        td_samples = len(test_y) - sum(test_y)
        
        print(f"  Fold {fold_idx + 1}: {asd_subj} ASD / {td_subj} TD subjects, "
              f"{asd_samples} ASD / {td_samples} TD samples")
    print()


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(X, y, groups, subjects, config):
    """Train with STRATIFIED Group K-Fold."""
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    print(f"\nOriginal features: {X_scaled.shape[1]}")
    n_select = min(config.N_FEATURES, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected features: {X_selected.shape[1]}")
    
    # KEY FIX: Use StratifiedGroupKFold with shuffle
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=config.SHUFFLE_FOLDS, 
                               random_state=config.RANDOM_STATE)
    
    # Verify fold balance
    verify_fold_balance(y, groups, subjects, cv)
    
    # Models
    models = {
        'SVM_RBF': (
            SVC(probability=True, random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 0.01, 0.1]}
        ),
        'Random_Forest': (
            RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'n_estimators': [200, 300], 'max_depth': [15, 20, None], 'min_samples_split': [2, 5]}
        ),
        'Extra_Trees': (
            ExtraTreesClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'n_estimators': [200, 300], 'max_depth': [15, 20, None]}
        ),
        'Gradient_Boosting': (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}
        ),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = (
            XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, 
                         eval_metric='logloss'),
            {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}
        )
    
    results = {}
    best_models = {}
    
    print("=" * 70)
    print("MODEL TRAINING WITH STRATIFIED GROUP K-FOLD")
    print("=" * 70)
    
    for name, (model, params) in models.items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print(f"{'─' * 50}")
        
        all_preds, all_probs, all_true = [], [], []
        all_groups, all_subjects = [], []
        fold_sample_accs = []
        fold_subject_accs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Grid search
            search = RandomizedSearchCV(model, params, n_iter=15, cv=3, scoring='accuracy',
                                       n_jobs=-1, random_state=config.RANDOM_STATE)
            search.fit(X_train, y_train)
            
            y_pred = search.predict(X_test)
            y_prob = search.predict_proba(X_test)[:, 1]
            
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_true.extend(y_test)
            all_groups.extend(groups[test_idx])
            all_subjects.extend([subjects[i] for i in test_idx])
            
            # Sample accuracy
            sample_acc = accuracy_score(y_test, y_pred)
            fold_sample_accs.append(sample_acc)
            
            # Subject accuracy for this fold
            subj_true, subj_pred, _ = subject_level_voting(
                y_test, y_pred, y_prob, groups[test_idx], [subjects[i] for i in test_idx]
            )
            subj_acc = accuracy_score(subj_true, subj_pred)
            fold_subject_accs.append(subj_acc)
            
            # Count subjects
            n_asd = len([s for s in set([subjects[i] for i in test_idx]) if 'ASD' in s])
            n_td = len([s for s in set([subjects[i] for i in test_idx]) if 'TD' in s])
            
            print(f"  Fold {fold_idx+1}: Sample={sample_acc:.4f}, Subject={subj_acc:.4f} "
                  f"({n_asd} ASD / {n_td} TD subjects)")
        
        # Overall metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        sample_acc = accuracy_score(all_true, all_preds)
        sample_auc = roc_auc_score(all_true, all_probs)
        
        subj_true, subj_pred, subj_prob = subject_level_voting(
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
            'fold_std': np.std(fold_subject_accs),
            'best_params': search.best_params_
        }
        
        best_models[name] = search.best_estimator_
        
        print(f"\n  OVERALL:")
        print(f"  ├── Sample:  Acc={sample_acc:.4f}, AUC={sample_auc:.4f}")
        print(f"  ├── Subject: Acc={subj_acc:.4f} (±{np.std(fold_subject_accs):.4f}), "
              f"F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
        
        cm = confusion_matrix(subj_true, subj_pred)
        print(f"  └── Subject CM: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Ensemble
    print(f"\n{'─' * 50}")
    print("Training: Voting Ensemble")
    print(f"{'─' * 50}")
    
    estimators = [(name.lower(), best_models[name]) for name in best_models]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    all_preds, all_probs, all_true = [], [], []
    all_groups, all_subjects = [], []
    fold_subject_accs = []
    
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
        
        subj_true, subj_pred, _ = subject_level_voting(
            y_test, y_pred, y_prob, groups[test_idx], [subjects[i] for i in test_idx]
        )
        subj_acc = accuracy_score(subj_true, subj_pred)
        fold_subject_accs.append(subj_acc)
        
        n_asd = len([s for s in set([subjects[i] for i in test_idx]) if 'ASD' in s])
        n_td = len([s for s in set([subjects[i] for i in test_idx]) if 'TD' in s])
        print(f"  Fold {fold_idx+1}: Subject={subj_acc:.4f} ({n_asd} ASD / {n_td} TD)")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    subj_true, subj_pred, subj_prob = subject_level_voting(
        all_true, all_preds, all_probs, all_groups, all_subjects
    )
    subj_acc = accuracy_score(subj_true, subj_pred)
    subj_f1 = f1_score(subj_true, subj_pred)
    subj_auc = roc_auc_score(subj_true, subj_prob)
    
    results['Voting_Ensemble'] = {
        'sample_accuracy': accuracy_score(all_true, all_preds),
        'sample_auc': roc_auc_score(all_true, all_probs),
        'subject_accuracy': subj_acc,
        'subject_f1': subj_f1,
        'subject_auc': subj_auc,
        'fold_std': np.std(fold_subject_accs),
    }
    
    print(f"\n  OVERALL:")
    print(f"  ├── Subject: Acc={subj_acc:.4f} (±{np.std(fold_subject_accs):.4f}), "
          f"F1={subj_f1:.4f}, AUC={subj_auc:.4f}")
    cm = confusion_matrix(subj_true, subj_pred)
    print(f"  └── Subject CM: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results, best_models


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ASD vs TD CLASSIFICATION - FIXED PIPELINE V4")
    print("KEY FIX: StratifiedGroupKFold for balanced folds")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  - Audio duration: {config.AUDIO_DURATION}s")
    print(f"  - Augmentation: {config.USE_AUGMENTATION} (x{config.N_AUGMENTATIONS})")
    print(f"  - CV: {config.CV_FOLDS}-fold StratifiedGroupKFold (shuffle={config.SHUFFLE_FOLDS})")
    
    # Load data
    X, y, groups, subjects = load_data(config)
    
    print(f"\nData loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  ASD: {np.sum(y == 1)}, TD: {np.sum(y == 0)}")
    print(f"  Subjects: {len(np.unique(groups))}")
    
    # Train
    results, best_models = train_and_evaluate(X, y, groups, subjects, config)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (SUBJECT-LEVEL)")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['subject_accuracy'], reverse=True)
    
    print(f"\n{'Model':<20} {'Subj Acc':<15} {'Subj F1':<10} {'Subj AUC':<10}")
    print("─" * 55)
    
    for name, r in sorted_results:
        std_str = f"±{r.get('fold_std', 0):.3f}" if 'fold_std' in r else ""
        print(f"{name:<20} {r['subject_accuracy']:.4f} {std_str:<8} {r['subject_f1']:.4f}      {r['subject_auc']:.4f}")
    
    best_name, best_r = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST: {best_name} - Subject Accuracy: {best_r['subject_accuracy']:.2%}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()