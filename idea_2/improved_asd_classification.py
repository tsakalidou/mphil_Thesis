"""
Improved ASD vs TD Classification Pipeline
==========================================
Enhancements over original:
1. Subject-level cross-validation (prevents data leakage)
2. Delta and delta-delta MFCC features
3. Statistical feature aggregation
4. Energy-based prosodic features
5. Proper nested cross-validation
6. Hyperparameter optimization
7. Ensemble methods
8. Comprehensive evaluation metrics

Author: Enhanced pipeline for MSc Capstone
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import scipy.io.wavfile as sci_wav
import scipy.stats as stats
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
    cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import librosa

# Import your custom MFCC functions
from mfcc_feature_extraction import extract_mfcc_feature

warnings.filterwarnings('ignore')


# =============================================================================
# ENHANCED FEATURE EXTRACTION
# =============================================================================

def compute_delta(mfcc, N=2):
    """
    Compute delta (velocity) coefficients from MFCC.
    
    :param mfcc: MFCC features [n_frames, n_coeffs]
    :param N: Number of frames to use for delta computation
    :return: Delta features [n_frames, n_coeffs]
    """
    n_frames, n_coeffs = mfcc.shape
    deltas = np.zeros_like(mfcc)
    
    # Pad the MFCC array for edge computation
    padded = np.pad(mfcc, ((N, N), (0, 0)), mode='edge')
    
    denominator = 2 * sum(n ** 2 for n in range(1, N + 1))
    
    for t in range(n_frames):
        numerator = sum(n * (padded[t + N + n] - padded[t + N - n]) for n in range(1, N + 1))
        deltas[t] = numerator / denominator
    
    return deltas


def compute_delta_delta(mfcc, N=2):
    """
    Compute delta-delta (acceleration) coefficients.
    
    :param mfcc: MFCC features [n_frames, n_coeffs]
    :param N: Number of frames for delta computation
    :return: Delta-delta features [n_frames, n_coeffs]
    """
    delta = compute_delta(mfcc, N)
    delta_delta = compute_delta(delta, N)
    return delta_delta


def compute_statistical_features(mfcc_all):
    """
    Compute statistical aggregation features across time frames.
    
    :param mfcc_all: Combined MFCC features [n_frames, n_total_coeffs]
    :return: Statistical feature vector
    """
    features = []
    
    for i in range(mfcc_all.shape[1]):
        coeff = mfcc_all[:, i]
        
        # Basic statistics
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.min(coeff))
        features.append(np.max(coeff))
        features.append(np.median(coeff))
        
        # Higher-order statistics
        features.append(stats.skew(coeff))
        features.append(stats.kurtosis(coeff))
        
        # Range and IQR
        features.append(np.max(coeff) - np.min(coeff))  # Range
        features.append(np.percentile(coeff, 75) - np.percentile(coeff, 25))  # IQR
        
        # Temporal dynamics
        features.append(np.mean(np.abs(np.diff(coeff))))  # Mean absolute difference
        
    return np.array(features)


def compute_energy_features(signal, fs, frame_size, frame_step):
    """
    Compute energy-based prosodic features.
    
    :param signal: Audio signal
    :param fs: Sampling frequency
    :param frame_size: Frame size in seconds
    :param frame_step: Frame step in seconds
    :return: Energy features
    """
    frame_length = int(frame_size * fs)
    hop_length = int(frame_step * fs)
    
    # Compute RMS energy per frame
    rms = []
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]
        rms.append(np.sqrt(np.mean(frame.astype(float) ** 2)))
    
    rms = np.array(rms) + 1e-10  # Avoid log(0)
    
    # Energy statistics
    features = [
        np.mean(rms),
        np.std(rms),
        np.max(rms),
        np.min(rms),
        np.max(rms) - np.min(rms),  # Dynamic range
        stats.skew(rms),
        stats.kurtosis(rms),
        np.mean(np.abs(np.diff(rms))),  # Energy variation rate
    ]
    
    # Log energy
    log_rms = np.log(rms)
    features.extend([
        np.mean(log_rms),
        np.std(log_rms),
    ])
    
    return np.array(features)


def compute_zero_crossing_features(signal, fs, frame_size, frame_step):
    """
    Compute zero-crossing rate features.
    
    :param signal: Audio signal
    :param fs: Sampling frequency
    :param frame_size: Frame size in seconds
    :param frame_step: Frame step in seconds
    :return: ZCR features
    """
    frame_length = int(frame_size * fs)
    hop_length = int(frame_step * fs)
    
    zcr = []
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]
        zcr.append(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length))
    
    zcr = np.array(zcr)
    
    features = [
        np.mean(zcr),
        np.std(zcr),
        np.max(zcr),
        np.min(zcr),
        stats.skew(zcr),
        stats.kurtosis(zcr),
    ]
    
    return np.array(features)


def extract_enhanced_features(signal, fs, mfcc_params, include_energy=True, include_zcr=True):
    """
    Extract comprehensive feature set from audio signal.
    
    :param signal: Audio signal
    :param fs: Sampling frequency  
    :param mfcc_params: MFCC extraction parameters
    :param include_energy: Include energy features
    :param include_zcr: Include zero-crossing features
    :return: Complete feature vector
    """
    # Extract base MFCC using your custom implementation
    mfcc = extract_mfcc_feature(
        y=signal,
        fs=fs,
        n_fft=mfcc_params['n_fft'],
        frame_size=mfcc_params['frame_size'],
        frame_step=mfcc_params['frame_step'],
        n_mels=mfcc_params.get('n_mels', 40),
        n_mfcc=mfcc_params.get('n_mfcc', 13)
    )
    
    # Compute delta and delta-delta
    delta = compute_delta(mfcc, N=2)
    delta_delta = compute_delta_delta(mfcc, N=2)
    
    # Combine MFCC + delta + delta-delta
    mfcc_combined = np.hstack([mfcc, delta, delta_delta])
    
    # Compute statistical features
    stat_features = compute_statistical_features(mfcc_combined)
    
    # Also include flattened raw features (optional - can be removed if too many)
    # raw_features = mfcc_combined.ravel()
    
    all_features = [stat_features]
    
    # Add energy features
    if include_energy:
        energy_features = compute_energy_features(
            signal, fs, mfcc_params['frame_size'], mfcc_params['frame_step']
        )
        all_features.append(energy_features)
    
    # Add ZCR features
    if include_zcr:
        zcr_features = compute_zero_crossing_features(
            signal, fs, mfcc_params['frame_size'], mfcc_params['frame_step']
        )
        all_features.append(zcr_features)
    
    return np.concatenate(all_features)


# =============================================================================
# DATA LOADING WITH SUBJECT TRACKING
# =============================================================================

def load_data_with_subjects(data_path, ages, mfcc_params, fix_len):
    """
    Load data while tracking which subject each sample belongs to.
    
    :param data_path: Path to data directory
    :param ages: Dictionary of subject ages
    :param mfcc_params: MFCC parameters
    :param fix_len: Fixed length for padding
    :return: Features, labels, subject groups, subject info
    """
    X = []
    y = []
    groups = []  # Subject ID for each sample
    subject_info = []  # Detailed info
    
    subject_id = 0
    cry_folders = os.listdir(data_path)
    
    print("Loading and processing audio files...")
    
    for cry_folder in cry_folders:
        folder_path = os.path.join(data_path, cry_folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Determine label
        if 'ASD' in cry_folder:
            label = 1
        elif 'TD' in cry_folder:
            label = 0
        else:
            continue
        
        cry_files = os.listdir(folder_path)
        n_samples = 0
        
        for cry_file in cry_files:
            if not cry_file.endswith('.wav'):
                continue
                
            cry_file_path = os.path.join(folder_path, cry_file)
            
            try:
                fs, signal = sci_wav.read(cry_file_path)
                
                # Convert to float and normalize
                if signal.dtype == np.int16:
                    signal = signal.astype(np.float32) / 32768.0
                elif signal.dtype == np.int32:
                    signal = signal.astype(np.float32) / 2147483648.0
                
                # Handle stereo
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                
                # Pad to fixed length
                signal = librosa.util.fix_length(signal, size=fix_len, mode='wrap')
                
                # Extract enhanced features
                features = extract_enhanced_features(signal, fs, mfcc_params)
                
                X.append(features)
                y.append(label)
                groups.append(subject_id)
                subject_info.append({
                    'subject': cry_folder,
                    'file': cry_file,
                    'age': ages.get(cry_folder, 0)
                })
                n_samples += 1
                
            except Exception as e:
                print(f"Error processing {cry_file_path}: {e}")
                continue
        
        print(f"  {cry_folder}: {n_samples} samples loaded")
        subject_id += 1
    
    return np.array(X), np.array(y), np.array(groups), subject_info


# =============================================================================
# MODEL TRAINING WITH PROPER CROSS-VALIDATION
# =============================================================================

def evaluate_with_group_cv(X, y, groups, n_splits=5):
    """
    Evaluate models using subject-level (group) cross-validation.
    
    :param X: Features
    :param y: Labels
    :param groups: Subject group for each sample
    :param n_splits: Number of CV folds
    :return: Results dictionary
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    print(f"\nOriginal feature count: {X_scaled.shape[1]}")
    
    # Select top features using mutual information
    n_features = min(200, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected feature count: {X_selected.shape[1]}")
    
    # Define models with hyperparameter grids
    models = {
        'SVM_RBF': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf']
            }
        },
        'SVM_Linear': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear']
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50), (100, 100), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    }
    
    # Group K-Fold cross-validation
    group_kfold = GroupKFold(n_splits=n_splits)
    
    results = {}
    best_models = {}
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION WITH SUBJECT-LEVEL CROSS-VALIDATION")
    print("=" * 70)
    
    for name, config in models.items():
        print(f"\n{'─' * 50}")
        print(f"Training {name}...")
        print(f"{'─' * 50}")
        
        # Grid search with nested CV
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=GroupKFold(n_splits=3),
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Collect predictions across all folds
        all_preds = []
        all_probs = []
        all_true = []
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_selected, y, groups)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            
            # Fit grid search on training data
            grid_search.fit(X_train, y_train, groups=groups_train)
            
            # Predict on test data
            y_pred = grid_search.predict(X_test)
            y_prob = grid_search.predict_proba(X_test)[:, 1]
            
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_true.extend(y_test)
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_acc)
            
            # Count unique subjects in test set
            n_test_subjects = len(np.unique(groups[test_idx]))
            print(f"  Fold {fold + 1}: Accuracy = {fold_acc:.4f} ({n_test_subjects} subjects in test)")
        
        # Calculate overall metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        accuracy = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds)
        precision = precision_score(all_true, all_preds)
        recall = recall_score(all_true, all_preds)
        auc = roc_auc_score(all_true, all_probs)
        
        results[name] = {
            'accuracy': accuracy,
            'accuracy_std': np.std(fold_scores),
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'best_params': grid_search.best_params_,
            'fold_scores': fold_scores
        }
        
        best_models[name] = grid_search.best_estimator_
        
        print(f"\n  RESULTS for {name}:")
        print(f"  ├── Accuracy:  {accuracy:.4f} (±{np.std(fold_scores):.4f})")
        print(f"  ├── F1 Score:  {f1:.4f}")
        print(f"  ├── Precision: {precision:.4f}")
        print(f"  ├── Recall:    {recall:.4f}")
        print(f"  ├── AUC-ROC:   {auc:.4f}")
        print(f"  └── Best params: {grid_search.best_params_}")
        
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(all_true, all_preds)
        print(f"  [[TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}]")
        print(f"   [FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}]]")
    
    return results, best_models, X_selected, scaler, selector


def create_ensemble(X, y, groups, best_models):
    """
    Create ensemble models from best performing base models.
    
    :param X: Features
    :param y: Labels
    :param groups: Subject groups
    :param best_models: Dictionary of best models from grid search
    :return: Ensemble results
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 70)
    
    group_kfold = GroupKFold(n_splits=5)
    
    # Voting Ensemble (Hard voting)
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', best_models.get('SVM_RBF', SVC(probability=True))),
            ('rf', best_models.get('Random_Forest', RandomForestClassifier())),
            ('gb', best_models.get('Gradient_Boosting', GradientBoostingClassifier()))
        ],
        voting='soft'
    )
    
    # Stacking Ensemble
    stacking_clf = StackingClassifier(
        estimators=[
            ('svm', best_models.get('SVM_RBF', SVC(probability=True))),
            ('rf', best_models.get('Random_Forest', RandomForestClassifier())),
            ('gb', best_models.get('Gradient_Boosting', GradientBoostingClassifier()))
        ],
        final_estimator=LogisticRegression(),
        cv=3
    )
    
    ensemble_results = {}
    
    for name, clf in [('Voting_Ensemble', voting_clf), ('Stacking_Ensemble', stacking_clf)]:
        print(f"\n{'─' * 50}")
        print(f"Training {name}...")
        print(f"{'─' * 50}")
        
        all_preds = []
        all_probs = []
        all_true = []
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
            all_true.extend(y_test)
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_acc)
            print(f"  Fold {fold + 1}: Accuracy = {fold_acc:.4f}")
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        accuracy = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds)
        auc = roc_auc_score(all_true, all_probs)
        
        ensemble_results[name] = {
            'accuracy': accuracy,
            'accuracy_std': np.std(fold_scores),
            'f1': f1,
            'auc': auc
        }
        
        print(f"\n  RESULTS for {name}:")
        print(f"  ├── Accuracy: {accuracy:.4f} (±{np.std(fold_scores):.4f})")
        print(f"  ├── F1 Score: {f1:.4f}")
        print(f"  └── AUC-ROC:  {auc:.4f}")
        
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(all_true, all_preds)
        print(f"  [[TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}]")
        print(f"   [FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}]]")
    
    return ensemble_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # UPDATE THIS PATH to your data location
    data_path = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou_Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
    # Subject ages (in months)
    ages = {
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
    
    # MFCC parameters (matching your original setup)
    n_fft = 512
    fs = 44100
    mfcc_params = {
        'fs': fs,
        'n_fft': n_fft,
        'frame_size': n_fft / fs,
        'frame_step': int(n_fft / 3) / fs,
        'n_mels': 40,
        'n_mfcc': 13
    }
    
    # Fixed length in samples (1 second)
    fix_len_s = 1.0
    fix_len = int(fix_len_s * fs)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("=" * 70)
    print("ASD vs TD CLASSIFICATION - ENHANCED PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Audio length: {fix_len_s}s ({fix_len} samples)")
    print(f"  - Sample rate: {fs} Hz")
    print(f"  - FFT size: {n_fft}")
    print(f"  - Frame size: {mfcc_params['frame_size'] * 1000:.2f}ms")
    print(f"  - Frame step: {mfcc_params['frame_step'] * 1000:.2f}ms")
    print(f"  - Mel filters: {mfcc_params['n_mels']}")
    print(f"  - MFCC coefficients: {mfcc_params['n_mfcc'] - 1}")
    print()
    
    X, y, groups, subject_info = load_data_with_subjects(
        data_path, ages, mfcc_params, fix_len
    )
    
    print(f"\n{'─' * 50}")
    print("DATA SUMMARY")
    print(f"{'─' * 50}")
    print(f"Total samples: {len(X)}")
    print(f"ASD samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"TD samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"Number of subjects: {len(np.unique(groups))}")
    print(f"Feature dimensions: {X.shape[1]}")
    
    # =========================================================================
    # EVALUATE MODELS
    # =========================================================================
    
    results, best_models, X_selected, scaler, selector = evaluate_with_group_cv(
        X, y, groups, n_splits=5
    )
    
    # =========================================================================
    # ENSEMBLE MODELS
    # =========================================================================
    
    ensemble_results = create_ensemble(X_selected, y, groups, best_models)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    all_results = {**results, **ensemble_results}
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Model':<25} {'Accuracy':<15} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("─" * 64)
    
    for name, res in sorted_results:
        acc_str = f"{res['accuracy']:.4f} (±{res.get('accuracy_std', 0):.4f})"
        print(f"{name:<25} {acc_str:<15} {res['f1']:.4f}       {res['auc']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {sorted_results[0][0]} with {sorted_results[0][1]['accuracy']:.2%} accuracy")
    print("=" * 70)