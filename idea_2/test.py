"""
DIAGNOSTIC ANALYSIS: Why are Folds 3 and 4 Failing?
====================================================
This script investigates:
1. Which subjects are in each fold
2. Which subjects are consistently misclassified
3. Subject characteristics (age, sample count, etc.)
4. Potential outliers or ambiguous cases

Run this BEFORE the main pipeline to understand your data better.
"""

import os
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy.io.wavfile as sci_wav
import scipy.stats as stats
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from mfcc_feature_extraction import extract_mfcc_feature

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# UPDATE THIS PATH
DATA_PATH = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou_Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'

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

SAMPLE_RATE = 44100
AUDIO_DURATION = 2.0
N_FFT = 512


# =============================================================================
# SIMPLE FEATURE EXTRACTION
# =============================================================================

def extract_simple_features(signal, fs):
    """Extract basic features for diagnostic purposes."""
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    fix_len = int(AUDIO_DURATION * fs)
    signal = librosa.util.fix_length(signal, size=fix_len, mode='wrap')
    
    # Basic MFCC
    mfcc_params = {
        'fs': fs,
        'n_fft': N_FFT,
        'frame_size': N_FFT / fs,
        'frame_step': int(N_FFT / 3) / fs,
        'n_mels': 40,
        'n_mfcc': 13
    }
    
    mfcc = extract_mfcc_feature(
        y=signal, fs=fs, n_fft=N_FFT,
        frame_size=mfcc_params['frame_size'],
        frame_step=mfcc_params['frame_step']
    )
    
    # Statistics
    features = []
    for i in range(mfcc.shape[1]):
        coeff = mfcc[:, i]
        features.extend([np.mean(coeff), np.std(coeff), np.min(coeff), np.max(coeff)])
    
    # Energy
    rms = np.sqrt(np.mean(signal ** 2))
    features.append(rms)
    
    return np.array(features)


# =============================================================================
# LOAD DATA WITH DETAILED TRACKING
# =============================================================================

def load_data_detailed():
    """Load data with detailed subject information."""
    X, y, groups, subjects = [], [], [], []
    subject_info = {}
    
    subject_id = 0
    folders = sorted(os.listdir(DATA_PATH))
    
    print("Loading data...")
    
    for folder in folders:
        folder_path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        
        if 'ASD' in folder:
            label = 1
            label_name = 'ASD'
        elif 'TD' in folder:
            label = 0
            label_name = 'TD'
        else:
            continue
        
        age = AGES.get(folder, 30)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        n_samples = 0
        
        for file in files:
            try:
                fs, signal = sci_wav.read(os.path.join(folder_path, file))
                features = extract_simple_features(signal, fs)
                
                X.append(features)
                y.append(label)
                groups.append(subject_id)
                subjects.append(folder)
                n_samples += 1
                
            except Exception as e:
                print(f"  Error: {file}: {e}")
        
        subject_info[folder] = {
            'subject_id': subject_id,
            'label': label_name,
            'age': age,
            'n_samples': n_samples
        }
        
        subject_id += 1
    
    return np.array(X), np.array(y), np.array(groups), subjects, subject_info


# =============================================================================
# ANALYZE FOLD COMPOSITION
# =============================================================================

def analyze_folds(X, y, groups, subjects, subject_info):
    """Analyze what's in each fold."""
    
    cv = GroupKFold(n_splits=5)
    
    print("\n" + "=" * 80)
    print("FOLD COMPOSITION ANALYSIS")
    print("=" * 80)
    
    fold_compositions = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        test_subjects = set([subjects[i] for i in test_idx])
        test_labels = [y[i] for i in test_idx]
        
        asd_subjects = [s for s in test_subjects if 'ASD' in s]
        td_subjects = [s for s in test_subjects if 'TD' in s]
        
        # Get ages
        asd_ages = [subject_info[s]['age'] for s in asd_subjects]
        td_ages = [subject_info[s]['age'] for s in td_subjects]
        
        # Get sample counts
        asd_samples = [subject_info[s]['n_samples'] for s in asd_subjects]
        td_samples = [subject_info[s]['n_samples'] for s in td_subjects]
        
        fold_compositions[fold_idx] = {
            'asd_subjects': asd_subjects,
            'td_subjects': td_subjects,
            'asd_ages': asd_ages,
            'td_ages': td_ages,
            'asd_samples': asd_samples,
            'td_samples': td_samples,
        }
        
        print(f"\n{'─' * 60}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'─' * 60}")
        print(f"ASD subjects ({len(asd_subjects)}): {sorted(asd_subjects)}")
        print(f"  Ages: {sorted(asd_ages)} (mean: {np.mean(asd_ages):.1f})")
        print(f"  Samples per subject: {asd_samples} (total: {sum(asd_samples)})")
        print(f"\nTD subjects ({len(td_subjects)}): {sorted(td_subjects)}")
        print(f"  Ages: {sorted(td_ages)} (mean: {np.mean(td_ages):.1f})")
        print(f"  Samples per subject: {td_samples} (total: {sum(td_samples)})")
        print(f"\nTotal test samples: {len(test_idx)} (ASD: {sum(test_labels)}, TD: {len(test_labels) - sum(test_labels)})")
    
    return fold_compositions


# =============================================================================
# IDENTIFY HARD-TO-CLASSIFY SUBJECTS
# =============================================================================

def identify_hard_subjects(X, y, groups, subjects, subject_info):
    """Identify which subjects are consistently misclassified."""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    selector = SelectKBest(mutual_info_classif, k=min(100, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    cv = GroupKFold(n_splits=5)
    
    # Track predictions for each subject
    subject_predictions = defaultdict(list)
    subject_true_labels = {}
    subject_fold = {}
    
    print("\n" + "=" * 80)
    print("SUBJECT-LEVEL CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Track per-subject
        for i, idx in enumerate(test_idx):
            subj = subjects[idx]
            subject_predictions[subj].append({
                'pred': y_pred[i],
                'prob': y_prob[i],
                'true': y_test[i],
                'fold': fold_idx + 1
            })
            subject_true_labels[subj] = y_test[i]
            subject_fold[subj] = fold_idx + 1
    
    # Analyze each subject
    subject_analysis = []
    
    for subj, preds in subject_predictions.items():
        true_label = subject_true_labels[subj]
        avg_prob = np.mean([p['prob'] for p in preds])
        correct_preds = sum(1 for p in preds if p['pred'] == true_label)
        total_preds = len(preds)
        accuracy = correct_preds / total_preds
        
        # Determine if this is a hard subject
        if true_label == 1:  # ASD
            is_hard = avg_prob < 0.6  # Should be > 0.5 for ASD
        else:  # TD
            is_hard = avg_prob > 0.4  # Should be < 0.5 for TD
        
        subject_analysis.append({
            'subject': subj,
            'true_label': 'ASD' if true_label == 1 else 'TD',
            'age': subject_info[subj]['age'],
            'n_samples': subject_info[subj]['n_samples'],
            'avg_prob': avg_prob,
            'sample_accuracy': accuracy,
            'fold': subject_fold[subj],
            'is_hard': is_hard,
            'predicted_as': 'ASD' if avg_prob > 0.5 else 'TD',
            'correct': (avg_prob > 0.5) == (true_label == 1)
        })
    
    # Sort by difficulty
    hard_subjects = sorted([s for s in subject_analysis if s['is_hard']], 
                          key=lambda x: abs(x['avg_prob'] - 0.5))
    
    easy_subjects = sorted([s for s in subject_analysis if not s['is_hard']], 
                          key=lambda x: abs(x['avg_prob'] - 0.5), reverse=True)
    
    # Print hard subjects
    print("\n🔴 HARD-TO-CLASSIFY SUBJECTS (likely misclassified):")
    print("-" * 90)
    print(f"{'Subject':<10} {'True':<6} {'Pred':<6} {'Prob':<8} {'Age':<6} {'Samples':<8} {'Fold':<6} {'Correct':<8}")
    print("-" * 90)
    
    for s in hard_subjects:
        correct_str = "✓" if s['correct'] else "✗"
        print(f"{s['subject']:<10} {s['true_label']:<6} {s['predicted_as']:<6} {s['avg_prob']:.3f}    "
              f"{s['age']:<6} {s['n_samples']:<8} {s['fold']:<6} {correct_str}")
    
    # Print easy subjects
    print("\n🟢 EASY-TO-CLASSIFY SUBJECTS (confident predictions):")
    print("-" * 90)
    print(f"{'Subject':<10} {'True':<6} {'Pred':<6} {'Prob':<8} {'Age':<6} {'Samples':<8} {'Fold':<6} {'Correct':<8}")
    print("-" * 90)
    
    for s in easy_subjects[:15]:  # Top 15 easiest
        correct_str = "✓" if s['correct'] else "✗"
        print(f"{s['subject']:<10} {s['true_label']:<6} {s['predicted_as']:<6} {s['avg_prob']:.3f}    "
              f"{s['age']:<6} {s['n_samples']:<8} {s['fold']:<6} {correct_str}")
    
    return subject_analysis, hard_subjects


# =============================================================================
# ANALYZE FOLD PERFORMANCE
# =============================================================================

def analyze_fold_performance(X, y, groups, subjects, subject_info, hard_subjects):
    """Analyze why specific folds are failing."""
    
    cv = GroupKFold(n_splits=5)
    
    hard_subject_names = set([s['subject'] for s in hard_subjects])
    
    print("\n" + "=" * 80)
    print("FOLD PERFORMANCE BREAKDOWN")
    print("=" * 80)
    
    # Scale and select features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(mutual_info_classif, k=min(100, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Get subjects in this fold
        test_subjects = [subjects[i] for i in test_idx]
        unique_subjects = list(set(test_subjects))
        
        # Count hard subjects in this fold
        hard_in_fold = [s for s in unique_subjects if s in hard_subject_names]
        
        # Subject-level accuracy
        subject_correct = 0
        subject_total = 0
        misclassified_subjects = []
        
        for subj in set(test_subjects):
            subj_indices = [i for i, s in enumerate(test_subjects) if s == subj]
            subj_preds = [y_pred[i] for i in subj_indices]
            subj_true = y_test[subj_indices[0]]
            
            # Majority vote
            final_pred = 1 if np.mean(subj_preds) > 0.5 else 0
            
            subject_total += 1
            if final_pred == subj_true:
                subject_correct += 1
            else:
                misclassified_subjects.append(subj)
        
        fold_acc = accuracy_score(y_test, y_pred)
        subj_acc = subject_correct / subject_total
        
        print(f"\n{'─' * 60}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'─' * 60}")
        print(f"Sample accuracy: {fold_acc:.4f}")
        print(f"Subject accuracy: {subj_acc:.4f} ({subject_correct}/{subject_total})")
        print(f"Hard subjects in fold: {len(hard_in_fold)}/{len(unique_subjects)} ({hard_in_fold})")
        print(f"Misclassified subjects: {misclassified_subjects}")
        
        # Age analysis
        test_ages = [subject_info[s]['age'] for s in unique_subjects]
        train_subjects = list(set([subjects[i] for i in train_idx]))
        train_ages = [subject_info[s]['age'] for s in train_subjects]
        
        print(f"\nAge distribution:")
        print(f"  Test ages: min={min(test_ages)}, max={max(test_ages)}, mean={np.mean(test_ages):.1f}")
        print(f"  Train ages: min={min(train_ages)}, max={max(train_ages)}, mean={np.mean(train_ages):.1f}")


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def generate_recommendations(subject_analysis, hard_subjects, fold_compositions):
    """Generate recommendations based on analysis."""
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Analyze hard subjects
    hard_asd = [s for s in hard_subjects if s['true_label'] == 'ASD']
    hard_td = [s for s in hard_subjects if s['true_label'] == 'TD']
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total hard subjects: {len(hard_subjects)}/62")
    print(f"  Hard ASD subjects: {len(hard_asd)} ({[s['subject'] for s in hard_asd]})")
    print(f"  Hard TD subjects: {len(hard_td)} ({[s['subject'] for s in hard_td]})")
    
    # Age analysis of hard subjects
    hard_ages = [s['age'] for s in hard_subjects]
    all_ages = [AGES[k] for k in AGES]
    
    print(f"\n📊 AGE ANALYSIS:")
    print(f"  Hard subjects avg age: {np.mean(hard_ages):.1f} months")
    print(f"  All subjects avg age: {np.mean(all_ages):.1f} months")
    
    # Sample count analysis
    hard_samples = [s['n_samples'] for s in hard_subjects]
    print(f"\n📊 SAMPLE COUNT:")
    print(f"  Hard subjects avg samples: {np.mean(hard_samples):.1f}")
    
    # Subjects with very few samples
    few_sample_subjects = [s for s in subject_analysis if s['n_samples'] <= 2]
    print(f"\n📊 LOW SAMPLE SUBJECTS (≤2 samples): {len(few_sample_subjects)}")
    for s in few_sample_subjects:
        print(f"  {s['subject']}: {s['n_samples']} samples, {'HARD' if s['is_hard'] else 'OK'}")
    
    print("\n" + "=" * 80)
    print("💡 ACTIONABLE RECOMMENDATIONS:")
    print("=" * 80)
    
    print("""
1. HANDLE HARD SUBJECTS:
   - Consider removing subjects with very low sample counts (1-2 samples)
   - Or: Use Leave-One-Subject-Out CV for more robust evaluation
   
2. AGE STRATIFICATION:
   - Try stratifying by age groups when splitting folds
   - Some age groups might have clearer ASD vs TD differences
   
3. DATA QUALITY CHECK:
   - Manually listen to samples from hard subjects
   - Check if recordings are high quality
   - Some might be mislabeled or ambiguous cases
   
4. SUBJECT-SPECIFIC AUGMENTATION:
   - Generate more augmented samples for subjects with few samples
   - This helps the model learn better representations
   
5. CUSTOM FOLD ASSIGNMENT:
   - Distribute hard subjects evenly across folds
   - Don't let them cluster in folds 3 and 4
   
6. THRESHOLD ADJUSTMENT:
   - For hard subjects, use confidence thresholds
   - Mark uncertain predictions as "needs review"
""")
    
    # Check fold 3 and 4 specifically
    print("\n" + "=" * 80)
    print("📍 FOLD 3 AND 4 SPECIFIC ANALYSIS:")
    print("=" * 80)
    
    fold3_hard = [s['subject'] for s in hard_subjects if s['fold'] == 3]
    fold4_hard = [s['subject'] for s in hard_subjects if s['fold'] == 4]
    other_folds_hard = [s['subject'] for s in hard_subjects if s['fold'] not in [3, 4]]
    
    print(f"\nHard subjects by fold:")
    print(f"  Fold 1: {len([s for s in hard_subjects if s['fold'] == 1])}")
    print(f"  Fold 2: {len([s for s in hard_subjects if s['fold'] == 2])}")
    print(f"  Fold 3: {len([s for s in hard_subjects if s['fold'] == 3])} ({fold3_hard})")
    print(f"  Fold 4: {len([s for s in hard_subjects if s['fold'] == 4])} ({fold4_hard})")
    print(f"  Fold 5: {len([s for s in hard_subjects if s['fold'] == 5])}")
    
    if len(fold3_hard) + len(fold4_hard) > len(other_folds_hard):
        print("\n⚠️  CONFIRMED: Folds 3 and 4 contain more hard subjects!")
        print("    This explains the performance drop.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS: Understanding Fold Performance")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load data
    X, y, groups, subjects, subject_info = load_data_detailed()
    
    print(f"\nLoaded {len(X)} samples from {len(subject_info)} subjects")
    print(f"ASD: {sum(y)} samples, TD: {len(y) - sum(y)} samples")
    
    # Analyze fold composition
    fold_compositions = analyze_folds(X, y, groups, subjects, subject_info)
    
    # Identify hard subjects
    subject_analysis, hard_subjects = identify_hard_subjects(X, y, groups, subjects, subject_info)
    
    # Analyze fold performance
    analyze_fold_performance(X, y, groups, subjects, subject_info, hard_subjects)
    
    # Generate recommendations
    generate_recommendations(subject_analysis, hard_subjects, fold_compositions)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return subject_analysis, hard_subjects


if __name__ == '__main__':
    subject_analysis, hard_subjects = main()