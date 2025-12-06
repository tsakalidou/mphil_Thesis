import os

import librosa
import numpy as np
import scipy.io.wavfile as sci_wav
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mfcc_feature_extraction import extract_mfcc_feature

# φορτώνει τα αρχεία ήχου για δύο κατηγορίες

def read_wav_files(wav_files):
    return sci_wav.read(wav_files) #[1]
    # return wavio.read(wav_files) #[1]

# τα διαμορφώνει ώστε όλα τα samples να έχουν το ίδιο μήκος ανεξάρτητα απο το πόσο διαρκούν, αυτό είναι απαραίτητο
# διότι οι ταξινομητές της machine learning απαιτούν όλα τα δεδομένα να έχουν το ίδιο αριθμό χαρακτηριστικών.
# Ο υπολογισμός MFCC δίνει σταθερό αριθμό coefficients ανα frame αλλά διαφορετικό αριθμό frames όταν αλλάζει το μήκος του ήχου

def pad_data(data_set: list, fix_length: int):
    """
    Pad each sample from data set to fix length

    :param data_set: Input data
    :param fix_length: Fix length in number of samples to pad data
    :return: Data set with fix length
    """

    if not isinstance(data_set, list):
        return librosa.util.fix_length(data_set, size=fix_length, axis=0, mode='wrap')

    else:
        data_set_fix_length = np.zeros((len(data_set), fix_length))
        for i, data in enumerate(data_set):
            data_set_fix_length[i, :] = librosa.util.fix_length(data, size=fix_length, axis=0, mode='wrap')
    return data_set_fix_length

 # υπολογίζει τα MFCC Feature απο κάθε δείγμα ήχου.

def mfcc_extraction(data_set_fix_length: list, fs: float, n_fft: int, frame_size: float, frame_step: float):
    """
    Extract MFCC features for each data sample.

    :param data_set_fix_length: Input data set
    :param fs: Sampling frequency
    :param n_fft: Num of Nfft points
    :param frame_size: Size of frame in sec
    :param frame_step: Frame step in sec
    :return: MFCC features for each data sample
    """

    flag = True
    for i, data in enumerate(data_set_fix_length):
        mfcc = extract_mfcc_feature(y=data, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)
        if flag:
            mfcc_features = np.zeros((len(data_set_fix_length), mfcc.shape[0], mfcc.shape[1]))
            flag = False
        mfcc_features[i, :, :] = mfcc
    return mfcc_features

# flatten αυτών των χαρακτηριστικών ώστε κάθε δείγμα να γίνει ένα μεγάλο διάνυσμα. ΤΙ ΚΑΝΕΙ: Για κάθε δείγμα που είναι
# ήδη σε ίδιο μήκος υπολογίζει MFCC χαρακτηριστικά
# Διασπά το σήμα σε frames, υπολογίζει το φάσμα FFT, Υπολογίζει Mel-Filterbanks, βγάζει MFCC coefficients --> Έτσι επιστρέφει ένα πίνακα 3 διαστάσεων

def preprocess_raw_data(data: list, fix_length: int, mfcc_parameters: dict):
    """
    Preprocess audio data. Returns MFCC features of each audio sample.

    :param data: List of audio data
    :param fix_length: Fix length to pad each audio sample
    :param mfcc_parameters: Parameters for MFCC extraction
    :return: MFCC data for each audio sample
    """

    # pad data to fix length
    data_set_fix_length = pad_data(data, fix_length)
    # extract MFCC for each sample
    mfcc_features = mfcc_extraction(data_set_fix_length, mfcc_parameters['fs'], mfcc_parameters['n_fft'],
                                    mfcc_parameters['frame_size'], mfcc_parameters['frame_step'])

    return mfcc_features

if __name__ == '__main__':
    data_path = r'C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
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
        'TD14': 20, 'TD15': 21, 'TD16': 24, 'TD17': 24, 'TD18':24, 'TD19': 24,
        'TD20': 24, 'TD21': 29, 'TD22': 30, 'TD23': 30, 'TD24': 43, 'TD25': 24,
        'TD26': 25, 'TD27': 29, 'TD28': 33, 'TD29': 45, 'TD30': 50, 'TD31': 51
    }

    asd_data = {
        'ASD': [],
        'Wav_Files': [],
        'Age': []
    }

    td_data = {
        'TD': [],
        'Wav_Files': [],
        'Age': []
    }
    n_fft = 512
    mfcc_parameters = {
        'fs': 44100,
        'n_fft': n_fft,
        'frame_size': n_fft / 44100,
        'frame_step': int(n_fft / 3) / 44100,
    }

    # Pick the fixed length in seconds\
    fix_len_s = 1.0  # 13.5 s
    fix_len = int(fix_len_s * 44100)

    cry_folders = os.listdir(data_path)
    for cry_f_idx, cry_folder in enumerate(cry_folders):

        if 'ASD' in cry_folder:
            cry_files = os.listdir(os.path.join(data_path, cry_folder))
            for cry_asd_idx, cry_asd_file in enumerate(cry_files):
                cry_file_path = os.path.join(data_path, cry_folder, cry_asd_file)
                fs_asd, cry_wav_asd = read_wav_files(cry_file_path)

                asd_data['ASD'].append(cry_file_path)
                asd_data['Wav_Files'].append(cry_wav_asd)
                asd_data['Age'].append(ages[cry_folder])
        else:
            cry_files = os.listdir(os.path.join(data_path, cry_folder))
            for cry_td_idx, cry_td_file in enumerate(cry_files):
                cry_file_path = os.path.join(data_path, cry_folder, cry_td_file)
                fs_td, cry_wav_td = read_wav_files(cry_file_path)


                td_data['TD'].append(cry_file_path)
                td_data['Wav_Files'].append(cry_wav_td)
                td_data['Age'].append(ages[cry_folder])

    X_ASD = preprocess_raw_data(asd_data['Wav_Files'], fix_len, mfcc_parameters)
    X_ASD = np.array([feature.ravel() for feature in X_ASD])

    X_TD = preprocess_raw_data(td_data['Wav_Files'], fix_len, mfcc_parameters)
    X_TD = np.array([feature.ravel() for feature in X_TD])

    # Create feature matrix (X) and labels (y)
    X = np.vstack((X_ASD, X_TD))
    y = np.hstack((np.ones(X_ASD.shape[0]), np.zeros(X_TD.shape[0])))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(                # Διαχωρισμός train/test
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )




    estimator = RandomForestClassifier()

    # Perform RFE   Γίνεται επιλογή χαρακτηριστικών με RFE (Recursive Feature Elimination) χρησιμοποιόντας Random Forest
    n_features_to_select = 50  # desired number of features
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Reduce dataset to selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    svc_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svc_model.fit(X_train_rfe, y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_rfe, y_train)

    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train_rfe, y_train)

    svc_y_pred = svc_model.predict(X_test_rfe)
    rf_y_pred = rf_model.predict(X_test_rfe)
    gb_y_pred = gb_model.predict(X_test_rfe)

    print(classification_report(y_test, svc_y_pred))
    print(f' SVC Accuracy: {accuracy_score(y_test, svc_y_pred)}')

    print(classification_report(y_test, rf_y_pred))
    print(f' RF Accuracy: {accuracy_score(y_test, rf_y_pred)}')

    print(classification_report(y_test, gb_y_pred))
    print(f' GB Accuracy: {accuracy_score(y_test, gb_y_pred)}')
    print()