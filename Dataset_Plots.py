import numpy as np
import os
import pandas as pd
import sklearn
import scipy.io.wavfile as sci_wav
from matplotlib import pyplot as plt
from mfcc_feature_extraction import extract_mfcc_feature
from mfcc_utility_functions import stft, signal_power_to_db
import wavio
import librosa
import librosa.display as libdisplay

def read_wav_files(wav_files):
    return sci_wav.read(wav_files) #[1]
    # return wavio.read(wav_files) #[1]

def visualize_spectrogram(y_spec: np.array, parameters: dict, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    libdisplay.specshow(y_spec, y_axis='linear', sr=parameters['fs'], cmap='autumn', x_axis='time', hop_length=parameters['window_step'])
    plt.savefig(f'{title}.png')
    # plt.show(block=False)
    plt.close()


def visualize_mfcc(y_spec: np.array, parameters: dict, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    libdisplay.specshow(y_spec, y_axis='frames', sr=parameters['fs'], x_axis='time', hop_length=parameters['window_step'])
    plt.colorbar()
    plt.ylabel('MFCC')
    plt.savefig(f'{title}.png')
    # plt.show(block=False)
    plt.close()

def get_spectrogram_of_signal(signal: np.array, parameters: dict, fs: int):
    """
    Get Spectrogram form librosa and algorithm implementation.

    :param signal: input signal
    :param parameters: Params used for Spectrogram extraction
    :return: Spectrogram from librosa and implementation
    """

    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']

    # Use librosa for spectrogram extraction
    signal_spec_lib = librosa.amplitude_to_db(np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=window_step,
                                                                  win_length=window_length)))

    frame_size = window_length / fs
    frame_step = window_step / fs
    # Use implemented function for spectrogram extraction
    power_spectrum_frames = np.abs(stft(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step))**2
    signal_spec = signal_power_to_db(power_spectrum_frames)

    return signal_spec_lib, signal_spec.T


def get_mfcc_of_signal(signal: np.array, parameters: dict, fs: int):
    """
    Get MFCC coefficient form librosa and algorithm implementation.

    :param signal: input signal
    :param parameters: Params used for MFCC extraction
    :return: MFCC from librosa and implementation
    """

    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']
    n_mels = 40
    n_mfcc = 13
    # Use librosa for MFCC extraction
    signal_mfcc_lib = librosa.feature.mfcc(sr=fs, y=signal, n_mfcc=n_mfcc, n_mels=n_mels, win_length=window_length,
                                           hop_length=window_step, lifter=22)[1:n_mfcc, :]

    frame_size = window_length / fs
    frame_step = window_step / fs
    # Use implemented function for MFCC extraction
    signal_mfcc = extract_mfcc_feature(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)

    return signal_mfcc_lib, signal_mfcc.T


if __name__ == '__main__':
    data_path = r'D:\Tsakalidou Thesis\data\Raw_and_cleaned_cry_data\Sounds\cleaned'
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

    # 11 to 24 Boys ASD
    # 25 to 31 Girls ASD

    # 11 to 24 Boys TD
    # 25 to 31 Girls TD

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
    cry_folders = os.listdir(data_path)
    n_fft = 512
    fix_len_s = 1.0  # 13.5 s
    fs = 44100
    fix_len = int(fix_len_s * fs)

    for cry_f_idx, cry_folder in enumerate(cry_folders):

        if 'ASD' in cry_folder:
            cry_files = os.listdir(os.path.join(data_path, cry_folder))
            for cry_asd_idx, cry_asd_file in enumerate(cry_files):
                cry_file_path = os.path.join(data_path, cry_folder, cry_asd_file)
                cry_wav_asd = read_wav_files(cry_file_path)
                fs_asd = cry_wav_asd.rate

                y = 1.0 * cry_wav_asd.data
                t = np.linspace(0, y.shape[0] / fs_asd, y.shape[0])

                params = {
                    'fs': fs_asd,
                    'n_fft': n_fft,
                    'window_length': n_fft,
                    'window_step': int(n_fft / 3),
                }

                plt.figure(figsize=(6, 4))
                plt.title(f"ASD sound audio wav {cry_asd_idx} {cry_f_idx}")
                plt.plot(t, y)
                plt.savefig(f'ASD sound audio wav {cry_asd_idx} {cry_f_idx}.png')
                # plt.show(block=False)
                plt.close()

                # show spectrogram
                y_spec_lib, y_spec = get_spectrogram_of_signal(y, params, fs_asd)
                visualize_spectrogram(y_spec, params, f"Spectrogram ASD cry wav {cry_asd_idx} fs {fs_asd} {cry_f_idx}")
                visualize_spectrogram(y_spec_lib, params, f"Librosa Spectrogram ASD cry wav {cry_asd_idx} fs {fs_asd} {cry_f_idx}")

                # show MFCC
                y_mffc_lib, y_mfcc = get_mfcc_of_signal(y, params, fs_asd)
                visualize_mfcc(y_mfcc, params, f"MFCC ASD cry wav {cry_asd_idx} fs {fs_asd} {cry_f_idx}")
                visualize_mfcc(y_mffc_lib, params, f"Librosa MFCC ASd cry wav {cry_asd_idx} fs {fs_asd} {cry_f_idx}")
                # plt.show()

                asd_data['ASD'].append(cry_file_path)
                asd_data['Wav_Files'].append(cry_wav_asd.data)
                asd_data['Age'].append(ages[cry_folder])
        else:
            cry_files = os.listdir(os.path.join(data_path, cry_folder))
            for cry_td_idx, cry_td_file in enumerate(cry_files):
                cry_file_path = os.path.join(data_path, cry_folder, cry_td_file)
                cry_wav_td = read_wav_files(cry_file_path)
                fs_td = cry_wav_td.rate

                y = 1.0 * cry_wav_td.data
                t = np.linspace(0, y.shape[0] / fs_td, y.shape[0])

                params = {
                    'fs': fs_td,
                    'n_fft': n_fft,
                    'window_length': n_fft,
                    'window_step': int(n_fft / 3),
                }

                plt.figure(figsize=(6, 4))
                plt.title(f"TD sound audio wav {cry_td_idx}")
                plt.plot(t, y)
                plt.savefig(f'TD sound audio wav {cry_td_idx}.png')
                # plt.show(block=False)
                plt.close()

                # show spectrogram
                y_spec_lib, y_spec = get_spectrogram_of_signal(y, params, fs_td)
                visualize_spectrogram(y_spec, params, f"Spectrogram TD cry wav {cry_td_idx} fs {fs_td} {cry_f_idx}")
                visualize_spectrogram(y_spec_lib, params, f"Librosa Spectrogram TD cry wav {cry_td_idx} fs {fs_td} {cry_f_idx}")

                # show MFCC
                y_mffc_lib, y_mfcc = get_mfcc_of_signal(y, params, fs_td)
                visualize_mfcc(y_mfcc, params, f"MFCC TD cry wav {cry_td_idx} fs {fs_td} {cry_f_idx}")
                visualize_mfcc(y_mffc_lib, params, f"Librosa MFCC TD cry wav {cry_td_idx} fs {fs_td} {cry_f_idx}")
                # plt.show()

                td_data['TD'].append(cry_file_path)
                td_data['Wav_Files'].append(cry_wav_td.data)
                td_data['Age'].append(ages[cry_folder])

    print()


