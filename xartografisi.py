import os
import pandas as pd
from scipy.io import wavfile

# Βάλε εδώ τον κύριο φάκελο που περιέχει τους υποφακέλους ASD, TD κλπ.
main_folder = r"C:\Users\User\Desktop\Tsakalidou Thesis\Tsakalidou Thesis\data\Raw_and_cleaned_cry_data\Sounds\original"

# Λίστα για να κρατάμε τα δεδομένα
data = []

# Διασχίζουμε όλους τους υποφακέλους
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    sample_rate, _ = wavfile.read(file_path)
                    # Διαρκεια σε δευτερόλεπτα
                    duration_sec = os.path.getsize(file_path) / (2 * sample_rate)  # αν stereo 16bit
                    data.append([folder_name, file_name, duration_sec])
                except Exception as e:
                    print(f"Error with {file_path}: {e}")

# Φτιάχνουμε DataFrame
df = pd.DataFrame(data, columns=['Folder', 'WAV File', 'Duration (sec)'])

# Αποθήκευση σε Excel
output_excel = os.path.join(main_folder, 'wav_durations.xlsx')
df.to_excel(output_excel, index=False)

print(f"Excel file created: {output_excel}")

