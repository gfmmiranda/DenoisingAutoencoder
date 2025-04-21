import os
import sys

sys.path.append('..')

import librosa
import numpy as np
from tqdm import tqdm
import torchaudio

from src.utils import compute_mag_spectrogram

# === Read Arguments ===
if len(sys.argv) != 6:
    print("Usage: python preprocess.py <DATASET_ROOT> <OUT_DIR> <N_FFT> <HOP_LENGTH> <WIN_LENGTH>")
    sys.exit(1)

DATASET_ROOT = sys.argv[1]
OUT_DIR = sys.argv[2]
N_FFT = int(sys.argv[3])
HOP_LENGTH = int(sys.argv[4])
WIN_LENGTH = int(sys.argv[5])

SAMPLE_RATE = 16000
TARGET_BINS = N_FFT // 2 + 1  # usually 257
TARGET_FRAMES = 1024          # 8s - you can tweak this as needed

SPLITS = ['train', 'validation', 'test']

def process_split(split):
    clean_folder = os.path.join(DATASET_ROOT, split, 'CleanSpeech')
    noisy_folder = os.path.join(DATASET_ROOT, split, 'NoisySpeech')
    
    out_clean = os.path.join(OUT_DIR, split, 'clean')
    out_noisy = os.path.join(OUT_DIR, split, 'noisy')
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    # Search all speaker directories
    for clean_file in tqdm(os.listdir(clean_folder), desc=f"[{split}] Processing clean files"):
        if not clean_file.endswith('.wav'):
            continue

        # Process clean file
        clean_path = os.path.join(clean_folder, clean_file)
        clean_mag = compute_mag_spectrogram(clean_path, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH)
        np.save(os.path.join(out_clean, f"{clean_file[:-4]}.npy"), clean_mag)

    
    for noisy_file in tqdm(os.listdir(noisy_folder), desc=f"[{split}] Processing noisy files"):
        if not noisy_file.endswith('.wav'):
            continue

        # Process noisy file
        noisy_path = os.path.join(noisy_folder, noisy_file)
        noisy_mag = compute_mag_spectrogram(noisy_path, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH)
        np.save(os.path.join(out_noisy, f"{noisy_file[:-4]}.npy"), noisy_mag)


if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)