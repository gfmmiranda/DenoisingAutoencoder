import os
import sys
import librosa
import numpy as np
from tqdm import tqdm
import torchaudio

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
TARGET_FRAMES = 1024          # you can tweak this as needed

SPLITS = ['train', 'validation', 'test']

def compute_complex_spectrogram(audio_path):
    y, _ = librosa.load(audio_path, sr=16000)
    D = librosa.stft(y, n_fft=512, hop_length=128, win_length=512)
    real = np.real(D)
    imag = np.imag(D)
    return np.stack([real, imag], axis=0).astype(np.float32)  # Shape: [2, F, T]

def compute_mag_spectrogram(audio_path):

    signal, sr = librosa.load(audio_path)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr = sr, target_sr=SAMPLE_RATE)
    
    # Compute Spectrogram
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mag = pad_or_crop_spectrogram(np.abs(D))

    return mag

def pad_or_crop_spectrogram(spec):
    freq_bins, time_frames = spec.shape

    # Pad frequency axis if needed
    if freq_bins < TARGET_BINS:
        pad_freq = TARGET_BINS - freq_bins
        spec = np.pad(spec, ((0, pad_freq), (0, 0)), mode='constant')
    elif freq_bins > TARGET_BINS:
        spec = spec[:TARGET_BINS, :]

    # Pad/crop time axis
    if time_frames < TARGET_FRAMES:
        pad_time = TARGET_FRAMES - time_frames
        spec = np.pad(spec, ((0, 0), (0, pad_time)), mode='constant')
    elif time_frames > TARGET_FRAMES:
        spec = spec[:, :TARGET_FRAMES]

    return spec

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
        clean_mag = compute_mag_spectrogram(clean_path)
        np.save(os.path.join(out_clean, f"{clean_file[:-4]}.npy"), clean_mag)

    
    for noisy_file in tqdm(os.listdir(noisy_folder), desc=f"[{split}] Processing noisy files"):
        if not noisy_file.endswith('.wav'):
            continue

        # Process noisy file
        noisy_path = os.path.join(noisy_folder, noisy_file)
        noisy_mag = compute_mag_spectrogram(noisy_path)
        np.save(os.path.join(out_noisy, f"{noisy_file[:-4]}.npy"), noisy_mag)


if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)