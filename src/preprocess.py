import os
import sys
import librosa
import numpy as np
from tqdm import tqdm

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
TARGET_BINS = N_FFT // 2 + 1
TARGET_FRAMES = 1024

SPLITS = ['train', 'test']
NOISE_TYPES = ['babb', 'musi', 'tele', 'none']
ROOMS = ['rm1', 'rm2', 'rm3', 'rm4']
MICS = ['mc01-stu-clo', 'mc05-stu-far']
DG = ['dg150', 'dg40']

def compute_complex_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    real = np.real(D)
    imag = np.imag(D)
    spec = np.stack([real, imag], axis=0)  # [2, F, T]
    return pad_or_crop_complex(spec)

def pad_or_crop_complex(spec):
    _, freq_bins, time_frames = spec.shape

    if freq_bins < TARGET_BINS:
        pad_freq = TARGET_BINS - freq_bins
        spec = np.pad(spec, ((0, 0), (0, pad_freq), (0, 0)), mode='constant')
    elif freq_bins > TARGET_BINS:
        spec = spec[:, :TARGET_BINS, :]

    if time_frames < TARGET_FRAMES:
        pad_time = TARGET_FRAMES - time_frames
        spec = np.pad(spec, ((0, 0), (0, 0), (0, pad_time)), mode='constant')
    elif time_frames > TARGET_FRAMES:
        spec = spec[:, :, :TARGET_FRAMES]

    return spec

def process_split(split):
    clean_root = os.path.join(DATASET_ROOT, 'source-16k', split)
    noisy_root = os.path.join(DATASET_ROOT, 'distant-16k', 'speech', split)
    out_clean = os.path.join(OUT_DIR, split, 'clean')
    out_noisy = os.path.join(OUT_DIR, split, 'noisy')
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    for speaker in tqdm(os.listdir(clean_root), desc=f"[{split}] Processing speakers"):
        speaker_dir = os.path.join(clean_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        for file in os.listdir(speaker_dir):
            if not file.endswith('.wav'):
                continue

            clean_path = os.path.join(speaker_dir, file)
            base_name = f"{speaker}-{file[-19:-4]}"
            clean_spec = compute_complex_spectrogram(clean_path)
            np.save(os.path.join(out_clean, f"{base_name}.npy"), clean_spec)

            for room in ROOMS:
                for noise in NOISE_TYPES:
                    noisy_dir = os.path.join(noisy_root, room, noise, speaker)
                    if not os.path.isdir(noisy_dir):
                        continue

                    for mic in MICS:
                        for dg in DG:
                            noisy_file = f"{file[:16]}-{room}-{noise}-{speaker}-{file[-19:-4]}-{mic}-{dg}.wav"
                            if noisy_file in os.listdir(noisy_dir):
                                noisy_path = os.path.join(noisy_dir, noisy_file)
                                noisy_spec = compute_complex_spectrogram(noisy_path)
                                noisy_name = f"{base_name}__{room}-{noise}-{mic}-{dg}.npy"
                                np.save(os.path.join(out_noisy, noisy_name), noisy_spec)

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)