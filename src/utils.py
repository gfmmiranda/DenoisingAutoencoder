import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import IPython.display as ipd
from torch.cuda.amp import autocast
import soundfile as sf

# Plot clean + noisy spectrograms side by side
def plot_spectrograms(clean_mag, noisy_mag, sr=16000, hop_length=128):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(clean_mag, ref=np.max), 
                             sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time')
    plt.title('Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(noisy_mag, ref=np.max), 
                             sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time')
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

# Helper function to listen to model outputs
def play_denoised_sample(
    model, 
    dataset, 
    index=[0], 
    n_fft=512, 
    hop_length=128, 
    win_length=512,
    sample_rate=16000,
    epoch=''
):
    device = next(model.parameters()).device
    model.eval()

    print(len(dataset), "samples in the dataset.")

    for idx in index:
        print(f"🎧 Playing sample index: {idx}")
        # Load sample
        noisy, clean = dataset[idx]  # [F, T]
        noisy_tensor = noisy.unsqueeze(0).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            with autocast():
                denoised = model(noisy_tensor).squeeze().cpu().numpy()

        # Reconstruct waveforms
        noisy_audio = librosa.griffinlim(noisy.numpy(), n_iter=64, hop_length=hop_length, win_length=win_length)
        denoised_audio = librosa.griffinlim(denoised, n_iter=64, hop_length=hop_length, win_length=win_length)
        clean_audio = librosa.griffinlim(clean.numpy(), n_iter=64, hop_length=hop_length, win_length=win_length)

        # Spectrogram to dB
        def to_db(x):
            return librosa.amplitude_to_db(np.maximum(x, 1e-5), ref=np.max)

        noisy_db = to_db(noisy.numpy())
        denoised_db = to_db(denoised)
        clean_db = to_db(clean.numpy())
        diff_db = clean_db - denoised_db

        # Plot
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

        librosa.display.specshow(noisy_db, sr=sample_rate, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[0])
        axs[0].set_title('Noisy')

        librosa.display.specshow(denoised_db, sr=sample_rate, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[1])
        axs[1].set_title('Denoised')

        librosa.display.specshow(clean_db, sr=sample_rate, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[2])
        axs[2].set_title('Clean')

        librosa.display.specshow(diff_db, sr=sample_rate, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[3], cmap='coolwarm')
        axs[3].set_title('Clean - Denoised (dB)')

        for ax in axs:
            ax.label_outer()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"spectrogram_epoch{epoch}_sample{idx}.png", dpi=600)

        # Playback
        print("🔊 Noisy")
        ipd.display(ipd.Audio(noisy_audio, rate=sample_rate))
        print("🔊 Denoised (model output)")
        ipd.display(ipd.Audio(denoised_audio, rate=sample_rate))
        print("🔊 Clean (reference)")
        ipd.display(ipd.Audio(clean_audio, rate=sample_rate))

        sf.write(f"denoised_epoch{epoch}_sample{idx}.wav", denoised_audio, sample_rate)


# Function to compute magnitude spectrogram
def compute_mag_spectrogram(audio_path, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, TARGET_BINS=None, TARGET_FRAMES=None):

    signal, sr = librosa.load(audio_path)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr = sr, target_sr=SAMPLE_RATE)
    
    # Compute Spectrogram
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mag = np.abs(D)
    
    if TARGET_BINS  and TARGET_FRAMES :
        mag = pad_or_crop_spectrogram(mag, TARGET_BINS, TARGET_FRAMES)

    return mag

def pad_or_crop_spectrogram(spec, TARGET_BINS, TARGET_FRAMES):
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