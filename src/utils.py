import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from torch.cuda.amp import autocast


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
        index=0, 
        n_fft=512, 
        hop_length=128, 
        win_length=512,
        sample_rate=16000,
    ):
    device = next(model.parameters()).device  # Get model's current device
    model.eval()

    # Load sample
    noisy, clean = dataset[index]  # [F, T]
    noisy_tensor = noisy.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, F, T]

    # Mixed precision inference
    with torch.no_grad():
        with autocast():
            denoised = model(noisy_tensor).squeeze().cpu().numpy()

    # Convert to waveform using Griffin-Lim
    noisy_audio = librosa.griffinlim(noisy.numpy(),  n_iter = 64, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    denoised_audio = librosa.griffinlim(denoised, n_iter = 64, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_audio = librosa.griffinlim(clean.numpy(), n_iter = 64, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Playback
    print("🔊 Noisy")
    ipd.display(ipd.Audio(noisy_audio, rate=sample_rate))
    print("🔊 Denoised (model output)")
    ipd.display(ipd.Audio(denoised_audio, rate=sample_rate))
    print("🔊 Clean (reference)")
    ipd.display(ipd.Audio(clean_audio, rate=sample_rate))