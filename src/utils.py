import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from torch.cuda.amp import autocast

# Complex STFT-based waveform reconstruction
def reconstruct_waveform(complex_spec, hop_length=128, win_length=512):
    real = complex_spec[0]
    imag = complex_spec[1]
    D = real + 1j * imag
    return librosa.istft(D, hop_length=hop_length, win_length=win_length)

# Plot clean + noisy complex spectrogram magnitudes side by side
def plot_spectrograms(clean_spec, noisy_spec, sr=16000, hop_length=128):
    clean_mag = np.sqrt(clean_spec[0]**2 + clean_spec[1]**2)
    noisy_mag = np.sqrt(noisy_spec[0]**2 + noisy_spec[1]**2)

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

# Play denoised sample using complex STFT output
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
    noisy, clean = dataset[index]  # [2, F, T]
    noisy_tensor = noisy.unsqueeze(0).to(device)  # [1, 2, F, T]

    # Mixed precision inference
    with torch.no_grad():
        with autocast():
            denoised = model(noisy_tensor).squeeze(0).cpu().numpy()  # [2, F, T]

    # Reconstruct waveforms
    noisy_audio = reconstruct_waveform(noisy.numpy(), hop_length, win_length)
    denoised_audio = reconstruct_waveform(denoised, hop_length, win_length)
    clean_audio = reconstruct_waveform(clean.numpy(), hop_length, win_length)

    # Playback
    print("🔊 Noisy")
    ipd.display(ipd.Audio(noisy_audio, rate=sample_rate))
    print("🔊 Denoised (model output)")
    ipd.display(ipd.Audio(denoised_audio, rate=sample_rate))
    print("🔊 Clean (reference)")
    ipd.display(ipd.Audio(clean_audio, rate=sample_rate))
