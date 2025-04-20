import librosa
import numpy as np

# Reconstruct waveform using Griffin-Lim
reconstructed = librosa.griffinlim(
    clean_mag,
    n_iter=32,
    n_fft=n_fft,
    hop_length=hop_length
)

# Save to file if you want
import soundfile as sf
sf.write('denoised_output.wav', reconstructed, sr)