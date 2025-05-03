
# Denoising Autoencoder

This project focuses on denoising audio signals using deep learning techniques, particularly Convolutional Neural Networks (CNNs) and U-Net architectures implemented in PyTorch. It encompasses audio preprocessing, model experimentation, hyperparameter tuning, and experiment tracking.

## Overview

The primary goal is to develop models capable of removing noise from audio signals. The process involves:

- **Audio Preprocessing**: Transforming raw audio into spectrograms using Short-Time Fourier Transform (STFT) with parameter tuning.
- **Model Development**: Implementing and experimenting with CNNs and U-Nets for effective denoising.
- **Hyperparameter Optimization**: Utilizing Optuna for automated hyperparameter tuning.
- **Experiment Tracking**: Employing Weights & Biases (W&B) for logging and visualizing experiments.


## Dataset

This project uses a subset of the [MS-SNSD dataset](https://github.com/microsoft/MS-SNSD), a high-quality corpus for speech denoising that combines clean speech with a variety of real-world noise recordings and room responses.

- **Audio format:** WAV, mono, 16kHz sample rate
- **Clip duration:** Each audio file is 8 seconds long
- **Noise types:** Babble, music, television, and background distractions
- **SNR range:** From 0.1 dB to 0.5 dB, simulating challenging denoising conditions

### 🔹 Training Breakdown:
- **Clean speech (training):** 8 hours
- **Clean speech (validation):** 4 hours
- **Clean speech (test):** 4 hours
- Noisy versions were created by combining clean speech with room impulse responses and real-world noise sources

All files were preprocessed into spectrograms and organized under:
```
PREPROCESSED/
├── train/
├── validation/
└── test/
```

For training on the full set, `train+validation` can be merged for final evaluation.

## Key Learnings

### 1. Audio Preprocessing

- **Spectrogram Generation**: Learned to convert audio signals into spectrograms, facilitating visual and computational analysis.
- **STFT Parameterization**: Explored the impact of different STFT parameters (e.g., window size, hop length) on the quality and resolution of spectrograms.

### 2. Model Experimentation

- **Convolutional Neural Networks**: Implemented CNNs to capture local temporal and frequency features in spectrograms for denoising tasks.
- **U-Net Architecture**: Adapted the U-Net model, known for its success in image segmentation, to perform audio denoising by leveraging its encoder-decoder structure with skip connections.

### 3. Hyperparameter Tuning and Experiment Tracking

- **Optuna Integration**: Automated the process of hyperparameter tuning, leading to more efficient and effective model optimization.
- **Weights & Biases**: Tracked experiments, visualized training progress, and compared model performances seamlessly.

## Denoised Spectrogram Examples

Below are examples showcasing the denoising capabilities of the implemented models:

### Example 1

![Noisy Spectrogram](path_to_noisy_spectrogram_image)  
*Noisy Input*

![Denoised Spectrogram](path_to_denoised_spectrogram_image)  
*Denoised Output*

### Example 2

![Noisy Spectrogram](path_to_noisy_spectrogram_image_2)  
*Noisy Input*

![Denoised Spectrogram](path_to_denoised_spectrogram_image_2)  
*Denoised Output*

*Note: Replace `path_to_*` with the actual paths to your spectrogram images.*

## Project Structure

```
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── src/                   # Source code for models and utilities
├── tuning/                # Hyperparameter tuning scripts and configurations
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
```

## Libraries

- **Librosa**: For audio processing and feature extraction.
- **PyTorch**: For building and training deep learning models.
- **Optuna**: For efficient hyperparameter optimization.
- **Weights & Biases**: For experiment tracking and visualization.
