# Speech Denoising with a Convolutional Autoencoder

An exploration of **sound representations, signal processing and deep learning for audio**, implemented in PyTorch. The project transforms noisy speech into magnitude spectrograms, learns to estimate their clean counterparts with a U-Net, and reconstructs audio using Griffin–Lim phase estimation.

On the evaluated test split, the final model reduced mean magnitude-spectrogram error by **75.64%** relative to the noisy input. [Notebook 04](notebooks/04_Evaluate.ipynb) contains the quantitative results, spectrogram comparisons and listening examples.

## Motivation

I built this project to connect my experience in data science with a deeper interest in sound and audio processing. Speech denoising offered a practical way to investigate the full path from a waveform to a learned representation and back to audible sound.

The central questions were:

- How do sampling, windowing and STFT parameters affect the representation available to a model?
- How can convolutional networks learn useful structure in time-frequency data?
- What is preserved or lost when modelling magnitude without phase?
- How well does a numerical reconstruction objective reflect what we hear?

The goal was to understand and implement this workflow, rather than establish a state-of-the-art speech-enhancement benchmark.

## From Waveform to Reconstructed Audio

| Stage | Approach |
|---|---|
| Audio representation | Mono speech at 16 kHz; short-time Fourier transform with a Hann window |
| Model input | Linear-magnitude spectrogram, padded or cropped to 257 × 1,024 bins/frames |
| Supervised learning | Noisy magnitude as input; paired clean magnitude as target |
| Prediction | U-Net estimates the clean magnitude spectrogram |
| Reconstruction | Griffin–Lim estimates phase and produces an audio waveform |

The STFT uses a **512-sample window** (32 ms) and **128-sample hop** (8 ms), with 75% overlap and 31.25 Hz frequency-bin spacing. These settings balance temporal localization, frequency detail and computational cost. Fixed-size inputs cover approximately eight seconds; the final training/evaluation workflow takes matching leading frames from clean and noisy recordings and pads shorter examples on the right.

The network operates on linear magnitudes. Decibel conversion is used for visualization, not as the training representation. Discarding phase simplifies the prediction target, but makes waveform reconstruction a separate estimation problem.

## Dataset

The experiments use paired clean/noisy speech from [**MS-SNSD (Microsoft Scalable Noisy Speech Dataset)**](https://github.com/microsoft/MS-SNSD), organized into training, validation and test folders. Magnitude spectrograms are saved as NumPy arrays and loaded in batches through a PyTorch dataset.

Initial exploration used **VOiCES**. Its far-field recording conditions brought reverberation and room effects into the problem, motivating a move to SNSD for a more focused investigation of additive-noise denoising. The VOiCES preprocessing code remains as part of that exploratory history; subsequent model experiments use SNSD.

### Generating a Dataset with the Same Settings

Clone the official dataset repository:

```bash
git clone https://github.com/microsoft/MS-SNSD.git
cd MS-SNSD
```

Follow the [MS-SNSD setup instructions](https://github.com/microsoft/MS-SNSD#usage) for dependencies and source speech/noise recordings. Set `noisyspeech_synthesizer.cfg` to the configuration used for this project:

```ini
[noisy_speech]

sampling_rate: 16000
audioformat: *.wav
audio_length: 10
silence_length: 0.2
total_hours: 8
snr_lower: 10
snr_upper: 40
total_snrlevels: 5

noise_dir: None
speech_dir: None
noise_types_excluded: None
```

This requests 8 hours of generated data, a minimum clip duration of 10 seconds, 0.2-second silences between concatenated speech utterances, and five SNR levels spanning 10–40 dB. Set `noise_dir` and `speech_dir` to local paths if your source files are outside the generator's default locations.

With the configuration file beside the script, run:

```bash
python noisyspeech_synthesizer.py
```

For remaining generation instructions and source-data details, refer to the [official repository](https://github.com/microsoft/MS-SNSD).

### Preparing the Project Splits

Organize generated clean/noisy pairs into the directories expected by this project's preprocessing script:

| Split | Clean recordings | Noisy recordings |
|---|---|---|
| Training | `raw/train/CleanSpeech/` | `raw/train/NoisySpeech/` |
| Validation | `raw/validation/CleanSpeech/` | `raw/validation/NoisySpeech/` |
| Test | `raw/test/CleanSpeech/` | `raw/test/NoisySpeech/` |

Keep each clean recording and all its noisy variants in the same split. Preserve the generated filename relationships so the dataset loader can match pairs. For speaker-independent evaluation, partition source speakers before synthesis.

These settings reproduce the **generation recipe**, not necessarily the exact recordings or reported scores. Exact reproduction additionally requires the same source-file versions, random choices and split membership; the configuration does not define the project's three-way split.

The generated recordings have a minimum length of **10 seconds**, while the model consumes **1,024 STFT frames**, approximately **8.2 seconds**. Preprocessing saves magnitude arrays; the final dataset loader selects matching leading frames from each pair and pads shorter inputs as needed.

Dataset reference: Chandan K. A. Reddy et al., *A Scalable Noisy Speech Dataset and Online Subjective Test Framework*, Interspeech 2019, pp. 1816–1820. The [MS-SNSD repository](https://github.com/microsoft/MS-SNSD#please-cite-us-if-you-use-this-dataset) provides the recommended citation.

## Model Development and Training

Development began with a plain convolutional encoder-decoder and progressed to a **U-Net with skip connections**. The encoder learns broader context at progressively lower resolution, while skip connections provide the decoder with finer time-frequency features that would otherwise have to pass through the bottleneck.

Optuna explored model width, depth, dropout, learning rate and batch size. The search also exposed practical constraints on an 8 GB GPU: the largest attempted configuration exceeded available memory. W&B integration supports experiment tracking.

The selected configuration was:

| Parameter | Value |
|---|---:|
| Base filters | 32 |
| Encoder/decoder depth | 4 |
| Encoder channels | 32 → 64 → 128 → 256 |
| Bottleneck channels | 512 |
| Dropout | ≈ 0.00456 |
| Initial learning rate | ≈ 0.0004014 |
| Batch size | 8 |

Training uses **magnitude MSE**, Adam, mixed precision, `ReduceLROnPlateau` and validation-based early stopping. The final run trains only on the training split, uses validation loss for scheduling and checkpoint selection, and reloads the best checkpoint before test evaluation.

## Results

| Test-set comparison | Mean magnitude MSE |
|---|---:|
| Noisy input vs. clean target | 0.054358 |
| U-Net prediction vs. clean target | 0.013243 |
| Relative error reduction | **75.64%** |

Final training stopped after **16 epochs**. The lowest validation loss, **0.012585**, occurred at epoch 11. Test MSE averages individual paired-excerpt errors and includes padded bins.

The listening examples were judged to give good reconstruction for door-slam noise and chatter, with a less successful result under loud chatter. These are illustrative observations, not category-level benchmarks. All three audio players in the evaluation notebook—including the clean reference—use Griffin–Lim reconstruction.

The numerical result demonstrates improved **magnitude reconstruction**. It does not establish a corresponding percentage improvement in perceived quality or intelligibility.

## Notebook Guide

| Notebook | Purpose |
|---|---|
| [01 — Preprocessing](notebooks/01_Preprocessing.ipynb) | Fourier/STFT foundations, windowing, representation choices, dataset exploration and spectrogram preparation |
| [02 — Training](notebooks/02_Training.ipynb) | Convolutional autoencoder development, U-Net architecture and initial training |
| [03 — Hyperparameter Tuning](notebooks/03_Hyperparameter_Tuning.ipynb) | Optuna search, GPU-memory constraints and configuration selection |
| [04 — Evaluation](notebooks/04_Evaluate.ipynb) | Final training, test-set comparison, spectrograms and reconstructed audio |
| [05 — Presentation](notebooks/05_Presentation.ipynb) | Auxiliary illustrations and a clean-audio reconstruction demonstration |

For a quick overview, start with **Notebook 04**. For the reasoning behind the approach, follow **01–03**. Notebook 05 is an appendix rather than another model experiment.

Reusable code lives in `src/`: model definitions in `src/models/`, data preparation/loading in `src/dataprocess/`, the training loop in `src/training.py`, and audio/visualization helpers in `src/utils.py`. Notebook 04 reads the exported trial table from `tuning/optuna_trials.csv`.

## Running the Project

Clone the repository and install the dependencies in a dedicated Python environment:

```bash
git clone https://github.com/gfmmiranda/DenoisingAutoencoder.git
cd DenoisingAutoencoder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torchsummary optuna wandb joblib ipykernel
```

Use that environment as your notebook kernel. A CUDA-capable GPU is recommended for training; configure the PyTorch installation for your hardware.

1. Generate SNSD pairs using the [configuration above](#generating-a-dataset-with-the-same-settings), organize the split folders, and run Notebook 01’s preprocessing steps.
2. Update the local dataset paths in the notebooks. They currently reflect the development machine.
3. Run Notebook 02 for the initial model experiment and Notebook 03 to reproduce the tuning workflow. Configure W&B if running its tracking cells.
4. Place the exported trial CSV at `tuning/optuna_trials.csv`, as expected by Notebook 04. Notebook 03's export location may need adjusting.
5. Run Notebook 04 to retrain the selected configuration, reload its best checkpoint and evaluate. To skip retraining, supply a compatible saved checkpoint and run the configuration/loading and evaluation cells.

Local datasets and checkpoint files must be available to execute these steps; saved notebook outputs can be read without rerunning training.

## What This Project Taught Me

**Representation choices shape the learning problem.** Sampling rate, window duration and hop size determine what time-frequency structure is available to the network and how much computation it requires.

**Architecture connects context with detail.** The encoder-decoder path builds context through downsampling, while U-Net skips give reconstruction access to features at finer resolutions.

**Denoising and reconstruction are separate sources of error.** Even an unchanged clean magnitude must undergo phase estimation before it becomes audio again. Listening to that reconstruction helps distinguish representation limits from model behaviour.

**Evaluation needs both numbers and listening.** MSE makes training and baseline comparisons straightforward, but spectrogram inspection and audio playback reveal effects that a single scalar cannot describe.

## Limitations

The model predicts magnitude only, and Griffin–Lim can introduce audible artifacts. Evaluation is limited to fixed-length SNSD excerpts and does not establish generalization to new speakers, recording environments or noise distributions. Perceptual metrics and intelligibility tests are not included.

Potential extensions include phase-aware modelling, reconstruction using the noisy phase, perceptual evaluation, testing on new noise conditions or allowing the model to process variable length inputs. They are outside the scope of this learning project.
