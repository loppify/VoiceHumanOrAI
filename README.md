# Voice I/O Lab

An academic audio-analysis application for exploring differences between recorded human speech and synthesized speech.

The repository is named `VoiceHumanOrAI`. Its Dash interface is called **Voice I/O Lab**. It combines audio visualization, a Random Forest classifier, an experimental signal-analysis method, and tools for preparing paired human/TTS samples.

This is a personal academic project by [Rostyslav Tarasov](https://github.com/loppify), developed with substantial AI assistance. It represents experimentation with audio processing and ML integration; it does not claim a reliable general-purpose deepfake detector.

[Demo video](demo/demo.mp4) · [Extended demo](demo/demo-full.mp4) · [Presentation](https://www.canva.com/design/DAHKv5u8qto/7ZaNPKdI9qOb6q9vSs2AvQ/view)

## Features

- Upload a WAV file or select a sample from a local dataset.
- Inspect waveforms and spectrograms in a Dash/Plotly interface.
- Extract MFCC, chroma, and spectral-rolloff features with librosa.
- Train and run a Random Forest classifier.
- Display classifier probabilities and feature importance.
- Explore the separate experimental analysis implemented in `src/bionic_core.py`.
- Build paired human and synthesized samples using a Hugging Face dataset and a TTS provider.
- Run a comparison script and generate an experiment report.

## Methods

The ML classifier uses 33 features: 20 mean MFCC values, 12 mean chroma values, and one mean spectral-rolloff value. Features pass through a fitted scaler before Random Forest inference.

The second analysis path is implemented separately in `src/bionic_core.py`. It explores signal measurements and a coordinate/topological representation. Its output is experimental; the presence of two methods does not establish that their combination improves detection.

## Install and launch

Requirements: Python compatible with the lockfile and the declared `>=3.11,<3.15` range, Poetry, and FFmpeg available on `PATH`. Python 3.12 is a reasonable starting point for the declared scientific dependencies. Dependency requirements may narrow the top-level Python range.

```bash
git clone https://github.com/loppify/VoiceHumanOrAI.git
cd VoiceHumanOrAI
poetry install
poetry run python src/app.py
```

Open [the local interface](http://127.0.0.1:8050/). Run commands from the repository root so dataset and model paths resolve correctly.

The application currently starts with Dash debug mode enabled. Use it locally for experimentation.

## Prepare a small dataset

Create `.env` from the supplied example if the selected dataset requires Hugging Face authentication:

```bash
cp .env.example .env
```

The variable name used by the application is case-sensitive:

```dotenv
HuggingFace_TOKEN=replace_with_your_token
```

Start with the Edge TTS provider, which does not require the local LuxTTS or MOSS servers:

```bash
poetry run python dataset_builder.py \
  --samples 10 \
  --provider edge \
  --dataset PolyAI/minds14 \
  --config en-US \
  --split train
```

The builder writes WAV samples under `dataset/human/` and `dataset/ai/`. `--samples` requests a number of pairs; successful completion depends on dataset access, compatible audio/text fields, and the TTS service. Ten pairs are a workflow check, not a meaningful benchmark.

If no usable samples are produced, inspect the logs and stop the process rather than assuming it is progressing; the builder retries failed attempts.

### Other TTS providers

The CLI also exposes `lux`, `mosstts`, and `voicebox`. Their integrations depend on external repositories or running services.

The repository contains `.gitmodules` declarations. Check whether your checkout actually contains the required external sources. Where submodule entries are available, initialize the required provider:

```bash
git submodule update --init --recursive
```

If this does not populate the provider directory, follow the corresponding upstream repository's installation instructions and place it at the path expected by `dataset_builder.py`. Upstream links are listed in `.gitmodules`.

Local server helpers can create environments, install packages, and download model resources. They currently assume Unix-style `venv/bin/python` paths. Their dependencies and hardware requirements are separate from the main application's environment.

## Train the classifier

After collecting both human and AI samples:

```bash
poetry run python src/train_model.py
```

This updates `models/rf_model.pkl` and `models/rf_scaler.pkl`. Preserve existing model files separately if you want to compare versions.

The script attempts five-fold cross-validation using a scaler/model pipeline. It then trains a final model on all available samples and prints a classification report on those same training samples. That final report is a training diagnostic, not a held-out test result.

## Evaluation boundaries

- No benchmark accuracy is claimed in this README.
- Current evaluation does not enforce separation by speaker, recording source, or TTS provider.
- Source-specific recording conditions can influence classification.
- Random Forest probabilities are model outputs, not independently calibrated guarantees of authenticity.
- Stronger evaluation would use independent test data, documented dataset provenance, and explicit failure analysis.

`run_experiment.py` compares Edge and Lux samples and writes `EXPERIMENT_REPORT.md`:

```bash
poetry run python run_experiment.py
```

This requires the data sources and both providers to be configured. The generated report contains fixed narrative conclusions; check them against the measured results before using the report.

## Tests

```bash
poetry run python -m pytest tests -q
```

The tests exercise selected signal-analysis and classifier behavior, including training on generated feature arrays. Passing these tests does not establish accuracy on real-world audio.

## Code map

| File | Responsibility |
| --- | --- |
| `src/app.py` | Dash interface and callbacks |
| `src/ml_core.py` | Feature extraction, model persistence, prediction |
| `src/bionic_core.py` | Experimental signal analysis |
| `src/train_model.py` | Training and evaluation output |
| `dataset_builder.py` | Dataset ingestion and TTS generation |
| `src/servers/` | Local TTS wrappers |
| `run_experiment.py` | Provider comparison and report generation |

## License

[MIT](LICENSE) for this repository's code. External models, datasets, and TTS repositories retain their own licenses and access conditions.
