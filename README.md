# Voice I/O Lab: Human or AI Classifier 🎙️🤖

A hybrid analysis laboratory designed to distinguish between natural human speech and AI-generated deepfakes. This project combines **Bionic Analysis** (Coordinate-Topological mapping in Helvag-Shcherba space) with **Classic Machine Learning** (Random Forest + MFCC) to provide a high-confidence verdict on voice authenticity.

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Architecture](https://img.shields.io/badge/architecture-Microservices-orange)

## ✨ Key Features

*   **Hybrid Detection:** Uses two independent classification engines for maximum reliability.
*   **Autonomous TTS Infrastructure:** Automatically manages external TTS servers (LuxTTS, MOSS-TTS) in isolated virtual environments.
*   **Real-time Analysis:** Interactive Dash UI for audio uploading, waveform visualization, and spectrogram generation.
*   **Dataset Builder:** Integrated tool to pull data from HuggingFace and generate synthetic pairs on the fly.
*   **Explainable AI (XAI):** Visualizes feature importance and topological clusters to explain *why* a voice is flagged as AI.

## 🏗️ Architecture

The project follows a modular, microservice-inspired architecture:
- **`src/app.py`**: The central Dash-based Web UI.
- **`src/bionic_core.py`**: Implementation of the bionic coordinate-topological method.
- **`src/ml_core.py`**: Feature extraction and Random Forest classifier.
- **`dataset_builder.py`**: Orchestrates data collection and automated TTS server lifecycles.
- **`external/`**: Cloned submodules of state-of-the-art TTS engines.
- **`src/servers/`**: Custom API wrappers to run external TTS engines as independent services.

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation)
- `ffmpeg` installed on your system

### Installation

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/your-repo/VoiceHumanOrAI.git
   cd VoiceHumanOrAI
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Setup environment variables:
   Create a `.env` file in the root directory:
   ```env
   HuggingFace_TOKEN=your_hf_token_here
   ```

### Running the Application

Launch the main analysis laboratory:
```bash
poetry run python src/app.py
```
Open your browser at `http://127.0.0.1:8050`.

## 🛠️ Usage

### 1. Analysis Laboratory
Upload any `.wav` file or select one from the local database. The system will generate:
- **Oscillogram:** Time-domain signal visualization.
- **Spectrogram:** Frequency-domain energy distribution.
- **Bionic Space:** Topological distribution of voice oscillations.
- **ML Verdict:** Classification probability based on MFCC features.

### 2. Training & Data Generation
- Select a dataset from HuggingFace (e.g., `minds14`).
- Choose a TTS Provider (e.g., `LuxTTS`).
- Click **"Start Generation"**. The system will automatically create a `venv` for the provider, install its dependencies, start the background server, and generate synthetic voice samples.
- Once finished, click **"Train Model"** to update the Random Forest classifier with new data.

## 🔬 Running Experiments

To run a comparative scientific experiment and generate a markdown report:
```bash
poetry run python run_experiment.py
```
Results will be saved in `EXPERIMENT_REPORT.md`.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created as part of an investigation into high-fidelity voice synthesis detection.*
