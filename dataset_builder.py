import argparse
import asyncio
import os
import random
import subprocess
import tempfile
import glob
import json
from abc import ABC, abstractmethod

import requests
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()
HF_TOKEN = os.environ.get("HuggingFace_TOKEN")

TARGET_SR = 22050
DATASET_DIR = "dataset"
AI_DIR = os.path.join(DATASET_DIR, "ai")
HUMAN_DIR = os.path.join(DATASET_DIR, "human")

def process_audio(input_path, output_path):
    """Конвертація аудіо у необхідний формат (22050 Hz, Mono)."""
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-acodec", "pcm_u8e",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-af",
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.1,atrim=0:10",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TTSProvider(ABC):
    """Базовий клас для генераторів мовлення."""
    @abstractmethod
    async def generate(self, text: str, output_path: str):
        pass

class EdgeTTSProvider(TTSProvider):
    """Провайдер для Edge TTS (хмарний)."""
    def __init__(self):
        import edge_tts
        self.edge_tts = edge_tts
        self.voices = [
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-GB-RyanNeural",
            "en-AU-NatashaNeural",
        ]

    async def generate(self, text: str, output_path: str):
        voice = random.choice(self.voices)
        tts = self.edge_tts.Communicate(text, voice)
        await tts.save(output_path)

class LuxTTSProvider(TTSProvider):
    """Провайдер для LuxTTS (локальне клонування)."""
    def __init__(self):
        import sys
        import torch
        sys.path.append(os.path.abspath("external/LuxTTS"))
        try:
            from zipvoice.luxvoice import LuxTTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Завантаження LuxTTS на {device}...")
            self.lux_tts = LuxTTS('YatharthS/LuxTTS', device=device)
        except ImportError:
            logger.error("Не вдалося імпортувати LuxTTS. Переконайтеся, що PYTHONPATH налаштований.")
            raise

    async def generate(self, text: str, output_path: str):
        import soundfile as sf
        human_refs = glob.glob(os.path.join(HUMAN_DIR, "*.wav"))
        if not human_refs:
            raise ValueError("Потрібен хоча б один файл у dataset/human для клонування.")
        
        ref_path = random.choice(human_refs)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        
        try:
            encoded_prompt = self.lux_tts.encode_prompt(ref_path, rms=0.01)
            final_wav = self.lux_tts.generate_speech(text, encoded_prompt, num_steps=4)
            final_wav_np = final_wav.numpy().squeeze()
            sf.write(tmp_path, final_wav_np, 48000)
            process_audio(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

class MossTTSProvider(TTSProvider):
    """Провайдер для MOSS-TTS-Nano (локальна генерація)."""
    async def generate(self, text: str, output_path: str):
        human_refs = glob.glob(os.path.join(HUMAN_DIR, "*.wav"))
        ref_path = random.choice(human_refs) if human_refs else None
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath("external/MOSS-TTS-Nano")
        
        cmd = ["venv/bin/python", "external/MOSS-TTS-Nano/infer.py", "--text", text, "--output-audio-path", output_path + ".raw.wav", "--device", "cpu"]
        if ref_path:
            cmd.extend(["--prompt-audio-path", ref_path])
        
        try:
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_audio(output_path + ".raw.wav", output_path)
            os.remove(output_path + ".raw.wav")
        except subprocess.CalledProcessError as e:
            logger.error(f"MOSS-TTS Error: {e.stderr.decode()}")
            raise

class VoiceboxProvider(TTSProvider):
    """Провайдер для Voicebox Studio (через локальний API)."""
    def __init__(self, api_url="http://127.0.0.1:17493"):
        self.api_url = api_url

    async def generate(self, text: str, output_path: str):
        # Voicebox очікує JSON
        payload = {"text": text, "language": "en"}
        try:
            response = requests.post(f"{self.api_url}/generate", json=payload, stream=True)
            response.raise_for_status()
            with open(output_path + ".raw.wav", 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            process_audio(output_path + ".raw.wav", output_path)
            os.remove(output_path + ".raw.wav")
        except Exception as e:
            logger.error(f"Voicebox API Error: {e}")
            raise

class OmniVoiceProvider(TTSProvider):
    """Провайдер для OmniVoice Studio (через локальний API)."""
    def __init__(self, api_url="http://127.0.0.1:8000"): # Типовий порт FastAPI
        self.api_url = api_url

    async def generate(self, text: str, output_path: str):
        # OmniVoice очікує Multipart Form
        human_refs = glob.glob(os.path.join(HUMAN_DIR, "*.wav"))
        files = None
        data = {"text": text, "language": "en"}
        
        if human_refs:
            ref_path = random.choice(human_refs)
            files = {'ref_audio': open(ref_path, 'rb')}
            
        try:
            response = requests.post(f"{self.api_url}/generate", data=data, files=files, stream=True)
            response.raise_for_status()
            with open(output_path + ".raw.wav", 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            process_audio(output_path + ".raw.wav", output_path)
            os.remove(output_path + ".raw.wav")
        except Exception as e:
            logger.error(f"OmniVoice API Error: {e}")
            raise
        finally:
            if files: files['ref_audio'].close()


async def generate_paired_dataset(dataset_name="PolyAI/minds14", config_name="en-US", split="train", samples=10, provider_name="edge"):
    os.makedirs(AI_DIR, exist_ok=True)
    os.makedirs(HUMAN_DIR, exist_ok=True)
    
    providers = {
        "edge": EdgeTTSProvider,
        "lux": LuxTTSProvider,
        "mosstts": MossTTSProvider,
        "voicebox": VoiceboxProvider,
        "omnivoice": OmniVoiceProvider
    }
    
    if provider_name not in providers:
        logger.error(f"Невідомий провайдер: {provider_name}")
        return
        
    logger.info(f"Ініціалізація {provider_name}...")
    tts_provider = providers[provider_name]()
    
    dataset = load_dataset(dataset_name, config_name if config_name else None, split=split, streaming=True, token=HF_TOKEN)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    count = 0
    pbar = tqdm(total=samples)
    
    for row in dataset:
        try:
            audio_data = row.get("audio")
            text = row.get("english_transcription") or row.get("text")
            if not audio_data or not text: continue

            out_human = os.path.join(HUMAN_DIR, f"human_{count:05}.wav")
            out_ai = os.path.join(AI_DIR, f"ai_{count:05}.wav")

            # Збереження людського оригіналу
            with open(out_human + ".tmp.wav", "wb") as f: f.write(audio_data["bytes"])
            process_audio(out_human + ".tmp.wav", out_human)
            os.remove(out_human + ".tmp.wav")

            # Генерація клону
            await tts_provider.generate(text, out_ai)

            count += 1
            pbar.update(1)
            if count >= samples: break
        except Exception as e:
            logger.warning(f"Семпл пропущено: {e}")
            continue
            
    pbar.close()
    logger.info("Генерацію завершено!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PolyAI/minds14")
    parser.add_argument("--config", default="en-US")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--provider", default="edge", choices=["edge", "lux", "mosstts", "voicebox", "omnivoice"])
    args = parser.parse_args()
    asyncio.run(generate_paired_dataset(args.dataset, args.config, samples=args.samples, provider_name=args.provider))
