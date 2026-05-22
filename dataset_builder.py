import argparse
import asyncio
import os
import random
import subprocess
import tempfile
import glob
from abc import ABC, abstractmethod

import requests
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from src.utils import convert_audio_format

load_dotenv()
HF_TOKEN = os.environ.get("HuggingFace_TOKEN")

class TTSProvider(ABC):
    @abstractmethod
    async def generate(self, text: str, output_path: str):
        pass

class EdgeTTSProvider(TTSProvider):
    def __init__(self):
        import edge_tts
        self.edge_tts = edge_tts
        self.voices = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural"]

    async def generate(self, text: str, output_path: str):
        voice = random.choice(self.voices)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            await self.edge_tts.Communicate(text, voice).save(tmp_path)
            convert_audio_format(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

class LuxTTSProvider(TTSProvider):
    def __init__(self):
        import sys
        import torch
        sys.path.append(os.path.abspath("external/LuxTTS"))
        try:
            from zipvoice.luxvoice import LuxTTS
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Завантаження LuxTTS на {self.device}...")
            self.lux_tts = LuxTTS('YatharthS/LuxTTS', device=self.device)
        except ImportError:
            logger.error("Не вдалося завантажити LuxTTS. Переконайтеся, що PYTHONPATH налаштований.")
            raise

    async def generate(self, text: str, output_path: str):
        import soundfile as sf
        # Використовуємо існуючий людський голос для клонування
        refs = glob.glob("dataset/human/*.wav")
        if not refs:
            raise ValueError("Для клонування потрібен хоча б один файл у dataset/human")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            prompt = self.lux_tts.encode_prompt(random.choice(refs), rms=0.01)
            wav = self.lux_tts.generate_speech(text, prompt, num_steps=4).numpy().squeeze()
            sf.write(tmp_path, wav, 48000)
            convert_audio_format(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

class MossTTSProvider(TTSProvider):
    async def generate(self, text: str, output_path: str):
        raise RuntimeError(
            "ПЛАН ДІЙ ДЛЯ MOSS-TTS:\n"
            "1. Модель MOSS-TTS вимагає конфліктних залежностей (pynini, WeTextProcessing).\n"
            "2. Будь ласка, відкрийте новий термінал і створіть для нього окреме віртуальне середовище.\n"
            "3. Запустіть локальний сервер: `python app_onnx.py` у папці external/MOSS-TTS-Nano.\n"
            "4. Якщо сервер запущено, змініть логіку цього провайдера для виконання POST-запиту."
        )

class VoiceboxProvider(TTSProvider):
    def __init__(self, api_url="http://127.0.0.1:17493"):
        self.api_url = api_url

    async def generate(self, text: str, output_path: str):
        try:
            response = requests.post(f"{self.api_url}/generate", json={"text": text, "language": "en"}, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            convert_audio_format(tmp_path, output_path)
            os.remove(tmp_path)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Не вдалося підключитись до Voicebox за адресою {self.api_url}.\n"
                "ПЛАН ДІЙ:\n"
                "1. Відкрийте новий термінал.\n"
                "2. Перейдіть у папку: cd external/voicebox\n"
                "3. Запустіть сервер: just dev (або npm run dev / python backend/main.py).\n"
                "4. Переконайтесь, що сервер працює, і повторіть спробу."
            )

async def build_dataset(samples: int, provider_name: str, human_dir: str = "dataset/human", ai_dir: str = "dataset/ai", dataset_name: str = "PolyAI/minds14", config_name: str = "en-US"):
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    providers = {"edge": EdgeTTSProvider, "lux": LuxTTSProvider, "mosstts": MossTTSProvider, "voicebox": VoiceboxProvider}
    
    if provider_name not in providers:
        raise ValueError(f"Невідомий провайдер: {provider_name}")
        
    provider = providers[provider_name]()

    from datasets import get_dataset_split_names

    # 1. Знаходимо валідний спліт автоматично
    try:
        valid_splits = get_dataset_split_names(dataset_name, config_name, token=HF_TOKEN)
        target_split = next((s for s in valid_splits if 'train' in s.lower()), valid_splits[0])
        logger.info(f"Знайдено спліти: {valid_splits}. Використовуємо: {target_split}")
    except Exception as e:
        logger.warning(f"Не вдалося отримати список сплітів: {e}. Пробуємо 'train'")
        target_split = 'train'

    # 2. Завантажуємо датасет та ПЕРЕМІШУЄМО його
    try:
        # Додано shuffle(buffer_size=1000) для гарантованої рандомізації при кожному запуску
        ds = load_dataset(dataset_name, config_name, split=target_split, streaming=True, token=HF_TOKEN).shuffle(buffer_size=1000, seed=random.randint(0, 10000))
    except Exception as e:
        logger.warning(f"Помилка завантаження з конфігом '{config_name}': {e}. Спроба без конфігу...")
        try:
            ds = load_dataset(dataset_name, split=target_split, streaming=True, token=HF_TOKEN).shuffle(buffer_size=1000, seed=random.randint(0, 10000))
        except Exception as e2:
            raise RuntimeError(
                f"Не вдалося завантажити датасет '{dataset_name}'.\n"
                f"Деталі: {e2}\n"
                "ПЛАН ДІЙ:\n"
                "1. Перевірте назву датасету на сайті HuggingFace.\n"
                "2. Можливо, він приватний і потребує валідного HuggingFace_TOKEN у файлі .env."
            )

    ds = ds.cast_column("audio", Audio(decode=False))

    count = 0
    skipped_count = 0
    
    for row in tqdm(ds, total=samples):
        if count >= samples: break
        
        # Захист від нескінченного циклу у "пустих" датасетах
        if skipped_count > samples * 10:
            raise RuntimeError(f"Знайдено забагато семплів без тексту. Перевірте структуру датасету {dataset_name}.")
            
        try:
            text = row.get("english_transcription") or row.get("text") or row.get("sentence")
            if not text:
                skipped_count += 1
                continue
            
            h_path = os.path.join(human_dir, f"human_{count:05}.wav")
            ai_path = os.path.join(ai_dir, f"ai_{count:05}.wav")
            
            with open(h_path + ".tmp", "wb") as f: f.write(row["audio"]["bytes"])
            convert_audio_format(h_path + ".tmp", h_path)
            os.remove(h_path + ".tmp")
            
            await provider.generate(text, ai_path)
            count += 1
        except RuntimeError as re:
            # Це наші "Плани дій", прокидаємо їх вище для виводу в UI
            raise re
        except Exception as e:
            logger.error(f"Помилка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PolyAI/minds14")
    parser.add_argument("--config", default="en-US")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--provider", default="edge")
    args = parser.parse_args()
    asyncio.run(build_dataset(args.samples, args.provider, dataset_name=args.dataset, config_name=args.config))
