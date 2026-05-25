import argparse
import asyncio
import glob
import os
import random
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod

import requests
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.utils import convert_audio_format

load_dotenv()
HF_TOKEN = os.environ.get("HuggingFace_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

import time


def get_venv_python(base_dir):
    """Шукає шлях до python у віртуальному середовищі."""
    for venv in ["venv", ".venv"]:
        path = os.path.join(base_dir, venv, "bin", "python")
        if os.path.exists(path):
            return os.path.abspath(path)
    return sys.executable  # Фоллбек на поточний інтерпретатор


class TTSProvider(ABC):
    """Базовий клас для генераторів мовлення."""

    async def __aenter__(self): return self

    async def __aexit__(self, *args): pass

    @abstractmethod
    async def generate(self, text: str, output_path: str):
        pass


class ServerManagedProvider(TTSProvider):
    """Базовий клас для провайдерів, що потребують запуску сервера."""

    def __init__(self, url, server_dir, server_script, health_url=None):
        self.url = url
        self.server_dir = os.path.abspath(server_dir)
        self.server_script = os.path.abspath(server_script)
        self.health_url = health_url or f"{url.rsplit('/', 1)[0]}/health"
        self.process = None

    async def _ensure_venv(self):
        """Створює venv та встановлює залежності, якщо потрібно."""
        venv_path = os.path.join(self.server_dir, "venv")
        python_exe = get_venv_python(self.server_dir)

        # Перевіряємо чи це системний python. Якщо так - venv не існує
        is_system_python = python_exe == sys.executable

        if is_system_python and not os.path.exists(venv_path):
            logger.info(f"Створення venv у {self.server_dir}...")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            # Примусово беремо шлях до нового venv
            python_exe = os.path.join(venv_path, "bin", "python")

            # Оновлюємо pip
            logger.info("Оновлення pip...")
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

            # Встановлюємо базові пакети для сервера
            logger.info("Встановлення базових пакетів (fastapi, uvicorn, etc.)...")
            subprocess.run([python_exe, "-m", "pip", "install", "fastapi", "uvicorn", "pydantic", "requests"],
                           check=True)

        # Додаткова перевірка на torch
        try:
            subprocess.run([python_exe, "-c", "import torch"], check=True, capture_output=True)
            # Перевірка на transformers сумісної версії
            subprocess.run(
                [python_exe, "-c", "import transformers; assert int(transformers.__version__.split('.')[0]) < 5"],
                check=True, capture_output=True)
        except:
            logger.warning(f"Залежності відсутні або несумісні у {self.server_dir}. Встановлення...")
            req_file = os.path.join(self.server_dir, "requirements.txt")
            if os.path.exists(req_file):
                # Намагаємось встановити все
                res = subprocess.run([python_exe, "-m", "pip", "install", "-r", req_file])
                if res.returncode != 0:
                    logger.error("Помилка встановлення вимог. Спроба встановити лише критичні пакети...")
                    subprocess.run(
                        [python_exe, "-m", "pip", "install", "torch", "torchaudio", "transformers<5.0", "sentencepiece",
                         "soundfile", "onnxruntime"], check=True)
            else:
                subprocess.run([python_exe, "-m", "pip", "install", "torch", "torchaudio", "transformers<5.0"],
                               check=True)

        return python_exe

    async def __aenter__(self):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not await self._is_running():
                    python_exe = await self._ensure_venv()
                    cmd = [python_exe, self.server_script]
                    logger.info(f"Запуск сервера {os.path.basename(self.server_script)} (Спроба {attempt + 1})...")

                    self.process = subprocess.Popen(
                        cmd,
                        cwd=self.server_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    # Потік для логів
                    def log_streamer(pipe):
                        with pipe:
                            buffer = ""
                            while True:
                                char = pipe.read(1)
                                if not char: break
                                if char in ('\n', '\r'):
                                    line = buffer.strip()
                                    if line:
                                        # Фільтруємо "шум" для кращого UX
                                        is_noise = any(x in line for x in [
                                            "Traceback (most recent call last)",
                                            "File \"",
                                            "    wav =",
                                            "    final_wav =",
                                            "RuntimeError:",
                                            "HTTP/1.1\" 400",
                                            "Fetching 11 files",
                                            "Device set to use"
                                        ])
                                        if not is_noise:
                                            logger.info(f"[{os.path.basename(self.server_script)}] {line}")
                                    buffer = ""
                                else:
                                    buffer += char

                    threading.Thread(target=log_streamer, args=(self.process.stdout,), daemon=True).start()

                    try:
                        await self._wait_for_server()
                        return self  # Успішний запуск
                    except RuntimeError as e:
                        # Якщо сервер не запустився, перевіряємо чи не через відсутність модулів
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Сервер не відповів. Можливо, пошкоджено venv. Спроба перевстановлення залежностей...")
                            # Видаляємо venv щоб форсувати перевстановлення
                            venv_path = os.path.join(self.server_dir, "venv")
                            if os.path.exists(venv_path):
                                import shutil
                                shutil.rmtree(venv_path)
                            continue
                        else:
                            raise e
                else:
                    return self
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                raise e
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            logger.info(f"Зупинка сервера {os.path.basename(self.server_script)}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    async def _is_running(self):
        try:
            r = requests.get(self.health_url, timeout=1)
            return r.status_code == 200
        except:
            return False

    async def _wait_for_server(self, timeout=120):  # Збільшив таймаут для важких моделей
        start = time.time()
        while time.time() - start < timeout:
            if await self._is_running():
                logger.info("Сервер готовий!")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Сервер {self.url} не запустився за {timeout} секунд")


class EdgeTTSProvider(TTSProvider):
    """Провайдер для Edge TTS (хмарний)."""

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


class LuxTTSProvider(ServerManagedProvider):
    """Провайдер для LuxTTS (через локальний мікросервіс)."""

    def __init__(self, url="http://127.0.0.1:8001/generate"):
        super().__init__(url, "external/LuxTTS", "src/servers/lux_server.py")

    async def generate(self, text: str, output_path: str):
        refs = glob.glob("dataset/human/*.wav")
        if not refs:
            raise ValueError("Для клонування потрібен хоча б один файл у dataset/human")

        raw_out = output_path + ".raw.wav"
        payload = {
            "text": text,
            "ref_path": os.path.abspath(random.choice(refs)),
            "output_path": os.path.abspath(raw_out)
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        convert_audio_format(raw_out, output_path)
        if os.path.exists(raw_out): os.remove(raw_out)


class MossTTSProvider(ServerManagedProvider):
    """Провайдер для MOSS-TTS (через локальний мікросервіс)."""

    def __init__(self, url="http://127.0.0.1:8002/generate"):
        super().__init__(url, "external/MOSS-TTS-Nano", "src/servers/moss_server.py")

    async def generate(self, text: str, output_path: str):
        refs = glob.glob("dataset/human/*.wav")
        ref = os.path.abspath(random.choice(refs)) if refs else None

        raw_out = output_path + ".raw.wav"
        payload = {
            "text": text,
            "output_path": os.path.abspath(raw_out),
            "ref_path": ref
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        convert_audio_format(raw_out, output_path)
        if os.path.exists(raw_out): os.remove(raw_out)


class VoiceboxProvider(TTSProvider):
    """Провайдер для Voicebox Studio API (передбачається, що запущено через Docker/інше)."""

    def __init__(self, url="http://127.0.0.1:17493/generate"):
        self.url = url

    async def generate(self, text: str, output_path: str):
        response = requests.post(self.url, json={"text": text, "language": "en"}, stream=True)
        response.raise_for_status()
        raw_out = output_path + ".raw.wav"
        with open(raw_out, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        convert_audio_format(raw_out, output_path)
        os.remove(raw_out)


async def build_dataset(samples: int,
                        provider_name: str,
                        dataset_name: str,
                        config_name: str,
                        split_name: str,
                        human_dir: str = "dataset/human",
                        ai_dir: str = "dataset/ai"):
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    providers = {
        "edge": EdgeTTSProvider,
        "lux": LuxTTSProvider,
        "mosstts": MossTTSProvider,
        "voicebox": VoiceboxProvider
    }

    provider_class = providers[provider_name]
    logger.info(f"Завантаження датасету {dataset_name} (конфігурація: {config_name}, split: {split_name})...")
    ds = load_dataset(dataset_name, config_name, split=split_name, streaming=True, token=HF_TOKEN).shuffle(
        buffer_size=1000)
    ds = ds.cast_column("audio", Audio(decode=False))

    count = 0
    pbar = tqdm(total=samples)

    while count < samples:
        try:
            async with provider_class() as provider:
                for row in ds:
                    if count >= samples: break
                    try:
                        # В minds14 `transcription` - це оригінал, `english_transcription` - переклад.
                        # Якщо ми вибрали en-US, нам потрібен `transcription`
                        text = row.get("transcription") or row.get("english_transcription") or row.get("text")

                        # Додаткова перевірка мови для minds14 (поле 'language' або 'lang')
                        # Це допоможе відсіяти артефакти, коли в en-US потрапляють інші мови
                        row_lang = str(row.get("language") or row.get("lang") or "").lower()
                        if config_name.lower().startswith("en") and row_lang and not row_lang.startswith(
                                "0"):  # 0 is en-US in minds14
                            # В minds14 мови кодуються числами. en-US = 0.
                            if row_lang != "0":
                                continue

                        if not text or len(text.strip()) < 2:
                            logger.warning("Пропущено порожній семпл.")
                            continue

                        h_path = os.path.join(human_dir, f"human_{count:05}.wav")
                        ai_path = os.path.join(ai_dir, f"ai_{count:05}.wav")

                        # Зберігаємо людський голос
                        with open(h_path + ".tmp", "wb") as f:
                            f.write(row["audio"]["bytes"])
                        convert_audio_format(h_path + ".tmp", h_path)
                        os.remove(h_path + ".tmp")

                        # Генеруємо ШІ через мікросервіс
                        try:
                            await provider.generate(text, ai_path)
                            count += 1
                            pbar.update(1)
                        except requests.exceptions.HTTPError as he:
                            if he.response.status_code == 400:
                                logger.warning(
                                    f"Провайдер відхилив текст (можливо, непідтримувана мова): {text[:30]}...")
                                # Видаляємо створений human файл, бо пара не вийшла
                                if os.path.exists(h_path): os.remove(h_path)
                                continue  # Йдемо до наступного семплу
                            else:
                                raise he
                    except requests.exceptions.ConnectionError:
                        logger.warning("З'єднання з сервером втрачено. Спроба перезапуску провайдера...")
                        break  # Виходимо з внутрішнього циклу для перезапуску контекст-менеджера
                    except Exception as e:
                        logger.error(f"Sample failed: {e}")
        except Exception as e:
            logger.error(f"Provider fatal error: {e}. Очікування 10 секунд перед повторною спробою...")
            await asyncio.sleep(10)

    pbar.close()
    logger.info(f"Генерацію завершено! Створено {count} пар.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--provider", default="edge", choices=["edge", "lux", "mosstts", "voicebox"])
    parser.add_argument("--dataset", default="PolyAI/minds14")
    parser.add_argument("--config", default="en-US")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    try:
        asyncio.run(build_dataset(args.samples, args.provider, args.dataset, args.config, args.split))
    except KeyboardInterrupt:
        logger.info("Процес перервано користувачем.")
        sys.exit(0)
