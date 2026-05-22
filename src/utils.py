import subprocess

import numpy as np


def normalize_signal(raw: np.ndarray) -> np.ndarray:
    """Приводить вхідний сигнал до стандартної 8-бітної шкали (-128...127)."""
    if raw.dtype == np.uint8:
        return raw.astype(np.float64) - 128.0
    elif raw.dtype == np.int16:
        return (raw.astype(np.float64) / 32768.0) * 128.0
    elif raw.dtype == np.int32:
        return (raw.astype(np.float64) / 2147483648.0) * 128.0
    else:
        return raw.astype(np.float64)


def convert_audio_format(input_path: str, output_path: str, sample_rate: int = 22050):
    """Використовує ffmpeg для конвертації в WAV 16-bit Mono (найбільш сумісний)."""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
