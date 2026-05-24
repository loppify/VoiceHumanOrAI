import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def create_spectrograms(human_path, ai_path, output_path):
    # Завантаження аудіо
    y_human, sr_human = librosa.load(human_path, sr=None)
    y_ai, sr_ai = librosa.load(ai_path, sr=None)

    # Обчислення Мел-спектрограм
    S_human = librosa.feature.melspectrogram(y=y_human, sr=sr_human, n_mels=128, fmax=8000)
    S_human_db = librosa.power_to_db(S_human, ref=np.max)

    S_ai = librosa.feature.melspectrogram(y=y_ai, sr=sr_ai, n_mels=128, fmax=8000)
    S_ai_db = librosa.power_to_db(S_ai, ref=np.max)

    # Побудова графіка
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    img1 = librosa.display.specshow(S_human_db, x_axis='time', y_axis='mel', sr=sr_human, fmax=8000, ax=ax1,
                                    cmap='magma')
    ax1.set_title('а) Спектрограма реального голосу (Людина)')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    img2 = librosa.display.specshow(S_ai_db, x_axis='time', y_axis='mel', sr=sr_ai, fmax=8000, ax=ax2, cmap='magma')
    ax2.set_title('б) Спектрограма синтезованого голосу (ШІ)')
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Спектрограму збережено у {output_path}")


if __name__ == "__main__":
    human = "dataset/human/human_00000.wav"
    ai = "dataset/ai/ai_00000.wav"
    out = "spectrogram_comparison.png"
    if os.path.exists(human) and os.path.exists(ai):
        create_spectrograms(human, ai, out)
    else:
        print("Файли не знайдено!")
