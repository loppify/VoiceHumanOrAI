import os
import numpy as np
from scipy.io import wavfile

def generate_test_audio():
    sample_rate = 22050
    duration = 10 # 10 seconds
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    
    # 1. Створюємо "AI-like" сигнал: ідеальна математична синусоїда без шуму
    ai_freq = 440.0
    ai_wave = np.sin(2 * np.pi * ai_freq * t)
    # Зміна амплітуди
    ai_wave *= 0.5
    # Конвертуємо в 16-bit
    ai_audio = np.int16(ai_wave * 32767)
    
    # 2. Створюємо "Human-like" сигнал: з шумом, зміною частоти (варіативність артикуляції)
    human_freq = 440.0 + 10 * np.sin(2 * np.pi * 2 * t) # Частотна модуляція
    noise = np.random.normal(0, 0.2, len(t))
    human_wave = np.sin(2 * np.pi * human_freq * t) + noise
    human_wave = np.clip(human_wave, -1.0, 1.0) * 0.5
    human_audio = np.int16(human_wave * 32767)
    
    os.makedirs("test_audio", exist_ok=True)
    ai_path = "test_audio/test_ai.wav"
    human_path = "test_audio/test_human.wav"
    
    wavfile.write(ai_path, sample_rate, ai_audio)
    wavfile.write(human_path, sample_rate, human_audio)
    
    print(f"Згенеровано файли для тестування:\n- {ai_path} (чистий ШІ)\n- {human_path} (з шумом Людини)")
    return ai_path, human_path

if __name__ == "__main__":
    ai_path, human_path = generate_test_audio()
    
    # Протестуємо біонічний алгоритм
    import sys
    sys.path.insert(0, os.path.abspath('src'))
    from bionic_core import BionicClassifier
    
    classifier = BionicClassifier()
    
    print("\nАналіз 'ШІ' файлу:")
    ai_res = classifier.analyze_file(ai_path)
    print(f"Середнє R: {ai_res['mean_r']:.2f}")
    print(f"Вердикт: {ai_res['verdict']}")
    
    print("\nАналіз 'Людина' файлу:")
    human_res = classifier.analyze_file(human_path)
    print(f"Середнє R: {human_res['mean_r']:.2f}")
    print(f"Вердикт: {human_res['verdict']}")
