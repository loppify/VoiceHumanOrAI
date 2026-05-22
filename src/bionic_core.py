from typing import Tuple, Optional, List, Dict, Any

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist

try:
    from utils import normalize_signal
except ImportError:
    from .utils import normalize_signal


class BionicClassifier:
    """
    Біонічний метод класифікації мовного сигналу "Людина-ШІ".
    Використовує координатно-топологічне відображення у просторі Хелвага-Щерби.
    """

    def __init__(self,
                 sample_rate: int = 22050,
                 t_analysis: int = 2500,
                 smoothing_window: int = 16,
                 peak_prominence: int = 10):
        self.sample_rate = sample_rate
        self.t_analysis = t_analysis
        self.smoothing_window = smoothing_window
        self.peak_prominence = peak_prominence

        # Параметри простору Хелвага-Щерби
        self.l_max = 200
        self.h_max = 200
        self.window_size = 15
        self.step_size = 3
        self.min_points_in_window = 2

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(signal, kernel, mode='same')

    def _find_best_oscillation(self, window: np.ndarray) -> Optional[Tuple[int, int, int]]:
        valleys, _ = find_peaks(-window, distance=self.smoothing_window + 1, prominence=self.peak_prominence)
        peaks, _ = find_peaks(window, distance=self.smoothing_window + 1, prominence=self.peak_prominence)

        if len(valleys) < 2 or len(peaks) < 1:
            return None

        best_osc = None
        max_energy = -np.inf

        for i in range(len(valleys) - 1):
            p0 = valleys[i]
            p1_list = peaks[peaks > p0]
            if len(p1_list) == 0: continue
            p1 = p1_list[0]
            p2_list = valleys[valleys > p1]
            if len(p2_list) == 0: continue
            p2 = p2_list[0]

            if 80 <= (p2 - p0) <= 200:
                energy = np.sum(np.abs(window[p0:p2]))
                if energy > max_energy:
                    max_energy, best_osc = energy, (p0, p1, p2)
        return best_osc

    def extract_features(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features, amplitudes = [], []
        pos = 0
        while pos + self.t_analysis < len(signal):
            chunk = signal[pos: pos + self.t_analysis]
            osc = self._find_best_oscillation(chunk)
            if osc:
                p0, p1, p2 = osc
                v1, v2 = float(p1 - p0), float(p2 - p1)
                amp = float(chunk[p1] - chunk[p0])
                if v1 > 0 and v2 > 0:
                    features.append([v1, v2])
                    amplitudes.append(amp)
                pos += p2
            else:
                pos += self.t_analysis // 4
        return np.array(features), np.array(amplitudes)

    def find_density_centers(self, points: np.ndarray) -> List[Tuple[float, float]]:
        if len(points) < self.min_points_in_window: return []
        candidates = []
        for x in range(0, self.l_max - self.window_size, self.step_size):
            for y in range(0, self.h_max - self.window_size, self.step_size):
                mask = (points[:, 0] >= x) & (points[:, 0] <= x + self.window_size) & \
                       (points[:, 1] >= y) & (points[:, 1] <= y + self.window_size)
                win_pts = points[mask]
                if len(win_pts) >= self.min_points_in_window:
                    r_w = np.sum(pdist(win_pts)) / (len(win_pts) ** 2)
                    candidates.append((r_w, x + self.window_size / 2, y + self.window_size / 2))

        candidates.sort(key=lambda x: x[0])
        centers = []
        for _, cx, cy in candidates:
            if len(centers) >= 3: break
            if all(np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) > 15 for _, ex, ey in [(0, c[0], c[1]) for c in centers]):
                centers.append((cx, cy))
        return centers[:3]

    def analyze_file(self, filepath: str, threshold: float = 61.0) -> Dict[str, Any]:
        import librosa
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не знайдено: {filepath}")
            
        # Завантаження через librosa для кращої підтримки форматів
        sig, rate = librosa.load(filepath, sr=self.sample_rate)
        # Приведення до 8-бітної шкали (-128...127)
        sig = sig * 128.0

        smoothed = self._smooth_signal(sig)
        points, amplitudes = self.extract_features(smoothed)
        centers = self.find_density_centers(points)

        dist = []
        if len(centers) > 1:
            # Обчислюємо відстані між усіма знайденими центрами (навіть якщо їх 2)
            dist = pdist(centers)

        mean_r = np.mean(dist) if len(dist) > 0 else 0.0

        # Захист від ділення на нуль та обчислення Jitter/Shimmer
        jitter = 0.0
        shimmer = 0.0
        variability = 0.0

        if len(points) > 5:
            periods = points.sum(axis=1)
            mean_period = np.mean(periods)
            if mean_period > 0:
                jitter = (np.std(periods) / mean_period) * 100

            mean_amp = np.mean(amplitudes)
            if mean_amp > 0:
                shimmer = (np.std(amplitudes) / mean_amp) * 100

            variability = np.std(points[:, 0]) + np.std(points[:, 1])

        # НОВА ФОРМУЛА SCORE: 
        # У людини природно вищий Jitter (тремор) та Shimmer, а також більший розкид (mean_r).
        # Навіть якщо центри не знайдено (короткий запис), високий Jitter врятує ситуацію.
        score = (mean_r * 0.5) + (variability * 0.3) + (jitter * 2.0) + shimmer

        # Обмежуємо score знизу нулем
        score = max(0.0, score)
        return {
            'points': points, 'centers': centers, 'distances': dist,
            'mean_r': score, 'jitter': jitter, 'shimmer': shimmer,
            'verdict': "Людина" if score > threshold else "ШІ (AI)",
            'signal': sig
        }
