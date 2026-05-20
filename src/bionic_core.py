import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from typing import Tuple, Optional, List

class BionicClassifier:
    """
    Біонічний метод класифікації мовного сигналу "Людина-ШІ".
    Базується на координатно-топологічному відображенні у просторі Хелвага-Щерби.
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
        
        # Параметри для пошуку центрів щільності
        self.l_range = (80, 200)
        self.l_max = 200
        self.h_max = 200
        self.window_l = 15
        self.window_h = 15
        self.step_size = 3
        self.min_points = 2

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Згладжування сигналу методом змінного середнього."""
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed

    def _find_best_oscillation(self, window: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Знаходження найкращої осциляції на ділянці (виділення піків)."""
        valleys, _ = find_peaks(-window, distance=self.smoothing_window + 1, prominence=self.peak_prominence)
        peaks, _ = find_peaks(window, distance=self.smoothing_window + 1, prominence=self.peak_prominence)
        
        if len(valleys) < 2 or len(peaks) < 1:
            return None
            
        best_oscillation = None
        max_energy = -np.inf
        l_min, l_max = self.l_range
        
        for i in range(len(valleys) - 1):
            p0 = valleys[i]
            suitable_peaks = peaks[peaks > p0]
            if len(suitable_peaks) == 0:
                continue
            p1 = suitable_peaks[0]
            suitable_valleys_after_peak = valleys[valleys > p1]
            if len(suitable_valleys_after_peak) == 0:
                continue
            p2 = suitable_valleys_after_peak[0]
            
            length = p2 - p0
            if l_min <= length <= l_max:
                energy = np.sum(window[p0:p2])
                if energy > max_energy:
                    max_energy = energy
                    best_oscillation = (p0, p1, p2)
                    
        return best_oscillation

    def extract_features(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Виділення просторових ознак v1, v2 та амплітуд для аналізу Jitter/Shimmer."""
        features = []
        amplitudes = []
        current_pos = 0
        
        while current_pos + self.t_analysis < len(signal):
            window_data = signal[current_pos: current_pos + self.t_analysis]
            oscillation_points = self._find_best_oscillation(window_data)
            
            if oscillation_points:
                p0, p1, p2 = oscillation_points
                v1, v2 = float(p1 - p0), float(p2 - p1)
                
                # Амплітуда (різниця між піком та основою)
                amp = float(window_data[p1] - window_data[p0])
                
                if v1 > 0 and v2 > 0:
                    features.append([v1, v2])
                    amplitudes.append(amp)
                current_pos += p2
            else:
                current_pos += self.t_analysis // 4
                
        return np.array(features), np.array(amplitudes)

    def find_density_centers(self, points: np.ndarray) -> Tuple[List[Tuple[float, float]], List]:
        """Алгоритм ковзного вікна для знаходження 3-х центрів максимальної щільності."""
        if len(points) == 0:
            return [], []
            
        candidates = []
        for x in range(0, self.l_max - self.window_l, self.step_size):
            for y in range(0, self.h_max - self.window_h, self.step_size):
                x_min, x_max = x, x + self.window_l
                y_min, y_max = y, y + self.window_h
                
                mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                       (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
                window_points = points[mask]
                n_p = len(window_points)
                
                if n_p < self.min_points:
                    continue
                    
                pairwise_dist_sum = np.sum(pdist(window_points, metric='euclidean'))
                r_w = pairwise_dist_sum / (n_p * n_p + 1e-6)
                
                center_x = x + self.window_l / 2
                center_y = y + self.window_h / 2
                candidates.append((r_w, center_x, center_y))
                
        candidates.sort(key=lambda item: item[0])
        final_centers = []
        
        for cand in candidates:
            if len(final_centers) >= 3:
                break
            r, cx, cy = cand
            is_far = True
            for _, ex, ey in final_centers:
                if np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) < 15:
                    is_far = False
                    break
            if is_far:
                final_centers.append(cand)
                
        # Доповнення до 3-х центрів, якщо не знайдено
        if len(final_centers) < 3 and len(candidates) >= 3:
            needed = 3 - len(final_centers)
            for cand in candidates:
                if needed == 0: break
                if cand not in final_centers:
                    final_centers.append(cand)
                    needed -= 1
                    
        centers_coords = [(c[1], c[2]) for c in final_centers]
        return centers_coords, candidates

    def calculate_distances(self, centers: List[Tuple[float, float]]) -> List[float]:
        """Обчислення відстаней між трьома знайденими центрами щільності."""
        if len(centers) < 3:
            return []
        c1, c2, c3 = centers[0], centers[1], centers[2]
        r12 = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        r13 = np.sqrt((c1[0] - c3[0]) ** 2 + (c1[1] - c3[1]) ** 2)
        r23 = np.sqrt((c2[0] - c3[0]) ** 2 + (c2[1] - c3[1]) ** 2)
        return [r12, r13, r23]

    def analyze_file(self, filepath: str, threshold: float = 61.0) -> dict:
        """Повний цикл аналізу аудіофайлу біонічним методом."""
        rate, raw = wavfile.read(filepath)
        
        # Конвертація в моно, якщо аудіо стерео
        if len(raw.shape) > 1:
            raw = raw[:, 0]
            
        # Нормалізація
        if raw.dtype == np.uint8:
            sig = raw.astype(np.float64) - 128.0
        elif raw.dtype == np.int16:
            sig = (raw.astype(np.float64) / 32768.0) * 128.0
        elif raw.dtype == np.int32:
            sig = (raw.astype(np.float64) / 2147483648.0) * 128.0
        else:
            sig = raw.astype(np.float64)
            
        smoothed = self._smooth_signal(sig)
        points, amplitudes = self.extract_features(smoothed)
        
        centers, _ = self.find_density_centers(points)
        distances = self.calculate_distances(centers)
        
        mean_r = np.mean(distances) if distances else 0.0
        
        # Розрахунок Jitter (варіативність довжини періоду) та Shimmer (варіативність амплітуди)
        jitter = 0.0
        shimmer = 0.0
        variability = 0.0
        
        if len(points) > 5:
            periods = points[:, 0] + points[:, 1]
            jitter = (np.std(periods) / np.mean(periods)) * 100 if np.mean(periods) > 0 else 0
            shimmer = (np.std(amplitudes) / np.mean(amplitudes)) * 100 if np.mean(amplitudes) > 0 else 0
            variability = np.std(points[:, 0]) + np.std(points[:, 1])

        # Комбінований показник
        score = mean_r + (variability * 0.5)
        verdict = "Людина" if score > threshold else "ШІ (AI)"
        
        return {
            'points': points,
            'centers': centers,
            'distances': distances,
            'mean_r': score,
            'jitter': jitter,
            'shimmer': shimmer,
            'verdict': verdict,
            'signal': sig
        }
