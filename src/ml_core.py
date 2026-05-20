import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class MLClassifier:
    """
    Класичний ML-підхід для бінарної класифікації мовного сигналу "Людина-ШІ".
    Використовує MFCCs, Chroma STFT, Spectral Rolloff та класифікатор (RandomForest / SVM).
    """
    def __init__(self, model_type='rf', model_dir='models'):
        self.model_type = model_type
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
        self.scaler_path = os.path.join(model_dir, f"{model_type}_scaler.pkl")
        
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_model()

    def extract_features(self, filepath: str) -> np.ndarray:
        """
        Екстракція ознак за допомогою librosa.
        Витягуємо MFCC (20), Chroma та Spectral Rolloff, усереднюємо їх по часу.
        """
        # Завантаження аудіо (librosa автоматично перетворює в mono і ресемплить)
        y, sr = librosa.load(filepath, sr=22050, mono=True)
        
        # Обробка коротких файлів або тиші
        if len(y) == 0:
            return np.zeros(20 + 12 + 1) # mfcc(20) + chroma(12) + rolloff(1)
            
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff, axis=1)
        
        # Об'єднання в один вектор
        features = np.hstack([mfcc_mean, chroma_mean, rolloff_mean])
        return features

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Навчання моделі на підготовлених ознаках X та мітках y.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.save_model()

    def save_model(self):
        """Збереження навченої моделі та скейлера на диск."""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
    def load_model(self):
        """Завантаження моделі з диску, якщо вона існує."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True

    def predict(self, filepath: str) -> Tuple[str, float]:
        """
        Аналіз аудіофайлу та класифікація.
        Повертає мітку ('Людина' або 'ШІ (AI)') та ймовірність.
        """
        # Спроба завантажити модель, якщо вона була навчена після запуску сервера
        if not self.is_trained:
            self.load_model()
            
        if not self.is_trained:
            raise ValueError("Модель ще не навчена! Запустіть скрипт навчання (src/train_model.py).")
            
        features = self.extract_features(filepath)
        X_scaled = self.scaler.transform([features])
        
        pred = self.model.predict(X_scaled)[0]
        prob = np.max(self.model.predict_proba(X_scaled)[0])
        
        verdict = "Людина" if pred == 0 else "ШІ (AI)"
        return verdict, prob

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Повертає список назв ознак та їх важливість для моделі."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return []
            
        # Формуємо назви ознак
        feature_names = [f"MFCC_{i+1}" for i in range(20)] + \
                        [f"Chroma_{i+1}" for i in range(12)] + \
                        ["Spectral Rolloff"]
        
        importances = self.model.feature_importances_
        importance_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return importance_list
