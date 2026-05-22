import os
from typing import Tuple, List, Optional

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLClassifier:
    """
    Класичний ML метод класифікації мовного сигналу.
    Використовує витяг ознак (MFCC, Chroma, Rolloff) та Random Forest.
    """

    def __init__(self, model_dir: str = "models", model_type: str = 'rf'):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
        self.scaler_path = os.path.join(self.model_dir, f"{model_type}_scaler.pkl")
        self.model_type = model_type

        os.makedirs(self.model_dir, exist_ok=True)

        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self._load_resources()

    def _load_resources(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
            except Exception as e:
                print(f"Помилка завантаження ресурсів ML: {e}")

    def extract_features(self, filepath: str) -> np.ndarray:
        y, sr = librosa.load(filepath, sr=None)

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

        return np.hstack([mfccs, chroma, rolloff])

    def train(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        self.is_trained = True

    def predict(self, filepath: str) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Модель не навчена!")
            
        features = self.extract_features(filepath)
        X_scaled = self.scaler.transform([features])
        
        pred = self.model.predict(X_scaled)[0]
        prob = np.max(self.model.predict_proba(X_scaled)[0])
        
        verdict = "Людина" if pred == 0 else "ШІ (AI)"
        return verdict, float(prob)

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return []

        names = [f"MFCC_{i + 1}" for i in range(20)] + \
                [f"Chroma_{i + 1}" for i in range(12)] + \
                ["Spectral Rolloff"]
        
        importances = self.model.feature_importances_
        return sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
