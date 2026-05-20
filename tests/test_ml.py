import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ml_core import MLClassifier

# Тимчасова директорія для тестової моделі
TEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'test_models')

@pytest.fixture
def clean_test_models():
    if not os.path.exists(TEST_MODEL_DIR):
        os.makedirs(TEST_MODEL_DIR)
    yield
    # Очистка після тесту
    for f in os.listdir(TEST_MODEL_DIR):
        os.remove(os.path.join(TEST_MODEL_DIR, f))
    os.rmdir(TEST_MODEL_DIR)

def test_ml_initialization(clean_test_models):
    classifier = MLClassifier(model_type='rf', model_dir=TEST_MODEL_DIR)
    assert classifier.model_type == 'rf'
    assert not classifier.is_trained

def test_ml_train_predict(clean_test_models):
    classifier = MLClassifier(model_type='rf', model_dir=TEST_MODEL_DIR)
    
    # Створюємо фейкові дані (2 класи, 10 зразків, 33 фічі)
    X = np.random.rand(10, 33)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    classifier.train(X, y)
    assert classifier.is_trained
    assert os.path.exists(classifier.model_path)
    assert os.path.exists(classifier.scaler_path)
    
    # Симулюємо виклик передбачення, підмінивши extract_features для тесту
    # Зазвичай librosa.load вимагатиме справжній файл, тому мокаємо цю частину:
    def mock_extract(filepath):
        return np.random.rand(33)
    
    classifier.extract_features = mock_extract
    
    label, prob = classifier.predict("dummy.wav")
    assert label in ["Людина", "ШІ (AI)"]
    assert 0.0 <= prob <= 1.0

def test_ml_predict_without_training(clean_test_models):
    classifier = MLClassifier(model_type='rf', model_dir=TEST_MODEL_DIR)
    with pytest.raises(ValueError, match="Модель ще не навчена"):
        classifier.predict("dummy.wav")
