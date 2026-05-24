import glob
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from ml_core import MLClassifier

DATASET_DIR = "dataset"
HUMAN_DIR = os.path.join(DATASET_DIR, "human")
AI_DIR = os.path.join(DATASET_DIR, "ai")


def create_mock_dataset_if_empty():
    """Створює структуру та пусті файли, якщо папок не існує (для тестів)"""
    os.makedirs(HUMAN_DIR, exist_ok=True)
    os.makedirs(AI_DIR, exist_ok=True)


def train_model():
    create_mock_dataset_if_empty()

    human_files = glob.glob(os.path.join(HUMAN_DIR, "*.wav"))
    ai_files = glob.glob(os.path.join(AI_DIR, "*.wav"))

    if len(human_files) == 0 or len(ai_files) == 0:
        print("ПОМИЛКА: Для навчання потрібні обидва класи (Людина та ШІ).")
        print(f"Знайдено: {len(human_files)} файлів 'Людина', {len(ai_files)} файлів 'ШІ'")
        return

    if len(human_files) < 2 or len(ai_files) < 2:
        print("ПОМИЛКА: Недостатньо даних для валідації. Потрібно хоча б по 2 семпли кожного класу.")
        return

    print(f"Знайдено: {len(human_files)} файлів 'Людина', {len(ai_files)} файлів 'ШІ'")

    classifier = MLClassifier(model_type='rf')
    X = []
    y = []

    print("Екстракція ознак для класу 'Людина' (Human)...")
    for file in tqdm(human_files):
        try:
            features = classifier.extract_features(file)
            X.append(features)
            y.append(0)  # 0 = Людина
        except Exception as e:
            print(f"Помилка обробки {file}: {e}")

    print("Екстракція ознак для класу 'ШІ' (AI)...")
    for file in tqdm(ai_files):
        try:
            features = classifier.extract_features(file)
            X.append(features)
            y.append(1)  # 1 = ШІ
        except Exception as e:
            print(f"Помилка обробки {file}: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("Не вдалося витягти ознаки з жодного файлу. Навчання скасовано.")
        return

    print("\n[ML] Проведення 5-fold крос-валідації...")
    try:
        # Для крос-валідації використовуємо модель без збереженого стану та пайплайн зі скейлером
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        cv_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
        cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='accuracy')
        print(f"Точність CV (Accuracy): {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    except Exception as e:
        print(f"Крос-валідація неможлива (можливо замало даних): {e}")

    print("\n[ML] Навчання фінальної моделі на всіх даних...")
    classifier.train(X, y)

    print("\n[ML] Оцінка на навчальній вибірці (Classification Report):")
    X_scaled = classifier.scaler.transform(X)
    y_pred = classifier.model.predict(X_scaled)
    print(classification_report(y, y_pred, target_names=["Людина", "ШІ"]))

    print(f"Модель успішно навчена та збережена у папку {classifier.model_dir}/!")


if __name__ == "__main__":
    train_model()
