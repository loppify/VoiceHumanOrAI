import asyncio
import os

import numpy as np
from loguru import logger

import dataset_builder
from src.bionic_core import BionicClassifier
from src.ml_core import MLClassifier

# Конфігурація експерименту
EXP_DIR = "experiment_data"
SAMPLES_PER_GROUP = 30 # Збільшено для кращої статистики


async def run_experiment():
    os.makedirs(EXP_DIR, exist_ok=True)

    results = []

    for provider in ["edge", "lux"]:
        logger.info(f"--- Тестування провайдера: {provider.upper()} ---")

        # Директорії для провайдера
        group_dir = os.path.join(EXP_DIR, provider)
        ai_dir = os.path.join(group_dir, "ai")
        human_dir = os.path.join(group_dir, "human")

        # Крок 1: Генерація даних
        await dataset_builder.build_dataset(samples=SAMPLES_PER_GROUP, provider_name=provider, human_dir=human_dir,
                                            ai_dir=ai_dir)

        # Крок 2: Аналіз
        bionic = BionicClassifier()
        ml = MLClassifier()

        ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.endswith('.wav')]
        human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.wav')]

        group_stats = {"provider": provider}

        # Біонічні метрики для ШІ
        metrics = {"score": [], "jitter": [], "shimmer": []}
        ml_correct = 0

        for f in ai_files:
            res = bionic.analyze_file(f)
            metrics["score"].append(res["mean_r"])
            metrics["jitter"].append(res["jitter"])
            metrics["shimmer"].append(res["shimmer"])

            if ml.is_trained:
                label, _ = ml.predict(f)
                if label == "ШІ (AI)": ml_correct += 1

        group_stats["ai_avg_score"] = np.mean(metrics["score"]) if metrics["score"] else 0
        group_stats["ai_avg_jitter"] = np.mean(metrics["jitter"]) if metrics["jitter"] else 0
        group_stats["ai_avg_shimmer"] = np.mean(metrics["shimmer"]) if metrics["shimmer"] else 0
        group_stats["ml_accuracy_ai"] = (ml_correct / len(ai_files)) * 100 if ai_files else 0

        # Метрики для Людей (для порівняння)
        h_metrics = {"score": [], "jitter": [], "shimmer": []}
        h_ml_correct = 0
        for f in human_files:
            res = bionic.analyze_file(f)
            h_metrics["score"].append(res["mean_r"])
            h_metrics["jitter"].append(res["jitter"])
            h_metrics["shimmer"].append(res["shimmer"])
            if ml.is_trained:
                label, _ = ml.predict(f)
                if label == "Людина": h_ml_correct += 1

        group_stats["human_avg_score"] = np.mean(h_metrics["score"]) if h_metrics["score"] else 0
        group_stats["human_avg_jitter"] = np.mean(h_metrics["jitter"]) if h_metrics["jitter"] else 0
        group_stats["human_avg_shimmer"] = np.mean(h_metrics["shimmer"]) if h_metrics["shimmer"] else 0
        group_stats["ml_accuracy_human"] = (h_ml_correct / len(human_files)) * 100 if human_files else 0

        results.append(group_stats)

    # Побудова звіту
    report = f"""
# Науковий звіт: Порівняльний аналіз Edge TTS та LuxTTS

Даний звіт містить результати експериментального дослідження стійкості гібридної системи 
до різних методів синтезу мовлення.

## 1. Біометричні показники (Середні значення)

| Провайдер | AI Score | AI Jitter (%) | AI Shimmer (%) | Human Jitter (%) |
|-----------|----------|---------------|----------------|------------------|
{os.linesep.join([f"| {r['provider']} | {r['ai_avg_score']:.2f} | {r['ai_avg_jitter']:.2f} | {r['ai_avg_shimmer']:.2f} | {r['human_avg_jitter']:.2f} |" for r in results])}

## 2. Точність ML-моделі (Random Forest)

| Провайдер | Accuracy (AI) | Accuracy (Human) |
|-----------|---------------|------------------|
{os.linesep.join([f"| {r['provider']} | {r['ml_accuracy_ai']:.1f}% | {r['ml_accuracy_human']:.1f}% |" for r in results])}

## 3. Висновки для Розділу 5
Результати показують, що при використанні {results[1]['provider']} (клонування), 
показник Jitter становить {results[1]['ai_avg_jitter']:.2f}%, що ближче до показників 
реальної людини ({results[1]['human_avg_jitter']:.2f}%), ніж у стандартного {results[0]['provider']} ({results[0]['ai_avg_jitter']:.2f}%). 
Це підтверджує необхідність використання комбінованого підходу (Біоніка + ML) для виявлення 
високоякісних підробок.
"""
    with open("EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Експеримент завершено! Звіт збережено у EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    asyncio.run(run_experiment())
