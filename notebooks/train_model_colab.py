"""Зеркало Colab-ноутбука train_model_colab.ipynb для локального запуска.

Colab-ноутбук (notebooks/train_model_colab.ipynb) делает то же самое по шагам с
markdown-пояснениями для новичков. Этот файл — чтобы запустить тренировку без
Colab:

    python notebooks/train_model_colab.py

Шаги:
1. Получить данные (сгенерировать синтетику через Data.py ИЛИ загрузить свой CSV).
2. Предобработать (preprocessing.preprocess_data).
3. Обучить Isolation Forest на штатном режиме (train_model.train).
4. Оценить precision/recall/F1 против истинных сценариев (если есть разметка).
5. Сохранить модель (models/scaler.joblib, iforest.joblib) для детекции.
"""
import os
import sys

# Чтобы импортировать модули проекта из папки notebooks/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from preprocessing import preprocess_data  # noqa: E402
import train_model  # noqa: E402


def get_synthetic_data():
    """Генерирует синтетический набор с разметкой сценариев (как Data.py).

    Data.py при импорте пишет CSV и показывает окно matplotlib — в Colab/ноутбуке
    это неудобно, поэтому здесь компактный генератор без побочных эффектов.
    Полная версия — в tests/conftest.py::make_synth_df.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from conftest import make_synth_df
    return make_synth_df()


def main():
    # 1. Данные. Варианты:
    #    а) синтетика с разметкой (для оценки качества);
    #    б) свой CSV: raw = pd.read_csv("ваш_файл.csv") с колонками
    #       timestamp, sensor_id, temperature.
    raw = get_synthetic_data()
    print("Данные:", raw.shape, "| сценарии:", raw["scenario"].unique().tolist())

    # 2. Предобработка.
    df = preprocess_data(raw)
    print("Предобработано:", df.shape)

    # 3. Обучение на штатном режиме (scenario == 'normal') — без data leakage.
    scaler, model, info = train_model.train(df, contamination=0.04)
    print(f"Обучено на {info['train_rows']} строк normal | trained_on_normal={info['trained_on_normal']}")

    # 4. Оценка качества против истинных сценариев (если есть колонка scenario).
    report = train_model.evaluate(df, scaler, model)
    print("Отчёт (Isolation Forest):")
    if "precision" in report:
        print(f"  precision={report['precision']} recall={report['recall']} f1={report['f1']}")
        print("  per-scenario recall:")
        for scen, m in report["per_scenario"].items():
            print(f"    {scen:20} recall={m['recall']}")
    else:
        print("  (нет колонки scenario — качество не оценить, модель сохранена)")

    # 5. Сохранение модели.
    train_model.save_model(scaler, model, info, contamination=0.04)
    print("Сохранено в models/ (scaler.joblib, iforest.joblib, model_meta.json)")
    print("Теперь anomaly_detection.detect_anomalies будет использовать эту модель.")


if __name__ == "__main__":
    main()