"""Обучение модели Isolation Forest на нормальных данных и её сохранение.

Раньше Isolation Forest обучался прямо на тех данных, которые оценивал
(`anomaly_detection.detect_anomalies` делал `fit_predict` на всём DataFrame).
Это data leakage: модель «видела» аномалии при обучении и не училась отличать
штатный режим от настоящих отклонений. Кроме того, модель нигде не
сохранялась — каждый запуск обучал её заново.

Этот скрипт исправляет оба недостатка:
- обучает StandardScaler + Isolation Forest ТОЛЬКО на строках со сценарием
  `normal` (модель учит штатный режим, а аномалии становятся для неё
  «нетипичными»);
- сохраняет артефакты (`models/scaler.joblib`, `models/iforest.joblib`,
  `models/model_meta.json`), которые потом загружает `detect_anomalies`;
- печатает отчёт precision/recall/F1 против истинных сценариев.

Запуск::

    python preprocessing.py     # если ещё нет preprocessed_temperature_data.csv
    python train_model.py        # обучить и сохранить модель
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Список признаков должен совпадать с тем, что использует
# anomaly_detection.detect_anomalies при работе с моделью. Дублируется сюда
# осознанно, чтобы train_model.py не depended от модуля детекции.
FEATURE_COLUMNS = [
    "temperature_filled",
    "rolling_mean",
    "rolling_std",
    "temp_diff",
    "abs_temp_diff",
    "abs_z_score",
    "is_missing",
    "is_stuck",
    "abs_diff_from_group_mean",
    "rolling_temp_diff_mean_20",
]

MODEL_DIR = "models"
DEFAULT_INPUT = "preprocessed_temperature_data.csv"
DEFAULT_CONTAMINATION = 0.04
RANDOM_STATE = 42
N_ESTIMATORS = 200


def prepare_features(df):
    """Добавляет rolling_temp_diff_mean_20 (как в detect_anomalies) и возвращает X.

    preprocessed_temperature_data.csv содержит все признаки кроме
    rolling_temp_diff_mean_20 — он считается на этапе детекции. Чтобы обучать
    модель на тех же признаках, что используются при инференсе, считаем его
    здесь идентично.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)
    df["rolling_temp_diff_mean_20"] = (
        df.groupby("sensor_id")["temp_diff"]
        .transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    )
    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, X


def _normal_mask(df):
    """Маска строк штатного режима для обучения."""
    if "scenario" in df.columns:
        mask = df["scenario"] == "normal"
        if mask.sum() == 0:
            warnings.warn(
                "Нет строк со scenario=='normal' — обучаю на всех данных "
                "(риск data leakage)."
            )
            return pd.Series(True, index=df.index), False
        return mask, True
    warnings.warn(
        "Нет колонки scenario — обучаю на всех данных (риск data leakage). "
        "Запустите Data.py / preprocessing.py, чтобы получить разметку."
    )
    return pd.Series(True, index=df.index), False


def train(df, contamination=DEFAULT_CONTAMINATION, random_state=RANDOM_STATE):
    """Обучает scaler+IsolationForest на штатном режиме.

    Возвращает (scaler, model, info) где info — словарь с числом train-строк и
    флагом, обучалась ли модель именно на `normal`.
    """
    df, X = prepare_features(df)
    train_mask, used_normal = _normal_mask(df)
    X_train = X.loc[train_mask]

    scaler = StandardScaler().fit(X_train)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
    ).fit(scaler.transform(X_train))

    info = {"train_rows": int(train_mask.sum()), "trained_on_normal": used_normal}
    return scaler, model, info


def evaluate(df, scaler, model):
    """Считает precision/recall/F1 модели против scenario!='normal' и per-scenario recall."""
    df, X = prepare_features(df)
    pred = (model.predict(scaler.transform(X)) == -1).astype(int)
    report = {"iforest_anomalies": int(pred.sum())}
    if "scenario" not in df.columns:
        return report
    gt = (df["scenario"] != "normal").astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    report.update({
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })
    per_scenario = {}
    for scenario, sub in df.groupby("scenario"):
        sp = pred[sub.index]
        total = int(len(sub))
        detected = int(sp.sum())
        per_scenario[scenario] = {
            "total": total,
            "detected": detected,
            "recall": round(detected / total, 3) if total else 0.0,
        }
    report["per_scenario"] = per_scenario
    return report


def save_model(scaler, model, info, model_dir=MODEL_DIR, contamination=DEFAULT_CONTAMINATION):
    os.makedirs(model_dir, exist_ok=True)
    dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    dump(model, os.path.join(model_dir, "iforest.joblib"))
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "train_rows": info["train_rows"],
        "trained_on_normal": info["trained_on_normal"],
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def main(input_file=DEFAULT_INPUT, contamination=DEFAULT_CONTAMINATION):
    df = pd.read_csv(input_file)
    scaler, model, info = train(df, contamination=contamination)
    save_model(scaler, model, info, contamination=contamination)
    report = evaluate(df, scaler, model)
    print("Модель обучена. train_rows =", info["train_rows"],
          "| trained_on_normal =", info["trained_on_normal"])
    print("Артефакты сохранены в:", MODEL_DIR)
    print("Отчёт по качеству (Isolation Forest):")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()