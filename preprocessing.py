import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Предобработка температурных данных.

    На вход получает DataFrame с колонками:
    - timestamp
    - sensor_id
    - temperature

    На выход возвращает DataFrame с дополнительными признаками:
    - is_missing
    - temperature_filled
    - rolling_mean
    - rolling_std
    - temp_diff
    - abs_temp_diff
    - z_score
    - abs_z_score
    - is_stuck
    - abs_diff_from_group_mean
    - preliminary_warning
    """

    df = df.copy()

    # Если нет колонки scenario, добавляем ее для пользовательских данных
    if "scenario" not in df.columns:
        df["scenario"] = "user_data"

    # Переводим timestamp в формат даты и времени
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Температуру принудительно переводим в число
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    # Сортируем данные по датчику и времени
    df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)

    # Признак пропуска сигнала
    df["is_missing"] = df["temperature"].isna().astype(int)

    # Заполнение пропусков внутри каждого датчика
    df["temperature_filled"] = (
        df.groupby("sensor_id")["temperature"]
        .ffill()
        .bfill()
    )

    # Если у какого-то датчика все значения пустые, после ffill/bfill могут остаться NaN
    df["temperature_filled"] = df["temperature_filled"].fillna(0)

    # Скользящие признаки
    window_size = 10

    df["rolling_mean"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    df["rolling_std"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    )

    df["rolling_std"] = df["rolling_std"].fillna(0)

    # Скорость изменения температуры
    df["temp_diff"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .diff()
        .fillna(0)
    )

    df["abs_temp_diff"] = df["temp_diff"].abs()

    # Z-score внутри каждого датчика
    sensor_mean = df.groupby("sensor_id")["temperature_filled"].transform("mean")
    sensor_std = df.groupby("sensor_id")["temperature_filled"].transform("std")

    df["z_score"] = (df["temperature_filled"] - sensor_mean) / sensor_std
    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], 0).fillna(0)
    df["abs_z_score"] = df["z_score"].abs()

    # Признак зависания датчика
    stuck_window = 15
    stuck_threshold = 0.05

    df["small_change"] = (df["abs_temp_diff"] < stuck_threshold).astype(int)

    df["stuck_score"] = (
        df.groupby("sensor_id")["small_change"]
        .transform(lambda x: x.rolling(window=stuck_window, min_periods=1).sum())
    )

    df["is_stuck"] = (df["stuck_score"] >= 14).astype(int)

    # Отклонение от группы датчиков
    df["mean_temp_all_sensors"] = (
        df.groupby("timestamp")["temperature_filled"]
        .transform("mean")
    )

    df["diff_from_group_mean"] = (
        df["temperature_filled"] - df["mean_temp_all_sensors"]
    )

    df["abs_diff_from_group_mean"] = df["diff_from_group_mean"].abs()

    # Предварительная техническая метка подозрительности
    df["preliminary_warning"] = 0

    df.loc[df["is_missing"] == 1, "preliminary_warning"] = 1
    df.loc[df["abs_temp_diff"] > 5, "preliminary_warning"] = 1
    df.loc[df["abs_z_score"] > 3, "preliminary_warning"] = 1
    df.loc[df["is_stuck"] == 1, "preliminary_warning"] = 1
    df.loc[df["abs_diff_from_group_mean"] > 8, "preliminary_warning"] = 1

    return df


if __name__ == "__main__":
    input_file = "synthetic_temperature_data.csv"
    output_file = "preprocessed_temperature_data.csv"

    df = pd.read_csv(input_file)
    processed_df = preprocess_data(df)

    processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nПредобработка завершена.")
    print(f"Файл сохранён: {output_file}")
    print("\nПервые строки обработанных данных:")
    print(processed_df.head())
    print("\nКоличество предварительно подозрительных точек:")
    print(processed_df["preliminary_warning"].sum())
