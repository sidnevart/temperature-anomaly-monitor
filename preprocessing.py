import pandas as pd
import numpy as np


# Параметры предобработки (вынесены из тела функции — единая точка настройки).
# Подобраны так, чтобы работать и на синтетике (несколько датчиков), и на реальных
# одноканальных трендовых данных (Т2.csv). См. tests/ и issue по объединению mod_AI_2.
ROLLING_WINDOW = 10        # окно скользящих признаков, точек
STUCK_MIN_RUN = 10        # сколько подряд одинаковых значений считать зависанием
STUCK_ABS_TOL = 1e-6       # точное равенство для зависания (ступени квантования не ловим)


def preprocess_data(df, rolling_window=ROLLING_WINDOW):
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
    window_size = rolling_window

    df["rolling_mean"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    df["rolling_std"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    )

    # 0/std на ранних точках (окно не накопилось) -> маленькая константа, чтобы
    # z-score не взрывался в inf.
    df["rolling_std"] = df["rolling_std"].fillna(0).replace(0, 1e-6)

    # Скорость изменения температуры
    df["temp_diff"] = (
        df.groupby("sensor_id")["temperature_filled"]
        .diff()
        .fillna(0)
    )

    df["abs_temp_diff"] = df["temp_diff"].abs()

    # Z-score: скользящий (локальный) — отклонение от собственного rolling_mean.
    # Раньше считался по глобальному среднему всего датчика: на растущем реальном
    # сигнале (Т2.csv: 49->109 °C) весь верхний полубас уходил в аномалию. Скользящий
    # z-score адаптируется к тренду и ловит локальные выбросы, а не сам тренд.
    df["z_score"] = (df["temperature_filled"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], 0).fillna(0)
    df["abs_z_score"] = df["z_score"].abs()

    # Признак зависания датчика: точное равенство подряд STUCK_MIN_RUN значений.
    # Пороговый критерий (|Δt|<0.05 за окно 15) ловил ступени квантования АЦП
    # реальных данных как зависание — тысячи ложных тревог. Точное равенство
    # срабатывает только на по-настоящему застывшем сигнале.
    is_stuck = np.zeros(len(df), dtype=int)
    for _sid, idx in df.groupby("sensor_id").groups.items():
        vals = df.loc[idx, "temperature_filled"].to_numpy()
        run = 0
        prev = None
        for j, i in enumerate(idx):
            v = vals[j]
            if (
                prev is not None
                and not (np.isnan(v) or np.isnan(prev))
                and abs(v - prev) < STUCK_ABS_TOL
            ):
                run += 1
            else:
                run = 0
            if run >= STUCK_MIN_RUN:
                is_stuck[i] = 1
            prev = v
    df["is_stuck"] = is_stuck

    # Совместимость со старыми выходами: small_change/stuck_score сохранены
    # (используются в отчётах), но is_stuck теперь считается точным равенством.
    df["small_change"] = (df["abs_temp_diff"] < 0.05).astype(int)
    df["stuck_score"] = (
        df.groupby("sensor_id")["small_change"]
        .transform(lambda x: x.rolling(window=15, min_periods=1).sum())
    )

    # Отклонение от группы. Для одного датчика кросс-сенсорное среднее равно самому
    # значению (отклонение 0, правило мёртвое). Поэтому при одном датчике берём
    # отклонение от собственного rolling_mean (подход mod_AI_2) — оно осмысленно.
    if df["sensor_id"].nunique() > 1:
        df["mean_temp_all_sensors"] = (
            df.groupby("timestamp")["temperature_filled"]
            .transform("mean")
        )
        df["diff_from_group_mean"] = (
            df["temperature_filled"] - df["mean_temp_all_sensors"]
        )
    else:
        df["mean_temp_all_sensors"] = df["rolling_mean"]
        df["diff_from_group_mean"] = df["temperature_filled"] - df["rolling_mean"]

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
