import pandas as pd
import numpy as np


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

input_file = "synthetic_temperature_data.csv"
output_file = "preprocessed_temperature_data.csv"

df = pd.read_csv(input_file)

# Переводим timestamp в формат даты и времени
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Сортируем данные по датчику и времени
df = df.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)


# ============================================================
# 2. БАЗОВАЯ ПРОВЕРКА ДАННЫХ
# ============================================================

print("Первые строки исходных данных:")
print(df.head())

print("\nРазмер исходной таблицы:")
print(df.shape)

print("\nКоличество пропусков по столбцам:")
print(df.isna().sum())


# ============================================================
# 3. ПРИЗНАК: ЕСТЬ ЛИ ПРОПУСК СИГНАЛА
# ============================================================

# Если температуры нет, значит есть потеря сигнала
df["is_missing"] = df["temperature"].isna().astype(int)


# ============================================================
# 4. ЗАПОЛНЕНИЕ ПРОПУСКОВ ДЛЯ РАСЧЁТА ПРИЗНАКОВ
# ============================================================

# Для расчёта скользящих признаков пропуски нужно временно заполнить.
# Заполняем пропуски предыдущим значением внутри каждого датчика.
df["temperature_filled"] = (
    df.groupby("sensor_id")["temperature"]
    .ffill()
    .bfill()
)


# ============================================================
# 5. СКОЛЬЗЯЩИЕ ПРИЗНАКИ
# ============================================================

window_size = 10
# окно 10 минут, так как шаг измерения 1 минута

# Скользящее среднее
df["rolling_mean"] = (
    df.groupby("sensor_id")["temperature_filled"]
    .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
)

# Скользящее стандартное отклонение
df["rolling_std"] = (
    df.groupby("sensor_id")["temperature_filled"]
    .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
)

# В первых строках std может быть NaN, заменяем на 0
df["rolling_std"] = df["rolling_std"].fillna(0)


# ============================================================
# 6. СКОРОСТЬ ИЗМЕНЕНИЯ ТЕМПЕРАТУРЫ
# ============================================================

# Разница температуры между текущей и предыдущей минутой
df["temp_diff"] = (
    df.groupby("sensor_id")["temperature_filled"]
    .diff()
    .fillna(0)
)

# Модуль скорости изменения температуры
df["abs_temp_diff"] = df["temp_diff"].abs()


# ============================================================
# 7. Z-SCORE
# ============================================================

# z-score показывает, насколько текущее значение отличается от среднего
# в пределах конкретного датчика

sensor_mean = df.groupby("sensor_id")["temperature_filled"].transform("mean")
sensor_std = df.groupby("sensor_id")["temperature_filled"].transform("std")

df["z_score"] = (df["temperature_filled"] - sensor_mean) / sensor_std

# На случай деления на ноль
df["z_score"] = df["z_score"].replace([np.inf, -np.inf], 0).fillna(0)

# Модуль z-score удобнее для поиска сильных отклонений
df["abs_z_score"] = df["z_score"].abs()


# ============================================================
# 8. ПРИЗНАК ЗАВИСАНИЯ ДАТЧИКА
# ============================================================

# Если температура почти не меняется несколько минут подряд,
# датчик может быть зависшим.

stuck_window = 15
# проверяем окно 15 минут

stuck_threshold = 0.05
# если изменение меньше 0.05 °C, считаем, что значение почти не менялось

df["small_change"] = (df["abs_temp_diff"] < stuck_threshold).astype(int)

df["stuck_score"] = (
    df.groupby("sensor_id")["small_change"]
    .transform(lambda x: x.rolling(window=stuck_window, min_periods=1).sum())
)

# Если почти все 15 минут изменение было очень маленьким,
# ставим признак зависания
df["is_stuck"] = (df["stuck_score"] >= 14).astype(int)


# ============================================================
# 9. ОТКЛОНЕНИЕ ОТ СОСЕДНИХ ДАТЧИКОВ
# ============================================================

# Для каждого момента времени считаем среднюю температуру по всем датчикам.
# Потом смотрим, насколько конкретный датчик отличается от общей картины.

df["mean_temp_all_sensors"] = (
    df.groupby("timestamp")["temperature_filled"]
    .transform("mean")
)

df["diff_from_group_mean"] = (
    df["temperature_filled"] - df["mean_temp_all_sensors"]
)

df["abs_diff_from_group_mean"] = df["diff_from_group_mean"].abs()


# ============================================================
# 10. ПРОСТАЯ ПРЕДВАРИТЕЛЬНАЯ МЕТКА ПОДОЗРИТЕЛЬНОСТИ
# ============================================================

# Это пока не финальная модель, а грубая техническая метка:
# 1 — точка выглядит подозрительно
# 0 — точка выглядит нормально

df["preliminary_warning"] = 0

df.loc[df["is_missing"] == 1, "preliminary_warning"] = 1
df.loc[df["abs_temp_diff"] > 5, "preliminary_warning"] = 1
df.loc[df["abs_z_score"] > 3, "preliminary_warning"] = 1
df.loc[df["is_stuck"] == 1, "preliminary_warning"] = 1
df.loc[df["abs_diff_from_group_mean"] > 8, "preliminary_warning"] = 1


# ============================================================
# 11. СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ============================================================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nПредобработка завершена.")
print(f"Файл сохранён: {output_file}")

print("\nПервые строки обработанных данных:")
print(df.head())

print("\nКоличество предварительно подозрительных точек:")
print(df["preliminary_warning"].sum())