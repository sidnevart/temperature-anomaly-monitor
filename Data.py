import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. НАСТРОЙКИ ГЕНЕРАЦИИ
# =========================

np.random.seed(42)  # чтобы при каждом запуске получались одинаковые данные

num_sensors = 6          # количество датчиков
duration_minutes = 8 * 60  # 8 часов наблюдения
time_step = 1            # шаг измерения: 1 минута

base_temperature = 70    # средняя нормальная температура, °C
noise_level = 1.0        # уровень случайного шума, °C

start_time = "2026-06-06 12:00"


# =========================
# 2. СОЗДАЁМ ВРЕМЕННУЮ ШКАЛУ
# =========================

timestamps = pd.date_range(
    start=start_time,
    periods=duration_minutes,
    freq=f"{time_step}min"
)

time = np.arange(duration_minutes)


# =========================
# 3. ФУНКЦИЯ НОРМАЛЬНОГО РЕЖИМА
# =========================

def generate_normal_temperature(sensor_shift=0):
    """
    Генерирует нормальный температурный сигнал:
    базовая температура + плавные колебания + случайный шум.
    """

    slow_fluctuation = 2 * np.sin(2 * np.pi * time / 180)
    # плавное колебание с периодом примерно 3 часа

    random_noise = np.random.normal(0, noise_level, duration_minutes)
    # случайный шум датчика

    temperature = base_temperature + sensor_shift + slow_fluctuation + random_noise

    return temperature


# =========================
# 4. СОЗДАЁМ ДАННЫЕ ДЛЯ ВСЕХ ДАТЧИКОВ
# =========================

data = []

for sensor_number in range(1, num_sensors + 1):
    sensor_id = f"T-{sensor_number:02d}"

    # небольшой индивидуальный сдвиг каждого датчика
    sensor_shift = np.random.uniform(-1.5, 1.5)

    temperature = generate_normal_temperature(sensor_shift)

    # по умолчанию все точки считаются нормальными
    scenario = np.array(["normal"] * duration_minutes, dtype=object)

    # =========================
    # 5. ДОБАВЛЯЕМ АНОМАЛИИ
    # =========================

    # T-02: резкий скачок температуры
    if sensor_id == "T-02":
        start = 180
        end = 200
        temperature[start:end] += 15
        scenario[start:end] = "sharp_jump"

    # T-03: медленный перегрев
    if sensor_id == "T-03":
        start = 240
        end = 360
        growth = np.linspace(0, 12, end - start)
        temperature[start:end] += growth
        scenario[start:end] = "slow_overheating"

    # T-04: дрейф датчика
    if sensor_id == "T-04":
        start = 250
        drift = np.linspace(0, 10, duration_minutes - start)
        temperature[start:] += drift
        scenario[start:] = "sensor_drift"

    # T-05: зависание датчика
    if sensor_id == "T-05":
        start = 300
        end = 360
        stuck_value = temperature[start]
        temperature[start:end] = stuck_value
        scenario[start:end] = "stuck_sensor"

    # T-06: шум и потеря сигнала
    if sensor_id == "T-06":
        start = 120
        end = 170

        strong_noise = np.random.normal(0, 5, end - start)
        temperature[start:end] += strong_noise
        scenario[start:end] = "high_noise"

        # часть значений делаем пропусками
        missing_indices = np.random.choice(
            np.arange(start, end),
            size=10,
            replace=False
        )
        temperature[missing_indices] = np.nan
        scenario[missing_indices] = "signal_loss"

    # T-02 и T-03: коррелированная аномалия в конце
    if sensor_id in ["T-02", "T-03"]:
        start = 400
        end = 450
        correlated_growth = np.linspace(0, 8, end - start)
        temperature[start:end] += correlated_growth
        scenario[start:end] = "correlated_growth"

    # сохраняем данные в общий список
    for i in range(duration_minutes):
        data.append({
            "timestamp": timestamps[i],
            "sensor_id": sensor_id,
            "temperature": temperature[i],
            "scenario": scenario[i]
        })


# =========================
# 6. СОЗДАЁМ DATAFRAME
# =========================

df = pd.DataFrame(data)

print(df.head())
print()
print("Размер таблицы:", df.shape)


# =========================
# 7. СОХРАНЯЕМ В CSV
# =========================

df.to_csv("synthetic_temperature_data.csv", index=False, encoding="utf-8-sig")

print("Файл synthetic_temperature_data.csv успешно создан")


# =========================
# 8. СТРОИМ ГРАФИК
# =========================

plt.figure(figsize=(14, 7))

for sensor_id in df["sensor_id"].unique():
    sensor_data = df[df["sensor_id"] == sensor_id]

    plt.plot(
        sensor_data["timestamp"],
        sensor_data["temperature"],
        label=sensor_id
    )

plt.xlabel("Время")
plt.ylabel("Температура, °C")
plt.title("Синтетические температурные данные по датчикам")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()