"""Адаптеры реальных данных под каноническую схему пайплайна.

Пайплайн (preprocessing.py, anomaly_detection.py) работает с DataFrame, у
которого есть колонки ``timestamp``, ``sensor_id``, ``temperature`` и
(необязательно) ``scenario``. Реальные данные лежат в других схемах —
например, ``Т2.csv`` имеет колонки ``time_s`` (секунды от начала) и ``temp_C``
и содержит один датчик без идентификатора. Этот модуль приводит такие файлы к
каноническому виду, чтобы их можно было обработать тем же пайплайном, что и
синтетику.

Запуск::

    python data_adapters.py            # конвертирует Т2.csv -> real_temperature_data.csv
"""
import re

import pandas as pd


CANONICAL_COLUMNS = ["timestamp", "sensor_id", "temperature", "scenario"]


def _parse_glued_body(body):
    """Разбор «склеенного» тела CSV без переносов строк.

    Иногда CSV сохраняется без переносов строк, и тело выглядит как
    ``0,49.001,49.592,50.15…`` (температуры ровно 2 знака после запятой, время —
    целые). Такое регулярное выражение однозначно восстанавливает пары
    ``(time_s, temp_C)``.
    """
    pairs = re.findall(r"(\d+),(\d+\.\d{2})", body)
    if not pairs:
        raise ValueError("Не удалось разобрать тело CSV регулярным выражением.")
    time_s = [int(t) for t, _ in pairs]
    temp_c = [float(v) for _, v in pairs]
    return pd.DataFrame({"time_s": time_s, "temp_C": temp_c})


def load_t2(path="Т2.csv", sensor_id="T-REAL-01",
            start_time="2026-06-06 12:00:00", freq_seconds=1):
    """Загружает реальный файл Т2.csv и приводит к канонической схеме.

    Параметры
    ---------
    path : путь к CSV с колонками ``time_s`` (секунды) и ``temp_C`` (°C).
    sensor_id : идентификатор единственного канала (в файле его нет).
    start_time : строка времени начала наблюдения — метка для ``time_s=0``.
    freq_seconds : шаг между измерениями в секундах (по умолчанию 1 с).

    Возвращает DataFrame с колонками ``timestamp, sensor_id, temperature,
    scenario`` (scenario='user_data'), готовый к ``preprocess_data``.
    """
    raw = pd.read_csv(path)

    # Если файл «склеен» в одну строку (без переносов), read_csv вернёт 1 строку,
    # а temp_C окажется строкой со всеми значениями. Тогда разбираем регуляркой.
    if len(raw) <= 1:
        with open(path, encoding="utf-8-sig") as fh:
            text = fh.read()
        m = re.match(r"\s*time_s\s*,\s*temp_C\s*", text)
        body = text[m.end():] if m else text
        raw = _parse_glued_body(body)

    # На случай альтернативных названий колонок.
    rename = {}
    if "time_s" in raw.columns and "temp_C" in raw.columns:
        pass
    elif len(raw.columns) >= 2:
        rename = {raw.columns[0]: "time_s", raw.columns[1]: "temp_C"}
    else:
        raise ValueError("Ожидались колонки 'time_s' и 'temp_C'.")
    raw = raw.rename(columns=rename)

    raw["time_s"] = pd.to_numeric(raw["time_s"], errors="coerce")
    raw["temp_C"] = pd.to_numeric(raw["temp_C"], errors="coerce")
    raw = raw.dropna(subset=["time_s"]).reset_index(drop=True)

    start = pd.to_datetime(start_time)
    timestamps = start + pd.to_timedelta(raw["time_s"] * freq_seconds, unit="s")

    return pd.DataFrame({
        "timestamp": timestamps,
        "sensor_id": sensor_id,
        "temperature": raw["temp_C"].to_numpy(),
        "scenario": "user_data",
    })[CANONICAL_COLUMNS]


def main(path="Т2.csv", output="real_temperature_data.csv"):
    df = load_t2(path)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Загружено строк: {len(df)}")
    print(f"Датчик: {df['sensor_id'].iloc[0]}")
    print(f"Диапазон времени: {df['timestamp'].iloc[0]} … {df['timestamp'].iloc[-1]}")
    print(f"Температура: {df['temperature'].min():.2f} … {df['temperature'].max():.2f} °C")
    print(f"Сохранено в: {output}")
    print("\nПервые строки:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()