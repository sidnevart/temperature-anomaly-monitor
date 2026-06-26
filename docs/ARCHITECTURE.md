# Архитектура

Проект — линейный пайплайн над DataFrame `timestamp, sensor_id, temperature`.
Каждый модуль берёт DataFrame, обогащает и отдаёт дальше.

## Модули

```
Data.py                     генератор синтетики (timestamp, sensor_id, temperature, scenario)
  │  пишет synthetic_temperature_data.csv
  ▼
preprocessing.py            preprocess_data(df) -> df + признаки
  │  признаки: is_missing, temperature_filled, rolling_mean, rolling_std,
  │           temp_diff, abs_temp_diff, z_score, abs_z_score, is_stuck,
  │           abs_diff_from_group_mean, preliminary_warning
  ▼
train_model.py              train(df) -> scaler + IsolationForest (обучение на scenario=='normal')
  │  save_model() -> models/scaler.joblib, iforest.joblib, model_meta.json
  ▼
anomaly_detection.py        detect_anomalies(df, model_dir) -> df + alarm_log
  │  1. правила (RULE_PARAMS) -> rule_anomaly, rule_event_type, risk, recommendation
  │  2. Isolation Forest (из models/ или fit на лету) -> iforest_anomaly, anomaly_score_norm
  │  3. final_anomaly = rule_anomaly | iforest_anomaly
  │  4. журнал тревог (только final_anomaly==1)
  ▼
app.py                      Streamlit: обзор, температурный тренд, anomaly score, журнал тревог
```

## Каноническая схема данных

Все модули работают с одним форматом:

| Колонка | Тип | Описание |
|---|---|---|
| `timestamp` | datetime | время измерения |
| `sensor_id` | str | идентификатор датчика |
| `temperature` | float | температура, °C (допускаются NaN — потеря сигнала) |
| `scenario` | str | метка режима (опционально; для пользовательских данных — `user_data`) |

Реальные данные `Т2.csv` (`time_s`, `temp_C`, один датчик) приводятся к этой
схеме адаптером `data_adapters.load_t2()`.

## Два слоя обнаружения

- **Правила** — детерминированные, объяснимые оператору. Пороги в
  `RULE_PARAMS` (в начале `anomaly_detection.py`).
- **Isolation Forest** — обучается на штатном режиме (`scenario == 'normal'`),
  сохраняется в `models/`. Если модели нет — обучается на лету (запасной режим,
  с data leakage и более низкой точностью).

`final_anomaly` = срабатывание хотя бы одного слоя. Если аномалию нашёл только
ИИ (без правил), она помечается «Нетипичное поведение по ИИ-модели».

## Ключевые параметры

| Где | Параметр | По умолчанию |
|---|---|---|
| `preprocessing.py` | `ROLLING_WINDOW` | 10 |
| `preprocessing.py` | `STUCK_MIN_RUN` | 10 (точное равенство) |
| `anomaly_detection.py` | `RULE_PARAMS` | см. файл |
| `train_model.py` | `DEFAULT_CONTAMINATION` | 0.04 |