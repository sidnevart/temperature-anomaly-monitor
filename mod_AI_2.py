import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# БЛОК 1 – УКАЗАНИЕ ПУТИ К ФАЙЛУ И ВЫХОДНЫМ ФАЙЛАМ
# ============================================================
INPUT_CSV = "Т2.csv"               # имя входного CSV
INPUT_PATH = "/Users/mac/Documents/mod_AI/"                             # папка, где лежит файл (если в текущей, оставить ".")
OUTPUT_DIR = "/Users/mac/Documents/mod_AI/"                             # папка для сохранения результатов (можно оставить ту же)

RESULTS_FILE = "temperature_anomaly_results.csv"
ALARM_LOG_FILE = "alarm_log.csv"
PLOT_FILE = "temperature_plot_with_anomalies.png"

# ============================================================
# БЛОК 2 – НАСТРОЙКА ПАРАМЕТРОВ ОБНАРУЖЕНИЯ
# ============================================================
JUMP_THRESHOLD = 5.0
Z_SCORE_THRESHOLD = 3.0
STUCK_SECONDS = 10
SLOW_GROWTH_RATE = 0.08
ROLLING_WINDOW = 60

IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.08
IF_RANDOM_STATE = 42

# ВКЛЮЧЕНИЕ / ОТКЛЮЧЕНИЕ СЦЕНАРИЕВ
ENABLE_AI_ONLY = True          # "Нетипичное поведение по ИИ-модели"
ENABLE_SLOW_GROWTH = True      # "Медленный рост температуры"
ENABLE_STUCK = True            # "Зависание датчика"
ENABLE_Z_SCORE = True          # "Сильное отклонение от нормы"

# ============================================================
# ОСНОВНОЙ КОД
# ============================================================

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    if 'time_s' in df.columns and 'temp_C' in df.columns:
        df = df.rename(columns={'time_s': 'timestamp', 'temp_C': 'temperature'})
    else:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'temperature'})
        else:
            raise ValueError("Не удалось определить колонки: нужны 'time_s' и 'temp_C'")
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['sensor_id'] = 0
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def compute_features(df):
    df['is_missing'] = df['temperature'].isna().astype(int)
    df['temperature_filled'] = df['temperature'].interpolate(method='linear', limit_direction='both')
    df['temperature_filled'] = df['temperature_filled'].fillna(method='bfill').fillna(method='ffill')

    df['temp_diff'] = df['temperature_filled'].diff()
    df['abs_temp_diff'] = df['temp_diff'].abs()

    df['rolling_mean'] = df['temperature_filled'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['rolling_std'] = df['temperature_filled'].rolling(window=ROLLING_WINDOW, min_periods=1).std()
    df['rolling_std'] = df['rolling_std'].replace(0, 1e-6)

    df['abs_z_score'] = ((df['temperature_filled'] - df['rolling_mean']) / df['rolling_std']).abs()

    df['is_stuck'] = 0
    stuck_counter = 0
    prev_temp = None
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'temperature_filled']):
            stuck_counter = 0
            continue
        if prev_temp is not None and abs(df.loc[i, 'temperature_filled'] - prev_temp) < 1e-6:
            stuck_counter += 1
        else:
            stuck_counter = 0
        if stuck_counter >= STUCK_SECONDS:
            df.loc[i, 'is_stuck'] = 1
        prev_temp = df.loc[i, 'temperature_filled']

    df['abs_diff_from_group_mean'] = (df['temperature_filled'] - df['rolling_mean']).abs()
    df['rolling_temp_diff_mean'] = df['temp_diff'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['scenario'] = os.path.basename(INPUT_CSV)
    return df

def apply_rules(df):
    df['rule_anomaly'] = 0
    df['rule_event_type'] = "normal"
    df['rule_risk_level'] = "Normal"
    df['rule_recommendation'] = "Наблюдение в штатном режиме"

    # Потеря сигнала
    mask_missing = df['is_missing'] == 1
    df.loc[mask_missing, 'rule_anomaly'] = 1
    df.loc[mask_missing, 'rule_event_type'] = "Потеря сигнала"
    df.loc[mask_missing, 'rule_risk_level'] = "Medium"
    df.loc[mask_missing, 'rule_recommendation'] = "Проверить канал измерения и наличие связи с датчиком"

    # Резкий скачок
    mask_jump = df['abs_temp_diff'] > JUMP_THRESHOLD
    df.loc[mask_jump, 'rule_anomaly'] = 1
    df.loc[mask_jump, 'rule_event_type'] = "Резкий скачок температуры"
    df.loc[mask_jump, 'rule_risk_level'] = "High"
    df.loc[mask_jump, 'rule_recommendation'] = "Проверить показания датчика и сравнить с соседними каналами"

    # Сильное отклонение от нормы
    if ENABLE_Z_SCORE:
        mask_z = df['abs_z_score'] > Z_SCORE_THRESHOLD
        df.loc[mask_z, 'rule_anomaly'] = 1
        df.loc[mask_z, 'rule_event_type'] = "Сильное отклонение от нормы"
        df.loc[mask_z, 'rule_risk_level'] = "Warning"
        df.loc[mask_z, 'rule_recommendation'] = "Проверить тренд температуры и устойчивость отклонения"

    # Зависание датчика
    if ENABLE_STUCK:
        mask_stuck = df['is_stuck'] == 1
        df.loc[mask_stuck, 'rule_anomaly'] = 1
        df.loc[mask_stuck, 'rule_event_type'] = "Зависание датчика"
        df.loc[mask_stuck, 'rule_risk_level'] = "Medium"
        df.loc[mask_stuck, 'rule_recommendation'] = "Проверить исправность датчика и цепь передачи данных"

    # Отклонение от скользящего среднего
    mask_group = df['abs_diff_from_group_mean'] > 8.0
    df.loc[mask_group, 'rule_anomaly'] = 1
    df.loc[mask_group, 'rule_event_type'] = "Отклонение от скользящего среднего"
    df.loc[mask_group, 'rule_risk_level'] = "Warning"
    df.loc[mask_group, 'rule_recommendation'] = "Сравнить показания с соседними температурными каналами"

    # Медленный рост
    if ENABLE_SLOW_GROWTH:
        mask_slow = (df['rolling_temp_diff_mean'] > SLOW_GROWTH_RATE) & (df['temperature_filled'] > df['rolling_mean'])
        df.loc[mask_slow, 'rule_anomaly'] = 1
        df.loc[mask_slow, 'rule_event_type'] = "Медленный рост температуры"
        df.loc[mask_slow, 'rule_risk_level'] = "Warning"
        df.loc[mask_slow, 'rule_recommendation'] = "Проверить устойчивость роста температуры и сравнить с соседними датчиками"

    return df

def apply_isolation_forest(df):
    feature_columns = [
        'temperature_filled', 'rolling_mean', 'rolling_std', 'temp_diff',
        'abs_temp_diff', 'abs_z_score', 'is_missing', 'is_stuck',
        'abs_diff_from_group_mean', 'rolling_temp_diff_mean'
    ]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE
    )
    df['iforest_prediction'] = model.fit_predict(X_scaled)
    df['iforest_anomaly'] = (df['iforest_prediction'] == -1).astype(int)
    df['iforest_score_raw'] = model.decision_function(X_scaled)
    df['anomaly_score'] = -df['iforest_score_raw']

    min_score = df['anomaly_score'].min()
    max_score = df['anomaly_score'].max()
    if max_score > min_score:
        df['anomaly_score_norm'] = (df['anomaly_score'] - min_score) / (max_score - min_score)
    else:
        df['anomaly_score_norm'] = 0.0
    df['anomaly_score_norm'] = df['anomaly_score_norm'].fillna(0)
    return df

def combine_and_finalize(df):
    df['final_anomaly'] = ((df['rule_anomaly'] == 1) | (df['iforest_anomaly'] == 1)).astype(int)

    # AI-only аномалии
    if ENABLE_AI_ONLY:
        mask_ai_only = (df['iforest_anomaly'] == 1) & (df['rule_anomaly'] == 0)
        df.loc[mask_ai_only, 'rule_event_type'] = "Нетипичное поведение по ИИ-модели"
        df.loc[mask_ai_only, 'rule_risk_level'] = "Warning"
        df.loc[mask_ai_only, 'rule_recommendation'] = "Проверить участок графика и сравнить с другими температурными каналами"
    else:
        # Убираем аномалии, найденные только AI
        mask_ai_only = (df['iforest_anomaly'] == 1) & (df['rule_anomaly'] == 0)
        df.loc[mask_ai_only, 'final_anomaly'] = 0
        df.loc[mask_ai_only, 'rule_event_type'] = "normal"

    # Уточнение риска
    mask_high = df['anomaly_score_norm'] > 0.85
    df.loc[(df['final_anomaly'] == 1) & mask_high, 'rule_risk_level'] = "High"
    mask_medium = (df['anomaly_score_norm'] > 0.60) & (df['anomaly_score_norm'] <= 0.85)
    df.loc[(df['final_anomaly'] == 1) & mask_medium & (df['rule_risk_level'] == "Normal"), 'rule_risk_level'] = "Medium"

    return df

def generate_alarm_log(df):
    alarm = df[df['final_anomaly'] == 1].copy()
    if alarm.empty:
        return alarm

    alarm = alarm[[
        'timestamp', 'sensor_id', 'temperature', 'temperature_filled',
        'rule_event_type', 'rule_risk_level', 'anomaly_score_norm',
        'rule_recommendation', 'scenario'
    ]]
    alarm = alarm.rename(columns={
        'timestamp': 'Время (с)',
        'sensor_id': 'Датчик',
        'temperature': 'Температура',
        'temperature_filled': 'Температура_заполненная',
        'rule_event_type': 'Тип_события',
        'rule_risk_level': 'Уровень',
        'anomaly_score_norm': 'Anomaly_score',
        'rule_recommendation': 'Рекомендация',
        'scenario': 'Исходный_файл'
    })
    for col in ['Температура', 'Температура_заполненная']:
        alarm[col] = alarm[col].round(2)
    alarm['Anomaly_score'] = alarm['Anomaly_score'].round(3)
    return alarm

def plot_results(df, output_plot):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['temperature'], 'b-', label='Температура', linewidth=1, alpha=0.7)

    anomalies = df[df['final_anomaly'] == 1]
    # Исключаем тип "normal" (на случай, если затесался)
    anomalies = anomalies[anomalies['rule_event_type'] != 'normal']

    if not anomalies.empty:
        types = anomalies['rule_event_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
        for typ, color in zip(types, colors):
            subset = anomalies[anomalies['rule_event_type'] == typ]
            plt.scatter(subset['timestamp'], subset['temperature'],
                        color=color, s=50, label=typ, alpha=0.8)

    plt.xlabel('Время, с', fontname='Times New Roman')
    plt.ylabel('Температура, °C', fontname='Times New Roman')
    plt.title('Термограмма с обнаруженными аномалиями', fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.show()
    print(f"График сохранён: {output_plot}")

def main():
    input_full = os.path.join(INPUT_PATH, INPUT_CSV)
    if not os.path.exists(input_full):
        print(f"❌ Файл {input_full} не найден!")
        sys.exit(1)

    print(f"📊 Загрузка данных из {input_full} ...")
    df = load_and_prepare_data(input_full)
    print(f"✅ Загружено {len(df)} записей.")

    print("🔍 Вычисление признаков...")
    df = compute_features(df)

    print("🔍 Применение правил...")
    df = apply_rules(df)

    print("🔍 Применение Isolation Forest...")
    df = apply_isolation_forest(df)

    print("🔍 Объединение результатов...")
    df = combine_and_finalize(df)

    alarm_log = generate_alarm_log(df)

    results_full = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    df.to_csv(results_full, index=False, encoding='utf-8-sig')
    print(f"Полные результаты сохранены: {results_full}")

    alarm_full = os.path.join(OUTPUT_DIR, ALARM_LOG_FILE)
    if not alarm_log.empty:
        alarm_log.to_csv(alarm_full, index=False, encoding='utf-8-sig')
        print(f"Журнал тревог сохранён: {alarm_full}")
        print("\nПервые строки журнала тревог:")
        print(alarm_log.head(20))
    else:
        print("⚠️ Аномалий не обнаружено. Журнал тревог пуст.")

    plot_full = os.path.join(OUTPUT_DIR, PLOT_FILE)
    plot_results(df, plot_full)

    print("\n✅ Анализ завершён.")

if __name__ == "__main__":
    main()