import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="Temperature Anomaly Monitor",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data
def load_data():
    results = pd.read_csv("temperature_anomaly_results.csv")
    alarms = pd.read_csv("alarm_log.csv")

    results["timestamp"] = pd.to_datetime(results["timestamp"])
    alarms["Время"] = pd.to_datetime(alarms["Время"])

    return results, alarms


df, alarm_log = load_data()


# ============================================================
# 3. ЗАГОЛОВОК
# ============================================================

st.title("ИИ-модуль мониторинга температурных показаний")
st.markdown(
    """
    Дашборд показывает работу прототипа системы раннего обнаружения температурных аномалий
    на условном радиохимическом участке.
    
    Система анализирует температурные ряды, выявляет отклонения и формирует журнал тревог
    для оператора.
    """
)


# ============================================================
# 4. БОКОВАЯ ПАНЕЛЬ ФИЛЬТРОВ
# ============================================================

st.sidebar.header("⚙️ Фильтры")

all_sensors = sorted(df["sensor_id"].unique())

selected_sensors = st.sidebar.multiselect(
    "Выберите датчики:",
    options=all_sensors,
    default=all_sensors
)

show_anomalies_only = st.sidebar.checkbox(
    "Показать только аномалии в таблице",
    value=False
)

risk_levels = sorted(alarm_log["Уровень"].unique())

selected_risk_levels = st.sidebar.multiselect(
    "Уровень тревоги:",
    options=risk_levels,
    default=risk_levels
)


# ============================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================================

filtered_df = df[df["sensor_id"].isin(selected_sensors)].copy()

filtered_alarm_log = alarm_log[
    (alarm_log["Датчик"].isin(selected_sensors)) &
    (alarm_log["Уровень"].isin(selected_risk_levels))
].copy()


# ============================================================
# 6. ОСНОВНЫЕ МЕТРИКИ
# ============================================================

total_points = len(filtered_df)
total_anomalies = int(filtered_df["final_anomaly"].sum())
anomaly_percent = total_anomalies / total_points * 100 if total_points > 0 else 0

high_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "High"])
medium_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Medium"])
warning_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Warning"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Всего точек", total_points)
col2.metric("Найдено аномалий", total_anomalies)
col3.metric("Доля аномалий", f"{anomaly_percent:.2f}%")
col4.metric("High-тревоги", high_count)


# ============================================================
# 7. ГРАФИК ТЕМПЕРАТУР
# ============================================================

st.subheader("📈 Температурные показания датчиков")

fig = px.line(
    filtered_df,
    x="timestamp",
    y="temperature_filled",
    color="sensor_id",
    title="Температурные ряды по датчикам",
    labels={
        "timestamp": "Время",
        "temperature_filled": "Температура, °C",
        "sensor_id": "Датчик"
    }
)

# Добавляем точки аномалий
anomaly_points = filtered_df[filtered_df["final_anomaly"] == 1]

fig.add_scatter(
    x=anomaly_points["timestamp"],
    y=anomaly_points["temperature_filled"],
    mode="markers",
    marker=dict(size=8, color="red"),
    name="Обнаруженные аномалии"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 8. ГРАФИК ANOMALY SCORE
# ============================================================

st.subheader("🧠 Anomaly score")

score_fig = px.line(
    filtered_df,
    x="timestamp",
    y="anomaly_score_norm",
    color="sensor_id",
    title="Оценка аномальности поведения датчиков",
    labels={
        "timestamp": "Время",
        "anomaly_score_norm": "Anomaly score",
        "sensor_id": "Датчик"
    }
)

st.plotly_chart(score_fig, use_container_width=True)


# ============================================================
# 9. ЖУРНАЛ ТРЕВОГ
# ============================================================

st.subheader("🚨 Журнал тревог")

if show_anomalies_only:
    table_df = filtered_alarm_log
else:
    table_df = filtered_alarm_log

if len(table_df) == 0:
    st.info("По выбранным фильтрам тревог не найдено.")
else:
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 10. СВОДКА ПО ТИПАМ СОБЫТИЙ
# ============================================================

st.subheader("📊 Сводка по типам событий")

event_counts = (
    filtered_alarm_log["Тип_события"]
    .value_counts()
    .reset_index()
)

event_counts.columns = ["Тип события", "Количество"]

if len(event_counts) > 0:
    bar_fig = px.bar(
        event_counts,
        x="Тип события",
        y="Количество",
        title="Количество тревог по типам событий"
    )

    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("Нет данных для построения сводки.")


# ============================================================
# 11. ПОЯСНЕНИЕ ДЛЯ ЗАЩИТЫ
# ============================================================

st.subheader("ℹ️ Описание работы MVP")

st.markdown(
    """
    В данном MVP используются синтетические температурные данные.
    Нормальный режим моделируется как сочетание базовой температуры,
    плавных технологических колебаний и случайного шума.

    Обнаружение аномалий выполняется двумя способами:

    1. **Правилами** — резкие скачки, пропуски сигнала, зависание датчика,
       отклонение от группы датчиков.
    2. **ИИ-моделью Isolation Forest** — поиск нетипичных комбинаций признаков.

    Система не управляет технологическим процессом и не заменяет оператора.
    Она предназначена для раннего выявления подозрительных температурных трендов
    и поддержки принятия решений.
    """
)
