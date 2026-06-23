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

st.title("🌡️ ИИ-модуль мониторинга температурных показаний")
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
# Фильтр по сценарию, если такая колонка есть в данных
possible_scenario_columns = ["scenario", "anomaly_type", "scenario_name", "mode", "true_scenario"]

scenario_column = None
for col in possible_scenario_columns:
    if col in df.columns:
        scenario_column = col
        break

# Русские названия сценариев для интерфейса
scenario_names_ru = {
    "normal": "Нормальная работа",
    "sharp_jump": "Резкий скачок температуры",
    "spike": "Резкий выброс",
    "slow_overheating": "Медленный перегрев",
    "sensor_drift": "Дрейф датчика",
    "stuck_sensor": "Зависание датчика",
    "high_noise": "Сильный шум",
    "signal_loss": "Потеря сигнала",
    "correlated_growth": "Коррелированный рост"
}

if scenario_column is not None:
    all_scenarios = sorted(df[scenario_column].dropna().unique())

    selected_scenarios_ru = st.sidebar.multiselect(
        "Сценарий:",
        options=all_scenarios,
        default=all_scenarios,
        format_func=lambda x: scenario_names_ru.get(x, x)
    )

    selected_scenarios = selected_scenarios_ru
else:
    selected_scenarios = None


risk_levels = sorted(alarm_log["Уровень"].unique())

selected_risk_levels = st.sidebar.multiselect(
    "Уровень тревоги:",
    options=risk_levels,
    default=risk_levels
)
show_anomalies_only = st.sidebar.checkbox(
    "Показывать только аномалии",
    value=True
)

# ============================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================================

filtered_df = df[df["sensor_id"].isin(selected_sensors)].copy()

# Применяем фильтр сценариев ко всей таблице результатов
if scenario_column is not None and selected_scenarios is not None:
    filtered_df = filtered_df[
        filtered_df[scenario_column].isin(selected_scenarios)
    ].copy()

filtered_alarm_log = alarm_log[
    (alarm_log["Датчик"].isin(selected_sensors)) &
    (alarm_log["Уровень"].isin(selected_risk_levels))
].copy()

# Применяем фильтр сценариев к журналу тревог
if "Истинный_сценарий" in filtered_alarm_log.columns and selected_scenarios is not None:
    filtered_alarm_log = filtered_alarm_log[
        filtered_alarm_log["Истинный_сценарий"].isin(selected_scenarios)
    ].copy()
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Обзор",
    "Температурный тренд",
    "Обнаружение аномалий",
    "Журнал аварийных сигналов",
    "О проекте"
])


# ============================================================
# 6. ОСНОВНЫЕ МЕТРИКИ
# ============================================================

with tab1:
    st.subheader("📊 Общий обзор системы")

    # Основные показатели
    total_points = len(filtered_df)
    total_anomalies = int(filtered_df["final_anomaly"].sum())
    anomaly_percent = total_anomalies / total_points * 100 if total_points > 0 else 0

    sensors_count = len(selected_sensors)
    total_alarms = len(filtered_alarm_log)

    high_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "High"])
    medium_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Medium"])
    warning_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Warning"])

    # Логика статуса системы
    if high_count > 0:
        system_status = "🔴 Требуется внимание"
        status_text = "Обнаружены тревоги высокого уровня. Необходимо проверить журнал событий."
    elif medium_count > 0 or warning_count > 0:
        system_status = "🟡 Есть предупреждения"
        status_text = "Обнаружены предупреждения или тревоги среднего уровня."
    else:
        system_status = "🟢 Нормальная работа"
        status_text = "Критических тревог по выбранным фильтрам не обнаружено."

    # Карточка статуса
    st.markdown("### Состояние системы")
    st.info(f"**{system_status}**  \n{status_text}")

    # Метрики первого ряда
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Датчиков выбрано", sensors_count)
    col2.metric("Всего точек", total_points)
    col3.metric("Найдено аномалий", total_anomalies)
    col4.metric("Доля аномалий", f"{anomaly_percent:.2f}%")

    # Метрики второго ряда
    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Всего тревог", total_alarms)
    col6.metric("High", high_count)
    col7.metric("Medium", medium_count)
    col8.metric("Warning", warning_count)
    st.markdown("### Выбранные сценарии")

    if scenario_column is not None and len(filtered_df) > 0:
        scenario_counts = (
            filtered_df[scenario_column]
            .value_counts()
            .reset_index()
        )

        scenario_counts.columns = ["Сценарий", "Количество точек"]

        scenario_counts["Сценарий"] = scenario_counts["Сценарий"].map(
            lambda x: scenario_names_ru.get(x, x)
        )

        st.dataframe(
            scenario_counts,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Сценарии не найдены в данных.")
    st.markdown("---")

    st.markdown(
        """
        **Что показывает этот раздел:**  
        Overview дает быстрый обзор состояния системы по выбранным фильтрам.  
        Здесь можно сразу увидеть, сколько данных анализируется, сколько найдено аномалий,
        есть ли тревоги высокого уровня и насколько стабильна работа датчиков.
        """
    )

# ============================================================
# 7. ГРАФИК ТЕМПЕРАТУР
# ============================================================

with tab2:
    st.subheader("📈 Температурные показания датчиков")

    st.markdown(
        """
        Здесь показаны температурные ряды по выбранным датчикам.  
        Красные точки обозначают моменты, которые система определила как аномальные.
        """
    )

    # Дополнительные настройки графика внутри вкладки
    col_a, col_b = st.columns(2)

    with col_a:
        chart_sensors = st.multiselect(
            "Датчики на графике:",
            options=sorted(filtered_df["sensor_id"].unique()),
            default=sorted(filtered_df["sensor_id"].unique())
        )

    with col_b:
        show_anomaly_points = st.checkbox(
            "Показывать точки аномалий",
            value=True
        )

    chart_df = filtered_df[filtered_df["sensor_id"].isin(chart_sensors)].copy()

    if len(chart_df) == 0:
        st.warning("Выберите хотя бы один датчик для отображения графика.")
    else:
        fig = px.line(
            chart_df,
            x="timestamp",
            y="temperature_filled",
            color="sensor_id",
            title="Температурные тренды по выбранным датчикам",
            labels={
                "timestamp": "Время",
                "temperature_filled": "Температура, °C",
                "sensor_id": "Датчик"
            }
        )

        # Улучшение внешнего вида графика
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Датчик",
            xaxis_title="Время",
            yaxis_title="Температура, °C"
        )

        # Отображение аномалий поверх линий
        if show_anomaly_points:
            anomaly_points = chart_df[chart_df["final_anomaly"] == 1]

            fig.add_scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["temperature_filled"],
                mode="markers",
                marker=dict(size=9, color="red"),
                name="Обнаруженные аномалии",
                text=anomaly_points["sensor_id"],
                hovertemplate=(
                    "Датчик: %{text}<br>"
                    "Время: %{x}<br>"
                    "Температура: %{y:.2f} °C<br>"
                    "<extra></extra>"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Краткая статистика по отображаемому графику
        chart_total_points = len(chart_df)
        chart_anomalies = int(chart_df["final_anomaly"].sum())
        chart_anomaly_percent = (
            chart_anomalies / chart_total_points * 100
            if chart_total_points > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Точек на графике", chart_total_points)
        col2.metric("Аномалий на графике", chart_anomalies)
        col3.metric("Доля аномалий", f"{chart_anomaly_percent:.2f}%")


# ============================================================
# 8. ГРАФИК ANOMALY SCORE
# ============================================================
with tab3:
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
with tab4:
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
with tab5:
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
