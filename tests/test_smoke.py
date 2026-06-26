"""Дымовые тесты: основные модули импортируются и предобработка работает.

Эти тесты живут в ветке CI (feature/tests-ci), чтобы pytest всегда находил хотя
бы один тест, даже до того, как смержены функциональные тесты из других PR.
"""
import pandas as pd


def test_core_modules_import():
    import preprocessing
    import anomaly_detection
    assert hasattr(preprocessing, "preprocess_data")
    assert hasattr(anomaly_detection, "detect_anomalies")


def test_preprocess_tiny():
    from preprocessing import preprocess_data
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:02"],
        "sensor_id": ["T-01", "T-01", "T-01"],
        "temperature": [70.0, 71.0, 70.5],
    })
    out = preprocess_data(df)
    # Ключевые признаки считаются.
    for col in ("is_missing", "rolling_mean", "abs_z_score", "abs_diff_from_group_mean"):
        assert col in out.columns