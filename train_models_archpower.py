"""5.4. Обучение на Linear Regression и Random Forest с ArchPower.

Първо стартирайте preprocess_archpower.py от т. 5.3.
После стартирайте: python train_models_archpower.py
Скриптът използва само подготвените данни и не обучава моделите върху тестовите наблюдения.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
input_file = ROOT / "processed_archpower.npz"
if not input_file.exists():
    raise FileNotFoundError(
        "Липсва processed_archpower.npz. Първо изпълнете preprocess_archpower.py от точка 5.3."
    )

# 1. Подготвени данни от т. 5.3
with np.load(input_file, allow_pickle=False) as data:
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

if len(X_train) != len(y_train) or len(X_test) != len(y_test):
    raise ValueError("Броят наблюдения и броят целеви стойности не съвпадат.")

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)

forest_model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)

# 4. Една и съща тестова извадка и едни и същи метрики за двата модела
def evaluate(name, actual, predicted):
    return {
        "Модел": name,
        "MAE (W)": mean_absolute_error(actual, predicted),
        "RMSE (W)": np.sqrt(mean_squared_error(actual, predicted)),
        "R²": r2_score(actual, predicted),
    }

metrics = pd.DataFrame([
    evaluate("Linear Regression", y_test, linear_predictions),
    evaluate("Random Forest", y_test, forest_predictions),
])
print("\nРезултати върху ТЕСТОВИТЕ данни:")
print(metrics.to_string(index=False, float_format=lambda n: f"{n:.4f}"))

# 5. Запис за графиките и анализа в т. 5.5
predictions = pd.DataFrame({
    "Референтна мощност (W)": y_test,
    "Linear Regression (W)": linear_predictions,
    "Random Forest (W)": forest_predictions,
})
predictions.to_csv(ROOT / "archpower_predictions.csv", index=False, encoding="utf-8-sig")
metrics.to_csv(ROOT / "archpower_metrics.csv", index=False, encoding="utf-8-sig")
print("\nЗапазени файлове: archpower_predictions.csv и archpower_metrics.csv")
