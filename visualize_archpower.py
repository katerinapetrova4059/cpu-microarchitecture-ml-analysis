"""5.5. Визуализация на реалните резултати от ArchPower.

Изпълнете първо preprocess_archpower.py и train_models_archpower.py.
После: python visualize_archpower.py

Изисква: pip install pandas numpy matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "archpower_predictions.csv"
METRIC_FILE = ROOT / "archpower_metrics.csv"

if not PRED_FILE.exists() or not METRIC_FILE.exists():
    raise FileNotFoundError(
        "Липсват archpower_predictions.csv или archpower_metrics.csv. "
        "Стартирайте първо train_models_archpower.py."
    )

predictions = pd.read_csv(PRED_FILE)
metrics = pd.read_csv(METRIC_FILE)
required_predictions = [
    "Референтна мощност (W)",
    "Linear Regression (W)",
    "Random Forest (W)",
]
required_metrics = ["Модел", "MAE (W)", "RMSE (W)", "R²"]
for col in required_predictions:
    if col not in predictions.columns:
        raise ValueError(f"Липсва колона в прогнозите: {col}")
for col in required_metrics:
    if col not in metrics.columns:
        raise ValueError(f"Липсва колона в метриките: {col}")

actual = predictions["Референтна мощност (W)"].to_numpy(dtype=float)
linear = predictions["Linear Regression (W)"].to_numpy(dtype=float)
forest = predictions["Random Forest (W)"].to_numpy(dtype=float)
if not np.isfinite(np.column_stack([actual, linear, forest])).all():
    raise ValueError("Прогнозите съдържат невалидни числови стойности.")

# Фигура 6: прогноза спрямо референтна мощност. Всяка точка е тестов запис.
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(actual, linear, label="Linear Regression", alpha=0.75, marker="o")
ax.scatter(actual, forest, label="Random Forest", alpha=0.75, marker="x")
min_val = min(actual.min(), linear.min(), forest.min())
max_val = max(actual.max(), linear.max(), forest.max())
margin = (max_val - min_val) * 0.05 or 0.1
lo, hi = min_val - margin, max_val + margin
ax.plot([lo, hi], [lo, hi], linestyle="--", label="Perfect prediction")
ax.set(xlabel="Reference power (W)", ylabel="Predicted power (W)",
       title="ArchPower: reference versus predicted power", xlim=(lo, hi), ylim=(lo, hi))
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(ROOT / "figure_6_predictions.png", dpi=200)
plt.close(fig)

# Фигура 7: MAE/RMSE сравнение. И двете метрики са във ватове.
fig, ax = plt.subplots(figsize=(7, 4.5))
names = metrics["Модел"].astype(str).tolist()
x = np.arange(len(names))
width = 0.35
bars_mae = ax.bar(x - width / 2, metrics["MAE (W)"], width, label="MAE (W)")
bars_rmse = ax.bar(x + width / 2, metrics["RMSE (W)"], width, label="RMSE (W)")
ax.set_xticks(x, names)
ax.set(ylabel="Error (W)", title="ArchPower: prediction errors on test data")
ax.bar_label(bars_mae, fmt="%.3f", padding=2, fontsize=9)
ax.bar_label(bars_rmse, fmt="%.3f", padding=2, fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "figure_7_errors.png", dpi=200)
plt.close(fig)

print("\nТаблица 4 — резултати от тестовите данни:\n")
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nСъздадени: figure_6_predictions.png и figure_7_errors.png")

by_mae = metrics.sort_values("MAE (W)")["Модел"].iloc[0]
by_rmse = metrics.sort_values("RMSE (W)")["Модел"].iloc[0]
if by_mae == by_rmse:
    print(f"\nПо MAE и RMSE по-добре се представя: {by_mae}.")
else:
    print("\nMAE и RMSE посочват различни модели. Обсъдете разликите в 5.5.")
