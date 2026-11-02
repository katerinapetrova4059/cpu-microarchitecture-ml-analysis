"""Подготовка на набора ArchPower за сравнение на Linear Regression и Random Forest.

Структура на папката:
    archpower_project/
        preprocess_archpower.py
        dataset/
            feature.npy
            label.npy

Стартиране: python preprocess_archpower.py
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
FEATURE_FILE = DATASET / "feature.npy"
LABEL_FILE = DATASET / "label.npy"

if not FEATURE_FILE.is_file() or not LABEL_FILE.is_file():
    raise FileNotFoundError(
        "Копирай feature.npy и label.npy в папката 'dataset' до този скрипт. "
        "Източник: https://github.com/hkust-zhiyao/ArchPower"
    )

# 1. Зареждане на оригиналните данни
X = np.asarray(np.load(FEATURE_FILE, allow_pickle=False), dtype=float)
labels = np.asarray(np.load(LABEL_FILE, allow_pickle=False), dtype=float)

# ArchPower: 25 конфигурации, 8 натоварвания за всяка от тях.
if X.ndim != 2 or X.shape[0] != 200 or labels.shape[0] != 200 or labels.size != 200 * 60:
    raise ValueError(f"Неочаквани размери на ArchPower: X={X.shape}, labels={labels.shape}")

# 2. От 60-те стойности избираме общата мощност (цял CPU, total power).
y = labels.reshape(200, 12, 5)[:, 0, 0]
if not np.isfinite(y).all():
    raise ValueError("Целевите стойности съдържат NaN или безкрайност.")

# 3. Проверка за невалидни входни стойности
invalid_count = int((~np.isfinite(X)).sum())
X[~np.isfinite(X)] = np.nan
print(f"Наблюдения: {len(X)}")
print(f"Входни характеристики: {X.shape[1]}")
print(f"Невалидни входни стойности: {invalid_count}")
print(f"Стойности на целта (W): min={y.min():.4f}, max={y.max():.4f}")

# 4. Групиране: 8 поредни реда = една процесорна конфигурация.
configurations = np.arange(25)
architecture = np.array(["BOOM"] * 15 + ["XiangShan"] * 10)
train_cfg, test_cfg = train_test_split(
    configurations, test_size=0.20, random_state=42, stratify=architecture
)
row_cfg = np.arange(200) // 8
train_rows = np.isin(row_cfg, train_cfg)
test_rows = np.isin(row_cfg, test_cfg)
X_train, X_test = X[train_rows], X[test_rows]
y_train, y_test = y[train_rows], y[test_rows]

# 5. Запълване с медианата на съответната характеристика в ОБУЧАВАЩИТЕ данни.
imputer = SimpleImputer(strategy="median", keep_empty_features=True)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 6. Мащабиране само за линейната регресия (без изтичане на информация).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Запазваме две версии на характеристиките за следващата точка (5.4).
output = ROOT / "processed_archpower.npz"
np.savez_compressed(
    output,
    X_train=X_train, X_test=X_test,
    X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
    y_train=y_train, y_test=y_test,
    train_configurations=train_cfg, test_configurations=test_cfg,
)
print(f"Обучаващи конфигурации: {len(train_cfg)}; наблюдения: {len(X_train)}")
print(f"Тестови конфигурации: {len(test_cfg)}; наблюдения: {len(X_test)}")
print(f"Запазен файл: {output.name}")
