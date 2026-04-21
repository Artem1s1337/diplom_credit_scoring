from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Конфигурация
DATA_PATH = "data/bureau.csv"  # путь к CSV с данными
TARGET_COL = "target"  # имя целевого столбца
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODELS_DIR = "models/second_best"
IMAGES_DIR = "images/metrics/second"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print(f"DATA_PATH={DATA_PATH}, TARGET_COL={TARGET_COL}")
print(f"Artifacts will be saved to: {MODELS_DIR}")
print(f"Metrics images will be saved to: {IMAGES_DIR}")


# Полезные функции: обработчики, метрики, архитектура нейросети
class PreparedData(NamedTuple):
    """Содержит подготовленные части набора данных."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет из CSV.

    Args:
        path (str): Путь к CSV-файлу.

    Returns:
        pd.DataFrame: Загруженный DataFrame.
    """
    return pd.read_csv(path)


def ensure_binary_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Кодирует метки в 0/1 с помощью LabelEncoder и проверяет, что задача является бинарной классификацией.

    Args:
        y (pd.Series): Целевая колонка.

    Returns:
        Tuple[np.ndarray, LabelEncoder]: Закодированный целевой массив и обученный LabelEncoder.

    Raises:
        ValueError: Если задача не является бинарной классификацией.
    """

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)
    classes = list(le.classes_)
    if len(classes) != 2:
        raise ValueError(
            f"Expected binary classification, but got {len(classes)} classes: {classes}"
        )
    return y_enc.astype(np.int64), le


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float, random_state: int
) -> PreparedData:
    """Разбивает данные на обучающую и тестовую выборки со стратификацией.

    Args:
        df (pd.DataFrame): Полный датасет.
        target_col (str): Имя целевого столбца.
        test_size (float): Доля тестовой выборки.
        random_state (int): Значение случайного зерна.

    Returns:
        PreparedData: Подготовленные части данных, включая кодировщик меток.
    """

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. Available: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    y, le = ensure_binary_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return PreparedData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, label_encoder=le
    )


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Создаёт препроцессор ColumnTransformer для числовых и категориальных признаков.

    - Числовые: медианное заполнение пропусков + StandardScaler
    - Категориальные: заполнение модой + OneHotEncoder(handle_unknown='ignore', dense)

    Args:
        df (pd.DataFrame): Датафрейм признаков (без целевой колонки).

    Returns:
        ColumnTransformer: Трансформер предобработки (обученный или нет).
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = PipelineSteps.numeric()
    categorical_pipeline = PipelineSteps.categorical()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
    return preprocessor


def safe_one_hot_encoder(dense: bool = True) -> OneHotEncoder:
    """Создаёт OneHotEncoder, совместимый с разными версиями scikit-learn.

    Пытается использовать 'sparse_output' (новый API, sklearn >= 1.2). Для старых версий использует 'sparse'.

    Args:
        dense (bool): Возвращает плотный (True) или разреженный (False) выход.

    Returns:
        OneHotEncoder: Сконфигурированный кодировщик.
    """
    try:
        # Newer API (sklearn >= 1.2); 'sparse' removed in 1.5+
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        # Older API fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


class PipelineSteps:
    """Фабрика шагов препроцессинга."""

    @staticmethod
    def numeric() -> Any:
        """Создаёт конвейер для числовых признаков: SimpleImputer(median) + StandardScaler.

        Returns:
            Any: Трансформер конвейера scikit-learn.
        """
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

    @staticmethod
    def categorical() -> Any:
        """Создаёт конвейер для категориальных признаков: SimpleImputer(most_frequent) + OneHotEncoder (dense).

        Returns:
            Any: Трансформер конвейера scikit-learn.
        """
        from sklearn.pipeline import Pipeline

        ohe = safe_one_hot_encoder(
            dense=True
        )  # ensures dense output compatible with all estimators
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]
        )


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Обучает препроцессор на X_train и трансформирует X_train/X_test.

    Args:
        preprocessor (ColumnTransformer): Колонночный трансформер.
        X_train (pd.DataFrame): Признаки обучения.
        X_test (pd.DataFrame): Признаки теста.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Преобразованные обучающие и тестовые признаки.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    return X_train_proc, X_test_proc


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Рассчитывает метрики классификации: ROC-AUC, accuracy, precision, recall, F1.

    Args:
        y_true (np.ndarray): Истинные бинарные метки (0/1).
        y_proba (np.ndarray): Предсказанные вероятности положительного класса.
        threshold (float): Порог для перевода вероятностей в метки.

    Returns:
        Dict[str, float]: Словарь метрик.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


class TorchMLP(nn.Module):
    """Простая MLP для бинарной классификации с функцией потерь BCEWithLogitsLoss."""

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2
    ) -> None:
        """Инициализирует MLP.

        Args:
            input_dim (int): Число входных признаков.
            hidden_dims (List[int]): Список размеров скрытых слоёв.
            dropout (float): Вероятность dropout.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямое распространение.

        Args:
            x (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Логиты формы (N, 1).
        """
        return self.net(x)


@torch.no_grad()
def predict_proba_torch(
    model: TorchMLP, X: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """Предсказывает вероятности с обученной TorchMLP.

    Args:
        model (TorchMLP): Обученная torch-модель.
        X (np.ndarray): Предобработанные признаки.
        device (Optional[str]): Устройство для вычислений ('cuda' или 'cpu').

    Returns:
        np.ndarray: Вероятности положительного класса.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(X_t).squeeze(1)
    proba = torch.sigmoid(logits).cpu().numpy()
    return proba


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    early_stopping_patience: int = 5,
    min_delta: float = 0.0,
) -> Tuple[TorchMLP, Dict[str, List[float]]]:
    """Обучает простую MLP (Torch) на предобработанных признаках с ранней остановкой.

    Ранняя остановка отслеживает ROC-AUC на валидации и прерывает обучение,
    если улучшение меньше ``min_delta`` не наблюдается в течение ``early_stopping_patience`` эпох.

    Args:
        X_train (np.ndarray): Обучающие признаки (после предобработки).
        y_train (np.ndarray): Обучающие метки (0/1).
        X_valid (np.ndarray): Валидационные/тестовые признаки (после предобработки).
        y_valid (np.ndarray): Валидационные/тестовые метки (0/1).
        input_dim (int): Число входных признаков.
        epochs (int): Количество эпох обучения.
        batch_size (int): Размер батча.
        lr (float): Скорость обучения (Adam).
        weight_decay (float): L2-регуляризация (weight decay).
        device (Optional[str]): Устройство для вычислений (например, 'cuda' или 'cpu').
        early_stopping_patience (int): Число эпох ожидания улучшения перед остановкой.
        min_delta (float): Минимальное приращение ROC-AUC, считающееся улучшением.

    Returns:
        Tuple[TorchMLP, Dict[str, List[float]]]: Обученная модель (лучшая по AUC) и история обучения.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchMLP(input_dim=input_dim)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds_train = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
    )
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    history: Dict[str, List[float]] = {"loss": [], "val_auc": []}
    best_auc = -np.inf
    best_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(loader.dataset)

        # validation AUC
        model.eval()
        y_proba = predict_proba_torch(model, X_valid, device=device)
        val_auc = float(roc_auc_score(y_valid, y_proba))

        history["loss"].append(epoch_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc + min_delta:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:02d}] loss={epoch_loss:.4f} val_auc={val_auc:.4f} (best={best_auc:.4f})"
            )

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no AUC improvement > {min_delta} for "
                f"{early_stopping_patience} epoch(s). Best AUC={best_auc:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# Загружаем данные
df = load_data(DATA_PATH)
df.columns = (
    df.columns.str.lower()
)  # приведём названия колонок к нижнему регистру, чтобы было проще работать
df.drop(columns=["sk_id_curr"], inplace=True)
print("Shape:", df.shape)
print("Columns:", list(df.columns)[:20], "..." if df.shape[1] > 20 else "")

# Split
prepared = split_data(df, TARGET_COL, TEST_SIZE, RANDOM_STATE)
X_train, X_test, y_train, y_test = (
    prepared.X_train,
    prepared.X_test,
    prepared.y_train,
    prepared.y_test,
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Делаем предобработку для всех моделей
preprocessor = build_preprocessor(X_train)
X_train_proc, X_test_proc = fit_transform_preprocessor(preprocessor, X_train, X_test)
input_dim = X_train_proc.shape[1]
print("Processed shapes:", X_train_proc.shape, X_test_proc.shape)

# Тренируем и оцениваем модели
results: Dict[str, Dict[str, Any]] = {}

# 1) Logistic Regression
lr = LogisticRegression(
    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=None
)
lr.fit(X_train_proc, y_train)
lr_proba = lr.predict_proba(X_test_proc)[:, 1]
results["logreg"] = {
    "estimator": lr,
    "proba": lr_proba,
    "metrics": compute_metrics(y_test, lr_proba),
}

# 2) Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_proc, y_train)
rf_proba = rf.predict_proba(X_test_proc)[:, 1]
results["random_forest"] = {
    "estimator": rf,
    "proba": rf_proba,
    "metrics": compute_metrics(y_test, rf_proba),
}

# 3) LightGBM
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)
lgbm.fit(X_train_proc, y_train)
lgbm_proba = lgbm.predict_proba(X_test_proc)[:, 1]
results["lightgbm"] = {
    "estimator": lgbm,
    "proba": lgbm_proba,
    "metrics": compute_metrics(y_test, lgbm_proba),
}

# 4) XGBoost
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    use_label_encoder=False,
)
xgb.fit(X_train_proc, y_train)
xgb_proba = xgb.predict_proba(X_test_proc)[:, 1]
results["xgboost"] = {
    "estimator": xgb,
    "proba": xgb_proba,
    "metrics": compute_metrics(y_test, xgb_proba),
}

# 5) CatBoost
cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False,
)
cat.fit(X_train_proc, y_train)
cat_proba = cat.predict_proba(X_test_proc)[:, 1]
results["catboost"] = {
    "estimator": cat,
    "proba": cat_proba,
    "metrics": compute_metrics(y_test, cat_proba),
}

# 6) PyTorch MLP
torch_model, torch_history = train_torch_model(
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
    input_dim=input_dim,
    epochs=30,
    batch_size=512,
    early_stopping_patience=10,
    min_delta=1e-3,
    lr=1e-3,
    weight_decay=1e-4,
)
torch_proba = predict_proba_torch(torch_model, X_test_proc)
results["torch_mlp"] = {
    "estimator": torch_model,
    "proba": torch_proba,
    "metrics": compute_metrics(y_test, torch_proba),
    "history": torch_history,
}

# Summary
summary = {name: vals["metrics"] for name, vals in results.items()}
print(json.dumps(summary, indent=2))


# Сохраняем лучшую модели и графики
def save_best_artifacts(
    results: Dict[str, Dict[str, Any]],
    preprocessor: ColumnTransformer,
    y_test: np.ndarray,
    models_dir: str,
    images_dir: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Сохраняет лучшую модель и графики.

    Args:
        results (Dict[str, Dict[str, Any]]): Словарь: имя_модели -> {estimator, proba, metrics, ...}.
        preprocessor (ColumnTransformer): Обученный препроцессор.
        y_test (np.ndarray): Метки тестовой выборки.
        models_dir (str): Папка для сохранения артефактов joblib.
        images_dir (str): Папка для сохранения графиков.
        threshold (float): Порог для матрицы ошибок.

    Returns:
        Dict[str, Any]: Метаданные сохранённых артефактов (имя лучшей модели, метрики, пути).
    """
    # Выбирает лучшую модель по ROC-AUC
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["roc_auc"])
    best_info = results[best_name]
    best_metrics = best_info["metrics"]
    best_proba = best_info["proba"]

    # Строит ROC-кривые для всех моделей
    plt.figure(figsize=(8, 6))
    for name, info in results.items():
        fpr, tpr, _ = roc_curve(y_test, info["proba"])
        auc = roc_auc_score(y_test, info["proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (all models)")
    plt.legend(loc="lower right")
    roc_path = os.path.join(images_dir, "roc_auc_all_models.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Строит матрицу ошибок для лучшей модели
    y_pred = (best_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({best_name}, thr={threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(images_dir, "cm_best_model.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Сохраняет артефакты в joblib
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    artifacts: Dict[str, Any] = {
        "model_name": best_name,
        "metrics_test": best_metrics,
        "threshold": threshold,
        "created_at": created_at,
        # Preprocessor shared by all models
        "preprocessor": preprocessor,
        # For sklearn/tree boosting models – fitted estimator
        # For torch – store state_dict and model init params
    }

    if best_name == "torch_mlp":
        # Save torch model state and architecture params
        artifacts.update(
            {
                "estimator_type": "torch",
                "torch_state_dict": {
                    k: v.cpu().numpy()
                    for k, v in best_info["estimator"].state_dict().items()
                },
                "torch_model_params": {
                    "input_dim": best_info["estimator"].net[0].in_features,
                    "hidden_dims": [128, 64],
                    "dropout": 0.2,
                },
            }
        )
    else:
        artifacts.update(
            {"estimator_type": "sklearn", "estimator": best_info["estimator"]}
        )

    joblib_path = os.path.join(models_dir, f"best_model_{best_name}.joblib")
    joblib.dump(artifacts, joblib_path)

    # Дополнительно сохраняет краткий JSON-отчёт
    summary = {
        "best_model": best_name,
        "metrics_test": best_metrics,
        "joblib_path": joblib_path,
        "roc_plot": roc_path,
        "cm_plot": cm_path,
    }
    with open(
        os.path.join(models_dir, f"summary_{best_name}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Best model:", best_name)
    print("Test metrics:", best_metrics)
    print("Saved:", joblib_path)
    print("ROC:", roc_path)
    print("CM:", cm_path)
    return summary


summary = save_best_artifacts(
    results, preprocessor, y_test, MODELS_DIR, IMAGES_DIR, threshold=0.5
)
summary
