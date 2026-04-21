from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_model_file(default_name: str = "best_model_xgboost.joblib") -> str:
    """Ищет файл модели в стандартных папках проекта.

    Порядок поиска:
    1) models/second_best/<default_name>
    2) models/first_best/<default_name>

    Args:
        default_name (str): Имя файла модели по умолчанию.

    Returns:
        str: Путь к найденному файлу.

    Raises:
        FileNotFoundError: Если файл не найден ни в одной из стандартных папок.
    """
    candidates = [
        os.path.join("models", "second_best", default_name),
        os.path.join("models", "first_best", default_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Файл {default_name} не найден. Проверьте наличие в models/second_best или models/first_best"
    )


def load_artifact(path: str) -> Dict[str, Any]:
    """Загружает артефакт модели из joblib-файла.

    Args:
        path (str): Путь к .joblib файлу артефакта.

    Returns:
        Dict[str, Any]: Содержимое артефакта (словарь с ключами estimator, preprocessor и др.).

    Raises:
        FileNotFoundError: Если путь не существует.
        ValueError: Если формат артефакта не соответствует ожидаемому словарю.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError("Ожидается артефакт в формате словаря (dict)")
    return obj


def get_feature_names(preprocessor: Any) -> List[str]:
    """Возвращает имена признаков после препроцессинга.

    Пытается использовать метод get_feature_names_out(). Если метод недоступен,
    возвращает индексы признаков в виде строковых имен.

    Args:
        preprocessor (Any): Обученный препроцессор (ColumnTransformer).

    Returns:
        List[str]: Список имён признаков после трансформации.
    """
    names: List[str]
    get_names = getattr(preprocessor, "get_feature_names_out", None)
    if callable(get_names):
        out = get_names()
        names = [str(x) for x in out]
    else:
        # Фолбэк: неизвестно число фич — вернёт пустой список, обработаем позже по длине важностей
        names = []
    return names


def extract_importances(estimator: Any) -> np.ndarray:
    """Извлекает важности признаков из оценщика.

    Args:
        estimator (Any): Обученная модель (например, XGBClassifier, LGBMClassifier и т.п.).

    Returns:
        np.ndarray: Вектор важностей признаков формы (D,).

    Raises:
        AttributeError: Если у модели отсутствуют важности признаков.
    """
    if hasattr(estimator, "feature_importances_"):
        imp = getattr(estimator, "feature_importances_")
        return np.asarray(imp)
    # На всякий случай фолбэк для линейных моделей
    if hasattr(estimator, "coef_"):
        coef = np.ravel(getattr(estimator, "coef_"))
        return np.abs(coef)
    raise AttributeError("У модели отсутствует атрибут feature_importances_ или coef_")


def top_importances(feature_names: List[str], importances: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """Формирует DataFrame с топ-N признаками по важности.

    Args:
        feature_names (List[str]): Имена признаков после препроцессинга.
        importances (np.ndarray): Важности признаков.
        top_n (int): Число топовых признаков для вывода.

    Returns:
        pd.DataFrame: Таблица с колонками ['feature', 'importance'], отсортированная по убыванию важности.
    """
    imp = np.asarray(importances).astype(float)
    n = imp.shape[0]
    if not feature_names or len(feature_names) != n:
        feature_names = [f"f_{i}" for i in range(n)]
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def plot_top_importances(df_top: pd.DataFrame, out_path: str) -> None:
    """Строит горизонтальный барчарт по топ-важностям и сохраняет на диск.

    Args:
        df_top (pd.DataFrame): Таблица с колонками ['feature', 'importance'].
        out_path (str): Путь для сохранения изображения.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_top.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orient="h",
        palette="Blues_r",
    )
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.title("Топ-10 важных признаков (XGBoost)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"График сохранён: {out_path}")


def main() -> None:
    """Запускает извлечение важностей XGBoost и строит график топ-10 признаков.

    Сценарий:
    1) Загружает артефакт best_model_xgboost.joblib (путь можно передать флагом --path)
    2) Извлекает feature_importances_ модели и имена признаков из препроцессора
    3) Формирует топ-10 по важности и строит горизонтальный барчарт
    4) Сохраняет картинку (флаг --out)
    """
    parser = argparse.ArgumentParser(description="Топ-10 важных признаков XGBoost")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Путь к артефакту .joblib. По умолчанию скрипт ищет best_model_xgboost.joblib "
            "в models/second_best или models/first_best"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Число топовых признаков для вывода (по умолчанию 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("images", "feature_importance_xgb_top10.png"),
        help="Путь для сохранения графика (PNG)",
    )
    args = parser.parse_args()

    model_path = args.path if args.path else find_model_file("best_model_xgboost.joblib")
    print(f"Загружаю артефакт: {model_path}")
    art = load_artifact(model_path)

    estimator = art.get("estimator")
    if estimator is None:
        raise ValueError("В артефакте отсутствует ключ 'estimator'")
    preprocessor = art.get("preprocessor")
    if preprocessor is None:
        raise ValueError("В артефакте отсутствует ключ 'preprocessor'")

    importances = extract_importances(estimator)
    feat_names = get_feature_names(preprocessor)
    top_df = top_importances(feat_names, importances, top_n=args.top)
    print("Топ признаков по важности:")
    print(top_df.to_string(index=False))

    plot_top_importances(top_df, args.out)


if __name__ == "__main__":
    main()
