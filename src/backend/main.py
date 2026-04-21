from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Определяет константы путей
FIRST_MODEL_DIR = os.path.join("models", "first_best")
SECOND_MODEL_DIR = os.path.join("models", "second_best")
BUREAU_CSV_PATH = os.path.join("data", "bureau.csv")


def _read_joblib(path: str) -> Dict[str, Any]:
    """Загружает joblib-артефакт.

    Args:
        path (str): Путь к .joblib файлу.

    Returns:
        Dict[str, Any]: Содержимое артефакта.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат файла не соответствует ожидаемому словарю.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден артефакт модели: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError("Ожидался словарь с артефактами модели (dict).")
    return obj


def _resolve_model_path(dir_path: str) -> str:
    """Возвращает путь к .joblib модели в указанной папке.

    Берёт файлы, начинающиеся с 'best_model'. Если таких нет, выбирает
    первый по алфавиту файл с расширением .joblib.

    Args:
        dir_path (str): Папка, в которой ищется модель.

    Returns:
        str: Абсолютный или относительный путь к файлу модели .joblib.

    Raises:
        FileNotFoundError: Если папка не существует или в ней нет .joblib файлов.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Папка с моделями не найдена: {dir_path}")
    candidates = sorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(".joblib")]
    )
    if not candidates:
        raise FileNotFoundError(f"В папке {dir_path} нет файлов .joblib")
    best = [f for f in candidates if f.startswith("best_model")]
    fname = best[0] if best else candidates[0]
    return os.path.join(dir_path, fname)


class TorchMLPProxy:
    """Лёгкая обёртка для инференса Torch-модели из артефакта без жёсткой зависимости на torch."""

    def __init__(
        self, state_dict: Dict[str, np.ndarray], model_params: Dict[str, Any]
    ) -> None:
        """Инициализирует прокси для Torch-модели.

        Args:
            state_dict (Dict[str, np.ndarray]): Состояние весов модели (в виде numpy-массивов).
            model_params (Dict[str, Any]): Параметры архитектуры (input_dim, hidden_dims, dropout).
        """
        self.state_dict = state_dict
        self.model_params = model_params
        self._torch = None
        self._nn = None
        self._model = None

    def _ensure_model(self) -> None:
        """Выполняет ленивую инициализацию torch-модели.

        Raises:
            ImportError: Если пакет torch не установлен.
            RuntimeError: Если отсутствуют необходимые параметры архитектуры.
        """
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except Exception as e:
            raise ImportError(
                "Для инференса Torch-модели требуется установленный пакет 'torch'."
            ) from e

        self._torch = torch
        self._nn = nn

        input_dim = int(self.model_params.get("input_dim"))
        hidden_dims = list(self.model_params.get("hidden_dims", [128, 64]))
        dropout = float(self.model_params.get("dropout", 0.2))
        if not input_dim:
            raise RuntimeError("В артефакте Torch отсутствует корректный 'input_dim'.")

        # Построим простую MLP: [Linear-ReLU-Dropout]xN + Linear(->1)
        layers: List[nn.Module] = []  # type: ignore[name-defined]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        model = nn.Sequential(*layers)

        # Загрузим веса из numpy-массивов
        sd = {}
        for k, v in self.state_dict.items():
            sd[k] = torch.from_numpy(v)  # type: ignore[name-defined]
        model.load_state_dict(sd)  # type: ignore[arg-type]
        model.eval()
        self._model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Считает вероятности положительного класса для входа X.

        Args:
            X (np.ndarray): Матрица признаков (N, D).

        Returns:
            np.ndarray: Вероятности положительного класса формы (N,).
        """
        self._ensure_model()
        assert self._torch is not None
        model = self._model
        torch = self._torch
        assert model is not None

        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            logits = model(X_t).squeeze(1)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba


class ModelArtifact:
    """Инкапсулирует препроцессор и модель (sklearn или torch)."""

    def __init__(self, artifact: Dict[str, Any]) -> None:
        """Инициализирует артефакт модели.

        Args:
            artifact (Dict[str, Any]): Загруженный словарь артефактов.
        """
        self.artifact = artifact
        self.preprocessor = artifact.get("preprocessor", None)
        self.estimator_type: str = artifact.get("estimator_type", "sklearn")
        self._estimator = None
        self._torch_proxy: Optional[TorchMLPProxy] = None

        if self.estimator_type == "sklearn":
            self._estimator = artifact.get("estimator", None)
        elif self.estimator_type == "torch":
            state = artifact.get("torch_state_dict", None)
            params = artifact.get("torch_model_params", None)
            if state is None or params is None:
                raise ValueError(
                    "Для Torch-артефакта требуются 'torch_state_dict' и 'torch_model_params'."
                )
            self._torch_proxy = TorchMLPProxy(state_dict=state, model_params=params)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.estimator_type}")

        if self.preprocessor is None:
            raise ValueError("В артефакте отсутствует препроцессор 'preprocessor'.")

    def expected_columns(self) -> List[str]:
        """Возвращает список ожидаемых входных колонок согласно препроцессору.

        Returns:
            List[str]: Список имён колонок.
        """
        cols = getattr(self.preprocessor, "feature_names_in_", None)
        if cols is None:
            # Падение на ранней стадии: корректный препроцессор sklearn>=1.0 должен иметь feature_names_in_
            raise RuntimeError(
                "Препроцессор не содержит 'feature_names_in_'. Проверьте версию sklearn и артефакт."
            )
        return list(cols)


    def _df_from_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Формирует DataFrame с ровно ожидаемыми колонками из словаря признаков.

        Недостающие значения заполняются NaN, лишние признаки отбрасываются.

        Args:
            features (Dict[str, Any]): Словарь входных признаков.

        Returns:
            pd.DataFrame: Однострочный DataFrame с нужными колонками.
        """
        cols = self.expected_columns()
        row = {c: features.get(c, np.nan) for c in cols}
        return pd.DataFrame([row], columns=cols)

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Предсказывает вероятность положительного класса для одной заявки.

        Args:
            features (Dict[str, Any]): Словарь признаков.

        Returns:
            float: Вероятность положительного класса.
        """
        X_df = self._df_from_features(features)
        X_proc = self.preprocessor.transform(X_df)  # type: ignore[attr-defined]
        if self.estimator_type == "sklearn":
            assert self._estimator is not None
            proba = self._estimator.predict_proba(X_proc)[:, 1]
            return float(proba[0])
        elif self.estimator_type == "torch":
            assert self._torch_proxy is not None
            proba = self._torch_proxy.predict_proba(np.asarray(X_proc))
            return float(proba[0])
        raise RuntimeError("Неизвестный тип модели при инференсе.")


from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Загружает переменные окружения
load_dotenv()

# Создаёт соединение с базой данных
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def _load_bureau_df(path: str) -> Optional[pd.DataFrame]:
    """Загружает bureau.csv, если существует.

    Args:
        path (str): Путь к CSV.

    Returns:
        Optional[pd.DataFrame]: Загруженный датафрейм или None, если файл отсутствует.
    """
    query = text("SELECT * FROM bureau")
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        # Нормализует имена столбцов к нижнему регистру
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из базы данных: {e}")
        return None


def _extract_features_from_bureau(
    df_bureau: pd.DataFrame, sk_id_curr: int, expected_cols: List[str]
) -> Optional[Dict[str, Any]]:
    """Извлекает признаки для указанного sk_id_curr из bureau.csv, приводя к ожидаемым колонкам.

    Args:
        df_bureau (pd.DataFrame): Датафрейм бюро.
        sk_id_curr (int): Идентификатор клиента.
        expected_cols (List[str]): Список колонок, ожидаемых моделью.

    Returns:
        Optional[Dict[str, Any]]: Словарь признаков или None, если id не найден.
    """
    if "sk_id_curr" not in df_bureau.columns:
        return None
    rows = df_bureau.loc[df_bureau["sk_id_curr"] == sk_id_curr]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    # Оставляет только нужные признаки и добавляет недостающие как NaN
    features = {c: row.get(c, np.nan) for c in expected_cols}
    return features


class PredictRequest(BaseModel):
    """Описывает запрос на предсказание."""

    sk_id_curr: Optional[int] = None
    features: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    """Ответ сервиса: скор (вероятность одобрения) и вероятности по классам."""

    score: float
    predict_proba: Dict[str, float]
    model_used: str
    source: str  # 'payload' | 'bureau'


class PrefillResponse(BaseModel):
    """Описывает ответ сервиса для автозаполнения признаков по sk_id_curr."""

    found: bool
    features: Optional[Dict[str, Any]] = None


class SchemaResponse(BaseModel):
    """Описывает схему ожидаемых признаков для обеих моделей."""

    first_expected_columns: List[str]
    second_expected_columns: List[str]



def create_app() -> FastAPI:
    """Создаёт и настраивает FastAPI-приложение.

    Returns:
        FastAPI: Приложение FastAPI.
    """
    app = FastAPI(title="Scoring Service", version="1.0.0")

    # Разрешает обращения со Streamlit (локально и извне) с помощью CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Загружает артефакты моделей
    try:
        first_path = _resolve_model_path(FIRST_MODEL_DIR)
        first_art = ModelArtifact(_read_joblib(first_path))
    except Exception as e:
        raise RuntimeError(
            f"Ошибка загрузки первой модели из {FIRST_MODEL_DIR}: {e}"
        ) from e

    try:
        second_path = _resolve_model_path(SECOND_MODEL_DIR)
        second_art = ModelArtifact(_read_joblib(second_path))
    except Exception as e:
        # Допускаем отсутствие второй модели, но предупреждаем
        second_art = None  # type: ignore[assignment]
        print(f"[WARN] Не удалось загрузить вторую модель из {SECOND_MODEL_DIR}: {e}")

    # Загружает данные бюро из базы данных
    bureau_df = _load_bureau_df("")

    @app.get("/schema", response_model=SchemaResponse)
    def schema() -> SchemaResponse:
        """Возвращает списки ожидаемых колонок для первой и второй моделей.

        Returns:
            SchemaResponse: Ожидаемые признаки для обеих моделей.
        """
        first_cols = first_art.expected_columns()
        second_cols = second_art.expected_columns() if second_art is not None else []
        return SchemaResponse(
            first_expected_columns=first_cols, second_expected_columns=second_cols
        )

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Проверяет здоровье сервиса.

        Returns:
            Dict[str, str]: Статус OK.
        """
        return {"status": "ok"}

    @app.get("/prefill/{sk_id}", response_model=PrefillResponse)
    def prefill(sk_id: int) -> PrefillResponse:
        """Возвращает признаки из bureau.csv для указанного sk_id (для автозаполнения на фронте).

        Args:
            sk_id (int): Идентификатор клиента.

        Returns:
            PrefillResponse: Флаги наличия и словарь признаков (если найден).
        """
        if bureau_df is None:
            return PrefillResponse(found=False, features=None)
        # Предпочитает использовать вторую модель для бюро
        art = second_art if second_art is not None else first_art
        try:
            feats = _extract_features_from_bureau(
                bureau_df, sk_id, art.expected_columns()
            )
        except Exception:
            feats = None
        if feats is None:
            return PrefillResponse(found=False, features=None)
        return PrefillResponse(found=True, features=feats)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Выполняет предсказание вероятности.

        Логика:
        - Если указан sk_id_curr и доступна вторая модель: пытаемся найти запись в bureau.csv.
          - Если id найден: используем вторую модель и признаки из bureau (source='bureau').
          - Если id не найден: выбрасываем 404 с сообщением «такого id нет, заполните поля вручную».
        - Если sk_id_curr не указан: используем первую модель и признаки из payload (source='payload').

        Args:
            req (PredictRequest): Запрос с опциональными sk_id_curr и признаками.

        Returns:
            PredictResponse: Скор (вероятность одобрения), вероятности по классам и метаданные.

        Raises:
            HTTPException: При отсутствии нужных данных или моделей.
        """
        # Обрабатывает ветку с указанным sk_id_curr: использует вторую модель (если доступна)
        if req.sk_id_curr is not None:
            if second_art is None:
                raise HTTPException(
                    status_code=400,
                    detail="Модель для работы по sk_id_curr (second_best) недоступна.",
                )
            if bureau_df is None:
                raise HTTPException(
                    status_code=400,
                    detail="Файл bureau.csv не найден. Невозможно авто-заполнение.",
                )
            feats = _extract_features_from_bureau(
                bureau_df, req.sk_id_curr, second_art.expected_columns()
            )
            if feats is None:
                # Если не нашли в бюро — если переданы признаки вручную, используем их для второй модели
                if req.features is not None:
                    try:
                        proba = second_art.predict_proba(req.features)
                        return PredictResponse(
                            score=(1.0 - proba),
                            predict_proba={
                                "class_0": float(1.0 - proba),
                                "class_1": float(proba),
                            },
                            model_used="second_best",
                            source="payload",
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Ошибка предсказания по переданным признакам: {e}",
                        )
                # Иначе просим заполнить вручную
                raise HTTPException(
                    status_code=404, detail="Такого id нет, заполните поля вручную."
                )
            proba = second_art.predict_proba(feats)
            return PredictResponse(
                score=(1.0 - proba),
                predict_proba={"class_0": float(1.0 - proba), "class_1": float(proba)},
                model_used="second_best",
                source="bureau",
            )

        # Обрабатывает ветку без sk_id_curr: использует первую модель и признаки из запроса
        if req.features is None:
            raise HTTPException(
                status_code=400,
                detail="Не переданы признаки (features) для предсказания.",
            )
        proba = first_art.predict_proba(req.features)
        return PredictResponse(
            score=(1.0 - proba),
            predict_proba={"class_0": float(1.0 - proba), "class_1": float(proba)},
            model_used="first_best",
            source="payload",
        )

    return app


app = create_app()