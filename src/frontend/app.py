from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

def get_backend_url() -> str:
    """Возвращает URL бэкенда из переменной окружения или состояния Streamlit.

    Returns:
        str: Базовый URL FastAPI.
    """
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")
    return st.session_state.get("backend_url", "http://backend:8000").rstrip("/")


@st.cache_data(show_spinner=False)
def fetch_schema(base_url: str) -> Tuple[List[str], List[str]]:
    """Возвращает списки ожидаемых колонок для обеих моделей с бэкенда.

    Args:
        base_url (str): Базовый URL FastAPI.

    Returns:
        Tuple[List[str], List[str]]: (first_expected_columns, second_expected_columns)
    """
    try:
        r = requests.get(f"{base_url}/schema", timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("first_expected_columns", []), js.get(
            "second_expected_columns", []
        )
    except Exception:
        return [], []



@st.cache_data(show_spinner=False)
def load_sample_application(
    path: str = "data/application.csv",
) -> Optional[pd.DataFrame]:
    """Загружает пример данных application.csv для определения типов столбцов и значений по умолчанию.

    Args:
        path (str): Путь к CSV.

    Returns:
        Optional[pd.DataFrame]: Датафрейм с примерами или None при отсутствии файла.
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return None


def default_value(series: pd.Series) -> Any:
    """Подбирает разумное значение по умолчанию для столбца.

    Args:
        series (pd.Series): Серия столбца.

    Returns:
        Any: Значение по умолчанию для поля формы.
    """
    if pd.api.types.is_bool_dtype(series):
        return (
            bool(series.mode(dropna=True).iloc[0])
            if not series.dropna().empty
            else False
        )
    if pd.api.types.is_integer_dtype(series):
        return int(series.median()) if not series.dropna().empty else 0
    if pd.api.types.is_float_dtype(series):
        return float(series.median()) if not series.dropna().empty else 0.0
    # для object/category берём наиболее частое
    if not series.dropna().empty:
        return str(series.mode(dropna=True).iloc[0])
    return ""


def infer_field_kind(series: Optional[pd.Series]) -> str:
    """Определяет тип виджета формы для столбца.

    Args:
        series (Optional[pd.Series]): Серия с типом столбца (может быть None).

    Returns:
        str: Один из 'number_int', 'number_float', 'checkbox', 'text'.
    """
    if series is None:
        return "text"
    if pd.api.types.is_bool_dtype(series):
        return "checkbox"
    if pd.api.types.is_integer_dtype(series):
        return "number_int"
    if pd.api.types.is_float_dtype(series):
        return "number_float"
    return "text"


def render_features_form(
    columns: List[str],
    sample_df: Optional[pd.DataFrame],
    prefill: Optional[Dict[str, Any]] = None,
    target_col: str = "target",
) -> Dict[str, Any]:
    """Строит форму ввода признаков и возвращает словарь заполненных значений.

    Args:
        columns (List[str]): Имена признаков, ожидаемых моделью.
        sample_df (Optional[pd.DataFrame]): Пример данных для типов и дефолтов.
        prefill (Optional[Dict[str, Any]]): Автозаполненные значения (из bureau), если есть.
        target_col (str): Имя целевой колонки, чтобы исключить из формы.

    Returns:
        Dict[str, Any]: Словарь признаков, заполненный пользователем.
    """
    feats: Dict[str, Any] = {}
    prefill = prefill or {}

    # Для удобства разбиваем поля на две колонки
    col_left, col_right = st.columns(2)

    for i, c in enumerate(columns):
        if c == target_col:
            continue
        series = (
            sample_df[c] if (sample_df is not None and c in sample_df.columns) else None
        )

        # Отображаем подписи для некоторых полей
        label_map = {
            "name_contract_type": "тип кредита",
            "flag_own_car": "владеет автомобилем",
            "flag_own_realty": "владеет недвижимостью",
            "amt_income_total": "годовой доход",
            "amt_credit": "сумма кредита",
            "name_education_type": "образование",
            "name_family_status": "семейное положение",
            "occupation_type": "профессия",
            "has_children": "наличие детей",
            "flag_mobil": "указан мобильный телефон",
            "age": "возраст",
            "years_employed": "текущий стаж",
        }
        label = label_map.get(c, c)

        # Определяем значение по умолчанию: префилл -> из данных -> эвристика
        default = prefill.get(c)
        if default is None and series is not None:
            default = default_value(series)

        container = col_left if i % 2 == 0 else col_right
        with container:
            # Обрабатываем специальные поля
            if c == "name_contract_type" and series is not None:
                options = sorted([str(x) for x in series.dropna().unique().tolist()])
                idx = (
                    options.index(str(default))
                    if (default is not None and str(default) in options)
                    else (0 if options else 0)
                )
                choice = st.selectbox(
                    label, options=options, index=idx if options else 0
                )
                if choice != "":
                    feats[c] = choice
                continue

            if c == "flag_own_car":
                options = ["Y", "N"]
                def_val = (
                    str(default)
                    if (default is not None and str(default) in options)
                    else "N"
                )
                idx = options.index(def_val)
                choice = st.selectbox(label, options=options, index=idx)
                feats[c] = choice
                continue

            if c == "flag_own_realty":
                options = ["Y", "N"]
                def_val = (
                    str(default)
                    if (default is not None and str(default) in options)
                    else "N"
                )
                idx = options.index(def_val)
                choice = st.selectbox(label, options=options, index=idx)
                feats[c] = choice
                continue

            if c == "name_education_type" and series is not None:
                options = sorted([str(x) for x in series.dropna().unique().tolist()])
                idx = (
                    options.index(str(default))
                    if (default is not None and str(default) in options)
                    else (0 if options else 0)
                )
                choice = st.selectbox(
                    label, options=options, index=idx if options else 0
                )
                if choice != "":
                    feats[c] = choice
                continue

            if c == "name_family_status" and series is not None:
                options = sorted([str(x) for x in series.dropna().unique().tolist()])
                idx = (
                    options.index(str(default))
                    if (default is not None and str(default) in options)
                    else (0 if options else 0)
                )
                choice = st.selectbox(
                    label, options=options, index=idx if options else 0
                )
                if choice != "":
                    feats[c] = choice
                continue

            if c == "occupation_type" and series is not None:
                options = sorted([str(x) for x in series.dropna().unique().tolist()])
                idx = (
                    options.index(str(default))
                    if (default is not None and str(default) in options)
                    else (0 if options else 0)
                )
                choice = st.selectbox(
                    label, options=options, index=idx if options else 0
                )
                if choice != "":
                    feats[c] = choice
                continue

            if c == "has_children":
                # Чекбокс возвращает 0/1
                try:
                    def_bool = bool(int(default)) if default is not None else False
                except Exception:
                    def_bool = bool(default) if default is not None else False
                val = st.checkbox(label, value=def_bool)
                feats[c] = 1 if val else 0
                continue

            if c == "flag_mobil":
                # Чекбокс возвращает 0/1
                try:
                    def_bool = bool(int(default)) if default is not None else False
                except Exception:
                    def_bool = bool(default) if default is not None else False
                val = st.checkbox(label, value=def_bool)
                feats[c] = 1 if val else 0
                continue

            # Применяем общее поведение по типам, если специальных правил нет
            kind = infer_field_kind(series)
            if kind == "checkbox":
                val = st.checkbox(
                    label, value=bool(default) if default is not None else False
                )
                feats[c] = bool(val)
            elif kind == "number_int":
                try:
                    int_default = int(default) if default is not None else 0
                except Exception:
                    int_default = 0
                val = st.number_input(label, value=int_default, step=1)
                feats[c] = int(val)
            elif kind == "number_float":
                try:
                    float_default = float(default) if default is not None else 0.0
                except Exception:
                    float_default = 0.0
                val = st.number_input(
                    label, value=float_default, step=0.1, format="%.6f"
                )
                feats[c] = float(val)
            else:
                val = st.text_input(
                    label, value=str(default) if default is not None else ""
                )
                # Пустые строки не передаём, чтобы на бэке заполнилось NaN
                if val.strip() != "":
                    feats[c] = val
    return feats


def backend_health(base_url: str) -> bool:
    """Проверяет доступность бэкенда.

    Args:
        base_url (str): Базовый URL FastAPI.

    Returns:
        bool: True если /health отвечает OK.
    """
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.ok
    except Exception:
        return False


def api_prefill(base_url: str, sk_id: int) -> Optional[Dict[str, Any]]:
    """Запрашивает автозаполнение признаков по sk_id_curr.

    Args:
        base_url (str): Базовый URL FastAPI.
        sk_id (int): Идентификатор клиента.

    Returns:
        Optional[Dict[str, Any]]: Словарь признаков или None, если не найден.
    """
    try:
        r = requests.get(f"{base_url}/prefill/{sk_id}", timeout=15)
        if not r.ok:
            return None
        js = r.json()
        return js.get("features") if js.get("found") else None
    except Exception:
        return None


def api_predict(
    base_url: str, sk_id: Optional[int], features: Optional[Dict[str, Any]]
) -> Tuple[float, str, str]:
    """Выполняет предсказание на бэкенде.

    Args:
        base_url (str): Базовый URL FastAPI.
        sk_id (Optional[int]): sk_id_curr, если задан.
        features (Optional[Dict[str, Any]]): Признаки.

    Returns:
        Tuple[float, str, str]: (score, model_used, source)

    Raises:
        RuntimeError: В случае ошибки ответа сервера.
    """
    payload: Dict[str, Any] = {}
    if sk_id is not None:
        payload["sk_id_curr"] = sk_id
    if features is not None:
        payload["features"] = features

    r = requests.post(f"{base_url}/predict", json=payload, timeout=60)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Ошибка предсказания: {detail}")
    js = r.json()
    return float(js["score"]), str(js["model_used"]), str(js["source"])


def decision_from_score(score: float, model_used: str) -> str:
    """Возвращает текстовое решение по скору (вероятности одобрения).

    Для первой модели (first_best) пороги по score:
    - score >= 0.6: одобрение
    - 0.3 <= score < 0.6: одобрение 40% от суммы кредита
    - score < 0.3: отказ

    Для второй модели по умолчанию порог score >= 0.5 — одобрение, иначе — отказ.

    Args:
        score (float): Вероятность одобрения (p_good).
        model_used (str): Имя модели (first_best/second_best).

    Returns:
        str: Сообщение для пользователя.
    """
    p_good = float(score)
    if model_used == "first_best":
        if p_good >= 0.6:
            return "Ваша заявка одобрена"
        if 0.3 <= p_good < 0.6:
            return "Ваша заявка одобрена на 40% от суммы кредита"
        return "Ваша заявка отклонена"
    else:
        return "Ваша заявка одобрена" if p_good >= 0.5 else "Ваша заявка отклонена"


def main() -> None:
    """Запускает Streamlit-приложение."""
    st.set_page_config(page_title="Скоринговый сервис", layout="wide")
    st.title("Заявка на кредитный продукт")

    # Отображаем панель настроек
    with st.sidebar:
        st.header("Настройки")
        backend_url_input = st.text_input("URL бэкенда", value=get_backend_url())
        st.session_state["backend_url"] = backend_url_input.strip().rstrip("/")
        is_up = backend_health(get_backend_url())
        st.caption(f"Статус бэкенда: {'🟢 доступен' if is_up else '🔴 недоступен'}")
        st.divider()
        st.caption(
            "Подсказка запуска:\n- Бэкенд: uvicorn src.backend.main:app --reload\n- Фронтенд: streamlit run src/frontend/app.py"
        )

    base_url = get_backend_url()

    # Выбираем сценарий: со sk_id_curr или без него
    st.subheader("Данные заявки")
    sk_id_curr: Optional[int] = None
    with st.container():
        left, right = st.columns([1, 1])
        with left:
            use_sk_id = st.checkbox("Есть в бюро", value=False)
        with right:
            sk_id_val = st.number_input(
                "Номер паспорта", value=0, step=1, min_value=0, disabled=not use_sk_id
            )
            if use_sk_id:
                sk_id_curr = int(sk_id_val)

    first_cols, second_cols = fetch_schema(base_url)
    sample_df = load_sample_application()

    prefill_feats: Optional[Dict[str, Any]] = None
    if sk_id_curr is not None and sk_id_curr > 0:
        if st.button("Поиск информации в бюро"):
            with st.spinner("Ищем в бюро..."):
                prefill_feats = api_prefill(base_url, sk_id_curr)
                if prefill_feats is None:
                    st.warning("Такого клиента нет, заполните поля вручную")
                else:
                    st.success("Данные найдены и подставлены в форму")
        # сохраняем автозаполнение между перерисовками
        if prefill_feats is not None:
            st.session_state["prefill_feats"] = prefill_feats
        elif "prefill_feats" in st.session_state:
            prefill_feats = st.session_state.get("prefill_feats")

    st.subheader("Заявочная форма")
    target_col = "target"

    # Берём поля для формы из application.csv (если доступен). Иначе используем схему бэкенда
    expected_cols: List[str] = []
    if sample_df is not None:
        # Исключаем целевую колонку и служебный sk_id_curr (он вводится отдельно сверху)
        expected_cols = [
            c for c in sample_df.columns if c not in (target_col, "Паспортные данные")
        ]
        st.caption("Поля берутся из НБКИ")
    else:
        # Если application.csv отсутствует — используем схему модели
        if sk_id_curr is not None and sk_id_curr > 0 and len(second_cols) > 0:
            expected_cols = second_cols
            st.caption("Используется схема второй модели (second_best)")
        else:
            expected_cols = first_cols if len(first_cols) > 0 else second_cols
            st.caption("Используется схема первой модели (first_best)")

    with st.form("features_form"):
        features = render_features_form(
            expected_cols, sample_df, prefill=prefill_feats, target_col=target_col
        )
        submitted = st.form_submit_button("Сделать предсказание")

    if submitted:
        try:
            with st.spinner("Считаем предсказание..."):
                score, model_used, source = api_predict(base_url, sk_id_curr, features)
            st.success("Готово")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Скор (вероятность одобрения)", f"{score:.4f}")
                st.write(f"Модель: {model_used} | Источник признаков: {source}")
            with col_b:
                st.info(decision_from_score(score, model_used))
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()