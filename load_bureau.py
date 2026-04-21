"""
Загрузка данных из bureau.csv в таблицу bureau (PostgreSQL).
Столбец TARGET пропускается.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Загружаем переменные окружения из .env файла
load_dotenv()


def get_db_connection_params() -> Dict[str, Any]:
    """Формирует параметры подключения к базе данных.

    Значения берутся из переменных окружения или используются значения по умолчанию.

    Returns:
        Dict[str, Any]: Словарь с ключами database, user, password, host, port.
    """
    return {
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST", "172.22.146.40"),
        "port": 5432,
    }


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame.

    Args:
        file_path (str): Путь к файлу bureau.csv.

    Returns:
        pd.DataFrame: Датафрейм с данными из файла.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    return pd.read_csv(file_path)


def prepare_tuples(df: pd.DataFrame) -> Tuple[List[Tuple], List[str]]:
    """Удаляет колонку TARGET и преобразует DataFrame в список кортежей.

    Args:
        df (pd.DataFrame): Исходный датафрейм.

    Returns:
        Tuple[List[Tuple], List[str]]:
            - Список кортежей для вставки в БД.
            - Список имён колонок (без TARGET) в нижнем регистре.
    """
    # Исключаем TARGET
    cols = [col for col in df.columns if col != "TARGET"]
    bureau = df[cols]

    # Формируем список кортежей
    tuples = list(bureau.itertuples(index=False, name=None))

    # Имена колонок в нижнем регистре для совместимости с PostgreSQL
    col_names = [col.lower() for col in cols]

    return tuples, col_names


def insert_data(
    conn: psycopg2.extensions.connection,
    table_name: str,
    columns: List[str],
    data: List[Tuple],
    page_size: int = 10000,
) -> None:
    """Выполняет пакетную вставку данных в таблицу PostgreSQL.

    Args:
        conn: Соединение с БД.
        table_name (str): Имя таблицы.
        columns (List[str]): Список имён колонок.
        data (List[Tuple]): Список кортежей с данными.
        page_size (int): Размер батча для execute_values.
    """
    cur = conn.cursor()
    try:
        col_names_sql = ", ".join(f'"{col}"' for col in columns)
        query = f"INSERT INTO {table_name} ({col_names_sql}) VALUES %s"
        execute_values(cur, query, data, page_size=page_size)
    finally:
        cur.close()


def main() -> None:
    """Основная функция загрузки данных из bureau.csv в PostgreSQL."""
    # Путь к файлу данных
    csv_path = os.getenv("BUREAU_CSV_PATH", "./data/bureau.csv")

    # 1. Загрузка DataFrame
    try:
        df = load_dataframe(csv_path)
        print(f"Загружено {len(df)} записей из {csv_path}")
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        return

    # 2. Подготовка данных
    tuples, columns = prepare_tuples(df)
    if not tuples:
        print("Нет данных для вставки (после удаления TARGET).")
        return

    # 3. Подключение к БД
    params = get_db_connection_params()
    try:
        conn = psycopg2.connect(**params)
        print("Подключение к БД установлено")
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return

    # 4. Вставка данных
    try:
        insert_data(conn, "bureau", columns, tuples)
        conn.commit()
        print(f"Вставлено {len(tuples)} записей в таблицу bureau")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при вставке данных: {e}")
    finally:
        conn.close()
        print("Соединение закрыто")


if __name__ == "__main__":
    main()