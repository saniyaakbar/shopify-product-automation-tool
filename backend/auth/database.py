import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "shops.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(BASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS shops (
            id INTEGER PRIMARY KEY,
            shop_domain TEXT UNIQUE,
            access_token TEXT,
            installed_at TIMESTAMP
        )
        """
        )
        conn.commit()
    finally:
        conn.close()


# initialize DB on import
init_db()
