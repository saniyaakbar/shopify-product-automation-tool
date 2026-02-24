from typing import Optional
from datetime import datetime
from .database import get_connection


def save_shop(shop: str, token: str) -> None:
    """Insert or update shop token and timestamp."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        now = datetime.utcnow()
        # Try insert, if conflict update
        cur.execute(
            """
            INSERT INTO shops (shop_domain, access_token, installed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(shop_domain) DO UPDATE SET access_token=excluded.access_token, installed_at=excluded.installed_at
            """,
            (shop, token, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_shop_token(shop: str) -> Optional[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT access_token FROM shops WHERE shop_domain = ?", (shop,))
        row = cur.fetchone()
        if not row:
            return None
        return row["access_token"]
    finally:
        conn.close()
