"""Local SQLite demo database tool."""

import sqlite3
from pathlib import Path

from langchain.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DB_PATH = PROJECT_ROOT / "memory" / "demo_business.db"


def init_demo_database() -> None:
    """初始化教学使用的商品数据库（非 Tool，程序初始化函数）。"""
    with sqlite3.connect(DEMO_DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )
            """
        )

        demo_products = [
            ("机械键盘", 299.0, 25),
            ("无线鼠标", 129.0, 40),
            ("27寸显示器", 1599.0, 12),
            ("USB-C扩展坞", 219.0, 18),
            ("笔记本支架", 89.0, 30),
        ]
        cursor.executemany(
            "INSERT OR IGNORE INTO products (name, price, stock) VALUES (?, ?, ?)",
            demo_products,
        )
        connection.commit()


@tool
def query_product(product_name: str) -> str:
    """查询本地商品数据库中的商品价格和库存。当用户询问数据库中某个商品的价格、库存或商品信息时使用。"""
    try:
        with sqlite3.connect(DEMO_DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT name, price, stock FROM products WHERE name LIKE ?",
                (f"%{product_name}%",),
            )
            rows = cursor.fetchall()

        if not rows:
            return f"数据库中没有找到与“{product_name}”相关的商品。"

        output = [
            f"商品：{name}，价格：{price}元，库存：{stock}件"
            for name, price, stock in rows
        ]
        return "\n".join(output)
    except Exception as exc:
        return f"数据库查询失败：{str(exc)}"
