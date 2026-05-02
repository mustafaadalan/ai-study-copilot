import os
from contextlib import contextmanager
from pathlib import Path

import mysql.connector
from mysql.connector import Error as MySQLError

# Proje kökünde .env varsa yükle (python-dotenv)
try:
    from dotenv import load_dotenv

    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.is_file():
        load_dotenv(_env)
except ImportError:
    pass


def mysql_config():
    """Ortam değişkenlerinden MySQL ayarları. Eksik alan boş string olabilir."""
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", ""),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", ""),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
    }


def get_connection():
    """Tek kullanımlık bağlantı. Kapatmak için conn.close() veya context manager kullan."""
    cfg = mysql_config()
    if not cfg["user"] or not cfg["database"]:
        raise MySQLError(
            "MYSQL_USER ve MYSQL_DATABASE ortam değişkenlerini ayarlayın (veya .env)."
        )
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        collation=cfg["collation"],
        autocommit=False,
    )


@contextmanager
def mysql_cursor(dict_cursor=False):
    """
    Bağlantıyı açıp kapatan bağlam yöneticisi.
    dict_cursor=True ise satırlar dict olarak gelir.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=dict_cursor)
        try:
            yield conn, cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()


def ping():
    """Bağlantının çalıştığını doğrular; başarılıysa True."""
    with mysql_cursor() as (conn, cur):
        cur.execute("SELECT 1")
        return cur.fetchone() is not None


def init_db():
    """Quiz sonuçları tablosunu oluşturur. DB yoksa sessizce geçer."""
    try:
        with mysql_cursor() as (_, cur):
            cur.execute("""
                CREATE TABLE IF NOT EXISTS quiz_results (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    session_id  VARCHAR(64)  NOT NULL,
                    pdf_name    VARCHAR(255),
                    section     VARCHAR(255),
                    page        VARCHAR(20),
                    question    TEXT,
                    correct_answer TEXT,
                    user_answer TEXT,
                    is_correct  BOOLEAN,
                    quiz_type   VARCHAR(20) DEFAULT 'cloze',
                    created_at  TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """)
    except Exception:
        pass


def save_quiz_result(session_id, pdf_name, section, page,
                     question, correct_answer, user_answer,
                     is_correct, quiz_type="cloze"):
    """Tek quiz sonucunu kaydeder. DB yoksa sessizce geçer."""
    try:
        with mysql_cursor() as (_, cur):
            cur.execute("""
                INSERT INTO quiz_results
                    (session_id, pdf_name, section, page, question,
                     correct_answer, user_answer, is_correct, quiz_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, pdf_name, str(section), str(page),
                  question, correct_answer, user_answer,
                  bool(is_correct), quiz_type))
    except Exception:
        pass
