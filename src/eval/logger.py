import sqlite3
import json
import time

DB_PATH = "eval_results.db"

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                query TEXT,
                answer TEXT,
                context_chunks TEXT,
                faithfulness REAL,
                answer_relevancy REAL,
                correctness REAL
            )
        """)
        
_init_db()

def save_result(result_dict: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO eval_results (
                timestamp, query, answer, context_chunks, 
                faithfulness, answer_relevancy, correctness
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result_dict.get("timestamp", time.time()),
            result_dict.get("query", ""),
            result_dict.get("answer", ""),
            json.dumps(result_dict.get("context_chunks", [])),
            result_dict.get("faithfulness"),
            result_dict.get("answer_relevancy"),
            result_dict.get("correctness")
        ))

def fetch_recent(n: int = 50) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT * FROM eval_results 
            ORDER BY timestamp DESC LIMIT ?
        """, (n,))
        rows = cur.fetchall()
        
    return [dict(row) for row in rows]
