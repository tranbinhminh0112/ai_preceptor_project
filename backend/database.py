"""
SQLite store for the saved question bank.

One table `questions` holds every saved item along with its difficulty and
metadata. Lists (options, knowledge_source) are stored as JSON strings.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from config import QUESTION_DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic            TEXT,
    question_type    TEXT,
    question         TEXT NOT NULL,
    answer           TEXT,
    options          TEXT,          -- JSON list (MCQ only)
    correct_answer   TEXT,
    difficulty_level TEXT,
    domain_knowledge TEXT,
    knowledge_source TEXT,          -- JSON list
    created_at       TEXT
);
"""


def _connect():
    conn = sqlite3.connect(QUESTION_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_question(item: dict, topic: str = "") -> int:
    """Insert a single question item. Returns the new row id."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO questions
                (topic, question_type, question, answer, options, correct_answer,
                 difficulty_level, domain_knowledge, knowledge_source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic,
                item.get("question_type", ""),
                item.get("question", ""),
                item.get("answer", ""),
                json.dumps(item.get("options", []), ensure_ascii=False),
                item.get("correct_answer", ""),
                item.get("difficulty_level", ""),
                item.get("domain_knowledge", ""),
                json.dumps(item.get("knowledge_source", []), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cur.lastrowid


def save_many(items, topic: str = "") -> int:
    """Save a list of items. Returns how many were saved."""
    count = 0
    for item in items:
        save_question(item, topic=topic)
        count += 1
    return count


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("options", "knowledge_source"):
        try:
            d[key] = json.loads(d.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            d[key] = []
    return d


def fetch_questions(difficulty: str = None, question_type: str = None, search: str = None):
    """Return saved questions, newest first, with optional filters."""
    init_db()
    clauses, params = [], []
    if difficulty and difficulty != "All":
        clauses.append("difficulty_level = ?")
        params.append(difficulty)
    if question_type and question_type != "All":
        clauses.append("question_type = ?")
        params.append(question_type)
    if search:
        clauses.append("(question LIKE ? OR answer LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM questions{where} ORDER BY id DESC"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_questions() -> int:
    init_db()
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]


def delete_question(qid: int):
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM questions WHERE id = ?", (qid,))
        conn.commit()


def clear_all():
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM questions")
        conn.commit()


def export_xlsx() -> bytes:
    """Return all saved questions as an .xlsx file (bytes), ready to download."""
    import io

    import pandas as pd

    rows = fetch_questions()  # all questions, newest first
    records = []
    for r in rows:
        records.append({
            "ID": r.get("id"),
            "Topic": r.get("topic", ""),
            "Type": r.get("question_type", ""),
            "Question": r.get("question", ""),
            "Answer": r.get("answer", ""),
            "Options": " | ".join(r.get("options", []) or []),
            "Correct answer": r.get("correct_answer", ""),
            "Difficulty": r.get("difficulty_level", ""),
            "Domain": r.get("domain_knowledge", ""),
            "Knowledge source": " ; ".join(r.get("knowledge_source", []) or []),
            "Created at": r.get("created_at", ""),
        })

    df = pd.DataFrame(records)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Questions")
    return buffer.getvalue()
