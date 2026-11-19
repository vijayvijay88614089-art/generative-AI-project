from __future__ import annotations
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("INSIGHTFORGE_DB", Path(__file__).resolve().parents[1] / "data" / "insightforge.sqlite3"))


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL, answer TEXT, answer_type TEXT, expression TEXT,
            success INTEGER NOT NULL, elapsed_seconds REAL, attempts INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS saved_prompts (
            id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            name TEXT NOT NULL, prompt TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prompt_comparisons (
            id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            technique TEXT NOT NULL, prompt TEXT NOT NULL, output TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            name TEXT NOT NULL, chunks INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            metrics_json TEXT NOT NULL
        );
        """)


def save_query(result: dict, question: str):
    answer = json.dumps(result.get("value"), ensure_ascii=False, default=str) if result.get("ok") else result.get("error")
    with connect() as db:
        db.execute("INSERT INTO query_history(question,answer,answer_type,expression,success,elapsed_seconds,attempts) VALUES(?,?,?,?,?,?,?)",
                   (question[:2000], answer[:6000], result.get("answer_type"), result.get("expression"), int(bool(result.get("ok"))),
                    float(result.get("elapsed_seconds", 0)), int(result.get("attempts", 0))))


def history(limit: int = 100):
    with connect() as db:
        return db.execute("SELECT created_at,question,answer,success,elapsed_seconds FROM query_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def save_prompt(name: str, prompt: str):
    with connect() as db:
        db.execute("INSERT INTO saved_prompts(name,prompt) VALUES(?,?)", (name[:100], prompt[:10000]))


def prompts():
    with connect() as db:
        return db.execute("SELECT id,created_at,name,prompt FROM saved_prompts ORDER BY id DESC").fetchall()


def save_prompt_comparison(technique: str, prompt: str, output: str):
    with connect() as db:
        db.execute("INSERT INTO prompt_comparisons(technique,prompt,output) VALUES(?,?,?)",
                   (technique[:100], prompt[:10000], output[:12000]))


def record_evaluation(metrics: dict):
    with connect() as db:
        db.execute("INSERT INTO evaluation_runs(metrics_json) VALUES(?)", (json.dumps(metrics, default=str),))
