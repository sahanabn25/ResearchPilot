"""Memory (guidelines: short-term AND long-term; say what is stored and where).

ShortTermMemory : per-session, in-process (lives in Streamlit session_state).
                  Holds conversation turns + shared workflow facts.
LongTermMemory  : SQLite file (stdlib, free). Holds
                  - search cache (avoids duplicate retrieval / API calls)
                  - finished runs (mode, topic, summary) for "recent work"
                  - user preferences
No API keys, uploaded-PDF text or questionnaire answers are written here unless a
caller explicitly passes them in a run summary.
"""
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)

_ROLES = {"user", "assistant", "system"}
_MAX_TURN_CHARS = 4000


class ShortTermMemory:
    """Bounded conversation history plus key/value facts shared between agents."""

    def __init__(self, max_turns: int = 20):
        if not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer.")
        self.max_turns = max_turns
        self.turns: list = []
        self.facts: dict = {}

    def add_turn(self, role: str, content: str) -> None:
        if role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}.")
        if not isinstance(content, str):
            raise ValueError("content must be a string.")
        self.turns.append({"role": role, "content": content[:_MAX_TURN_CHARS]})
        self.turns = self.turns[-self.max_turns:]

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def context_summary(self, max_chars: int = 1500) -> str:
        """Compact text of what is currently known, for inclusion in prompts."""
        parts = []
        for key, value in self.facts.items():
            if isinstance(value, (list, tuple, set, dict)):
                parts.append(f"{key}: {len(value)} item(s)")
            else:
                parts.append(f"{key}: {str(value)[:200]}")
        recent = [f"{t['role']}: {t['content'][:200]}" for t in self.turns[-4:]]
        text = "Facts: " + "; ".join(parts) + ("\nRecent: " + " | ".join(recent) if recent else "")
        return text[:max_chars]

    def to_dict(self) -> dict:
        return {"max_turns": self.max_turns, "turns": list(self.turns), "facts": dict(self.facts)}

    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory":
        mem = cls(max_turns=data.get("max_turns", 20))
        mem.turns = list(data.get("turns", []))[-mem.max_turns:]
        mem.facts = dict(data.get("facts", {}))
        return mem


def search_key(topic: str, limit: int, sources) -> str:
    topic_norm = re.sub(r"\s+", " ", topic.strip().lower())
    return f"{topic_norm}|{limit}|{','.join(sorted(sources))}"


class LongTermMemory:
    """SQLite-backed memory. Use path=':memory:' for tests."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or config.MEMORY_DB_PATH
        self._lock = threading.Lock()
        self._mem_conn = None
        if self.path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._init_schema()

    def _connect(self):
        return self._mem_conn or sqlite3.connect(self.path)

    def _execute(self, sql: str, params: tuple = (), fetch: str = "none"):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params)
                result = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else cur.lastrowid
                conn.commit()
                return result
            finally:
                if conn is not self._mem_conn:
                    conn.close()

    def _init_schema(self) -> None:
        for sql in (
            "CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, topic TEXT, "
            "payload TEXT, created_at TEXT)",
            "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT, "
            "topic TEXT, summary TEXT, created_at TEXT)",
            "CREATE TABLE IF NOT EXISTS prefs (key TEXT PRIMARY KEY, value TEXT)",
        ):
            self._execute(sql)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ---- search cache ----
    def save_search(self, key: str, topic: str, papers: list, source_status: dict) -> None:
        payload = json.dumps({"papers": papers, "source_status": source_status}, ensure_ascii=False)
        self._execute("INSERT OR REPLACE INTO search_cache VALUES (?,?,?,?)",
                      (key, topic, payload, self._now().isoformat()))

    def get_search(self, key: str, max_age_days: Optional[int] = None) -> Optional[dict]:
        row = self._execute("SELECT payload, created_at FROM search_cache WHERE key=?", (key,), fetch="one")
        if not row:
            return None
        if max_age_days is not None:
            created = datetime.fromisoformat(row[1])
            if self._now() - created > timedelta(days=max_age_days):
                return None
        try:
            return json.loads(row[0])
        except ValueError:
            log.warning("corrupt search-cache row ignored")
            return None

    # ---- runs ----
    def save_run(self, mode: str, topic: str, summary: Optional[dict] = None) -> int:
        if mode not in {"topic", "upload"}:
            raise ValueError("mode must be 'topic' or 'upload'.")
        return self._execute("INSERT INTO runs (mode, topic, summary, created_at) VALUES (?,?,?,?)",
                             (mode, topic, json.dumps(summary or {}, ensure_ascii=False), self._now().isoformat()))

    def recent_runs(self, n: int = 5) -> list:
        rows = self._execute("SELECT id, mode, topic, summary, created_at FROM runs "
                             "ORDER BY id DESC LIMIT ?", (int(n),), fetch="all")
        return [{"id": r[0], "mode": r[1], "topic": r[2], "summary": json.loads(r[3] or "{}"),
                 "created_at": r[4]} for r in rows]

    # ---- preferences ----
    def set_pref(self, key: str, value: Any) -> None:
        self._execute("INSERT OR REPLACE INTO prefs VALUES (?,?)", (key, json.dumps(value)))

    def get_pref(self, key: str, default: Any = None) -> Any:
        row = self._execute("SELECT value FROM prefs WHERE key=?", (key,), fetch="one")
        return json.loads(row[0]) if row else default
