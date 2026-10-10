"""OpenAI chat + embedding wrappers with token accounting and a disk cache for embeddings."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time

import numpy as np
import tiktoken
from openai import OpenAI

from . import config, tracing

_client: OpenAI | None = None
_enc = tiktoken.get_encoding("o200k_base")


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(max_retries=6, timeout=120)
    return _client


def count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def chat(messages: list[dict], *, name: str, tools: list[dict] | None = None, json_mode: bool = False,
         max_tokens: int = 800, context_tokens: int = 0, temperature: float = 0.0, tool_choice=None):
    """One chat completion.  Returns the message object; usage is recorded on the active trace."""
    kwargs = dict(model=config.LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=7)
    if tools:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = True
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    with tracing.op("llm", name, model=config.LLM_MODEL) as o:
        for attempt in range(7):
            try:
                resp = client().chat.completions.create(**kwargs)
                break
            except Exception:  # rate limits / transient API errors beyond the SDK's own retries
                if attempt == 6:
                    raise
                time.sleep(min(30, 2 ** attempt * 2))
        u = resp.usage
        o.input_tokens = u.prompt_tokens
        o.output_tokens = u.completion_tokens
        o.context_tokens = context_tokens
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None):
            o.detail["cached_input_tokens"] = details.cached_tokens
        msg = resp.choices[0].message
        if msg.tool_calls:
            o.detail["tool_calls"] = [tc.function.name for tc in msg.tool_calls]
    return msg


def chat_json(messages: list[dict], *, name: str, max_tokens: int = 600, context_tokens: int = 0) -> dict:
    msg = chat(messages, name=name, json_mode=True, max_tokens=max_tokens, context_tokens=context_tokens)
    try:
        return json.loads(msg.content or "{}")
    except json.JSONDecodeError:
        return {}


class _EmbedCache:
    def __init__(self):
        config.CACHE_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(config.CACHE_DIR / "embeddings.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(f"{config.EMBED_MODEL}|{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
                for k, v in self._db.execute(q, part):
                    out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]):
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                 [(k, v.astype(np.float32).tobytes()) for k, v in items.items()])
            self._db.commit()


_cache: _EmbedCache | None = None


def embed(texts: list[str], *, name: str = "embed", batch: int = 256) -> list[np.ndarray]:
    """Embed texts (cached).  Only cache misses cost tokens; those are recorded on the trace."""
    global _cache
    if _cache is None:
        _cache = _EmbedCache()
    keys = [_cache.key(t) for t in texts]
    found = _cache.get_many(keys)
    missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
    with tracing.op("embed", name, n=len(texts), cache_hits=len(texts) - len(missing)) as o:
        # Report what the embeddings would cost without the local cache, so pipelines compare fairly;
        # the tokens actually billed this run are kept separately.
        o.embed_tokens = sum(count_tokens(t[:30000]) for t in texts)
        o.detail["billed_tokens"] = 0
        seen = set()
        todo = [(k, t) for k, t in missing if not (k in seen or seen.add(k))]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            resp = client().embeddings.create(model=config.EMBED_MODEL, input=[t[:30000] for _, t in part])
            o.detail["billed_tokens"] += resp.usage.total_tokens
            new = {k: np.array(d.embedding, dtype=np.float32) for (k, _), d in zip(part, resp.data)}
            _cache.put_many(new)
            found.update(new)
    return [found[k] for k in keys]


def embed_one(text: str, name: str = "embed_query") -> list[float]:
    return embed([text], name=name)[0].tolist()
