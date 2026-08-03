from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json

GROUNDING_CACHE_SCHEMA_VERSION = 1


def build_grounding_cache_signature(
    *,
    base_url: str,
    model_name: str,
    prompt_template: str,
) -> dict[str, str]:
    prompt_sha256 = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    return {
        "base_url": str(base_url),
        "model_name": str(model_name),
        "prompt_sha256": prompt_sha256,
    }


class GroundingResultCache:
    def __init__(self, path: str | Path, signature: dict[str, str]):
        self.path = Path(path)
        self.signature = dict(signature)
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = load_json(self.path)
        except Exception as error:
            print(f"Warning: failed to load grounding cache {self.path}: {error}")
            return

        if not isinstance(payload, dict):
            return
        if payload.get("schema_version") != GROUNDING_CACHE_SCHEMA_VERSION:
            print(f"Ignoring grounding cache with unsupported schema: {self.path}")
            return
        if payload.get("signature") != self.signature:
            print(
                f"Ignoring grounding cache with a different model/prompt signature: {self.path}"
            )
            return

        items = payload.get("items")
        if isinstance(items, dict):
            self._items = {
                str(key): dict(value)
                for key, value in items.items()
                if isinstance(value, dict)
            }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": GROUNDING_CACHE_SCHEMA_VERSION,
            "signature": self.signature,
            "items": self._items,
        }

    def successful_items(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                key: dict(value)
                for key, value in self._items.items()
                if not bool(value.get("vlm_failed", False))
            }

    def record(self, key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._items[str(key)] = dict(result)
            atomic_write_json(self.path, self._payload())

    def counts(self) -> dict[str, int]:
        with self._lock:
            failed = sum(
                bool(item.get("vlm_failed", False)) for item in self._items.values()
            )
            return {
                "total": len(self._items),
                "successful": len(self._items) - failed,
                "failed": failed,
            }
