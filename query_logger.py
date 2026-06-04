"""Persistent per-query logging for SignSeek.

Each query gets its own directory under query_logs/:
  - record.json: parameters, step events, raw LLM responses, final results
  - frames/: copied input frames so web temporary uploads remain inspectable
"""
from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

import config


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def candidate_snapshot(candidate: dict) -> dict:
    return {
        "id": candidate.get("id"),
        "words": candidate.get("words", ""),
        "description": candidate.get("description", ""),
        "image_path": candidate.get("image_path", ""),
        "recall_rank": candidate.get("recall_rank"),
        "recall_sim": candidate.get("recall_sim"),
        "frame_score": candidate.get("frame_score"),
        "motion_score": candidate.get("motion_score"),
        "final_recall_score": candidate.get("final_recall_score"),
        "estimated_frames": candidate.get("estimated_frames"),
        "motion_level": candidate.get("motion_level"),
        "motion_tags": candidate.get("motion_tags", []),
        "motion_confidence": candidate.get("motion_confidence"),
        "confidence": candidate.get("confidence"),
        "reason": candidate.get("reason", ""),
    }


class QueryLogger:
    def __init__(self, source: str, meta: dict | None = None, root: Path | None = None):
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.log_id = f"{stamp}_{source}_{uuid.uuid4().hex[:8]}"
        self.dir = (root or config.QUERY_LOGS_DIR) / self.log_id
        self.frames_dir = self.dir / "frames"
        self.path = self.dir / "record.json"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "log_id": self.log_id,
            "source": source,
            "started_at": _now(),
            "meta": _json_safe(meta or {}),
            "input_frames": [],
            "parameters": {},
            "events": [],
            "output": {},
            "status": "running",
        }
        self.flush()

    def flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(_json_safe(self.record), ensure_ascii=False, indent=2), encoding="utf-8")

    def set(self, key: str, value: Any) -> None:
        self.record[key] = _json_safe(value)
        self.flush()

    def event(self, name: str, data: Any | None = None) -> None:
        self.record.setdefault("events", []).append({
            "time": _now(),
            "name": name,
            "data": _json_safe(data or {}),
        })
        self.flush()

    def copy_input_frames(self, frame_paths) -> None:
        frames = []
        for index, src in enumerate([Path(p) for p in frame_paths], start=1):
            suffix = src.suffix.lower() or ".jpg"
            dst = self.frames_dir / f"frame_{index:02d}{suffix}"
            shutil.copy2(src, dst)
            info = {
                "index": index,
                "source_path": str(src),
                "saved_path": str(dst.relative_to(self.dir)),
                "bytes": dst.stat().st_size,
            }
            try:
                with Image.open(dst) as img:
                    info["width"], info["height"] = img.size
                    info["format"] = img.format
            except Exception as exc:  # noqa: BLE001
                info["image_error"] = str(exc)
            frames.append(info)
        self.record["input_frames"] = frames
        self.flush()

    def finish(self, status: str = "ok", error: BaseException | None = None) -> None:
        self.record["status"] = status
        self.record["finished_at"] = _now()
        if error is not None:
            self.record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        self.flush()
