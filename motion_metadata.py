"""Motion/frame metadata for sign retrieval.

The source database does not have a hand-authored frame count.  This module
derives a conservative estimate from existing corpus text and llm_struct, then
uses it as a soft retrieval prior.  It never hard-filters candidates.
"""
from __future__ import annotations

import json
from typing import Any


MOTION_LEVELS = {"unknown", "static", "simple_motion", "path_motion", "multi_step"}

_UNCERTAIN_TOKENS = {"", "uncertain", "未知", "不确定", "无法确定", "看不到", "不可见", "none", "n/a", "na"}
_STATIC_TOKENS = ("静止", "不动", "无明显运动", "保持", "停在", "置于")
_PATH_KEYWORDS = ("从", "至", "到", "经", "顺", "沿", "绕", "轨迹", "前襟线", "弧线", "路线")
_MOVE_KEYWORDS = (
    "移动", "划", "转", "绕", "弯动", "点", "捶", "拍", "向前", "向后", "向下", "向上",
    "上下", "左右", "内外", "移", "摆", "摇", "扇动", "来回", "交替", "搓", "摩擦",
    "伸出", "收回", "张开", "合拢", "打开", "闭合", "翻转", "抖动", "弹动", "敲",
)
_REPEAT_KEYWORDS = ("两下", "几下", "反复", "重复", "来回", "交替", "连续", "多次", "一下")
_UPWARD_KEYWORDS = ("向上", "上移", "上举", "抬起", "由下向上", "往上")
_DOWNWARD_KEYWORDS = ("向下", "下移", "下划", "下落", "放下", "由上向下", "往下")
_LATERAL_KEYWORDS = ("横", "左右", "向左", "向右", "侧", "横向", "平移")
_CIRCLE_KEYWORDS = ("绕", "圈", "圆", "旋转", "转动", "弧形")
_TOUCH_KEYWORDS = ("触", "碰", "贴", "接触", "抵", "按", "点", "搭", "置于", "靠近")
_FACE_ANCHORS = ("下巴", "颈前", "咽喉", "喉", "嘴", "唇", "口", "脸", "面颊", "脸颊", "腮")
_TORSO_ANCHORS = ("肩", "胸", "腰", "腹", "肚", "身体一侧", "体侧")
_WAIST_ANCHORS = ("腰", "腹", "肚")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "，".join(_clean_text(v) for v in value if _clean_text(v))
    return str(value).strip()


def parse_llm_struct(raw: Any) -> dict:
    """Return a dict for llm_struct, tolerating blank or malformed values."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _steps_from_struct(data: dict) -> list[str]:
    steps = data.get("steps") or []
    if not isinstance(steps, list):
        return []
    return [str(s).strip() for s in steps if str(s or "").strip()]


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def _tag_set(text: str) -> set[str]:
    tags: set[str] = set()
    if _has_any(text, _REPEAT_KEYWORDS):
        tags.add("repeat")
    if _has_any(text, _UPWARD_KEYWORDS):
        tags.add("upward")
    if _has_any(text, _DOWNWARD_KEYWORDS):
        tags.add("downward")
    if _has_any(text, _LATERAL_KEYWORDS):
        tags.add("lateral")
    if _has_any(text, _CIRCLE_KEYWORDS):
        tags.add("circle")
    if _has_any(text, _TOUCH_KEYWORDS):
        tags.add("touch")

    face = _has_any(text, _FACE_ANCHORS)
    torso = _has_any(text, _TORSO_ANCHORS)
    waist = _has_any(text, _WAIST_ANCHORS)
    if face and torso:
        tags.add("body_path")
    if face and waist:
        tags.add("face_to_waist")
    return tags


def derive_motion_metadata(description: str = "", llm_struct: Any = None, llm_description: str = "") -> dict:
    """Estimate frame need and motion class from available sign metadata.

    estimated_frames is intentionally capped at 5 because the local UI accepts
    at most five user frames.
    """
    data = parse_llm_struct(llm_struct)
    movement = _clean_text(data.get("movement"))
    location = _clean_text(data.get("location"))
    relation = _clean_text(data.get("two_hand_relation"))
    resembles = _clean_text(data.get("resembles"))
    steps = _steps_from_struct(data)

    text_parts = [description, llm_description, movement, location, relation, resembles, *steps]
    text = "，".join(part for part in (_clean_text(p) for p in text_parts) if part)
    lowered_movement = movement.lower()

    has_steps = len(steps) >= 2
    has_path = _has_any(text, _PATH_KEYWORDS)
    has_motion = _has_any(text, _MOVE_KEYWORDS) or has_path or has_steps
    motion_unknown = lowered_movement in _UNCERTAIN_TOKENS and not has_motion
    static_hint = _has_any(movement or text, _STATIC_TOKENS)

    tags = _tag_set(text)
    if has_steps:
        tags.add("multi_step")
    if has_path:
        tags.add("path")
    if static_hint and not has_motion:
        tags.add("static")

    if motion_unknown and not text:
        motion_level = "unknown"
        estimated_frames = 1
        confidence = 0.15
    elif has_steps and len(steps) >= 3:
        motion_level = "multi_step"
        estimated_frames = min(5, max(3, len(steps)))
        confidence = 0.86
    elif has_steps:
        motion_level = "multi_step"
        estimated_frames = 2
        confidence = 0.78
    elif has_path:
        motion_level = "path_motion"
        estimated_frames = 3 if "body_path" in tags or "face_to_waist" in tags or _has_any(text, _REPEAT_KEYWORDS) else 2
        confidence = 0.72
    elif has_motion:
        motion_level = "simple_motion"
        estimated_frames = 2
        confidence = 0.64
    elif static_hint:
        motion_level = "static"
        estimated_frames = 1
        confidence = 0.7
    elif text:
        motion_level = "static"
        estimated_frames = 1
        confidence = 0.42
    else:
        motion_level = "unknown"
        estimated_frames = 1
        confidence = 0.12

    return {
        "estimated_frames": int(max(1, min(5, estimated_frames))),
        "motion_level": motion_level,
        "motion_tags": sorted(tags),
        "motion_confidence": round(float(confidence), 3),
    }


def ensure_motion_metadata(candidate: dict) -> dict:
    """Return motion metadata stored on a candidate, or derive a fallback."""
    level = candidate.get("motion_level")
    est = candidate.get("estimated_frames")
    if level in MOTION_LEVELS and isinstance(est, int):
        return {
            "estimated_frames": int(max(1, min(5, est))),
            "motion_level": level,
            "motion_tags": list(candidate.get("motion_tags") or []),
            "motion_confidence": float(candidate.get("motion_confidence") or 0),
        }
    return derive_motion_metadata(
        candidate.get("description", ""),
        candidate.get("llm_struct"),
        candidate.get("llm_description", ""),
    )


def frame_match_score(query_frame_count: int, candidate: dict) -> float:
    """Soft prior for matching uploaded frame count with candidate motion needs."""
    frame_count = max(1, min(5, int(query_frame_count or 1)))
    meta = ensure_motion_metadata(candidate)
    estimated = int(meta["estimated_frames"])
    level = meta["motion_level"]
    tags = set(meta.get("motion_tags") or [])

    score = 0.0
    if frame_count == 1:
        if level == "static":
            score += 0.015
        elif level == "simple_motion":
            score += 0.005
        elif level in {"path_motion", "multi_step"}:
            score -= 0.010
    elif frame_count == 2:
        if level in {"simple_motion", "path_motion"}:
            score += 0.020
        elif level == "multi_step":
            score += 0.012
        elif level == "static":
            score -= 0.015
    else:
        if level in {"path_motion", "multi_step"}:
            score += 0.035
        elif level == "simple_motion":
            score += 0.012
        elif level == "static":
            score -= 0.025
        if estimated >= 3:
            score += 0.020
        if tags.intersection({"body_path", "face_to_waist"}):
            score += 0.015

    diff = abs(frame_count - estimated)
    if diff == 0:
        score += 0.008
    elif diff == 1:
        score += 0.003
    elif diff >= 3:
        score -= 0.008

    if level == "unknown":
        score *= 0.35
    return round(max(-0.06, min(0.06, score)), 6)


def derive_query_motion_profile(query_text: str = "", description: dict | None = None) -> dict:
    """Extract coarse motion tags from the user's rendered query text."""
    parts = [_clean_text(query_text)]
    if isinstance(description, dict):
        for key in ("location", "movement", "iconicity", "handshape", "orientation"):
            parts.append(_clean_text(description.get(key)))
    text = "，".join(part for part in parts if part)
    tags = _tag_set(text)
    if _has_any(text, _PATH_KEYWORDS):
        tags.add("path")
    if _has_any(text, _MOVE_KEYWORDS):
        tags.add("moving")
    return {
        "motion_tags": sorted(tags),
        "has_body_path": "body_path" in tags,
        "has_face_to_waist": "face_to_waist" in tags,
        "has_path": "path" in tags,
    }


def motion_alignment_score(query_profile: dict | None, candidate: dict) -> float:
    """Soft prior for matching query motion tags with candidate motion tags."""
    if not query_profile:
        return 0.0
    query_tags = set(query_profile.get("motion_tags") or [])
    if not query_tags:
        return 0.0

    meta = ensure_motion_metadata(candidate)
    candidate_tags = set(meta.get("motion_tags") or [])
    level = meta["motion_level"]
    score = 0.0

    if "face_to_waist" in query_tags:
        if "face_to_waist" in candidate_tags:
            score += 0.060
        elif "body_path" in candidate_tags:
            score += 0.032
        elif level == "static":
            score -= 0.016
    elif "body_path" in query_tags:
        if "body_path" in candidate_tags:
            score += 0.038
        elif level == "static":
            score -= 0.012

    if "path" in query_tags and "path" in candidate_tags:
        score += 0.010
    for tag in ("downward", "upward", "lateral", "circle", "repeat"):
        if tag in query_tags and tag in candidate_tags:
            score += 0.006
    if "touch" in query_tags and "touch" in candidate_tags:
        score += 0.004

    directional = {"downward", "upward", "lateral", "circle"}
    if query_tags.intersection(directional) and not candidate_tags.intersection(directional):
        score -= 0.004

    return round(max(-0.07, min(0.07, score)), 6)


def motion_label(candidate: dict) -> str:
    """Human-readable compact metadata label."""
    meta = ensure_motion_metadata(candidate)
    level_name = {
        "unknown": "动作未知",
        "static": "静态",
        "simple_motion": "简单动作",
        "path_motion": "路径动作",
        "multi_step": "多步动作",
    }.get(meta["motion_level"], meta["motion_level"])
    return f"预计{meta['estimated_frames']}帧 · {level_name}"
