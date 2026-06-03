"""GPT-5.5（OpenAI 兼容网关）调用封装。

第三方网关（api.mindracode.com）对参数支持不确定，因此 chat() 内置三级降级并记忆能力：
  - token 上限：max_completion_tokens → max_tokens
  - 结构化输出：json_schema(strict) → json_object → 纯文本抽 JSON
  - reasoning_effort：失败则丢弃
  - 端点 /v1：get_client 默认补 /v1（裸域名时）

确切行为请先用 smoke_test.py 实测。
"""
import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

import config

# 能力记忆（首次探测后固定，避免每次重试）
_TOKEN_PARAM = None      # "max_completion_tokens" | "max_tokens"
_JSON_CAP = None         # "json_schema" | "json_object" | "text"
_EFFORT_OK = True        # reasoning_effort 是否被接受
_client = None


# --- 客户端 -------------------------------------------------------------
def _normalize_base_url(url: str) -> str:
    """裸域名（无路径）补 /v1；已带版本路径则原样。"""
    p = urlparse(url)
    if p.path in ("", "/"):
        return url.rstrip("/") + "/v1"
    return url


def get_client(base_url: str | None = None):
    global _client
    if base_url is not None:  # smoke_test 探测不同 URL 时绕过缓存
        from openai import OpenAI
        return OpenAI(base_url=_normalize_base_url(base_url), api_key=config.LLM_API_KEY)
    if _client is None:
        from openai import OpenAI
        if not config.LLM_API_KEY:
            raise RuntimeError("未设置 LLM_API_KEY，请在 reverse-lookup/.env 中配置")
        _client = OpenAI(base_url=_normalize_base_url(config.LLM_BASE_URL), api_key=config.LLM_API_KEY)
    return _client


# --- 消息内容构造 -------------------------------------------------------
def build_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def build_image_content(path, detail: str = "auto", max_px: int = 768) -> dict:
    """图片 → 缩放 ≤max_px → JPEG → base64 data URL 的 image_url content。"""
    img = Image.open(Path(path)).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}


# --- JSON 抽取 ----------------------------------------------------------
def extract_json(text: str):
    """从可能含 markdown 围栏/多余文字的回复中抽出首个 JSON 对象或数组。"""
    if not text or not text.strip():
        raise ValueError("LLM 返回为空")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 扫描首个平衡的 {...} 或 [...]（粗略跳过字符串内的括号）
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"无法从回复中解析 JSON：{text[:200]}")


# --- 核心 chat（带降级与重试）------------------------------------------
def _response_format(json_mode: bool, json_schema):
    global _JSON_CAP
    if not json_mode and not json_schema:
        return None
    cap = _JSON_CAP
    if cap is None:  # 未探测：有 schema 先试 schema，否则 json_object
        cap = "json_schema" if json_schema else "json_object"
    if cap == "json_schema" and json_schema:
        return {"type": "json_schema", "json_schema": {**json_schema, "strict": True}}
    if cap in ("json_schema", "json_object"):
        return {"type": "json_object"}
    return None  # "text"：靠 extract_json 兜底


def _downgrade_json():
    global _JSON_CAP
    _JSON_CAP = {"json_schema": "json_object", "json_object": "text", None: "json_object"}.get(_JSON_CAP, "text")


def _stream_chat(client, kwargs):
    """流式兜底：网关偶发只肯流式返回时，用 stream=True 拼装完整内容与 usage。"""
    sk = dict(kwargs)
    sk["stream"] = True
    try:
        sk["stream_options"] = {"include_usage": True}
    except Exception:  # noqa: BLE001
        pass
    parts = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for chunk in client.chat.completions.create(**sk):
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                parts.append(delta.content)
        u = getattr(chunk, "usage", None)
        if u:
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                     "total_tokens": getattr(u, "total_tokens", 0) or 0}
    return "".join(parts), usage


def chat(messages, *, max_tokens: int = 1024, json_mode: bool = False,
         json_schema=None, effort: str | None = None) -> str:
    """调用 GPT-5.5，返回回复文本。失败时逐项降级重试。"""
    text, _ = chat_with_usage(messages, max_tokens=max_tokens, json_mode=json_mode,
                              json_schema=json_schema, effort=effort)
    return text


def chat_with_usage(messages, *, max_tokens: int = 1024, json_mode: bool = False,
                    json_schema=None, effort: str | None = None):
    """同 chat，但额外返回 usage dict（prompt_tokens/completion_tokens/total_tokens）。"""
    global _TOKEN_PARAM, _EFFORT_OK
    from openai import APIConnectionError, BadRequestError, RateLimitError, APIStatusError

    client = get_client()
    for attempt in range(6):  # 至多 6 次：覆盖参数降级 + 网络退避
        kwargs = {"model": config.LLM_MODEL, "messages": messages}
        kwargs[_TOKEN_PARAM or "max_completion_tokens"] = max_tokens
        rf = _response_format(json_mode, json_schema)
        if rf:
            kwargs["response_format"] = rf
        if effort and _EFFORT_OK:
            kwargs["reasoning_effort"] = effort
        try:
            resp = client.chat.completions.create(**kwargs)
            if not hasattr(resp, "choices"):  # 网关偶发改用流式返回 → 改用 stream 重取并拼装
                try:
                    content, usage = _stream_chat(client, kwargs)
                    if content:
                        return content, usage
                except Exception:  # noqa: BLE001
                    pass
                if attempt < 4:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"网关返回非预期结构: {str(resp)[:200]}")
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("回复 content 为空")
            u = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            } if u else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return content, usage
        except BadRequestError as e:
            msg = str(e).lower()
            if _TOKEN_PARAM is None and ("max_completion_tokens" in msg or "max_tokens" in msg or "unsupported" in msg and "token" in msg):
                _TOKEN_PARAM = "max_tokens"
                continue
            if rf is not None and ("response_format" in msg or "json" in msg or "schema" in msg):
                _downgrade_json()
                continue
            if effort and _EFFORT_OK and "reasoning" in msg:
                _EFFORT_OK = False
                continue
            raise
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if isinstance(e, APIStatusError) and status is not None and 400 <= status < 500 and status != 429:
                raise  # 非限流的 4xx 不重试
            if attempt >= 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("chat 多次重试后仍失败")


# --- 三段式高层调用 -----------------------------------------------------
def describe_frames(frame_paths) -> dict:
    """阶段1：关键帧 → 结构化描述 dict。"""
    content = [build_text_content(config.DESCRIBE_PROMPT)]
    for p in frame_paths:
        content.append(build_image_content(p, detail="auto"))
    raw = chat([{"role": "user", "content": content}],
               max_tokens=900, json_mode=True, json_schema=config.QUERY_JSON_SCHEMA, effort="low")
    return extract_json(raw)


def describe_corpus_image(image_path, original_description: str, detail: str = "high"):
    """v2 语料重描述：读「线描图 + 词典原文」→ 结构化描述 dict + usage。"""
    content = [
        build_text_content(config.CORPUS_DESCRIBE_PROMPT + "\n\n【官方打法文字】\n" + (original_description or "")),
        build_image_content(image_path, detail=detail),
    ]
    raw, usage = chat_with_usage([{"role": "user", "content": content}],
                                 max_tokens=700, json_mode=True,
                                 json_schema=config.CORPUS_JSON_SCHEMA, effort="low")
    return extract_json(raw), usage


def _fmt_candidate(c: dict, with_id: bool = True) -> str:
    desc = (c.get("description") or "").replace("\n", "／")
    head = f'id={c["id"]}｜' if with_id else ""
    return f'{head}词：{c.get("words", "")}｜打法：{desc}'


def rerank_text(query_json: dict, candidates: list[dict], top_text: int = 10) -> list[int]:
    """阶段4a：纯文本重排 → 返回排序后的候选 id 列表（≤top_text）。"""
    valid_ids = {c["id"] for c in candidates}
    lines = "\n".join(_fmt_candidate(c) for c in candidates)
    user = (config.RERANK_TEXT_PROMPT % top_text
            + "\n\n【用户动作结构化描述】\n" + json.dumps(query_json, ensure_ascii=False)
            + "\n\n【候选词条】\n" + lines)
    try:
        raw = chat([{"role": "user", "content": user}], max_tokens=400, json_mode=True, effort="low")
        ranking = extract_json(raw).get("ranking", [])
        out, seen = [], set()
        for x in ranking:
            try:
                i = int(x)
            except (ValueError, TypeError):
                continue
            if i in valid_ids and i not in seen:
                out.append(i)
                seen.add(i)
        if out:
            return out[:top_text]
    except Exception as e:  # 解析/调用失败 → 退回召回原序
        print(f"  [rerank_text 降级：{e}]")
    return [c["id"] for c in candidates][:top_text]


def _pick_frames(frame_paths, max_n: int):
    """帧过多时取首/中/尾 ≤max_n 张，省 token。"""
    n = len(frame_paths)
    if n <= max_n:
        return list(frame_paths)
    idxs = sorted({round(i * (n - 1) / (max_n - 1)) for i in range(max_n)})
    return [frame_paths[i] for i in idxs]


def rerank_visual(frame_paths, candidates: list[dict], top_final: int = 5,
                  batch_size: int = 10, max_query_frames: int = 4) -> list[dict]:
    """阶段4b：原始帧 + 候选线描图 → top-K {id,confidence,reason}。"""
    frames = _pick_frames(list(frame_paths), max_query_frames)
    by_id = {c["id"]: c for c in candidates}
    scored: dict[int, dict] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        content = [build_text_content(config.RERANK_VISUAL_PROMPT % top_final),
                   build_text_content("【用户关键帧】")]
        for p in frames:
            content.append(build_image_content(p, detail="auto"))
        content.append(build_text_content("【候选词条】"))
        for c in batch:
            content.append(build_text_content("候选 " + _fmt_candidate(c)))
            content.append(build_image_content(config.resolve_image(c["image_path"]), detail="low"))
        try:
            raw = chat([{"role": "user", "content": content}], max_tokens=800)
            for r in extract_json(raw).get("results", []):
                try:
                    rid = int(r.get("id"))
                except (ValueError, TypeError):
                    continue
                if rid not in by_id:
                    continue
                conf = float(r.get("confidence", 0) or 0)
                if rid not in scored or conf > scored[rid]["confidence"]:
                    scored[rid] = {"id": rid, "confidence": conf, "reason": str(r.get("reason", ""))}
        except Exception as e:
            print(f"  [rerank_visual 批次降级：{e}]")
    if not scored:  # 全失败 → 退回前 top_final 候选；confidence 不可用，避免误显示为 0%
        return [{"id": c["id"], "confidence": None, "reason": "（视觉重排暂不可用，按文本召回顺序展示）"}
                for c in candidates[:top_final]]
    ranked = sorted(scored.values(), key=lambda d: d["confidence"], reverse=True)
    return ranked[:top_final]
