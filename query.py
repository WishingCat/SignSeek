"""见手知意 SignSeek — 在线查询主流程：关键帧 → top-K 候选词条。

  (1) 理解：GPT-5.5 看帧 → 结构化描述
  (2) 渲染：结构化描述 → 贴库措辞的检索文本
  (3) 召回：BGE 向量余弦 → top-recall
  (4a) 文本重排：GPT-5.5 → top-text
  (4b) 视觉重排：GPT-5.5 看帧+候选线描图 → top-final
  (5) 输出：stdout 摘要 + reports/query_*.html 并排报告

用法：
  python3 query.py "../测试题/测试题 1.jpg"
  python3 query.py f1.jpg f2.jpg f3.jpg --top-k 5 --open
  python3 query.py img.jpg --no-visual-rerank      # 省 token 调试
"""
import argparse
import base64
import html
import json
import sys
import time
import webbrowser
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

import config
import llm
from embedder import get_embedder
from motion_metadata import (
    derive_query_motion_profile,
    ensure_motion_metadata,
    frame_match_score,
    motion_alignment_score,
    motion_label,
)
from query_logger import QueryLogger, candidate_snapshot

_LABELS = {"hands": "手数", "handshape": "手型", "orientation": "朝向", "location": "位置",
           "movement": "运动", "expression": "表情", "iconicity": "象形提示", "uncertain": "不确定要素"}


def load_index():
    if not config.EMB_PATH.exists():
        raise FileNotFoundError(f"未找到索引 {config.EMB_PATH}，请先运行 python3 build_index.py")
    vecs = np.load(config.EMB_PATH)
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    imeta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    return vecs, meta, imeta


def cosine_topk(qvec: np.ndarray, matrix: np.ndarray, k: int):
    if len(matrix) == 0:
        raise ValueError("索引为空，请重新运行 python3 build_index.py")
    sims = matrix @ qvec
    k = max(1, min(int(k), len(sims)))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def frame_aware_topk(qvec: np.ndarray, matrix: np.ndarray, meta: list[dict], k: int,
                     frame_count: int, query_motion: dict | None = None):
    """Vector recall with a soft prior for uploaded frame count."""
    if len(matrix) == 0:
        raise ValueError("索引为空，请重新运行 python3 build_index.py")
    sims = matrix @ qvec
    frame_scores = np.array([frame_match_score(frame_count, m) for m in meta], dtype=np.float32)
    motion_scores = np.array([motion_alignment_score(query_motion, m) for m in meta], dtype=np.float32)
    final_scores = sims + frame_scores + motion_scores
    k = max(1, min(int(k), len(final_scores)))
    idx = np.argpartition(-final_scores, k - 1)[:k]
    idx = idx[np.argsort(-final_scores[idx])]
    return idx, sims[idx], frame_scores[idx], motion_scores[idx], final_scores[idx]


def attach_recall_scores(candidate: dict, recall_sim: float, frame_count: int,
                         query_motion: dict | None = None,
                         final_recall_score: float | None = None,
                         motion_score: float | None = None) -> dict:
    motion = ensure_motion_metadata(candidate)
    candidate.update(motion)
    candidate["recall_sim"] = float(recall_sim)
    candidate["frame_score"] = float(frame_match_score(frame_count, candidate))
    candidate["motion_score"] = (
        float(motion_score)
        if motion_score is not None
        else float(motion_alignment_score(query_motion, candidate))
    )
    candidate["final_recall_score"] = (
        float(final_recall_score)
        if final_recall_score is not None
        else candidate["recall_sim"] + candidate["frame_score"] + candidate["motion_score"]
    )
    return candidate


def run_query(frame_paths, *, top_recall=40, top_text=10, top_final=5,
              visual=True, max_rerank_images=5, backend=None,
              log_query=True, log_source="cli", log_meta=None):
    frame_paths = [Path(p) for p in frame_paths]
    if not frame_paths:
        raise ValueError("至少需要一张关键帧图片")
    top_final = max(1, int(top_final))
    top_text = max(top_final, int(top_text))
    top_recall = max(top_text, int(top_recall))
    max_rerank_images = max(1, int(max_rerank_images))
    query_frame_count = max(1, min(5, len(frame_paths)))
    logger = QueryLogger(log_source, log_meta) if log_query else None
    if logger:
        logger.set("parameters", {
            "top_recall": top_recall,
            "top_text": top_text,
            "top_final": top_final,
            "visual": visual,
            "max_rerank_images": max_rerank_images,
            "backend": backend,
            "query_frame_count": query_frame_count,
        })
    try:
        for p in frame_paths:
            if not p.exists():
                raise FileNotFoundError(f"找不到关键帧：{p}")
        if logger:
            logger.copy_input_frames(frame_paths)

        vecs, meta, imeta = load_index()
        embedder = get_embedder(backend or imeta.get("backend", "local"))
        if logger:
            logger.event("index.loaded", {
                "index_meta": imeta,
                "vector_shape": list(vecs.shape),
                "meta_count": len(meta),
                "embedder_backend": getattr(embedder, "backend", backend or imeta.get("backend", "local")),
                "embedder_model": getattr(embedder, "model_name", ""),
            })

        # (1) 理解
        print("· 理解关键帧…")
        desc = llm.describe_frames(frame_paths, logger=logger)
        qtext = config.render_query_text(desc)
        query_motion = derive_query_motion_profile(qtext, desc)
        if logger:
            logger.event("query_text.rendered", {
                "description": desc,
                "query_text": qtext,
                "query_motion": query_motion,
            })
        print(f"  检索文本：{qtext}")

        # (3) 召回
        qvec = embedder.embed_query(qtext)
        idx, sims, frame_scores, motion_scores, final_scores = frame_aware_topk(
            qvec, vecs, meta, top_recall, query_frame_count, query_motion)
        candidates = []
        for rank, (i, s, fs, ms, final) in enumerate(zip(idx, sims, frame_scores, motion_scores, final_scores)):
            c = dict(meta[int(i)])
            c["recall_rank"] = rank + 1
            attach_recall_scores(c, float(s), query_frame_count, query_motion,
                                 final_recall_score=float(final), motion_score=float(ms))
            candidates.append(c)
        by_id = {c["id"]: c for c in candidates}
        if logger:
            logger.event("recall.completed", {
                "query_frame_count": query_frame_count,
                "query_motion": query_motion,
                "scoring": "recall_sim + frame_score + motion_score",
                "top_recall": len(candidates),
                "highest_similarity": candidates[0]["recall_sim"] if candidates else None,
                "highest_final_score": candidates[0]["final_recall_score"] if candidates else None,
                "candidates": [candidate_snapshot(c) for c in candidates],
            })
        print(f"· 召回 top-{len(candidates)}（最高综合分 {candidates[0]['final_recall_score']:.3f}，向量相似度 {candidates[0]['recall_sim']:.3f}）")

        # 形近字增召回：王/十/口/工 这类"比字形"词条靠语义描述，向量召回会漏，按字符补进重排池
        res = (desc.get("resembles") or "").strip()
        if res and len(res) <= 4:
            boosted = []
            for j, m in enumerate(meta):
                if m["id"] in by_id:
                    continue
                d = m.get("description") or ""
                words = (m.get("words") or "").split("、")
                if (res in d and "字" in d) or res in words:
                    c = dict(m)
                    c["recall_rank"] = 0          # 0 = 词法注入
                    attach_recall_scores(c, float(vecs[j] @ qvec), query_frame_count, query_motion)
                    boosted.append(c)
            boosted.sort(key=lambda c: -c["final_recall_score"])
            for c in boosted[:6]:
                candidates.append(c)
                by_id[c["id"]] = c
            if logger:
                logger.event("lexical_boost.completed", {
                    "resembles": res,
                    "hit_count": len(boosted),
                    "injected_count": min(len(boosted), 6),
                    "injected": [candidate_snapshot(c) for c in boosted[:6]],
                })
            if boosted:
                print(f"· 词法增召回：“{res}” 命中 {len(boosted)} 条，注入 {min(len(boosted), 6)} 条")
        elif logger:
            logger.event("lexical_boost.skipped", {"resembles": res})

        # (4a) 文本重排 → 选出 top_text 候选（含召回兜底补齐）
        print("· 文本重排…")
        text_ids = llm.rerank_text(desc, candidates, top_text, logger=logger)
        ordered = [by_id[i] for i in text_ids]
        for c in candidates:
            if len(ordered) >= top_text:
                break
            if c["id"] not in text_ids:
                ordered.append(c)
        text_candidates = ordered[:top_text]
        kept_boosted = []
        for c in [x for x in candidates if x.get("recall_rank") == 0]:
            if c["id"] in {x["id"] for x in text_candidates}:
                continue
            if len(text_candidates) < top_text:
                text_candidates.append(c)
            else:
                text_candidates[-1] = c
            kept_boosted.append(c)
        if logger and kept_boosted:
            logger.event("text_candidates.boosted_kept", {
                "ids": [c["id"] for c in kept_boosted],
                "candidates": [candidate_snapshot(c) for c in kept_boosted],
            })
        if logger:
            logger.event("text_candidates.selected", {
                "ids": [c["id"] for c in text_candidates],
                "candidates": [candidate_snapshot(c) for c in text_candidates],
            })

        # (4b) 视觉重排
        if visual:
            print("· 视觉重排…")
            vres = llm.rerank_visual(
                frame_paths,
                text_candidates,
                top_final,
                batch_size=max_rerank_images,
                logger=logger,
            )
            results = []
            for r in vres:
                c = dict(by_id[r["id"]])
                c["confidence"] = r["confidence"]
                c["reason"] = r["reason"]
                results.append(c)
        else:
            results = []
            for c in text_candidates[:top_final]:
                c = dict(c)
                c["confidence"] = None
                c["reason"] = "（未做视觉重排）"
                results.append(c)
            if logger:
                logger.event("visual_rerank.skipped", {"reason": "disabled"})

        result = {
            "frames": [str(p) for p in frame_paths],
            "description": desc,
            "query_text": qtext,
            "query_motion": query_motion,
            "results": results,
        }
        if logger:
            result["log_id"] = logger.log_id
            result["log_dir"] = str(logger.dir)
            logger.set("output", {
                "description": desc,
                "query_text": qtext,
                "query_motion": query_motion,
                "results": [candidate_snapshot(c) for c in results],
            })
            logger.finish("ok")
        return result
    except Exception as exc:
        if logger:
            logger.finish("error", exc)
            try:
                setattr(exc, "log_id", logger.log_id)
                setattr(exc, "log_dir", str(logger.dir))
            except Exception:  # noqa: BLE001
                pass
        raise


# --- 输出 ---------------------------------------------------------------
def print_summary(result: dict):
    print("\n" + "=" * 60)
    print(f"查询帧：{', '.join(Path(f).name for f in result['frames'])}")
    print(f"检索文本：{result['query_text']}")
    print("-" * 60)
    for rank, c in enumerate(result["results"], 1):
        conf = f"{c['confidence'] * 100:.0f}%" if c.get("confidence") is not None else "—"
        rtag = "词法注入" if c.get("recall_rank") == 0 else f"召回#{c.get('recall_rank', '—')}"
        first_desc = ((c.get("description") or "").splitlines() or [""])[0][:50]
        frame_score = c.get("frame_score")
        frame_tag = f"，帧数权重 {frame_score:+.3f}" if isinstance(frame_score, (int, float)) else ""
        motion_score = c.get("motion_score")
        motion_tag = f"，动作权重 {motion_score:+.3f}" if isinstance(motion_score, (int, float)) else ""
        print(f"{rank}. [{conf}] {c.get('words', '')}   (id={c.get('id')}, {rtag}，{motion_label(c)}{frame_tag}{motion_tag})")
        print(f"     打法：{first_desc}")
        if c.get("reason"):
            print(f"     理由：{c['reason']}")
    print("=" * 60)


def _img_data_uri(path, max_px=480) -> str:
    img = Image.open(Path(path)).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html_report(result: dict, out_path: Path) -> Path:
    e = html.escape
    desc = result["description"]
    desc_order = ["hands", "handshape", "orientation", "location", "movement",
                  "expression", "iconicity", "resembles", "uncertain"]
    desc_chips = []
    for key in desc_order:
        if key not in desc:
            continue
        value = desc.get(key)
        if isinstance(value, list):
            value = "、".join(str(x) for x in value)
        value = str(value or "").strip()
        if not value:
            continue
        muted = value.lower() == "uncertain"
        desc_chips.append(
            f'<div class="chip {"muted" if muted else ""}">'
            f'<span>{e(_LABELS.get(key, key))}</span><b>{e(value)}</b></div>')
    frame_imgs = "".join(
        f'<figure class="frame"><img src="{_img_data_uri(f)}" alt="{e(Path(f).name)}">'
        f'<figcaption>{e(Path(f).name)}</figcaption></figure>' for f in result["frames"])

    cards = []
    for rank, c in enumerate(result["results"], 1):
        conf = f"{c['confidence'] * 100:.0f}%" if c.get("confidence") is not None else "—"
        try:
            cand_img = f'<img src="{_img_data_uri(config.resolve_image(c["image_path"]))}" alt="{e(c.get("words") or "")}">'
        except Exception:
            cand_img = '<div class="noimg">无插图</div>'
        recall = "词法注入" if c.get("recall_rank") == 0 else "召回#" + str(c.get("recall_rank", "—"))
        frame_score = c.get("frame_score")
        frame_score_text = f" · 帧数权重 {frame_score:+.3f}" if isinstance(frame_score, (int, float)) else ""
        motion_score = c.get("motion_score")
        motion_score_text = f" · 动作权重 {motion_score:+.3f}" if isinstance(motion_score, (int, float)) else ""
        final_score = c.get("final_recall_score")
        final_score_text = f" · 综合分 {final_score:.3f}" if isinstance(final_score, (int, float)) else ""
        top_class = " top-card" if rank == 1 else ""
        cards.append(f"""
        <article class="candidate{top_class}">
          <div class="candidate-rank">
            <span>#{rank}</span>
            <strong>{conf}</strong>
          </div>
          <div class="illu">{cand_img}</div>
          <div class="candidate-main">
            <div class="words">{e(c.get('words') or '')}</div>
            <div class="meta">id={e(str(c.get('id')))} · {e(recall)} · {e(motion_label(c))} · 相似度 {float(c.get('recall_sim') or 0):.3f}{e(frame_score_text)}{e(motion_score_text)}{e(final_score_text)}</div>
            <div class="reason">{e(c.get('reason') or '')}</div>
            <pre class="desc">{e(c.get('description') or '')}</pre>
          </div>
        </article>""")

    htmldoc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>见手知意 SignSeek · 反查报告</title><style>
:root {{
  color-scheme: light;
  --ink: #2a1a10;
  --muted: #806c5a;
  --hairline: rgba(126, 78, 27, .18);
  --glass: rgba(255, 250, 238, .72);
  --glass-strong: rgba(255, 247, 225, .9);
  --amber: #f59e0b;
  --gold: #fbbf24;
  --green: #2f7d64;
  --rose: #be123c;
  --shadow: 0 24px 70px rgba(120, 72, 20, .16);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  letter-spacing: 0;
  background:
    linear-gradient(135deg, rgba(255, 252, 244, .98), rgba(255, 238, 199, .84) 46%, rgba(255, 250, 238, .96)),
    linear-gradient(45deg, rgba(245, 158, 11, .18), rgba(251, 191, 36, .14), rgba(47, 125, 100, .07));
  background-attachment: fixed;
}}
body::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255,255,255,.42) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.32) 1px, transparent 1px);
  background-size: 72px 72px;
  mask-image: linear-gradient(to bottom, rgba(0,0,0,.72), transparent);
}}
.shell {{
  position: relative;
  width: min(1160px, calc(100vw - 36px));
  margin: 0 auto;
  padding: 30px 0 46px;
}}
.hero {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 22px;
  margin-bottom: 22px;
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 14px;
}}
.mark {{
  width: 48px;
  height: 48px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  color: #fff;
  font-weight: 820;
  background: linear-gradient(135deg, #6b3a12, #d97706 50%, #fbbf24);
  box-shadow: 0 14px 30px rgba(146, 86, 20, .24);
}}
h1 {{
  margin: 0;
  font-size: clamp(30px, 4.5vw, 54px);
  line-height: 1.1;
  font-weight: 840;
  background: linear-gradient(135deg, #7c3f12, var(--amber) 48%, var(--green));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}
.subtitle {{
  margin-top: 8px;
  color: var(--muted);
  font-size: 14px;
}}
.pill {{
  border: 1px solid var(--hairline);
  background: rgba(255,255,255,.52);
  backdrop-filter: blur(22px) saturate(1.35);
  border-radius: 999px;
  padding: 10px 14px;
  color: #5c3b1a;
  box-shadow: 0 12px 32px rgba(120, 72, 20, .1);
  white-space: nowrap;
  font-size: 13px;
}}
.panel {{
  position: relative;
  border: 1px solid var(--hairline);
  border-radius: 24px;
  background: var(--glass);
  backdrop-filter: blur(28px) saturate(1.35);
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.panel::before {{
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(135deg, rgba(255,255,255,.7), transparent 36%, rgba(255,255,255,.22) 74%, transparent);
}}
.panel > * {{ position: relative; }}
.summary {{
  padding: 22px;
  margin-bottom: 20px;
}}
.eyebrow {{
  margin: 0 0 10px;
  color: var(--green);
  font-size: 13px;
  font-weight: 760;
}}
.query-text {{
  margin: 0;
  color: #3f2916;
  font-size: 18px;
  line-height: 1.75;
  font-weight: 620;
}}
.two-col {{
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
  gap: 20px;
  align-items: stretch;
  margin-bottom: 24px;
}}
.section {{
  padding: 20px;
}}
h2 {{
  margin: 0 0 16px;
  font-size: 17px;
  line-height: 1.3;
  font-weight: 780;
}}
.frames {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
}}
.frame {{
  margin: 0;
}}
.frame img {{
  display: block;
  width: 100%;
  height: 240px;
  object-fit: contain;
  border: 1px solid rgba(126, 78, 27, .14);
  border-radius: 18px;
  background: rgba(255, 253, 247, .88);
}}
figcaption {{
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.chips {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}}
.chip {{
  min-width: 0;
  max-width: 100%;
  border: 1px solid rgba(47, 125, 100, .14);
  border-radius: 16px;
  background: rgba(255, 253, 247, .7);
  padding: 10px 12px;
}}
.chip span {{
  display: block;
  margin-bottom: 4px;
  color: var(--green);
  font-size: 12px;
  font-weight: 760;
}}
.chip b {{
  color: #3f2916;
  font-size: 13px;
  line-height: 1.55;
  font-weight: 620;
  word-break: normal;
  overflow-wrap: break-word;
}}
.chip.muted b {{ color: #9b826b; }}
.results-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
  margin: 4px 0 14px;
}}
.results-head h2 {{
  margin: 0;
  font-size: 22px;
}}
.candidates {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.candidate {{
  border: 1px solid var(--hairline);
  border-radius: 22px;
  background: rgba(255, 253, 247, .82);
  box-shadow: 0 18px 46px rgba(120, 72, 20, .1);
  padding: 16px;
  display: grid;
  grid-template-columns: 74px 178px minmax(420px, 1fr);
  gap: 18px;
  align-items: start;
}}
.top-card {{
  grid-template-columns: 88px 250px minmax(460px, 1fr);
  padding: 20px;
  background: linear-gradient(135deg, rgba(255, 247, 222, .95), rgba(255, 253, 247, .86));
  border-color: rgba(245, 158, 11, .32);
  box-shadow: 0 24px 70px rgba(146, 86, 20, .18);
}}
.candidate-rank {{
  min-height: 76px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(245,158,11,.16), rgba(47,125,100,.1));
}}
.candidate-rank span {{
  color: #8a4b12;
  font-size: 24px;
  font-weight: 850;
}}
.candidate-rank strong {{
  color: var(--green);
  font-size: 14px;
  font-weight: 820;
}}
.illu img {{
  display: block;
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  border: 1px solid rgba(126, 78, 27, .12);
  border-radius: 18px;
  background: #fffdf7;
}}
.candidate-main {{
  min-width: 0;
  padding-top: 2px;
}}
.noimg {{
  min-height: 150px;
  display: grid;
  place-items: center;
  border-radius: 18px;
  background: rgba(255, 248, 232, .9);
  color: #a1866d;
  font-size: 13px;
}}
.words {{
  color: #2a1a10;
  font-size: 22px;
  line-height: 1.25;
  font-weight: 840;
  word-break: normal;
  overflow-wrap: break-word;
}}
.meta {{
  margin-top: 7px;
  color: var(--muted);
  font-size: 12px;
}}
.reason {{
  margin-top: 12px;
  color: var(--rose);
  font-size: 14px;
  line-height: 1.65;
  font-weight: 650;
  word-break: normal;
  overflow-wrap: break-word;
}}
.desc {{
  margin: 12px 0 0;
  padding: 12px;
  white-space: pre-wrap;
  color: #594232;
  font: 13px/1.7 -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  border-radius: 16px;
  background: rgba(255, 247, 235, .78);
  word-break: normal;
  overflow-wrap: break-word;
}}
.note {{
  margin: 22px 0 0;
  color: #8b725f;
  font-size: 12px;
  line-height: 1.7;
}}
.footer {{
  margin-top: 24px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  color: #8b725f;
  font-size: 12px;
}}
@media (max-width: 820px) {{
  .shell {{ width: min(100vw - 24px, 1160px); padding-top: 20px; }}
  .hero, .footer {{ align-items: flex-start; flex-direction: column; }}
  .two-col {{ grid-template-columns: 1fr; }}
  .chips {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .candidate {{ grid-template-columns: 64px 144px minmax(0, 1fr); }}
  .top-card {{ grid-template-columns: 72px 170px minmax(0, 1fr); }}
  .words {{ font-size: 19px; }}
}}
@media (max-width: 560px) {{
  .candidate, .top-card {{ grid-template-columns: 1fr; }}
  .chips {{ grid-template-columns: 1fr; }}
  .candidate-rank {{ min-height: 54px; display: flex; justify-content: space-between; padding: 0 18px; }}
  .illu img {{ max-height: 220px; }}
}}
</style></head><body>
<main class="shell">
  <header class="hero">
    <div class="brand">
      <div class="mark">S</div>
      <div>
        <h1>见手知意 SignSeek</h1>
        <div class="subtitle">手语视觉反查报告</div>
      </div>
    </div>
    <div class="pill">生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}</div>
  </header>

  <section class="panel summary">
    <p class="eyebrow">检索文本</p>
    <p class="query-text">{e(result['query_text'])}</p>
  </section>

  <section class="two-col">
    <div class="panel section">
      <h2>查询关键帧</h2>
      <div class="frames">{frame_imgs}</div>
    </div>
    <div class="panel section">
      <h2>模型理解</h2>
      <div class="chips">{''.join(desc_chips)}</div>
    </div>
  </section>

  <section>
    <div class="results-head">
      <h2>匹配结果</h2>
      <div class="pill">Top {len(result['results'])}</div>
    </div>
    <div class="candidates">{''.join(cards)}</div>
  </section>

  <p class="note">提示：单帧或缺少身体参照时，运动、位置和面部线索可能不完整，结果适合作为候选召回参考。</p>
  <footer class="footer">
    <span>由见手知意 SignSeek 本地生成</span>
    <span>开发者：手语分社心创组</span>
  </footer>
</main>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(htmldoc, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="见手知意 SignSeek｜手语视觉反查：关键帧 → top-K 词条")
    ap.add_argument("frames", nargs="+", help="一个或多个关键帧图片路径")
    ap.add_argument("--top-k", type=int, default=5, help="最终返回数（默认 5）")
    ap.add_argument("--top-recall", type=int, default=40, help="向量召回数（默认 40）")
    ap.add_argument("--top-text", type=int, default=10, help="文本重排保留数（默认 10）")
    ap.add_argument("--no-visual-rerank", action="store_true", help="跳过视觉重排（省 token）")
    ap.add_argument("--max-rerank-images", type=int, default=5, help="单批视觉重排候选上限（默认 5）")
    ap.add_argument("--backend", default=None, choices=["local", "api"], help="向量后端（默认随索引）")
    ap.add_argument("--json-only", action="store_true", help="只输出 JSON，不生成 HTML")
    ap.add_argument("--open", action="store_true", help="生成后自动打开 HTML 报告")
    args = ap.parse_args()

    try:
        result = run_query(
            args.frames, top_recall=args.top_recall, top_text=args.top_text, top_final=args.top_k,
            visual=not args.no_visual_rerank, max_rerank_images=args.max_rerank_images, backend=args.backend)
    except Exception as exc:
        if getattr(exc, "log_dir", None):
            print(f"查询日志：{exc.log_dir}", file=sys.stderr)
        raise

    if args.json_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print_summary(result)
    stem = Path(args.frames[0]).stem
    out = config.REPORTS_DIR / f"query_{time.strftime('%Y%m%d_%H%M%S')}_{stem}.html"
    build_html_report(result, out)
    print(f"\n报告已生成：{out}")
    if result.get("log_dir"):
        print(f"查询日志：{result['log_dir']}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
