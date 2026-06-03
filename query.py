"""见手知意 SignSeek — 在线查询主流程：关键帧 → top-5 候选词条。

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
import time
import webbrowser
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

import config
import llm
from embedder import get_embedder

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
    sims = matrix @ qvec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def run_query(frame_paths, *, top_recall=40, top_text=10, top_final=5,
              visual=True, max_rerank_images=10, backend=None):
    frame_paths = [Path(p) for p in frame_paths]
    for p in frame_paths:
        if not p.exists():
            raise FileNotFoundError(f"找不到关键帧：{p}")

    vecs, meta, imeta = load_index()
    embedder = get_embedder(backend or imeta.get("backend", "local"))

    # (1) 理解
    print("· 理解关键帧…")
    desc = llm.describe_frames(frame_paths)
    qtext = config.render_query_text(desc)
    print(f"  检索文本：{qtext}")

    # (3) 召回
    qvec = embedder.embed_query(qtext)
    idx, sims = cosine_topk(qvec, vecs, top_recall)
    candidates = []
    for rank, (i, s) in enumerate(zip(idx, sims)):
        c = dict(meta[int(i)])
        c["recall_rank"] = rank + 1
        c["recall_sim"] = float(s)
        candidates.append(c)
    by_id = {c["id"]: c for c in candidates}
    print(f"· 召回 top-{len(candidates)}（最高相似度 {candidates[0]['recall_sim']:.3f}）")

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
                c["recall_sim"] = float(vecs[j] @ qvec)
                boosted.append(c)
        boosted.sort(key=lambda c: -c["recall_sim"])
        for c in boosted[:6]:
            candidates.append(c)
            by_id[c["id"]] = c
        if boosted:
            print(f"· 词法增召回：“{res}” 命中 {len(boosted)} 条，注入 {min(len(boosted), 6)} 条")

    # (4a) 文本重排 → 选出 top_text 候选（含召回兜底补齐）
    print("· 文本重排…")
    text_ids = llm.rerank_text(desc, candidates, top_text)
    ordered = [by_id[i] for i in text_ids]
    for c in candidates:
        if len(ordered) >= top_text:
            break
        if c["id"] not in text_ids:
            ordered.append(c)
    text_candidates = ordered[:top_text]

    # (4b) 视觉重排
    if visual:
        print("· 视觉重排…")
        vres = llm.rerank_visual(frame_paths, text_candidates, top_final, batch_size=max_rerank_images)
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

    return {
        "frames": [str(p) for p in frame_paths],
        "description": desc,
        "query_text": qtext,
        "results": results,
    }


# --- 输出 ---------------------------------------------------------------
def print_summary(result: dict):
    print("\n" + "=" * 60)
    print(f"查询帧：{', '.join(Path(f).name for f in result['frames'])}")
    print(f"检索文本：{result['query_text']}")
    print("-" * 60)
    for rank, c in enumerate(result["results"], 1):
        conf = f"{c['confidence'] * 100:.0f}%" if c.get("confidence") is not None else "—"
        rtag = "词法注入" if c.get("recall_rank") == 0 else f"召回#{c['recall_rank']}"
        print(f"{rank}. [{conf}] {c['words']}   (id={c['id']}, {rtag})")
        print(f"     打法：{(c['description'] or '').splitlines()[0][:50]}")
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
    desc_rows = "".join(
        f"<tr><th>{_LABELS.get(k, k)}</th><td>{e(str(v))}</td></tr>"
        for k, v in desc.items())
    frame_imgs = "".join(
        f'<img src="{_img_data_uri(f)}" title="{e(Path(f).name)}">' for f in result["frames"])

    cards = []
    for rank, c in enumerate(result["results"], 1):
        conf = f"{c['confidence'] * 100:.0f}%" if c.get("confidence") is not None else "—"
        try:
            cand_img = f'<img src="{_img_data_uri(config.resolve_image(c["image_path"]))}">'
        except Exception:
            cand_img = '<div class="noimg">无插图</div>'
        cards.append(f"""
        <div class="card">
          <div class="rank">#{rank}<span class="conf">{conf}</span></div>
          <div class="illu">{cand_img}</div>
          <div class="info">
            <div class="words">{e(c['words'])}</div>
            <div class="meta">id={c['id']} · {'词法注入' if c.get('recall_rank') == 0 else '召回#' + str(c['recall_rank'])} · 相似度 {c['recall_sim']:.3f}</div>
            <div class="reason">{e(c.get('reason') or '')}</div>
            <pre class="desc">{e(c['description'])}</pre>
          </div>
        </div>""")

    htmldoc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>见手知意 SignSeek · 反查报告</title><style>
 body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;color:#222;background:#fafafa}}
 h1{{font-size:20px}} h2{{font-size:15px;color:#555;margin-top:24px}}
 .query{{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}}
 .frames img{{height:220px;border:1px solid #ddd;border-radius:8px;margin-right:8px;background:#fff}}
 table{{border-collapse:collapse;font-size:13px}} th,td{{border:1px solid #e3e3e3;padding:3px 8px;text-align:left}}
 th{{background:#f0f0f0;white-space:nowrap}}
 .qtext{{font-size:15px;background:#eef4ff;padding:8px 12px;border-radius:8px;margin:10px 0}}
 .card{{display:flex;gap:16px;background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:14px;margin:12px 0}}
 .rank{{font-size:22px;font-weight:700;color:#3b6fd4;min-width:54px}}
 .conf{{display:block;font-size:13px;color:#888;font-weight:400}}
 .illu img{{width:240px;border:1px solid #eee;border-radius:6px;background:#fff}} .noimg{{color:#aaa}}
 .words{{font-size:18px;font-weight:600}} .meta{{color:#999;font-size:12px;margin:4px 0}}
 .reason{{color:#3b6fd4;font-size:13px;margin:6px 0}}
 .desc{{white-space:pre-wrap;font-size:13px;color:#444;background:#f7f7f7;padding:8px;border-radius:6px;margin:6px 0 0}}
</style></head><body>
<h1>见手知意 · SignSeek　<span style="font-size:14px;color:#888;font-weight:400">手语视觉反查报告</span></h1>
<div class="qtext">检索文本：{e(result['query_text'])}</div>
<div class="query">
  <div class="frames"><h2>查询关键帧</h2>{frame_imgs}</div>
  <div><h2>模型理解</h2><table>{desc_rows}</table></div>
</div>
<h2>最匹配的 {len(result['results'])} 个候选</h2>
{''.join(cards)}
<p style="color:#aaa;font-size:12px;margin-top:24px">⚠ 单帧/无身体参照时，运动与位置信息缺失，匹配以手型为主，结果供参考。</p>
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
    ap.add_argument("--max-rerank-images", type=int, default=10, help="单批视觉重排候选上限")
    ap.add_argument("--backend", default=None, choices=[None, "local", "api"], help="向量后端（默认随索引）")
    ap.add_argument("--json-only", action="store_true", help="只输出 JSON，不生成 HTML")
    ap.add_argument("--open", action="store_true", help="生成后自动打开 HTML 报告")
    args = ap.parse_args()

    result = run_query(
        args.frames, top_recall=args.top_recall, top_text=args.top_text, top_final=args.top_k,
        visual=not args.no_visual_rerank, max_rerank_images=args.max_rerank_images, backend=args.backend)

    if args.json_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print_summary(result)
    stem = Path(args.frames[0]).stem
    out = config.REPORTS_DIR / f"query_{time.strftime('%Y%m%d_%H%M%S')}_{stem}.html"
    build_html_report(result, out)
    print(f"\n报告已生成：{out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
