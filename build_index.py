"""离线建索引：signs.db → index/embeddings.npy + meta.json + index_meta.json。

索引粒度 = sign（不是 meaning），每条聚合该 sign 的全部同义词。
embedding 文本 = description（手语"打法"描述，即动作形态，正是要与查询比对的内容）。
对数据库只读（仅 SELECT）。
"""
import argparse
import json
import sqlite3
import time

import numpy as np

import config
from embedder import get_embedder
from motion_metadata import derive_motion_metadata


def load_signs(db_path, limit: int | None = None, v2: bool = False) -> list[dict]:
    """读取 6699 个 sign，聚合同义词。返回按 id 升序的 dict 列表。

    v2=True 时额外读取 llm_description / llm_struct，并令 embed_text = 原文 + LLM重描述（叠加策略）。
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # 只读打开，杜绝误写
    llm_col = ", s.llm_description, s.llm_struct" if v2 else ", NULL, NULL"
    try:
        sql = f"""
            SELECT s.id,
                   (SELECT group_concat(text, '、')
                      FROM (SELECT text FROM meanings WHERE sign_id = s.id ORDER BY order_in_entry)) AS words,
                   s.description,
                   s.image_path
                   {llm_col}
            FROM signs s
            ORDER BY s.id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        desc, llm_desc, llm_struct = r[2] or "", (r[4] or ""), (r[5] or "")
        embed_text = f"{desc}。{llm_desc}" if (v2 and llm_desc) else desc
        motion = derive_motion_metadata(desc, llm_struct, llm_desc)
        out.append({"id": r[0], "words": r[1] or "", "description": desc,
                    "image_path": r[3], "embed_text": embed_text, **motion})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="构建手语词条向量索引")
    ap.add_argument("--backend", default="local", choices=["local", "api"], help="向量后端（默认 local BGE）")
    ap.add_argument("--limit", type=int, default=None, help="只索引前 N 条（调试用）")
    ap.add_argument("--v2", action="store_true", help="用 v2 库的叠加文本（原文+LLM重描述）建索引")
    ap.add_argument("--rebuild", action="store_true", help="存在索引时覆盖（默认也会覆盖，仅作语义提示）")
    args = ap.parse_args()

    db_path = (config.PROJECT_ROOT / "sign-language-database-v2" / "signs.db") if args.v2 else config.DB_PATH
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return 1

    t0 = time.time()
    signs = load_signs(db_path, args.limit, v2=args.v2)
    print(f"读取 {len(signs)} 个 sign（来源：{'v2叠加' if args.v2 else 'v1原描述'}，{time.time() - t0:.1f}s）")

    print(f"加载向量模型（backend={args.backend}）…")
    embedder = get_embedder(args.backend)

    texts = [s["embed_text"] for s in signs]
    t1 = time.time()
    vecs = embedder.embed_documents(texts)
    print(f"编码完成：{vecs.shape}（{time.time() - t1:.1f}s，device={getattr(embedder, 'device', args.backend)}）")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMB_PATH, vecs)
    config.META_PATH.write_text(json.dumps(signs, ensure_ascii=False), encoding="utf-8")
    config.INDEX_META_PATH.write_text(json.dumps({
        "model": getattr(embedder, "model_name", args.backend),
        "backend": args.backend,
        "dim": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "embed_field": "description+llm_description" if args.v2 else "description",
        "source": "v2" if args.v2 else "v1",
        "normalized": True,
        "motion_metadata": True,
        "motion_metadata_source": "description+llm_description+llm_struct" if args.v2 else "description",
        "max_query_frames": 5,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已写入 {config.INDEX_DIR}")
    print(f"  embeddings.npy  {vecs.shape}  ({config.EMB_PATH.stat().st_size // 1024} KB)")
    print(f"  meta.json       {len(signs)} 条")
    # 自检：用第一条描述当查询，应能召回自身
    if len(signs) > 1:
        q = embedder.embed_query(signs[0]["embed_text"])
        top = int(np.argmax(vecs @ q))
        flag = "✓" if top == 0 else f"✗(召回到 idx={top})"
        print(f"  自检：用 signs[0] 描述检索 → 命中自身 {flag}")
    print(f"完成，总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
