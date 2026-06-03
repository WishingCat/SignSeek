"""A/B 评测：同样 100 条，对比「原词典描述(v1)」vs「GPT重描述(v2)」的召回排名。

对每张测试题：理解→渲染查询文本→分别在 v1/v2 向量矩阵里查目标词条的排名。
查询文本两边相同，故排名差异纯粹来自"词库侧描述"的不同。纯 embedding 排名，不含词法增召回。
"""
import json

import numpy as np

import config
import llm
from embedder import get_embedder

# 测试题 → (目标词条 id, 是否用户确认)
TARGETS = {
    "测试题 1.jpg": (878, False),    # 出租车（系统推定）
    "测试题 2.jpg": (5653, False),   # 姓（推定）
    "测试题 3.jpg": (3598, False),   # 名词（推定）
    "测试题 4.jpg": (5208, True),    # 王（用户确认）
    "测试题 5.jpg": (1848, False),   # 工会/工（推定，新题5=工字形）
}


def rank_of(sims: np.ndarray, target_idx: int) -> int:
    """目标在降序排名中的位次（1-based）。"""
    return int((sims > sims[target_idx]).sum()) + 1


def main() -> int:
    sc = config.INDEX_DIR / "v2_descriptions.json"
    if not sc.exists():
        print("缺少 index/v2_descriptions.json，请先跑 build_corpus_v2.py")
        return 1
    data = json.loads(sc.read_text(encoding="utf-8"))
    ids = [int(k) for k in data]
    id_idx = {i: k for k, i in enumerate(ids)}
    orig_texts = [data[str(i)]["orig"] for i in ids]
    llm_texts = [data[str(i)]["llm"] for i in ids]

    emb = get_embedder("local")
    print(f"评测集 {len(ids)} 条；编码 v1(原描述) / v2(重描述) / v3(原文+重描述叠加)…")
    M1 = emb.embed_documents(orig_texts)
    M2 = emb.embed_documents(llm_texts)
    M3 = emb.embed_documents([f"{o}。{l}" for o, l in zip(orig_texts, llm_texts)])

    print(f"\n{'测试题':<13}{'目标词':<13}{'v1原':>7}{'v2重描述':>9}{'v3叠加':>8}")
    print("-" * 56)
    win = {"v1": 0, "v2": 0, "v3": 0}
    for img, (tid, confirmed) in TARGETS.items():
        if tid not in id_idx:
            print(f"{img:<13} 目标 id={tid} 不在评测集，跳过")
            continue
        desc = llm.describe_frames([config.TEST_DIR / img])
        q = config.render_query_text(desc)
        qv = emb.embed_query(q)
        i = id_idx[tid]
        r1, r2, r3 = rank_of(M1 @ qv, i), rank_of(M2 @ qv, i), rank_of(M3 @ qv, i)
        best = min(r1, r2, r3)
        for k, r in (("v1", r1), ("v2", r2), ("v3", r3)):
            if r == best:
                win[k] += 1
        words = data[str(tid)]["words"][:9]
        tag = "✅" if confirmed else "推定"
        print(f"{img:<13}{words+'('+tag+')':<13}{('#'+str(r1)):>7}{('#'+str(r2)):>9}{('#'+str(r3)):>8}")

    print("-" * 56)
    print(f"各方案取得最佳排名的次数：v1={win['v1']}  v2={win['v2']}  v3={win['v3']}（越多越好）")
    print("注：查询文本两侧一致，差异纯来自词库侧描述；100 条内相对排名。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
