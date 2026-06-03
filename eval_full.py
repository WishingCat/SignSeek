"""全库规模 A/B：目标词条在 v1(原描述) vs v2(叠加) 全 6699 库中的纯向量召回排名。

不含词法增召回，隔离"词库侧描述质量"的影响。
"""
import numpy as np

import config
import llm
from build_index import load_signs
from embedder import get_embedder

V2DB = config.PROJECT_ROOT / "sign-language-database-v2" / "signs.db"
TARGETS = {
    "测试题 1.jpg": (878, False),    # 出租车
    "测试题 2.jpg": (5653, False),   # 姓
    "测试题 3.jpg": (3598, False),   # 名词
    "测试题 4.jpg": (5208, True),    # 王（确认）
    "测试题 5.jpg": (1848, False),   # 工会
}


def rank_of(sims, idx):
    return int((sims > sims[idx]).sum()) + 1


def main():
    emb = get_embedder("local")
    v1 = load_signs(config.DB_PATH)
    v2 = load_signs(V2DB, v2=True)
    ids = [s["id"] for s in v1]
    id_idx = {i: k for k, i in enumerate(ids)}
    print(f"全库 {len(ids)} 条；编码 v1 与 v2（各 6699）…")
    M1 = emb.embed_documents([s["embed_text"] for s in v1])
    M2 = emb.embed_documents([s["embed_text"] for s in v2])

    print(f"\n{'测试题':<13}{'目标词':<13}{'v1原描述':>10}{'v2叠加':>9}   变化")
    print("-" * 56)
    for img, (tid, conf) in TARGETS.items():
        desc = llm.describe_frames([config.TEST_DIR / img])
        qv = emb.embed_query(config.render_query_text(desc))
        i = id_idx[tid]
        r1, r2 = rank_of(M1 @ qv, i), rank_of(M2 @ qv, i)
        words = next(s["words"] for s in v1 if s["id"] == tid)[:9]
        tag = "✅" if conf else "推定"
        arrow = f"↑改善 {r1}→{r2}" if r2 < r1 else (f"↓变差 {r1}→{r2}" if r2 > r1 else "持平")
        print(f"{img:<13}{words+'('+tag+')':<13}{('#'+str(r1)):>10}{('#'+str(r2)):>9}   {arrow}")
    print("-" * 56)
    print("（全 6699 库内纯向量召回排名；#越小越好。在线 query.py 还有词法增召回+视觉重排叠加增益）")


if __name__ == "__main__":
    main()
