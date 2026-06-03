"""v2 语料重描述（全量 + 可断点续跑）：GPT-5.5 读「线描图 + 词典原文」重生成结构化描述。

采用「叠加」策略：原文 description 保持不动，新增 llm_description / llm_struct 两列；
下游检索文本 = description + "。" + llm_description（既保留原文语义锚点，又获得规范手型细节）。

特性：
- 增量写库：每条完成即写入 v2 库，崩溃可续跑（只处理 llm_description IS NULL 的）。
- 并发：默认 100 路（仅 API 调用走线程，DB 写入只在主线程，SQLite 安全）。
- 完成后自动写 sign-language-database-v2/README.md 变更说明。

原数据库只读，绝不修改。
"""
import argparse
import concurrent.futures as cf
import json
import shutil
import sqlite3
import time

import config
import llm

IN_PRICE, OUT_PRICE = 5.0 / 1e6, 30.0 / 1e6
V2DIR = config.PROJECT_ROOT / "sign-language-database-v2"
V2DB = V2DIR / "signs.db"


def ensure_v2_db() -> None:
    """首次：复制原库 + 加列 + images 符号链接；已存在则原样复用（续跑）。"""
    V2DIR.mkdir(exist_ok=True)
    if not V2DB.exists():
        shutil.copy(config.DB_PATH, V2DB)
    con = sqlite3.connect(V2DB)
    for col in ("llm_description", "llm_struct"):
        try:
            con.execute(f"ALTER TABLE signs ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.close()
    img_link = V2DIR / "images"
    if not img_link.exists():
        img_link.symlink_to(config.DB_DIR / "images")


def write_changelog(con) -> None:
    total = con.execute("SELECT COUNT(*) FROM signs").fetchone()[0]
    done = con.execute("SELECT COUNT(*) FROM signs WHERE llm_description IS NOT NULL").fetchone()[0]
    md = f"""# 中国手语词典数据库 v2（GPT-5.5 重描述增强版）

> 为 **见手知意 · SignSeek** 手语视觉反查项目构建的增强词库。

本目录是 `sign-language-database/` 的**增强副本**，在不改动原数据的前提下，为每个手势词条**叠加**了一份由 GPT-5.5 重新生成的、措辞规范的手型动作描述，用于提升「拍手语动作反查词条」的召回准确率。

## 相比 v1 的改变

| 项 | v1（原库） | v2（本库） |
|---|---|---|
| `signs` 表原有列 | id / image_path / description / source_entry / letter / volume | **完全保留，未改动** |
| 新增列 `llm_description` | — | GPT-5.5 读「线描图+词典原文」生成的规范化手型描述（已渲染为检索文本） |
| 新增列 `llm_struct` | — | 同一描述的结构化 JSON（hands/handshape/orientation/location/movement/two_hand_relation/resembles/steps） |
| `meanings` 表 | 不变 | 不变 |
| `images/` | 实体目录 | **符号链接**指向 v1 的 images（不重复占用 ~350MB） |

进度：**{done} / {total}** 个词条已生成 llm_description（{done*100//total if total else 0}%）。

## 为什么这样做（核心：叠加而非替换）

- 原库 `description` 是词典原文，措辞与「用户拍照后由大模型生成的描述」风格不一致，导致纯向量召回会漏掉一些词条（实测"王"曾排到第 5457 名）。
- 让 GPT-5.5 用与查询侧一致的词汇重新描述词库图，可显著改善召回（"王" #32→#1）。
- 但**重描述会丢失原文里"搭成X字形"这类语义锚点**（实测"工会"纯替换后 #2→#17）。
- 因此采用**叠加**：检索文本 = `description`（原文锚点）+ `llm_description`（规范手型细节）。100 条 A/B/C 验证中，叠加方案在 5 题里 4 题取得最佳召回。

## 生成方式

- 模型：GPT-5.5（OpenAI 兼容网关），图片 `detail=high`。
- 输入：每个词条的线描示意图 + 词典原文打法，图文互证（以原文为准、用图补全运动/朝向/双手关系）。
- 输出：结构化 JSON（见 `llm_struct`）+ 渲染后的检索文本（`llm_description`）。

## 如何使用

```sql
-- 推荐的检索文本（叠加）
SELECT id, description || '。' || COALESCE(llm_description,'') AS embed_text,
       image_path
FROM signs;
```
配套：`reverse-lookup/build_index.py --v2` 直接用叠加文本重建向量索引。

## 说明

- 原 `sign-language-database/` 始终只读，本库为独立副本。
- 数据为《国家通用手语词典》衍生，仅限研究/教育/无障碍**非商业**用途（与原库 license 一致）。
- 重描述由模型生成，可能存在个别误读；原文 `description` 始终保留为准绳。
"""
    (V2DIR / "README.md").write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="v2 全量语料重描述（可续跑）")
    ap.add_argument("--workers", type=int, default=100, help="并发数（默认 100）")
    ap.add_argument("--detail", default="high", choices=["high", "low", "auto"])
    ap.add_argument("--limit", type=int, default=None, help="本次最多处理多少条（默认全部待处理）")
    args = ap.parse_args()

    ensure_v2_db()
    con = sqlite3.connect(V2DB)  # 主线程唯一写者
    total = con.execute("SELECT COUNT(*) FROM signs").fetchone()[0]
    already = con.execute("SELECT COUNT(*) FROM signs WHERE llm_description IS NOT NULL").fetchone()[0]
    sql = "SELECT id, description, image_path FROM signs WHERE llm_description IS NULL ORDER BY id"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    pending = con.execute(sql).fetchall()
    rows = {r[0]: r for r in pending}
    print(f"全库 {total} 条；已完成 {already}；本次待处理 {len(pending)} 条；并发 {args.workers}，detail={args.detail}",
          flush=True)
    if not pending:
        write_changelog(con)
        print("无待处理项，已是完整 v2。已刷新 README.md", flush=True)
        con.close()
        return 0

    def work(sid):
        _, desc, img = rows[sid]
        d, u = llm.describe_corpus_image(config.resolve_image(img), desc, detail=args.detail)
        return sid, config.render_corpus_text(d), d, u

    done = fail = 0
    cost = 0.0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in rows}
        for fut in cf.as_completed(futs):
            sid = futs[fut]
            try:
                sid, txt, struct, u = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                if fail <= 20:
                    print(f"  ✗ id={sid} 失败：{type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            con.execute("UPDATE signs SET llm_description=?, llm_struct=? WHERE id=?",
                        (txt, json.dumps(struct, ensure_ascii=False), sid))
            done += 1
            cost += u["prompt_tokens"] * IN_PRICE + u["completion_tokens"] * OUT_PRICE
            if done % 25 == 0:
                con.commit()
            if done % 100 == 0 or done == len(pending):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(pending) - done) / rate if rate else 0
                print(f"  [{done}/{len(pending)}] 成功{done} 失败{fail} | "
                      f"{rate:.1f}条/s | 已花${cost:.2f} | 已用{el/60:.1f}min ETA{eta/60:.1f}min", flush=True)
    con.commit()
    write_changelog(con)
    total_done = con.execute("SELECT COUNT(*) FROM signs WHERE llm_description IS NOT NULL").fetchone()[0]
    con.close()
    print(f"\n本次完成 {done} 条，失败 {fail} 条，用时 {(time.time()-t0)/60:.1f}min，本次成本 ≈ ${cost:.2f}（¥{cost*7.2:.0f}）", flush=True)
    print(f"v2 库累计完成 {total_done}/{total}。", flush=True)
    if total_done < total:
        print(f"仍有 {total-total_done} 条未完成，重跑本脚本可自动续跑。", flush=True)
    print(f"已写入 {V2DIR/'README.md'} 变更说明。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
