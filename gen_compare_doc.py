"""生成 v1 vs v2 描述对比文档（格式优美的 md，含插图）。

从 sign-language-database-v2/signs.db 抽取代表性词条，渲染原描述 / LLM 增强描述 / 结构化字段对比。
输出：sign-language-database-v2/v1-v2描述对比.md（图片用相对 images/ 路径）。
"""
import json
import sqlite3
import time

import config

V2DB = config.PROJECT_ROOT / "sign-language-database-v2" / "signs.db"
OUT = config.PROJECT_ROOT / "sign-language-database-v2" / "v1-v2描述对比.md"

# (分组标题, 看点说明, [词条 id])
GROUPS = [
    ("🀄 比划字形类手势", "用手摆出一个汉字。LLM 既保留原文“搭成X字形”的语义锚点，又逐指写清手型——这类词条 v2 召回改善最大（“王”从纯向量召回 #330 → #1）。",
     [5208, 1848, 1647]),
    ("🎯 测试题目标词", "本项目测试题对应的目标词条，LLM 描述与“用户拍照后由模型生成的描述”风格一致，利于匹配。",
     [878, 5653, 1201, 3598]),
    ("✋ 单手手势", "LLM 补全了原文未明说的掌心朝向、运动方向与具体位置。",
     [100, 6670, 1095]),
    ("👐 双手手势", "LLM 明确了双手的相对关系与接触方式（一前一后 / 相对 / 搭接）。",
     [31, 248]),
    ("📐 多步骤复杂手势", "LLM 把（一）（二）多步动作的手型、朝向、运动逐步拆解得更细。",
     [25, 500]),
]
TEST_TAG = {5208: "测试题 4", 1848: "测试题 5", 878: "测试题 1", 5653: "测试题 2", 3598: "测试题 3"}
FIELD_LABEL = [("hands", "手数"), ("handshape", "手型"), ("orientation", "朝向"),
               ("location", "位置"), ("movement", "运动"), ("two_hand_relation", "双手关系"),
               ("resembles", "比划字形")]


def fetch(con, sid):
    row = con.execute(
        """SELECT id, description, llm_description, llm_struct, image_path,
                  (SELECT group_concat(text,'、') FROM meanings WHERE sign_id=signs.id)
           FROM signs WHERE id=?""", (sid,)).fetchone()
    return row


def quote(text: str) -> str:
    """渲染为 md 引用块，多步 \\n → <br>。"""
    return "> " + (text or "").strip().replace("\n", "<br>")


def struct_line(js: str) -> str:
    try:
        d = json.loads(js or "{}")
    except json.JSONDecodeError:
        return ""
    parts = []
    for key, label in FIELD_LABEL:
        v = d.get(key, "")
        if isinstance(v, str) and v.strip() and v.strip().lower() not in ("uncertain", "无", ""):
            parts.append(f"`{label}` {v.strip()}")
    steps = d.get("steps") or []
    if steps:
        parts.append(f"`步骤` {len(steps)} 步")
    return "　｜　".join(parts)


def main():
    con = sqlite3.connect(f"file:{V2DB}?mode=ro", uri=True)
    total = con.execute("SELECT COUNT(*) FROM signs").fetchone()[0]

    L = []
    L.append("# 见手知意 SignSeek · 描述增强前后对比（v1 → v2）\n")
    L.append("本文档对比每个手势词条在**加入 GPT-5.5 视觉描述前后**的内容差异，便于直观感受 v2 增强的价值。\n")
    L.append("- **📖 v1 原描述**：词典自带的官方打法文字。")
    L.append("- **🤖 v2 LLM 增强**：GPT-5.5 同时阅读「词典线描图 + 原文」后重写的规范化描述（图文互证、措辞与查询侧对齐）。")
    L.append("- **🔖 结构化字段**：从 LLM 描述抽取的机读要素（手型 / 朝向 / 运动 / 比字形 …）。")
    L.append(f"\n> 实际检索时用的是**叠加文本** = 原描述 + LLM 增强描述（既保留原文锚点，又获得规范手型细节）。\n")
    L.append(f"> 全库 {total} 条均已增强；下面是各类别的代表性示例。\n")

    # 总览表
    L.append("## 📑 速览\n")
    L.append("| 词条 | id | 类别 | 关联 |")
    L.append("|---|---|---|---|")
    for title, _, ids in GROUPS:
        cat = title.split(" ", 1)[1] if " " in title else title
        for sid in ids:
            r = fetch(con, sid)
            if not r:
                continue
            words = (r[5] or "")[:14]
            tag = TEST_TAG.get(sid, "—")
            L.append(f"| [{words}](#sign-{sid}) | {sid} | {cat} | {tag} |")
    L.append("")

    # 分组明细
    for title, note, ids in GROUPS:
        L.append(f"\n## {title}\n")
        L.append(f"*{note}*\n")
        for sid in ids:
            r = fetch(con, sid)
            if not r:
                continue
            _id, desc, llm_desc, struct, img, words = r
            tagline = f"　·　🎯 {TEST_TAG[sid]}" if sid in TEST_TAG else ""
            L.append(f'<a id="sign-{sid}"></a>')
            L.append(f"### {words}")
            L.append(f"`id {sid}`　·　`{img.split('/')[-1]}`{tagline}\n")
            L.append(f'<img src="{img}" width="240" alt="{words}" />\n')
            L.append("**📖 v1 · 词典原描述**\n")
            L.append(quote(desc) + "\n")
            L.append("**🤖 v2 · LLM 增强描述**\n")
            L.append(quote(llm_desc) + "\n")
            sline = struct_line(struct)
            if sline:
                L.append(f"**🔖 结构化字段**　{sline}\n")
            L.append("---\n")

    L.append(f"\n*由 `reverse-lookup/gen_compare_doc.py` 生成于 {time.strftime('%Y-%m-%d %H:%M')}　·　模型 GPT-5.5　·　仅供研究/教育/无障碍非商业用途*")
    con.close()

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"已生成 {OUT}")
    print(f"  词条数：{sum(len(g[2]) for g in GROUPS)}　·　分组：{len(GROUPS)}　·　行数：{len(L)}")


if __name__ == "__main__":
    main()
