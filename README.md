# 见手知意 · SignSeek

> 看见手势，便知其意。

一个手语视觉反查工具：拍下自己做手语动作的**多张关键帧照片**，从《国家通用手语词典》6699 个词条里找出**最匹配的 top-5**（含词典插图、中文词、打法描述、置信度与理由）。

> 代码目录 `reverse-lookup/`；数据库 `sign-language-database/`（v2 增强版见 `sign-language-database-v2/`）。

## 思路

用户照片是「真人单手彩照」，词典图是「黑白线描半身像＋运动箭头」，画风与信息维度差异大，**不走图对图像素匹配**。改由多模态大模型换赛道：

1. **理解**：GPT-5.5 看关键帧 → 用词典词汇输出结构化描述（手型/朝向/位置/运动）。
2. **召回**：把描述渲染成贴库措辞的一句话 → 本地 BGE 中文向量 → 余弦检索 6699 条 → top-40。
3. **重排**：先文本重排 40→10（省 token），再把原始帧＋候选线描图一起喂模型视觉重排 10→5。

> 局限：单帧丢失「位置/运动」两个手语要素，故定位为 top-K 召回而非精确命中。建议多帧、带上半身入镜以提升准确率。

> **「比字形」手势**（王/十/口/工 等用双手摆出一个汉字）：这类词条在词典里是语义化描述（如"搭成'王'字形"），纯向量召回会漏（实测"王"曾排到第 5457 名）。理解阶段会输出 `resembles` 字段识别所比划的字符，并对该字符做**词法增召回**注入重排池——修复后"王"回到 Top-1。

## 安装与配置

```bash
cd reverse-lookup
python3 -m pip install -r requirements.txt     # 主要补 sentence-transformers
cp .env.example .env                            # 编辑填入 LLM_API_KEY（.env 已被 gitignore）
```

`.env`：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL=gpt-5.5`。

## 使用

```bash
python3 smoke_test.py                            # ① 先锁定网关格式（关键）
python3 build_index.py                           # ② 建索引（首次下 BGE ~100MB；6699 条秒级）
python3 query.py "../测试题/测试题 1.jpg" --open    # ③ 查询（单帧）
python3 query.py f1.jpg f2.jpg f3.jpg --open      #    多关键帧
python3 query.py img.jpg --no-visual-rerank       #    省 token 调试档
python3 web_app.py --open                         #    本地网页前端（上传多帧，展示 top-9）
```

常用参数：`--top-k`（默认 5）、`--top-recall`（默认 40）、`--no-visual-rerank`、`--max-rerank-images`、`--json-only`、`--open`、`--backend local|api`。

## 文件

| 文件 | 职责 |
|---|---|
| `config.py` | 路径、prompt、`QUERY_JSON_SCHEMA`、`render_query_text` |
| `embedder.py` | 本地 BGE 向量（默认）＋ 网关 embeddings 兜底 |
| `llm.py` | GPT-5.5 调用封装（三级降级）＋ 理解/文本重排/视觉重排 |
| `build_index.py` | signs.db →(只读)→ `index/`（embeddings.npy + meta.json） |
| `query.py` | 在线主流程 ＋ HTML 报告 |
| `smoke_test.py` | 网关连通与格式探测 |
| `web_app.py` | 本地网页前端与上传 API，复用 `query.run_query()` |

## 网关降级矩阵（`llm.chat` 自动处理，首次探测后记忆）

| 能力 | 首选 | 退化 1 | 退化 2 |
|---|---|---|---|
| token 上限 | `max_completion_tokens` | `max_tokens` | — |
| 结构化输出 | `json_schema`(strict) | `json_object` | 纯文本抽 JSON |
| 端点 | 裸域名自动补 `/v1` | 原样 URL | — |
| reasoning | `reasoning_effort` | 丢弃 | — |

## 验收

- `smoke_test.py` 全 PASS（端点/token/json/识图/端到端描述）。
- `build_index.py` 产出 `index/embeddings.npy`（6699×512）且自检命中自身。
- 两测试题各出可读 HTML 报告（左查询帧／右 top-5 候选线描图＋词＋打法＋置信＋理由）。
- 反向自测：拿某词条描述/插图当查询，能在 top-5 召回自身。

## 注意

- 数据库目录为只读事实源，本项目一律只 `SELECT`，从不写入。
- 仅供研究/教育/无障碍非商业用途（与数据库 license 一致）。
- 🔐 `.env` 含密钥，切勿提交；key 若曾泄露建议轮换。

## v2 实验：用 GPT-5.5 重描述词库（已验证）

动机：词库侧是词典原文（措辞与查询侧 GPT 描述不对齐），"王"这类靠语义描述的词条纯向量召回会漏（曾 #5457）。
做法：`build_corpus_v2.py` 用 GPT-5.5 读「线描图 + 词典原文」重生成结构化描述，存入 `sign-language-database-v2/signs.db`（新增 `llm_description` / `llm_struct` 列）。`eval_v2.py` 用 6 张测试题做 A/B/C 召回对比。

**100 条验证结论**（成本实测 ≈ $1.44 / ¥10.4，全库 6699 约 $98）：

| 方案 | 说明 | 6699 全量成本 |
|---|---|---|
| v1 原描述 | 现状 | $0 |
| v2 纯重描述（替换） | 救回"王"(#32→#1)，但丢原文锚点，"工会"反降(#2→#17) | ~$98 |
| **v3 原文+重描述叠加** | **5 题 4 题最佳，综合赢家** | ~$98 |

**关键结论：思路有效，但要"叠加"而非"替换"**——重描述会丢失原文里"工字形"这类语义锚点，须保留原文再追加 LLM 描述。iconic 锚点还可配合 `resembles` 字段单独加权。注意：在线 `query.py` 已有 `resembles`+词法增召回，iconic 词条在实际流程中已被兜住；本实验是纯 embedding 召回（不含词法增召回）以隔离"描述质量"的影响。

> 若要全量推进：建议走 v3 叠加 + `detail=low`（图 token 降 ~70%，约 $25–30 跑完 6699），并先扩充带答案评测集。
