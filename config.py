"""见手知意 SignSeek — 路径常量、环境配置、各阶段 prompt 与查询文本渲染。

被所有其他模块 import。不做任何外部调用（import 即安全）。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- 路径 ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent                 # reverse-lookup/
PROJECT_ROOT = ROOT.parent                             # 手语视觉反查查询项目/
DB_DIR = PROJECT_ROOT / "sign-language-database"       # 只读事实源
DB_PATH = DB_DIR / "signs.db"
TEST_DIR = PROJECT_ROOT / "测试题"

INDEX_DIR = ROOT / "index"                             # 生成物
REPORTS_DIR = ROOT / "reports"                          # 生成物
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.json"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

load_dotenv(ROOT / ".env")

# --- 模型 / 网关 --------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.mindracode.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.5")
EMB_MODEL = "BAAI/bge-small-zh-v1.5"                    # 本地中文向量，512 维


def resolve_image(image_path: str) -> Path:
    """词条 image_path（形如 images/v1_xxx.jpg）→ 绝对路径。"""
    return DB_DIR / image_path


# --- 查询结构化描述的 JSON schema（GPT-5.5 理解阶段输出）---------------
# strict 模式要求：所有字段 required + additionalProperties=false。
QUERY_JSON_SCHEMA = {
    "name": "sign_description",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "hands": {"type": "string", "description": "一只手 / 双手 / 不确定"},
            "handshape": {"type": "string", "description": "手型，必要时分别描述两只手，用词典词汇：食指直立、拇指与中指相捏、五指张开、握拳伸拇指等"},
            "orientation": {"type": "string", "description": "掌心/手背/指尖朝向，如 掌心向外、手背向外、虎口朝上"},
            "location": {"type": "string", "description": "相对身体的位置，如 胸前、耳旁；画面无身体参照则填 uncertain"},
            "movement": {"type": "string", "description": "运动轨迹/方向，如 从后向前、上下弯动；单帧静态无法判断则填 uncertain"},
            "expression": {"type": "string", "description": "面部表情；不可见则填 uncertain"},
            "iconicity": {"type": "string", "description": "象形/语义提示（手型像在模仿什么），可填空字符串"},
            "resembles": {"type": "string", "description": "若双手整体在比划一个具体的汉字/数字/字母/图形（如 王、十、口、工、3、Z、十字），写出那个字符或形状；否则填空字符串"},
            "uncertain": {"type": "array", "items": {"type": "string"}, "description": "因输入受限而无法确定的要素，如 movement、location"},
        },
        "required": ["hands", "handshape", "orientation", "location", "movement", "expression", "iconicity", "resembles", "uncertain"],
    },
}

_UNCERTAIN_TOKENS = {"", "uncertain", "未知", "不确定", "无法确定", "看不到", "不可见", "none", "n/a", "na"}


def render_query_text(desc: dict) -> str:
    """把结构化描述渲染成一句"贴库措辞"的描述，用于 embedding 召回。

    刻意模仿词典 description 风格（手型，朝向，位置，运动），跳过 uncertain 项，
    以缩小"GPT 口语描述 vs 线描书面语"的向量域差异。
    """
    def keep(v) -> bool:
        return isinstance(v, str) and v.strip().lower() not in _UNCERTAIN_TOKENS

    parts = []
    hands = desc.get("hands", "")
    if keep(hands) and hands.strip() not in {"一只手", "单手"}:
        parts.append(hands.strip())
    for key in ("handshape", "orientation", "location", "movement"):
        v = desc.get(key, "")
        if keep(v):
            parts.append(v.strip())
    text = "，".join(parts)
    icon = desc.get("iconicity", "")
    if keep(icon):
        text = f"{text}。{icon.strip()}" if text else icon.strip()
    res = desc.get("resembles", "")
    if keep(res):
        text = f"{text}，双手比划“{res.strip()}”字形" if text else f"双手比划“{res.strip()}”字形"
    return text or "（描述不足）"


# --- 各阶段 prompt ------------------------------------------------------
DESCRIBE_PROMPT = """你是中国《国家通用手语词典》编纂专家。下面是用户做某个手语动作的若干关键帧照片（按时间先后排列，可能只有一帧）。请仔细观察手部，用词典打法描述的词汇与风格，客观描述这个动作。

要求：
- 只描述画面中确实看到的内容。看不到或无法确定的要素（尤其单帧时的运动轨迹 movement、相对身体位置 location）必须填 "uncertain"，绝不臆造。
- 自拍画面可能左右镜像，请关注手指之间的相对关系，而非绝对左右手。
- 若画面只有手、没有身体参照，location 填 uncertain；只有单帧、看不出运动，movement 填 uncertain。
- **特别注意"比字形"手势**：中国手语里很多词是用单手/双手摆出一个汉字、数字或字母的字形——例如"王"=一手三指横伸当三横、另一手食指竖立当一竖，搭成"王"字；"十"=两食指交叉成十字；"口"=双手围成方框；"工"=类似搭法。只要你判断双手或单手在拼出某个字符或图形，务必在 resembles 字段写出那个字符（如 王、十、口、工、3、Z），并在 iconicity 说明它像什么。
- 用词参考：手型（食指直立、拇指与中指相捏成圆形、五指张开、握拳伸拇指、食中指分开…），朝向（掌心向内/向外、手背向外、虎口朝上…），位置（置于身体一侧、耳旁、胸前、头顶…），运动（从后向前、上下弯动、弧形转动…）。
严格按给定 JSON schema 输出。"""

RERANK_TEXT_PROMPT = """以下是用户某个手语动作的结构化描述（可能缺少运动 movement / 位置 location 信息），以及若干候选词典词条（含 id、中文词、打法描述）。

请只依据可对比的要素（主要是手型 handshape、朝向 orientation）判断哪些候选最可能与用户动作匹配，按可能性从高到低排序。用户描述中标为 uncertain 的要素，不要作为否决候选的理由。

只输出 JSON：{"ranking": [候选id, ...]}，最多 %d 个，按可能性降序。"""

RERANK_VISUAL_PROMPT = """第一组图是用户做手语动作的关键帧照片（真人单手实拍）。随后每个候选是词典里的一个词条：黑白线描示意图（含人物半身、虚线表示运动方向）+ 中文词 + 打法描述。

请对照判断用户动作最可能对应哪些词条，挑出最匹配的最多 %d 条，按可能性排序。

判断要点：
- 画风不同（真人实拍 vs 线描示意），请比对语义而非像素。
- 用户输入可能是单帧、缺少运动/位置信息——以手型、朝向为主要依据；运动、位置作为加分项，而非否决项。
- 自拍可能左右镜像，关注手指相对关系。

只输出 JSON：{"results": [{"id": 候选id, "confidence": 0到1的小数, "reason": "简短中文理由"}, ...]}。"""


# --- 语料重描述（v2）：用 GPT-5.5 读「词典线描图 + 词典原文」生成结构化描述 ----
# 与查询侧 QUERY_JSON_SCHEMA 对称（同字段名），但库侧允许更完整（含运动/分步），
# 以便两侧"逐字段可比"。图文互证：以原文为准、用图补全细节。
CORPUS_JSON_SCHEMA = {
    "name": "sign_corpus_description",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "hands": {"type": "string", "description": "一只手 / 双手"},
            "handshape": {"type": "string", "description": "手型，双手则分别描述。用规范词汇：食指直立、拇指与中指相捏、五指张开、握拳伸拇指、伸拇指与小指等"},
            "orientation": {"type": "string", "description": "掌心/手背/指尖朝向"},
            "location": {"type": "string", "description": "相对身体的位置，如 胸前、耳旁、身体一侧；图中无身体参照则填 uncertain"},
            "movement": {"type": "string", "description": "运动轨迹/方向（线描图常用虚线箭头表示），如 从后向前、上下弯动、弧形移动；静止填 静止"},
            "two_hand_relation": {"type": "string", "description": "双手关系：接触/交叉/一前一后/搭成字形等；单手填空字符串"},
            "resembles": {"type": "string", "description": "若整体比划一个汉字/数字/字母/图形（王/十/口/工/3/Z），写出该字符；否则空字符串"},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "若动作分多步（一）（二），逐步简述；单步则单元素或空数组"},
        },
        "required": ["hands", "handshape", "orientation", "location", "movement", "two_hand_relation", "resembles", "steps"],
    },
}

CORPUS_DESCRIBE_PROMPT = """你是中国《国家通用手语词典》编纂专家。下面给你一个词条的【官方打法文字】和【词典线描示意图】。

线描图说明：黑白线条画，常含人物半身；虚线/箭头表示手的运动方向；用（一）（二）等分格表示多步骤动作。

请图文互证，为这个词条生成一份**结构化、措辞规范**的手型动作描述：
- **以官方文字为准**（它是权威答案），用线描图补全文字未明说的细节（如具体朝向、运动方向、双手接触关系）。
- 若文字与图有出入，以文字为准。
- 用规范统一的词汇描述手型/朝向/位置/运动，便于与"他人对同一手势的描述"逐项比对。
- 多步骤动作在 steps 里逐步写清；运动方向尽量明确（从箭头读）。
- 若在比划某汉字/字母/数字，resembles 写出该字符。
严格按给定 JSON schema 输出。"""


def render_corpus_text(desc: dict) -> str:
    """库侧结构化描述 → 检索文本。与 render_query_text 同构，但纳入更完整要素。"""
    def keep(v) -> bool:
        return isinstance(v, str) and v.strip().lower() not in _UNCERTAIN_TOKENS

    parts = []
    hands = desc.get("hands", "")
    if keep(hands):
        parts.append(hands.strip())
    for key in ("handshape", "orientation", "two_hand_relation", "location", "movement"):
        v = desc.get(key, "")
        if keep(v) and v.strip() not in {"静止"}:
            parts.append(v.strip())
    text = "，".join(parts)
    res = desc.get("resembles", "")
    if keep(res):
        text = f"{text}，比划“{res.strip()}”字形" if text else f"比划“{res.strip()}”字形"
    return text or "（描述不足）"
