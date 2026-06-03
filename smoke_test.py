"""连通性冒烟测试 —— 最先运行，锁定 mindracode 网关的确切调用格式。

逐项探测并打印 PASS/FAIL：
  1. base_url 端点（/v1 还是裸域名）
  2. token 上限参数（max_completion_tokens vs max_tokens）
  3. 结构化输出（json_schema / json_object / 纯文本）
  4. 单图多模态识别
  5. 端到端 describe_frames

任一项 FAIL 时给出修复建议；建议据此修正 .env 再继续 build_index / query。
"""
import sys

import config
from llm import _normalize_base_url, build_image_content, build_text_content, extract_json

OK = "\033[32mPASS\033[0m"
NO = "\033[31mFAIL\033[0m"


def _try(client, **kwargs):
    try:
        r = client.chat.completions.create(model=config.LLM_MODEL, **kwargs)
        content = r.choices[0].message.content
        if not content:
            return False, "content 为空（可能仅返回 reasoning）"
        return True, content
    except Exception as e:  # noqa: BLE001 —— 冒烟测试需要捕获一切并报告
        return False, f"{type(e).__name__}: {e}"


def _ping(client, token_param="max_completion_tokens"):
    return _try(client, messages=[{"role": "user", "content": "回复一个字：好"}], **{token_param: 16})


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print(f"模型: {config.LLM_MODEL}   配置 base_url: {config.LLM_BASE_URL}")
    if not config.LLM_API_KEY:
        print(f"{NO} 未检测到 LLM_API_KEY。请 `cp .env.example .env` 并填入 key。")
        return 1
    print(f"{OK} 已检测到 LLM_API_KEY（{config.LLM_API_KEY[:4]}…，长度 {len(config.LLM_API_KEY)}）")

    # 1. 端点探测：试多个 base_url 变体，找出能连通的
    print("\n[1] 端点 / 连通性")
    raw = config.LLM_BASE_URL.rstrip("/")
    variants = list(dict.fromkeys([_normalize_base_url(config.LLM_BASE_URL), raw, raw + "/v1"]))
    working_url, token_param = None, "max_completion_tokens"
    for url in variants:
        client = OpenAI(base_url=url, api_key=config.LLM_API_KEY)
        ok, out = _ping(client, "max_completion_tokens")
        if not ok and ("max_completion_tokens" in out or "max_tokens" in out or "token" in out.lower()):
            ok2, out2 = _ping(client, "max_tokens")
            if ok2:
                ok, out, token_param = True, out2, "max_tokens"
        if ok:
            print(f"  {OK} {url}  → 回复：{out.strip()[:30]}")
            working_url = url
            break
        print(f"  {NO} {url}  → {out[:80]}")
    if not working_url:
        print(f"\n{NO} 所有端点均连不通。请核对 base_url / key / 网关是否可用。")
        return 1
    if working_url != _normalize_base_url(config.LLM_BASE_URL):
        print(f"  ⚠ 建议把 .env 的 LLM_BASE_URL 改为：{working_url}")
    client = OpenAI(base_url=working_url, api_key=config.LLM_API_KEY)

    # 2. token 参数
    print("\n[2] token 上限参数")
    ok_mct, _ = _ping(client, "max_completion_tokens")
    ok_mt, _ = _ping(client, "max_tokens")
    token_param = "max_completion_tokens" if ok_mct else ("max_tokens" if ok_mt else token_param)
    print(f"  max_completion_tokens: {OK if ok_mct else NO}   max_tokens: {OK if ok_mt else NO}   → 采用 {token_param}")

    # 3. 结构化输出
    print("\n[3] 结构化输出模式")
    jmsg = [{"role": "user", "content": '返回 JSON：{"ok": true}'}]
    schema = {"type": "json_schema", "json_schema": {"name": "t", "strict": True,
              "schema": {"type": "object", "additionalProperties": False,
                         "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}}
    ok_schema, _ = _try(client, messages=jmsg, response_format=schema, **{token_param: 32})
    ok_obj, _ = _try(client, messages=jmsg, response_format={"type": "json_object"}, **{token_param: 32})
    json_cap = "json_schema" if ok_schema else ("json_object" if ok_obj else "text(抽取兜底)")
    print(f"  json_schema: {OK if ok_schema else NO}   json_object: {OK if ok_obj else NO}   → 采用 {json_cap}")

    # 4. 单图多模态
    print("\n[4] 多模态识图")
    test_img = config.TEST_DIR / "测试题 1.jpg"
    if not test_img.exists():
        print(f"  ⚠ 找不到测试图 {test_img}，跳过")
    else:
        content = [build_text_content("这是一只手的照片。用一句话客观描述手型（哪些手指伸直/弯曲/相捏）。"),
                   build_image_content(test_img, detail="auto")]
        ok_v, out_v = _try(client, messages=[{"role": "user", "content": content}], **{token_param: 200})
        print(f"  {OK if ok_v else NO}  {out_v.strip()[:120] if ok_v else out_v[:120]}")

    # 5. 端到端 describe_frames（用探测到的可用 url）
    print("\n[5] 端到端 describe_frames")
    import llm
    config.LLM_BASE_URL = working_url
    llm._client = None
    llm._TOKEN_PARAM = token_param
    llm._JSON_CAP = {"json_schema": "json_schema", "json_object": "json_object"}.get(json_cap, "text")
    try:
        desc = llm.describe_frames([config.TEST_DIR / "测试题 1.jpg"])
        print(f"  {OK} 结构化描述：")
        for k, v in desc.items():
            print(f"      {k}: {v}")
        print(f"  渲染检索文本：{config.render_query_text(desc)}")
    except Exception as e:  # noqa: BLE001
        print(f"  {NO} {type(e).__name__}: {e}")
        return 1

    print("\n" + "=" * 60)
    print("冒烟测试完成。若 [1] 有 ⚠ 建议，请同步修改 .env 后再 build_index / query。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
