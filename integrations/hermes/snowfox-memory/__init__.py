"""DBA 三层记忆 + 向量层插件（Hermes 侧）。

三层与写入者（用户定的分工）：
    core.md     核心记忆   云端模型写   上限 15KB   每轮全量注入
    stream.md   记忆流     本地模型写   不限长度    仅检索
    ws/*.md     工作区册   云端模型写   不限长度    仅检索

本插件做三件事：
    1. pre_llm_call  以本轮用户输入为 query 检索记忆 → 组装 → override_messages
    2. 把本轮对话追加为记忆流候选（写入 pending 区，供本地模型后续提炼成时间线）
    3. 索引维护：数据文件比索引新时重建，避免「刚写的记忆检索不到」

⭐ 2026-10-06 迁移：记忆数据与工具链已从 hermes 下搬到 **DBA 仓库 `D:\\AI\\DBA`**。
   本插件现在只是个**适配器**：它不再自己拼路径，而是导入 DBA 的脚本，
   由 `nb_config` 决定数据在哪（可用 DBA_HOME 覆盖）。这样换客户端不必搬记忆。

索引规则（用户定义）：
    用户输入 → 本地模型索引记忆 → 云端模型
                                       ↓ 记忆不清晰
                                 云端模型可主动再次索引
本插件实现「第一次索引」：把用户输入当查询打出检索结果。
二次索引由云端模型通过 nb_search.py / nb_vec.py / nb_assemble.py 主动发起
（检索为空或命中分数低时），插件不擅自重试。

与旧五级插件的差别：不再有 L1/L2/L3/L4 分层压缩、不再有 mem_compress 级联。
旧数据已迁为 *.legacy，明细见 DBA 仓库的 memory/ws/*.md。
"""
import logging
import os
import sys
import time
from pathlib import Path

# ⭐ 记忆工具链现在在 DBA 仓库内，跟着数据走。
#   以前是「数据在 hermes/memories、脚本在 ~/.dsh/scripts」，两处路径靠约定同步；
#   现在统一到 DBA_HOME 一个根，数据和代码不可能再各走一边。
_DBA_HOME = Path(os.environ.get("DBA_HOME") or r"D:\AI\DBA")
_DBA_SCRIPTS = _DBA_HOME / "scripts"
if str(_DBA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DBA_SCRIPTS))

logger = logging.getLogger("dba-memory")


def _m() -> Path:
    """记忆数据目录。

    ⚠️ 不再是 hermes/memories —— 那只是**旧位置**。现在由 DBA 的 nb_config
    决定（它读 DBA_HOME 环境变量），本函数只是给还没迁移的调用点一个统一入口。
    保留这个函数而不是删除，是为了让仍写着 `_m() / "xxx"` 的代码在迁移期
    不至于立刻崩；但**新增代码一律用 nb_config 的路径常量**，不要在这里拼。
    """
    try:
        import nb_config
        return nb_config.MEMORY
    except Exception:
        return _DBA_HOME / "memory"


def _imports():
    """惰性导入 DBA 记忆模块。失败返回 None，插件整体降级不影响会话。"""
    try:
        import nb_assemble
        import nb_bridge
        import nb_config
        import nb_write
        return nb_assemble, nb_bridge, nb_config, nb_write
    except Exception as e:
        logger.error(f"[dba] import failed: {e}")
        return None


# ── 系统注入过滤：平台合成的英文指令不属于真实用户输入 ──
_SYSTEM_INJECT_MARKERS = (
    "review the conversation above and update the skill library",
    "consider saving to memory if appropriate",
    "a pass that does nothing is a missed learning opportunity",
    "target shape of the library",
    "has the user revealed things about themselves",
    "be active — most sessions",
)

_USER_LIMIT = 300
_ASST_LIMIT = 600


def _is_system_inject(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    tl = t.lower()
    return any(m in tl for m in _SYSTEM_INJECT_MARKERS)


def _clip(text: str, limit: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[:limit] + " …"


def _extract_turn(conversation_history):
    """从会话历史取出最近一轮 user/assistant 文本。"""
    lu = la = pu = None
    for msg in conversation_history or []:
        r = msg.get("role", "")
        c = msg.get("content", "")
        if c is None:
            continue
        src = msg.get("source") or {}
        if isinstance(src, dict) and src.get("kind") and src["kind"] != "user":
            continue                      # 平台合成消息不记
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c
                          if isinstance(p, dict) and p.get("type") == "text")
        sc = str(c).strip()
        if r == "user" and sc:
            if _is_system_inject(sc):
                continue
            if lu is not None:
                la = None
            lu, pu = sc, sc
        elif r == "assistant" and sc and lu is not None:
            la, pu = sc, None
    return lu, la, pu


def _write_pending(user_msg: str, session_id: str) -> None:
    """把本轮用户输入记为「待补全」，回复到达后由 post_llm_call 补上助手侧。

    为什么仍要 pending：记忆流要落的是「干过什么」（= 助手侧动作），
    而 pre_llm_call 时助手还没产出。先占位、后回填，避免丢事件。
    """
    mods = _imports()
    if not mods:
        return
    _, _, nb_config, _ = mods
    try:
        nb_config.ensure_dirs()
        p = _m() / ".pending.jsonl"
        import json
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               "session": session_id, "user": _clip(user_msg, _USER_LIMIT)}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"[dba] pending write skipped: {e}")


def _flush_pending(assistant_msg: str, session_id: str) -> int:
    """把配对上的 pending 记录提炼为记忆流条目（一行一条）。"""
    mods = _imports()
    if not mods:
        return 0
    _, _, nb_config, nb_write = mods
    p = _m() / ".pending.jsonl"
    if not p.exists():
        return 0
    import json
    kept, written = [], 0
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    act = _clip(assistant_msg, _ASST_LIMIT)
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if rec.get("session") != session_id:
            kept.append(ln)
            continue
        # 记忆流存「干过什么」：取助手侧动作，不存用户提问
        nb_write.append_stream(
            f"{act[:120]}",
            at=rec.get("ts"),
            tag="",
            session=session_id,
        )
        written += 1
    try:
        if kept:
            p.write_text("\n".join(kept) + "\n", encoding="utf-8")
        else:
            p.unlink()
    except OSError:
        pass
    return written


def _on_pre_llm_call(session_id="", conversation_history=None, **kw):
    mods = _imports()
    if not mods or not session_id:
        return None
    nb_assemble, nb_bridge, _nb_config, _nb_write = mods

    lu, _la, pu = _extract_turn(conversation_history)
    if pu:
        _write_pending(pu, session_id)

    query = lu or kw.get("user_message", "") or ""
    try:
        res = nb_bridge.rebuild_assembly(query)
    except Exception as e:
        logger.error(f"[dba] assembly failed: {e}")
        return None

    logger.info(f"[dba] assembly {res['bytes']}B core={res['core_kb']}KB "
                f"hits={len(res['retrieval']['hits'])} index={res['index']}")

    asm = _m() / "_assembled_context.md"
    if not asm.exists():
        return None
    try:
        body = asm.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # 剥首行 built 时间戳：每轮都变，留着会让 prompt 前缀缓存全失效
    import re
    body = re.sub(r"^<!--[^\n]*-->\r?\n?", "", body)
    combined = f"{body}\n\n---\n\n## Current Turn\n\n{kw.get('user_message', '') or query}"
    return {"override_messages": [{"role": "user", "content": combined}]}


def _on_post_llm_call(response_text="", session_id="", **kw):
    if not session_id or not response_text:
        return
    n = _flush_pending(response_text, session_id)
    if n:
        logger.info(f"[dba] stream += {n} event(s)")


def _on_session_end(session_id="", completed=False, **kw):
    if not session_id or not completed:
        return
    mods = _imports()
    if not mods:
        return
    _, nb_bridge, _, _ = mods
    try:
        nb_bridge.ensure_index(force=True)
        logger.info("[dba] index rebuilt at session end")
    except Exception as e:
        logger.debug(f"[dba] index rebuild skipped: {e}")


def register(ctx):
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("[dba] hooks: pre_llm, post_llm, session_end")
    mods = _imports()
    if mods:
        try:
            _, nb_bridge, nb_config, _ = mods
            nb_bridge.ensure_index()
            logger.info("[dba] ready\n" + nb_config.summary())
        except Exception as e:
            logger.error(f"[dba] startup init failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = _imports()
    print("dba-memory: core(全量) + stream/ws(检索) + 二级检索")
    if m:
        print(m[2].summary())
