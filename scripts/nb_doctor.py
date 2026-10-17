"""DBA 环境自检 —— 迁移/改配置后跑一次，确认没有静默坏掉的东西。

═══════════════════════════════════════════════════════════════
为什么需要这个
═══════════════════════════════════════════════════════════════
本项目的故障几乎都是**静默**的：脚本 exit 0、日志有输出、看起来一切正常，
只是记忆写去了错的地方、或检索永远返回 0 命中。人工逐项检查不可靠，
所以把检查固化成一个命令。

它检查的是**「会不会静默坏」**这一类问题，不是功能测试：

  1. 路径        —— 记忆是不是只有一份？旧位置有没有复活？
  2. 编码        —— 含中文的 .ps1 有没有 BOM？（丢了会整脚本解析失败）
  3. 索引        —— 在不在、新不新、能不能读（损坏必须显式报错）
  4. 向量        —— 可用吗？gpu 是不是真的在用？（**绝不接受 CPU 回退**）
  5. 集成副本    —— 仓库副本与生效位置是否一致？（漂移的集成代码比没有更坏）
  6. 计划任务    —— 上次跑成功了吗？步骤数对不对？
  7. 写入闸门    —— 记忆流闸门还在拦垃圾吗？

  py nb_doctor.py          # 全部检查
  py nb_doctor.py --quiet  # 只报失败项（给计划任务/CI 用）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_config as C  # noqa: E402

PASS, FAIL, WARN = "  ✓", "  ✗", "  !"
_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def bad(msg: str) -> None:
    print(f"{FAIL} {msg}")
    _failures.append(msg)


def warn(msg: str) -> None:
    print(f"{WARN} {msg}")
    _warnings.append(msg)


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 52 - len(title)))


# ═══════════ 1. 路径 ═══════════

def check_paths() -> None:
    section("路径")
    print(f"  ROOT = {C.ROOT}")
    if not (C.ROOT / "memory").is_dir():
        bad(f"记忆目录不存在：{C.MEMORY}")
        return

    for name, path, limit_kb in (
        ("core.md", C.CORE, C.CORE_MAX_KB),
        ("stream.md", C.STREAM, None),
    ):
        if not os.path.exists(path):
            bad(f"{name} 缺失：{path}")
            continue
        kb = os.path.getsize(path) / 1024
        if limit_kb and kb > limit_kb:
            bad(f"{name} {kb:.2f}KB **超上限 {limit_kb}KB** —— 每轮注入会超预算")
        else:
            ok(f"{name} {kb:.2f}KB" + (f" / {limit_kb}KB" if limit_kb else ""))

    ws = sorted(C.WS_DIR.glob("*.md")) if C.WS_DIR.is_dir() else []
    ok(f"ws/ {len(ws)} 册")

    # 旧位置不该复活：它复活 = 还有进程在用旧路径写
    old = Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "memories" / "notebook"
    if old.exists():
        warn(f"旧记忆目录**已复活**：{old}\n"
             f"      → 说明有进程仍在写旧路径。检查是否忘了重启客户端。")
    else:
        ok("旧位置 notebook/ 未复活")


# ═══════════ 2. 编码 ═══════════

def check_encoding() -> None:
    section("编码（含中文的脚本带 BOM）")
    # ⚠️ 这条检查是真踩出来的：edit 类工具改写 .ps1 会**剥掉 BOM**，
    # 而 powershell.exe 5.1 没有 BOM 就按 GBK 读 → 中文全部乱码 →
    # 字符串引号提前闭合 → **整个脚本解析失败**，且错误信息指向毫不相干的行。
    # 现象是计划任务 Last Result=1、日志一个字都没写。
    candidates = [
        Path.home() / ".dsh" / "scripts" / "memory-daily.ps1",
        C.ROOT / "integrations" / "dsh" / "memory-daily.ps1",
    ]
    bom = b"\xef\xbb\xbf"
    found = False
    for p in candidates:
        if not p.exists():
            continue
        found = True
        head = p.read_bytes()[:3]
        if head != bom:
            bad(f"缺 BOM（PS 5.1 会按 GBK 解析中文而崩）：{p}")
        else:
            ok(f"BOM 正常：{p.name}")
    if not found:
        warn("未找到 memory-daily.ps1，跳过")


# ═══════════ 3. 索引 ═══════════

def check_index() -> None:
    section("关键词索引")
    if not C.INDEX.exists():
        bad(f"索引不存在，检索会返回 0 命中。建：py nb_rebuild.py --kw")
        return
    try:
        import nb_index as I
        idx = I.load()
    except Exception as e:
        bad(f"索引损坏（{type(e).__name__}: {e}）—— 检索会崩或静默空。建：py nb_rebuild.py --kw")
        return
    if idx is None:
        bad("索引读不到")
        return
    n_docs = len(idx["docs"])
    ok(f"可读：{n_docs} 段 / {len(idx['postings'])} 词，{C.kb(C.INDEX):.1f}KB")

    # 索引应涵盖全部记忆正文；少了说明陈旧
    import nb_index as I2
    expected = len(I2.collect_docs())
    if n_docs < expected:
        warn(f"索引陈旧：{n_docs} 段 vs 源文件 {expected} 段。重建：py nb_rebuild.py --kw")
    else:
        ok(f"与源文件段数一致（{expected}）")

    # ── 时间权重的前提：索引里得有日期 ──
    # ⚠️ 这一段存在的原因是：时间权重**失效时表现得和正常一模一样**。
    #   索引没有 date 字段 → 所有段落权重都是 1.0 → 排序完全不变，
    #   而检索照常返回结果、没有任何报错。不主动查，没人会发现它没工作。
    cd = idx.get("corpus_dates") or {}
    if not cd:
        bad("索引无 corpus_dates 字段 → **时间权重不会生效**（旧版索引）。重建：py nb_rebuild.py --kw")
        return
    dated, undated = cd.get("dated", 0), cd.get("undated", 0)
    total = dated + undated
    if not dated:
        bad("索引内**没有任何带日期的段落** → 时间权重完全失效（每次检索都不按时间）")
    else:
        pct = 100.0 * undated / max(1, total)
        msg = (f"时间权重前提：{dated} 段有日期 / {undated} 段无日期"
               f"（{pct:.0f}% 无日期，跨 {cd.get('oldest')} ~ {cd.get('newest')}，"
               f"中性权重 {cd.get('neutral_weight')}，下限 {cd.get('min_weight')}）")
        if pct >= 80:
            # 无日期的段落用中性权重，不会被惩罚；但**时间权重对它们整体无效**。
            # 比例过高说明绝大多数记忆参与不了时间排序，机制形同虚设。
            warn(msg + "　⚠️ 无日期比例过高，时间权重对绝大多数段落无效")
        else:
            ok(msg)
        # 跨度太小 → 机制在跑但没有区分度
        # `_days_between` 对畸形日期返回 None（不再抛异常）→ 必须区分
        # 「算不出」与「跨 0 天」：后者是合法事实（语料真在同一天），
        # 前者是数据有问题。两者都说成 0 天就是把故障讲成事实。
        span = None
        try:
            span = I2._days_between(cd["oldest"], cd["newest"])
        except Exception as e:
            warn(f"跨度算不出（{type(e).__name__}: {e}）")
        if span is None:
            warn(f"跨度算不出：日期不合法（oldest={cd.get('oldest')} newest={cd.get('newest')}）")
        elif span <= 1:
            warn(f"语料只跨 {span} 天 → 时间权重**不改变任何排序**（等于空操作）")
        else:
            ok(f"语料跨 {span} 天，时间权重有实际区分度")

        # ── 畸形日期：记忆数据里的真实错误，必须报出来 ──
        # ⚠️ 这不是「没有日期」，是「日期写错了」。前者无事可做，后者该去改。
        # 不报的话它就只是「安静地没有时间权重」，永远没人知道记忆里存着错。
        n_bad = cd.get("malformed", 0)
        if n_bad:
            samples = "、".join(str(s) for s in (cd.get("malformed_samples") or [])[:3])
            warn(f"{n_bad} 段日期不合法（日历上不存在）如 {samples}"
                 f"　← 这些段落拿不到时间权重，请修正源文件里的日期")


# ═══════════ 4. 向量 ═══════════

def check_vector() -> None:
    section("向量层")
    try:
        import nb_vec
    except Exception as e:
        bad(f"nb_vec 导入失败：{type(e).__name__}: {e}")
        return
    if not C.VEC_INDEX.exists():
        warn("向量索引不存在（语义检索降级，关键词不受影响）。建：py nb_rebuild.py --vec")
        return
    # ⚠️ 不能用 stats() 判健康：它内部把 VecCorrupt 捕获掉、只返回一句人话
    # 字符串（"🚨 损坏：…"）。doctor 如果只打印那句话，**索引损坏时它照样
    # 退出码 0** —— 巡检工具自己变成静默降级，正是它该抓的东西。
    # 所以这里直接 load()，让异常爆出来；stats() 只用来说人话。
    try:
        loaded = nb_vec.load()
    except Exception as e:
        bad(f"向量索引损坏（{type(e).__name__}: {e}）→ 语义检索不可用。"
            f"建：py nb_rebuild.py --vec")
        return
    try:
        arr, meta = loaded
        n, dim = (arr.shape + (0,))[:2]
        ok(f"可读：{n} 段 × {dim} 维，{C.kb(C.VEC_INDEX):.1f}KB，模型 {meta.get('model', '?')}")
    except Exception as e:
        bad(f"向量索引结构异常（{type(e).__name__}: {e}）")
        return

    # ⚠️ GPU 检查是真格的：本项目硬性要求不走 CPU。
    # CPU 回退不会报错，只会慢到不可用，而使用者以为在用 GPU。
    # 判据也**不能用字符串包含**去猜：必须问出结构化状态。
    try:
        st = nb_vec.gpu_probe()  # → {"loaded": bool, "gpu_percent": float|None, "raw": str}
        if not st["loaded"]:
            # 未加载时问不到 GPU 占用。这不是故障，但**也不能当成通过** ——
            # 那样这条检查永远只会说「不知道」。主动加载一次再问。
            nb_vec.search("GPU 自检", top_k=1)
            st = nb_vec.gpu_probe()
        pct = st["gpu_percent"]
        if pct is None:
            bad(f"GPU 状态无法确认：{st['raw']} —— 无法排除 CPU 回退，不接受"
                f"（本项目禁止 CPU 推理）")
        elif pct >= 99.0:
            ok(f"GPU：{st['raw']}（{pct:.0f}% 在显存）")
        else:
            bad(f"**未完全在 GPU 上**：{st['raw']}（仅 {pct:.0f}%）"
                f" —— 不接受部分 CPU 回退，检查 GPU/驱动/Ollama")
    except Exception as e:
        bad(f"GPU 检查失败（{type(e).__name__}: {e}）—— 无法排除 CPU 回退")

    # 陈旧判定：不能只比文件是否存在，要看源文件是否比索引新
    try:
        if nb_vec.stale():
            warn("向量索引陈旧（源文件比索引新，新记忆召不回）。建：py nb_rebuild.py --vec")
        else:
            ok("向量索引不陈旧")
    except Exception as e:
        warn(f"陈旧检查跳过（{type(e).__name__}: {e}）")


# ═══════════ 5. 集成副本一致性 ═══════════

def check_integrations() -> None:
    section("集成副本与生效位置一致性")
    # ⚠️ 只在「本仓库就是本机装的那一份」时才比 —— 否则刚 clone 到别处的
    # 仓库会拿自己的副本去比本机的 live 安装，必然全部「漂移」，
    # 报一堆假错。假错比不报更坏：它会训练人忽略自检输出。
    # 判据：本机 live 目录是否真在用这个仓库（DBA_HOME 指向这里）。
    dba_home = os.environ.get("DBA_HOME")
    if dba_home:
        try:
            if Path(dba_home).resolve() != C.ROOT.resolve():
                warn(f"本仓库不是本机生效的那一份（DBA_HOME={dba_home}）"
                     f"→ 跳过一致性比较")
                return
        except OSError:
            pass
    else:
        # 没有 DBA_HOME：插件默认用 D:\AI\DBA。这里若不在那个位置，同样跳过。
        if C.ROOT.resolve() != Path(r"D:\AI\DBA").resolve():
            warn(f"本仓库不是本机生效的那一份（默认应为 D:\\AI\\DBA）"
                 f"→ 跳过一致性比较")
            return

    # 漂移的集成代码比没有更危险：它看起来是对的。
    home = Path.home()
    pairs = [
        (home / ".dsh" / "profiles" / "web" / "snowfox-inject.mjs",
         C.ROOT / "integrations" / "dsh" / "snowfox-inject.mjs"),
        (home / ".dsh" / "profiles" / "web" / "snowfox-sync.mjs",
         C.ROOT / "integrations" / "dsh" / "snowfox-sync.mjs"),
        (home / ".dsh" / "profiles" / "web" / "cordis.patch.yml",
         C.ROOT / "integrations" / "dsh" / "cordis.patch.yml"),
        (home / ".dsh" / "scripts" / "memory-daily.ps1",
         C.ROOT / "integrations" / "dsh" / "memory-daily.ps1"),
        (Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "plugins" / "snowfox-memory" / "__init__.py",
         C.ROOT / "integrations" / "hermes" / "snowfox-memory" / "__init__.py"),
    ]
    import hashlib

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for live, repo in pairs:
        if not live.exists():
            warn(f"生效位置不存在（未安装？）：{live.name}")
        elif not repo.exists():
            warn(f"仓库副本缺失：{repo.name}")
        elif live.name == "cordis.patch.yml":
            # ⚠️ 这一份**不能**按字节相等判：integrations/README.md 要求「把 `insert:`
            # 那两条合并进去，**别覆盖**你原有的条目」—— 生效版天然比仓库副本长。
            # 判据改成**包含**：仓库里每个 `- id:` 都得在生效版里找得到。
            # 反方向（生效版缺条目）才是真漂移，仍会被抓到；多余条目属本机自有配置。
            import re as _re
            ids = _re.findall(r"(?m)^\s*-?\s*id:\s*(\S+)", repo.read_text(encoding="utf-8"))
            live_ids = set(_re.findall(r"(?m)^\s*-?\s*id:\s*(\S+)", live.read_text(encoding="utf-8")))
            missing = [i for i in ids if i not in live_ids]
            if missing:
                bad(f"**已漂移**：{live.name} —— 仓库的 insert 条目在生效版里缺失：{', '.join(missing)}")
            else:
                ok(f"包含一致：{live.name}（仓库 {len(ids)} 条 insert 都在；多出的是本机自有条目）")
        elif h(live) == h(repo):
            ok(f"一致：{live.name}")
        else:
            bad(f"**已漂移**：{live.name} —— 生效版本与仓库副本不同，改哪份要想清楚")


# ═══════════ 6. 计划任务 ═══════════

def check_schedule() -> None:
    section("计划任务")
    import subprocess

    # ⚠️ 不能只问「任务上次成功了吗」—— 得先问「这个任务跑的是**本仓库**的脚本吗」。
    # 否则把仓库 clone 到别处一扫，会用本机 D:\AI\DBA 那个任务的结果去给这份
    # 仓库背书（或反过来判它失败）。**假错会训练人忽略自检输出**，和静默失败
    # 一样有害，只是方向相反。
    try:
        r = subprocess.run(
            ["schtasks", "/query", "/tn", "DSH_Memory_Daily", "/fo", "LIST", "/v"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
    except Exception as e:
        warn(f"查不到计划任务（{type(e).__name__}: {e}）")
        return
    out = r.stdout or ""
    if not out.strip():
        warn("计划任务 DSH_Memory_Daily 不存在或查不到")
        return

    # 任务命令行里应出现本仓库路径（或靠 DBA_HOME 指过来）
    task_cmd = ""
    last = ""
    for line in out.splitlines():
        s = line.strip()
        if s.lower().startswith(("task to run", "要运行的任务")):
            task_cmd = s.split(":", 1)[-1].strip()
        elif s.lower().startswith("last result"):
            last = s.split(":", 1)[-1].strip()

    root_str = str(C.ROOT)
    dba_home = os.environ.get("DBA_HOME") or ""
    if task_cmd and root_str.lower() not in task_cmd.lower() and dba_home.lower() not in task_cmd.lower():
        # 任务跑的是**另一份**仓库 —— 本仓库的脚本是否被调度，这里无从判断。
        warn(f"计划任务指向别的仓库：{task_cmd[:80]}\n"
             f"      → 本仓库不由它维护，跳过结果判定")
        return

    if last == "0":
        ok("上次运行结果 = 0")
    elif last:
        bad(f"上次运行结果 = {last}（非 0）。读日志：%LOCALAPPDATA%\\hermes\\memories\\cron.log")
    else:
        warn("未取到 Last Result")


# ═══════════ 7. 写入闸门 ═══════════

def check_write_gate() -> None:
    section("记忆流写入闸门")
    try:
        import nb_write
        junk = [
            "ok", "好的", "Hermes hit repeated errors",
            "Traceback (most recent call last):",
            "sk-abcdefghijklmnop", "https://example.com/x", "短",
        ]
        leaked = [j for j in junk if not nb_write._stream_junk_reason(j)]
        if leaked:
            bad(f"闸门放行了 {len(leaked)} 条垃圾：{leaked[:3]}")
        else:
            ok(f"闸门拦下全部 {len(junk)} 条测试垃圾")
        good = "完成 DBA 仓库迁移并建立向量层"
        if nb_write._stream_junk_reason(good):
            bad(f"闸门**误杀**了正常记录：{good}")
        else:
            ok("正常记录未被误杀")
    except Exception as e:
        warn(f"闸门检查跳过（{type(e).__name__}: {e}）")


CHECKS = [check_paths, check_encoding, check_index, check_vector,
          check_integrations, check_schedule, check_write_gate]


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    if not quiet:
        print("═" * 58)
        print("  DBA 环境自检")
        print("═" * 58)

    for fn in CHECKS:
        if quiet:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn()
        else:
            fn()

    print()
    if _failures:
        print(f"结论: ❌ {len(_failures)} 项失败")
        for f in _failures:
            print(f"   · {f.splitlines()[0]}")
        return 1
    if _warnings:
        print(f"结论: ⚠️ 通过，但有 {len(_warnings)} 项提醒")
        return 0
    print("结论: ✓ 全部正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
