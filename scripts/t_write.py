"""nb_write 写入侧闸门复验。

⚠️ 只测 **core.md 整体替换的缩水闸门**（2026-10-11 实测事故驱动）；
记忆流的写入闸门（`_stream_junk_reason`）在 `t_index_health.py` 里，不重复。

事故原形：`nb_write.py core --title X --file notes.md` 本意是追加一段笔记，
实际走的是 `replace_core`（整体覆盖），把 15331B 的 core.md 换成 716B，
**退出码 0、回执只有一行尺寸** —— 破坏性操作没有防呆，且失败不报警。

闸门要钉死三件事，缺一不可：
  1. 该拒的必须拒，**且 core.md 一个字节都没动**（拒绝 ≠ 写一半）；
  2. 该放的必须放 —— 闸门太紧同样有害：压缩 core 是 replace 的正当用法，
     误拦会让云端模型丢掉刚整理好的内容，那是**丢记忆**（比超限严重）；
  3. 判据必须与落盘同一把尺子（`_disk_bytes`，Windows 下 `\\n`→`\\r\\n` 每行 1 字节）。
"""
import sys, tempfile, shutil
sys.path.insert(0, r'D:\AI\DBA\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import nb_config as C
import nb_write as W
from pathlib import Path

saved = {k: getattr(C, k) for k in ('ROOT', 'MEMORY', 'CORE', 'WS_DIR')}
tmp = Path(tempfile.mkdtemp())
fails = []


def check(label: str, ok: bool, detail: str = '') -> None:
    if not ok:
        fails.append(label)
    print('  %s %-42s %s' % ('✓' if ok else '✗ 错', label, detail))


def replace_with(body_text: str, force: bool = False):
    """把 body_text 写成源文件后调 replace_core。返回 (是否被拒, 原因首行)。"""
    src = tmp / 'src.md'
    src.write_text(body_text, encoding='utf-8')
    try:
        W.replace_core(str(src), force=force)
        return False, ''
    except W.WriteError as e:
        return True, str(e).split('\n')[0]


def set_core(body_text: str) -> int:
    """按生产路径写入 core，返回磁盘字节数。"""
    C.CORE.write_text(body_text, encoding='utf-8')
    return C.core_bytes()


try:
    C.ROOT = tmp
    C.MEMORY = tmp / 'memory'
    C.CORE = C.MEMORY / 'core.md'
    C.WS_DIR = C.MEMORY / 'ws'
    C.MEMORY.mkdir(parents=True, exist_ok=True)

    # 接近真实体积的 core（写入方式与生产路径一致，保证「旧值」与「新值」同尺）
    old_body = ''.join('- 第 %d 条核心记忆，内容用于占位；⚠️ 含中文与符号 `x`\n' % i
                       for i in range(300))
    old_disk = set_core(old_body)
    print('旧 core = %dB，闸门阈值 = %dB\n' % (old_disk, int(old_disk * W._CORE_SHRINK_RATIO)))

    print('══ A. 误用 --file 当追加：必须拒，且 core.md 原封不动 ══')
    before = C.CORE.read_bytes()
    rejected, why = replace_with('- V4.1 Pro 已按账号灰度，API 未放出。\n- 官方无日期/id/价格。\n')
    after = C.CORE.read_bytes()
    check('缩水到 3 行 → 拒绝', rejected, why[:64])
    check('拒绝后 core.md 字节不变', before == after,
          '%dB → %dB' % (len(before), len(after)))

    print()
    print('══ B. 同一份内容 + --force = 明确意图，必须放行 ══')
    one_line = '- 只有一行。\n'
    rejected_f, why_f = replace_with(one_line, force=True)
    check('force 覆盖 → 通过', not rejected_f, why_f[:64])
    check('落盘字节 = 源文件换算字节',
          C.core_bytes() == W._disk_bytes(one_line), '%dB' % C.core_bytes())

    print()
    print('══ C. 正当压缩（缩水不到一半）：不该拦 ══')
    set_core(old_body)
    rejected_c, why_c = replace_with(old_body[:int(len(old_body) * 0.8)])
    check('缩到 80% → 通过', not rejected_c, why_c[:64])

    print()
    print('══ D. 边界：恰好一半放行，少 1 字节就拒 ══')
    line = '- 边界测试行 ' + 'x' * 20 + '\n'
    line_disk = W._disk_bytes(line)
    half_disk = set_core(line * 200) // 2          # 200 行 → 一半恰好 100 行
    rejected_d, why_d = replace_with(line * 100)
    check('恰好 50%%（%dB / %dB）→ 通过' % (line_disk * 100, half_disk),
          not rejected_d and line_disk * 100 == half_disk, why_d[:50])
    set_core(line * 200)                            # ⚠️ 复原：上一步已经把 core 换掉了
    below = line * 99 + line[:-2] + '\n'
    rejected_d2, why_d2 = replace_with(below)
    check('少 1 字节（%dB / %dB）→ 拒绝' % (W._disk_bytes(below), half_disk * 2),
          rejected_d2, why_d2[:50])

    print()
    print('══ E. 空源文件：单独判，必须拒（空 ≠ 内容少）══')
    set_core(old_body)
    rejected_e, why_e = replace_with('   \n\n')
    check('空白正文 → 拒绝', rejected_e, why_e[:64])

    print()
    print('══ F. core.md 不存在时不该被闸门误拦（首次写入）══')
    C.CORE.unlink()
    rejected_g, why_g = replace_with('- 首次写入的一行。\n')
    check('core 缺失 → 通过', not rejected_g, why_g[:64])

finally:
    for k, v in saved.items():
        setattr(C, k, v)
    shutil.rmtree(tmp, ignore_errors=True)

print()
print('结论: %s' % ('✓ 全部通过' if not fails else '✗ 失败项：' + '; '.join(fails)))
sys.exit(1 if fails else 0)
