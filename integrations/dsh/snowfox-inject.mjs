/**
 * snowfox-inject — DBA 三层记忆 + 向量层 → DSH 系统提示注入插件。
 *
 * 背景：记忆已迁到 **DBA（双脑架构）** 仓库 `D:\AI\DBA`，2026-10-06 起不再放
 * hermes 下（记忆是可移植资产，不该埋在某个客户端的安装目录里）。
 * 组装由 `D:\AI\DBA\scripts\nb_assemble.py` 产出，本插件只负责注入系统提示。
 *
 * ⚠️ 分两个 section 注入，因为两段的「变化频率」不同：
 *   `_core_context.md` 核心记忆，几天才变 → 排在 prompt 前部（order -50），
 *                      稳定内容在前，DeepSeek 前缀缓存才能命中。
 *   `_hits_context.md` 本轮检索命中，每轮随用户输入变 → 排在**最末尾**
 *                      （order 200，在工具定义 150 之后）。
 *   两者若合成一个 section 注入，一旦检索命中，system prompt 从该 section
 *   起整段变化，其后所有段落的缓存全部失效。
 *   （用户先前实测：变化内容进前缀 → 命中率 0%；稳定在前、变化在后 → 92.7%。）
 *
 * 降级：文件不存在 / 读取失败 / 超限时返回空 section，绝不影响会话提交。
 * 容量：各段另有 64KB 硬上限，防异常膨胀挤爆上下文。
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

export const name = 'snowfox-inject'

/** 使用 ctx.systemPrompt 服务必须声明 inject 依赖（cordis 约定）。 */
export const inject = ['systemPrompt']

/**
 * 记忆仓库根。允许用 DBA_HOME 覆盖，便于同一份插件跑在多台机器/多份仓库上
 * （仓库是设计成可整体拷走的，路径不该硬编码进插件）。
 */
const DBA_HOME = process.env.DBA_HOME
  || (process.platform === 'win32' ? 'D:\\AI\\DBA' : join(process.env.HOME || '.', 'repos', 'DBA'))
/** 组装产物落在 memory/ 下（与记忆数据同级，便于整体搬迁）。 */
const MEMORY_DIR = join(DBA_HOME, 'memory')
const CORE_FILE = join(MEMORY_DIR, '_core_context.md')
const HITS_FILE = join(MEMORY_DIR, '_hits_context.md')

/** 硬上限：超过即丢弃，避免异常组装产物挤爆上下文。 */
const MAX_BYTES = 64 * 1024

/**
 * 组装产物首行的时间戳注释——每次重建都变，必须剥离，否则 prompt 前缀
 * 每轮都不同，DeepSeek 前缀缓存全部失效（实测命中率会被拖到 0）。
 *
 * 三种前缀都要认（历史遗留 + 新格式）：
 *   旧：`<!-- SnowFox Memory Assembly | built: ... -->`
 *   中：`<!-- Notebook Memory | built: ... | core=... -->`
 *   新：`<!-- Notebook core | built: ... | ...KB/15KB -->`
 * 曾只匹配旧前缀 → 迁移后时间戳留在正文里 → 缓存全失效（2026-10-05）。
 */
const BUILT_LINE_RE = /^<!--\s*(?:SnowFox Memory Assembly|Notebook Memory|Notebook core)\b[^\n]*-->[ \t]*\r?\n/

/**
 * 会话级快照缓存：同一会话内核心记忆只读一次并固定快照，prompt 前缀
 * 保持稳定 → DeepSeek 前缀缓存可命中（命中率从 53% 回到 90%+）。
 *
 * 刷新时机：`session/created` 事件（新会话）清空缓存，下次 assembly 重读。
 * 会话中途重建核心记忆不影响已固定的前缀。
 *
 * ⚠️ 检索命中段（hits）**不做会话级缓存**：它本来就该随本轮输入变化，
 * 缓存住了就等于永远检索不到新东西。它排在末尾，变化也不破坏前缀。
 *
 * ⚠️ 此处不注册「字面量变量」来放过正文里的 {{char}} 之类占位符。
 * 那样做等于**永久关闭 DSH 的未知变量校验器**，任何一次真正的拼错变量
 * 都会被静默放过（2026-10-05 事故的修法）。现在改为在生成端
 * （nb_assemble.escape_braces）把 `{{` 打断为 `{\u200b{`，
 * 校验器保持有效，正文原样可读。
 */
let cachedCore = null

function readSection(file, cache) {
  try {
    const raw = readFileSync(file, 'utf8')
    const text = raw
      .replace(/^\uFEFF/, '')
      .replace(BUILT_LINE_RE, '')
      .trim()
    if (!text) return ''
    if (Buffer.byteLength(text, 'utf8') > MAX_BYTES) return ''
    // 兜底：写入端（nb_assemble.escape_braces）万一漏掉，这里再打断一次，
    // 保证 DSH 的未知变量校验器永远不会被记忆正文里的占位符触发（2026-10-05 事故）。
    const safe = text.includes('{{') ? text.replace(/\{/g, '{\u200b') : text
    if (cache) cachedCore = safe
    return safe
  } catch {
    return ''
  }
}

function loadCore() {
  if (cachedCore !== null) return cachedCore
  return readSection(CORE_FILE, true)
}

function loadHits() {
  // 不缓存：每轮都要读最新检索结果
  return readSection(HITS_FILE, false)
}

export function apply(ctx) {
  // 新会话创建时清缓存，下次 prompt assembly 读最新核心记忆。
  ctx.on('session/created', () => {
    cachedCore = null
  })
  // 稳定段：核心记忆（全量）。harness identity 为 -100、persona 为 0；
  // -50 使记忆排在 persona 之前。
  ctx.systemPrompt.section({
    name: 'snowfox:memory',
    order: -50,
    text: () => loadCore(),
  })
  // 变化段：本轮检索命中。排在最后（工具定义 order 150 之后），
  // 保证它变化时不会让前面的内容失去缓存。
  ctx.systemPrompt.section({
    name: 'snowfox:retrieval',
    order: 200,
    text: () => loadHits(),
  })
}
