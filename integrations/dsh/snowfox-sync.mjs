/** Durable pending user+assistant snapshots; no restart or composition changes. */
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { randomUUID } from 'node:crypto'
export const name = 'snowfox-sync'
const DBA_HOME = process.env.DBA_HOME || 'D:\\AI\\DBA'
const SCRIPTS = join(DBA_HOME, 'scripts')
const MEMORY = join(DBA_HOME, 'memory')
function textOf(blocks) {
  if (typeof blocks === 'string') return blocks.trim()
  if (!Array.isArray(blocks)) return ''
  return blocks.filter(b => b?.type === 'text' && typeof b.text === 'string').map(b => b.text).join('\n').trim()
}
function python(script, args, data) {
  return execFileSync('py', ['-3.14', script, ...args], {
    cwd: SCRIPTS, input: data === undefined ? undefined : JSON.stringify(data),
    encoding: 'utf8', stdio: ['pipe', 'pipe', 'inherit'], timeout: 15000, windowsHide: true,
  })
}
function retryQueue() {
  // Queue recovery is handled manually to avoid creating a console host that can steal fullscreen focus.
}
const io = {
  snapshot: e => python('nb_turn_draft.py', ['snapshot'], e),
  list: () => JSON.parse(python('nb_turn_draft.py', ['list'])),
  remove: e => python('nb_turn_draft.py', ['remove', e.session_id, e.event_id]),
  enqueue: e => python('nb_turn.py', ['enqueue', '--no-spawn'], Object.fromEntries(['session_id','event_id','user','assistant','at'].map(k => [k, e[k]]))),
  retry: retryQueue,
  assembly(query) {
    try { execFileSync('py', ['-3.14', 'nb_bridge.py'], { cwd: SCRIPTS, input: query, stdio: ['pipe','ignore','inherit'], timeout: 60000, windowsHide: true }) }
    catch (e) { console.error('[snowfox-sync] assembly failed:', e.message) }
  },
  error: e => console.error('[snowfox-sync] draft failed:', e.message),
  uuid: randomUUID, now: () => new Date().toISOString(),
}
export function createTracker(storage = io) {
  const pending = new Map()
  // A successful enqueue is durable before draft deletion. Failed deletion is
  // harmless: the next recovery enqueues the identical immutable event again.
  function flush(event, stage) {
    const saved = { ...event, stage }
    storage.snapshot(saved)
    if (!saved.assistant.trim()) return false // Worker rejects empty assistant; retain the interrupted draft.
    // Derive the warning deterministically from stage. The snapshot keeps the
    // exact assistant content, so enqueue-success/remove-failure retries use
    // the identical payload without stacking warning prefixes.
    const payload = stage === 'interrupted'
      ? { ...saved, assistant: '[系统生命周期记录：本轮未正常结束，以下为已产生的部分输出，不代表任务完成]\n--- 部分输出开始 ---\n' + saved.assistant + '\n--- 部分输出结束 ---' }
      : saved
    storage.enqueue(payload)
    storage.remove(saved)
    return true
  }
  function recover() {
    try {
      for (const event of storage.list()) {
        if (event.error) { storage.error(new Error(event.error)); continue }
        try {
          if (!flush(event, event.stage === 'ready' ? 'ready' : 'interrupted')) {
            pending.set(event.session_id, { ...event, stage: 'interrupted' })
          }
        } catch (e) { pending.set(event.session_id, event); storage.error(e) }
      }
      storage.retry()
    } catch (e) { storage.error(e) }
  }
  function handle(session, event) {
    const id = session.id
    try {
      if (event?.type === 'user/message') {
        if (event.data?.source?.kind !== 'user') return
        const user = textOf(event.data.content)
        if (!user) return
        const previous = pending.get(id)
        if (previous) {
          // Persist the prior identity before starting a new one. A failed enqueue
          // may leave a draft on disk, but a failed snapshot blocks replacement.
          const priorStage = previous.stage === 'ready' ? 'ready' : 'interrupted'
          storage.snapshot({ ...previous, stage: priorStage })
          try { flush(previous, priorStage) } catch (e) { storage.error(e) }
        }
        const next = { session_id: id, event_id: storage.uuid(), user, assistant: '', at: storage.now(), stage: 'ongoing' }
        storage.snapshot(next) // Never replace memory before persistence succeeds.
        pending.set(id, next)
        storage.retry(); storage.assembly(user)
      } else if (event?.type === 'assistant/message') {
        const previous = pending.get(id)
        const text = textOf(event.data?.message?.content)
        if (!previous || !text) return
        if (previous.stage !== 'ongoing') {
          storage.error(new Error('late assistant message for a finalized draft; payload remains immutable'))
          return
        }
        const next = { ...previous, assistant: previous.assistant ? previous.assistant + '\n\n' + text : text }
        storage.snapshot(next)
        pending.set(id, next)
      } else if (event?.type === 'turn/end') {
        const previous = pending.get(id)
        if (!previous) return
        const stage = previous.stage === 'ready' || (previous.assistant.trim() && previous.stage === 'ongoing') ? 'ready' : 'interrupted'
        storage.snapshot({ ...previous, stage })
        pending.set(id, { ...previous, stage })
        if (flush(previous, stage)) pending.delete(id)
        storage.retry()
      }
    } catch (e) { storage.error(e) }
  }
  return { handle, recover, pending }
}
export function apply(ctx) {
  mkdirSync(join(MEMORY, '.turns'), { recursive: true })
  writeFileSync(join(MEMORY, '.turns', 'hook-runtime.json'), JSON.stringify({ version: 'joint-turn-draft-v3', pid: process.pid, loaded_at: new Date().toISOString() }), 'utf8')
  const tracker = createTracker()
  tracker.recover()
  ctx.on('session/event', tracker.handle)
  ctx.effect(() => () => tracker.pending.clear())
}
