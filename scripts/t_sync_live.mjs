// Real hook -> durable outbox -> GPU worker -> cloud injection, in a temporary home.
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, cpSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'
const repo = dirname(dirname(fileURLToPath(import.meta.url)))
const home = mkdtempSync(join(tmpdir(), 'dba-hook-live-'))
process.env.DBA_HOME = home
process.env.DBA_RWKV_PYTHON = join(repo, '.venv-rwkv', 'Scripts', 'python.exe')
try {
  cpSync(join(repo, 'scripts'), join(home, 'scripts'), { recursive: true })
  mkdirSync(join(home, 'memory/ws'), { recursive: true })
  writeFileSync(join(home, 'memory/core.md'), '# 测试核心\n')
  writeFileSync(join(home, 'memory/stream.md'), '# 测试流\n')
  const { createTracker } = await import('../integrations/dsh/snowfox-sync.mjs')
  const run = (script, args = [], input) => execFileSync('py', ['-3.14', '-B', join(home, 'scripts', script), ...args], {
    env: process.env, encoding: 'utf8', input, timeout: 300000,
  })
  const tracker = createTracker({
    snapshot: e => run('nb_turn_draft.py', ['snapshot'], JSON.stringify(e)),
    list: () => JSON.parse(run('nb_turn_draft.py', ['list'])),
    remove: e => run('nb_turn_draft.py', ['remove', e.session_id, e.event_id]),
    enqueue: e => run('nb_turn.py', ['enqueue', '--no-spawn'], JSON.stringify(Object.fromEntries(['session_id','event_id','user','assistant','at'].map(k => [k,e[k]])))),
    retry() {}, assembly() {}, error(e) { throw e }, uuid: () => 'live-event', now: () => '2026-10-06T12:00:00Z',
  })
  const session = { id: 'isolated-hook-live' }
  tracker.handle(session, { type: 'user/message', data: { source: { kind: 'user' }, content: '请记录本地机器人电机控制器使用蓝色齿轮测试台' } })
  tracker.handle(session, { type: 'assistant/message', data: { message: { content: '已收到蓝色齿轮测试台的电机控制器配置，尚未进行硬件验证。' } } })
  tracker.handle(session, { type: 'turn/end' })
  const states = JSON.parse(run('nb_turn.py', ['worker']))
  assert.equal(Object.values(states)[0].status, 'done')
  const recall = JSON.parse(run('nb_turn.py', ['recall', '蓝色齿轮测试台电机控制器']))
  assert.equal(recall.degraded, false)
  assert.match(recall.hits[0].event.user, /蓝色齿轮/)
  assert.match(recall.hits[0].event.assistant, /尚未/)
  run('nb_bridge.py', [], '蓝色齿轮测试台电机控制器')
  const hits = readFileSync(join(home, 'memory/_hits_context.md'), 'utf8')
  assert.match(hits, /双方对话记忆/)
  assert.match(hits, /用户：/)
  assert.match(hits, /助手：/)
  console.log(JSON.stringify({ passed: true, stage: 'real-hook-outbox-gpu-recall-cloud-injection', score: recall.hits[0].score, isolated: true }))
} finally { rmSync(home, { recursive: true, force: true }) }
