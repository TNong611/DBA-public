import assert from 'node:assert/strict'
import { createTracker } from '../integrations/dsh/snowfox-sync.mjs'
function harness() {
 const files=new Map(), submitted=[], errors=[]; let n=0
 const store={uuid:()=>`e${++n}`,now:()=> '2026-10-06T12:00:00Z', snapshot:e=>{if(store.diskFail)throw Error('snapshot failed');files.set(e.event_id,{...e})},list:()=>[...files.values()],enqueue:e=>{if(store.fail)throw Error('enqueue failed');submitted.push({...e})},remove:e=>{if(store.removeFail)throw Error('remove failed');files.delete(e.event_id)},retry(){},assembly(){},error:e=>errors.push(e.message)}
 return {store,files,submitted,errors}
}
const user=(text,source='user')=>({type:'user/message',data:{source:{kind:source},content:[{type:'text',text}]}})
const assistant=text=>({type:'assistant/message',data:{message:{content:[{type:'text',text}]}}})
const end={type:'turn/end'}, s={id:'s'}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('question'));t.handle(s,assistant('one'));t.handle(s,assistant('two'));t.handle(s,end);assert.equal(h.submitted[0].assistant,'one\n\ntwo');assert.equal(h.files.size,0);t.handle(s,end);assert.equal(h.submitted.length,1)
}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('empty'));t.handle(s,end);assert.equal(h.submitted.length,0);assert.equal([...h.files.values()][0].stage,'interrupted')
}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('prior'));t.handle(s,assistant('partial'));h.store.fail=true;t.handle(s,user('next'));assert(h.files.has('e1'));assert(h.files.has('e2'));h.store.fail=false;createTracker(h.store).recover();assert.equal(h.submitted.length,1);assert.equal(h.submitted[0].stage,'interrupted');assert(h.files.has('e2'))
}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('prior'));t.handle(s,assistant('partial'));h.store.diskFail=true;t.handle(s,user('next'));assert.equal(t.pending.get('s').event_id,'e1');assert.equal(h.files.size,1)
}
{
 const h=harness(),t=createTracker(h.store);h.store.removeFail=true;t.handle(s,user('cut'));t.handle(s,assistant('partial'));t.handle(s,end);assert.equal(h.submitted.length,1);h.store.removeFail=false;createTracker(h.store).recover();assert.equal(h.submitted.length,2);assert.equal(h.submitted[0].event_id,h.submitted[1].event_id);assert.equal(h.submitted[0].assistant,h.submitted[1].assistant)
}
{
 const h=harness(),t=createTracker(h.store);h.store.removeFail=true;t.handle(s,user('ready'));t.handle(s,assistant('complete'));t.handle(s,end);const first=h.submitted[0];assert.equal(first.stage,'ready');t.handle(s,user('next'));assert.equal(t.pending.get('s').event_id,'e2');assert.equal(h.submitted.length,2);assert.equal(h.submitted[1].event_id,'e1');assert.equal(h.submitted[1].stage,'ready');assert.equal(h.submitted[1].assistant,first.assistant);assert.equal(h.files.get('e1').stage,'ready')
}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('ignore','plugin'));assert.equal(h.files.size,0);t.handle(s,user('recover'));t.handle(s,assistant('answer'));createTracker(h.store).recover();assert.equal(h.submitted[0].stage,'interrupted');assert.equal(h.files.size,0)
}
{
 const h=harness(),t=createTracker(h.store);t.handle(s,user('q'));t.handle(s,assistant('exact partial\ncontent'));h.store.removeFail=true;t.recover();const first=h.submitted[0];assert(first.assistant.startsWith('[系统生命周期记录：本轮未正常结束，以下为已产生的部分输出，不代表任务完成]'));assert.equal(h.files.get('e1').assistant,'exact partial\ncontent');assert(first.assistant.includes('--- 部分输出开始 ---\nexact partial\ncontent\n--- 部分输出结束 ---'));h.store.removeFail=false;createTracker(h.store).recover();assert.equal(h.submitted[1].assistant,first.assistant);assert.equal(h.files.size,0)
}
console.log('PASS: eight lifecycle groups including ready retry after next user and interrupted remove-failure idempotency')
