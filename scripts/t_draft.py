import json, os, subprocess, tempfile, unittest
from pathlib import Path
ROOT = tempfile.TemporaryDirectory(); os.environ['DBA_HOME'] = ROOT.name
import nb_turn_draft as D

class T(unittest.TestCase):
 def setUp(self):
  import shutil
  self.mem=Path(ROOT.name)/'memory'
  if self.mem.exists(): shutil.rmtree(self.mem)
  self.mem.mkdir()
  self.assertEqual(D.C.MEMORY.resolve(), self.mem.resolve())
  self.assertEqual(D.items(), [])
  self.assertFalse((D.root()/'submitted').exists())
 def event(self, **kw):
  x={'session_id':'s','event_id':'e','user':'用户 token=secret123','assistant':'','at':'2026-10-06T12:00:00+08:00','stage':'ongoing'}; x.update(kw); return x
 def test_snapshot_sanitizes_only_leaves_and_roundtrips(self):
  k=D.snapshot(self.event()); saved=D.items()[0]; self.assertEqual(saved['event_id'],'e'); self.assertNotIn('secret123', saved['user']); self.assertEqual(k,D.key(saved)); self.assertEqual(saved['assistant'],'')
 def test_protocol_identity_not_sanitized(self):
  ident='a'*64; D.snapshot(self.event(event_id=ident,assistant='apikey=anothersecret123 response', extra='leak')); saved=D.items()[0]; self.assertEqual(saved['event_id'],ident); self.assertNotIn('anothersecret123',saved['assistant']); self.assertNotIn('extra',saved)
 def test_metadata_timestamp_is_typed(self):
  with self.assertRaises(D.TurnError): D.snapshot(self.event(updated_at='not-a-date'))
 def test_stage_interrupted_explicit(self):
  D.snapshot(self.event(stage='interrupted')); self.assertEqual(D.items()[0]['stage'],'interrupted')
 def test_identity_collision(self):
  D.snapshot(self.event());
  with self.assertRaises(D.TurnError): D.snapshot(self.event(user='different'))
 def test_corrupt_visible(self):
  D.snapshot(self.event()); p=next(D.root().glob('*.json')); p.write_text('{"bad":1}',encoding='utf8'); self.assertIn('error',D.items()[0])
 def test_remove_archives(self):
  e=self.event(); D.snapshot(e); D.remove('e','s'); self.assertTrue((D.root()/'submitted').glob('*.json').__iter__().__next__())
if __name__=='__main__': unittest.main()
