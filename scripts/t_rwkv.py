"""Pure Python transaction tests; no torch, weights, GPU or real memory."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import nb_rwkv as R


class FakeEngine:
    identity = {"sha256": "fake-weight-identity", "architecture": "RWKV-7"}

    def __init__(self):
        self.observations = []
        self.start_values = []
        self.fail_summary = False

    def assert_gpu(self, state):
        pass  # Fake engine explicitly models state mechanics only.

    def observe(self, state, text):
        self.start_values.append(list(state or []))
        self.observations.append(text)
        state = list(state or [])
        state.append(text)
        return state, len(text)

    def clone(self, state):
        return copy.deepcopy(state)

    def summarize(self, branch):
        branch.append("GENERATED BRANCH CONTAMINATION")
        return "invalid\nsummary" if self.fail_summary else "用户请求修复索引；助手提出方案，尚未执行"

    def to_cpu(self, state):
        return copy.deepcopy(state)

    def to_gpu(self, state):
        return copy.deepcopy(state)

    def save(self, path, checkpoint):
        Path(path).write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")

    def load(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = FakeEngine()
        self.runner = R.Runner(self.engine, self.tmp.name)

    def event(self, **changes):
        e = {"session_id": "s", "event_id": "e1", "user": "请求修复索引", "assistant": "提出方案，尚未执行", "at": "2026-10-06T12:00:00Z"}
        e.update(changes)
        return e

    def test_session_continuity_and_branch_isolation(self):
        a = self.runner.process(self.event())
        b = self.runner.process(self.event(event_id="e2", parent_revision=1))
        self.assertEqual((a["revision"], b["revision"]), (1, 2))
        self.assertEqual(self.engine.start_values[1], [self.engine.observations[0]])
        current = self.runner.latest(self.runner.session_dir("s"), "s")
        self.assertEqual(current["source_state"], self.engine.observations)
        self.assertNotIn("GENERATED BRANCH CONTAMINATION", current["source_state"])

    def test_new_session_separate(self):
        self.runner.process(self.event())
        b = self.runner.process(self.event(session_id="other"))
        self.assertEqual(b["revision"], 1)
        self.assertEqual(self.engine.start_values[1], [])

    def test_retry_in_fresh_runner_exactly_once(self):
        first = self.runner.process(self.event())
        self.runner.process(self.event(event_id="e2"))
        engine = FakeEngine()
        cached = R.Runner(engine, self.tmp.name).process(self.event(parent_revision=0))
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["revision"], first["revision"])
        self.assertEqual(engine.observations, [])

    def test_conflicting_event_id(self):
        self.runner.process(self.event())
        with self.assertRaisesRegex(R.RWKVError, "different sanitized content"):
            self.runner.process(self.event(assistant="实际完成了"))
        self.assertEqual(len(self.engine.observations), 1)

    def test_parent_revision_rejects_out_of_order(self):
        self.runner.process(self.event())
        with self.assertRaisesRegex(R.RWKVError, "parent revision conflict"):
            self.runner.process(self.event(event_id="e2", parent_revision=0))
        self.assertEqual(len(self.engine.observations), 1)

    def test_snapshot_crash_preserves_previous_and_retry(self):
        self.runner.process(self.event())
        def crash(tmp, dst):
            self.assertTrue(tmp.exists())
            raise RuntimeError("injected crash")
        with self.assertRaisesRegex(RuntimeError, "injected crash"):
            R.Runner(self.engine, self.tmp.name, after_save=crash).process(self.event(event_id="e2"))
        checkpoint = self.runner.latest(self.runner.session_dir("s"), "s")
        self.assertEqual(checkpoint["revision"], 1)
        self.assertEqual(list(self.runner.root.rglob("*.tmp")), [])
        fresh = R.Runner(FakeEngine(), self.tmp.name)
        result = fresh.process(self.event(event_id="e2"))
        self.assertEqual(result["revision"], 2)
        self.assertEqual(fresh.latest(fresh.session_dir("s"), "s")["source_state"], self.engine.observations[:2])

    def test_summary_failure_explicit_and_source_committed_once(self):
        self.engine.fail_summary = True
        result = self.runner.process(self.event())
        self.assertFalse(result["ok"])
        self.assertTrue(result["source_committed"])
        self.assertIsNone(result["summary"])
        self.assertIn("single line", result["summary_error"])
        self.assertTrue(self.runner.process(self.event())["cached"])
        self.assertEqual(len(self.engine.observations), 1)

    def test_model_identity_change_refused(self):
        self.runner.process(self.event())
        changed = FakeEngine()
        changed.identity = {"sha256": "different"}
        with self.assertRaisesRegex(R.RWKVError, "model identity mismatch"):
            R.Runner(changed, self.tmp.name).process(self.event(event_id="e2"))

    def test_sanitized_text_only_and_protocol_preserved(self):
        event = self.event(event_id="a" * 64, user="用户 token=supersecret123")
        result = self.runner.process(event)
        self.assertEqual(result["event_id"], "a" * 64)
        self.assertNotIn("supersecret123", self.engine.observations[0])
        self.assertIn("a" * 64, self.engine.observations[0])

    def test_summary_fact_guard_rejects_unperformed_action(self):
        text = "助手进行了硬件验证，但未能成功。"
        with self.assertRaisesRegex(R.RWKVError, "contradicts"):
            R.validate_summary(text, {"user": "请求验证", "assistant": "尚未进行硬件验证。"})

    def test_output_file_atomic_json(self):
        path = Path(self.tmp.name) / "result.json"
        R.output_json(path, {"ok": True})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
