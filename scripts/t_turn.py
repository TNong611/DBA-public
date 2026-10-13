"""Isolated tests: temporary DBA_HOME, fake HTTP/writer; no live memory/model."""
import os
import tempfile

_TEMP = tempfile.TemporaryDirectory()
os.environ["DBA_HOME"] = _TEMP.name

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import nb_turn as T
import nb_config as C


class FakeOllama:
    def __init__(self):
        self.calls = 0
        self.seen = []
        self.fail = False
        self.model = "qwen3:8b"
        self.event_model = "bge-m3"

    def summary(self, event):
        self.calls += 1
        if self.fail:
            raise T.TurnError("model unavailable")
        self.seen.append(event)
        return "用户请求修复索引；助手提出方案，尚未执行"

    def require_gpu(self, model):
        return None

    def embed(self, texts, model):
        return [[float(len(text) or 1), 1.0] for text in texts]


class FakeBackend:
    def __init__(self, stream):
        self.stream = stream
        self.calls = {"append": 0, "kw": 0, "vec": 0}
        self.fail_kw = False
        self.fail_vec = False
        self.crash = False

    def append(self, text, at, marker):
        if T.nb_write._stream_junk_reason(text):
            raise T.nb_write.WriteError("gate rejected")
        self.calls["append"] += 1
        with self.stream.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"- {at} | tag={marker} | {text}\n")
        if self.crash:
            self.crash = False
            raise RuntimeError("crash after append")

    def kw(self):
        self.calls["kw"] += 1
        if self.fail_kw:
            raise RuntimeError("kw unavailable")

    def vec(self):
        self.calls["vec"] += 1
        if self.fail_vec:
            raise RuntimeError("vec GPU unconfirmed")


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TEMP.name)
        self.memory = Path(self.tmp.name) / "memory"
        self.memory.mkdir()
        (self.memory / "core.md").write_text("# core\n", encoding="utf-8")
        self.stream = self.memory / "stream.md"
        self.stream.write_text("# 记忆流\n", encoding="utf-8")
        self.o = FakeOllama()
        self.b = FakeBackend(self.stream)
        self.p = T.Pipeline(self.memory, self.o, self.b)
        # Real CLI backend references global paths. Point each at this sandbox.
        self.config_patches = [patch.object(C, name, value) for name, value in {
            "ROOT": self.memory.parent, "MEMORY": self.memory, "CORE": self.memory / "core.md",
            "STREAM": self.stream, "WS_DIR": self.memory / "ws", "INDEX": self.memory / ".index" / "inverted.json",
            "VEC_DIR": self.memory / ".vec", "VEC_INDEX": self.memory / ".vec" / "vectors.npy",
            "VEC_META": self.memory / ".vec" / "meta.json"}.items()]
        for p in self.config_patches:
            p.start()

    def tearDown(self):
        for p in self.config_patches:
            p.stop()
        self.tmp.cleanup()

    def event(self, **changes):
        event = {"session_id": "s1", "event_id": "e1", "user": "token=supersecret123 用户请求修复索引",
                 "assistant": "助手提出方案，尚未执行", "at": "2026-10-06T12:00:00+08:00"}
        event.update(changes)
        return event

    def test_sanitize_identity_conflict_and_raw_preservation(self):
        key = self.p.enqueue(self.event())
        self.assertEqual(key, self.p.enqueue(self.event()))
        stored = (self.p.events / (key + ".json")).read_text(encoding="utf-8")
        self.assertNotIn("supersecret123", stored)
        self.assertIn("用户请求修复索引", stored)
        self.assertIn("助手提出方案", stored)
        with self.assertRaises(T.TurnError):
            self.p.enqueue(self.event(user="different"))
        self.assertFalse(self.p.state_path(key).exists())

    def test_crash_after_append_recovers_without_duplicate_line(self):
        key = self.p.enqueue(self.event())
        self.b.crash = True
        self.assertEqual(self.p.drain()[key]["status"], "failed")
        self.assertEqual(self.p.drain()[key]["status"], "done")
        self.assertEqual(self.b.calls["append"], 1)
        self.assertEqual(self.o.calls, 1)
        self.assertEqual(self.stream.read_text(encoding="utf-8").count("tag=turn:"), 1)
        self.assertTrue((self.p.vectors / (key + ".json")).exists())
        self.assertEqual(self.p.drain()[key]["status"], "done")

    def test_independent_index_failure_retries_only_failed_channel(self):
        key = self.p.enqueue(self.event())
        self.b.fail_kw = True
        result = self.p.drain()[key]
        self.assertTrue(result["committed"])
        self.assertEqual(result["status"], "index_failed")
        self.assertEqual(result["indexes"]["vec"]["status"], "done")
        self.b.fail_kw = False
        self.assertEqual(self.p.drain()[key]["status"], "done")
        self.assertEqual(self.b.calls, {"append": 1, "kw": 2, "vec": 1})
        self.assertTrue((self.p.vectors / (key + ".json")).is_file())
        recalled = self.p.recall("修复索引")
        self.assertEqual(recalled["hits"][0]["event"]["event_id"], "e1")

    def test_explicit_durable_input_limit_remains_enforced(self):
        with patch.dict(os.environ, {"DBA_TURN_MAX_CHARS": "4"}):
            with self.assertRaises(T.TurnError):
                self.p.enqueue(self.event(user="12345"))
        self.assertFalse(self.p.root.exists())

    def test_protocol_identity_is_not_sanitized(self):
        event = self.event()
        event["event_id"] = "a" * 64
        event["session_id"] = "b" * 64
        key = self.p.enqueue(event)
        record = json.loads((self.p.events / (key + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(record["event"]["event_id"], event["event_id"])
        self.assertEqual(record["event"]["session_id"], event["session_id"])

    def test_long_turn_enqueues_without_ordinary_turn_cap(self):
        event = self.event(user="用户内容" * 4000, assistant="助手内容" * 3500)
        key = self.p.enqueue(event)
        stored = json.loads((self.p.events / (key + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(stored["event"]["user"], event["user"])
        self.assertEqual(stored["event"]["assistant"], event["assistant"])

    def test_hierarchical_summary_consumes_all_source_chunks_and_bounds_requests(self):
        o = T.Ollama()
        observed, endpoints = [], []
        def request(endpoint, payload=None):
            endpoints.append((endpoint, payload))
            if endpoint == "/api/ps":
                return {"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 100}]}
            if endpoint == "/api/chat":
                content = payload["messages"][1]["content"]
                self.assertLessEqual(len(T.canonical(payload).encode("utf-8")), 2000)
                self.assertEqual(payload["options"]["num_ctx"], T.LOCAL_NUM_CTX)
                data = json.loads(content)
                if data["stage"] == "chunk":
                    observed.append((data["speaker"], data["offset"], data["text"]))
                    summary = ("用户片段" if data["speaker"] == "user" else "助手片段") + str(data["offset"])
                else:
                    summary = "用户请求与助手回应按顺序保留；尚未验证"
                return {"message": {"content": json.dumps({"summary": summary}, ensure_ascii=False)}}
            return {}
        event = self.event(user="用户甲\n" * 5000, assistant="助手乙\n" * 4500)
        with patch.object(o, "request", side_effect=request):
            result = o.summary(event)
        self.assertTrue(result)
        for speaker in ("user", "assistant"):
            pieces = sorted((offset, text) for who, offset, text in observed if who == speaker)
            reconstructed = "".join(text for _, text in pieces)
            self.assertEqual(reconstructed, event[speaker])
            self.assertEqual(pieces[0][0], 0)
            self.assertEqual(pieces[-1][0] + len(pieces[-1][1]), len(event[speaker]))
        self.assertTrue(any(json.loads(p["messages"][1]["content"]).get("stage") == "reduce"
                            for ep, p in endpoints if ep == "/api/chat"))

    def test_model_failure_preserves_event_and_retries(self):
        key = self.p.enqueue(self.event())
        self.o.fail = True
        self.assertEqual(self.p.drain()[key]["status"], "failed")
        self.assertEqual(self.b.calls["append"], 0)
        self.o.fail = False
        self.assertEqual(self.p.drain()[key]["status"], "done")

    def test_partial_line_fails_closed(self):
        key = self.p.enqueue(self.event())
        with patch.object(self.b, "append", side_effect=RuntimeError("before append")):
            self.p.process(key)
        with self.stream.open("a", encoding="utf-8") as f:
            f.write("- broken tag=turn:" + key)
        result = self.p.drain()[key]
        self.assertFalse(result["committed"])
        self.assertIn("ambiguous", result["error"])

    def test_validation_before_disk(self):
        for change in ({"at": "2026-13-45"}, {"user": ""}, {"event_id": None}):
            with self.assertRaises(T.TurnError):
                self.p.enqueue(self.event(**change))
        self.assertFalse(self.p.root.exists())

    def test_lock_cross_process_and_release(self):
        path = self.p.root / "worker.lock"
        script = ("import sys; from pathlib import Path; import nb_turn as T; "
                  "\ntry:\n with T.file_lock(Path(sys.argv[1])): pass"
                  "\nexcept T.LockBusy: sys.exit(7)")
        with T.file_lock(path):
            r = subprocess.run([sys.executable, "-c", script, str(path)], cwd=Path(T.__file__).parent)
            self.assertEqual(r.returncode, 7)
        r = subprocess.run([sys.executable, "-c", script, str(path)], cwd=Path(T.__file__).parent)
        self.assertEqual(r.returncode, 0)

    def test_gpu_switch_unloads_qwen_before_loading_bge(self):
        o = T.Ollama()
        calls = []
        loaded = {"qwen3:8b"}
        def request(endpoint, payload=None):
            calls.append((endpoint, payload))
            if endpoint == "/api/ps":
                return {"models": [{"name": name if ":" in name else name + ":latest",
                                    "size": 100, "size_vram": 100} for name in loaded]}
            if endpoint == "/api/generate":
                if payload.get("keep_alive") == 0:
                    loaded.discard(payload["model"])
                else:
                    loaded.add(payload["model"])
                return {}
            if endpoint == "/api/embed" and payload.get("keep_alive") == 0:
                loaded.discard(payload["model"])
                return {}
            if endpoint == "/api/embed":
                loaded.add(payload["model"])
                return {"embeddings": [[1.0, 0.0] for _ in payload["input"]]}
            return {}
        with patch.object(o, "request", side_effect=request):
            o.embed(["测试"], "bge-m3")
            o.ensure_gpu_model(o.model)
        generations = [payload for endpoint, payload in calls if endpoint == "/api/generate"]
        embeds = [payload for endpoint, payload in calls if endpoint == "/api/embed"]
        self.assertEqual(generations[0]["model"], "qwen3:8b")
        self.assertEqual(generations[0]["keep_alive"], 0)
        self.assertEqual(embeds[0]["model"], "bge-m3")
        self.assertEqual(embeds[0]["input"], [""])
        self.assertIn("qwen3:8b", loaded)
        embed_unloads = [p for ep, p in calls
                         if ep == "/api/embed" and p.get("keep_alive") == 0
                         and p.get("input") == []]
        self.assertEqual(len(embed_unloads), 1)
        self.assertEqual(embed_unloads[0]["model"], "bge-m3")
        self.assertTrue(all(p["model"] != "bge-m3" for p in generations))

    def test_gpu_requires_exact_full_vram(self):
        o = T.Ollama()
        cases = [[], [{"name": "other:8b", "size": 100, "size_vram": 100}],
                 [{"name": "qwen3:8b", "size": 100000, "size_vram": 99999}],
                 [{"name": "qwen3:8b", "size": 0, "size_vram": 100}]]
        for models in cases:
            with patch.object(o, "request", return_value={"models": models}):
                with self.assertRaises(T.TurnError):
                    o.require_gpu("qwen3:8b")
        with patch.object(o, "request", return_value={"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 100}]}):
            o.require_gpu("qwen3:8b")

    def test_structured_chat_sources_prompt_and_output_limit(self):
        o = T.Ollama()
        calls = []
        def request(endpoint, payload=None):
            calls.append((endpoint, payload))
            if endpoint == "/api/ps":
                return {"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 100}]}
            if endpoint == "/api/chat":
                return {"message": {"content": json.dumps({"summary": "用户请求修复；助手提出方案，尚未执行"}, ensure_ascii=False)}}
            return {}
        with patch.object(o, "request", side_effect=request):
            self.assertIn("尚未执行", o.summary(self.event()))
        chat = next(payload for endpoint, payload in calls if endpoint == "/api/chat")
        self.assertEqual(chat["format"]["properties"]["summary"]["maxLength"], 90)
        self.assertEqual(chat["options"]["num_gpu"], 999)
        self.assertIn("用户", chat["messages"][1]["content"])
        self.assertIn("助手", chat["messages"][1]["content"])
        for content in (json.dumps({"summary": "x" * 91}), "not JSON", json.dumps({"summary": "ok", "extra": 1})):
            with patch.object(o, "ensure_gpu_model"), patch.object(o, "require_gpu"), patch.object(o, "request", return_value={"message": {"content": content}}):
                with self.assertRaises((T.TurnError, json.JSONDecodeError)):
                    o.summary(self.event())

    def test_cpu_refuses_stream_commit(self):
        o = T.Ollama()
        def request(endpoint, payload=None):
            if endpoint == "/api/ps":
                return {"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 0}]}
            return {}
        self.p.ollama = o
        key = self.p.enqueue(self.event())
        with patch.object(o, "request", side_effect=request):
            result = self.p.drain()[key]
        self.assertFalse(result["committed"])
        self.assertEqual(self.b.calls["append"], 0)

    def test_real_append_stream_is_scoped_and_gated(self):
        backend = T.Backend(self.o)
        before = self.stream.read_text(encoding="utf-8")
        with self.assertRaises(T.nb_write.WriteError):
            backend.append("Now let me continue", "2026-10-06T12:00:00", "turn:test")
        self.assertEqual(self.stream.read_text(encoding="utf-8"), before)
        backend.append("已修复", "2026-10-06T12:00:00", "turn:test")
        self.assertIn("tag=turn:test", self.stream.read_text(encoding="utf-8"))

    def test_outbox_excluded_from_collect_docs(self):
        import nb_index
        self.p.enqueue(self.event())
        docs = nb_index.collect_docs()
        self.assertFalse(any(".turns" in d["file"] for d in docs))


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _TEMP.cleanup()
