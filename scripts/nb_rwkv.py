"""GPU-only RWKV-7 recurrent observations with isolated summary branches.

CLI: process --output FILE < event.json | probe --output FILE
Run with .venv-rwkv/Scripts/python.exe. Never writes stream/core/ws.
A per-session lock serializes complete observation+checkpoint transactions.
Immutable revision files contain CPU state, model identity, receipts and summary
in ONE torch.save. Latest revision is the commit point; no separate head pointer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

import nb_config as C
import nb_migrate
import nb_write
from nb_turn import file_lock

DEFAULT_MODEL = r"D:\AI\models\rwkv7\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
FORMAT = 1


class RWKVError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise RWKVError("RWKV weight file missing: " + str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RWKVError("model changed while fingerprinting")
    return {"sha256": h.hexdigest(), "bytes": after.st_size,
            "architecture": "RWKV-7", "strategy": "cuda:0 bf16",
            "tokenizer": "rwkv_vocab_v20230424", "rwkv_package": "0.8.32"}


def clean_event(raw):
    required = {"session_id", "event_id", "user", "assistant", "at"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {"parent_revision"}:
        raise RWKVError("event requires session_id,event_id,user,assistant,at; optional parent_revision")
    if any(not isinstance(raw[k], str) or not raw[k].strip() for k in required):
        raise RWKVError("event fields must be nonempty strings")
    if len(raw["session_id"]) > 512 or len(raw["event_id"]) > 512:
        raise RWKVError("identity too long")
    try:
        datetime.fromisoformat(raw["at"])
    except ValueError as exc:
        raise RWKVError("at must be an ISO timestamp") from exc
    if "parent_revision" in raw and (type(raw["parent_revision"]) is not int or raw["parent_revision"] < 0):
        raise RWKVError("parent_revision must be a nonnegative integer")
    event = {k: raw[k] for k in required}
    for k in ("user", "assistant"):
        event[k] = nb_migrate.sanitize(event[k])
    return event


def validate_summary(text, event=None):
    if not isinstance(text, str):
        raise RWKVError("summary is not text")
    text = nb_migrate.sanitize(text).strip()
    if not text or len(text) > 90 or any(c in text for c in "\r\n\t"):
        raise RWKVError("summary must be a nonempty single line <=90 characters")
    if not re.search(r"[\u3400-\u9fff]", text):
        raise RWKVError("summary must contain Chinese")
    if event is not None:
        # Narrow, conservative hard-fact guard. This is NOT a semantic verifier.
        # Reject known completion reversals and provenance hallucinations.
        source = event["user"] + "\n" + event["assistant"]
        cleaned = re.sub(r"(?:尚未|未能|未曾|没有|未|待|尚待)[^，。；！？\n]{0,12}", "", text)
        for action in ("验证", "执行", "测试", "部署", "完成", "安装", "修复"):
            if re.search(r"(?:尚未|未曾|没有|未)[^，。；！？\n]{0,10}" + action, source):
                if re.search(r"(?:已|进行|完成|成功)[^，。；！？\n]{0,10}" + action, cleaned):
                    raise RWKVError("summary contradicts an explicit unperformed action")
        if "记忆正文" in text and "记忆正文" not in source:
            raise RWKVError("summary invented memory-output provenance")
    reason = nb_write._stream_junk_reason(text)
    if reason:
        raise RWKVError("summary gate rejected: " + str(reason))
    return text


class TorchEngine:
    """Imports heavy dependencies only when actual GPU inference is requested."""
    def __init__(self, model_path=None, chunk_tokens=64, generation_tokens=192):
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "0"  # torch CUDA tensors; no custom compiler
        import torch
        from importlib.metadata import version
        if version("rwkv") != "0.8.32":
            raise RWKVError("this adapter was validated for rwkv==0.8.32")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RWKVError("CUDA with BF16 support is required; CPU inference forbidden")
        self.torch = torch
        self.path = Path(model_path or os.environ.get("DBA_RWKV_MODEL", DEFAULT_MODEL)).resolve()
        self.identity = fingerprint(self.path)
        stat = self.path.stat()
        self.model_stat = (stat.st_size, stat.st_mtime_ns)
        if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 512:
            raise RWKVError("chunk_tokens must be 1..512")
        self.chunk_tokens = chunk_tokens
        self.generation_tokens = generation_tokens
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE
        # RWKV-7 __init__ appends .pth itself; do not pass the extension twice.
        self.model = RWKV(model=str(self.path)[:-4] if self.path.suffix == ".pth" else str(self.path),
                          strategy="cuda:0 bf16")
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.assert_gpu(None)

    def assert_model_unchanged(self):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.model_stat:
            raise RWKVError("RWKV weight changed during active session; reload before continuing")

    def assert_gpu(self, state):
        self.assert_model_unchanged()
        t = self.torch
        weights = getattr(self.model, "z", None)
        if not isinstance(weights, dict) or not weights:
            raise RWKVError("RWKV-7 model.z tensor dictionary missing")
        for name, value in weights.items():
            if not isinstance(value, t.Tensor) or value.device.type != "cuda" or value.device.index != 0:
                raise RWKVError("non-GPU model tensor: " + name)
            if value.dtype != t.bfloat16:
                raise RWKVError("model tensor is not BF16: " + name)
        if state is not None:
            if not isinstance(state, list) or len(state) != self.model.args.n_layer * 3:
                raise RWKVError("invalid recurrent state tensor count")
            for index, value in enumerate(state):
                if not isinstance(value, t.Tensor) or value.device.type != "cuda" or value.device.index != 0:
                    raise RWKVError("non-GPU recurrent state")
                expected_shape = ((self.model.args.n_embd // self.model.args.head_size,
                                   self.model.args.head_size, self.model.args.head_size)
                                  if index % 3 == 1 else (self.model.args.n_embd,))
                expected_dtype = t.float32 if index % 3 == 1 else t.bfloat16
                if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
                    raise RWKVError("recurrent state shape/dtype mismatch")
                if not t.isfinite(value).all().item():
                    raise RWKVError("nonfinite recurrent state")

    def clone(self, state):
        self.assert_gpu(state)
        return None if state is None else [v.detach().clone() for v in state]

    def observe(self, state, text):
        self.assert_gpu(state)
        tokens = self.pipeline.encode(text)
        if not tokens:
            raise RWKVError("empty tokenized observation")
        with self.torch.inference_mode():
            for i in range(0, len(tokens), self.chunk_tokens):
                logits, state = self.model.forward(tokens[i:i + self.chunk_tokens], state)
                self.assert_gpu(state)
        return state, len(tokens)

    def summarize(self, branch):
        # Prompt and generated tokens live only in a CLONED source-state branch.
        prompt = ("\nUser: 根据刚刚观察的用户与助手事件，用中文一行概括双方事实和结果，最多90字。"
                  "请求、方案、失败和未验证不得说成已完成。只输出记忆正文，不要解释。\nAssistant:")
        tokens = self.pipeline.encode(prompt)
        state = branch
        generated = []
        with self.torch.inference_mode():
            for i in range(0, len(tokens), self.chunk_tokens):
                logits, state = self.model.forward(tokens[i:i + self.chunk_tokens], state)
                self.assert_gpu(state)
            for _ in range(self.generation_tokens):
                if logits.device.type != "cuda" or not self.torch.isfinite(logits).all().item():
                    raise RWKVError("invalid GPU generation logits")
                token = int(self.torch.argmax(logits).item())
                if token == 0:
                    break
                generated.append(token)
                text = self.pipeline.decode(generated)
                if "\n" in text or "\r" in text:
                    return text.splitlines()[0]
                # Fail on oversize instead of truncating an unfinished assertion.
                if len(text) > 90 and "\ufffd" not in text:
                    raise RWKVError("generated summary exceeded 90 characters")
                logits, state = self.model.forward([token], state)
                self.assert_gpu(state)
        text = self.pipeline.decode(generated)
        if "\ufffd" in text:
            raise RWKVError("incomplete UTF-8 generation")
        return text

    def to_cpu(self, state):
        self.assert_gpu(state)
        return [v.detach().cpu().clone() for v in state]

    def to_gpu(self, state):
        if not isinstance(state, list) or any(not isinstance(v, self.torch.Tensor) or v.device.type != "cpu" for v in state):
            raise RWKVError("checkpoint must contain only CPU state tensors")
        restored = [v.to("cuda:0") for v in state]
        self.assert_gpu(restored)
        return restored

    def save(self, path, checkpoint):
        self.torch.save(checkpoint, path)

    def load(self, path):
        return self.torch.load(path, map_location="cpu", weights_only=True)

    def probe(self):
        state, count = self.observe(None, "你")
        return {"ok": True, "backend": "rwkv7", "identity": self.identity,
                "device": "cuda:0", "dtype": "bf16", "tokens": count,
                "state_tensors": len(state), "gpu": self.torch.cuda.get_device_name(0)}


class Runner:
    """Dependency-injected state transaction layer; fake engines need no torch."""
    def __init__(self, engine, memory=None, after_save=None):
        self.engine = engine
        self.root = Path(memory or C.MEMORY) / ".rwkv"
        self.after_save = after_save  # crash-injection hook, before atomic replacement

    def session_dir(self, session_id):
        return self.root / digest(session_id)

    def latest(self, directory, session_id):
        paths = sorted(directory.glob("revision-*.pth"))
        if not paths:
            return None
        checkpoint = self.engine.load(paths[-1])
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
            raise RWKVError("checkpoint format invalid")
        if checkpoint.get("session_id") != session_id or checkpoint.get("identity") != self.engine.identity:
            raise RWKVError("checkpoint session/model identity mismatch")
        if type(checkpoint.get("revision")) is not int or checkpoint["revision"] < 1:
            raise RWKVError("invalid checkpoint revision")
        if paths[-1].name != f"revision-{checkpoint['revision']:020d}.pth":
            raise RWKVError("checkpoint filename/revision mismatch")
        if checkpoint.get("parent_revision") != checkpoint["revision"] - 1:
            raise RWKVError("checkpoint parent revision invalid")
        if not isinstance(checkpoint.get("receipts"), dict) or len(checkpoint["receipts"]) != checkpoint["revision"]:
            raise RWKVError("checkpoint receipt ledger invalid")
        return checkpoint

    def atomic_checkpoint(self, path, checkpoint):
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            self.engine.save(tmp, checkpoint)
            with tmp.open("r+b") as f:
                os.fsync(f.fileno())
            if self.after_save:
                self.after_save(tmp, path)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def process(self, raw):
        event = clean_event(raw)
        directory = self.session_dir(event["session_id"])
        directory.mkdir(parents=True, exist_ok=True)
        with file_lock(directory / "session.lock", wait=True, timeout=300):
            previous = self.latest(directory, event["session_id"])
            revision = previous["revision"] if previous else 0
            receipts = dict(previous["receipts"]) if previous else {}
            event_key = digest(event["event_id"])
            content_hash = digest(event)
            if event_key in receipts:
                receipt = receipts[event_key]
                if receipt["content_hash"] != content_hash:
                    raise RWKVError("event identity reused with different sanitized content")
                return {**receipt["result"], "cached": True}
            if "parent_revision" in raw and raw["parent_revision"] != revision:
                raise RWKVError(f"parent revision conflict: expected {revision}")
            state = self.engine.to_gpu(previous["source_state"]) if previous else None
            self.engine.assert_gpu(state)
            # Explicit data envelope with boundaries; no prior generated summaries fed.
            observation = "\n<observed_turn>\n" + canonical(event) + "\n</observed_turn>\n"
            state, token_count = self.engine.observe(state, observation)
            self.engine.assert_gpu(state)
            summary, summary_error = None, None
            try:
                summary = validate_summary(self.engine.summarize(self.engine.clone(state)), event)
            except Exception as exc:
                summary_error = nb_migrate.sanitize(f"{type(exc).__name__}: {exc}")
            # Source observation is committed even when summary generation/gate fails.
            # Receipt records explicit summary failure; retry never feeds event twice.
            self.engine.assert_gpu(state)
            result = {"ok": summary_error is None, "backend": "rwkv7",
                      "session_id": event["session_id"], "event_id": event["event_id"],
                      "revision": revision + 1, "parent_revision": revision,
                      "content_hash": content_hash, "source_committed": True,
                      "summary": summary, "summary_error": summary_error,
                      "summary_status": "done" if summary_error is None else "failed",
                      "observation_tokens": token_count, "cached": False}
            receipts[event_key] = {"content_hash": content_hash, "result": result}
            checkpoint = {"format": FORMAT, "identity": self.engine.identity,
                          "session_id": event["session_id"], "revision": revision + 1,
                          "parent_revision": revision, "source_state": self.engine.to_cpu(state),
                          "receipts": receipts, "summary": summary, "summary_error": summary_error}
            self.atomic_checkpoint(directory / f"revision-{revision + 1:020d}.pth", checkpoint)
            return result


def output_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(canonical(value) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("process", "probe"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    args = parser.parse_args()
    try:
        raw = json.load(sys.stdin) if args.command == "process" else None
        if raw is not None:
            clean_event(raw)  # validate before allocating model VRAM
        engine = TorchEngine(args.model, chunk_tokens=args.chunk_tokens)
        result = engine.probe() if args.command == "probe" else Runner(engine).process(raw)
        output_json(args.output, result)
        return 0 if result["ok"] else 2
    except Exception as exc:
        output_json(args.output, {"ok": False, "backend": "rwkv7", "error": nb_migrate.sanitize(f"{type(exc).__name__}: {exc}")})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
