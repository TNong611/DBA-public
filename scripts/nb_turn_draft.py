"""Durable DSH pending-turn snapshots. Only user/assistant leaves are sanitized."""
from __future__ import annotations
import argparse, json, os, sys, uuid
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C
import nb_migrate
from nb_turn import atomic_json, canonical, digest, file_lock, TurnError

FIELDS = ("session_id", "event_id", "user", "assistant", "at")

def root() -> Path:
    p = C.MEMORY / ".turns" / "drafts"
    p.mkdir(parents=True, exist_ok=True)
    return p

def clean(raw: dict) -> dict:
    if not isinstance(raw, dict) or any(k not in raw for k in FIELDS):
        raise TurnError("draft requires session_id,event_id,user,assistant,at")
    if any(not isinstance(raw[k], str) for k in FIELDS):
        raise TurnError("draft fields must be strings")
    stage = raw.get("stage", "ongoing")
    if not isinstance(stage, str) or stage not in ("ongoing", "interrupted", "ready"):
        raise TurnError("invalid draft stage")
    updated_at = raw.get("updated_at")
    if updated_at is None:
        updated_at = datetime.now().astimezone().isoformat()
    if not isinstance(updated_at, str):
        raise TurnError("updated_at must be an ISO timestamp")
    try:
        datetime.fromisoformat(updated_at)
    except ValueError as exc:
        raise TurnError("updated_at must be an ISO timestamp") from exc
    # Strict whitelist: arbitrary metadata never enters the persisted record.
    out = {k: raw[k] for k in FIELDS}
    out["user"] = nb_migrate.sanitize(raw["user"])
    out["assistant"] = nb_migrate.sanitize(raw["assistant"])
    out["stage"] = stage
    out["updated_at"] = updated_at
    if not out["session_id"].strip() or not out["event_id"].strip() or not out["user"].strip():
        raise TurnError("draft identity and user must be nonempty")
    return out

def key(event: dict) -> str:
    return digest([event["session_id"], event["event_id"]])

def path_for(event: dict) -> Path:
    return root() / (key(event) + ".json")

def snapshot(raw: dict) -> str:
    event = clean(raw)
    path = path_for(event)
    owned = {"event": event, "draft_hash": digest(event), "updated_at": event["updated_at"]}
    with file_lock(root() / ".lock", wait=True):
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("draft_hash") != digest(old.get("event", {})):
                raise TurnError("existing draft hash mismatch")
            if any(old.get("event", {}).get(k) != event[k] for k in ("event_id", "session_id", "user", "at")):
                raise TurnError("draft identity reused with different source")
        atomic_json(path, owned)
    return key(event)

def items() -> list[dict]:
    out = []
    for p in sorted(root().glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            event = obj["event"]
            if obj.get("draft_hash") != digest(event):
                raise TurnError("draft hash mismatch")
            out.append(event)
        except Exception as e:
            out.append({"error": f"{p.stem}: {type(e).__name__}: {e}"})
    return out

def remove(event_id: str, session_id: str) -> None:
    e = {"event_id": event_id, "session_id": session_id}
    p = root() / (digest([session_id, event_id]) + ".json")
    with file_lock(root() / ".lock", wait=True):
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if obj.get("event", {}).get("event_id") != event_id:
                raise TurnError("draft identity mismatch")
            # Keep lifecycle provenance separate from nb_turn's exact event schema.
            archive = root() / "submitted" / p.name
            atomic_json(archive, obj)
            p.unlink()

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("list")
    rm = sub.add_parser("remove"); rm.add_argument("session_id"); rm.add_argument("event_id")
    a = ap.parse_args()
    try:
        if a.cmd == "snapshot": print(canonical({"key": snapshot(json.load(sys.stdin)), "persisted": True}))
        elif a.cmd == "list": print(canonical(items()))
        else: remove(a.event_id, a.session_id); print(canonical({"removed": True}))
        return 0
    except Exception as e:
        print(canonical({"error": f"{type(e).__name__}: {e}"}), file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
