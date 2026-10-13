"""Export the train-split normalisation statistics of an RWKV-slots run, once,
into a self-contained `normalization.pt`.

WHY THIS EXISTS
---------------
`train_rwkv_slots.py` standardises the feature matrix with train-split mean/std
computed in memory, and saves only `run_identity.feature_hash` -- a SHA-256 prefix
of the STANDARDISED train matrix. The statistics themselves are not in the
checkpoint. The evaluator can re-derive them from the run's raw feature cache and
prove the result against that hash, but the cache is a large, deletable,
gitignored directory under `training/`, so a checkpoint outlives its own ability to
be evaluated.

This exporter performs that same verified recovery ONCE, while the cache still
exists, and writes the two tensors plus everything needed to prove their
provenance into a small file next to the checkpoint. After that the checkpoint is
self-contained.

WHAT IT PROVES BEFORE WRITING (all strict; any failure writes nothing)
----------------------------------------------------------------------
1. The checkpoint's own contract: format, label, frozen backbone, decoder config,
   case_ids, data_counts, feature_hash (see eval_rwkv_slots_checkpoint).
2. The raw rows for `case_ids[:train_count]` are present in the training cache, and
   that cache's key/identity match the ones reconstructed from the checkpoint's
   recorded backbone identity + data_hash. The cache is opened READ-ONLY.
3. Recomputing mean/std with the trainer's own operators and standardising the
   train matrix reproduces `run_identity.feature_hash` exactly. Only TRAIN rows are
   read; no test row is touched, and test statistics are never used.
4. If `--train-cases` is given: that file must reproduce the checkpoint's
   `data_version` AND its `case_ids` in order, i.e. it must be the very file this
   run trained on. This is the check that catches "the data file was edited after
   the run started", which is exactly the situation the live risawoz runs hit.
   Without `--train-cases` the exported file records
   `train_cases_verified: false` honestly instead of implying it was checked.

The exporter is CPU-only: it never loads the 1.5B backbone and never touches the
GPU. It does not modify the trainer, the checkpoint or the cache.

USAGE
-----
    py -3.14 export_rwkv_normalization.py \
        --checkpoint D:\\AI\\DBA\\training\\<run>\\checkpoint-last.pt \
        --train-cases D:\\AI\\DBA\\training\\web-sources\\risawoz\\memory_cases.jsonl \
        --out D:\\AI\\DBA\\training\\<run>\\normalization.pt

    py -3.14 export_rwkv_normalization.py --self-test     # CPU unit tests

Exit codes: 0 exported (or self-test passed), 1 refused/failed, 2 nothing to do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_rwkv_slots as T
import eval_rwkv_slots_checkpoint as E


def export(args, torch=None, log=print):
    torch = torch or __import__("torch")
    if torch.cuda.is_available() and not args.allow_gpu_present:
        # Not a failure: the exporter simply must not use the GPU, and saying so
        # keeps a silently GPU-hungry invocation from going unnoticed.
        log("note: a CUDA device is present; this exporter stays on the CPU")

    checkpoint = E.load_and_verify_checkpoint(args.checkpoint, torch,
                                              allow_ablated=args.allow_ablated_checkpoint)
    checkpoint_sha256 = E.sha256_file(checkpoint["path"])
    config = T.DecoderConfig.from_dict(checkpoint["decoder_config"])
    feature_shape = (config.feature_count, config.feature_dim)
    n_layer = config.feature_count // 3
    if config.feature_count % 3:
        raise E.EvalError("feature_count %d is not 3 x layers" % config.feature_count)
    log("checkpoint step=%d identity=%s feature_hash=%s"
        % (checkpoint["step"], checkpoint["identity_hash"][:16], checkpoint["feature_hash"]))

    train_cases = None
    if args.train_cases:
        train_cases = E.verify_training_cases(args.train_cases, checkpoint)
        if not train_cases["matches_checkpoint_data_hash"]:
            raise E.EvalError(
                "--train-cases is NOT the training set of this checkpoint: its data_version "
                "is %s but the checkpoint recorded %s.\n"
                "  This is the normal outcome when the data file was edited after the run "
                "started: the original revision is gone, and the cache's raw rows are then "
                "the only surviving copy of what was actually trained on. Re-run without "
                "--train-cases to export from the cache alone (the file will record "
                "train_cases_verified=false)."
                % (train_cases["data_version"][:16], str(checkpoint["data_hash"])[:16]))
        if not train_cases["case_ids_match_in_order"]:
            raise E.EvalError("--train-cases matches the content hash but not the "
                              "checkpoint's case order/composition")
        log("--train-cases verified: data_version %s and %d case ids match in order"
            % (train_cases["data_version"][:16], len(checkpoint["case_ids"])))

    # The cache directory is found from the CHECKPOINT's recorded data hash, so it
    # is located correctly even when the source file on disk has moved on.
    cache_identity, cache_key = E.feature_key_for(checkpoint["backbone_identity"],
                                                 checkpoint["data_hash"], n_layer)
    root = (Path(args.train_cache_root) if args.train_cache_root
            else Path(args.checkpoint).resolve().parent)
    cache_dir = root / T.FEATURE_CACHE_DIR / cache_key[:16]
    reader = E.ReadOnlyFeatureCache(cache_dir, cache_key, cache_identity, torch)
    reader.load()
    cache_report = reader.describe()
    log("training cache: %s (%d cases, %d shards, closed=%s, read-only)"
        % (cache_dir, cache_report["cases_on_disk"], cache_report["shards"],
           cache_report["closed_manifest"]))
    missing = [case_id for case_id in checkpoint["case_ids"][:checkpoint["train_count"]]
               if case_id not in reader.index]
    if missing:
        raise E.EvalError(
            "the training cache is missing %d of %d training rows (first: %s); the "
            "statistics cannot be exported from a partial cache"
            % (len(missing), checkpoint["train_count"], ", ".join(missing[:5])))

    mean, std, recovery = E.recover_train_statistics(
        reader, checkpoint["case_ids"], checkpoint["train_count"], torch,
        checkpoint["feature_hash"])
    log("statistics recovered and VERIFIED: feature_hash %s (%d train rows, TRain only)"
        % (recovery["recomputed_feature_hash"], recovery["rows"]))

    train_ids = [str(value) for value in checkpoint["case_ids"][:checkpoint["train_count"]]]
    payload = {
        "format": E.NORMALIZATION_FORMAT,
        "tool": E.NORMALIZATION_TOOL,
        "created_at": time.time(),
        "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mean": mean.to(torch.float32).contiguous(),
        "std": std.to(torch.float32).contiguous(),
        "payload_digest": E.normalization_payload_digest(mean, std, torch),
        "train_count": checkpoint["train_count"],
        "identity": {
            "identity_hash": checkpoint["identity_hash"],
            "data_hash": checkpoint["data_hash"],
            "feature_hash": checkpoint["feature_hash"],
            "feature_version": T.FEATURE_VERSION,
            "feature_function": T.feature_function_fingerprint(),
            "feature_shape": list(feature_shape),
            "train_count": checkpoint["train_count"],
            "case_ids_sha256": hashlib_join(train_ids),
            "backbone_identity_sha256": (checkpoint["backbone_identity"] or {}).get("sha256"),
            "tokenizer": checkpoint["tokenizer"],
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": checkpoint["step"],
            "train_cases_path": str(args.train_cases) if args.train_cases else None,
            "train_cases_sha256": train_cases["sha256"] if train_cases else None,
            "train_cases_verified": bool(train_cases),
            "train_cases_check": train_cases,
            "cache": {**cache_report, "manifest_path": str(cache_dir / T.FEATURE_CACHE_MANIFEST),
                      "manifest_case_count": (reader.manifest or {}).get("case_count"),
                      "manifest_shards": len((reader.manifest or {}).get("shards") or [])},
        },
        "verification": {
            "feature_hash_recomputed": recovery["recomputed_feature_hash"],
            "matches_checkpoint_feature_hash": recovery["verified"],
            "stats_are_train_only": True,
            "test_rows_read": 0,
            "train_rows_read": recovery["rows"],
            "method": "rows assembled in run_identity.case_ids[:train_count] order; mean and "
                      "std computed with the trainer's own operators; result verified by "
                      "hashing the standardised train matrix",
            "std_min": recovery["std_min"], "std_max": recovery["std_max"],
        },
        "limits": [
            "train_cases_verified=false means the original training file was not available "
            "for the data_version/case-order check; the statistics are still the verified "
            "train-split reduction over the checkpoint's own case_ids.",
            "This file is only valid for the run identity it records. The evaluator "
            "re-checks identity_hash/data_hash/feature_hash/feature_shape/train_count.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    E.atomic_torch_save(out, payload, torch)
    written = torch.load(out, map_location="cpu", weights_only=False)
    if E.normalization_payload_digest(written["mean"], written["std"], torch) \
            != payload["payload_digest"]:
        raise E.EvalError("the written normalization file did not read back intact")
    log("wrote %s (%.1f KB) payload_digest=%s"
        % (out, out.stat().st_size / 1024, payload["payload_digest"]))
    log("checkpoint dir now contains a self-contained normalisation contract; the evaluator "
        "will find it automatically as %s" % E.NORMALIZATION_NAME)
    if not args.train_cases:
        log("WARNING: exported without --train-cases, so train_cases_verified=false. "
            "Pass the ORIGINAL training cases file next time to record that check.")
    return payload


def hashlib_join(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:32]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export verified TRAIN-split normalisation statistics for an "
                    "RWKV-slots checkpoint into a self-contained normalization.pt. "
                    "CPU only; reads the feature cache read-only; never modifies the "
                    "trainer or the checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="[required]")
    parser.add_argument("--train-cases", default=None,
                        help="the ORIGINAL training cases file (strictly verified if given)")
    parser.add_argument("--out", default=None,
                        help="where to write, normally <checkpoint dir>/%s" % E.NORMALIZATION_NAME)
    parser.add_argument("--train-cache-root", default=None,
                        help="training run dir holding features/<key>/ "
                             "(default: the checkpoint's directory)")
    parser.add_argument("--allow-ablated-checkpoint", action="store_true",
                        help="export for a --state-mode none/shuffled checkpoint")
    parser.add_argument("--allow-gpu-present", action="store_true",
                        help="silence the note that a CUDA device is present (the exporter "
                             "stays on the CPU either way)")
    parser.add_argument("--self-test", action="store_true",
                        help="CPU unit tests; needs no checkpoint, cache, data or GPU")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [name for name in ("checkpoint", "out") if not getattr(args, name)]
        if missing:
            raise E.EvalError("required: " + ", ".join("--" + n.replace("_", "-")
                                                       for n in missing))
    return args


def self_test():
    """Round-trip and refusal tests for the exported contract (CPU only)."""
    import tempfile
    import torch

    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))

    identity = {"sha256": "a" * 64, "bytes": 1, "architecture": "x", "strategy": "y",
                "tokenizer": "t", "rwkv_pack": "0.8.32"}
    raw = torch.randn(6, 2, 4, generator=torch.Generator().manual_seed(21))
    mean_ref = raw.mean(dim=(0, 1), keepdim=True)
    std_ref = raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    standardised = (raw - mean_ref) / std_ref
    import hashlib as _hashlib
    feature_hash = _hashlib.sha256(
        standardised.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]
    case_ids = ["c%d" % index for index in range(6)]

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _, key = E.feature_key_for(identity, "datahash", 2)
        cache_dir = root / T.FEATURE_CACHE_DIR / key[:16]
        writer = T.FeatureCache(cache_dir, key, E.feature_key_for(identity, "datahash", 2)[0],
                                torch, every=100)
        for index, case_id in enumerate(case_ids):
            writer.add(case_id, raw[index], index + 1)
        writer.close()

        reader = E.ReadOnlyFeatureCache(cache_dir, key,
                                       E.feature_key_for(identity, "datahash", 2)[0], torch)
        reader.load()
        mean, std, recovery = E.recover_train_statistics(reader, case_ids, 6, torch, feature_hash)
        check("exporter recovery verifies against feature_hash", recovery["verified"])
        check("exporter recovery equals the trainer's statistics exactly",
              bool(torch.equal(mean, mean_ref) and torch.equal(std, std_ref)))

        digest = E.normalization_payload_digest(mean, std, torch)
        check("payload digest is stable", digest == E.normalization_payload_digest(mean, std, torch))
        check("payload digest changes when a statistic changes",
              digest != E.normalization_payload_digest(mean + 1e-3, std, torch))

        # A payload built exactly as `export` builds it must pass the evaluator's
        # loader, and each identity field must be able to refuse it on its own.
        def payload_for(overrides=None, mean_value=None, digest_value=None):
            body = {"format": E.NORMALIZATION_FORMAT, "tool": E.NORMALIZATION_TOOL,
                    "created_at": time.time(),
                    "mean": (mean_value if mean_value is not None else mean).clone(),
                    "std": std.clone(),
                    "payload_digest": digest_value or digest,
                    "identity": {"identity_hash": "IH", "data_hash": "DH",
                                 "feature_hash": "FH", "feature_version": T.FEATURE_VERSION,
                                 "feature_function": T.feature_function_fingerprint(),
                                 "feature_shape": [2, 4], "train_count": 6,
                                 "checkpoint_sha256": "cs"}}
            if overrides:
                body["identity"].update(overrides)
            return body

        fake_checkpoint = {"identity_hash": "IH", "data_hash": "DH", "feature_hash": "FH",
                           "train_count": 6, "path": str(root / "ck.pt"), "step": 1}
        (root / "ck.pt").write_bytes(b"pretend checkpoint")
        checkpoint_sha256 = E.sha256_file(root / "ck.pt")
        path = root / E.NORMALIZATION_NAME
        torch.save(payload_for({"checkpoint_sha256": checkpoint_sha256}), path)
        loaded_mean, loaded_std, report = E.load_normalization_file(path, fake_checkpoint,
                                                                   torch, (2, 4))
        check("evaluator loader accepts a matching file",
              bool(torch.equal(loaded_mean, mean) and torch.equal(loaded_std, std)),
              json.dumps(report.get("failed_fields")))
        check("evaluator loader recomputes the payload digest",
              report["payload_digest"] == digest)
        check("checkpoint artifact hash is reported as unchanged",
              report["checkpoint_artifact_unchanged"] is True,
              str(report["checkpoint_sha256_now"]))

        # A rolling checkpoint overwritten by a later step must NOT be rejected on
        # its file hash alone; the run identity is the authority.
        torch.save(payload_for({"checkpoint_sha256": "0" * 64}), path)
        _, _, report2 = E.load_normalization_file(path, fake_checkpoint, torch, (2, 4))
        check("a changed checkpoint file hash is reported, not fatal",
              report2["checkpoint_artifact_unchanged"] is False
              and not report2["failed_fields"], json.dumps(report2["failed_fields"]))
        torch.save(payload_for(), path)

        torch.save(payload_for(digest_value="0" * 32), path)
        try:
            E.load_normalization_file(path, fake_checkpoint, torch, (2, 4))
            check("a tampered payload is refused", False, "no error raised")
        except E.EvalError as exc:
            check("a tampered payload is refused", "payload digest mismatch" in str(exc))
        torch.save(payload_for(), path)

        for field, wrong in (("identity_hash", "OTHER"), ("data_hash", "OTHER"),
                             ("feature_hash", "OTHER"), ("train_count", 99)):
            torch.save(payload_for({field: wrong}), path)
            try:
                E.load_normalization_file(path, fake_checkpoint, torch, (2, 4))
                check("a file from another run is refused on %s" % field, False, "accepted")
            except E.EvalError as exc:
                check("a file from another run is refused on %s" % field, field in str(exc))
        torch.save(payload_for(), path)
        try:
            E.load_normalization_file(path, fake_checkpoint, torch, (2, 8))
            check("a wrong feature shape is refused", False, "accepted")
        except E.EvalError as exc:
            check("a wrong feature shape is refused", "shape" in str(exc))
        torch.save(payload_for(mean_value=torch.full_like(mean, float("nan"))), path)
        try:
            E.load_normalization_file(path, fake_checkpoint, torch, (2, 4))
            check("non-finite statistics are refused", False, "accepted")
        except E.EvalError as exc:
            check("non-finite statistics are refused", "non-finite" in str(exc))
        torch.save(payload_for(mean_value=mean.clone(), digest_value=digest), path)

        # atomic write helper used by the exporter
        target = root / "atomic.pt"
        E.atomic_torch_save(target, {"a": 1}, torch)
        check("atomic_torch_save writes a readable payload",
              torch.load(target, map_location="cpu", weights_only=False) == {"a": 1})
        check("atomic_torch_save leaves no temporary behind",
              not list(root.glob("*.tmp")))
        check("the cache survived read-only use", len(list(cache_dir.glob("shard-*"))) == 2)

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print("%s %s%s" % ("PASS" if ok else "FAIL", name, ("  <- " + detail) if detail and not ok else ""))
    print("\n%d/%d exporter self-test checks passed" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.self_test:
            return self_test()
        export(args)
    except (E.EvalError, T.TrainingError) as exc:
        print("FAILED: %s" % exc, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
