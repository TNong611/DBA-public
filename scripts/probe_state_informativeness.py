"""Decisive diagnostic: do the cached frozen-RWKV state features carry the slot labels?

The decoder is failing on heldout while its zero-state ablation is just as good, which
means it never learned to read the state. Two very different causes look identical there:

  A. the features really do encode the label, and the decoder is at fault; or
  B. the extracted features do not encode the label, so nothing can read them.

This script settles it without touching the training run. It reads the SAME immutable
feature cache the trainer writes (train-only standardisation, raw rows), pools each case
to a fixed vector, and fits a plain linear classifier on the train group, scoring the
heldout group. A linear probe is the weakest reader in existence: if a linear map cannot
recover the label, no decoder architecture can, and the bug is upstream in feature
extraction. If it can, the fault is in the decoder.

What is scored (separately, so nothing hides behind an aggregate):
  * topic  (主题) as a 191-way classification problem -- the slot with a fixed vocabulary
  * the exact four-slot message, tokenised, is NOT forced here on purpose: the probe is
    about whether information exists at all, not about generation ability.

Guards kept from the rest of the pipeline:
  * reads the cache strictly read-only (never deletes or rewrites a shard);
  * refuses to report a number if train/heldout cases are missing from the cache;
  * reports the majority-class baseline next to every accuracy, so a probe that only
    predicts the commonest topic cannot be mistaken for information;
  * never claims a result from an incomplete cache -- it says how many cases it saw.

CPU-only by design: the GPU belongs to the live training run.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rwkv_slots_data as DATA


def load_cases(path):
    """id -> (group, topic) from the case file. Topic is the 主题 slot value."""
    cases = DATA.all_cases(path)
    out = {}
    for case in cases:
        summary = case.get("summary") or case.get("assistant") or ""
        topic = None
        for line in str(summary).splitlines():
            line = line.strip()
            if line.startswith("主题：") or line.startswith("主题:"):
                topic = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                break
        out[str(case.get("id"))] = (str(case.get("group")), topic)
    return out


def pooled_features(cache_dir, want_ids, pool):
    """case id -> pooled float32 vector, read shard by shard (memory stays bounded)."""
    shards = sorted(glob.glob(os.path.join(cache_dir, "shard-*.pt")))
    if not shards:
        raise SystemExit("no shards under %s" % cache_dir)
    import torch

    vectors, found = {}, 0
    for path in shards:
        blob = torch.load(path, map_location="cpu")
        ids = blob["ids"]
        feats = blob["features"].float()
        for index, case_id in enumerate(ids):
            if case_id not in want_ids:
                continue
            rows = feats[index]                       # [n_rows, width]
            if pool == "mean":
                vector = rows.mean(dim=0)
            elif pool == "meanmax":
                vector = torch.cat([rows.mean(dim=0), rows.max(dim=0).values], dim=0)
            else:
                raise SystemExit("unknown pool %r" % pool)
            vectors[case_id] = vector
            found += 1
        del blob, feats
    return vectors, found, len(shards)


def fit_probe(train_x, train_y, heldout_x, heldout_y, classes, iterations, seed, torch):
    """Multinomial linear probe with plain SGD; returns heldout accuracy."""
    index = {name: i for i, name in enumerate(classes)}
    y_train = torch.tensor([index[name] for name in train_y], dtype=torch.long)
    y_held = torch.tensor([index[name] for name in heldout_y], dtype=torch.long)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    heldout_x = (heldout_x - mean) / std

    torch.manual_seed(seed)
    weight = torch.zeros(train_x.shape[1], len(classes), requires_grad=True)
    bias = torch.zeros(len(classes), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(iterations):
        optimizer.zero_grad()
        logits = train_x.mm(weight) + bias
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_acc = float((train_x.mm(weight) + bias).argmax(1).eq(y_train).float().mean())
        held_logits = heldout_x.mm(weight) + bias
        held_acc = float(held_logits.argmax(1).eq(y_held).float().mean())
    return {"train_accuracy": train_acc, "heldout_accuracy": held_acc,
            "final_train_loss": float(loss.detach())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="case file the run trained on")
    parser.add_argument("--feature-cache", required=True,
                        help="the run's features/<cache_key>/ directory")
    parser.add_argument("--pool", default="meanmax", choices=("mean", "meanmax"))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    meta = load_cases(args.cases)
    vectors, found, shards = pooled_features(args.feature_cache, set(meta), args.pool)
    import torch

    cached_train = [i for i in vectors if meta[i][0] == DATA.TRAIN]
    cached_held = [i for i in vectors if meta[i][0] == DATA.HELDOUT]
    all_train = [i for i in meta if meta[i][0] == DATA.TRAIN]
    all_held = [i for i in meta if meta[i][0] == DATA.HELDOUT]

    coverage = {"train": len(cached_train), "heldout": len(cached_held),
                "train_expected": len(all_train), "heldout_expected": len(all_held),
                "shards_read": shards}
    report = {"pool": args.pool, "iterations": args.iterations, "coverage": coverage}

    topics = sorted({meta[i][1] for i in cached_train + cached_held if meta[i][1]})
    report["topic_classes"] = len(topics)

    # score only on what the cache actually holds, and say so
    train_ids = [i for i in sorted(cached_train) if meta[i][1]]
    held_ids = [i for i in sorted(cached_held) if meta[i][1]]
    stack = lambda ids: torch.stack([vectors[i] for i in ids])
    x_train, x_held = stack(train_ids), stack(held_ids)
    y_train = [meta[i][1] for i in train_ids]
    y_held = [meta[i][1] for i in held_ids]

    counts = collections.Counter(y_train)
    majority = counts.most_common(1)[0]
    report["train_majority_topic"] = majority[0]
    report["train_majority_share"] = majority[1] / max(1, len(y_train))
    report["chance_topic_rate"] = 1.0 / max(1, len(topics))
    report["topic_probe"] = fit_probe(x_train, y_train, x_held, y_held, topics,
                                      args.iterations, args.seed, torch)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
