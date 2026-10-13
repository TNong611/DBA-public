"""Is the CONSTRAINT information in the frozen state at all?

The pair run reached `triple` loss 0.005 while `oracle_topk_recall` was only 0.3159:
even when handed the exact number of top-scoring triples, it recovered under a third of
the gold positives. Threshold sweeps made it worse, and 83.8% of dev positives come from
triples seen 20+ times in train, so neither calibration nor rarity explains it.

Before changing the model again, the cheap and decisive question is whether the
information is in the features: a linear probe is the weakest possible reader, and the
topic probe already showed the same pooled features carry topic at 66% (chance 0.47%).

This trains one independent logistic probe per frequent triple on the TRAIN split and
scores AUC on dev. Reading the result:
  * high AUC on frequent triples -> the state carries it, and the pair model's
    shared-encoder multi-label head is the thing to fix;
  * AUC near 0.5 -> the pooled features do not carry that information, and the lever is
    the observation text or the pooling, not another head.

CPU only; the GPU is left free. Writes nothing; prints a table.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_rwkv_slot_pairs as P  # noqa: E402


def auc(scores, labels):
    """Rank AUC with ties handled by midpoint ranks; no sklearn dependency."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--pool", default="meanmax")
    parser.add_argument("--top", type=int, default=40, help="how many frequent triples to probe")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-identity", default=None,
                        help="data_hash the cache was built for; read from --identity-from "
                             "when omitted")
    parser.add_argument("--identity-from", default=None,
                        help="a run report whose cache_identity should be reused, so this "
                             "cannot disagree with what the checkpoint trained on")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch = P.base._torch()
    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)

    # The cache verifies its own identity against the dataset hash, so that hash is an
    # input, not a guess. Preferring an existing run report keeps this probe on exactly
    # the rows a real run used; `--cache-identity` is the explicit escape hatch.
    if args.cache_identity:
        identity = args.cache_identity
    elif args.identity_from:
        identity = json.loads(
            Path(args.identity_from).read_text(encoding="utf-8"))["cache_identity"]
    else:
        raise SystemExit("pass --cache-identity or --identity-from; the cache refuses to be "
                         "read without the dataset hash it was extracted for")
    cache = P.ReadOnlyFeatureCache(Path(args.feature_cache), identity, None,
                                   allow_any_backbone=True)
    cache.open()
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    pooled = P.build_pooled(cache, ordered, args.pool)
    train_count = len(train_cases)
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    x_train = ((pooled[:train_count] - mean) / std).to(args.device)
    x_dev = ((pooled[train_count:] - mean) / std).to(args.device)

    frequency = collections.Counter()
    for entry in train_pairs:
        for triple in entry["constraint"]:
            frequency[triple] += 1
    probe_targets = [triple for triple, _ in frequency.most_common(args.top)]

    train_sets = [set(entry["constraint"]) for entry in train_pairs]
    dev_sets = [set(entry["constraint"]) for entry in dev_pairs]

    rows = []
    for triple in probe_targets:
        y_train = torch.tensor([1.0 if triple in s else 0.0 for s in train_sets])
        y_dev = [1 if triple in s else 0.0 for s in dev_sets]
        positives = int(y_train.sum())
        if positives == 0 or positives == len(y_train):
            continue
        torch.manual_seed(20261008)
        weight = torch.zeros(x_train.shape[1], 1, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        # Positive weighting: these labels are sparse, and an unweighted fit would just
        # predict the majority class and report a meaningless AUC of 0.5.
        positive_weight = torch.tensor((len(y_train) - positives) / positives)
        optimizer = torch.optim.Adam([weight, bias], lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        for _ in range(args.iterations):
            optimizer.zero_grad()
            logits = x_train.mm(weight).squeeze(1) + bias
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            dev_scores = (x_dev.mm(weight).squeeze(1) + bias).tolist()
        rows.append({"triple": list(triple), "train_frequency": frequency[triple],
                     "train_positives": positives,
                     "dev_positives": int(sum(y_dev)),
                     "dev_auc": auc(dev_scores, y_dev)})

    scored = [row for row in rows if row["dev_auc"] is not None and row["dev_positives"] >= 5]
    scored.sort(key=lambda row: row["dev_auc"], reverse=True)
    report = {
        "probed": len(rows),
        "scored_with_dev_support": len(scored),
        "mean_auc": (sum(row["dev_auc"] for row in scored) / len(scored)) if scored else None,
        "median_auc": (sorted(row["dev_auc"] for row in scored)[len(scored) // 2]
                       if scored else None),
        "auc_above_0.7": sum(1 for row in scored if row["dev_auc"] > 0.7),
        "auc_above_0.8": sum(1 for row in scored if row["dev_auc"] > 0.8),
        "top": scored[:12],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
