"""Does predicting the SET SIZE instead of thresholding 931 logits break the wall?

Measured on this dev split:
  * a plain linear multi-label head over 931 triples tops out at 8.07% exact, and at its
    best threshold it emits 49.3 triples per case against a true mean of 2.24 -- it is not
    failing to rank, it is failing to stop;
  * per-triple AUC is 0.9788, so the ranking carries real signal;
  * the true set size takes only five values (0..4, mean 2.24), which is a 5-class
    problem, not a 931-class one.

Those three facts point at one testable fix: predict the size, then take exactly that many
top-scoring triples. This script measures the best possible outcome of that fix by supplying
the TRUE size (an oracle upper bound on the size-prediction strategy) and comparing it with
the best fixed threshold.

Reading the result:
  * top-k with the true k far above every threshold result -> the wall was the stopping
    rule, and the run should add a size head (5 classes, trivially learnable);
  * top-k with the true k still near the floor             -> the ranking itself is the wall
    and no decoding change will help.

CPU only. Fits one linear head exactly as `measure_constraint_linear_ceiling.py` does, then
compares decoders on the same logits so the comparison is apples to apples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_rwkv_slot_pairs as P  # noqa: E402


def exact_rate(predicted_sets, references):
    hits = sum(1 for predicted, reference in zip(predicted_sets, references)
               if predicted == reference["constraint"])
    return hits, hits / len(references)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cache-identity", required=True)
    parser.add_argument("--pool", default="meanmax")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    args = parser.parse_args(argv)

    torch = P.base._torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA; pass --device cpu")

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)

    cache = P.ReadOnlyFeatureCache(Path(args.feature_cache), args.cache_identity, None,
                                   allow_any_backbone=True)
    cache.open()
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    pooled = P.build_pooled(cache, ordered, args.pool)
    train_count = len(train_cases)
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    features = (pooled - mean) / std
    x_train = features[:train_count].to(args.device)
    x_dev = features[train_count:].to(args.device)

    targets = P.encode_targets(train_pairs, vocab, torch).multi.to(args.device)
    pos_weight = P.triple_positive_weight(
        P.encode_targets(train_pairs, vocab, torch).multi, torch,
        power=args.pos_weight_power, clamp=args.pos_weight_clamp).to(args.device)

    torch.manual_seed(20261008)
    weight = torch.zeros(x_train.shape[1], len(vocab.triples),
                         device=args.device).normal_(0, 0.01).requires_grad_(True)
    bias = torch.zeros(len(vocab.triples), device=args.device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(20261008)
    for step in range(args.steps):
        index = torch.randint(0, x_train.shape[0], (args.batch_size,), generator=generator)
        optimizer.zero_grad()
        loss = loss_fn(x_train[index].mm(weight) + bias, targets[index])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dev_logits = (x_dev.mm(weight) + bias).float().cpu()

    report = {"threshold_decoders": [], "oracle_size_topk": [], "fixed_size_topk": []}

    for raw in args.thresholds.split(","):
        threshold = float(raw)
        mask = torch.sigmoid(dev_logits) >= threshold
        predicted = []
        for row in range(mask.shape[0]):
            predicted.append(frozenset(
                vocab.triples.decode(position)
                for position in mask[row].nonzero(as_tuple=False).flatten().tolist()))
        hits, rate = exact_rate(predicted, dev_pairs)
        report["threshold_decoders"].append({
            "threshold": threshold, "exact": hits, "rate": rate,
            "mean_size": sum(len(item) for item in predicted) / len(predicted)})

    # Oracle size: decode exactly as many triples as the reference has. This is the ceiling
    # of the "add a small size head" strategy if that head were perfect.
    predicted = []
    for row in range(dev_logits.shape[0]):
        k = len(dev_pairs[row]["constraint"])
        if k == 0:
            predicted.append(frozenset())
            continue
        top = torch.topk(dev_logits[row], k).indices.tolist()
        predicted.append(frozenset(vocab.triples.decode(p) for p in top))
    hits, rate = exact_rate(predicted, dev_pairs)
    report["oracle_size_topk"] = {"exact": hits, "rate": rate}

    # Fixed k, so it is clear how sharp the size decision has to be.
    for k in (1, 2, 3, 4):
        predicted = []
        for row in range(dev_logits.shape[0]):
            top = torch.topk(dev_logits[row], k).indices.tolist()
            predicted.append(frozenset(vocab.triples.decode(p) for p in top))
        hits, rate = exact_rate(predicted, dev_pairs)
        report["fixed_size_topk"].append({"k": k, "exact": hits, "rate": rate})

    # How well is the size itself predicted from the same features? A 5-class problem; if
    # this is high, the size-head strategy is only as limited as the oracle above.
    size_train = torch.tensor([len(entry["constraint"]) for entry in train_pairs],
                              device=args.device)
    size_dev = [len(entry["constraint"]) for entry in dev_pairs]
    torch.manual_seed(20261008)
    size_weight = torch.zeros(x_train.shape[1], 5, device=args.device).normal_(
        0, 0.01).requires_grad_(True)
    size_bias = torch.zeros(5, device=args.device, requires_grad=True)
    size_optimizer = torch.optim.AdamW([size_weight, size_bias], lr=1e-3, weight_decay=0.01)
    for step in range(2000):
        index = torch.randint(0, x_train.shape[0], (256,), generator=generator)
        size_optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            x_train[index].mm(size_weight) + size_bias, size_train[index])
        loss.backward()
        size_optimizer.step()
    with torch.no_grad():
        predicted_size = (x_dev.mm(size_weight) + size_bias).argmax(dim=1).tolist()
    correct = sum(1 for a, b in zip(predicted_size, size_dev) if a == b)
    report["size_head"] = {
        "accuracy": correct / len(size_dev),
        "accuracy_if_always_2": sum(1 for b in size_dev if b == 2) / len(size_dev),
        "correct": correct,
        "confusion": {str(actual): {str(p): sum(1 for a, b in zip(predicted_size, size_dev)
                                            if b == actual and a == p)
                                    for p in range(5)} for actual in range(5)},
    }

    # And the realistic strategy: predicted size + top-k on the same logits.
    predicted = []
    for row in range(dev_logits.shape[0]):
        k = predicted_size[row]
        if k == 0:
            predicted.append(frozenset())
            continue
        top = torch.topk(dev_logits[row], k).indices.tolist()
        predicted.append(frozenset(vocab.triples.decode(p) for p in top))
    hits, rate = exact_rate(predicted, dev_pairs)
    report["predicted_size_topk"] = {"exact": hits, "rate": rate}

    report["reference"] = {"pair_model_dev_constraint_rate": 0.0344,
                           "pair_model_oracle_topk_recall": 0.3159,
                           "linear_best_threshold_dev_rate": 0.0812}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
