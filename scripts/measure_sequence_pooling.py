"""Does pooling away the event sequence cost the constraint slot?

The cache stores each case as [72, 2048] per-event state rows. Every run so far collapsed
that to one 4096-vector with `meanmax`, which keeps no ordering: event 3 and event 60 are
interchangeable afterwards. The constraint's markers (本轮用户条件 / 本轮条件 / 沿用 /
沿用已告知条件) say WHEN a condition appeared, so ordering is plausibly load-bearing.

This compares pooled representations that differ ONLY in how much sequence they keep:
  * chunked mean+max over N chunks (N=1 reproduces the run's 4096-dim meanmax exactly);
  * optionally a straight concatenation of chunk means.

All heads are the same plain linear multi-label head with the same weighting, and the same
size-aware decoder is used for every variant, so a difference can only come from the
representation. N=1 is the control: if it does not reproduce the known ~11.9% baseline,
something else differs and the comparison is void.

Writes nothing; prints JSON. Reads the feature cache read-only.
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


def chunked_features(cache, cases, chunks, torch, log=None):
    """[N, chunks * width * 2] -- per-chunk mean and max, chunk-major.

    With chunks=1 this is exactly `meanmax`: cat([mean, max]) over all rows.
    """
    rows = None
    for case in cases:
        row = cache.row(case["id"])
        if row is None:
            raise P.TrainingError("case %s missing from cache" % case["id"])
        if rows is None:
            rows = row.shape[0]
        break
    width = int(cache.shape[1])
    matrix = torch.empty((len(cases), chunks * width * 2), dtype=torch.float32)
    for index, case in enumerate(cases):
        row = cache.row(case["id"])
        if chunks == 1:
            matrix[index] = torch.cat([row.mean(dim=0), row.max(dim=0).values])
        else:
            pieces = []
            edges = [round(i * rows / chunks) for i in range(chunks + 1)]
            for chunk in range(chunks):
                block = row[edges[chunk]:max(edges[chunk] + 1, edges[chunk + 1])]
                pieces.append(block.mean(dim=0))
                pieces.append(block.max(dim=0).values)
            matrix[index] = torch.cat(pieces)
        if log is not None and (index + 1) % 2000 == 0:
            log("chunk-pooled %d/%d (chunks=%d)" % (index + 1, len(cases), chunks))
    return matrix


def fit_and_evaluate(x_train, x_dev, targets, size_targets, references, vocab, torch,
                     device, steps, batch_size, lr, pos_weight, seed=20261008):
    torch.manual_seed(seed)
    weight = torch.zeros(x_train.shape[1], len(vocab.triples), device=device).normal_(
        0, 0.01).requires_grad_(True)
    bias = torch.zeros(len(vocab.triples), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        index = torch.randint(0, x_train.shape[0], (batch_size,), generator=generator)
        optimizer.zero_grad()
        loss = loss_fn(x_train[index].mm(weight) + bias, targets[index])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        dev_logits = (x_dev.mm(weight) + bias).float().cpu()

    # The size head must be fit OUTSIDE `no_grad`: leaving it inside silently produced a
    # loss with no graph, and `loss.backward()` raised "element 0 of tensors does not
    # require grad". Setting `requires_grad=True` on the parameters is not enough -- the
    # ambient no_grad context strips the graph from the forward pass regardless.
    torch.manual_seed(seed)
    size_weight = torch.zeros(x_train.shape[1], 5, device=device).normal_(
        0, 0.01).requires_grad_(True)
    size_bias = torch.zeros(5, device=device, requires_grad=True)
    size_optimizer = torch.optim.AdamW([size_weight, size_bias], lr=lr, weight_decay=0.01)
    for step in range(2000):
        index = torch.randint(0, x_train.shape[0], (batch_size,), generator=generator)
        size_optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            x_train[index].mm(size_weight) + size_bias, size_targets[index])
        loss.backward()
        size_optimizer.step()
    with torch.no_grad():
        predicted_size = (x_dev.mm(size_weight) + size_bias).argmax(dim=1).tolist()

    def decode(kind):
        predicted = []
        if kind == "threshold":
            mask = torch.sigmoid(dev_logits) >= 0.5
            for row in range(dev_logits.shape[0]):
                predicted.append(frozenset(
                    vocab.triples.decode(position)
                    for position in mask[row].nonzero(as_tuple=False).flatten().tolist()))
        else:
            for row in range(dev_logits.shape[0]):
                k = (len(references[row]["constraint"]) if kind == "oracle"
                     else predicted_size[row])
                if k == 0:
                    predicted.append(frozenset())
                    continue
                top = torch.topk(dev_logits[row], k).indices.tolist()
                predicted.append(frozenset(vocab.triples.decode(p) for p in top))
        hits = sum(1 for p, r in zip(predicted, references) if p == r["constraint"])
        return hits, hits / len(references)

    size_correct = sum(1 for a, b in zip(predicted_size, size_targets_dev(references))
                       if a == b)
    return {
        "constraint_threshold_0.5": decode("threshold")[1],
        "constraint_predicted_size_topk": decode("size")[1],
        "constraint_oracle_size_topk": decode("oracle")[1],
        "size_head_accuracy": size_correct / len(references),
        "dim": int(x_train.shape[1]),
    }


def size_targets_dev(references):
    return [len(reference["constraint"]) for reference in references]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cache-identity", required=True)
    parser.add_argument("--chunks", default="1,4,8,18,36")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch = P.base._torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA; pass --device cpu")

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)
    references = [dict(entry) for entry in dev_pairs]

    cache = P.ReadOnlyFeatureCache(Path(args.feature_cache), args.cache_identity, None,
                                   allow_any_backbone=True)
    cache.open()
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    train_count = len(train_cases)

    raw_targets = P.encode_targets(train_pairs, vocab, torch).multi
    targets = raw_targets.to(args.device)
    pos_weight = P.triple_positive_weight(raw_targets, torch, power=args.pos_weight_power,
                                          clamp=args.pos_weight_clamp).to(args.device)
    size_targets = torch.tensor([len(entry["constraint"]) for entry in train_pairs],
                                device=args.device)

    results = []
    for raw in args.chunks.split(","):
        chunks = int(raw)
        features = chunked_features(cache, ordered, chunks, torch,
                                    log=lambda message: print(message, flush=True))
        mean = features[:train_count].mean(dim=0, keepdim=True)
        std = features[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
        features = (features - mean) / std
        x_train = features[:train_count].to(args.device)
        x_dev = features[train_count:].to(args.device)
        variant = fit_and_evaluate(x_train, x_dev, targets, size_targets, references, vocab,
                                   torch, args.device, args.steps, args.batch_size, args.lr,
                                   pos_weight)
        variant["chunks"] = chunks
        results.append(variant)
        print(json.dumps(variant, ensure_ascii=False), flush=True)
        del features, x_train, x_dev
        torch.cuda.empty_cache()

    report = {
        "variants": results,
        "control_expectation": ("chunks=1 reproduces the run's meanmax representation and "
                                "should land near the known 11.9% constraint / 21.2% "
                                "size-decoder baseline; a large deviation invalidates the "
                                "comparison"),
        "reference": {"pair_model_constraint_rate": 0.0344,
                      "meanmax_linear_threshold": 0.119,
                      "meanmax_linear_size_decoder": 0.2115,
                      "meanmax_linear_oracle_size": 0.2493},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
