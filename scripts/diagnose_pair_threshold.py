"""Is the constraint failure a calibration problem or a representation problem?

The pair run ended with `triple` loss 0.005 and 64% of predicted constraints EMPTY,
while the reference is empty in only 32/1773 cases. That combination is the signature of
a sparse multi-label collapse: positive rate is 0.2299% (931 classes, ~2.14 positives per
case), so an all-negative output already scores 0.023 BCE, and a fixed 0.5 threshold on
sigmoid essentially never fires at that base rate.

Those two explanations need opposite fixes, and they look identical in the headline
number, so this script separates them WITHOUT retraining:

  * calibration   -> sweeping the decision threshold should recover most of the gap;
  * representation-> no threshold helps, because the positives are not ranked above the
                     negatives.

It re-reads the SAME sealed-off dev split and the SAME checkpoint, and touches neither
`memory_test.jsonl` nor any training artifact. It writes nothing: it prints a table.

Reading the result:
  * if the best threshold is far below 0.5 and lifts constraint accuracy by a lot, the
    fix is a calibrated loss (per-class pos_weight) plus a threshold chosen on dev;
  * if constraint accuracy stays near the floor at every threshold, the state does not
    rank the right triples and the lever is the label space, not the loss.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pool", default="meanmax")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002")
    parser.add_argument("--top-k", default="3,4,5,6")
    args = parser.parse_args(argv)

    torch = P.base._torch()
    if not torch.cuda.is_available():
        raise SystemExit("this diagnostic needs the same CUDA path the run used")
    device = "cuda"

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)

    loaded = P.load_checkpoint(Path(args.checkpoint), torch)
    config = P.PairModelConfig(pooled_dim=4096, d_state=512, vocab=vocab, d_char=192,
                               dropout=0.0)
    model = P.build_model(config, torch).to(device)
    model.load_state_dict(loaded["model_state"], strict=True)
    model.eval()

    # The cache verifies its own identity against the dataset hash and refuses to be
    # read otherwise, which is why the hash has to be passed rather than assumed. The
    # expected value comes from the training run's own report so this cannot silently
    # disagree with what the checkpoint was trained on.
    run_report = json.loads(
        (Path(args.checkpoint).parent / "dev-report.json").read_text(encoding="utf-8"))
    cache_identity = run_report["cache_identity"]
    cache = P.ReadOnlyFeatureCache(Path(args.feature_cache), cache_identity, None,
                                   allow_any_backbone=True)
    cache.open()
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    pooled = P.build_pooled(cache, ordered, args.pool)
    train_count = len(train_cases)
    # Standardise with TRAIN-only statistics, exactly as training did: reusing the
    # run's own transform is the only way the logits here mean what they meant there.
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    dev_pooled = ((pooled[train_count:] - mean) / std).to(device)
    references = [dict(entry) for entry in dev_pairs]

    # One forward pass over dev, keeping the raw triple logits.
    all_logits = []
    with torch.no_grad():
        for start in range(0, int(dev_pooled.shape[0]), args.eval_batch_size):
            block = dev_pooled[start:start + args.eval_batch_size]
            outputs = model(block)
            all_logits.append(outputs["triple"].float().cpu())
    logits = torch.cat(all_logits, 0)
    probs = torch.sigmoid(logits)

    gold = P.encode_targets(dev_pairs, vocab, torch).multi
    scores = torch.sigmoid(logits)

    report = {"cases": len(dev_pairs), "classes": int(logits.shape[1]), "thresholds": []}

    for raw in args.thresholds.split(","):
        threshold = float(raw)
        mask = probs >= threshold
        exact = 0
        sizes = []
        for row in range(mask.shape[0]):
            predicted = frozenset(
                vocab.triples.decode(position)
                for position in mask[row].nonzero(as_tuple=False).flatten().tolist())
            sizes.append(len(predicted))
            if predicted == references[row]["constraint"]:
                exact += 1
        report["thresholds"].append({
            "threshold": threshold,
            "constraint_exact": exact,
            "constraint_exact_rate": exact / len(dev_pairs),
            "mean_predicted_size": sum(sizes) / len(sizes),
        })

    # Top-k variant: is the right triple at least RANKED highly, even if the score is
    # too low to pass a threshold? This is what distinguishes calibration from ranking.
    report["top_k"] = []
    for raw in args.top_k.split(","):
        k = int(raw)
        predicted_sets = []
        for row in range(scores.shape[0]):
            top = torch.topk(scores[row], k).indices.tolist()
            predicted_sets.append(frozenset(vocab.triples.decode(p) for p in top))
        exact = sum(1 for row, predicted in enumerate(predicted_sets)
                    if predicted == references[row]["constraint"])
        report["top_k"].append({"k": k, "constraint_exact": exact,
                                "constraint_exact_rate": exact / len(dev_pairs)})

    # Ranking quality without any threshold: how many of the gold positives are in the
    # top-|gold| scores, averaged over cases.
    recall_sum, coverage = 0.0, 0
    for row in range(scores.shape[0]):
        gold_positions = gold[row].nonzero(as_tuple=False).flatten().tolist()
        if not gold_positions:
            continue
        k = len(gold_positions)
        top = set(torch.topk(scores[row], k).indices.tolist())
        recall_sum += len(top & set(gold_positions)) / k
        coverage += 1
    report["oracle_topk_recall"] = recall_sum / max(1, coverage)
    report["oracle_topk_cases"] = coverage

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
