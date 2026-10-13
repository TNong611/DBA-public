"""Is the MARKER the reason constraint exact match stays near zero?

`meanmax` pooling collapses each case's [72, 2048] event sequence to one 4096-vector by
taking a mean and a max. The constraint's four markers -- 本轮用户条件 / 本轮条件 /
沿用 / 沿用已告知条件 -- describe WHEN a condition was introduced, so they are inherently
temporal. Pooling may erase exactly the information that decides them.

That hypothesis makes a sharp prediction: if the model gets the (attribute, value) pairs
right and only the marker wrong, then dropping the marker should raise exact match a lot.

This reads an existing dev dump, so it is instant and needs no GPU. It also separates the
two failure shapes:
  * pairs right, marker wrong  -> temporal information lost in pooling; the fix is a head
    that sees the sequence, not a better pooled head;
  * pairs wrong               -> the pooled representation itself is insufficient.

Writes nothing; prints a table.
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


def parse_predicted_constraint(text):
    """The dump stores rendered canonical text; recover the triple set from it."""
    for line in text.split("\n"):
        if line.startswith("约束："):
            body = line[3:]
            break
    else:
        return None
    if body.strip() in ("", "无"):
        return frozenset()
    return frozenset(P.constraint_triples(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args(argv)

    rows = []
    with open(args.predictions, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    totals = collections.Counter()
    pair_right_marker_wrong = 0
    pair_wrong = 0
    marker_only_cases = []
    for row in rows:
        predicted = parse_predicted_constraint(row["prediction_canonical"])
        reference = row["reference_raw"]["constraint"]
        gold = frozenset(P.constraint_triples(reference)) if reference.strip() not in ("", "无") \
            else frozenset()
        if predicted is None:
            continue
        totals["cases"] += 1
        full = predicted == gold
        totals["full_exact"] += int(full)
        pred_pairs = frozenset((attribute, value) for _, attribute, value in predicted)
        gold_pairs = frozenset((attribute, value) for _, attribute, value in gold)
        pairs = pred_pairs == gold_pairs
        totals["pairs_exact_ignoring_marker"] += int(pairs)
        if full:
            continue
        if pairs:
            pair_right_marker_wrong += 1
            if len(marker_only_cases) < 5:
                marker_only_cases.append({
                    "id": row["id"],
                    "gold": sorted(gold),
                    "predicted": sorted(predicted),
                })
        else:
            pair_wrong += 1

    # What would a marker-free task score? Compare (attribute, value) sets only, which is
    # the task that survives pooling if the temporal marker is what pooling destroys.
    report = {
        "cases": totals["cases"],
        "full_triple_exact": totals["full_exact"],
        "full_triple_exact_rate": totals["full_exact"] / max(1, totals["cases"]),
        "marker_ignored_exact": totals["pairs_exact_ignoring_marker"],
        "marker_ignored_exact_rate": (totals["pairs_exact_ignoring_marker"]
                                      / max(1, totals["cases"])),
        "failures_where_only_marker_differs": pair_right_marker_wrong,
        "failures_where_pairs_differ": pair_wrong,
        "share_of_failures_that_are_marker_only": (
            pair_right_marker_wrong / max(1, pair_right_marker_wrong + pair_wrong)),
        "examples_marker_only": marker_only_cases,
        "reading": ("if marker_ignored_exact_rate is far above full_triple_exact_rate, the "
                    "temporal marker is the bottleneck and a sequence-aware head is the fix"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
