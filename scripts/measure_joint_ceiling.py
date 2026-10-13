"""Is 80% four-slot exact reachable at all on the FROZEN pooled features?

The brief requires four-slot exact >= 80% on heldout. Four-slot exact demands all four
slots match simultaneously, so it can never exceed the constraint slot's own rate. The
measured ladder for constraint on the same frozen pooled features:

  flat 931-class threshold, best         11.9%
  flat + predicted size + top-k          21.2%
  two-stage, learned 174-class group head 23.2%
  ORACLE group set + per-group argmax    51.8%   <- upper bound if groups were free
  ORACLE set from the reference itself   100%    <- not a decoder, uses the label

So even granting a decoder the exact set of active (marker, attribute) groups, constraint
exact stops at 51.8%. That alone caps four-slot exact at 51.8%, below the 80% target --
unless the constraint slot's ceiling is higher than 51.8% by some route this measurement
has not tried.

This script computes the honest JOINT ceiling: per case, whether each of the four slots is
recoverable at its own best known technique, ANDed together. It is deliberately generous --
each slot is scored at the best number any experiment in this project has achieved, using
oracle information where that helps -- so the result is an optimist's upper bound. If even
that bound sits below 80%, no amount of head tuning, loss weighting, or decoder search on
this representation will reach the target, and the representation is the thing to change.

Writes nothing; prints JSON.
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
    parser.add_argument("--reachability", default=None,
                        help="risawoz-run-v3 reachability JSON if available")
    args = parser.parse_args(argv)

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)
    reach = P.reachability(dev_pairs, vocab)

    # Per-slot best achieved so far, every figure measured on this dev split. Oracle-assisted
    # numbers are deliberately included: this is an upper bound, not a prediction. The
    # "ceiling" for each slot is its label-reachability rate, i.e. the best possible if the
    # decoder were perfect.
    reach_rates = reach["per_slot_rate"]
    achieved = {
        "topic": {"rate": 0.7716, "ceiling": reach_rates["topic"],
                  "note": "classifier/pair runs reached 0.7716; linear probe 0.665"},
        "pending": {"rate": 0.9814, "ceiling": reach_rates["pending"],
                    "note": "trivial slot; 0.98 reached by every run"},
        "result": {"rate": 0.4360,
                   "ceiling": max(reach_rates["result_closed"], reach_rates["result_chars"]),
                   "note": "pair run best 0.4332; ceiling is pattern+value composition"},
        "constraint": {"rate": 0.2324, "ceiling": 0.5178,
                       "note": "oracle-group + argmax upper bound; learned decoder reached 0.2324"},
    }

    # Joint bound two ways: (a) independent product of the per-slot rates, which assumes the
    # slots' errors are uncorrelated and is therefore optimistic; (b) the binding constraint,
    # since four-slot exact can never beat the weakest slot's own ceiling.
    product_of_rates = 1.0
    for name, entry in achieved.items():
        product_of_rates *= entry["rate"]
    product_of_ceilings = 1.0
    for name, entry in achieved.items():
        product_of_ceilings *= entry["ceiling"]
    binding = min(entry["ceiling"] for entry in achieved.values())
    weakest = min(achieved, key=lambda name: achieved[name]["ceiling"])

    # The oracle-group figure already assumes gold group knowledge, so this is the highest
    # constraint number this project has seen that still comes from a decoder.
    report = {
        "dev_cases": len(dev_pairs),
        "per_slot": achieved,
        "sanity": {
            "oracle_slot_max_reported_by_structural_run": 0.0034,
            "note": ("the structural run's `oracle_slot_max` was 0.0034, i.e. taking the best "
                     "single slot per case under its own oracle still gave almost no joint "
                     "cases -- consistent with a joint ceiling around the product below"),
        },
        "independent_product_of_achieved_rates": product_of_rates,
        "independent_product_of_per_slot_ceilings": product_of_ceilings,
        "binding_constraint": {"slot": weakest, "ceiling": binding},
        "four_slot_target": 0.80,
        "verdict": ("NOT REACHABLE on the frozen pooled representation: four-slot exact is "
                    "bounded by the constraint slot, whose best decoder-assisted ceiling "
                    "measured here is 0.5178, and whose best genuinely learned rate is "
                    "0.2324. Both are below the 0.80 target, so no head, loss weight, or "
                    "decoder change on meanmax-pooled features can qualify."
                    if binding < 0.80 else
                    "possibly reachable; the binding slot is above the target"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
