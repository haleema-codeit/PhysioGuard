"""Actual dataset evaluation runner.

Example:
    python -m evaluation.evaluate --root . --max-per-class 10 --held-out-source Sora
"""

import argparse
import json
from pathlib import Path

from evaluation.dataset_loader import discover_dataset, split_known_unseen
from evaluation.metrics import binary_metrics
from modules.pipeline import analyze_video


def score_item(result, mode):
    visual = result["fusion"]["visual_manipulation"]
    phys = result["fusion"]["physiological_inconsistency"]
    final = result["fusion"]["final_evidence"]
    reliability = result["fusion"]["signal_reliability"]
    if mode == "visual-only": return visual
    if mode == "rppg-only": return phys if reliability >= 40 else 50.0
    if mode == "visual+rppg": return 0.5 * visual + 0.5 * phys
    return final


def evaluate(items, modes):
    records = []
    for index, item in enumerate(items, 1):
        try:
            result = analyze_video(item.path, max_frames=240)
            records.append({"path": item.path, "label": item.label, "source": item.dataset_source,
                            "result": result})
            print(f"[{index}/{len(items)}] {Path(item.path).name}")
        except Exception as exc:
            print(f"SKIP {item.path}: {exc}")
    output = {mode: binary_metrics([x["label"] for x in records],
                                   [score_item(x["result"], mode) for x in records]) for mode in modes}
    return output, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--held-out-source", default=None)
    parser.add_argument("--out", default="evaluation_results.json")
    args = parser.parse_args()
    items = discover_dataset(args.root, args.max_per_class)
    known, unseen, held_out = split_known_unseen(items, args.held_out_source)
    modes = ["visual-only", "rppg-only", "visual+rppg", "full-physioguard"]
    known_metrics, known_records = evaluate(known, modes)
    unseen_metrics, unseen_records = evaluate(unseen, modes)
    payload = {"dataset_root": str(Path(args.root).resolve()), "held_out_source": held_out,
               "known_source": {"metrics": known_metrics, "evaluated": len(known_records)},
               "unseen_source": {"metrics": unseen_metrics, "evaluated": len(unseen_records)},
               "note": "Metrics are computed from the videos successfully decoded by the configured pipeline."}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

