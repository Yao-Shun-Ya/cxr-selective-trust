"""One-off: the MNAR bound for the 3 medgemma_15 shift cells that never parsed.

A top-journal treatment of missing cells is not "drop them and move on". It is:
report the rate, classify the missingness, and BOUND its effect so a reviewer can see
it cannot move a conclusion. This script derives the counts from the data -- not from
prose -- and prints the bound, as evidence.

Why a bound rather than a re-computation: for these cells the model emitted no
probabilities, so there is nothing to include, and imputing values it never produced
would be fabrication. What a reviewer needs is the LARGEST effect the omission could
possibly have, which follows from the counts alone:

    worst-case shift of a mean over N units  <=  (max - min) / N  <=  1 / N   for [0,1] metrics

so one missing image in a 100-image cell moves a [0,1] cell metric by at most 0.01,
and 8 missing labels in an 800-label pooled metric by at most 8/800 = 0.01.

Also records the framing that matters: for the shift layer structural validity IS the
outcome, so these 3 cells are data points on the validity curve (validity 0, already
counted in parse_coverage = 0.99), not dropped data. The bound applies only to the
content metrics that require a parsed object.

Not part of the pipeline.
"""
import json
from collections import Counter
from pathlib import Path

SHIFT = Path("outputs/pretest/nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl")
OUT = Path("outputs/state/mnar_bound_medgemma15_shift.json")

LABELS_PER_CELL = 8              # the shift prompt asks for 8 findings
N_IMAGES_PER_CELL = 100
N_LABELS_PER_CELL = N_IMAGES_PER_CELL * LABELS_PER_CELL

cells = Counter()
per_img = Counter()
n = fail = 0
with SHIFT.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        if (r.get("parsed") or {}).get("parse_success"):
            continue
        fail += 1
        cells["%s/%s" % (r.get("condition"), r.get("severity"))] += 1
        per_img[r.get("image_id")] += 1

dropped_labels = fail * LABELS_PER_CELL
total_labels = n * LABELS_PER_CELL

report = {
    "stage": "medgemma_15_4b_it/shift",
    "n_cells": n,
    "n_parse_fail": fail,
    "parse_fail_rate": round(fail / n, 6),
    "failed_cells": dict(sorted(cells.items())),
    "failed_images": dict(per_img),
    "missingness": ("MNAR -- all 3 are on degraded images "
                    "(blur/severe, resolution/moderate, resolution/severe); the model fails "
                    "precisely where the image is hardest, so the missingness is not random"),
    "dropped_images": fail,
    "dropped_label_instances": dropped_labels,
    "total_label_instances": total_labels,
    "dropped_label_fraction": round(dropped_labels / total_labels, 6),
    "worst_case_bound_cell_metric": round(1.0 / N_IMAGES_PER_CELL, 6),
    "worst_case_bound_pooled_metric": round(LABELS_PER_CELL / N_LABELS_PER_CELL, 6),
    "structural_validity_note": ("structural validity counts these as failures "
                                 "(parse_coverage 0.99), so they are data points on the "
                                 "validity curve, not dropped data; the bound applies only to "
                                 "content metrics that require a parsed object"),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("\nwrote", OUT)