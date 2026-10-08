#!/usr/bin/env python
"""59 — probe the pre-run artifacts for findings worth putting in the paper.

This is a *reader*: it never writes into `outputs/pretest`. It answers three
questions that the run-level summaries do not answer on their own.

  A. Image reliance as a *fingerprint*.  The published summary reports a mean
     sensitivity per condition. That is an average over 100 images and it hides
     the strongest possible form of the same fact: does the model produce the
     *same bytes* when the image is destroyed? A model that answers from the
     text prior alone must map blank, blur and shuffle onto the same output —
     so `blank == blur` is a testable prediction, not an average.

  B. Structural validity under shift, per cell.  Layer 1's 19 cells exist to
     separate "the format held" from "the model answered". Parsing rate and
     positive-call rate per cell show where the two come apart.

  C. The ablation's prompt-robustness ordering.  Five prompts, one model, both
     conditions on disk; the contract/parse rates are already summarised, but
     the *blank-image survival* of each prompt is the interesting axis.

Usage:
    python scripts/59_probe_pretest_findings.py [--root outputs/pretest]

Exit code is 0 even when a model is broken — a broken model is a result here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

CONDITIONS = ("original", "blank", "blur", "shuffle")
PERTURBATIONS = ("blank", "blur", "shuffle")


# --------------------------------------------------------------------------- io


def read_jsonl(path):
    """Read a jsonl file, tolerating a torn final line (the driver can be killed)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # torn tail: everything after it is untrustworthy
    return rows


def prob_vector(row):
    """The 7 (status, probability) pairs a row carries, or None if unparseable.

    Returns None rather than a default vector on purpose: a row that did not
    parse has no opinion, and treating it as all-negative would manufacture
    agreement that the model never expressed.
    """
    parsed = row.get("parsed") or {}
    if not parsed.get("parse_success"):
        return None
    findings = parsed.get("findings") or {}
    if not findings:
        return None
    out = []
    for label in sorted(findings):
        item = findings[label]
        if isinstance(item, dict):
            # `probability_positive` is the schema key (src/cxr_trust/inference/
            # schema.py); the earlier `probability_pos` matched nothing, so every
            # vector compared as (label, status, None) and the `prob=` column was
            # silently a *status*-agreement column. It was never cited, but a
            # column that cannot see probabilities is worse than a missing one.
            out.append((label, str(item.get("status", "")), item.get("probability_positive")))
        else:
            out.append((label, str(item), None))
    return tuple(out)


def raw(row):
    value = row.get("raw_output")
    return value if isinstance(value, str) else ""


def keyed(rows):
    """condition -> image_id -> row (last write wins, matching the durable writer)."""
    table = defaultdict(dict)
    for row in rows:
        condition = row.get("condition")
        image_id = row.get("image_id")
        if condition and image_id:
            table[condition][image_id] = row
    return table


# ------------------------------------------------------------------ section A


def section_a(root):
    print("=" * 78)
    print("A. IMAGE RELIANCE — the blindness fingerprint")
    print("=" * 78)
    print("columns: n = paired images; text=/prob= = fraction byte/probability-identical to")
    print("         the ORIGINAL image; bb = fraction where blank and blur agree exactly")
    print()
    header = f"{'model/variant':34s} {'cond':8s} {'n':>4s} {'text=':>7s} {'prob=':>7s} {'bb':>6s}"
    print(header)
    print("-" * len(header))

    pattern = os.path.join(root, "nih_eval", "*", "reliance*", "image_reliance_predictions.jsonl")
    import glob

    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, os.path.join(root, "nih_eval"))
        name = os.path.dirname(rel).replace(os.sep, "/")
        table = keyed(read_jsonl(path))
        base = table.get("original", {})
        if not base:
            print(f"{name:34s} {'-':8s} {'0':>4s} {'-':>7s} {'-':>7s} {'-':>6s}")
            continue
        blank, blur = table.get("blank", {}), table.get("blur", {})
        shared = sorted(set(blank) & set(blur))
        bb = sum(1 for i in shared if raw(blank[i]) == raw(blur[i]) and raw(blank[i]) != "")
        for condition in PERTURBATIONS:
            other = table.get(condition, {})
            ids = sorted(set(base) & set(other))
            if not ids:
                continue
            same_text = sum(1 for i in ids if raw(base[i]) == raw(other[i]) and raw(base[i]) != "")
            usable = [(i, prob_vector(base[i]), prob_vector(other[i])) for i in ids]
            comparable = [(a, b) for _, a, b in usable if a is not None and b is not None]
            same_prob = sum(1 for a, b in comparable if a == b)
            print(
                f"{name:34s} {condition:8s} {len(ids):4d} "
                f"{same_text / len(ids):7.3f} "
                f"{(same_prob / len(comparable)) if comparable else float('nan'):7.3f} "
                f"{(bb / len(shared)) if condition == 'blank' and shared else float('nan'):6.3f}"
            )
        print()


# ------------------------------------------------------------------ section B


def section_b(root):
    print("=" * 78)
    print("B. SHIFT — structural validity per cell (layer 1)")
    print("=" * 78)
    print("parse = fraction of rows that parsed; pos = mean positive calls per row")
    print("        (of 7 labels) among parsed rows; a drop to ~0 with a high parse")
    print("        rate is the 'valid-looking all-negative' failure mode.")
    print()
    header = f"{'model':24s} {'rows':>5s} {'parse':>6s} {'pos':>6s}  worst cells by positive-call rate"
    print(header)
    print("-" * len(header))

    import glob

    for path in sorted(glob.glob(os.path.join(root, "nih_eval", "*", "shift", "shift_predictions.jsonl"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows = read_jsonl(path)
        if not rows:
            print(f"{name:24s} {0:5d} {'-':>6s} {'-':>6s}")
            continue
        parse_ok = sum(1 for r in rows if (r.get("parsed") or {}).get("parse_success"))
        positives, cells = [], defaultdict(lambda: [0, 0, 0])  # parsed, positive, total
        for row in rows:
            vector = prob_vector(row)
            cell = f"{row.get('condition')}/{row.get('severity')}"
            cells[cell][2] += 1
            if vector is None:
                continue
            cells[cell][0] += 1
            n_pos = sum(1 for _, status, _ in vector if status == "positive")
            cells[cell][1] += n_pos
            positives.append(n_pos)
        mean_pos = (sum(positives) / len(positives)) if positives else float("nan")
        ranked = sorted(
            ((c, v) for c, v in cells.items() if v[0]),
            key=lambda kv: kv[1][1] / kv[1][0],
        )[:3]
        worst = "  ".join(f"{c}:{v[1] / v[0]:.2f}(n={v[0]})" for c, v in ranked)
        print(f"{name:24s} {len(rows):5d} {parse_ok / len(rows):6.3f} {mean_pos:6.2f}  {worst}")
    print()


# ------------------------------------------------------------------ section C


def section_c(root):
    print("=" * 78)
    print("C. ABLATION — does the prompt survive an image with no information?")
    print("=" * 78)
    path = os.path.join(root, "nih_eval", "medgemma_4b_it", "ablation", "ablation_run_summary.json")
    if not os.path.isfile(path):
        print("(ablation summary not found)")
        return
    data = json.load(open(path, encoding="utf-8"))
    by_variant = defaultdict(dict)
    for cell in data.get("cells", []):
        by_variant[cell["variant"]][cell["condition"]] = cell
    print(f"{'prompt variant':24s} {'parse orig':>10s} {'parse blank':>11s} "
          f"{'viol orig':>9s} {'viol blank':>10s} {'miss orig':>9s} {'miss blank':>10s}")
    print("-" * 88)
    for variant in sorted(by_variant):
        cells = by_variant[variant]
        orig, blank = cells.get("original", {}), cells.get("blank", {})
        print(
            f"{variant:24s} "
            f"{orig.get('parse_failure_rate', float('nan')):10.3f} "
            f"{blank.get('parse_failure_rate', float('nan')):11.3f} "
            f"{orig.get('contract_violation_rate', float('nan')):9.3f} "
            f"{blank.get('contract_violation_rate', float('nan')):10.3f} "
            f"{orig.get('mean_probs_missing', float('nan')):9.2f} "
            f"{blank.get('mean_probs_missing', float('nan')):10.2f}"
        )
    print()
    print("miss = mean label probabilities NOT emitted per row (of 7); it is the")
    print("cleanest measure of a prompt losing the ability to answer at all.")
    print()
    print("--- blank-vs-original text identity, per variant (byte level) ---")
    print("each condition lives in its own file, so the pair is built across files;")
    print("identical_text is over images where BOTH sides produced non-empty text.")
    base = os.path.join(root, "nih_eval", "medgemma_4b_it", "ablation")
    for variant in sorted(by_variant):
        orig = {r["image_id"]: r for r in read_jsonl(
            os.path.join(base, variant, "original", "image_reliance_predictions.jsonl")) if r.get("image_id")}
        blk = {r["image_id"]: r for r in read_jsonl(
            os.path.join(base, variant, "blank", "image_reliance_predictions.jsonl")) if r.get("image_id")}
        ids = [i for i in sorted(set(orig) & set(blk))
               if raw(orig[i]).strip() and raw(blk[i]).strip()]
        if not ids:
            print(f"{variant:24s} (no image produced text on both sides)")
            continue
        same = sum(1 for i in ids if raw(orig[i]) == raw(blk[i]))
        print(f"{variant:24s} informative_pairs={len(ids):4d} identical_text={same / len(ids):6.3f}")
    print()


def section_d(root):
    """Per-cell parse matrix for the shift stage.

    Section B ranks cells by positive-call rate. This section asks the other
    question: is the *structural* failure uniform across the 19 cells, or does
    it cluster in particular corruptions? Those are different claims and only
    the second one supports a "loss of visual information breaks the format"
    story.
    """
    print("=" * 78)
    print("D. SHIFT — parse rate per cell (structural failure is not uniform)")
    print("=" * 78)
    import glob

    for path in sorted(glob.glob(os.path.join(root, "nih_eval", "*", "shift", "shift_predictions.jsonl"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows = read_jsonl(path)
        if not rows:
            continue
        cells = defaultdict(lambda: [0, 0])
        for row in rows:
            cell = f"{row.get('condition')}/{row.get('severity')}"
            cells[cell][1] += 1
            if (row.get("parsed") or {}).get("parse_success"):
                cells[cell][0] += 1
        overall = sum(v[0] for v in cells.values()) / sum(v[1] for v in cells.values())
        if overall >= 0.999 and len({v[0] for v in cells.values()}) == 1:
            print(f"{name}: 100% parse in all {len(cells)} cells — no cell-level collapse")
            continue
        print(f"{name}: overall {overall:.3f}, {len(cells)} cells; cells below 95%:")
        ranked = sorted(cells.items(), key=lambda kv: kv[1][0] / kv[1][1])
        for cell, (ok, total) in ranked:
            if ok / total >= 0.95:
                break
            print(f"    {cell:28s} {ok:4d}/{total:<4d} = {ok / total:.3f}")
        print()
    print()


THINK_MARKERS = ("<unused94>thought", "<unused95>", "thought\n", "Thinking Process", "<|think|>")
SEVERITY_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}


def section_e(root):
    """Test the mechanism hypothesis on data we already have.

    Hypothesis: the parse failures are not a parser problem — the model switches
    into a free-form reasoning channel, and the switch is driven by how much the
    image carries. If that is true then (i) the failing rows must contain a
    reasoning marker, and (ii) the models that never collapse must never emit
    one. Both are measurable without another forward pass.
    """
    print("=" * 78)
    print("E. MECHANISM — is the failure a reasoning-channel switch?")
    print("=" * 78)
    import glob

    print("(i) reasoning markers in the RAW text, per model/stage")
    print(f"{'model/stage':40s} {'rows':>6s} {'marker':>7s} {'unparsed':>9s} {'unparsed&marker':>16s}")
    print("-" * 82)
    for path in sorted(glob.glob(os.path.join(root, "nih_eval", "*", "*", "*predictions.jsonl"))):
        rel = os.path.relpath(path, os.path.join(root, "nih_eval")).replace(os.sep, "/")
        rows = read_jsonl(path)
        if not rows:
            continue
        marked = sum(1 for r in rows if any(m in raw(r) for m in THINK_MARKERS))
        bad = [r for r in rows if not (r.get("parsed") or {}).get("parse_success")]
        both = sum(1 for r in bad if any(m in raw(r) for m in THINK_MARKERS))
        print(f"{rel.replace('/' + os.path.basename(path), ''):40s} {len(rows):6d} "
              f"{marked:7d} {len(bad):9d} {both:16d}")
    print()

    print("(ii) parse rate by severity, pooled over conditions (all 19 cells)")
    print("     ordinal severity 0-3; a monotone fall is the 'information gradient' claim.")
    print(f"{'model':26s} " + " ".join(f"{'sev' + str(v):>14s}" for v in range(4)))
    print("-" * 88)
    for path in sorted(glob.glob(os.path.join(root, "nih_eval", "*", "shift", "shift_predictions.jsonl"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows = read_jsonl(path)
        if not rows:
            continue
        buckets = defaultdict(lambda: [0, 0])
        for row in rows:
            level = SEVERITY_ORDER.get(str(row.get("severity")))
            if level is None:
                continue
            cell = buckets[level]
            cell[1] += 1
            if (row.get("parsed") or {}).get("parse_success"):
                cell[0] += 1
        cells = []
        for level in range(4):
            ok, total = buckets.get(level, (0, 0))
            cells.append(f"{(ok / total):.3f}({total})" if total else "-")
        print(f"{name:26s} " + " ".join(f"{c:>14s}" for c in cells))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("outputs", "pretest"))
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2
    section_a(args.root)
    section_b(args.root)
    section_c(args.root)
    section_d(args.root)
    section_e(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
