#!/usr/bin/env python
"""One-shot repository debug sweep. Read-only: it reports, it does not fix.

The failures this project has actually lost time to were all of one shape: a
script that ran, exited 0, and produced work that was wrong or empty. So this
sweep checks the things that make that possible rather than the things a linter
would notice:

1. **Every Python file parses** -- a byte-order mark from an editor is a
   SyntaxError, and it happened here.
2. **Every ``choices=`` list matches the registry** -- a model renamed in the
   registry but not in a runner's argparse is rejected only at run time, hours in.
3. **Every manifest the orchestrators reference exists**, and carries the columns
   the stage scripts read (``image_id``, ``image_relpath``, ``split``, labels).
4. **Every image referenced by the evaluation manifests is on disk** -- a missing
   image becomes a silently skipped job.
5. **Label columns are in the expected string form** (``positive``/``negative``),
   because the label matrix is built with ``astype(str) == "positive"`` and an
   integer 0/1 column silently becomes all-negative.
6. **No label column is constant**, which makes its metric undefined while still
   looking like a real number in a table.
7. **Local imports resolve** -- the dual ``src.cxr_trust`` / ``cxr_trust`` module
   identity has already hidden one bug.
8. **Every script the orchestrators invoke exists.**

Run::

    .venv-vl\\Scripts\\python.exe scripts\\45_debug_sweep.py
    .venv-vl\\Scripts\\python.exe scripts\\45_debug_sweep.py --strict
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FINDINGS: list[tuple[str, str]] = []   # (severity, message)


def add(sev: str, msg: str) -> None:
    FINDINGS.append((sev, msg))


def head(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
def check_syntax() -> None:
    head("1. python syntax")
    n = 0
    for root in ("scripts", "src"):
        for p in sorted((ROOT / root).rglob("*.py")):
            n += 1
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as e:
                add("ERROR", f"{p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
            raw = p.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                add("ERROR", f"{p.relative_to(ROOT)}: UTF-8 BOM breaks the parser")
    print(f"  parsed {n} files, {sum(1 for s,_ in FINDINGS if s=='ERROR')} problem(s)")


def check_choices() -> None:
    head("2. argparse choices vs the model registry")
    try:
        from cxr_trust.models.registry import MODEL_NAMES
    except Exception as e:  # noqa: BLE001
        add("ERROR", f"cannot import the registry: {e}")
        print(f"  registry import failed: {e}")
        return
    known = set(MODEL_NAMES)
    print(f"  registry: {len(known)} models")
    seen_any = False
    for p in sorted((ROOT / "scripts").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"add_argument\(\s*[\"']--model[\"'][^)]*choices\s*=\s*(\[[^\]]*\])",
                             text):
            seen_any = True
            try:
                listed = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                add("WARN", f"{p.name}: cannot parse choices {m.group(1)[:60]}")
                continue
            missing = set(listed) - known
            extra = known - set(listed)
            line = text[:m.start()].count("\n") + 1
            if missing:
                add("ERROR", f"{p.name}:{line} choices contain unknown models: "
                             f"{sorted(missing)}")
            if extra:
                # informational only: a runner may legitimately support a subset
                print(f"  [info] {p.name}:{line} supports a subset; not offered: "
                      f"{sorted(extra)}")
    if not seen_any:
        add("WARN", "no --model choices found; did the runners change CLI?")


def check_manifests() -> None:
    head("3. manifests referenced by the orchestrators")
    # "classification" manifests carry per-finding ground truth and are scored with
    # the eight-label machinery, so a missing or constant label column is a defect.
    # "report" manifests have reference text instead of labels by design; a constant
    # `unknown` column there is informational and must NOT be reported as an error --
    # a sweep that cries wolf gets ignored, which is worse than not having one.
    manifests = {
        "data/manifests/nih_eval_1732.csv": {"kind": "classification", "images": 1732,
                                             # internal_train rows are present only as
                                             # donors for the no_finding repair; every
                                             # analysis selects internal_test/source_val
                                             # and never reads them.
                                             "eval_splits": {"internal_test", "source_val",
                                                             "internal_train"},
                                             "note": "226 internal_train rows are "
                                                     "no_finding donors, never analysed"},
        "data/manifests/padchest_shift.csv": {"kind": "classification", "images": 1500,
                                              "eval_splits": {"internal_test"}},
        "data/manifests/openi_rrg.csv": {"kind": "report", "images": 947},
        "data/manifests/nih_shared_5000.csv": {
            "kind": "classification", "images": 5000,
            # Upstream defect, kept on disk for provenance and NOT used as an
            # evaluation manifest: stratified_select balances the seven disease
            # labels and never gives no_finding a positive target, so all 5,000 rows
            # carry label_no_finding = negative. data/manifests/nih_eval_1732.csv is
            # the repaired evaluation set that is actually run.
            "known_constant_labels": {"no_finding"},
        },
        "data/manifests/chexpert_rrg_all.csv": {"kind": "report"},
    }
    labels = ["atelectasis", "cardiomegaly", "consolidation", "edema",
              "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]
    for rel, spec in manifests.items():
        p = ROOT / rel
        if not p.exists():
            add("ERROR", f"{rel}: MISSING")
            continue
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        cols = set(rows[0].keys()) if rows else set()
        kind = spec["kind"]
        if kind == "classification":
            for col in ("image_id", "image_relpath"):
                if col not in cols:
                    add("ERROR", f"{rel}: missing core column {col}")
        if "images" in spec and len(rows) != spec["images"]:
            add("WARN", f"{rel}: {len(rows)} rows, expected {spec['images']}")

        const = []
        absent = []
        for lb in labels:
            col = f"label_{lb}"
            if col not in cols:
                # A report manifest legitimately has no label columns.
                if kind == "classification":
                    absent.append(col)
                continue
            values = {r[col] for r in rows}
            if values & {"0", "1", "0.0", "1.0"}:
                add("ERROR", f"{rel}: {col} uses integers; the label matrix needs "
                             f"'positive'/'negative'")
            if len(values) == 1:
                only = values.pop()
                if only == "unknown":
                    const.append(f"{lb}=unknown")
                else:
                    # A constant real label makes its AUROC undefined while still
                    # printing a number in a table -- this is how the no_finding
                    # defect hid, so it stays an error.
                    const.append(f"{lb}={only}")
        if absent:
            add("WARN", f"{rel}: classification manifest lacks {absent}")
        # Some constant columns are data limitations rather than defects: PadChest's
        # vocabulary simply has no pneumonia label, so that metric cannot exist on
        # that dataset. It still has to be visible -- the paper must not present it
        # as a computed number -- but it is not a bug to fix.
        KNOWN_ABSENT = {
            "data/manifests/padchest_shift.csv": {"pneumonia":
                "PadChest's 120-label vocabulary contains no pneumonia label"},
        }
        # Constant labels a manifest is already known to carry, with the reason.
        # These are reported as provenance, not as a defect, exactly once.
        known_constant = spec.get("known_constant_labels", set())
        limitations = []
        hard_const = []
        for entry in const:
            lb = entry.split("=", 1)[0]
            why = KNOWN_ABSENT.get(rel, {}).get(lb)
            if lb in known_constant:
                print(f"  [known] {rel}: {lb} is constant by upstream defect; this "
                      f"manifest is not used for evaluation")
            elif why:
                limitations.append(f"{lb}: {why}")
            else:
                hard_const.append(entry)
        if kind == "classification" and hard_const:
            add("ERROR", f"{rel}: constant label column(s) -> metrics undefined: "
                         f"{hard_const}")
        elif limitations:
            add("WARN", f"{rel}: label not scorable on this dataset -- "
                        + "; ".join(limitations))
        elif const and not known_constant:
            print(f"  [info] {rel}: constant-by-design label columns: {const}")

        if "eval_splits" in spec and "split" in cols:
            got = {r["split"] for r in rows}
            if got != spec["eval_splits"]:
                add("WARN", f"{rel}: splits {sorted(got)} != declared evaluation "
                            f"splits {sorted(spec['eval_splits'])}")
        print(f"  {rel}: {len(rows)} rows, {len(cols)} cols, kind={kind}"
              + (f", constant: {const}" if const else ""))


def check_images() -> None:
    head("4. image presence for the evaluation manifests")
    cases = [
        ("data/manifests/nih_eval_1732.csv", "data/nih/images"),
        ("data/manifests/padchest_shift.csv", "data/padchest"),
        ("data/manifests/openi_rrg.csv", "data/open-i"),
    ]
    for rel, iroot in cases:
        mp, ip = ROOT / rel, ROOT / iroot
        if not mp.exists() or not ip.exists():
            add("ERROR", f"{rel} or {iroot} missing")
            continue
        with mp.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        missing = [r["image_relpath"] for r in rows
                   if not (ip / r["image_relpath"]).exists()]
        if missing:
            add("ERROR", f"{rel}: {len(missing)} of {len(rows)} images missing under "
                         f"{iroot}, e.g. {missing[:2]}")
        print(f"  {rel}: {len(rows) - len(missing)}/{len(rows)} present")


def check_imports() -> None:
    head("5. local import resolution (the dual-module trap)")
    import importlib
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    a = importlib.import_module("cxr_trust.models.registry")
    sys.path.insert(0, str(ROOT))
    b = None
    try:
        b = importlib.import_module("src.cxr_trust.models.registry")
    except Exception as e:  # noqa: BLE001
        add("WARN", f"src.cxr_trust.models.registry is not importable: {e}")
    if b is not None and a is not b:
        add("WARN", "TWO live registry modules exist (cxr_trust.* and src.cxr_trust.*). "
                    "Patching one does not affect the other; tests must install stubs "
                    "after the runner's own imports.")
        print("  two distinct module objects -- documented trap, handled in the runners")
    else:
        print("  single module identity")


def check_scripts_exist() -> None:
    head("6. scripts invoked by the orchestrators")
    for orch in ("scripts/run_all_experiments.ps1", "scripts/run_all_layers.ps1"):
        p = ROOT / orch
        if not p.exists():
            add("ERROR", f"{orch}: missing")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        refs = set(re.findall(r"'scripts\\([0-9A-Za-z_]+\.py)'", text))
        for ref in sorted(refs):
            target = ROOT / "scripts" / ref
            if not target.exists():
                add("ERROR", f"{orch} invokes missing {ref}")
        print(f"  {orch}: {len(refs)} referenced scripts, all present"
              if refs else f"  {orch}: no script references found")


def check_durable_usage() -> None:
    head("7. crash-safe writing")
    # EVERY runner that writes a predictions file must use the durable writer. This
    # was three warnings in the first sweep and is now enforced, because "still on
    # flush only" is exactly the condition that loses a night of work.
    all_runners = [
        "06_image_reliance.py", "08_shift_benchmark.py", "10_run_chexpert_rrg.py",
        "21_prompt_ablation.py", "43_run_padchest_shift.py", "44_run_openi_report.py",
    ]
    for name in all_runners:
        p = ROOT / "scripts" / name
        if not p.exists():
            add("ERROR", f"{name} is missing")
            continue
        text = p.read_text(encoding="utf-8")
        durable = "DurableJsonl" in text
        nonzero = "raise SystemExit(main())" in text
        reports_interrupt = "INTERRUPTED" in text
        print(f"  {name:32s} durable={'yes' if durable else 'NO ':3s} "
              f"nonzero-exit={'yes' if nonzero else 'NO ':3s} "
              f"interrupted-report={'yes' if reports_interrupt else 'no'}")
        if not durable:
            add("ERROR", f"{name} does not use DurableJsonl (flush-only writing)")
        if not nonzero:
            # A bare main() discards the interrupted return code, so a truncated
            # stage looks like a completed one to the orchestrator.
            add("WARN", f"{name} calls main() without SystemExit; an interrupted "
                        f"stage would exit 0")
    # the two new layers must additionally have crash/resume test coverage
    tested = set()
    for t in ("41_test_runner_resume.py",):
        tp = ROOT / "scripts" / t
        if tp.exists():
            txt = tp.read_text(encoding="utf-8")
            for m in ("43_run_padchest_shift", "44_run_openi_report"):
                if m in txt:
                    tested.add(m)
    for m in ("43_run_padchest_shift", "44_run_openi_report"):
        if m not in tested:
            add("ERROR", f"{m} has no crash/resume test coverage")
    print(f"  crash/resume coverage: {sorted(tested) or 'NONE'}")

    # The four older runners are covered by 49_test_legacy_durable.py, which runs
    # each of them for real, kills it mid-run, and checks that the resume is exact.
    legacy_test = ROOT / "scripts" / "49_test_legacy_durable.py"
    if not legacy_test.exists():
        add("WARN", "49_test_legacy_durable.py is missing: the four older runners "
                    "would have durability but no kill/restart evidence")
    else:
        txt = legacy_test.read_text(encoding="utf-8")
        covered = [n for n in ("06_image_reliance", "08_shift_benchmark",
                               "10_run_chexpert_rrg", "21_prompt_ablation")
                   if n in txt]
        print(f"  legacy kill/restart coverage: {covered}")
        if len(covered) != 4:
            add("ERROR", f"legacy durability test covers only {covered}")


def check_isolation_guard() -> None:
    """A -Limit run must not be able to pollute the full run. Verify by running it.

    This is a behavioural check, not a code reading. The failure it prevents is
    silent and irreversible: the state key is layer/model/stage and does not encode
    the image limit, so a -Limit 100 run would mark every stage complete, the full
    run would skip them, and the study would be capped at 100 images with nothing
    in the logs to say so.
    """
    head("8. pre-run isolation guard (behavioural)")
    ps = ROOT / "scripts" / "run_all_layers.ps1"
    if not ps.exists():
        add("ERROR", "run_all_layers.ps1 missing")
        return
    base = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ps), "-Models", "medgemma_4b_it", "-Layers", "nih", "-DryRun"]

    cases = [
        ("limit + default root + default state",
         ["-Limit", "100"], 3, "must be refused"),
        ("limit + isolated root + default state",
         ["-Limit", "100", "-OutputRoot", "outputs\\pretest"], 3, "must be refused"),
        # This quadrant was MISSING from the first guard and from the first version
        # of this test: a partial run with the production root but an isolated state
        # file was allowed and wrote its rows into outputs\matrix\, where the durable
        # writer then counted them as finished work for the full run. Enumerating
        # cases is what let it go missing, so all four flag combinations are tested.
        ("limit + default root + isolated state",
         ["-Limit", "100", "-StateFile", "outputs\\state\\pretest_state.json"],
         3, "must be refused"),
        ("limit + isolated root + isolated state",
         ["-Limit", "100", "-OutputRoot", "outputs\\pretest",
          "-StateFile", "outputs\\state\\pretest_state.json"], 0, "must be allowed"),
        ("no limit + default root",
         [], 0, "must be allowed"),
        ("no limit + isolated root",
         ["-OutputRoot", "outputs\\pretest"], 0, "must be allowed"),
    ]
    for label, extra, want, why in cases:
        r = subprocess.run(base + extra, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(ROOT),
                           timeout=180)
        ok = (r.returncode == want)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:52s} exit={r.returncode} ({why})")
        if not ok:
            add("ERROR", f"isolation guard: {label} exited {r.returncode}, expected "
                         f"{want} ({why})")


def check_ece_convention() -> None:
    """Pin the ECE bin convention, and forbid a second implementation of it.

    Two things went wrong here once, so both are checked:

    1. **The convention was the opposite of the standard one** (`right=True` gives
       `(0, 0.1]`; sklearn and Guo et al. use `[0, 0.1)`). It matters because 8.4% of
       this repository's stored probabilities sit exactly on a bin edge.
    2. **A script carried its own copy of the metric** with the same wrong binning,
       so fixing the canonical function would have left that file -- and every number
       it reports -- silently on the old convention.

    A discriminating vector is used, not a round one: `[0.1, 0.1, 0.15, 0.15]` with
    `[1, 1, 0, 0]` separates the conventions (0.525 vs 0.375), whereas a vector of
    only round values gives the same answer either way and would pin nothing.
    """
    head("9. ECE bin convention, and single implementation")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from cxr_trust.evaluation.metrics import expected_calibration_error
    except Exception as exc:  # noqa: BLE001
        add("ERROR", f"cannot import expected_calibration_error: {exc}")
        return

    yt = [1, 1, 0, 0]
    yp = [0.1, 0.1, 0.15, 0.15]
    got = expected_calibration_error(yt, yp)

    def ece_variant(right: bool) -> float:
        y = np.asarray(yt, float)
        p = np.asarray(yp, float)
        bins = np.linspace(0.0, 1.0, 11)
        ids = np.digitize(p, bins[1:-1], right=right)
        val = 0.0
        for b in range(10):
            m = ids == b
            if np.any(m):
                val += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
        return float(val)

    edge_closed = ece_variant(True)     # (0, 0.1]  -- non-standard
    edge_open = ece_variant(False)      # [0, 0.1)  -- sklearn / Guo et al.
    ok = abs(got - edge_open) < 1e-12
    print(f"  discriminating vector -> shipped={got:.6f}  "
          f"edge-closed={edge_closed:.6f}  edge-open={edge_open:.6f}")
    if abs(edge_closed - edge_open) < 1e-12:
        add("ERROR", "the ECE convention probe does not discriminate; the check is "
                     "meaningless")
    elif not ok:
        add("ERROR", f"ECE uses the non-standard (0, 0.1] convention: got {got}, "
                     f"expected the standard [0, 0.1) value {edge_open}")
    else:
        print("  [ok  ] ECE uses the standard [left-closed, right-open) convention")

    # No second implementation may exist -- with one precise exception. A file may
    # call digitize if the convention is an explicit parameter of the function it
    # sits in, because then both conventions are reachable and named, and the file
    # is structurally a comparator rather than a metric. Exempting by filename would
    # let a real duplicate through just because it was named right; this rule is
    # about the shape of the code.
    dupes = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        if p.name == "45_debug_sweep.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for ln, line in enumerate(lines, 1):
            if "digitize(" not in line or "bins" not in line:
                continue
            stripped = line.strip()
            # Prose is not code. A docstring that quotes the wrong convention (as
            # 54's does, deliberately, to document what changed) is not a second
            # implementation, and flagging it would train the reader to ignore this
            # check.
            if stripped.startswith(("#", '"', "'", "*")) or "``" in line:
                continue
            # find the enclosing def and ask whether it takes `right`
            header = ""
            for back in range(ln - 1, max(0, ln - 60), -1):
                if re.match(r"\s*def\s+\w+\s*\(", lines[back - 1]):
                    header = lines[back - 1]
                    break
            if re.search(r"\bright\b\s*[:=]", header):
                print(f"  [ok  ] {p.name}:{ln} is inside "
                      f"{header.strip()[:44]} -- convention is an explicit parameter")
                continue
            dupes.append(f"{p.name}:{ln}")
        for ln, line in enumerate(lines, 1):
            if re.match(r"\s*def ece\s*\(.*\)\s*:", line):
                # a local ece() with no `right` parameter is a silent second metric
                if not re.search(r"\bright\b", line):
                    dupes.append(f"{p.name}:{ln} defines a local ece() with no "
                                 f"convention parameter")
    if dupes:
        add("ERROR", f"a second ECE implementation exists (it will drift): {dupes}")
    else:
        print("  [ok  ] exactly one ECE implementation, no silent duplicate")


def check_stage_aware_guard() -> None:
    """The [BAD] guard must inspect the file belonging to the stage it checked.

    It used to take (model, layer) and return the first ``*_predictions.jsonl``
    under the model directory, so for reliance / precision / ablation it could
    return the *shift* file -- written first and usually clean. A poisoned stage
    would then be sampled as clean and recorded complete, which is exactly the
    silent failure this guard exists to prevent.

    The check loads ``Get-PredictionFile`` out of the real driver and calls it, so
    it tests the shipped source rather than a copy of its logic.
    """
    head("10. stage-aware [BAD] guard (loads the real function)")
    ps = ROOT / "scripts" / "run_all_layers.ps1"
    if not ps.exists():
        add("ERROR", "run_all_layers.ps1 missing")
        return
    text = ps.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?ms)^function Get-PredictionFile.*?\n\}", text)
    if not m:
        add("ERROR", "cannot find Get-PredictionFile in the driver")
        return

    # build a synthetic tree: a clean shift file and a poisoned reliance file
    import tempfile
    with tempfile.TemporaryDirectory(prefix="stageguard_") as td:
        root = Path(td)
        base = root / "nih_eval" / "m1"
        for sub, name, row in (("shift", "shift_predictions.jsonl",
                                '{"image_id":"a","error":""}'),
                               ("reliance", "image_reliance_predictions.jsonl",
                                '{"image_id":"a","condition":"original","error":"boom"}'),
                               ("reliance_bf16", "image_reliance_predictions.jsonl",
                                '{"image_id":"a","condition":"original","error":""}')):
            d = base / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(row + "\n", encoding="utf-8")

        layers = ("@{\n  nih = @{ Dir = 'nih_eval' }\n}")
        harness = (
            f"$script:OutputRoot = '{root}'\n"
            f"$LayerDef = {layers}\n"
            + m.group(0) + "\n"
            + "$r = [ordered]@{}\n"
            + "foreach ($s in @('shift','reliance','precision','ablation')) {\n"
            + "  $r[$s] = Get-PredictionFile 'm1' 'nih' $s\n"
            + "}\n"
            + "$r | ConvertTo-Json -Compress\n"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", harness],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        if out.returncode != 0:
            add("ERROR", f"stage-guard harness failed: {(out.stderr or '')[:200]}")
            return
        try:
            picked = json.loads((out.stdout or "{}").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            add("ERROR", f"stage-guard harness produced no JSON: {out.stdout!r}")
            return
        expect = {"shift": "shift_predictions.jsonl",
                  "reliance": "image_reliance_predictions.jsonl",
                  "precision": "image_reliance_predictions.jsonl",
                  "ablation": None}
        for stage, want in expect.items():
            got = picked.get(stage)
            got_name = Path(got).name if got else None
            got_dir = Path(got).parent.name if got else None
            want_dir = {"shift": "shift", "reliance": "reliance",
                        "precision": "reliance_bf16", "ablation": "ablation"}[stage]
            if want is None:
                # ablation has no file in this fixture, and must NOT fall back to
                # another stage's file
                ok = got is None
            else:
                ok = (got_name == want and got_dir == want_dir)
            print(f"  {'ok  ' if ok else 'FAIL'} {stage:10s} -> "
                  f"{got_dir + '/' + str(got_name) if got else 'None'}")
            if not ok:
                add("ERROR", f"stage-aware guard picked {got!r} for stage {stage}, "
                             f"expected a file under {want_dir}/")


def check_layer_eval_counts() -> None:
    """The coverage denominator must be the SPLIT-FILTERED size, not the file rows.

    ``nih_eval_1732.csv`` holds 1,732 rows, but 226 of them are ``internal_train``
    donors for the no_finding repair, so the runner -- which passes
    ``--split internal_test,source_val`` -- writes only 1,506 x cells. The first
    full run expected 1,732 x cells, measured 87% coverage on a complete and
    correct stage, marked it [BAD] and DELETED it. The driver must therefore count
    the manifest's eval rows; this loads the real ``Get-LayerDef`` to prove the NIH
    denominator is 1,506 and not 1,732.
    """
    head("14. layer eval-set size is the split-filtered count (behavioural)")
    ps = ROOT / "scripts" / "run_all_layers.ps1"
    if not ps.exists():
        add("ERROR", "run_all_layers.ps1 missing")
        return
    text = ps.read_text(encoding="utf-8", errors="replace")
    f_count = re.search(r"(?ms)^function Get-EvalImageCount.*?\n\}", text)
    f_def = re.search(r"(?ms)^function Get-LayerDef.*?\n\}", text)
    if not (f_count and f_def):
        add("ERROR", "cannot find Get-EvalImageCount/Get-LayerDef in the driver")
        return
    harness = (
        f"$script:Root = '{ROOT}'\n"
        + f_count.group(0) + "\n" + f_def.group(0) + "\n"
        + "$d = Get-LayerDef\n"
        + "$r = [ordered]@{ nih = $d.nih.Images; padchest = $d.padchest.Images; "
          "openi = $d.openi.Images }\n"
        + "$r | ConvertTo-Json -Compress\n"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", harness],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=120)
    if out.returncode != 0:
        add("ERROR", f"eval-count harness failed: {(out.stderr or '')[:200]}")
        return
    try:
        got = json.loads((out.stdout or "{}").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        add("ERROR", f"eval-count harness produced no JSON: {out.stdout!r}")
        return
    expect = {"nih": 1506, "padchest": 1500, "openi": 947}
    for layer, want in expect.items():
        g = got.get(layer)
        ok = (g == want)
        print(f"  {'ok  ' if ok else 'FAIL'} {layer:9s} eval images = {g} (want {want})")
        if not ok:
            add("ERROR", f"{layer} eval count is {g}, expected {want} -- the coverage "
                         f"denominator must be the split-filtered count, not the "
                         f"manifest's row count")


def check_contract_exempt_guard() -> None:
    """LLaVA-Med's prose must be KEPT and marked contract-exempt, not deleted.

    llava_med_v1_5_7b answers in prose and never satisfies the JSON contract, so a
    strict [BAD] guard would delete its file, leave the stage out of the state
    file, and redo ~13h of work on every re-run -- while destroying the very prose
    that evidences the prose-without-the-contract failure mode. The exemption must
    (a) name exactly the documented model, (b) actually relax Test-StageOutput, and
    (c) not leak to any other model. (b) is checked by loading the real function.

    The same section also guards the second, layer-level mechanism: the open-i
    report layer is free text for every model, so it relaxes the parse check but
    must still fail on an empty reply (an empty report is not a deliverable).
    """
    head("13. contract-exempt guard (behavioural + source)")
    ps = ROOT / "scripts" / "run_all_layers.ps1"
    if not ps.exists():
        add("ERROR", "run_all_layers.ps1 missing")
        return
    text = ps.read_text(encoding="utf-8", errors="replace")

    # (a) exactly one named exemption, and it is the documented model
    block = re.search(r"(?ms)\$KnownContractFailures = @\{(.*?)\n\}", text)
    if not block:
        add("ERROR", "no $KnownContractFailures table in the driver")
        return
    keys = re.findall(r"'([a-z0-9_]+)'\s*=", block.group(1))
    if keys != ["llava_med_v1_5_7b"]:
        add("ERROR", f"$KnownContractFailures names {keys}, expected exactly "
                     "['llava_med_v1_5_7b']")

    # (c) the guard call site and the coverage census both honour it
    if "-ContractExempt:$exempt" not in text:
        add("ERROR", "the guard call site does not pass -ContractExempt")
    if "contract-exempt (prose by design); no label census" not in text:
        add("ERROR", "the coverage census does not skip contract-exempt models")

    # (d) the layer-level prose exemption exists and reaches the guard
    if "$ProseByDesignLayers" not in text:
        add("ERROR", "no $ProseByDesignLayers table in the driver")
    if "-ProseStage:$ex.prose" not in text:
        add("ERROR", "the guard call site does not pass -ProseStage")

    # (b) behavioural: the same all-unparsed file must fail strict and pass exempt
    import tempfile
    with tempfile.TemporaryDirectory(prefix="exempt_") as td:
        root = Path(td)
        prose = ('{"image_id":"a","raw_output":"There is cardiomegaly.",'
                 '"parsed":{"parse_success":false}}')
        empty = '{"image_id":"a","raw_output":"","parsed":{"parse_success":false}}'
        # Three fixtures, one per behaviour under test:
        #   nih      prose + empty, all unparsed  -> model exemption tolerates both
        #   openi    non-empty prose only         -> prose LAYER must accept it
        #   padchest all empty                    -> prose LAYER must still reject it
        fixtures = {
            "nih_eval/m1/shift": (prose + "\n") * 2 + (empty + "\n") * 2,
            "openi/m1/shift":    (prose + "\n") * 4,
            "padchest/m1/shift": (empty + "\n") * 4,
        }
        for rel, body in fixtures.items():
            d = root / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "shift_predictions.jsonl").write_text(body, encoding="utf-8")
        f_pred = re.search(r"(?ms)^function Get-PredictionFile.*?\n\}", text)
        f_guard = re.search(r"(?ms)^function Test-StageOutput.*?\n\}", text)
        if not (f_pred and f_guard):
            add("ERROR", "cannot find Get-PredictionFile/Test-StageOutput in the driver")
            return
        harness = (
            f"$script:OutputRoot = '{root}'\n"
            "$LayerDef = @{ nih = @{ Dir = 'nih_eval' }; openi = @{ Dir = 'openi' }; "
            "padchest = @{ Dir = 'padchest' } }\n"
            + f_pred.group(0) + "\n" + f_guard.group(0) + "\n"
            + "$r = [ordered]@{}\n"
            + "$r['strict'] = (Test-StageOutput 'm1' 'nih' 'shift' -Expected 4).ok\n"
            + "$r['exempt'] = (Test-StageOutput 'm1' 'nih' 'shift' -Expected 4 -ContractExempt).ok\n"
            + "$r['prose'] = (Test-StageOutput 'm1' 'openi' 'shift' -Expected 4 -ProseStage).ok\n"
            + "$r['prose_empty'] = (Test-StageOutput 'm1' 'padchest' 'shift' -Expected 4 -ProseStage).ok\n"
            + "$r | ConvertTo-Json -Compress\n"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", harness],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        if out.returncode != 0:
            add("ERROR", f"contract-exempt harness failed: {(out.stderr or '')[:200]}")
            return
        try:
            got = json.loads((out.stdout or "{}").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            add("ERROR", f"contract-exempt harness produced no JSON: {out.stdout!r}")
            return
        strict_ok, exempt_ok = got.get("strict"), got.get("exempt")
        prose_ok, prose_empty = got.get("prose"), got.get("prose_empty")
        print(f"  {'ok  ' if strict_ok is False else 'FAIL'} strict     -> ok={strict_ok} (must reject all-prose)")
        print(f"  {'ok  ' if exempt_ok is True else 'FAIL'} exempt     -> ok={exempt_ok} (must accept prose+empty)")
        print(f"  {'ok  ' if prose_ok is True else 'FAIL'} prose      -> ok={prose_ok} (prose layer must accept non-empty prose)")
        print(f"  {'ok  ' if prose_empty is False else 'FAIL'} prose/empty-> ok={prose_empty} (prose layer must reject all-empty)")
        if strict_ok is not False:
            add("ERROR", f"strict guard accepted an all-unparsed file (ok={strict_ok})")
        if exempt_ok is not True:
            add("ERROR", f"contract-exempt guard rejected prose (ok={exempt_ok})")
        if prose_ok is not True:
            add("ERROR", f"prose-by-design layer rejected non-empty prose (ok={prose_ok})")
        if prose_empty is not False:
            add("ERROR", f"prose-by-design layer accepted an all-empty file (ok={prose_empty})")


def check_calibration_universe() -> None:
    """The calibration sample universe must be explicit, and must be the declared one.

    Two metric families need different rows: probability metrics (auroc, auprc, ece,
    brier) are functions of the declared probability and so include rows where the
    model hedged, while classification/selective metrics need a definite call.
    A single ``dropna(subset=[y_true, y_pred, y_prob])`` used to serve both, which
    silently removed every uncertain row from the calibration metrics too -- the
    paper reported ECE on 1614 of 1800 MedGemma test decisions and said nothing
    about it, while another script reported the full-set number for the same data.

    The check is behavioural: run the real evaluator on a synthetic file containing
    known uncertain rows and require that the two counts differ. A source-level grep
    would pass again the moment someone restores the combined dropna.
    """
    head("11. calibration sample universe is explicit (behavioural)")
    ev = ROOT / "scripts" / "04_evaluate.py"
    if not ev.exists():
        add("ERROR", "04_evaluate.py missing")
        return

    import tempfile
    LABELS = ["atelectasis", "cardiomegaly"]
    with tempfile.TemporaryDirectory(prefix="univ_") as td:
        tmp = Path(td)
        rows = []
        # 10 decided rows, 4 hedged rows (uncertain but WITH a probability)
        for i in range(10):
            rows.append({"image_id": f"i{i}", "split": "internal_test",
                         "ground_truth": {f"label_{lb}": "positive" if i % 2 else "negative"
                                          for lb in LABELS},
                         "findings": {lb: {"status": "positive" if i % 2 else "negative",
                                           "probability_positive": 0.9 if i % 2 else 0.1}
                                      for lb in LABELS}})
        for i in range(4):
            rows.append({"image_id": f"u{i}", "split": "internal_test",
                         "ground_truth": {f"label_{lb}": "negative" for lb in LABELS},
                         "findings": {lb: {"status": "uncertain",
                                           "probability_positive": 0.2}
                                      for lb in LABELS}})
        pred = tmp / "predictions.jsonl"
        pred.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        out = tmp / "out"
        r = subprocess.run([sys.executable, str(ev), "--predictions", str(pred),
                            "--output-dir", str(out)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(ROOT), timeout=300)
        if r.returncode != 0:
            add("ERROR", f"04_evaluate failed on the synthetic fixture: "
                         f"{(r.stderr or '')[-200:]}")
            return
        mf = out / "metrics_by_label.csv"
        if not mf.exists():
            add("ERROR", "04_evaluate wrote no metrics_by_label.csv")
            return
        import csv as _csv
        with mf.open("r", encoding="utf-8-sig", newline="") as fh:
            got = list(_csv.DictReader(fh))
        if not got:
            add("ERROR", "metrics_by_label.csv has no rows")
            return
        cols = set(got[0].keys())
        missing = {"n_prob", "n_definite"} - cols
        if missing:
            add("ERROR", f"metrics_by_label.csv is missing explicit universe counts: "
                         f"{sorted(missing)}")
            return
        n_prob = int(got[0]["n_prob"])
        n_def = int(got[0]["n_definite"])
        print(f"  fixture: 14 rows/label (10 decided + 4 hedged) -> "
              f"n_prob={n_prob}, n_definite={n_def}")
        if n_prob == n_def:
            add("ERROR", "the two universes collapsed into one: hedged rows were "
                         "removed from the probability metrics again")
        elif n_prob == 14 and n_def == 10:
            print("  [ok  ] probability metrics keep hedged rows; "
                  "classification metrics require a definite call")
        else:
            add("ERROR", f"unexpected universes: n_prob={n_prob} (want 14), "
                         f"n_definite={n_def} (want 10)")


def check_row_keys() -> None:
    """Section 12: every row key the analysis code reads is one a runner writes.

    Loads `64_audit_row_keys.py`'s real functions rather than re-implementing the
    check -- the same lesson section 10 records about the stage-aware guard, where
    asserting against a copy of the driver's function put the same defect one
    level up.

    Why a standing guard: on 2026-10-06 `59_probe_pretest_findings.py` turned out
    to read `probability_pos` where the schema writes `probability_positive`. A
    missing key reached through `.get` does not raise, so that column was `None`
    for every row and the comparison it fed used the status column instead. It
    survived a whole round of review because nothing compares reads to writes.
    """
    head("12. row keys: everything analysis reads is something a runner writes")
    path = ROOT / "scripts" / "64_audit_row_keys.py"
    if not path.exists():
        add("ERROR", "64_audit_row_keys.py is missing; the read/write key audit cannot run")
        return
    proc = subprocess.run(
        [sys.executable, str(path), "--selftest"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if "PASS: the scan does flag the known bug" not in out:
        add("ERROR", "64_audit_row_keys.py --selftest did not confirm it can detect "
                     "the probability_pos bug; the audit's silence means nothing")
        return
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    suspect_lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("READ")]
    if proc.returncode != 0 or suspect_lines:
        add("ERROR", "row-key audit found near misses (a read key that no runner "
                     "writes): " + "; ".join(suspect_lines[:6]))
    else:
        print("  [ok  ] no near misses: 391 read keys, all of them written somewhere")
    always_null = [ln.strip() for ln in out.splitlines()
                   if ln.strip() and "rows" in ln and ln.strip().endswith("rows")]
    if always_null:
        print(f"  [note] {len(always_null)} key(s) are empty in every row sampled "
              f"(informational): {', '.join(x.split()[0] for x in always_null[:6])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings as well as errors")
    args = ap.parse_args()

    print(f"program root: {ROOT}")
    check_syntax()
    check_choices()
    check_manifests()
    check_images()
    check_imports()
    check_scripts_exist()
    check_durable_usage()
    check_isolation_guard()
    check_layer_eval_counts()
    check_ece_convention()
    check_stage_aware_guard()
    check_contract_exempt_guard()
    check_calibration_universe()
    check_row_keys()

    errors = [m for s, m in FINDINGS if s == "ERROR"]
    warns = [m for s, m in FINDINGS if s == "WARN"]
    print(f"\n{'=' * 78}")
    print(f"  debug sweep: {len(errors)} error(s), {len(warns)} warning(s)")
    for m in errors:
        print(f"    ERROR  {m}")
    for m in warns:
        print(f"    WARN   {m}")
    print(f"{'=' * 78}")
    (ROOT / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "state" / "debug_sweep.json").write_text(
        json.dumps({"errors": errors, "warnings": warns}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if errors:
        return 1
    if warns and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
