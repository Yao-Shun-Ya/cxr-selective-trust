#!/usr/bin/env python
"""RRG report-evidence labelling: replace rule_based_v0 with a judged labeler.

The problem being fixed
-----------------------
`papers/main_v2/sections/03_methods.tex` describes the CheXpert-RRG evidence
extractor as "a rule-based keyword + negation-window method (rule_based_v0)" and
states that the final analysis should use a validated report labeller. Every
hallucination and omission number in the paper inherits that weakness, and a
reviewer attacking it attacks the external-validation section wholesale.

What this script does
---------------------
Extracts, for every (image, finding), whether the *reference report* asserts the
finding, using one of three evidence sources, then reports the agreement between
them so the choice is defensible rather than asserted:

``rule``
    The existing ``rule_based_v0`` behaviour, reproduced here so it can be
    compared against the others on identical data.
``judge``
    A local text LLM acting as adjudicator (an "LLM-as-judge" labeler). This is
    the domain-standard alternative to keyword matching: CheXbert / CheXpert
    labeler style extraction, but promptable and auditable.
``gold``
    Human adjudication, when available as a CSV. This is what actually converts
    the numbers from pilot to publishable, and it is also the only source that
    can yield an inter-rater statistic.

Design notes
------------
* **The judge must not be the model under evaluation.** Grading a model with
  itself is circular; a *different* model family is used and the judge's
  identity is recorded in the output.
* **The judge is asked for a constrained answer** (present / absent / uncertain)
  with a short evidence quote, so it can be audited and so its output parses.
* **Agreement is computed pairwise** (rule vs judge, judge vs gold) with Cohen's
  kappa, plus per-finding breakdowns, because aggregate agreement can hide a
  label that is systematically wrong.
* **The judge prompt is versioned** (`JUDGE_PROMPT_VERSION`) and written into
  every output, since a judged labeler is only reproducible together with its
  exact prompt.

Usage
-----
::

    # rule-only baseline (no model needed) -- always run this first
    python scripts/20_rrg_labeler_eval.py --mode rule \
        --output-dir outputs/rrg_evidence/eval

    # add the LLM judge (loads a local model; --mode takes several values, and
    # "all" is NOT one of them -- the docstring used to say it was, and argparse
    # then rejected the command with "invalid choice: 'all'")
    python scripts/20_rrg_labeler_eval.py --mode rule judge report \
        --reports data/manifests/openi_rrg.csv \
        --judge-model models/Qwen2.5-VL-7B-Instruct --judge-load-in-4bit \
        --output-dir outputs/rrg_evidence/eval

    # score against a human adjudication file
    python scripts/20_rrg_labeler_eval.py --mode report \
        --gold human_adjudication.csv --output-dir outputs/rrg_evidence/eval

The human file needs columns: ``image_id,label,status`` where status is one of
``present`` / ``absent`` / ``uncertain``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JUDGE_PROMPT_VERSION = "judge_v1.0.0"

REPORT_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax",
]

# --------------------------------------------------------------------------- #
# negation / uncertainty cues, kept explicit for auditability
# --------------------------------------------------------------------------- #
NEGATION_CUES = [
    "no ", "no evidence", "without", "negative for", "free of", "clear of",
    "absence of", "not ", "denies", "unremarkable", "no acute", "rules out",
    "resolved", "improved", "interval resolution",
]
UNCERTAIN_CUES = [
    "possible", "possibly", "may ", "might", "suggest", "suspicious",
    "cannot exclude", "cannot be excluded", "questionable", "equivocal",
    "likely", "probable", "concerning for", "borderline",
]

# label -> surface forms
LABEL_SYNONYMS = {
    "atelectasis": ["atelectasis", "atelectatic", "collapse", "volume loss"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "enlarged cardiac",
                     "cardiac enlargement", "heart size is enlarged"],
    "consolidation": ["consolidation", "consolidative", "airspace opacity",
                      "air space opacity", "airspace disease"],
    "edema": ["edema", "oedema", "pulmonary vascular congestion", "congestion",
              "vascular congestion", "interstitial edema"],
    "pleural_effusion": ["pleural effusion", "effusion", "pleural fluid",
                         "costophrenic angle blunting"],
    "pneumonia": ["pneumonia", "infectious", "infection", "consolidation consistent"],
    "pneumothorax": ["pneumothorax", "pneumothoraces", "collapsed lung"],
}


def _window(text: str, pos: int, before: int = 60, after: int = 40) -> str:
    return text[max(0, pos - before): pos + after]


def rule_label_report(text: str) -> dict[str, str]:
    """rule_based_v0, reproduced: keyword hit + negation / uncertainty window."""
    t = (text or "").lower()
    out = {}
    for lb in REPORT_LABELS:
        status = "absent"
        for syn in LABEL_SYNONYMS.get(lb, [lb]):
            for m in re.finditer(re.escape(syn), t):
                w = _window(t, m.start())
                if any(c in w for c in UNCERTAIN_CUES):
                    status = "uncertain"
                elif any(c in w for c in NEGATION_CUES):
                    if status != "uncertain":
                        status = "absent"
                else:
                    status = "present"
                    break
            if status == "present":
                break
        out[lb] = status
    return out


# --------------------------------------------------------------------------- #
# judge prompt
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = ("You are a careful radiology report annotator. You extract "
                "whether a report asserts a finding. You never guess: if the "
                "report does not address a finding, say absent.")

JUDGE_TEMPLATE = """Read this chest radiograph report and decide, for each finding,
whether the report ASSERTS the finding is present.

Report:
\"\"\"
{report}
\"\"\"

For each finding answer with exactly one of:
- "present": the report states or clearly implies the finding is there
- "absent": the report states it is not there, or does not mention/address it
- "uncertain": the report hedges (possible / suspicious / cannot exclude)

Answer with one JSON object only:
{{"findings": {{{keys}}}}}

Rules:
- Judge only what the report says. Do not use your own image impression.
- A finding that is not mentioned at all is "absent".
- Hedged phrases are "uncertain", not "present".
"""


def _load_judge(model_path: str, load_in_4bit: bool):
    """Load a judge model, tolerating text-only and multimodal checkpoints alike.

    A judge only needs to read text, but the convenient local checkpoints are
    often *multimodal* (Qwen2.5-VL, MedGemma, ...), and those refuse to load
    through ``AutoModelForCausalLM``. So we try the causal-LM class first (right
    for a pure text model) and fall back to the image-text classes, which accept
    a text-only prompt perfectly well. Which class succeeded is reported, since
    a judged labeler is only reproducible together with its exact model.
    """
    import torch
    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "auto"

    classes = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq"):
        try:
            import transformers
            classes.append((name, getattr(transformers, name)))
        except AttributeError:
            continue

    last_error: Exception | None = None
    for name, cls in classes:
        try:
            model = cls.from_pretrained(model_path, **kwargs)
        except Exception as exc:      # noqa: BLE001 - fall through to next class
            last_error = exc
            continue
        model.eval()
        # tokenizer: prefer AutoTokenizer, fall back to the multimodal processor
        try:
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tok = getattr(proc, "tokenizer", proc)
        return model, tok, name

    raise RuntimeError(
        f"could not load judge {model_path!r} with any of "
        f"{[n for n, _ in classes]}: {last_error}")


def build_judge_prompt(report: str) -> tuple[str, str]:
    keys = ", ".join(f'"{lb}": "present|absent|uncertain"' for lb in REPORT_LABELS)
    return JUDGE_SYSTEM, JUDGE_TEMPLATE.format(report=(report or "").strip()[:4000], keys=keys)


def parse_judge_output(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    findings = obj.get("findings") if isinstance(obj, dict) else None
    if not isinstance(findings, dict):
        return {}
    out = {}
    for lb in REPORT_LABELS:
        v = str(findings.get(lb, "")).strip().lower()
        if v in {"present", "absent", "uncertain"}:
            out[lb] = v
    return out


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #
def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    m = pd.notna(a) & pd.notna(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= len(a)
    pe = float(O.sum(axis=0) @ O.sum(axis=1))
    po = float(np.trace(O))
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float("nan")


def agreement_block(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    m = pd.notna(a) & pd.notna(b)
    if m.sum() == 0:
        return {"comparison": name, "n": 0}
    return {
        "comparison": name,
        "n": int(m.sum()),
        "agreement": float(np.mean(np.asarray(a)[m] == np.asarray(b)[m])),
        "cohen_kappa": cohens_kappa(np.asarray(a), np.asarray(b)),
    }


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_rrg_reports() -> pd.DataFrame:
    """RRG reference reports.

    The project manifest already carries the decoded reference text
    (``data/manifests/chexpert_rrg_all.csv`` with a ``reference_text`` column),
    so there is no need to re-read the parquet files.
    """
    man = ROOT / "data/manifests/chexpert_rrg_all.csv"
    if not man.exists():
        raise FileNotFoundError(
            f"{man} not found; run scripts/01_build_manifests.py first")
    df = pd.read_csv(man)
    if "reference_text" not in df.columns:
        raise KeyError(f"no reference_text column in {man}: {list(df.columns)}")
    return df


def find_reference_text_column(df: pd.DataFrame) -> str:
    for c in ("reference_text", "reference", "findings_section", "report", "text"):
        if c in df.columns:
            return c
    raise KeyError(f"no reference-text column found in {list(df.columns)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", nargs="+", default=["rule"],
                    choices=["rule", "judge", "report"],
                    help="rule = keyword baseline; judge = LLM labeler; report = agreement tables")
    ap.add_argument("--reports", default=None,
                    help="CSV with image_id + reference text; default = project RRG loader")
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--judge-model", default=None,
                    help="local path or HF id of a TEXT model used as adjudicator")
    ap.add_argument("--judge-load-in-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gold", default=None, help="human adjudication CSV for kappa")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_rule = out_dir / "evidence_rule.csv"
    path_judge = out_dir / "evidence_judge.csv"

    # ---- load reports ---------------------------------------------------
    if args.reports:
        rep = pd.read_csv(args.reports if Path(args.reports).is_absolute()
                          else ROOT / args.reports)
        text_col = args.text_column or find_reference_text_column(rep)
    else:
        rep = load_rrg_reports()
        text_col = args.text_column or find_reference_text_column(rep)
    if args.limit:
        rep = rep.head(args.limit)
    print(f"reports: {len(rep)} rows, text column = {text_col}")
    if "reference_text" in rep.columns:
        empty = int(rep["reference_text"].isna().sum())
        if empty:
            print(f"  [warn] {empty} reports have empty reference text "
                  f"(dropped before labelling)")
        rep = rep[rep["reference_text"].notna() & (rep["reference_text"].astype(str).str.strip() != "")]
        print(f"  usable reports: {len(rep)}")

    # ---- rule baseline --------------------------------------------------
    if "rule" in args.mode or not path_rule.exists():
        rows = []
        for _, r in rep.iterrows():
            labels = rule_label_report(str(r[text_col]))
            for lb, st in labels.items():
                rows.append({"image_id": r["image_id"], "label": lb, "rule_status": st})
        rule_df = pd.DataFrame(rows)
        rule_df.to_csv(path_rule, index=False)
        print(f"[rule] wrote {path_rule}  ({len(rule_df)} rows)")
    else:
        rule_df = pd.read_csv(path_rule)

    # ---- LLM judge ------------------------------------------------------
    judge_df = pd.read_csv(path_judge) if path_judge.exists() else pd.DataFrame()
    if "judge" in args.mode:
        if not args.judge_model:
            print("[judge] skipped: --judge-model not given")
        else:
            import torch
            print(f"[judge] loading {args.judge_model}")
            model, tok, cls_name = _load_judge(args.judge_model, args.judge_load_in_4bit)
            print(f"[judge] loaded via {cls_name}")

            done = set()
            if len(judge_df):
                done = set(judge_df["image_id"].astype(str))
            rows = judge_df.to_dict("records") if len(judge_df) else []
            from tqdm import tqdm
            for _, r in tqdm(rep.iterrows(), total=len(rep), desc="judge", unit="report"):
                iid = str(r["image_id"])
                if iid in done:
                    continue
                system, user = build_judge_prompt(str(r[text_col]))
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                raw = tok.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                labels = parse_judge_output(raw)
                if not labels:
                    print(f"[warn] judge output unparseable for {iid}: {raw[:80]!r}")
                for lb in REPORT_LABELS:
                    rows.append({"image_id": iid, "label": lb,
                                 "judge_status": labels.get(lb, "unparsed")})
                if len(rows) % 20 == 0:
                    pd.DataFrame(rows).to_csv(path_judge, index=False)
            judge_df = pd.DataFrame(rows)
            judge_df.to_csv(path_judge, index=False)
            n_unparsed = int((judge_df["judge_status"] == "unparsed").sum()) if len(judge_df) else 0
            print(f"[judge] wrote {path_judge}  ({len(judge_df)} rows, {n_unparsed} unparsed)")

    # ---- agreement ------------------------------------------------------
    if "report" in args.mode:
        merged = rule_df.copy()
        if len(judge_df):
            merged = merged.merge(judge_df, on=["image_id", "label"], how="outer")
        if args.gold:
            gold = pd.read_csv(args.gold)
            gold = gold.rename(columns={"status": "gold_status"})
            merged = merged.merge(gold[["image_id", "label", "gold_status"]],
                                  on=["image_id", "label"], how="outer")
        merged.to_csv(out_dir / "evidence_merged.csv", index=False)

        blocks, per_label = [], []
        pairs = [("rule_status", "judge_status", "rule_vs_judge"),
                 ("judge_status", "gold_status", "judge_vs_gold"),
                 ("rule_status", "gold_status", "rule_vs_gold")]
        for ca, cb, name in pairs:
            if ca in merged.columns and cb in merged.columns:
                blk = agreement_block(merged[ca].to_numpy(dtype=object),
                                      merged[cb].to_numpy(dtype=object), name)
                blocks.append(blk)
                if blk.get("n"):
                    print(f"  {name:16s} n={blk['n']:5d}  agreement={blk['agreement']:.3f}  "
                          f"kappa={blk['cohen_kappa']:.3f}")
                for lb, g in merged.groupby("label"):
                    b = agreement_block(g[ca].to_numpy(dtype=object),
                                        g[cb].to_numpy(dtype=object), name)
                    b["label"] = lb
                    per_label.append(b)
        pd.DataFrame(blocks).to_csv(out_dir / "agreement_summary.csv", index=False)
        pd.DataFrame(per_label).to_csv(out_dir / "agreement_by_label.csv", index=False)

        summary = {
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_model": args.judge_model,
            "mode": args.mode,
            "n_reports": int(len(rep)),
            "n_rows": int(len(merged)),
            "agreement": blocks,
            "caveats": [
                "rule = keyword + negation-window (rule_based_v0), reproduced for comparison.",
                "judge = LLM adjudicator; it must NOT be the model under evaluation.",
                "A judged labeler is reproducible only together with its prompt version.",
                "Only a human adjudication file can support an inter-rater claim.",
            ],
        }
        (out_dir / "labeler_eval_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
