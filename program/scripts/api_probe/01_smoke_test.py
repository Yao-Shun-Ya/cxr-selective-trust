"""Step 1: environment + multimodal token probe.

Eight image calls (2 per modality) plus one text-only call. The point is to catch
a relay that silently drops ``image_url``: if the image is ignored, prompt_tokens
collapses to the text baseline while every other field still looks healthy.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from PIL import Image

from common import (
    API_KEY, BASE_URL, MODEL, OUT_DIR, ROOT, Request, build_condition_image,
    call_api, encode_image, ensure_dirs, parse_response, payload_for,
    prompt_for, resolve_image, setup_logging,
)

MODALITIES = [
    ("cxr", ROOT / "data" / "manifests" / "nih_eval_1506.csv"),
    ("mri", ROOT / "data" / "manifests" / "crossmodal_mri.csv"),
    ("oct", ROOT / "data" / "manifests" / "crossmodal_oct.csv"),
    ("derm", ROOT / "data" / "manifests" / "crossmodal_derm.csv"),
]


def _make_request(row: dict, cohort: str) -> tuple[Request, dict]:
    system, user, labels = prompt_for(cohort, "v0")
    img = build_condition_image(Image.open(resolve_image(row, cohort)), "clean",
                                str(row["image_id"]))
    b64, w, h, nbytes = encode_image(img)
    req = Request(probe_id=f"smoke|{row['image_id']}", cohort=cohort,
                  schema="v0", condition="clean", image_b64=b64,
                  system=system, user=user, labels=labels, meta={})
    return req, {"w": w, "h": h, "bytes": nbytes}


async def main() -> int:
    ensure_dirs()
    log = setup_logging("01_smoke_test")
    if not API_KEY:
        log.error("OPENAI_API_KEY is empty; export it before running")
        return 2
    log.info("model=%s base_url=%s", MODEL, BASE_URL)

    cases = []
    for cohort, man in MODALITIES:
        for _, row in pd.read_csv(man).head(2).iterrows():
            req, sent = _make_request(row.to_dict(), cohort)
            cases.append((req, sent, row.to_dict()))

    results = []
    ctx = None
    baseline = payload_for(cases[0][0], MODEL)
    baseline["messages"][1]["content"] = [{"type": "text",
                                           "text": cases[0][0].user}]
    try:
        body, _ = await call_api(baseline, log=log, probe_id="text_baseline")
        ctx = (body.get("usage") or {}).get("prompt_tokens")
    except Exception as exc:  # noqa: BLE001
        log.warning("text baseline failed: %s", exc)

    for req, sent, row in cases:
        try:
            body, meta = await call_api(payload_for(req, MODEL), log=log,
                                        probe_id=req.probe_id)
            parsed = parse_response(body, req.labels, req.cohort)
            pt = parsed["usage"]["prompt_tokens"]
            results.append({
                "cohort": req.cohort, "image_id": row["image_id"],
                "sent_w": sent["w"], "sent_h": sent["h"],
                "sent_kb": round(sent["bytes"] / 1024, 1),
                "prompt_tokens": pt,
                "completion_tokens": parsed["usage"]["completion_tokens"],
                "model_returned": parsed["model_returned"],
                "finish_reason": parsed["finish_reason"],
                "parse_success": parsed["parsed"].get("parse_success"),
                "ratio_vs_text": round(pt / ctx, 2) if pt and ctx else None,
                "content_head": parsed["content"][:80].replace("\n", " "),
                "attempts": meta["attempts"], "http_status": meta["http_status"],
            })
        except Exception as exc:  # noqa: BLE001
            log.error("call failed for %s: %s", row["image_id"], exc)
            results.append({"cohort": req.cohort, "image_id": row["image_id"],
                            "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(results)
    out = OUT_DIR / "smoke_test.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\ntext-only prompt_tokens={ctx}")
    print(f"wrote {out}")

    ok = ctx is not None
    if ctx is None:
        print("FAIL: text baseline call failed")
    else:
        for cohort in ("cxr", "mri", "oct", "derm"):
            sub = df[(df["cohort"] == cohort) & df["prompt_tokens"].notna()]
            if sub.empty:
                print(f"FAIL {cohort}: no successful call")
                ok = False
                continue
            ratio = sub["ratio_vs_text"].min()
            if ratio is not None and ratio < 1.5:
                print(f"FAIL {cohort}: prompt_tokens only {ratio}x the text "
                      "baseline; the relay may be dropping image_url")
                ok = False
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
