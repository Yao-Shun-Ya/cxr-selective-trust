# Frontier-API probe (`scripts/api_probe/`)

A deliberately isolated arm that measures one closed-weight multimodal model on
the same three CXR layers and the same three non-CXR modalities as the local
matrix, without touching any existing artifact.

## Isolation contract

| Rule | Implementation |
|---|---|
| New outputs only | everything lands in `outputs/api_probe/<model_slug>/` |
| Existing artifacts read-only | manifests are read with `pandas`, never rewritten |
| Shared protocol | images, prompts, parser and `DurableJsonl` are imported from `src/cxr_trust` |
| No secrets on disk | the key is read from `OPENAI_API_KEY`; nothing logs it |

## Pipeline

```bash
export OPENAI_API_KEY=...          # never commit this
export OPENAI_BASE_URL=https://amr-link.open-design.ai/v1
export OPENAI_MODEL=deepseek-v4.1-flash

python 01_smoke_test.py            # 8 image calls + 1 text baseline
python 02_build_probe_manifest.py --profile full_multimodal   # 9,400 rows
python 03_run_api_probe.py --dry-run                          # 20 rows
python 03_run_api_probe.py --profile full_multimodal          # resumable
python 04_audit_api_results.py                                # P1-P4 CSVs
```

`--profile lite` builds 2,700 rows for tighter quotas.

## What the smoke test actually checks

A relay that silently drops `image_url` still returns HTTP 200 and valid JSON.
The only cheap signal is `usage.prompt_tokens`: the test compares every image
call against a text-only call with the same prompt and fails the run if the
ratio is below 1.5x. Measured ratios on this relay: 1.93x (MRI), 2.16x
(dermoscopy), 2.73x (OCT), 4.64x (CXR).

## Two endpoint behaviours worth knowing

1. **`max_tokens` must cover hidden reasoning.** At 1,024 tokens the model spends
   the whole budget on its trace and returns `choices: []` with
   `completion_tokens = 1024`. At 4,096 roughly one CXR cell in sixteen still
   died; the runner uses 8,192 and records `finish_reason = empty_choices`
   rather than pretending the call succeeded.
2. **TLS through `httpx` fails on this gateway** (`EOF occurred in violation of
   protocol`), while `curl` and `urllib` negotiate normally. The client is the
   standard library in a worker thread, not `httpx`; see `common._post_sync`.

## Concurrency and durability

- AIMD: start 8, +1 after 50 consecutive successes (cap 10), halve on
  429/502/503/timeout (floor 2), exponential backoff, five attempts.
- One `asyncio.Queue`, one consumer, one `DurableJsonl`: concurrent workers never
  touch the file. A rerun skips every durable `probe_id`.
- `provenance.json` fingerprints the model, manifest SHA-256, prompt set, image
  geometry and token budget; a mismatch refuses the resume instead of mixing
  two experiments.

## Logs

| File | Contents |
|---|---|
| `01_smoke_test.log` | token ratios, parse status, sent image sizes |
| `run_console.log` | per-50-call progress: rate, error count, AIMD limit, ETA |
| `03_run_api_probe.log` | same, plus every retry with exception type |
| `events.jsonl` | reserved for per-request events |
| `run_summary.json` | rows, ok/error, wall time, final limit, provenance id |
