# 45 — Fetching large model weights through Chinese mirrors

> Recorded from a working, verified download of
> `cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit` (17.88 GB, 4 shards) on a host
> with **no usable HF proxy**. Every step below was necessary; none is optional.
> The result was verified by SHA256 against the official LFS hashes.

---

## 1. The winning combination

```text
mirror endpoint  +  disable Xet  +  single-threaded  +  chunked resume
                 +  low-speed timeout  +  outer retry  +  SHA256 verification
```

**Single-threaded is not a compromise, it is the fix.** Multi-threaded downloads
were the slowest configuration tried.

---

## 2. The five traps, in the order they bite

### 2.1 Direct `huggingface.co` is unreachable

`huggingface.co` times out. `hf-mirror.com` returns 200. Point the client at the
mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**Note for our own tooling:** the mirror's *API* path (`/api/models/...`) returned
403 in our tests, while its *file* path
(`/resolve/main/<file>`) worked. So "the mirror is down" is the wrong conclusion
to draw from an API 403 — test a file URL before giving up.

### 2.2 The `hf` CLI hangs under the mirror (Xet protocol)

Symptom: `Fetching 20 files: 15/20` then 0 bytes for minutes, while all small
files succeed. Cause: the mirror 302-redirects large files to
`cas-bridge.xethub.hf.co` (Xet storage) and the Xet path stalls.

```powershell
$env:HF_HUB_DISABLE_XET = "1"     # fall back to plain HTTP
```

This helps but does not fully fix it — the transfer still stalls after a while.
Hence the chunking below.

### 2.3 Concurrency gets rate-limited into uselessness

Measured: a single stream gives a stable 19–26 MB/s; four concurrent streams all
sit at high CPU with **zero** throughput. Counter-intuitive but reproducible.

**Use one stream.**

### 2.4 Long connections stall, and `curl` will wait forever

Symptom: after several GB the flow drops to zero and `curl` never exits, because
the default has no low-speed timeout. It sometimes restarts with a 0-byte temp
file, silently discarding progress.

Fix — **chunked download with `Range` requests**, ~100 MB per chunk, plus:

* `--speed-limit` / `--speed-time` so a stalled chunk dies after ~25 s;
* resume (`-C -`) so a retry continues instead of restarting;
* an outer retry loop;
* on signature-expiry errors (403 / 22 / 63) re-fetch a fresh signed URL.

A failed chunk then costs one chunk, not the whole file. Sustained throughput
with this setup: **~27 MB/s**.

### 2.5 PowerShell 5.1 mis-decodes a non-ASCII username in the path

Symptom: `Access to the path 'C:\Users\鐖昏垳\...' is denied` — the Chinese
characters in the user name were read as GBK.

Root cause: the `.ps1` was saved as UTF-8 **without BOM**, and PowerShell 5.1
parses scripts as ANSI unless a BOM is present.

Fix (either, but the first is more robust):

* **write no non-ASCII characters in the script at all**; build paths at runtime
  from `$env:USERPROFILE`;
* or save the script with a UTF-8 BOM.

---

## 3. Two self-inflicted bugs worth remembering

1. **`[Math]::Min` overflows on byte counts.** Passing a value above `Int32`
   range throws, which silently corrupts the computed chunk range. Cast to
   `[long]` explicitly.
2. **File size is not an integrity check.** A truncated or zero-padded shard has
   the wrong byte count only *sometimes*. Verify each shard against the official
   LFS **SHA256**; delete anything that fails and re-fetch that shard only.

---

## 4. Applying this to the project

We hit traps 2.1 and a variant of 2.3 with our own tooling:

* `scripts/28_ms_download.py` — direct ModelScope REST downloader, written because
  the bundled `modelscope` SDK in this environment raises
  `AttributeError: 'HubConfig' object has no attribute '_logged_out'` before it
  even reaches the network. It lists a repo's files, filters by regex, and streams
  with resume. Measured **17.3 MB/s** on ModelScope. Use it for anything mirrored
  there.
* Use the MethodScope path when the repo exists on ModelScope; fall back to
  `hf-mirror.com` **file** URLs (not the API) plus the chunked-curl approach above
  for HF-only repos such as the AWQ quantisations.

**Verified outcome:** `Qwen3-VL-30B-A3B-Instruct-AWQ4bit`, 17.88 GB, 4 shards,
all SHA256 matching, 56,466 tensors consistent with the index. Staged into
`program/models/` and registered as `qwen3_vl_30b_a3b_awq4bit`.

---

## 5. Comparability caveat for the paper

This model's weights are **pre-quantized** (`pack-quantized`, 4-bit, group 32,
compressed-tensors format). Every other model in the matrix uses on-the-fly
bitsandbytes NF4.

The two are not the same quantisation scheme, so a reviewer may ask whether the
30B's numbers are comparable. The planned handling:

* state the scheme explicitly per model in the model table;
* include at least one model measured under **both** schemes, so the
  quantisation-induced offset is estimated rather than assumed;
* frame the 30B as scale evidence, not as a like-for-like entry.
