"""Why does CheXagent-8b ignore ``torch_dtype=torch.float16``?

Fix A (pin ``device_map={"": 0}``) removed the CPU offload but did NOT reduce the
footprint: the card filled to 31.2 GiB -- the fp32 size of a checkpoint whose
on-disk metadata says ``torch_dtype: float32`` -- and the first forward pass went
straight into the WDDM paging regime (100% "utilisation", 82 W, ~2 s per token).

So the real question is not "which device map" but "which dtype actually landed".
This probe loads the model with three kwarg styles and reports what is on the
card, without ever running a forward pass (a forward on a mis-loaded model is
what makes the desktop unresponsive):

    plain      torch_dtype=torch.float16, device_map={"": 0}
    lowcpumem  the same + low_cpu_mem_usage=True
    newdtype   dtype=torch.float16 (the 4.5x-style kwarg), device_map={"": 0}

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\_diag_chexagent_dtype.py --mode plain
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "lowcpumem", "newdtype"], default="plain")
    ap.add_argument("--model", default=str(ROOT / "models" / "CheXagent-8b"))
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    print(f"[{args.mode}] gpu before: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")

    kwargs = dict(trust_remote_code=True, device_map={"": 0})
    if args.mode == "plain":
        kwargs["torch_dtype"] = torch.float16
    elif args.mode == "lowcpumem":
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True
    else:
        kwargs["dtype"] = torch.float16
    print(f"[{args.mode}] kwargs: {{{', '.join(f'{k}={v}' for k, v in kwargs.items())}}}")

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    by_dtype: dict[str, int] = collections.Counter()
    for _, p in model.named_parameters():
        by_dtype[str(p.dtype)] += p.numel() * p.element_size()
    total_bytes = sum(by_dtype.values())
    resident = torch.cuda.memory_allocated() / 2**30

    print(f"[{args.mode}] config.torch_dtype      : {model.config.torch_dtype}")
    print(f"[{args.mode}] parameter bytes by dtype :")
    for dt, nb in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
        print(f"[{args.mode}]     {dt:16s} {nb / 2**30:7.2f} GiB")
    print(f"[{args.mode}] parameter total         : {total_bytes / 2**30:.2f} GiB")
    print(f"[{args.mode}] cuda memory_allocated   : {resident:.2f} GiB")
    verdict = "fp16 OK" if total_bytes < 20 * 2**30 else "STILL fp32 (dtype ignored)"
    print(f"[{args.mode}] VERDICT: {verdict}")

    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
