"""fxlm.sample: generate text from a checkpoint.

    python ml\\fxlm\\sample.py --ckpt OUT\\ckpt_last.pt --prompt "..." \
        --max-new 200 --temperature 0.8 --top-k 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxlm.env import setup_lm_env  # noqa: E402 — before torch
setup_lm_env(tunable=False)

import torch  # noqa: E402

from fxlm.env import require_gpu  # noqa: E402
from fxlm.model import GPT, GPTConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = require_gpu()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**sd["config"])
    meta = sd.get("meta", {})
    tok = meta.get("tokenizer", "bytes")

    model = GPT(cfg).to(device)
    # load from the fp32 master, then half()
    with torch.no_grad():
        msd = {k: v.to(device) for k, v in sd["model"].items()}
        missing = []
        for name, p in model.named_parameters():
            if name in msd:
                p.data.copy_(msd[name].to(p.dtype))
            else:
                missing.append(name)
        if missing:
            print("missing parameters:", missing)
    model.half().eval()

    if tok == "bytes":
        raw = list(args.prompt.encode("utf-8"))
        if not raw:
            raw = [10]            # empty prompt: seed with "\n"
        ids = torch.tensor(raw, dtype=torch.long, device=device)
        decode = lambda t: bytes(t.tolist()).decode("utf-8",
                                                   errors="replace")
    else:
        raise SystemExit(f"no encode/decode defined for tokenizer '{tok}' "
                         "(only 'bytes')")

    x = ids.view(1, -1)
    for _ in range(args.max_new):
        xc = x[:, -cfg.seq_len:]
        with torch.inference_mode():
            logits = model(xc)[:, -1, :].float()
        logits = logits / max(args.temperature, 1e-5)
        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.shape[-1]))
            logits = logits.masked_fill(logits < v[:, -1:], -float("inf"))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)

    out = x[0, ids.numel():]
    sys.stdout.write(decode(out))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
