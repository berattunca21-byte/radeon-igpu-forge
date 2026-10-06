"""fxlm.prepare_bytes: metin dosyasi/glob -> byte-level token .bin.

vocab=256 (bayt degerleri). Yazar:
    <out>_train.bin, <out>_val.bin  (uint16)
    <out>.json  {vocab_size, tokenizer:"bytes", n_train, n_val}

Ornek:
    python ml\\fxlm\\prepare_bytes.py --out data\\demo "C:\\...\\Lib\\*.py"
Bagimliliksiz (numpy+stdlib).
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="metin dosyasi, dizin veya glob (birden cok olur)")
    ap.add_argument("--out", required=True,
                    help="cikis on-eki: <out>_train.bin + <out>.json")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="0 -> sinirsiz; buyuk kaynaklarda kesme")
    ap.add_argument("--ext", default=".txt,.md,.py",
                    help="dizin taramasinda alinacak uzantilar "
                         "(virgulle; acikca verilen dosya/glob'lara "
                         "uygulanmaz)")
    args = ap.parse_args()

    exts = tuple(e.strip().lower() for e in args.ext.split(",") if e.strip())
    files = []
    for pat in args.inputs:
        if os.path.isdir(pat):
            for root, _d, names in os.walk(pat):
                for n in names:
                    if n.lower().endswith(exts):
                        files.append(os.path.join(root, n))
        else:
            hits = sorted(glob.glob(pat))
            files += hits if hits else [pat]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit("girdi dosyasi bulunamadi: " + " ".join(args.inputs))

    # binary eleme: ilk 8 KB'ta NUL varsa metin degildir
    kept, skipped = [], 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                head = fh.read(8192)
        except OSError:
            skipped += 1
            continue
        if b"\x00" in head:
            skipped += 1
            continue
        kept.append(f)
    files = kept
    if skipped:
        print(f"{skipped} binary dosya atlandi")
    if not files:
        raise SystemExit("filtre sonrasi metin dosyasi kalmadi")
    print(f"{len(files)} dosya okunuyor...")

    chunks = []
    total = 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if args.max_bytes and total + len(data) > args.max_bytes:
            data = data[:args.max_bytes - total]
        chunks.append(data)
        total += len(data)
        if args.max_bytes and total >= args.max_bytes:
            break
    raw = b"".join(chunks)
    toks = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    n = toks.size
    n_val = max(1, int(n * args.val_frac))
    n_tr = n - n_val
    os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                exist_ok=True)
    tr_p, va_p = args.out + "_train.bin", args.out + "_val.bin"
    toks[:n_tr].tofile(tr_p)
    toks[n_tr:].tofile(va_p)
    meta = {"vocab_size": 256, "tokenizer": "bytes",
            "n_train": int(n_tr), "n_val": int(n_val),
            "n_files": len(files)}
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"yazildi: {tr_p} ({n_tr:,} token) | {va_p} ({n_val:,}) | "
          f"{args.out}.json")


if __name__ == "__main__":
    main()
