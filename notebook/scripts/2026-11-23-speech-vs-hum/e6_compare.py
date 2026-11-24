"""E6: one table per checkpoint: speech-as-hum and hum / whistle detection on the Fifine clips,
the Mac clips, and external val data raw and pushed through the measured Fifine response.

  .venv/bin/python scratch/svh/e6_compare.py <out-name> tag=checkpoint.pt [tag=checkpoint.pt ...]
"""
import sys
import numpy as np
import svh
from e5_eq import ltas
from svh import external_jobs, groups, mic_jobs, run, summarize, with_variants

KEEP = ("clips", "minutes", "hum%", "hum%_of_voiced", "whistle%", "hum_ev/min", "wh_ev/min", "match/min dirk>=0.1", "match/min strict>=0.5")


def main():
    out = sys.argv[1]
    quick = out.startswith("quick")  # Fifine clips + a little MUSAN speech only, few workers
    variants = [(a.split("=")[0], a.split("=")[1], []) for a in sys.argv[2:]]
    mic = [j for j in mic_jobs() if not quick or j["group"].startswith("fifine")]
    sp = [j for j in mic if j["group"].endswith(":speech")]
    fif = [j for j in sp if j["group"].startswith("fifine")]
    mus = external_jobs("musan", "speech", "val", 60, seed=1)
    grid, lf = ltas([j for j in mic_jobs("speech") if j["group"].startswith("fifine")])
    _, lmu = ltas(mus[:30])
    ref = (grid > 90) & (grid < 130)
    d = lf - lmu
    d -= d[ref].mean()
    eq = ("eqcurve", grid.tolist(), np.clip(d, -30, 30).tolist(), "keep_rms")
    hums = external_jobs("mlend_hums", "hum", "val", 60, seed=5)
    whs = external_jobs("mlend_whistles", "whistle", "val", 40, seed=6)
    jobs = []
    for j in mic:
        split = j["meta"]["split"]
        held = "" if j["group"].startswith("fifine") else (" (held out)" if split != "train" else " (train)")
        jobs.append({**j, "group": j["group"] + held})
        if j["group"].startswith("fifine") and split != "train":  # unseen by a mic fine-tune
            jobs.append({**j, "group": j["group"] + " (val+test clips only)"})
    # robustness on the Fifine clips: other levels and a darker / brighter position in the room
    if not quick:
        fs = [j for j in mic if j["group"] == "fifine:speech"]
        fh = [j for j in mic if j["group"] == "fifine:hum"]
        fw = [j for j in mic if j["group"] == "fifine:whistle"]
        for name, base, opsets in (
            ("fifine:speech", fs, {"+12 dB": [("gain", 12)], "+24 dB": [("gain", 24)], "-12 dB": [("gain", -12)], "tilt -3 dB/oct": [("tilt", -3, 110)], "tilt +3 dB/oct": [("tilt", 3, 110)], "reverb 0.5 s / DRR 0": [("reverb", 0.5, 0)]}),
            ("fifine:hum", fh, {"+12 dB": [("gain", 12)], "-12 dB": [("gain", -12)], "tilt -3 dB/oct": [("tilt", -3, 110)], "tilt +3 dB/oct": [("tilt", 3, 110)], "reverb 0.5 s / DRR 0": [("reverb", 0.5, 0)]}),
            ("fifine:whistle", fw, {"-12 dB": [("gain", -12)], "+12 dB": [("gain", 12)]}),
        ):
            for on, ops in opsets.items():
                jobs += [{**j, "group": f"{name} {on}", "ops": ops, "seed": i} for i, j in enumerate(base)]
    if quick:
        mus, hums, whs = mus[:24], hums[:16], []
    for name, base in (("musan speech val", mus), ("mlend hums val", hums), ("mlend whistles val", whs)):
        for on, ops in (("raw", []), ("+fifine response", [eq]), ("+reverb 0.5 s / DRR 0", [("reverb", 0.5, 0)]), ("+both", [eq, ("reverb", 0.5, 0)])):
            if on != "raw" and "whistle" in name and "both" not in on:
                continue
            jobs += [{**j, "group": f"{name} {on}", "ops": ops, "seed": i} for i, j in enumerate(base)]
    if not quick:
        jobs += external_jobs("musan", "music", "val", 40, seed=3) + external_jobs("musan", "noise", "val", 40, seed=4)
    res = run(with_variants(jobs, variants), f"e6_{out}.pkl", workers=12 if quick else 32)
    g = groups(res)
    order = sorted(g)
    for tag, _, _ in variants:
        print(f"\n=== {tag}")
        print(svh.table({k: {kk: vv for kk, vv in summarize(g[k], tag).items() if kk in KEEP} for k in order}))
    # compact: one row per group, hum% per variant (frames) and hum events / min
    print("\n=== hum % of frames, by checkpoint")
    print(svh.table({k: {t: summarize(g[k], t)["hum%"] for t, _, _ in variants} for k in order}))
    print("\n=== whistle % of frames, by checkpoint")
    print(svh.table({k: {t: summarize(g[k], t)["whistle%"] for t, _, _ in variants} for k in order}))
    # operating point: speech frames called hum against hum frames found, per presence threshold
    print("\n=== threshold sweep on the Fifine clips: speech frames called hum % / voiced hum frames found %")
    rows = {}
    for t, _, _ in variants:
        row = {}
        for thr in (0.3, 0.5, 0.7, 0.9):
            a = summarize(g["fifine:speech"], t, thr)["hum%"]
            b = summarize(g["fifine:hum"], t, thr)["hum%_of_voiced"]
            row[f"thr {thr}"] = f"{a:.1f} / {b:.0f}"
        rows[t] = row
    print(svh.table(rows))
    print("\n=== Fifine speech: hum events/min, tune matches/min (dirk settings >=0.1, >=0.5; default settings >=0.5)")
    print(svh.table({t: {k: v for k, v in summarize(g["fifine:speech"], t).items() if "min" in k and "wh_" not in k} for t, _, _ in variants}))
    print("\n=== Fifine hums + whistles recorded for a tune: matches per clip (default settings >=0.5)")
    rows = {}
    for t, _, _ in variants:
        row = {}
        for grp in ("fifine:hum", "fifine:whistle"):
            clips = [r for r in g[grp] if r["meta"].get("tune")]
            hit = sum(any(m[3] >= 0.5 for m in r["variants"][t]["matches"]["strict"]) for r in clips)
            row[grp] = f"{hit}/{len(clips)}"
        rows[t] = row
    print(svh.table(rows))


if __name__ == "__main__":
    main()
