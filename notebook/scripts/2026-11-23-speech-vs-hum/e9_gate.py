"""E9: models trained with gated examples (`--augment room-gate`), with and without a gate in
the pipeline. Every checkpoint is run three ways over the recordings:

  (plain)  no filter
  +g6      adaptive_gate over all six Fifine profiles (the pipeline on dirk, incl. `weird-hum`)
  +g5      the same without `weird-hum`: the room has a hum the gate has no profile of

and, for checkpoints with a speech output, also with a 1 s speech hold.

  .venv/bin/python scratch/svh/e9_gate.py <out-name> tag=checkpoint.pt ...
"""

import sys

import svh
from svh import external_jobs, groups, mic_jobs, run, summarize, with_variants

PROFILES = [
    "fifine",
    "fifine-fan-door-open",
    "fifine-fan-door-closed",
    "fifine-fan-both-doors-closed",
    "fifine-fridge",
    "weird-hum",
]
WEIRD = {"a73ef3fe863c", "34809240aec2", "84d4029aa168"}  # recorded with the 120 Hz hum on


def gate(profiles):
    return {"name": "adaptive_gate", "params": {"profiles": profiles, "dwell_s": 60}}


FRONTS = {"": [], "+g6": [gate(PROFILES)], "+g5": [gate(PROFILES[:-1])]}


def has_speech(ckpt: str) -> bool:
    import torch

    ck = torch.load(svh.DATA / "models" / ckpt, map_location="cpu", weights_only=False)
    return "speech" in ck["arch"].get("aux", ())


def main():
    out = sys.argv[1]
    models = [a.split("=") for a in sys.argv[2:]]
    variants, held = [], []
    for tag, ckpt in models:
        sp = has_speech(ckpt)
        for fk, filters in FRONTS.items():
            variants.append((tag + fk, ckpt, filters))
            if sp:
                variants.append((f"{tag}{fk} hold1", ckpt, filters, {"speech_hold_s": 1.0}))
                held.append(f"{tag}{fk} hold1")
    plain = [v[0] for v in variants if " hold" not in v[0]]
    jobs = []
    for j in mic_jobs():
        mic, label = j["group"].split(":")
        if mic == "fifine" and j["id"] in WEIRD:
            jobs.append({**j, "group": "fifine: 120 Hz hum on, nobody there (3 clips)"})
            continue
        jobs.append(j)
        if mic == "fifine" and j["meta"]["split"] != "train":
            jobs.append({**j, "group": f"fifine:{label} (val+test clips)"})
    for name, ds, label, n, seed in (
        ("musan speech val", "musan", "speech", 40, 1),
        ("musan music val", "musan", "music", 20, 3),
        ("musan noise val", "musan", "noise", 20, 4),
        ("mlend hums val", "mlend_hums", "hum", 40, 5),
        ("mlend whistles val", "mlend_whistles", "whistle", 20, 6),
    ):
        jobs += [{**j, "group": name} for j in external_jobs(ds, label, "val", n, seed=seed)]
    res = run(with_variants(jobs, variants), f"e9_{out}.pkl", workers=32)
    g = groups(res)
    order = sorted(g)

    def col(metric, tags, pick=None):
        return svh.table(
            {
                f"{k} ({len(g[k])}, {sum(r['seconds'] for r in g[k]) / 60:.1f} min)": {
                    t: summarize(g[k], t).get(metric, "-") for t in tags
                }
                for k in order
                if pick is None or pick(k)
            }
        )

    def neg(k):
        return not ("hum" in k.split(" (")[0].split(":")[-1] or "whistle" in k)

    print("=== hum presence >= 0.5, % of frames")
    print(col("hum%", plain))
    print("\n=== hum presence >= 0.5, % of voiced frames (pYIN), hum clips")
    print(col("hum%_of_voiced", plain, lambda k: "hum" in k and "120" not in k))
    print("\n=== whistle presence >= 0.5, % of frames")
    print(col("whistle%", plain))
    print("\n=== hum events per minute")
    print(col("hum_ev/min", plain))
    print("\n=== hum events per minute, 1 s speech hold")
    print(col("hum_ev/min", held))
    print("\n=== seconds of hum events per minute")
    print(col("hum_ev_s/min", plain))
    print("\n=== seconds of hum events per minute, 1 s speech hold")
    print(col("hum_ev_s/min", held))
    print("\n=== whistle events per minute")
    print(col("wh_ev/min", plain))
    print("\n=== speech output on, % of frames")
    print(col("speech%", [t for t in plain if t + " hold1" in held]))
    for rk, thr in (("dirk", 0.1), ("dirk", 0.5), ("strict", 0.5)):
        print(f"\n=== tune matches per minute, recognizer {rk} >= {thr}")
        print(col(f"match/min {rk}>={thr}", plain + held, neg))
    print("\n=== clips recorded for a tune: matched (default recognizer settings, >= 0.5)")
    rows = {}
    for t in plain + held:
        row = {}
        for grp in ("fifine:hum", "fifine:whistle"):
            clips = [r for r in g.get(grp, []) if r["meta"].get("tune")]
            hit = sum(any(m[3] >= 0.5 for m in r["variants"][t]["matches"]["strict"]) for r in clips)
            row[grp] = f"{hit}/{len(clips)}"
        rows[t] = row
    print(svh.table(rows))
    print("\n=== the clips with the 120 Hz hum: hum% of frames | hum events")
    rows = {}
    for r in sorted(g["fifine: 120 Hz hum on, nobody there (3 clips)"], key=lambda r: r["id"]):
        rows[f"{r['id']} {r['seconds']:.0f} s"] = {
            t: f"{summarize([r], t)['hum%']} | "
            f"{sum(e[0] == 'hum' for e in r['variants'][t]['events'])}"
            for t in plain
        }
    print(svh.table(rows))


if __name__ == "__main__":
    main()
