"""E8: a model with a speech output (`--aux speech`), and what `speech_hold_s` on the learned
detector does with it. For every checkpoint: how often the speech output is on over speech,
hums, whistles and the rest (it must stay off on hums, or it would hold real hums back), and
hum events / tune matches with the hold at several lengths.

  .venv/bin/python scratch/svh/e8_speech_head.py <out-name> tag=checkpoint.pt ... [holds=0,0.5,1,2]
  a checkpoint without a speech output is run without the hold (baseline).
"""
import sys

import svh
from svh import external_jobs, groups, mic_jobs, run, summarize, with_variants

FOLDS = {  # clips each leave-out fine-tune did not see (data-foldA / data-foldB)
    "foldA": {"192d159e773e", "ace0fcf83406", "cf109ad6a79c"},
    "foldB": {"1405acf41f03", "1de8a07ce00f", "dfd73785c931"},
}


def has_speech(ckpt: str) -> bool:
    import torch

    ck = torch.load(svh.DATA / "models" / ckpt, map_location="cpu", weights_only=False)
    return "speech" in ck["arch"].get("aux", ())


def main():
    out = sys.argv[1]
    holds = [0.0, 0.5, 1.0, 2.0]
    models = []
    for a in sys.argv[2:]:
        k, v = a.split("=")
        if k == "holds":
            holds = [float(x) for x in v.split(",")]
        else:
            models.append((k, v))
    variants = []
    for tag, ckpt in models:
        if not has_speech(ckpt):
            variants.append((tag, ckpt, []))
            continue
        for h in holds:
            variants.append((f"{tag} hold{h:g}", ckpt, [], {"speech_hold_s": h}))
    tags = [v[0] for v in variants]
    old = set((svh.HERE / "old_ids.txt").read_text().split())
    jobs = []
    for j in mic_jobs():
        mic, label = j["group"].split(":")
        jobs.append(j)
        if mic != "fifine":
            continue
        if j["meta"]["split"] != "train":
            jobs.append({**j, "group": f"fifine:{label} (val+test clips)"})
        for fold, ids in FOLDS.items():
            if j["id"] in ids:
                jobs.append({**j, "group": f"fifine:{label} (left out of {fold})"})
        if j["id"] not in old and label == "speech":
            jobs.append({**j, "group": "fifine:speech (recorded after the first models)"})
    for name, ds, label, n, seed in (
        ("musan speech val", "musan", "speech", 60, 1),
        ("musan music val", "musan", "music", 30, 3),
        ("musan noise val", "musan", "noise", 30, 4),
        ("mlend hums val", "mlend_hums", "hum", 60, 5),
        ("mlend whistles val", "mlend_whistles", "whistle", 30, 6),
    ):
        jobs += [{**j, "group": name} for j in external_jobs(ds, label, "val", n, seed=seed)]
    res = run(with_variants(jobs, variants), f"e8_{out}.pkl", workers=32)
    g = groups(res)
    order = sorted(g)

    def col(metric, pick=None):
        return svh.table(
            {
                f"{k} ({len(g[k])}, {sum(r['seconds'] for r in g[k]) / 60:.1f} min)": {
                    t: summarize(g[k], t).get(metric, "-") for t in tags
                }
                for k in order
                if pick is None or pick(k)
            }
        )

    first = [t for t in tags if t.endswith("hold0") or " hold" not in t]
    print("=== speech output on, % of frames (hold does not change it)")
    print(svh.table({k: {t: summarize(g[k], t).get("speech%", "-") for t in first} for k in order}))
    print("\n=== speech output on, % of voiced frames (pYIN)")
    print(svh.table({k: {t: summarize(g[k], t).get("speech%_of_voiced", "-") for t in first} for k in order}))
    print("\n=== hum presence >= 0.5, % of frames (before any hold)")
    print(svh.table({k: {t: summarize(g[k], t)["hum%"] for t in first} for k in order}))
    print("\n=== hum events per minute")
    print(col("hum_ev/min"))
    print("\n=== seconds of hum events per minute")
    print(col("hum_ev_s/min"))
    print("\n=== whistle events per minute")
    print(col("wh_ev/min", lambda k: "whistle" in k or k.startswith("fifine:speech")))
    for rk, thr in (("dirk", 0.1), ("dirk", 0.5), ("strict", 0.5)):
        print(f"\n=== tune matches per minute, recognizer {rk} >= {thr}")
        print(col(f"match/min {rk}>={thr}", lambda k: not ("hum" in k or "whistle" in k)))
    print("\n=== clips recorded for a tune: matched (default recognizer settings, >= 0.5)")
    rows = {}
    for t in tags:
        row = {}
        for grp in ("fifine:hum", "fifine:whistle", "mac:hum", "mac:whistle"):
            clips = [r for r in g.get(grp, []) if r["meta"].get("tune")]
            hit = sum(any(m[3] >= 0.5 for m in r["variants"][t]["matches"]["strict"]) for r in clips)
            row[grp] = f"{hit}/{len(clips)}"
        rows[t] = row
    print(svh.table(rows))
    print("\n=== Fifine speech per clip: speech% | hum% | hum events at each hold")
    rows = {}
    for r in sorted(g["fifine:speech"], key=lambda r: r["id"]):
        row = {}
        for tag, _ in models:
            ts = [t for t in tags if t == tag or t.startswith(tag + " hold")]
            s0 = summarize([r], ts[0])
            ev = "/".join(str(sum(e[0] == "hum" for e in r["variants"][t]["events"])) for t in ts)
            row[tag] = f"{s0.get('speech%', '-')} | {s0['hum%']} | {ev}"
        rows[f"{r['id']} {r['meta']['split']:5s} {(r['meta']['notes'] or '')[:24]:24s}"] = row
    print(svh.table(rows))
    print("\n=== Fifine hums per clip: speech% | hum% of voiced | hum event seconds at each hold")
    rows = {}
    for r in sorted(g["fifine:hum"], key=lambda r: r["id"]):
        row = {}
        for tag, _ in models:
            ts = [t for t in tags if t == tag or t.startswith(tag + " hold")]
            s0 = summarize([r], ts[0])
            ev = "/".join(f"{sum(e[2] - e[1] for e in r['variants'][t]['events'] if e[0] == 'hum'):.1f}" for t in ts)
            row[tag] = f"{s0.get('speech%', '-')} | {s0['hum%_of_voiced']} | {ev}"
        rows[f"{r['id']} {r['meta']['split']:5s} {(r['meta']['notes'] or '')[:24]:24s}"] = row
    print(svh.table(rows))


if __name__ == "__main__":
    main()
