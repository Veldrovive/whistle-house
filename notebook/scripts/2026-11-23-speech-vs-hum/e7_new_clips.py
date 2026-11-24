"""E7: the Fifine clips recorded after the first investigation (ids not in old_ids.txt).
Unseen by every model trained before them; a later fine-tune sees their train split.

  .venv/bin/python scratch/svh/e7_new_clips.py <out-name> tag=checkpoint.pt ...
"""
import sys
import svh
from svh import groups, mic_jobs, run, summarize, with_variants

KEEP = ("clips", "minutes", "hum%", "hum%_of_voiced", "whistle%", "hum_ev/min", "hum_ev_s/min", "wh_ev/min", "match/min dirk>=0.1", "match/min dirk>=0.5", "match/min strict>=0.5")


def main():
    out = sys.argv[1]
    variants = [(a.split("=")[0], a.split("=")[1], []) for a in sys.argv[2:]]
    old = set((svh.HERE / "old_ids.txt").read_text().split())
    jobs = []
    for j in mic_jobs():
        if not j["group"].startswith("fifine"):
            continue
        new = j["id"] not in old
        held = j["meta"]["split"] != "train"
        label = j["group"].split(":")[1]
        tags = [f"{'new' if new else 'old'} {label}: all"]
        if held:
            tags.append(f"{'new' if new else 'old'} {label}: val+test")
            tags.append(f"all {label}: val+test")
        jobs += [{**j, "group": t} for t in tags]
    res = run(with_variants(jobs, variants), f"e7_{out}.pkl", workers=24)
    g = groups(res)
    order = sorted(g)
    print("=== hum % of frames")
    print(svh.table({f"{k} ({len(g[k])} clips, {sum(r['seconds'] for r in g[k]):.0f} s)": {t: summarize(g[k], t)["hum%"] for t, _, _ in variants} for k in order}))
    print("\n=== hum % of voiced frames")
    print(svh.table({k: {t: summarize(g[k], t)["hum%_of_voiced"] for t, _, _ in variants} for k in order if "hum" in k.split(":")[0]}))
    print("\n=== whistle % of frames")
    print(svh.table({k: {t: summarize(g[k], t)["whistle%"] for t, _, _ in variants} for k in order}))
    print("\n=== hum events per minute")
    print(svh.table({k: {t: summarize(g[k], t)["hum_ev/min"] for t, _, _ in variants} for k in order if "speech" in k or "noise" in k or "silence" in k}))
    for rk, thr in (("dirk", 0.1), ("dirk", 0.5), ("strict", 0.5)):
        print(f"\n=== tune matches per minute, recognizer {rk} >= {thr}")
        print(svh.table({k: {t: round(sum(1 for r in g[k] for m in r["variants"][t]["matches"][rk] if m[3] >= thr) / (sum(r["seconds"] for r in g[k]) / 60), 2) for t, _, _ in variants} for k in order if "speech" in k or "noise" in k or "silence" in k}))
    print("\n=== new speech, per clip: hum % of frames | hum events | matches (dirk settings, any score)")
    rows = {}
    for r in sorted(g.get("new speech: all", []), key=lambda r: r["id"]):
        rows[f"{r['id']} {r['meta']['split']:5s} {r['meta']['notes'][:22]:22s}"] = {
            t: f"{summarize([r], t)['hum%']:.1f} | {sum(e[0] == 'hum' for e in r['variants'][t]['events'])} | {' '.join(f'{m[3]:.2f}' for m in r['variants'][t]['matches']['dirk']) or '-'}" for t, _, _ in variants
        }
    if rows:
        print(svh.table(rows))
    print("\n=== thresholds, new clips: speech frames called hum % / voiced hum frames found %")
    rows = {}
    for t, _, _ in variants:
        rows[t] = {f"thr {thr}": f"{summarize(g['new speech: all'], t, thr)['hum%']:.1f} / {summarize(g['new hum: all'], t, thr)['hum%_of_voiced']:.0f}" for thr in (0.3, 0.5, 0.7, 0.9)}
    print(svh.table(rows))


if __name__ == "__main__":
    main()
