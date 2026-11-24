"""Frame-level replay of the learned detector speech hold from saved scores (e8 pickle):
share of hum frames (p_hum >= 0.5) that survive, for speech_threshold x speech_min_s x hold."""
import pickle, sys, numpy as np
from e8_speech_head import FOLDS
res = pickle.load(open(f"results/e8_{sys.argv[1]}.pkl","rb"))
tag = sys.argv[2]
PERIOD = 512/44100
def survive(ps, ph, thr, min_s, hold_s):
    until, since, keep = -1e9, None, np.zeros(len(ph), bool)
    for i in range(len(ph)):
        t = i*PERIOD
        if ps[i] >= thr:
            since = t if since is None else since
            if t - since >= min_s: until = t + hold_s
        else: since = None
        keep[i] = ph[i] >= 0.5 and t > until
    return keep
def groups():
    g = {}
    for r in res:
        if "(" in r["group"] and "left out" not in r["group"]: continue
        g.setdefault(r["group"], []).append(r)
    g["fifine:hum fb3c (fast melody, test)"] = [r for r in res if r["id"]=="fb3c1d2fe46f" and r["group"]=="fifine:hum"]
    g["fifine:hum other val+test"] = [r for r in res if r["group"]=="fifine:hum (val+test clips)" and r["id"]!="fb3c1d2fe46f"]
    return g
g = groups()
want = [k for k in g if k.startswith("fifine:speech") or k.startswith("fifine:hum") or k in ("mlend hums val","musan speech val","mac:speech","mac:hum","musan music val")]
cfgs = [(None,0,0)] + [(thr, m, h) for thr in (0.5, 0.7, 0.9) for m in (0.1, 0.3, 0.6) for h in (0.5, 1.0)]
print(tag, "hum frames kept, % of all frames (hum clips: % of voiced frames)")
print(" "*44 + " ".join(("no hold" if c[0] is None else "t%.1f m%.1f h%.1f" % c).rjust(14) for c in cfgs))
for k in sorted(want):
    row = []
    for thr, m, h in cfgs:
        num = den = 0
        for r in g[k]:
            v = r["variants"][tag + " hold0"]
            ps, ph = v["p_speech"].astype(float), v["p_hum"].astype(float)
            keep = ph >= 0.5 if thr is None else survive(ps, ph, thr, m, h)
            if "hum" in k.split(" ")[0] or k.startswith("mlend hums"):
                vo = r["f0"] > 0; num += (keep & vo).sum(); den += vo.sum()
            else:
                num += keep.sum(); den += len(keep)
        row.append("%.2f" % (100*num/max(den,1)))
    print(k[:43].ljust(44) + " ".join(x.rjust(14) for x in row))
