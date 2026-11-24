"""E3: per-clip rates, reverberance proxy per corpus, and what the false-hum frames look like."""
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
import svh

res = pickle.loads((svh.HERE / "results/e1_n150.pkl").read_bytes())
g = svh.groups(res)
tag = "long-e240"
print("per-clip, fifine speech + hum")
for r in g["fifine:speech"] + g["fifine:hum"] + g["mac:speech"][:4]:
    s = svh.summarize([r], tag); s2 = svh.summarize([r], "long-e240+gate")
    print(r["group"], r["id"], r["meta"].get("split"), f"{r['seconds']:.0f}s rms {r['rms_db']:.0f}", "hum%", s["hum%"], "of voiced", s["hum%_of_voiced"], "ev/min", s["hum_ev/min"], "| gate hum%", s2["hum%"])

def decay(path, start=0.0, dur=None):
    """Free-decay rate of the 300-3000 Hz envelope: 90th percentile of the fall (dB/s) over
    100 ms steps, among steps starting >15 dB above the clip's floor. A dry recording falls as
    fast as the voice stops; a reverberant one cannot fall faster than 60/RT60 dB/s."""
    x = svh.load(path, start, dur)
    x = sosfilt(butter(4, [300, 3000], "band", fs=svh.SR, output="sos"), x)
    hop = 441
    e = 10 * np.log10(np.convolve(x**2, np.ones(882) / 882, "same")[::hop] + 1e-12)
    floor = np.percentile(e, 10)
    k = 10  # 100 ms
    d = (e[:-k] - e[k:]) / 0.1
    ok = (e[:-k] > floor + 20) & (e[k:] > floor + 3)
    if ok.sum() < 20: return np.nan
    return float(np.percentile(d[ok], 90))

from concurrent.futures import ProcessPoolExecutor
import json
def job(a): return decay(*a)
if __name__ == "__main__":
    sets = {}
    for grp in ("fifine:speech","mac:speech","fifine:hum","mac:hum","musan:speech:val","mlend_hums:hum:val","fifine:whistle","mac:whistle","mlend_whistles:whistle:val"):
        rs = g[grp][:60]
        args = []
        for r in rs:
            if ":" in grp and grp.split(":")[0] in ("fifine","mac"):
                args.append((str(svh.DATA / "recordings" / f"{r['id']}.wav"),))
            else:
                path = r["id"].split("@")[0]; start = float(r["id"].split("@")[1].split("#")[0])
                ds = grp.split(":")[0]
                args.append((str(svh.EXT / "prepared" / ds / path), start, 30.0))
        with ProcessPoolExecutor(16) as p:
            v = np.array(list(p.map(job, args)))
        v = v[np.isfinite(v)]
        if not len(v):
            print(grp, 'no usable decays'); continue
        print(f"{grp:28s} n={len(v):3d} decay dB/s p25/50/75: {np.percentile(v,25):6.0f} {np.percentile(v,50):6.0f} {np.percentile(v,75):6.0f}  -> RT60-equivalent {60/np.percentile(v,50):.2f} s")

    # what do false-hum frames look like? pitch steadiness and position in the voiced run
    def frame_feats(r):
        f0 = r["f0"]; v = f0 > 0
        st = np.where(v, 12 * np.log2(np.maximum(f0, 1) / 100), np.nan)
        k = 9  # ~100 ms
        slope = np.full(len(st), np.nan)
        slope[k:] = np.abs(st[k:] - st[:-k]) / (k / 86.13)   # st/s over the last 100 ms
        # position: time since the voiced run began, time until it ends
        since = np.zeros(len(v)); until = np.zeros(len(v)); c = 0
        for i in range(len(v)):
            c = c + 1 if v[i] else 0; since[i] = c / 86.13
        c = 0
        for i in range(len(v) - 1, -1, -1):
            c = c + 1 if v[i] else 0; until[i] = c / 86.13
        return v, slope, since, until, since + until
    for grp in ("fifine:speech", "musan:speech:val", "mac:speech", "fifine:hum", "mlend_hums:hum:val"):
        V=[];S=[];L=[];P=[];LV=[];F=[]
        for r in g[grp]:
            v, slope, since, until, runlen = frame_feats(r)
            p = r["variants"][tag]["p_hum"].astype(np.float32)
            V.append(v); S.append(slope); L.append(runlen); P.append(p); LV.append(r["frame_max_db"]); F.append(r["f0"])
        V=np.concatenate(V);S=np.concatenate(S);L=np.concatenate(L);P=np.concatenate(P);F=np.concatenate(F)
        fp = P >= 0.5
        print(f"\n{grp}: frames {len(P)}, voiced {V.mean():.2f}, hum {fp.mean():.3f}; of hum frames, pYIN-voiced: {V[fp].mean() if fp.any() else float('nan'):.2f}")
        vv = V & np.isfinite(S)
        for name, x, edges in (("pitch slope st/s", S, [0, 2, 5, 10, 20, 1e9]), ("voiced-run length s", L, [0, .15, .3, .6, 1.2, 1e9]), ("f0 Hz", F, [60, 100, 130, 170, 230, 600])):
            row = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = vv & (x >= a) & (x < b)
                row.append(f"[{a:g},{b:g}): share {m.sum()/max(vv.sum(),1):.2f} hum {P[m].__ge__(0.5).mean() if m.any() else float('nan'):.3f}")
            print("  ", name, " | ".join(row))
