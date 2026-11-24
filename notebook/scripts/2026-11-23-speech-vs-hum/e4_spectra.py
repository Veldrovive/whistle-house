"""E4: what the model sees at the true pitch. For voiced frames (pYIN f0), the front-end
spectrum (dB) sampled at h*f0 for h = 1..8 and at the midpoints between harmonics, per group,
and for the false-hum frames of the Fifine speech against its other voiced frames."""
import pickle
import numpy as np
import svh
from concurrent.futures import ProcessPoolExecutor

tag = "long-e240"
H = np.array([1, 2, 3, 4, 5, 6, 7, 8.0]); G = H[:-1] + 0.5

def feats(r):
    svh._limit_threads()
    grp = r["group"]
    if grp.split(":")[0] in ("fifine", "mac"):
        x = svh.load(str(svh.DATA / "recordings" / f"{r['id']}.wav"))
    else:
        path = r["id"].split("@")[0]; start = float(r["id"].split("@")[1].split("#")[0])
        x = svh.load(str(svh.EXT / "prepared" / grp.split(":")[0] / path), start, 30.0)
    fe = svh._frontend(())
    blocks = [fe.push(x[i:i + svh.CHUNK])[1] for i in range(0, len(x), svh.CHUNK)]
    vals = np.concatenate([b.values for b in blocks]); freqs = fe.axis.freqs
    f0 = r["f0"][: len(vals)]; vals = vals[: len(f0)]
    p = r["variants"][tag]["p_hum"].astype(np.float32)[: len(f0)]
    v = (f0 > 70) & (f0 < 400)
    lf = np.log(freqs)
    out = np.full((len(f0), len(H) + len(G)), np.nan, np.float32)
    for i in np.flatnonzero(v):
        out[i] = np.interp(np.log(np.concatenate([H, G]) * f0[i]), lf, vals[i], right=np.nan)
    floor = np.percentile(vals, 10, axis=0)  # per-bin quiet level of the clip
    fl = np.full_like(out, np.nan)
    for i in np.flatnonzero(v):
        fl[i] = np.interp(np.log(np.concatenate([H, G]) * f0[i]), lf, floor, right=np.nan)
    return grp, out[v], fl[v], p[v], f0[v]

if __name__ == "__main__":
    res = pickle.loads((svh.HERE / "results/e1_n150.pkl").read_bytes())
    g = svh.groups(res)
    sel = []
    for grp in ("fifine:speech", "mac:speech", "musan:speech:val", "fifine:hum", "mac:hum", "mlend_hums:hum:val"):
        sel += g[grp][:50]
    with ProcessPoolExecutor(24) as pool:
        out = list(pool.map(feats, sel))
    def show(name, X, FL):
        if not len(X): return
        med = np.nanmedian(X, 0); fl = np.nanmedian(FL, 0)
        rel = med[: len(H)] - med[0]
        gaps = med[len(H):]
        comb = np.nanmedian(X[:, 1:len(H)] - X[:, len(H):], 0)  # harmonic h+1 over the gap below it
        print(f"{name:34s} n={len(X):6d} h1 abs {med[0]:6.1f} dB (floor {fl[0]:6.1f}) | rel h2..h8: " + " ".join(f"{a:5.1f}" for a in rel[1:]) + " | comb depth h2..h8 over gap: " + " ".join(f"{a:4.1f}" for a in comb) + f" | harmonics above floor+10: {np.nanmean((X[:, :len(H)] > FL[:, :len(H)] + 10).sum(1)):.1f}")
    by = {}
    for grp, X, FL, p, f0 in out:
        by.setdefault(grp, []).append((X, FL, p, f0))
    for grp, lst in by.items():
        X = np.concatenate([a[0] for a in lst]); FL = np.concatenate([a[1] for a in lst]); p = np.concatenate([a[2] for a in lst])
        show(grp, X, FL)
        if "speech" in grp:
            show("   frames called hum", X[p >= 0.5], FL[p >= 0.5])
            show("   frames p<0.1", X[p < 0.1], FL[p < 0.1])
