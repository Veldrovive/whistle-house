"""E5: is it the channel's frequency response? Measure the long-term spectrum of the owner's
speech on the Fifine against the Mac mic (same voice), apply the difference to Mac / MUSAN
speech, and its inverse to the Fifine speech."""
import pickle, sys, json
import numpy as np
from scipy.signal import welch
import svh
from svh import external_jobs, groups, mic_jobs, run, summarize, table, with_variants

VARIANTS = [("long-e240", "humnet-long-pretrain-e240.pt", []), ("deep-ft", "humnet-deep.pt", [])]

def ltas(jobs):
    """Mean power spectrum over the loud (speech) parts of the clips, dB, on a 1/6-octave grid."""
    grid = np.geomspace(50, 8000, 45)
    acc = []
    for j in jobs:
        x = svh.load(j["path"])
        n = 4096
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2]
        e = (fr**2).mean(1)
        fr = fr[e > np.percentile(e, 60)]
        f = np.fft.rfftfreq(n, 1 / svh.SR)
        P = (np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2).mean(0)
        band = [P[(f >= g / 2 ** (1 / 6)) & (f < g * 2 ** (1 / 6))].mean() for g in grid]
        acc.append(10 * np.log10(np.array(band) + 1e-20))
    return grid, np.mean(acc, 0)

def main():
    sp = mic_jobs("speech"); hum = mic_jobs("hum")
    fif = [j for j in sp if j["group"].startswith("fifine")]; mac = [j for j in sp if j["group"].startswith("mac")]
    grid, lf = ltas(fif); _, lm = ltas(mac)
    _, hf = ltas([j for j in hum if j["group"].startswith("fifine")]); _, hm = ltas([j for j in hum if j["group"].startswith("mac")])
    mus = external_jobs("musan", "speech", "val", 60, seed=1)
    _, lmu = ltas(mus[:30])
    d = lf - lm
    ref = (grid > 90) & (grid < 130)
    d = d - d[ref].mean()   # relative to the fundamental region
    dmu = (lf - lmu); dmu = dmu - dmu[ref].mean()
    dh = (hf - hm); dh = dh - dh[ref].mean()
    print("freq Hz           :", " ".join(f"{g:6.0f}" for g in grid[::3]))
    print("fifine-mac speech :", " ".join(f"{v:6.1f}" for v in d[::3]))
    print("fifine-mac hums   :", " ".join(f"{v:6.1f}" for v in dh[::3]))
    print("fifine-musan      :", " ".join(f"{v:6.1f}" for v in dmu[::3]))
    eq = ("eqcurve", grid.tolist(), np.clip(d, -30, 30).tolist(), "keep_rms")
    inv = ("eqcurve", grid.tolist(), np.clip(-d, -30, 30).tolist(), "keep_rms")
    eqmu = ("eqcurve", grid.tolist(), np.clip(dmu, -30, 30).tolist(), "keep_rms")
    sets = {
        "musan": (mus, {"raw": [], "fifineEQ(mac ref)": [eq], "fifineEQ(musan ref)": [eqmu], "tilt-3dB/oct": [("tilt", -3, 110)], "tilt-6dB/oct": [("tilt", -6, 110)], "tilt+3dB/oct": [("tilt", 3, 110)],
                         "fifineEQ(musan)+reverb0.5/0": [eqmu, ("reverb", 0.5, 0)], "tilt-6+reverb0.5/0": [("tilt", -6, 110), ("reverb", 0.5, 0)], "fifineEQ(musan) level-50": [eqmu, ("rms_to", -50)]}),
        "mac": (mac, {"raw": [], "fifineEQ": [eq], "fifineEQ+gain+12": [eq, ("gain", 12)], "fifineEQ+reverb0.5/0": [eq, ("reverb", 0.5, 0)], "tilt-3dB/oct": [("tilt", -3, 110)], "tilt-6dB/oct": [("tilt", -6, 110)], "fifineEQ level-50": [eq, ("rms_to", -50)]}),
        "fifine": (fif, {"raw": [], "inverseEQ": [inv], "inverseEQ+gain+12": [inv, ("gain", 12)], "tilt+3dB/oct": [("tilt", 3, 110)], "tilt+6dB/oct": [("tilt", 6, 110)], "tilt-3dB/oct": [("tilt", -3, 110)]}),
        "fifine-hum": ([j for j in hum if j["group"].startswith("fifine")], {"raw": [], "inverseEQ": [inv], "tilt+6dB/oct": [("tilt", 6, 110)]}),
        "mac-hum": ([j for j in hum if j["group"].startswith("mac")], {"raw": [], "fifineEQ": [eq], "tilt-6dB/oct": [("tilt", -6, 110)]}),
    }
    jobs = []
    for name, (base, opsets) in sets.items():
        for on, ops in opsets.items():
            jobs += [{**j, "group": f"{name} {on}", "ops": ops, "seed": i} for i, j in enumerate(base)]
    res = run(with_variants(jobs, VARIANTS), "e5.pkl")
    g = groups(res)
    for tag, _, _ in VARIANTS:
        print(f"\n=== {tag}")
        print(table({k: {kk: vv for kk, vv in summarize(v, tag).items() if kk in ("clips", "minutes", "hum%", "hum%_of_voiced", "hum_ev/min", "match/min dirk>=0.1")} for k, v in g.items()}))

if __name__ == "__main__":
    main()
