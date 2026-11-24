import pickle, sys, numpy as np
res = pickle.load(open("scratch/svh/results/e8_ft1.pkl","rb"))
for cid in sys.argv[1:]:
    r = next(r for r in res if r["id"]==cid and r["group"].startswith("fifine:") and "(" not in r["group"])
    f0 = r["f0"]; v = r["variants"]["sp hold0"]; t = r["times"]
    ps, ph = v["p_speech"].astype(float), v["p_hum"].astype(float)
    print(cid, "dur %.1f voiced %.2f f0 median %.0f p10 %.0f p90 %.0f level %.1f dB" % (r["seconds"], (f0>0).mean(), np.median(f0[f0>0]), *np.percentile(f0[f0>0],[10,90]), r["rms_db"]))
    # per-second summary: speech prob, hum prob, median f0
    for s in range(int(r["seconds"])):
        m = (t>=s)&(t<s+1)
        ff = f0[m]; ff = ff[ff>0]
        print("  %2d s  speech %.2f  hum %.2f  f0 %s  voiced %.2f" % (s, ps[m].mean(), ph[m].mean(), ("%.0f" % np.median(ff)) if len(ff) else "-", (f0[m]>0).mean()))
    print("  room2 events:", [(round(e[1],1), round(e[2],1), round(e[3])) for e in r["variants"]["room2"]["events"] if e[0]=="hum"])
