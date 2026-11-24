"""E2: what makes speech fire as hum? Waveform manipulations of the same speech segments.

MUSAN val speech is pushed towards the deployment condition (level, room noise, reverb), and
the Fifine speech clips away from it (louder, rumble removed).
"""

import sys

from svh import DATA, external_jobs, groups, mic_jobs, run, summarize, table, with_variants

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REC = DATA / "recordings"
QUIET = str(REC / "bb3c12342551.wav")  # "Another near complete silence" (Fifine)
FRIDGE = str(REC / "72b0840d2a4e.wav")  # "Hum of the fridge"
FAN = str(REC / "259b0607aa98.wav")  # "Bathroom fan with door open"

VARIANTS = [
    ("long-e240", "humnet-long-pretrain-e240.pt", []),
    ("deep-ft", "humnet-deep.pt", []),
]

MUSAN_OPS = {
    "raw": [],
    "level-45": [("rms_to", -45)],
    "level-55": [("rms_to", -55)],
    "level-45+quiet-room": [("rms_to", -45), ("add", QUIET, 0)],
    "level-45+fridge": [("rms_to", -45), ("add", FRIDGE, 0)],
    "level-45+fan": [("rms_to", -45), ("add", FAN, 0)],
    "level-55+quiet-room": [("rms_to", -55), ("add", QUIET, 0)],
    "reverb0.3/drr+6": [("reverb", 0.3, 6)],
    "reverb0.5/drr0": [("reverb", 0.5, 0)],
    "reverb0.7/drr-6": [("reverb", 0.7, -6)],
    "white-snr20": [("rms_to", -30), ("white", -50)],
    "white-snr10": [("rms_to", -30), ("white", -40)],
    "far-field: reverb0.5/0 level-45 quiet-room": [
        ("reverb", 0.5, 0),
        ("rms_to", -45),
        ("add", QUIET, 0),
    ],
    "far-field: reverb0.7/-6 level-50 fridge": [
        ("reverb", 0.7, -6),
        ("rms_to", -50),
        ("add", FRIDGE, 0),
    ],
    "lowpass1500": [("lowpass", 1500)],
    "highpass200": [("highpass", 200)],
}
FIFINE_OPS = {
    "raw": [],
    "gain+12": [("gain", 12)],
    "gain+24": [("gain", 24)],
    "gain-12": [("gain", -12)],
    "highpass150": [("highpass", 150)],
    "highpass150+gain+24": [("highpass", 150), ("gain", 24)],
    "reverb0.5/drr0": [("reverb", 0.5, 0)],
}


def main():
    base = external_jobs("musan", "speech", "val", N, seed=1)
    jobs = []
    for name, ops in MUSAN_OPS.items():
        jobs += [{**j, "group": f"musan {name}", "ops": ops, "seed": i} for i, j in enumerate(base)]
    for mic in ("fifine", "mac"):
        sp = [j for j in mic_jobs("speech") if j["group"].startswith(mic)]
        extra = {k: v for k, v in MUSAN_OPS.items() if k != "raw"} if mic == "mac" else {}
        for name, ops in {**FIFINE_OPS, **extra}.items():
            jobs += [{**j, "group": f"{mic} {name}", "ops": ops, "seed": i} for i, j in enumerate(sp)]
    res = run(with_variants(jobs, VARIANTS), f"e2_n{N}.pkl")
    g = groups(res)
    for tag, _, _ in VARIANTS:
        print(f"\n=== {tag}")
        rows = {k: summarize(v, tag) for k, v in g.items()}
        for r in rows.values():
            r["rms_db"] = 0
        for k, v in g.items():
            rows[k]["rms_db"] = round(sum(x["rms_db"] for x in v) / len(v), 1)
        print(table(rows))


if __name__ == "__main__":
    main()
