"""E1: speech-as-hum on the pretraining speech (MUSAN) vs the owner's speech clips, per model."""

import sys

from svh import external_jobs, groups, mic_jobs, run, summarize, table, with_variants

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
GATE = {
    "name": "adaptive_gate",
    "params": {
        "profiles": [
            "fifine",
            "fifine-fan-door-open",
            "fifine-fan-door-closed",
            "fifine-fan-both-doors-closed",
            "fifine-fridge",
        ],
        "dwell_s": 60,
    },
}
VARIANTS = [
    ("long-e240", "humnet-long-pretrain-e240.pt", []),
    ("long-e240+gate", "humnet-long-pretrain-e240.pt", [GATE]),
    ("long-b128", "humnet-long-pretrain-b128.pt", []),
    ("deep-pre", "humnet-deep-pretrain.pt", []),
    ("deep-ft", "humnet-deep.pt", []),
    ("harm-pre", "humnet-pretrain.pt", []),
    ("harm-ft", "humnet.pt", []),
]


def main():
    jobs = (
        external_jobs("musan", "speech", "val", N, seed=1)
        + external_jobs("musan", "speech", "train", N, seed=2)
        + external_jobs("musan", "music", "val", 60, seed=3)
        + external_jobs("musan", "noise", "val", 60, seed=4)
        + external_jobs("mlend_hums", "hum", "val", 100, seed=5)
        + external_jobs("mlend_whistles", "whistle", "val", 60, seed=6)
        + mic_jobs()
    )
    res = run(with_variants(jobs, VARIANTS), f"e1_n{N}.pkl")
    g = groups(res)
    for tag, _, _ in VARIANTS:
        print(f"\n=== {tag}")
        print(table({k: summarize(v, tag) for k, v in sorted(g.items())}))


if __name__ == "__main__":
    main()
