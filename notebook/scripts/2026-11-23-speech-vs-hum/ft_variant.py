"""Run the whistle-house CLI with the `room` preset's mic interferer weight changed
(experiment knob only: the status file does not record it, so say it in the run's name).

  .venv/bin/python scratch/svh/ft_variant.py <mic weight> [--data ...] train-detector ...
"""
import sys

from whistle_house import cli
from whistle_house.data import training

if __name__ == "__main__":  # workers re-import this file (forkserver)
    training.AUGMENT_PRESETS["room"]["interferer_weights"] = {
        **training._SPEECH_HEAVY,
        "mic": float(sys.argv[1]),
    }
    sys.argv = ["whistle-house", *sys.argv[2:]]
    cli.main()
