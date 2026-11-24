"""ONNX export of a WhistleNet checkpoint, and the runtime that streams the exported file.

``export_onnx`` (training machine: needs torch, ``onnx`` and ``onnxruntime``, all in the ``ml``
extra) writes ``<name>.onnx`` next to ``<name>.pt``. The graph is one streaming step, the same
computation as ``WhistleNet.forward_stream`` followed by the sigmoids:

    inputs   x (1, T, F) normalised frames, any T >= 1;  s0..sN  each block's last input frames
    outputs  presence (1, T, K + A) and salience (1, T, K, F) probabilities;  n0..nN  the new state

Everything else a detector needs travels in the file's metadata (key ``whistle_house``, JSON):
the architecture, the normalisation stats, the front end the model was trained on, and the
SHA-256 of the checkpoint it was exported from. So the Pi needs the ``.onnx`` file and
``onnxruntime`` only (the ``onnx`` extra): no torch, no ``.pt``.

``OnnxStream`` loads such a file with onnxruntime + numpy alone. On the Pi 5 it runs the
``small`` and ``harmonic`` models in about half the time PyTorch takes per chunk.

The export is checked before it is kept: random frames go through both the exported graph and
``forward_stream`` in chunks of several sizes with the state carried over, and the outputs must
agree to ``PARITY_TOL``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

METADATA_KEY = "whistle_house"
FORMAT = 1
OPSET = 17
PARITY_TOL = 1e-4  # on probabilities; float32 round-off between the two runtimes is ~1e-6


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def onnx_path(checkpoint: Path) -> Path:
    """Where the export of a checkpoint lives: same folder, same name, ``.onnx``."""
    return checkpoint.with_suffix(".onnx")


class OnnxStream:
    """An exported checkpoint: metadata + the streaming step. Needs onnxruntime, not torch."""

    def __init__(self, path: Path, threads: int = 1):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError(
                "the onnx backend needs onnxruntime: `uv sync --extra onnx` (or --extra ml)"
            ) from e

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        self.path = path
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if METADATA_KEY not in meta:
            raise ValueError(f"{path.name} was not written by `whistle-house export-onnx`")
        info = json.loads(meta[METADATA_KEY])
        if info.get("format") != FORMAT:
            raise ValueError(f"{path.name}: unknown export format {info.get('format')}; re-export")
        self.arch: dict = info["arch"]
        self.norm: dict = info["norm"]
        self.frontend: dict = info["frontend"]
        self.source: dict = info["source"]  # {"name", "sha256"} of the .pt it came from
        self.classes: tuple[str, ...] = tuple(self.arch["classes"])
        # presence-only outputs (e.g. "speech"): presence columns after the classes
        self.aux: tuple[str, ...] = tuple(self.arch.get("aux", ()))
        self.lookahead: int = int(self.arch["lookahead"])
        self.n_bins: int = int(self.arch["n_bins"])
        inputs = self.session.get_inputs()
        self._x = inputs[0].name
        self._states = [(i.name, tuple(i.shape)) for i in inputs[1:]]

    def init_state(self) -> list[np.ndarray]:
        """State of a fresh stream (the zero padding an offline pass starts with)."""
        return [np.zeros(shape, np.float32) for _, shape in self._states]

    def stream(
        self, x: np.ndarray, state: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Next frames of a stream. x: (1, T, F) normalised float32. Returns presence (T, K + A)
        and salience (T, K, F) probabilities, delayed by ``lookahead`` frames, + the new state."""
        feed = {self._x: x, **{name: s for (name, _), s in zip(self._states, state, strict=True)}}
        pres, sal, *new_state = self.session.run(None, feed)
        return pres[0], sal[0], new_state

    def stale_against(self, checkpoint: Path) -> bool:
        """True when ``checkpoint`` exists and is not the file this export was made from."""
        return checkpoint.exists() and file_sha256(checkpoint) != self.source["sha256"]


def export_onnx(checkpoint: Path, out: Path | None = None) -> Path:
    """Export ``checkpoint`` (.pt) to ONNX, verify it against PyTorch, return the path."""
    import onnx
    import torch

    from whistle_house.ml.model import load_checkpoint

    model, ck = load_checkpoint(checkpoint)
    out = out or onnx_path(checkpoint)
    state = model.init_state()

    class Step(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, *state):
            pres, sal, new = self.net.forward_stream(x, list(state))
            return (torch.sigmoid(pres), torch.sigmoid(sal), *new)

    in_names = ["x", *(f"s{i}" for i in range(len(state)))]
    out_names = ["presence", "salience", *(f"n{i}" for i in range(len(state)))]
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer and deprecation chatter
        torch.onnx.export(
            Step(model).eval(),
            (torch.zeros(1, 8, model.cfg.n_bins), *state),
            str(tmp),
            input_names=in_names,
            output_names=out_names,
            dynamic_axes={"x": {1: "frames"}, "presence": {1: "frames"}, "salience": {1: "frames"}},
            opset_version=OPSET,
            dynamo=False,
        )
    proto = onnx.load(str(tmp))
    entry = proto.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = json.dumps(
        {
            "format": FORMAT,
            # the built config, not ck["arch"]: old checkpoints lack the newer fields
            "arch": asdict(model.cfg),
            "norm": ck["norm"],
            "frontend": ck["frontend"],
            "source": {"name": checkpoint.name, "sha256": file_sha256(checkpoint)},
        }
    )
    onnx.save(proto, str(tmp))
    try:
        err = _parity_error(model, OnnxStream(tmp))
        if not err <= PARITY_TOL:
            raise RuntimeError(
                f"export of {checkpoint.name} does not match PyTorch (max difference {err:.2g}, "
                f"allowed {PARITY_TOL:g}); nothing was written"
            )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return out


def _parity_error(model, exported: OnnxStream, chunks: tuple[int, ...] = (8, 3, 1, 13, 8)) -> float:
    """Largest output difference between the two runtimes over one stream cut into ``chunks``."""
    import torch

    gen = torch.Generator().manual_seed(0)
    t_state, o_state = model.init_state(), exported.init_state()
    err = 0.0
    for n in chunks:
        x = torch.randn(1, n, model.cfg.n_bins, generator=gen)
        with torch.no_grad():
            pres, sal, t_state = model.forward_stream(x, t_state)
        o_pres, o_sal, o_state = exported.stream(x.numpy(), o_state)
        err = max(
            err,
            float(np.abs(o_pres - torch.sigmoid(pres)[0].numpy()).max()),
            float(np.abs(o_sal - torch.sigmoid(sal)[0].numpy()).max()),
        )
    return err
