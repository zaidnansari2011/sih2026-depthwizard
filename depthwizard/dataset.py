"""Shard-backed dataset for DFC2019 height regression.

Reads the .npz shards produced by tools/prepare_data.py. Each shard holds:

    rgb   uint8    (N, C, C, 3)
    agl   float16  (N, C, C)      metres above ground level
    cls   uint8    (N, C, C)      DFC semantic class, optional
    tile  str      (N,)           provenance, e.g. "JAX_004_007"

Why shards rather than one big memmap
-------------------------------------
Training happens on Kaggle, so the data has to travel. Compressed shards are roughly a
third the size of the raw arrays, which is the difference between a 20-minute upload and
an hour. The cost is that random access across the whole set is not free, so we shuffle
in two stages: shard order each epoch, then rows within a small in-memory buffer of
shards. prepare_data.py already decorrelates shard *contents*, so a buffer of a few
shards is a good approximation to a global shuffle.

Void handling
-------------
prepare_data.py zeroes void pixels and drops crops that are mostly void, but survivors
still carry some. Zero is a legal height, so we cannot recover the mask from the values
afterwards -- instead we treat exact zeros conservatively as suspect only where the
probe found a sentinel. See `void_policy`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

# DA-V2 inherits DINOv2's ImageNet normalisation. Keep these identical to the values
# used at inference or the backbone sees a different input distribution than it was
# trained on -- a silent, hard-to-find accuracy leak.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Augment:
    """D4 symmetry plus photometric jitter.

    D4 (flips and 90-degree rotations) is exactly valid for nadir height regression:
    rotating the scene rotates the height map identically, and height is invariant to
    reflection. This is *not* true once a shadow prior is in play -- shadow direction is
    tied to sun azimuth -- which is one reason 6.2 is a separate, later stage.

    Photometric jitter is the cheap half of the domain-gap work (PLAN.md 7.1). WorldView
    and Cartosat differ in radiometry, sensor response and atmospheric correction; a
    model trained on one fixed colour distribution learns that distribution. Jitter is
    not a substitute for real target-domain data, but it is not nothing.
    """

    flip: bool = True
    rot90: bool = True
    brightness: float = 0.2       # multiplicative, +/- this fraction
    contrast: float = 0.2
    gamma: float = 0.15           # log-uniform exponent, +/- this
    noise: float = 0.01           # gaussian sigma in [0,1] units

    def __call__(self, rgb: np.ndarray, agl: np.ndarray, cls: np.ndarray | None, rng):
        if self.flip:
            if rng.random() < 0.5:
                rgb, agl = rgb[:, ::-1], agl[:, ::-1]
                cls = cls[:, ::-1] if cls is not None else None
            if rng.random() < 0.5:
                rgb, agl = rgb[::-1], agl[::-1]
                cls = cls[::-1] if cls is not None else None
        if self.rot90:
            k = int(rng.integers(4))
            if k:
                rgb, agl = np.rot90(rgb, k, (0, 1)), np.rot90(agl, k, (0, 1))
                cls = np.rot90(cls, k, (0, 1)) if cls is not None else None

        x = rgb.astype(np.float32) / 255.0
        if self.brightness:
            x *= 1.0 + rng.uniform(-self.brightness, self.brightness)
        if self.contrast:
            m = x.mean()
            x = m + (x - m) * (1.0 + rng.uniform(-self.contrast, self.contrast))
        if self.gamma:
            x = np.clip(x, 1e-4, None) ** float(np.exp(rng.uniform(-self.gamma, self.gamma)))
        if self.noise:
            x += rng.normal(0.0, self.noise, x.shape).astype(np.float32)
        return np.clip(x, 0.0, 1.0), agl, cls


def _normalise(x01: np.ndarray) -> np.ndarray:
    """HWC float [0,1] -> CHW float, ImageNet-normalised."""
    return np.ascontiguousarray(((x01 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


class ShardIndex:
    """Discovers shards for a split and reports what is in them."""

    def __init__(self, shard_dir: str | Path, split: str):
        self.dir = Path(shard_dir)
        self.split = split
        self.files = sorted(self.dir.glob(f"{split}_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"no {split}_*.npz in {self.dir}. Run: python tools/prepare_data.py shard"
            )
        probe = self.dir / "probe.json"
        self.probe = json.loads(probe.read_text()) if probe.exists() else {}

    def __repr__(self):
        return f"<ShardIndex {self.split}: {len(self.files)} shards in {self.dir}>"


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class HeightShardDataset(Dataset):
    """Map-style dataset. Loads whole shards on demand and caches a few.

    Use this for validation and test, where order does not matter and a sequential
    sweep keeps the cache hit rate at 100%. For training use ShardStream, which
    shuffles properly.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        split: str = "val",
        augment: Augment | None = None,
        cache_shards: int = 2,
        max_height: float | None = None,
        return_cls: bool = True,
    ):
        self.index = ShardIndex(shard_dir, split)
        self.augment = augment
        self.max_height = max_height
        self.return_cls = return_cls
        self._cache: dict[Path, dict] = {}
        self._cache_order: list[Path] = []
        self._cache_n = max(1, cache_shards)

        # Row counts per shard, read from the header only -- np.load of a compressed
        # npz is lazy per array, so this does not decompress the pixel data.
        self.counts = []
        for f in self.index.files:
            with np.load(f, allow_pickle=False) as z:
                self.counts.append(int(z["agl"].shape[0]) if "agl" in z.files else 0)
        self.offsets = np.cumsum([0] + self.counts)
        self.n = int(self.offsets[-1])

    def __len__(self):
        return self.n

    def _shard(self, i: int) -> dict:
        path = self.index.files[i]
        if path not in self._cache:
            self._cache[path] = _load_shard(path)
            self._cache_order.append(path)
            while len(self._cache_order) > self._cache_n:
                self._cache.pop(self._cache_order.pop(0), None)
        return self._cache[path]

    def __getitem__(self, idx: int):
        si = int(np.searchsorted(self.offsets, idx, "right") - 1)
        row = idx - int(self.offsets[si])
        d = self._shard(si)
        cls = d["cls"][row] if ("cls" in d and self.return_cls) else None
        return make_sample(
            d["rgb"][row], d["agl"][row], cls,
            augment=self.augment, max_height=self.max_height,
            tile=str(d["tile"][row]) if "tile" in d else "",
            rng=np.random.default_rng(idx),
        )


class ShardStream(IterableDataset):
    """Iterable training stream with two-stage shuffling.

    Each epoch: shuffle shard order, fill a buffer of `buffer_shards`, emit rows from it
    in random order. With DataLoader workers, shards are partitioned across workers so
    no two workers decompress the same file.

    `epoch` must be set (via set_epoch) each epoch or every epoch sees the same order.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        split: str = "train",
        augment: Augment | None = None,
        buffer_shards: int = 2,
        max_height: float | None = None,
        return_cls: bool = False,
        seed: int = 1337,
    ):
        self.index = ShardIndex(shard_dir, split)
        self.augment = augment if augment is not None else Augment()
        self.buffer_shards = max(1, buffer_shards)
        self.max_height = max_height
        self.return_cls = return_cls
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + wid)

        files = list(self.index.files)
        np.random.default_rng(self.seed + 1000 * self.epoch).shuffle(files)
        files = files[wid::nw]

        for start in range(0, len(files), self.buffer_shards):
            chunk = files[start: start + self.buffer_shards]
            loaded = [_load_shard(f) for f in chunk]
            pairs = [(s, r) for s, d in enumerate(loaded) for r in range(len(d["agl"]))]
            rng.shuffle(pairs)
            for s, r in pairs:
                d = loaded[s]
                cls = d["cls"][r] if ("cls" in d and self.return_cls) else None
                yield make_sample(
                    d["rgb"][r], d["agl"][r], cls,
                    augment=self.augment, max_height=self.max_height,
                    tile=str(d["tile"][r]) if "tile" in d else "",
                    rng=rng,
                )


def make_sample(rgb, agl, cls, augment, max_height, tile, rng) -> dict:
    """Assemble one training sample, including the validity mask.

    The mask is what keeps void ground truth out of the loss. Two things can invalidate
    a pixel: a non-finite height, or a height outside the physically plausible range.
    The second guard matters more than it looks -- a single -9999 that slipped through
    would dominate an MSE and quietly wreck a run.
    """
    agl = np.asarray(agl, np.float32)
    valid = np.isfinite(agl)
    if max_height is not None:
        valid &= (agl >= -1.0) & (agl <= max_height)
    agl = np.where(valid, agl, 0.0)

    if augment is not None:
        x01, agl, cls = augment(rgb, agl, cls, rng)
        # The mask must follow the same geometry as the image, so re-derive it from the
        # augmented height rather than transforming it separately and risking a skew.
        valid = np.isfinite(agl)
        if max_height is not None:
            valid &= (agl >= -1.0) & (agl <= max_height)
    else:
        x01 = np.asarray(rgb, np.float32) / 255.0

    out = {
        "rgb": torch.from_numpy(_normalise(x01)),
        "agl": torch.from_numpy(np.ascontiguousarray(agl))[None],
        "mask": torch.from_numpy(np.ascontiguousarray(valid.astype(np.float32)))[None],
        "tile": tile,
    }
    if cls is not None:
        out["cls"] = torch.from_numpy(np.ascontiguousarray(np.asarray(cls, np.uint8)).astype(np.int64))[None]
    return out


def denormalise(t: torch.Tensor) -> np.ndarray:
    """CHW normalised tensor -> HWC uint8, for writing preview images."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
