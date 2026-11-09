'Choosing code generation settings by timing them on sample data.'

from __future__ import annotations

import hashlib, json, os, platform, subprocess, tempfile, time
from pathlib import Path

import numpy as np

from lgbmjit import _lgbmjit

GRID = dict(block_trees=(4, 8, 17, 34), branch_free_leaves=(0, 4), tile_rows=(256, 1024, 2048))
COLD_FRACTION = 0.01
MAX_SAMPLE_ROWS = 20_000
SETTING_TYPES = dict(block_trees=int, tile_rows=int, branch_free_leaves=int, cold_fraction=float)


def cpu_name() -> str:
    'The CPU model, or an empty string when it is unknown.'
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError): pass
    return ""


def cache_key(model_text: str) -> str:
    "Key of a model's settings in a tuning cache: model SHA-256, architecture and CPU name."
    return f"{hashlib.sha256(model_text.encode()).hexdigest()}/{platform.machine()}/{cpu_name()}"


def valid(settings: object) -> bool:
    return (
        isinstance(settings,
            dict) and set(settings) == set(SETTING_TYPES) and all((type(settings[k]) is t for k, t in SETTING_TYPES.items())))


def search(forest: _lgbmjit.Forest, sample: np.ndarray) -> dict[str, int | float]:
    'The fastest grid point for predicting `sample` on one thread (the best of 3 runs each).'
    sample = sample[:MAX_SAMPLE_ROWS]
    if sample.shape[0] == 0: raise ValueError("the tuning sample has no rows")
    best = None
    for block_trees in GRID["block_trees"]:
        for branch_free_leaves in GRID["branch_free_leaves"]:
            compiled = forest.compile(sample.dtype.name, 0, 0, block_trees, GRID['tile_rows'][0], branch_free_leaves, COLD_FRACTION)
            for tile_rows in GRID["tile_rows"]:
                compiled.set_tile_rows(tile_rows)
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    compiled.predict(sample, False, 1)
                    times.append(time.perf_counter() - t0)
                settings = dict(block_trees=block_trees, tile_rows=tile_rows, branch_free_leaves=branch_free_leaves, cold_fraction=COLD_FRACTION)
                if best is None or min(times) < best[0]: best = (min(times), settings)
    assert best is not None
    return best[1]


def read_cache(path: str | os.PathLike, key: str) -> dict[str, int | float] | None:
    'The valid settings stored under `key` in the cache file at `path`, if any.'
    try: settings = json.loads(Path(path).read_text()).get(key)
    except (OSError, ValueError, AttributeError): return None
    return settings if valid(settings) else None


def write_cache(path: str | os.PathLike, key: str, settings: dict[str, int | float]) -> None:
    'Stores `settings` under `key` in the cache file at `path`, replacing the file atomically.'
    path = Path(path)
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict): data = {}
    except (OSError, ValueError): data = {}
    data[key] = settings
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f: json.dump(data, f, indent=1, sort_keys=True)
    os.replace(f.name, path)
