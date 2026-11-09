# /// script
# requires-python = ">=3.11"
# ///
"""Check, then publish, the files in `dist/` for a release tag: `uv run release.py v0.2.0 [--check]`.

`dist/` must hold the wheels, the source distribution and the `tested-*.json` records of a `tasks.py test` run on every platform. Every file must carry the tag's version and match its recorded hash. `--check` stops after checking; without it, `uv publish` uploads the files, using the token in `UV_PUBLISH_TOKEN`.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

DIST = Path(__file__).parent / "dist"
# one test record per wheel platform that tasks.py builds, plus the sdist
PLATFORMS = ["darwin-arm64", "linux-aarch64", "linux-x86_64", "windows-amd64", "sdist"]


def version(f): return f.name.removesuffix(".tar.gz").split("-")[1]


def main(tag, check):
    files = sorted([*DIST.glob("*.whl"), *DIST.glob("*.tar.gz")])
    if not files: sys.exit("dist/ is empty: the build for this commit is gone or was never made. Rebuild and test it first.")
    missing = [p for p in PLATFORMS if not (DIST / f"tested-{p}.json").exists()]
    if missing: sys.exit(f"no test record for: {', '.join(missing)}")
    tested = {n: h for r in DIST.glob("tested-*.json") for n, h in json.loads(r.read_text())["files"].items()}
    untested = [f.name for f in files if tested.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest()]
    if untested: sys.exit("not tested, or changed since testing: " + ", ".join(untested))
    wrong = [f.name for f in files if version(f) != tag.removeprefix("v")]
    if wrong: sys.exit(f"version differs from tag {tag}: " + ", ".join(wrong))
    if not any(f.name.endswith(".tar.gz") for f in files): sys.exit("dist/ has no source distribution")
    print("\n".join(f.name for f in files))
    if not check: subprocess.run(["uv", "publish", *map(str, files)], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tag", help="release tag, such as v0.2.0")
    p.add_argument("--check", action="store_true", help="check the files without publishing")
    a = p.parse_args()
    main(a.tag, a.check)
