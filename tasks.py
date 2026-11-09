# /// script
# requires-python = ">=3.11"
# ///
"""lgbmjit's build pipeline, the same on any machine or CI: `uv run tasks.py check|build|test`.

- `check`: formatting, style, types, lints and Rust tests. Also the pre-commit hook.
- `build`: a release wheel for this machine into `dist/`, or with `--sdist` the source distribution.
- `test`: installs this machine's wheels from `dist/` into fresh environments, one with the newest dependencies on the newest Python and one with the declared floors on the oldest, and runs the test suite against them. Files that pass are recorded in `dist/tested-*.json` for `release.py`.
"""
import argparse, hashlib, json, platform, re, subprocess, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
MATURIN_VERSION = "1.16.0"
OLDEST, NEWEST = "3.11", "3.15"
# file name patterns of the wheels this machine builds and can run
WHEELS = {("Darwin", "arm64"): "*macosx*_arm64.whl", ("Darwin", "x86_64"): "*macosx*_x86_64.whl",
    ("Linux", "x86_64"): "*linux*_x86_64.whl", ("Linux", "aarch64"): "*linux*_aarch64.whl", ("Windows", "AMD64"): "*win_amd64.whl"}


def run(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def check():
    # the Python-side tools only read sources: install them from the lock without building the package
    run("uv", "sync", "--inexact", "--no-install-project", "--extra", "dev", "--extra", "test")
    for tool in (["cargo", "fastfmt", "--check"], ["chkstyle"], ["pyright", "python"]): run("uv", "run", "--no-sync", *tool)
    run("cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings")
    run("cargo", "test", "--workspace")


def build(sdist=False):
    if sdist: return run("uvx", f"maturin=={MATURIN_VERSION}", "sdist", "--out", "dist")
    # zig links Linux wheels against an old glibc, so they install on any manylinux2014 system
    if platform.system() == "Linux": run("uvx", "--from", f"maturin[zig]=={MATURIN_VERSION}", "maturin", "build", "--release", "--zig", "--compatibility", "manylinux2014", "--out", "dist")
    else: run("uvx", f"maturin=={MATURIN_VERSION}", "build", "--release", "--out", "dist")


def floors():
    "The `>=` floor of each runtime and test dependency in pyproject.toml, as exact pins."
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    reqs = project["dependencies"] + project["optional-dependencies"]["test"]
    return [f"{m[1]}=={m[2]}" for r in reqs if (m := re.fullmatch(r"([\w-]+)>=([\w.]+)", r))]


def environments():
    "Name, Python version and pins of each test environment: the newest versions, and the declared floors."
    yield "newest", NEWEST, []
    yield "floors", OLDEST, floors()
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def test(only=None, sdist=False):
    pattern = WHEELS[platform.system(), platform.machine()]
    files = sorted(DIST.glob("*.tar.gz" if sdist else pattern))
    if not files: sys.exit(f"no {'sdist' if sdist else pattern} in dist/; run `uv run tasks.py build{' --sdist' if sdist else ''}` first")
    envs = [e for e in environments() if not only or e[0] in only]
    if sdist: envs = envs[:1]
    for f in files:
        for name, python, pins in envs:
            print(f"# {f.name}: {name}", flush=True)
            with_ = [a for p in pins for a in ("--with", p)]
            run("uv", "run", "--isolated", "--no-project", "-p", python, "--refresh-package", "lgbmjit",
                "--with", f"lgbmjit[test] @ {f.resolve().as_uri()}", *with_, "pytest", "-q", "tests")
    tag = "sdist" if sdist else f"{platform.system()}-{platform.machine()}".lower()
    record = dict(files={f.name: sha256(f) for f in files}, environments=[e[0] for e in envs], machine=platform.platform())
    (DIST / f"tested-{tag}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("build").add_argument("--sdist", action="store_true", help="build the source distribution instead of a wheel")
    t = sub.add_parser("test")
    t.add_argument("--only", action="append", help="test only this environment (repeatable), e.g. newest, floors")
    t.add_argument("--sdist", action="store_true", help="test the source distribution instead of the wheels")
    a = p.parse_args()
    if a.cmd == "check": check()
    elif a.cmd == "build": build(a.sdist)
    else: test(a.only, a.sdist)
