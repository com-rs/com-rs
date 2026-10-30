"""Run every scenario with every library and write ``results/<scenario>.json``.

Every (scenario, library, threads) runs in its own process with the thread settings
of ``common.thread_env``: one thread for every pool (``--threads 1``), or the library
defaults, every core (``--threads all``). The error rates are computed in the
single-thread runs; the all-thread runs only time (the results do not depend on the
threads). OptiCommPy without numba runs the same code as OptiCommPy with numba, so
it is only timed. The machine, the versions, the commit and the commands are recorded
in every result file; ``python ../../tools/benchmarks_page.py`` renders them into
``docs/python/benchmarks.rst``.

    python run_all.py [--quick] [--scenarios s1_awgn,s4_coding] [--libs comrs,sionna]
                      [--threads 1,all] [--no-rust-build] [--keep-work]

Run it on a quiet machine: the timings are wrong on a loaded one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import common

SCENARIOS = ["s1_awgn", "s2_rayleigh", "s3_mimo", "s4_coding", "s5_ssfm"]


def first_paragraph(doc: str | None) -> str:
    return " ".join((doc or "").strip().split("\n\n")[0].split())


def build_rust(log: list[str]) -> bool:
    cmd = ["cargo", "build", "--release", "-p", "com-spec", "--bin", "comrs"]
    log.append(" ".join(cmd))
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=common.ROOT).returncode == 0


def run_one(python: str, mod, lib: str, threads: str, args, work: Path) -> dict:
    script = common.HERE / f"{mod.SCENARIO}.py"
    out = work / f"{mod.SCENARIO}_{lib}_{threads}.json"
    timing_only = threads != "1" or lib == "opticommpy-nojit"
    cmd = [
        python,
        str(script),
        "--lib",
        lib,
        "--threads",
        threads,
        "--work",
        str(work),
        "--out",
        str(out),
    ]
    if args.quick:
        cmd.append("--quick")
    if timing_only:
        cmd.append("--timing-only")
    env = common.thread_env(lib, threads)
    shown = " ".join(
        f"{v}={env[v]}" for v in [*common.THREAD_VARS, "NUMBA_DISABLE_JIT"] if v in env
    )
    command = f"{shown} python {script.name} " + " ".join(cmd[2:]).replace(
        str(work), "results/_work"
    )
    print(
        f"[{mod.SCENARIO}] {lib}, threads = {threads}{' (timing only)' if timing_only else ''}",
        flush=True,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=common.HERE, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        print("  failed:", *tail, sep="\n    ", flush=True)
        return {
            "library": lib,
            "threads": threads,
            "command": command,
            "process_wall_s": wall,
            "error": "\n".join(tail),
        }
    run = json.loads(out.read_text())
    run.update(command=command, process_wall_s=wall)
    print(f"  {wall:.1f} s", flush=True)
    return run


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quick", action="store_true", help="small sizes: checks that everything runs")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--libs", default=",".join(common.LIBRARIES))
    p.add_argument("--threads", default="1,all", help="thread modes, among 1 and all")
    p.add_argument(
        "--python", default=sys.executable, help="the interpreter of the scenario scripts"
    )
    p.add_argument("--no-rust-build", action="store_true", help="use target/release/comrs as it is")
    p.add_argument(
        "--keep-work", action="store_true", help="keep results/_work (fields, per-run JSON)"
    )
    args = p.parse_args()
    libs = [x for x in args.libs.split(",") if x]
    modes = [x for x in args.threads.split(",") if x]
    work = common.RESULTS / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    commands = [" ".join(["python", "run_all.py", *sys.argv[1:]])]
    rust_ok = (common.ROOT / "target" / "release" / "comrs").exists()
    if "comrs-rust" in libs and not args.no_rust_build:
        rust_ok = build_rust(commands)
    machine, git, versions = common.machine_info(), common.git_commit(), common.versions()
    t_all = time.perf_counter()
    for name in [x for x in args.scenarios.split(",") if x]:
        mod = importlib.import_module(name)
        runs = []
        for lib in common.LIBRARIES:
            if lib not in libs:
                continue
            supported = [c for c, sup in mod.CASES.items() if sup.get(lib) is True]
            if not supported:
                continue
            for threads in modes:
                if lib == "comrs-rust" and not rust_ok:
                    runs.append(
                        {
                            "library": lib,
                            "threads": threads,
                            "error": "target/release/comrs not built",
                        }
                    )
                    continue
                runs.append(run_one(args.python, mod, lib, threads, args, work))
        merged = {
            "scenario": mod.SCENARIO,
            "title": mod.TITLE,
            "description": first_paragraph(mod.__doc__),
            "script": f"benchmarks/libraries/{mod.SCENARIO}.py",
            "parameters": mod.PARAMETERS,
            "quick": args.quick,
            "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "machine": machine,
            "git": git,
            "versions": versions,
            "libraries": {k: v for k, v in common.LIBRARIES.items() if k in libs},
            "support": {
                c: {k: v for k, v in sup.items() if k in libs} for c, sup in mod.CASES.items()
            },
            "thread_modes": modes,
            "commands": commands,
            "runs": runs,
        }
        if hasattr(mod, "postprocess"):
            merged["agreement"] = mod.postprocess(merged, work)
        common.write_json(common.RESULTS / f"{mod.SCENARIO}.json", merged)
        print(f"wrote results/{mod.SCENARIO}.json", flush=True)
    print(f"total {time.perf_counter() - t_all:.0f} s")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
