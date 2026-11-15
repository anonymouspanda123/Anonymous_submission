#!/usr/bin/env python3
"""
run_all.py - execute the whole data-generation pipeline with one command.

    python run_all.py --smoke --yes     # cheap end-to-end test: 1 strategy, 1 category, main track only, in a scratch copy
    python run_all.py --dry-run         # print every command and the estimated number of API calls, call nothing
    python run_all.py --yes             # the full run: 9 strategies x 3 feedback tracks (Mistral, DeepSeek, GPT-OSS)

What it runs (per strategy S; scripts are in pipeline/):
  main track (feedback judge Mistral):  01 answers -> 02 scores -> 03 improvements -> 04 merge -> 01 answers on test
                                        questions with the merged prompt -> 05 pairwise (original first) -> 05 pairwise (improved first)
  other tracks (DeepSeek, GPT-OSS):     02 scores by that judge -> 03 improvements on the SAME improvement questions -> 04 merge ->
                                        01 test answers -> 05 pairwise (both orders)
Every script skips outputs that already exist, so an interrupted or partly failed run is resumed by running the same command again.
A step counts as failed if it exits non-zero or its log reports errors (the scripts exit 0 even when single API calls failed).
Logs go to pipeline/logs/, a machine-readable record of the run (versions, prompt hashes, steps, timings) to pipeline/run_manifest.json.
No user name or host name is written anywhere. Logs contain absolute paths, so delete pipeline/logs/ before sharing the folder.
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"
MODEL = "gemma-4-31b-it"
STRATEGIES = ["base", "Chain_of_thought", "Hierarchical_MoE_ToT", "Least_to_most", "MoE",
              "Reflection", "Self_consistency", "Step_back", "Tree_of_thought"]
JUDGES = ["deepseek:deepseek-v4-pro:thinking", "mistral:mistral-large-2512", "openrouter:openai/gpt-oss-120b"]
SMOKE_STRATEGY, SMOKE_CATEGORY = "Step_back", "cat08"
SCRIPTS = {"01": "01_run_inference_with_prompts.py", "02": "02_judge_answers.py", "03": "03_improve_prompt.py",
           "04": "04_merge_improvements.py", "05": "05_pairwise_compare.py"}
PRINT_LOCK = threading.Lock()


def track_defs(gptoss_model):
    slug = gptoss_model.replace("/", "-").replace(":", "-")
    return {
        "mistral": {"slug": None, "provider": "mistral", "model": "mistral-large-2512"},
        "deepseek": {"slug": "deepseek-v4-pro-thinking", "provider": "deepseek", "model": "deepseek-v4-pro:thinking"},
        "gptoss": {"slug": slug, "provider": "openrouter", "model": gptoss_model},
    }


def steps_for(strategy, track, tdefs, category):
    """List of (step id, script key, argument list) for one strategy and one feedback track."""
    pre = f"{MODEL}_{strategy}_"
    cat = ["--category", category] if category else []
    jargs = [x for j in JUDGES for x in ("--judge", j)]
    t = tdefs[track]
    if track == "mistral":
        merged = f"{strategy}_merged"
        return [
            ("answers", "01", ["--strategy", strategy] + cat),
            ("scores", "02", ["--strategy", strategy] + cat),
            ("improve", "03", ["--strategy", strategy] + cat),
            ("merge", "04", ["--strategy", strategy] + cat),
            ("test-answers", "01", ["--strategy", merged, "--base-prefix", pre] + cat),
            ("pairwise", "05", ["--strategy-a", strategy, "--strategy-b", merged, "--base-prefix", pre] + jargs + cat),
            ("pairwise-swapped", "05", ["--strategy-a", merged, "--strategy-b", strategy, "--base-prefix", pre] + jargs
             + ["--scores-dir", "pairwise_scores_reversed"] + cat),
        ]
    slug = t["slug"]
    merged = f"{strategy}_merged_{slug}"
    return [
        ("scores", "02", ["--strategy", strategy, "--provider", t["provider"], "--judge-model", t["model"],
                          "--scores-dir", f"{slug}/scores"] + cat),
        ("improve", "03", ["--strategy", strategy, "--output-strategy", merged, "--all-pairs", "--train-only",
                           "--base-prefix", pre, "--base-improvements-dir", "improvements",
                           "--scores-dir", f"{slug}/scores", "--improvements-dir", f"{slug}/improvements"] + cat),
        ("merge", "04", ["--strategy", merged, "--output-name", merged, "--original-strategy", strategy,
                         "--improvements-dir", f"{slug}/improvements"] + cat),
        ("test-answers", "01", ["--strategy", merged, "--base-prefix", pre] + cat),
        ("pairwise", "05", ["--strategy-a", strategy, "--strategy-b", merged, "--base-prefix", pre] + jargs + cat),
        ("pairwise-swapped", "05", ["--strategy-a", merged, "--strategy-b", strategy, "--base-prefix", pre] + jargs
         + ["--scores-dir", "pairwise_scores_reversed"] + cat),
    ]


# --------------------------------------------------------------------------- preflight
def question_stems(workdir, category):
    return sorted(p.stem for p in (workdir / "benchmark_questions").glob("cat*.txt") if not category or p.stem.startswith(category))


def estimate(stems, strategies, tracks):
    n = len(stems)
    train, test = n // 2, n - n // 2
    cats = len({s.split("_")[0] for s in stems})
    per_main = {"generation (gemma)": n + test, "improvement+merge (gemma)": train + cats, "grading (judge)": n, "pairwise (3 judges x 2 orders)": 6 * test}
    per_track = {"generation (gemma)": test, "improvement+merge (gemma)": train + cats, "grading (judge)": n, "pairwise (3 judges x 2 orders)": 6 * test}
    total = {k: 0 for k in per_main}
    for t in tracks:
        for k, v in (per_main if t == "mistral" else per_track).items():
            total[k] += v * len(strategies)
    return n, train, test, total


def missing_keys():
    miss = []
    if not os.environ.get("GOOGLE_API_KEY"):
        miss.append("GOOGLE_API_KEY (Gemma generation, improvement, merge)")
    if not any(os.environ.get(k) for k in ["MISTRAL_API_KEY", "MISTRAL_API_KEY2", "MISTRAL_API_KEY3", "MISTRAL_API_KEY4", "MISTRAL_APL_KEY6", "MISTRAL_APL_KEY7"]):
        miss.append("MISTRAL_API_KEY (Mistral judge)")
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_AI_API_KEY_PAID")):
        miss.append("OPENROUTER_API_KEY (GPT-OSS judge)")
    ds = (PIPE / "providers" / "deepseek.py").read_text(encoding="utf-8")
    if re.search(r"^ROUTE_VIA_OPENROUTER\s*=\s*True", ds, re.M):
        if not os.environ.get("OPEN_ROUTER_API_KEY"):
            miss.append("OPEN_ROUTER_API_KEY (DeepSeek, routed through OpenRouter)")
    elif not os.environ.get("DEEPSEEK_API_KEY"):
        miss.append("DEEPSEEK_API_KEY (DeepSeek judge)")
    return miss


def make_smoke_dir():
    d = ROOT / "smoke_run"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for f in list(PIPE.glob("*.py")):
        shutil.copy2(f, d / f.name)
    shutil.copytree(PIPE / "providers", d / "providers", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PIPE / "eval_prompts", d / "eval_prompts")
    shutil.copytree(PIPE / "benchmark_questions", d / "benchmark_questions")
    (d / "system_prompts").mkdir()
    for s in STRATEGIES:
        shutil.copy2(PIPE / "system_prompts" / f"{s}.txt", d / "system_prompts" / f"{s}.txt")
    return d


# --------------------------------------------------------------------------- execution
ERR_PATTERNS = [r"[Ee]rrors?\s*[:=]\s*(\d+)"]


def count_errors(text):
    """Errors reported by a script: the larger of its own summary counts and the number of error lines."""
    summary = sum(int(m) for p in ERR_PATTERNS for m in re.findall(p, text))
    lines = len(re.findall(r"^\s*(?:\[?ERROR\]?)\b", text, re.M))
    return max(summary, lines)


def run_step(sid, script, args, workdir, logdir, dry):
    cmd = [sys.executable, SCRIPTS[script]] + args
    shown = "python " + SCRIPTS[script] + " " + " ".join(args)
    if dry:
        with PRINT_LOCK:
            print(f"  [dry-run] {sid}: {shown}")
        return {"id": sid, "cmd": shown, "returncode": 0, "errors": 0, "seconds": 0.0, "log": None}
    log = logdir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", sid) + ".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    errs = count_errors(log.read_text(encoding="utf-8", errors="replace"))
    rec = {"id": sid, "cmd": shown, "returncode": rc, "errors": errs, "seconds": round(time.time() - t0, 1),
           "log": str(log.relative_to(workdir)).replace("\\", "/")}
    with PRINT_LOCK:
        status = "ok" if rc == 0 and errs == 0 else f"FAILED (exit {rc}, {errs} error(s), see {rec['log']})"
        print(f"  {sid:55s} {rec['seconds']:8.1f}s  {status}", flush=True)
    return rec


def run_chain(strategy, track, tdefs, category, workdir, logdir, dry):
    recs = []
    for step, script, args in steps_for(strategy, track, tdefs, category):
        r = run_step(f"{track}/{strategy}/{step}", script, args, workdir, logdir, dry)
        recs.append(r)
        if r["returncode"] != 0 or r["errors"] > 0:
            break      # later steps need this step's output; rerun the same command to resume
    return recs


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def manifest(args, workdir, strategies, tracks, tdefs, records, category, started, finished):
    initial = {f"{s}.txt" for s in STRATEGIES}
    files = {}
    for pat in ["*.py", "providers/*.py", "eval_prompts/*.txt"]:
        for p in sorted(workdir.glob(pat)):
            files[str(p.relative_to(workdir)).replace("\\", "/")] = sha256(p)
    for s in STRATEGIES:
        files[f"system_prompts/{s}.txt"] = sha256(workdir / "system_prompts" / f"{s}.txt")
    qh = hashlib.sha256()
    for p in sorted((workdir / "benchmark_questions").glob("cat*.txt")):
        qh.update(p.name.encode() + b"\0" + p.read_bytes())
    generated = {str(p.relative_to(workdir)).replace("\\", "/"): sha256(p)
                 for p in sorted((workdir / "system_prompts").rglob("*.txt")) if p.name not in initial or p.parent.name != "system_prompts"}
    pkgs = {}
    for name in ["google-genai", "openai", "mistralai"]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    stems = question_stems(workdir, category)
    return {
        "started_utc": started, "finished_utc": finished,
        "command_line": sys.argv[1:],
        "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
        "packages": pkgs,
        "generator_model": MODEL, "pairwise_judges": JUDGES,
        "feedback_tracks": {t: tdefs[t] for t in tracks},
        "settings": {"improvement_sampling_seed": 42, "temperature": "not set (provider default)",
                     "deepseek_route": "openrouter (thinking via reasoning flag)" if re.search(r"^ROUTE_VIA_OPENROUTER\s*=\s*True", (workdir / "providers" / "deepseek.py").read_text(encoding="utf-8"), re.M) else "direct (thinking budget 8192 tokens)",
                     "category_filter": category},
        "questions": len(stems), "expected_split": {"improvement": len(stems) // 2, "test": len(stems) - len(stems) // 2},
        "strategies": strategies,
        "sha256_code_and_initial_prompts": files,
        "sha256_questions_combined": qh.hexdigest(),
        "sha256_generated_prompts": generated,
        "steps": records,
        "failed_steps": [r["id"] for r in records if r["returncode"] != 0 or r["errors"] > 0],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help=f"1 strategy ({SMOKE_STRATEGY}), 1 category ({SMOKE_CATEGORY}), main track, in ./smoke_run/")
    ap.add_argument("--dry-run", action="store_true", help="print commands and the estimated API calls; call nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--tracks", nargs="+", default=["mistral", "deepseek", "gptoss"], choices=["mistral", "deepseek", "gptoss"])
    ap.add_argument("--workers", type=int, default=1, help="strategies processed in parallel (each worker has its own rate limiter; mind API limits)")
    ap.add_argument("--gptoss-model", default="openai/gpt-oss-120b:free", help="model of the GPT-OSS feedback judge (the study used the free route)")
    a = ap.parse_args()

    category = None
    strategies, tracks = a.strategies, a.tracks
    workdir = PIPE
    if a.smoke:
        strategies, tracks, category = [SMOKE_STRATEGY], ["mistral"], SMOKE_CATEGORY
        workdir = PIPE if a.dry_run else make_smoke_dir()
    tdefs = track_defs(a.gptoss_model)
    # order: the main track defines the improvement questions that the other tracks reuse
    tracks = sorted(set(tracks), key=lambda t: ["mistral", "deepseek", "gptoss"].index(t))

    for key in SCRIPTS.values():
        if not (workdir / key).is_file():
            sys.exit(f"missing script {key}")
    stems = question_stems(workdir, category)
    if not stems:
        sys.exit("no benchmark questions found in pipeline/benchmark_questions/")
    if "mistral" not in tracks and not a.dry_run:
        have = list((workdir / "improvements").glob(f"{MODEL}_*_improved_prompt.txt"))
        if not have:
            sys.exit("tracks deepseek/gptoss reuse the improvement questions of the main track: run the 'mistral' track first (it creates improvements/).")

    n, train, test, total = estimate(stems, strategies, tracks)
    print(f"strategies: {', '.join(strategies)}\ntracks: {', '.join(tracks)}   questions: {n}"
          f" ({'category ' + category if category else 'all categories'}) -> {train} improvement / {test} test")
    print("estimated API calls (upper bound, scripts skip finished work):")
    for k, v in total.items():
        print(f"  {k:36s}{v:8d}")
    print(f"  {'total':36s}{sum(total.values()):8d}")
    miss = missing_keys()
    if miss:
        print("\nmissing environment variables:\n  " + "\n  ".join(miss))
        if not a.dry_run:
            sys.exit(1)
    if a.dry_run:
        for t in tracks:
            for s in strategies:
                print(f"\n[{t}] {s}")
                run_chain(s, t, tdefs, category, workdir, None, True)
        return
    if not a.yes:
        try:
            answer = input("\nThis calls external APIs and may cost money. Proceed? [y/N] ") if sys.stdin.isatty() else ""
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            sys.exit("not confirmed: pass --yes to run without asking (the run calls external, possibly paid, APIs)")

    logdir = workdir / "logs"
    logdir.mkdir(exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    records = []
    for t in tracks:
        print(f"\n=== track {t} ===", flush=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(run_chain, s, t, tdefs, category, workdir, logdir, False) for s in strategies]
            for f in futs:
                records.extend(f.result())
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    m = manifest(a, workdir, strategies, tracks, tdefs, records, category, started, finished)
    (workdir / "run_manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    failed = m["failed_steps"]
    print(f"\n{len(records)} steps run, {len(failed)} failed. Manifest: {(workdir / 'run_manifest.json').relative_to(ROOT)}")
    if failed:
        print("failed steps (rerun the same command to resume):\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
