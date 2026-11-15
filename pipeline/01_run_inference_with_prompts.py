#!/usr/bin/env python3
"""
01_run_inference_with_prompts.py

Generate answers for any strategy defined in system_prompts/, saving results into:

  answers/   <model>_<strategy>_<question_id>_response.txt

When --base-prefix is supplied, only questions that have NO corresponding
improvement file under improvements/<base-prefix>*_improved_prompt.txt are
processed (i.e. the held-out test set).  Use this for evaluating improved
strategies so the model is never tested on questions it was optimised against.

Two prompt-strategy layouts are supported:

  1. Single-file strategy:
       system_prompts/<strategy>.txt
     One prompt is applied to every question regardless of category.

  2. Per-category strategy:
       system_prompts/<strategy>/cat01.txt
       system_prompts/<strategy>/cat02.txt
       …
     Each category gets its own prompt; the matching file is selected
     automatically from the question filename.

Usage
-----
    # List available strategies
    python 01_run_inference_with_prompts.py --list

    # Dry-run to verify plan without API calls
    python 01_run_inference_with_prompts.py --strategy Chain_of_thought --dry-run

    # Generate answers for a strategy
    python 01_run_inference_with_prompts.py --strategy my_strategy

    # Restrict to one category
    python 01_run_inference_with_prompts.py --strategy my_strategy --category cat01

    # Use a different model or provider
    python 01_run_inference_with_prompts.py --strategy my_strategy --model gemma-4-27b-it
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    from providers import gemini, openrouter, deepseek
    from utils import RateLimiter, parse_retry_after, QUESTIONS_DIR
except ImportError as exc:
    sys.exit(f"Cannot import providers/utils: {exc}\nRun from the project root directory.")

# ── Configuration ───────────────────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "google":      gemini,
    "openrouter":  openrouter,
    "deepseek":    deepseek,
}

DEFAULT_PROVIDER = "google"
DEFAULT_MODEL    = "gemma-4-31b-it"

BASE_DIR           = Path(__file__).parent
SYSTEM_PROMPTS_DIR = BASE_DIR / "system_prompts"
ANSWERS_DIR        = BASE_DIR / "answers"
IMPROVEMENTS_DIR   = BASE_DIR / "improvements"

CAT_RE      = re.compile(r"(cat\d+)")
MAX_RETRIES = 4

_gen_limiter = RateLimiter(max_calls=1, period=6.0)

# ── Strategy loading ─────────────────────────────────────────────────────────────

def available_strategies() -> list[str]:
    names = set()
    for p in SYSTEM_PROMPTS_DIR.iterdir():
        if p.is_file() and p.suffix == ".txt":
            names.add(p.stem)
        elif p.is_dir() and any(p.glob("cat*.txt")):
            names.add(p.name)
    return sorted(names)


def load_strategy(strategy: str) -> str | dict[str, str]:
    single_file = SYSTEM_PROMPTS_DIR / f"{strategy}.txt"
    if single_file.exists():
        return single_file.read_text(encoding="utf-8").strip()

    cat_dir = SYSTEM_PROMPTS_DIR / strategy
    if cat_dir.is_dir():
        prompts = {
            f.stem: f.read_text(encoding="utf-8").strip()
            for f in sorted(cat_dir.glob("cat*.txt"))
        }
        if prompts:
            return prompts

    sys.exit(
        f"Strategy '{strategy}' not found.\n"
        f"Expected either:\n"
        f"  {single_file}          (single-file)\n"
        f"  {cat_dir}/cat01.txt …  (per-category directory)\n"
        f"Available strategies: {available_strategies()}"
    )


def prompt_for_question(strategy_prompts: str | dict, q_path: Path) -> str:
    if isinstance(strategy_prompts, str):
        return strategy_prompts
    m = CAT_RE.search(q_path.stem)
    cat = m.group(1) if m else None
    if cat and cat in strategy_prompts:
        return strategy_prompts[cat]
    sys.exit(
        f"No prompt found for category '{cat}' in the per-category strategy.\n"
        f"Available categories in strategy: {sorted(strategy_prompts)}"
    )


def strategy_mode(strategy_prompts: str | dict) -> str:
    return "single-file" if isinstance(strategy_prompts, str) else "per-category"


# ── Helpers ─────────────────────────────────────────────────────────────────────

def file_prefix(model: str, strategy: str) -> str:
    return f"{model.replace(':', '_').replace('/', '_')}_{strategy}_"


def call_with_retry(fn, *args, limiter, **kwargs) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.wait()
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = parse_retry_after(exc, attempt)
            print(f"    [WARN] Attempt {attempt} failed ({exc}); retrying in {wait:.0f}s ...")
            time.sleep(wait)


# ── Generate answers ─────────────────────────────────────────────────────────────

def generate_answers(
    question_files: list[Path],
    strategy_prompts: str | dict,
    answers_dir: Path,
    prefix: str,
    query_fn,
    model: str,
    dry_run: bool,
) -> None:
    mode = strategy_mode(strategy_prompts)
    print("\n" + "=" * 60)
    print("  Generating answers")
    print("=" * 60)
    print(f"  Prompt mode  : {mode}")
    if mode == "per-category":
        print(f"  Categories   : {sorted(strategy_prompts)}")
    print(f"  Output dir   : {answers_dir}")

    answers_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for qf in question_files:
        out = answers_dir / f"{prefix}{qf.stem}_response.txt"
        if out.exists() and not out.read_text(encoding="utf-8", errors="replace").startswith("[ERROR]"):
            skipped += 1
        else:
            pending.append(qf)

    print(f"\n  Already done : {skipped}")
    print(f"  To generate  : {len(pending)}")

    if not pending:
        print("  Nothing to do.")
        return

    if dry_run:
        for qf in pending:
            print(f"  [DRY RUN] {qf.stem}")
        return

    print()
    generated = errors = 0
    for qf in pending:
        q_id          = qf.stem
        out           = answers_dir / f"{prefix}{q_id}_response.txt"
        system_prompt = prompt_for_question(strategy_prompts, qf)
        question_text = qf.read_text(encoding="utf-8", errors="replace").strip()
        print(f"  {q_id} ...", end=" ", flush=True)
        try:
            response = call_with_retry(
                query_fn, model, system_prompt, question_text,
                limiter=_gen_limiter,
            )
            out.write_text(response, encoding="utf-8")
            print("done")
            generated += 1
        except Exception as exc:
            out.write_text(f"[ERROR]: {exc}", encoding="utf-8")
            print(f"ERROR — {exc}")
            errors += 1

    print(f"\n  Generated: {generated}  |  Skipped: {skipped}  |  Errors: {errors}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answers using a strategy from system_prompts/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strategy", help="Strategy name.")
    parser.add_argument("--list", action="store_true", help="List available strategies and exit.")
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDER_REGISTRY),
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model identity used for the answer-file prefix and to match downstream scripts "
             f"(default: {DEFAULT_MODEL}). Also used as the generation model unless "
             f"--gen-provider/--gen-model override it.",
    )
    parser.add_argument(
        "--gen-provider", default=None, choices=list(PROVIDER_REGISTRY),
        help="Provider to actually call for generation, if different from --provider "
             "(e.g. route the same model through OpenRouter instead of Google's direct API). "
             "Defaults to --provider.",
    )
    parser.add_argument(
        "--gen-model", default=None,
        help="Model id to actually call for generation, if different from --model "
             "(e.g. 'google/gemma-4-31b-it' on OpenRouter). File naming still uses --model. "
             "Defaults to --model.",
    )
    parser.add_argument("--category", default=None, help="Restrict to one category, e.g. cat01.")
    parser.add_argument(
        "--base-prefix", default=None,
        help="If set, restrict to the test set: questions that have no improvement file "
             "under improvements/<base-prefix>*_improved_prompt.txt.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="When used with --base-prefix, run on TRAINING questions only "
             "(those that DO have improvement files) instead of the test set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without calling any API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        strategies = available_strategies()
        print("Available strategies in system_prompts/:")
        for s in strategies:
            single = (SYSTEM_PROMPTS_DIR / f"{s}.txt").exists()
            tag    = "(single-file)" if single else "(per-category)"
            print(f"  {s}  {tag}")
        return

    if not args.strategy:
        print("Error: --strategy is required. Use --list to see available strategies.")
        sys.exit(1)

    gen_provider = args.gen_provider or args.provider
    gen_model    = args.gen_model or args.model

    strategy_prompts = load_strategy(args.strategy)
    prefix           = file_prefix(args.model, args.strategy)
    questions_dir    = Path(str(QUESTIONS_DIR))
    query_fn         = PROVIDER_REGISTRY[gen_provider].query

    print(f"Strategy     : {args.strategy}  [{strategy_mode(strategy_prompts)}]")
    print(f"Provider     : {args.provider}")
    print(f"Model        : {args.model}")
    print(f"Gen provider : {gen_provider}")
    print(f"Gen model    : {gen_model}")
    print(f"File prefix  : {prefix}")
    print(f"Answers dir  : {ANSWERS_DIR}")
    print(f"Category     : {args.category or 'all'}")
    print(f"Base prefix  : {args.base_prefix or '(none — full set)'}")
    print(f"Dry run      : {args.dry_run}")

    question_files = sorted(questions_dir.glob("cat*.txt"))
    if args.category:
        question_files = [f for f in question_files if f.stem.startswith(args.category)]

    if args.base_prefix:
        trained = {
            qid
            for f in IMPROVEMENTS_DIR.glob(f"{args.base_prefix}*_improved_prompt.txt")
            for qid in [f.stem.removeprefix(args.base_prefix).removesuffix("_improved_prompt")]
            if qid.startswith("cat")
        }
        if args.train_only:
            question_files = [f for f in question_files if f.stem in trained]
            print(f"  Train-set filter: {len(question_files)} questions selected "
                  f"({len(trained)} improvement files found)")
        else:
            question_files = [f for f in question_files if f.stem not in trained]
            print(f"  Test-set filter: {len(question_files)} questions remain "
                  f"({len(trained)} excluded as train set)")

    if not question_files:
        sys.exit(f"No question files found for category filter '{args.category}'.")

    generate_answers(
        question_files=question_files,
        strategy_prompts=strategy_prompts,
        answers_dir=ANSWERS_DIR,
        prefix=prefix,
        query_fn=query_fn,
        model=gen_model,
        dry_run=args.dry_run,
    )

    print("\nDone.")
    print(f"  Answers : {ANSWERS_DIR}")


if __name__ == "__main__":
    main()
