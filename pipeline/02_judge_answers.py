#!/usr/bin/env python3
"""
02_judge_answers.py

Score existing answer files using an LLM judge, saving results into:

  scores/   <model>_<strategy>_<question_id>_score.txt

Reads answers from the answers/ directory produced by
01_run_inference_with_prompts.py.  The --model and --strategy flags are used
only to reconstruct the file-prefix pattern; they do not affect the judge.

Usage
-----
    # Dry-run to verify plan without API calls
    python 02_judge_answers.py --strategy Chain_of_thought --dry-run

    # Score all answers for a strategy
    python 02_judge_answers.py --strategy my_strategy

    # Restrict to one category
    python 02_judge_answers.py --strategy my_strategy --category cat01

    # Use a different judge model or provider
    python 02_judge_answers.py --strategy my_strategy --judge-model mistral-large-2512
    python 02_judge_answers.py --strategy my_strategy --provider google --judge-model gemma-4-27b-it
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from providers import gemini, mistral, openrouter, deepseek
    from utils import RateLimiter, parse_retry_after, QUESTIONS_DIR
except ImportError as exc:
    sys.exit(f"Cannot import providers/utils: {exc}\nRun from the project root directory.")

# ── Configuration ───────────────────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "google":     gemini,
    "mistral":    mistral,
    "openrouter": openrouter,
    "deepseek":   deepseek,
}

DEFAULT_PROVIDER    = "mistral"
DEFAULT_MODEL       = "gemma-4-31b-it"
DEFAULT_JUDGE_MODEL = "mistral-large-2512"

BASE_DIR         = Path(__file__).parent
ANSWERS_DIR      = BASE_DIR / "answers"
SCORES_DIR       = BASE_DIR / "scores"
DEFAULT_SCORES_DIR = SCORES_DIR
EVAL_PROMPT_PATH = BASE_DIR / "eval_prompts" / "eval_prompt.txt"
IMPROVEMENTS_DIR = BASE_DIR / "improvements"

GRADE_RE       = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\b")
GRADE_LABEL_RE = re.compile(r"Grade\*?\*?:\*?\*?\s*(\d{1,2}(?:\.\d+)?)", re.IGNORECASE)
BOXED_RE       = re.compile(r"boxed\{(\d{1,2}(?:\.\d+)?)\}")
MAX_RETRIES = 4

_judge_limiter = RateLimiter(max_calls=1, period=6.0)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def file_prefix(model: str, strategy: str) -> str:
    return f"{model.replace(':', '_').replace('/', '_')}_{strategy}_"


def load_eval_template() -> str:
    if not EVAL_PROMPT_PATH.exists():
        sys.exit(f"Eval prompt not found: {EVAL_PROMPT_PATH}")
    return EVAL_PROMPT_PATH.read_text(encoding="utf-8").strip()


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


def extract_grade(text: str) -> float | None:
    # Prefer explicit "Grade: X" label (search from end so we catch the final verdict)
    for m in reversed(list(GRADE_LABEL_RE.finditer(text))):
        val = float(m.group(1))
        if 1.0 <= val <= 10.0:
            return val
    # Fallback: \boxed{X}
    for m in reversed(list(BOXED_RE.finditer(text))):
        val = float(m.group(1))
        if 1.0 <= val <= 10.0:
            return val
    # Last resort: first number in valid range
    for m in GRADE_RE.finditer(text):
        val = float(m.group(1))
        if 1.0 <= val <= 10.0:
            return val
    return None


# ── Score answers ───────────────────────────────────────────────────────────────

def score_answers(
    questions_dir: Path,
    answers_dir: Path,
    scores_dir: Path,
    prefix: str,
    judge_model: str,
    judge_query_fn,
    eval_template: str,
    category_filter: str | None,
    dry_run: bool,
    exclude_qids: set[str] | None = None,
    verbose: bool = False,
    limiter: "RateLimiter | None" = None,
) -> None:
    if limiter is None:
        limiter = _judge_limiter
    print("\n" + "=" * 60)
    print("  Scoring answers")
    print("=" * 60)
    print(f"  Output dir   : {scores_dir}")

    answer_files = sorted(answers_dir.glob(f"{prefix}*_response.txt"))
    if category_filter:
        answer_files = [f for f in answer_files if category_filter in f.name]
    if exclude_qids:
        answer_files = [
            f for f in answer_files
            if f.stem.removeprefix(prefix).removesuffix("_response") not in exclude_qids
        ]

    if not answer_files:
        print("  No answer files found — skipping scoring.")
        return

    scores_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for af in answer_files:
        q_id      = af.stem.removeprefix(prefix).removesuffix("_response")
        score_out = scores_dir / f"{prefix}{q_id}_score.txt"
        if score_out.exists():
            skipped += 1
        else:
            pending.append(af)

    print(f"\n  Already scored : {skipped}")
    print(f"  To score       : {len(pending)}")

    if not pending:
        print("  Nothing to do.")
        return

    if dry_run:
        for af in pending:
            q_id = af.stem.removeprefix(prefix).removesuffix("_response")
            print(f"  [DRY RUN] score {q_id}")
        return

    print()
    scored = errors = 0
    for af in pending:
        q_id      = af.stem.removeprefix(prefix).removesuffix("_response")
        score_out = scores_dir / f"{prefix}{q_id}_score.txt"

        answer_text = af.read_text(encoding="utf-8", errors="replace").strip()
        if answer_text.startswith("[ERROR]"):
            score_out.write_text("?\n\n(answer file contained an error)", encoding="utf-8")
            print(f"  [SKIP] Error answer — {af.name}")
            skipped += 1
            continue

        q_file = questions_dir / f"{q_id}.txt"
        if not q_file.exists():
            print(f"  [SKIP] Question file not found — {q_id}.txt")
            skipped += 1
            continue

        question_text = q_file.read_text(encoding="utf-8", errors="replace").strip()
        judge_prompt  = (
            f"Given the question:\n{question_text}\n\n"
            f"{eval_template}\n\n"
            f"Answer:\n{answer_text}"
        )

        print(f"  {q_id} ...", end=" ", flush=True)
        try:
            result    = call_with_retry(
                judge_query_fn, judge_model, "", judge_prompt,
                limiter=limiter,
            )
            if verbose:
                raw, thinking = result
            else:
                raw, thinking = result, None
            grade     = extract_grade(raw)
            grade_str = f"{grade:.1f}" if grade is not None else "?"
            content   = f"{grade_str}\n\n{raw}"
            if thinking:
                content += f"\n\n=== THINKING ===\n{thinking}"
            score_out.write_text(content, encoding="utf-8")
            print(f"score={grade_str}")
            scored += 1
        except Exception as exc:
            score_out.write_text(f"?\n\n[ERROR]: {exc}", encoding="utf-8")
            print(f"ERROR — {exc}")
            errors += 1

    print(f"\n  Scored: {scored}  |  Skipped: {skipped}  |  Errors: {errors}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing answers from answers/ using an LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy", required=True,
        help="Strategy name (used to match answer file prefix).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model used during inference (for prefix matching, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDER_REGISTRY),
        help=f"LLM provider for the judge (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument("--category", default=None, help="Restrict to one category, e.g. cat01.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without calling any API.")
    parser.add_argument(
        "--scores-dir", default=None,
        help="Override output directory for score files (default: scores/).",
    )
    parser.add_argument(
        "--base-prefix", default=None,
        help="If set, restrict to the test set: answer files whose question ID has no "
             "corresponding improvement file under improvements/<base-prefix>*_improved_prompt.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    prefix     = file_prefix(args.model, args.strategy)
    questions_dir = Path(str(QUESTIONS_DIR))
    provider_mod = PROVIDER_REGISTRY[args.provider]
    judge_fn     = getattr(provider_mod, "query_verbose", None) or provider_mod.query
    verbose      = hasattr(provider_mod, "query_verbose")
    scores_dir = Path(args.scores_dir) if args.scores_dir else DEFAULT_SCORES_DIR

    test_set: set[str] | None = None
    if args.base_prefix:
        trained = {
            f.stem.removeprefix(args.base_prefix).removesuffix("_improved_prompt")
            for f in IMPROVEMENTS_DIR.glob(f"{args.base_prefix}*_improved_prompt.txt")
        }
        test_set = None  # will filter inside score_answers via exclude set
        _trained_set = trained
    else:
        _trained_set = set()

    print(f"Strategy     : {args.strategy}")
    print(f"Model prefix : {args.model}")
    print(f"Judge model  : {args.judge_model}")
    print(f"File prefix  : {prefix}")
    print(f"Answers dir  : {ANSWERS_DIR}")
    print(f"Scores dir   : {scores_dir}")
    print(f"Category     : {args.category or 'all'}")
    print(f"Test-set only: {bool(args.base_prefix)}")
    print(f"Dry run      : {args.dry_run}")

    eval_template = load_eval_template()
    score_answers(
        questions_dir=questions_dir,
        answers_dir=ANSWERS_DIR,
        scores_dir=scores_dir,
        prefix=prefix,
        judge_model=args.judge_model,
        judge_query_fn=judge_fn,
        eval_template=eval_template,
        category_filter=args.category,
        dry_run=args.dry_run,
        exclude_qids=_trained_set if args.base_prefix else None,
        verbose=verbose,
    )

    print("\nDone.")
    print(f"  Scores : {scores_dir}")


if __name__ == "__main__":
    main()
