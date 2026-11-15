#!/usr/bin/env python3
"""
03_improve_prompt.py

For a given strategy, samples half of the questions that have both an answer
and a score, then for each sampled question sends the original system prompt +
question + answer + score to an LLM and asks it to produce an improved prompt
for that specific question type.

One improved prompt is saved per question:
    improvements/<model>_<strategy>_<question_id>_improved_prompt.txt

These per-question improved prompts are later merged by process 4.

Usage
-----
    python 03_improve_prompt.py --strategy Chain_of_thought
    python 03_improve_prompt.py --strategy Chain_of_thought --category cat01
    python 03_improve_prompt.py --strategy Chain_of_thought --seed 99
    python 03_improve_prompt.py --strategy Chain_of_thought --prompts-only
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

try:
    from providers import gemini, openrouter
    from utils import RateLimiter, parse_retry_after
except ImportError as exc:
    sys.exit(f"Cannot import providers: {exc}\nRun from the project root directory.")

# ── Provider registry ──────────────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "google":     gemini,
    "openrouter": openrouter,
}

DEFAULT_PROVIDER = "google"
DEFAULT_MODEL    = "gemma-4-31b-it"
DEFAULT_SEED     = 42

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent
SYSTEM_PROMPTS_DIR = BASE_DIR / "system_prompts"
QUESTIONS_DIR      = BASE_DIR / "benchmark_questions"
ANSWERS_DIR        = BASE_DIR / "answers"
SCORES_DIR         = BASE_DIR / "scores"
IMPROVEMENTS_DIR   = BASE_DIR / "improvements"

CAT_RE = re.compile(r"(cat\d+)")

# ── Strategy loading (mirrors process 1) ──────────────────────────────────────

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
        f"  {SYSTEM_PROMPTS_DIR / strategy}/cat01.txt …  (per-category directory)"
    )


def strategy_mode(strategy_prompts: str | dict) -> str:
    return "single-file" if isinstance(strategy_prompts, str) else "per-category"


def file_prefix(model: str, strategy: str) -> str:
    return f"{model.replace(':', '_').replace('/', '_')}_{strategy}_"


def prompt_for_category(strategy_prompts: str | dict, cat: str) -> str:
    if isinstance(strategy_prompts, str):
        return strategy_prompts
    return strategy_prompts.get(cat, "")

# ── Prompt template ────────────────────────────────────────────────────────────

IMPROVEMENT_SYSTEM_PROMPT = """\
You are an expert in process mining and prompt engineering.
You receive the question, the answer, and the score.

=== QUESTION ===
{question}

=== ANSWER PRODUCED ===
{answer}

=== SCORE & EVALUATOR FEEDBACK ===
{score_block}

Please produce improvements that would lead to 10/10 answers:
1) Analyse what went wrong based on the judge answer — list the mistakes
2) Propose improvements
3) List what the model did well
4) Propose one system prompt that includes the improvements, maintains the good things done, and avoids the mistakes\
"""

IMPROVEMENT_USER_TEMPLATE = """\
=== ORIGINAL SYSTEM PROMPT ===
{system_prompt}

=== QUESTION ===
{question}

=== ANSWER PRODUCED ===
{answer}

=== SCORE & EVALUATOR FEEDBACK ===
{score_block}

---
Produce an improved system prompt for this type of question.\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

MAX_RETRIES   = 4
_rate_limiter = RateLimiter(max_calls=1, period=6.0)


def call_llm(query_fn, model: str, system_prompt: str, user_text: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _rate_limiter.wait()
            return query_fn(model, system_prompt, user_text)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = parse_retry_after(exc, attempt)
            print(f"    [WARN] Attempt {attempt} failed ({exc}); retrying in {wait:.0f}s …")
            time.sleep(wait)


def train_ids_from_base(base_prefix: str, improvements_dir: Path = IMPROVEMENTS_DIR) -> set[str]:
    """Return train question IDs — those with improvement files for the base prefix."""
    return {
        f.stem.removeprefix(base_prefix).removesuffix("_improved_prompt")
        for f in improvements_dir.glob(f"{base_prefix}*_improved_prompt.txt")
        if f.stem.removeprefix(base_prefix).removesuffix("_improved_prompt").startswith("cat")
    }


def collect_pairs(prefix: str, category: str | None,
                  allowed_ids: set[str] | None = None,
                  scores_dir: Path = SCORES_DIR) -> list[dict]:
    """Return all (question, answer, score, cat) triples that exist for the prefix/category."""
    pattern = f"{prefix}{category}_*_response.txt" if category else f"{prefix}cat*_response.txt"
    answer_files = sorted(ANSWERS_DIR.glob(pattern))

    pairs = []
    for af in answer_files:
        q_id       = af.stem.removeprefix(prefix).removesuffix("_response")
        if allowed_ids is not None and q_id not in allowed_ids:
            continue
        score_file = scores_dir    / f"{prefix}{q_id}_score.txt"
        q_file     = QUESTIONS_DIR / f"{q_id}.txt"

        if not score_file.exists() or not q_file.exists():
            continue

        m = CAT_RE.search(q_id)
        pairs.append({
            "q_id":        q_id,
            "cat":         m.group(1) if m else "",
            "question":    q_file.read_text(encoding="utf-8", errors="replace").strip(),
            "answer":      af.read_text(encoding="utf-8", errors="replace").strip(),
            "score_block": score_file.read_text(encoding="utf-8", errors="replace").strip(),
        })
    return pairs


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_improvement(
    strategy_name: str,
    strategy_prompts: str | dict,
    prefix: str,
    output_prefix: str,
    category: str | None,
    rng: random.Random,
    query_fn,
    model: str,
    prompts_only: bool,
    all_pairs: bool,
    base_prefix: str | None = None,
    train_only: bool = False,
    scores_dir: Path = SCORES_DIR,
    improvements_dir: Path = IMPROVEMENTS_DIR,
    base_improvements_dir: Path | None = None,
) -> None:
    improvements_dir.mkdir(parents=True, exist_ok=True)

    allowed_ids: set[str] | None = None
    if train_only and base_prefix:
        lookup_dir = base_improvements_dir if base_improvements_dir is not None else improvements_dir
        allowed_ids = train_ids_from_base(base_prefix, lookup_dir)
        print(f"  Train-only mode: {len(allowed_ids)} training questions")

    # Collect pairs — filtered to train set when requested
    pairs = collect_pairs(prefix, category, allowed_ids, scores_dir=scores_dir)
    if not pairs:
        print("  [SKIP] No answer/score pairs found.")
        return

    # Pre-flight: split into already done vs pending
    pending = []
    skipped = 0
    for pair in pairs:
        out = improvements_dir / f"{output_prefix}{pair['q_id']}_improved_prompt.txt"
        if out.exists():
            skipped += 1
        else:
            pending.append(pair)

    total = len(pairs)
    print(f"\n  Total pairs   : {total}")
    print(f"  Already done  : {skipped}")

    # Sample half of the pending questions (unless --all-pairs)
    if not all_pairs:
        n_sample = max(1, total // 2)
        if len(pending) > n_sample:
            pending = sorted(rng.sample(pending, n_sample - skipped if n_sample > skipped else n_sample),
                             key=lambda x: x["q_id"])
    print(f"  To improve    : {len(pending)}\n")

    if not pending:
        print("  Nothing to do.")
        return

    improved = errors = 0
    for pair in pending:
        q_id      = pair["q_id"]
        cat       = pair["cat"]
        out       = improvements_dir / f"{output_prefix}{q_id}_improved_prompt.txt"
        sys_prompt = prompt_for_category(strategy_prompts, cat)

        system_prompt = IMPROVEMENT_SYSTEM_PROMPT.format(
            question=pair["question"],
            answer=pair["answer"],
            score_block=pair["score_block"],
        )
        user_prompt = IMPROVEMENT_USER_TEMPLATE.format(
            system_prompt=sys_prompt,
            question=pair["question"],
            answer=pair["answer"],
            score_block=pair["score_block"],
        )

        if prompts_only:
            print(f"  [PROMPTS ONLY] {q_id}  ({len(user_prompt)} chars)")
            continue

        print(f"  {q_id} …", end=" ", flush=True)
        try:
            result = call_llm(query_fn, model, system_prompt, user_prompt).strip()
            out.write_text(result, encoding="utf-8")
            print("saved")
            improved += 1
        except Exception as exc:
            print(f"ERROR — {exc}")
            errors += 1

    if not prompts_only:
        print(f"\n  Improved: {improved}  |  Skipped: {skipped}  |  Errors: {errors}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one improved prompt per question for a given strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strategy",     required=True,
                        help="Strategy name (must match a file or dir in system_prompts/).")
    parser.add_argument("--provider",     default=DEFAULT_PROVIDER, choices=list(PROVIDER_REGISTRY))
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"Model for improvement (default: {DEFAULT_MODEL})")
    parser.add_argument("--seed",         default=DEFAULT_SEED, type=int,
                        help=f"Random seed for sampling (default: {DEFAULT_SEED})")
    parser.add_argument("--category",     default=None,
                        help="Process only this category (e.g. cat01).")
    parser.add_argument("--prompts-only", action="store_true",
                        help="Build prompts and preview without calling the LLM.")
    parser.add_argument("--output-strategy", default=None,
                        help="Save improvement files under this strategy name instead of --strategy. "
                             "Useful for round-based training (e.g. Chain_of_thought_r1).")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Use all available question pairs instead of sampling half.")
    parser.add_argument("--base-prefix", default=None,
                        help="File prefix of the base strategy (e.g. gemma-4-31b-it_base_). "
                             "Required when using --train-only.")
    parser.add_argument("--train-only", action="store_true",
                        help="Only improve training questions (those with improvement files "
                             "under the base prefix). Requires --base-prefix.")
    parser.add_argument("--scores-dir", default=None,
                        help="Override scores input directory (default: scores/).")
    parser.add_argument("--improvements-dir", default=None,
                        help="Override improvements output directory (default: improvements/).")
    parser.add_argument("--base-improvements-dir", default=None,
                        help="Directory to look up train IDs from when using --train-only. "
                             "Defaults to --improvements-dir if not set.")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    rng      = random.Random(args.seed)
    query_fn = None if args.prompts_only else PROVIDER_REGISTRY[args.provider].query

    strategy_prompts  = load_strategy(args.strategy)
    prefix            = file_prefix(args.model, args.strategy)
    output_strategy   = args.output_strategy or args.strategy
    output_prefix     = file_prefix(args.model, output_strategy)
    mode             = strategy_mode(strategy_prompts)
    scores_dir            = Path(args.scores_dir) if args.scores_dir else SCORES_DIR
    improvements_dir      = Path(args.improvements_dir) if args.improvements_dir else IMPROVEMENTS_DIR
    base_improvements_dir = Path(args.base_improvements_dir) if args.base_improvements_dir else None

    print(f"Strategy          : {args.strategy}  [{mode}]")
    print(f"Provider          : {args.provider}")
    print(f"Model             : {args.model}")
    print(f"Seed              : {args.seed}")
    print(f"Prompts only      : {args.prompts_only}")
    print(f"Output strat      : {output_strategy}")
    print(f"All pairs         : {args.all_pairs}")
    print(f"Train only        : {args.train_only}  (base prefix: {args.base_prefix or '-'})")
    print(f"Answers dir       : {ANSWERS_DIR}")
    print(f"Scores dir        : {scores_dir}")
    print(f"Output dir        : {improvements_dir}")
    print(f"Base improv dir   : {base_improvements_dir or '(same as output dir)'}")

    run_improvement(
        strategy_name=args.strategy,
        strategy_prompts=strategy_prompts,
        prefix=prefix,
        output_prefix=output_prefix,
        category=args.category,
        rng=rng,
        query_fn=query_fn,
        model=args.model,
        prompts_only=args.prompts_only,
        all_pairs=args.all_pairs,
        base_prefix=args.base_prefix,
        train_only=args.train_only,
        scores_dir=scores_dir,
        improvements_dir=improvements_dir,
        base_improvements_dir=base_improvements_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
