#!/usr/bin/env python3
"""
04_merge_improvements.py

Scans improvements/ for per-question improved prompts, groups them by
(strategy, category), and for each pair asks an LLM to merge all
per-question improvements with the original category prompt into one
final, clean system prompt.

Output — one file per (strategy, category):
    system_prompts/<strategy>_merged/
        cat01.txt
        cat02.txt
        …

This directory can be used directly in process 1:
    python 01_run_inference_with_prompts.py --strategy <strategy>_merged

Usage
-----
    # Process all strategies and all categories found in improvements/
    python 04_merge_improvements.py

    # Filter to one strategy
    python 04_merge_improvements.py --strategy Chain_of_thought

    # Filter to one strategy and one category
    python 04_merge_improvements.py --strategy Chain_of_thought --category cat01

    python 04_merge_improvements.py --dry-run

    # One prompt per strategy, pooling improvements across ALL categories
    # (no per-category split) -> system_prompts/<strategy>_single.txt
    python 04_merge_improvements.py --single-prompt --strategy base

    # Same, but batched to stay under free-tier per-minute token quotas:
    # merges groups of --batch-size questions first, then merges those
    # batch outputs into one final prompt.
    python 04_merge_improvements.py --single-prompt --strategy base --batch-size 5

    # --model identifies WHICH improvement files to read (matches the answering
    # model's filename prefix, e.g. gemma-4-31b-it). --gen-provider/--gen-model
    # optionally route the actual merge LLM call elsewhere (e.g. the same Gemma
    # model served via OpenRouter instead of Google's direct API, to dodge
    # Google's free-tier rate limits/outages) without changing which files are
    # discovered:
    python 04_merge_improvements.py --single-prompt --strategy base \\
        --gen-provider openrouter --gen-model google/gemma-4-31b-it:free
"""

import argparse
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

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent
SYSTEM_PROMPTS_DIR = BASE_DIR / "system_prompts"
IMPROVEMENTS_DIR   = BASE_DIR / "improvements"
QUESTIONS_DIR      = BASE_DIR / "benchmark_questions"
BATCH_CACHE_DIR    = BASE_DIR / ".merge_batch_cache"

# ── Prompt templates ───────────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = (
    "You are an expert in process mining and prompt engineering. "
    "You receive question → improvement pairs. Each improvement contains: "
    "mistakes made, proposed improvements, what the model did well, and a proposed system prompt. "
    "Your task is to produce one single system prompt that: "
    "- defines what a good answer looks like, based on what the model did well across all analyses; "
    "- includes all proposed improvements; "
    "- explicitly tells the model what mistakes to avoid. "
    "Remove redundancy, keep it clean and direct. "
    "Output ONLY the final system prompt — no preamble, no explanation."
)

MERGE_USER_TEMPLATE = """\
=== QUESTION → IMPROVEMENT PAIRS ({n} questions) ===
{improved_blocks}

---
Using the improved prompts above, merge them into one final system prompt \
optimised to respond to the questions listed above.\
"""

# Second-level merge: combine several batch-level candidate prompts (each
# already a merge of a subset of questions) into one final prompt. Used by
# --single-prompt --batch-size to stay under per-minute token quotas.

FINAL_MERGE_SYSTEM_PROMPT = (
    "You are an expert in process mining and prompt engineering. "
    "You receive several candidate system prompts, each already merged from a subset of "
    "question/improvement analyses for the same strategy. "
    "Your task is to combine them into one final system prompt that: "
    "- keeps every distinct improvement and quality standard mentioned across all candidates; "
    "- resolves overlaps and contradictions in favor of the more specific/stronger instruction; "
    "- stays clean, direct, and free of redundancy. "
    "Output ONLY the final system prompt — no preamble, no explanation."
)

FINAL_MERGE_USER_TEMPLATE = """\
=== CANDIDATE SYSTEM PROMPTS ({n} batches) ===
{candidate_blocks}

---
Combine the candidate system prompts above into one final system prompt.\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

MAX_RETRIES   = 4
_rate_limiter = RateLimiter(max_calls=1, period=6.0)


def call_llm(query_fn, model: str, user_text: str, system_prompt: str = MERGE_SYSTEM_PROMPT) -> str:
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


def build_improved_blocks(bodies: list[dict]) -> str:
    return "\n\n".join(
        f"--- {i+1}. ({entry['q_id']}) ---\n"
        f"QUESTION:\n{entry['question']}\n\n"
        f"IMPROVEMENT:\n{entry['analysis']}"
        for i, entry in enumerate(bodies)
    )


def merge_in_batches(query_fn, model: str, bodies: list[dict],
                      batch_size: int, batch_delay: float, cache_key: str) -> str:
    """
    Two-phase merge to stay under free-tier per-minute input-token quotas:
      1. Split `bodies` into chunks of `batch_size`; merge each chunk with the
         normal question->improvement merge prompt, producing one candidate
         system prompt per chunk.
      2. If more than one chunk, merge the candidate prompts together into a
         single final prompt.
    `batch_delay` seconds are slept between every LLM call in this function
    (not just between batches) so cumulative tokens/minute stay bounded.

    Each batch's candidate is cached to disk under BATCH_CACHE_DIR/<cache_key>_batchN.txt
    right after it succeeds. If a later batch or the final combine step fails
    (e.g. hits the per-minute token quota), re-running picks up the cached
    candidates instead of re-calling the LLM for batches already done.
    """
    chunks = [bodies[i:i + batch_size] for i in range(0, len(bodies), batch_size)]
    candidates: list[str] = []
    BATCH_CACHE_DIR.mkdir(exist_ok=True)

    for i, chunk in enumerate(chunks, 1):
        cache_file = BATCH_CACHE_DIR / f"{cache_key}_batch{i}.txt"
        print(f"      batch {i}/{len(chunks)} ({len(chunk)} question(s)) ...", end=" ", flush=True)
        if cache_file.exists():
            candidates.append(cache_file.read_text(encoding="utf-8"))
            print("cached")
            continue
        user_prompt = MERGE_USER_TEMPLATE.format(n=len(chunk), improved_blocks=build_improved_blocks(chunk))
        result = call_llm(query_fn, model, user_prompt, MERGE_SYSTEM_PROMPT).strip()
        cache_file.write_text(result, encoding="utf-8")
        candidates.append(result)
        print("done")
        if i < len(chunks) or len(chunks) > 1:
            time.sleep(batch_delay)

    if len(candidates) == 1:
        result = candidates[0]
    else:
        print(f"      combining {len(candidates)} batch outputs into final prompt ...", end=" ", flush=True)
        candidate_blocks = "\n\n".join(f"--- Candidate {i+1} ---\n{c}" for i, c in enumerate(candidates))
        final_user_prompt = FINAL_MERGE_USER_TEMPLATE.format(n=len(candidates), candidate_blocks=candidate_blocks)
        result = call_llm(query_fn, model, final_user_prompt, FINAL_MERGE_SYSTEM_PROMPT).strip()
        print("done")

    for i in range(1, len(chunks) + 1):
        (BATCH_CACHE_DIR / f"{cache_key}_batch{i}.txt").unlink(missing_ok=True)
    return result


def file_prefix(model: str, strategy: str) -> str:
    return f"{model.replace(':', '_').replace('/', '_')}_{strategy}_"


def discover_jobs(model: str, improvements_dir: Path = IMPROVEMENTS_DIR,
                   pool_categories: bool = False) -> dict[tuple[str, str], list[dict]]:
    """
    Scan improvements/ for per-question files and group by (strategy, category).

    Filename format:  <model>_<strategy>_<cat>_<rest>_improved_prompt.txt
    Returns: { (strategy, cat): [{"q_id": str, "question": str, "analysis": str}, …] }

    If pool_categories is True, all categories for a strategy are pooled into a
    single group keyed (strategy, "ALL") — used to build one prompt with no
    per-category split.
    """
    model_prefix = model.replace(":", "_").replace("/", "_") + "_"
    pattern      = re.compile(r"^(.+?)_(cat\d+_.+)_improved_prompt\.txt$")

    jobs: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(improvements_dir.glob("*_improved_prompt.txt")):
        name = f.name
        if not name.startswith(model_prefix):
            continue
        rest = name[len(model_prefix):]
        m    = pattern.match(rest)
        if not m:
            continue
        strategy = m.group(1)
        q_id     = m.group(2)
        cat      = "ALL" if pool_categories else q_id.split("_")[0]

        q_file   = QUESTIONS_DIR / f"{q_id}.txt"
        question = q_file.read_text(encoding="utf-8", errors="replace").strip() if q_file.exists() else ""
        analysis = f.read_text(encoding="utf-8", errors="replace").strip()

        jobs.setdefault((strategy, cat), []).append({
            "q_id":     q_id,
            "question": question,
            "analysis": analysis,
        })

    return dict(sorted(jobs.items()))


def load_original_prompt(strategy: str, category: str, original_strategy: str | None = None) -> str:
    src = original_strategy or strategy
    cat_file = SYSTEM_PROMPTS_DIR / src / f"{category}.txt"
    if cat_file.exists():
        return cat_file.read_text(encoding="utf-8").strip()
    single = SYSTEM_PROMPTS_DIR / f"{src}.txt"
    if single.exists():
        return single.read_text(encoding="utf-8").strip()
    sys.exit(f"Original prompt not found for strategy '{src}', category '{category}'.")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-question improvements into one prompt per (strategy, category).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strategy", default=None,
                        help="Filter to one strategy. Omit to process all strategies found.")
    parser.add_argument("--category", default=None,
                        help="Filter to one category (e.g. cat01).")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDER_REGISTRY))
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Model used to identify which improvement files to read "
                             f"(matches the answering model's filename prefix; default: {DEFAULT_MODEL}). "
                             f"Also used as the merge LLM unless --gen-provider/--gen-model override it.")
    parser.add_argument("--gen-provider", default=None, choices=list(PROVIDER_REGISTRY),
                        help="Provider to actually call for the merge LLM, if different from --provider "
                             "(e.g. route the same model through OpenRouter instead of Google's direct API). "
                             "Defaults to --provider.")
    parser.add_argument("--gen-model", default=None,
                        help="Model id to actually call for the merge LLM, if different from --model "
                             "(e.g. 'google/gemma-4-31b-it:free' on OpenRouter). File discovery still "
                             "uses --model. Defaults to --model.")
    parser.add_argument("--output-name", default=None,
                        help="Directory name for the merged output under system_prompts/ "
                             "(default: <strategy>_merged). Use for round-based naming, e.g. Chain_of_thought_r1.")
    parser.add_argument("--original-strategy", default=None,
                        help="Load the original system prompt from this strategy instead of "
                             "the one derived from the improvement filename. Needed for round-based training.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview merge prompts without calling the LLM.")
    parser.add_argument("--improvements-dir", default=None,
                        help="Override improvements input directory (default: improvements/).")
    parser.add_argument("--single-prompt", action="store_true",
                        help="Pool improvements across ALL categories into one prompt per strategy "
                             "(no per-category split). Output: system_prompts/<output-name>.txt "
                             "(default output-name: <strategy>_single).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Only used with --single-prompt. Split a strategy's pooled improvements "
                             "into batches of this many questions, merge each batch, then merge the "
                             "batch outputs into one final prompt. Use this to stay under free-tier "
                             "per-minute input-token quotas when a strategy has many improvement files.")
    parser.add_argument("--batch-delay", type=float, default=65.0,
                        help="Seconds to sleep between LLM calls when --batch-size is set (default: 65). "
                             "Keeps cumulative input tokens/minute under the free-tier cap.")
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
    gen_provider = args.gen_provider or args.provider
    gen_model    = args.gen_model or args.model
    query_fn    = None if args.dry_run else PROVIDER_REGISTRY[gen_provider].query
    improvements_dir = Path(args.improvements_dir) if args.improvements_dir else IMPROVEMENTS_DIR

    print(f"File-match model : {args.model}  (provider: {args.provider})")
    print(f"Gen provider     : {gen_provider}")
    print(f"Gen model        : {gen_model}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Improvements dir : {improvements_dir}")
    if args.single_prompt and args.batch_size:
        print(f"Batch size       : {args.batch_size}")
        print(f"Batch delay      : {args.batch_delay}s")

    jobs = discover_jobs(args.model, improvements_dir, pool_categories=args.single_prompt)

    if not jobs:
        sys.exit(
            f"\nNo improvement files found in {improvements_dir} for model '{args.model}'.\n"
            "Run process 3 first: python 03_improve_prompt.py --strategy <name>"
        )

    # Apply filters
    if args.strategy:
        jobs = {k: v for k, v in jobs.items() if k[0] == args.strategy}
    if args.category:
        jobs = {k: v for k, v in jobs.items() if k[1] == args.category}

    if not jobs:
        sys.exit("No jobs match the given --strategy / --category filters.")

    # Show plan
    strategies = sorted({k[0] for k in jobs})
    print(f"\nStrategies : {strategies}")
    print(f"Jobs       : {len(jobs)}  ({len(jobs)} strategy×category pairs)\n")

    merged = skipped = errors = 0

    for (strategy, cat), bodies in jobs.items():
        if args.single_prompt:
            out_name = args.output_name if args.output_name else f"{strategy}_single"
            out_file = SYSTEM_PROMPTS_DIR / f"{out_name}.txt"
        else:
            out_dir_name = args.output_name if args.output_name else f"{strategy}_merged"
            out_dir  = SYSTEM_PROMPTS_DIR / out_dir_name
            out_file = out_dir / f"{cat}.txt"

        use_batching = args.single_prompt and args.batch_size and len(bodies) > args.batch_size

        print(f"[{strategy}] [{cat}]  ({len(bodies)} question(s))", end="  ")

        if out_file.exists():
            print("[SKIP] already exists.")
            skipped += 1
            continue

        if args.dry_run:
            if use_batching:
                n_batches = -(-len(bodies) // args.batch_size)
                print(f"[DRY RUN]  {n_batches} batch(es) of up to {args.batch_size}, then 1 final merge")
            else:
                improved_blocks = build_improved_blocks(bodies)
                user_prompt = MERGE_USER_TEMPLATE.format(n=len(bodies), improved_blocks=improved_blocks)
                print(f"[DRY RUN]  prompt={len(user_prompt)} chars")
            continue

        print("merging ..." if not use_batching else "")
        try:
            if use_batching:
                cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{out_file.relative_to(SYSTEM_PROMPTS_DIR)}")
                result = merge_in_batches(query_fn, gen_model, bodies, args.batch_size, args.batch_delay, cache_key)
            else:
                improved_blocks = build_improved_blocks(bodies)
                user_prompt = MERGE_USER_TEMPLATE.format(n=len(bodies), improved_blocks=improved_blocks)
                result = call_llm(query_fn, gen_model, user_prompt).strip()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(result, encoding="utf-8")
            print(f"  saved -> {out_file.relative_to(SYSTEM_PROMPTS_DIR)}")
            merged += 1
        except Exception as exc:
            print(f"  ERROR - {exc}")
            errors += 1

    print(f"\nMerged: {merged}  |  Skipped: {skipped}  |  Errors: {errors}")
    print("\nDone.")


if __name__ == "__main__":
    main()
