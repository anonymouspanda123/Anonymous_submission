#!/usr/bin/env python3
"""
05_pairwise_compare.py

Compare two strategies answer-by-answer using multiple LLM judges running in parallel.
Each judge evaluates A1 vs A2 and returns structured JSON with scores + winner.

Position bias is mitigated by randomly swapping A1/A2 assignment per question.

Usage
-----
    # Dry run
    python 05_pairwise_compare.py --strategy-a base --strategy-b base_merged --dry-run \\
        --judge mistral:mistral-large-2512 \\
        --judge openrouter:openai/gpt-oss-120b \\
        --judge deepseek:deepseek-v4-pro:thinking

    # Full run with randomization
    python 05_pairwise_compare.py --strategy-a base --strategy-b base_merged --randomize \\
        --judge mistral:mistral-large-2512 \\
        --judge openrouter:openai/gpt-oss-120b \\
        --judge deepseek:deepseek-v4-pro:thinking

    # Custom output dir
    python 05_pairwise_compare.py --strategy-a base --strategy-b base_merged \\
        --judge mistral:mistral-large-2512 --scores-dir my_pairwise
"""

import argparse
import hashlib
import json
import re
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from providers import REGISTRY
    from utils import RateLimiter, parse_retry_after, QUESTIONS_DIR
except ImportError as exc:
    sys.exit(f"Cannot import providers/utils: {exc}\nRun from the project root directory.")

BASE_DIR         = Path(__file__).parent
ANSWERS_DIR      = BASE_DIR / "answers"
IMPROVEMENTS_DIR = BASE_DIR / "improvements"
EVAL_PROMPT_PATH = BASE_DIR / "eval_prompts" / "pairwise_eval_prompt_v2.txt"

CATEGORY_LABELS: dict[str, str] = {
    "cat01": "Event Log Preprocessing",
    "cat02": "Conformance Checking",
    "cat03": "Process Discovery",
    "cat04": "Process Simulation",
    "cat05": "Performance Analysis",
    "cat06": "Temporal Profiling",
    "cat08": "Declarative Modeling",
}
MAX_RETRIES     = 5
MODEL_PREFIX    = "gemma-4-31b-it"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_answers(strategy: str) -> dict[str, str]:
    prefix = f"{MODEL_PREFIX}_{strategy}_"
    out = {}
    for f in sorted(ANSWERS_DIR.glob(f"{prefix}cat*_response.txt")):
        q_id = f.stem.removeprefix(prefix).removesuffix("_response")
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if not text.startswith("[ERROR]"):
            out[q_id] = text
    return out


def load_questions() -> dict[str, str]:
    qdir = Path(str(QUESTIONS_DIR))
    return {f.stem: f.read_text(encoding="utf-8").strip() for f in qdir.glob("cat*.txt")}


def load_eval_template() -> str:
    if not EVAL_PROMPT_PATH.exists():
        sys.exit(f"Pairwise eval prompt not found: {EVAL_PROMPT_PATH}")
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


def extract_json(text: str) -> dict | None:
    for pattern in [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _flip_winner(w: str) -> str:
    if w == "Answer A": return "Answer B"
    if w == "Answer B": return "Answer A"
    return w


def _flip_answer_labels(text: str) -> str:
    """Swap 'Answer A' <-> 'Answer B' mentions in judge free text after a position swap.
    Without this, winner fields get flipped but the prose justification still refers
    to the pre-swap labels, contradicting the (correctly flipped) winner."""
    if not text:
        return text
    return re.sub(
        r"Answer A|Answer B",
        lambda m: "Answer B" if m.group(0) == "Answer A" else "Answer A",
        text,
    )


def unswap_result(parsed: dict) -> dict:
    """Reverse A/B swap so results always refer to original strategy_a/strategy_b."""
    result = dict(parsed)
    result["criteria"] = {
        k: {
            **v,
            "winner": _flip_winner(v.get("winner", "")),
            "justification": _flip_answer_labels(v.get("justification", "")),
        }
        for k, v in parsed.get("criteria", {}).items()
    }
    result["overall_winner"] = _flip_winner(parsed.get("overall_winner", ""))
    result["overall_justification"] = _flip_answer_labels(parsed.get("overall_justification", ""))
    return result


# ── Per-judge worker ──────────────────────────────────────────────────────────

def _is_valid_result(path: Path) -> bool:
    """A previously-written judge file only counts as done if it's a
    successful v2-schema result — not a saved error, an unparsed response,
    or a leftover pre-v2-prompt record (missing overall_winner/criteria)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if "error" in d:
        return False
    p = d.get("parsed")
    return bool(p and p.get("overall_winner") and "criteria" in p)


def run_judge(
    judge_key: str,
    provider_name: str,
    judge_model: str,
    common_qids: list[str],
    questions: dict[str, str],
    answers_a: dict[str, str],
    answers_b: dict[str, str],
    strategy_a: str,
    strategy_b: str,
    eval_template: str,
    judge_dir: Path,
    randomize: bool,
    dry_run: bool,
) -> None:
    provider = REGISTRY[provider_name]
    query_fn = getattr(provider, "query_verbose", None) or provider.query
    verbose  = hasattr(provider, "query_verbose")
    # openrouter is on the paid gpt-oss-120b model (routed to deepinfra/akashml,
    # bf16), not the free-tier upstream-limited one, so it no longer needs
    # extra-gentle pacing. mistral rotates across healthy keys, so it can take
    # >1 call/sec. deepseek's paid account handles >1 call/sec cleanly too.
    max_calls, period = {"mistral": (2, 1.0), "deepseek": (2, 1.0), "openrouter": (2, 1.0)}.get(provider_name, (1, 1.0))
    limiter  = RateLimiter(max_calls=max_calls, period=period)

    judge_dir.mkdir(parents=True, exist_ok=True)

    scored = skipped = errors = 0
    for q_id in common_qids:
        out_file = judge_dir / f"{q_id}.json"
        if out_file.exists() and _is_valid_result(out_file):
            skipped += 1
            continue

        q_text = questions.get(q_id, "")
        if not q_text:
            skipped += 1
            continue

        if dry_run:
            print(f"  [{judge_key}] DRY RUN: {q_id}")
            continue

        swap = randomize and random.random() < 0.5
        a1, a2 = (answers_b[q_id], answers_a[q_id]) if swap else (answers_a[q_id], answers_b[q_id])

        cat_id = q_id.split("_")[0]  # "cat01_01_..." -> "cat01"
        category = CATEGORY_LABELS.get(cat_id, cat_id)
        prompt = (eval_template
                  .replace("{Category}", category)
                  .replace("{Question}", q_text)
                  .replace("{Answer_A}", a1)
                  .replace("{Answer_B}", a2))

        print(f"  [{judge_key}] {q_id} ...", end=" ", flush=True)
        try:
            result = call_with_retry(query_fn, judge_model, "", prompt, limiter=limiter)
            raw, thinking = result if verbose else (result, None)

            parsed = extract_json(raw)
            if parsed and swap:
                parsed = unswap_result(parsed)

            record = {
                "q_id": q_id,
                "judge": judge_key,
                "judge_model": judge_model,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "swapped": swap,
                "eval_prompt_sha256": hashlib.sha256(eval_template.encode("utf-8")).hexdigest(),  # added for reproducibility
                "parsed": parsed,
                "raw": raw,
            }
            if thinking:
                record["thinking"] = thinking

            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            winner = (parsed or {}).get("overall_winner", "?")
            degree = (parsed or {}).get("overall_degree", "?")
            print(f"winner={winner} degree={degree}")
            scored += 1
        except Exception as exc:
            out_file.write_text(json.dumps({"q_id": q_id, "judge": judge_key, "error": str(exc)}, indent=2), encoding="utf-8")
            print(f"ERROR — {exc}")
            errors += 1

    print(f"  [{judge_key}] Finished — scored={scored}  skipped={skipped}  errors={errors}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(scores_dir: Path, strategy_a: str, strategy_b: str, judge_keys: list[str]) -> None:
    print("\n" + "=" * 60)
    print("  PAIRWISE SUMMARY")
    print("=" * 60)
    print(f"  {strategy_a}  vs  {strategy_b}\n")

    for judge_key in judge_keys:
        judge_dir = scores_dir / judge_key
        if not judge_dir.exists():
            continue

        wins_a = wins_b = ties = errors = 0
        degree_counts: dict[str, int] = {}

        for f in sorted(judge_dir.glob("*.json")):
            try:
                data   = json.loads(f.read_text(encoding="utf-8"))
                p      = data.get("parsed") or {}
                if not p:
                    errors += 1
                    continue
                winner = p.get("overall_winner", "")
                degree = p.get("overall_degree", "?")
                if winner == "Answer A":
                    degree_counts[degree] = degree_counts.get(degree, 0) + 1
                    wins_a += 1
                elif winner == "Answer B":
                    degree_counts[degree] = degree_counts.get(degree, 0) + 1
                    wins_b += 1
                elif winner == "Tie":
                    degree_counts[degree] = degree_counts.get(degree, 0) + 1
                    ties += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

        total = wins_a + wins_b + ties
        print(f"  Judge : {judge_key}")
        print(f"  {strategy_a:<30} wins: {wins_a}/{total}")
        print(f"  {strategy_b:<30} wins: {wins_b}/{total}")
        print(f"  Ties                           : {ties}/{total}")
        print(f"  Degree distribution            : {degree_counts}")
        if errors:
            print(f"  Errors                         : {errors}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise comparison of two answer strategies with parallel judges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strategy-a", required=True, help="First strategy (A1).")
    parser.add_argument("--strategy-b", required=True, help="Second strategy (A2).")
    parser.add_argument(
        "--judge", action="append", dest="judges", default=[], metavar="PROVIDER:MODEL",
        help="Judge as provider:model. Repeat to add multiple judges (run in parallel).",
    )
    parser.add_argument("--scores-dir", default="pairwise_scores", help="Output base directory.")
    parser.add_argument("--randomize", action="store_true", help="Randomly swap A1/A2 to reduce position bias.")
    parser.add_argument("--category", default=None, help="Restrict to one category, e.g. cat01.")
    parser.add_argument(
        "--base-prefix", default=None,
        help="If set, restrict comparison to the test set: questions with no improvement "
             "file under improvements/<base-prefix>*_improved_prompt.txt. Keeps all "
             "comparisons on the same held-out set regardless of how many answers exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.judges:
        sys.exit("Specify at least one judge with --judge PROVIDER:MODEL")

    judges = []
    for j in args.judges:
        if ":" not in j:
            sys.exit(f"Invalid judge format '{j}'. Use PROVIDER:MODEL")
        provider_name, model = j.split(":", 1)
        if provider_name not in REGISTRY:
            sys.exit(f"Unknown provider '{provider_name}'. Available: {list(REGISTRY)}")
        safe = model.replace("/", "_").replace(":", "_")
        judges.append((f"{provider_name}_{safe}", provider_name, model))

    answers_a  = load_answers(args.strategy_a)
    answers_b  = load_answers(args.strategy_b)
    questions  = load_questions()
    template   = load_eval_template()

    common = sorted(set(answers_a) & set(answers_b))
    if args.category:
        common = [q for q in common if q.startswith(args.category)]
    if args.base_prefix:
        train_ids = {
            qid
            for f in IMPROVEMENTS_DIR.glob(f"{args.base_prefix}*_improved_prompt.txt")
            for qid in [f.stem.removeprefix(args.base_prefix).removesuffix("_improved_prompt")]
        }
        common = [q for q in common if q not in train_ids]

    scores_dir = BASE_DIR / args.scores_dir / f"{args.strategy_a}_vs_{args.strategy_b}"

    if not args.dry_run:
        answers_out_dir = scores_dir / "answers"
        answers_out_dir.mkdir(parents=True, exist_ok=True)
        for q_id in common:
            out_file = answers_out_dir / f"{q_id}.json"
            if out_file.exists():
                continue
            record = {
                "q_id": q_id,
                "strategy_a": args.strategy_a,
                "strategy_b": args.strategy_b,
                "answer_a": answers_a[q_id],
                "answer_b": answers_b[q_id],
            }
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Strategy A   : {args.strategy_a}  ({len(answers_a)} answers)")
    print(f"Strategy B   : {args.strategy_b}  ({len(answers_b)} answers)")
    print(f"Common Qs    : {len(common)}")
    print(f"Judges       : {[j[0] for j in judges]}")
    print(f"Randomize    : {args.randomize}")
    print(f"Scores dir   : {scores_dir}")
    print(f"Dry run      : {args.dry_run}")

    with ThreadPoolExecutor(max_workers=len(judges)) as executor:
        futures = {
            executor.submit(
                run_judge,
                judge_key, provider_name, judge_model,
                common, questions, answers_a, answers_b,
                args.strategy_a, args.strategy_b,
                template, scores_dir / judge_key,
                args.randomize, args.dry_run,
            ): judge_key
            for judge_key, provider_name, judge_model in judges
        }
        for future in as_completed(futures):
            jk = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[{jk}] THREAD FAILED: {exc}")

    if not args.dry_run:
        print_summary(scores_dir, args.strategy_a, args.strategy_b, [j[0] for j in judges])

    print("Done.")


if __name__ == "__main__":
    main()
