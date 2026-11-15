# Data-generation pipeline

Code that generates the data for the paper: answers from `gemma-4-31b-it` to 51 process-mining questions under 9 prompting strategies, judge scores, improved prompts, and pairwise comparisons of original vs. improved answers.

The scripts that turn the data into tables and figures are not included.

## How to run

```bash
pip install -r requirements.txt        # Python 3.10+, no GPU
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=...
export OPEN_ROUTER_API_KEY=...

python run_all.py --dry-run            # free: shows the commands and the number of API calls
python run_all.py --smoke --yes        # small test run (~34 API calls)
python run_all.py --yes                # full run (~7,600 API calls, many hours)
```

On Windows PowerShell, set keys with `$env:GOOGLE_API_KEY="..."`.

If a run stops, run the same command again. Existing outputs are skipped.
Use `python run_all.py --help` for more options.

## What is in here

```
run_all.py        runs everything
requirements.txt  dependencies
pipeline/
  01_...  02_...  03_...  04_...  05_...   the five steps (answer, judge, improve, merge, compare)
  system_prompts/        the 9 starting prompts
  eval_prompts/          judging prompts
  benchmark_questions/   the 51 questions
```

## Models and keys

| Role | Model | Key |
|---|---|---|
| Answers, improvements | `gemma-4-31b-it` | `GOOGLE_API_KEY` |
| Judge | `mistral-large-2512` | `MISTRAL_API_KEY` |
| Judge | `openai/gpt-oss-120b` | `OPENROUTER_API_KEY` |
| Judge | `deepseek-v4-pro` (thinking) | `OPEN_ROUTER_API_KEY` |

No temperature or other generation settings are set. The only seed is 42, used for the improvement/test question split.

## Data

The questions come from [PM-LLM-Benchmark v2.2](https://github.com/fit-alessandro-berti/pm-llm-benchmark), licensed GPL-3.0 (see `pipeline/benchmark_questions/NOTICE`).

## Reproducibility

Models are called through hosted APIs, so reruns give similar but not identical results. The data in the paper was made with earlier versions of these scripts and prompts, so exact numbers will differ.

`run_manifest.json` records versions, settings and file hashes for each run. Delete `pipeline/logs/` before sharing, since it contains local paths.
