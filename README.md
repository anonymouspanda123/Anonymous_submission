# Data-generation pipeline

Code that implements the data-generation procedure of the paper: answers of `gemma-4-31b-it` to 51 process-mining questions under 9 prompting
strategies, LLM-judge scores, LLM-written improved prompts, and pairwise judge verdicts (original vs improved answer,
in both presentation orders, by three judges). Everything is run with one command, `run_all.py`.

**Scope.** This package regenerates the *data*. The scripts that turn the data into the paper's tables and figures
(win rates, confidence intervals, tests) are not part of it.

## Quick start

```bash
pip install -r requirements.txt                  # Python 3.11.4 tested; >= 3.10 required; no GPU
export GOOGLE_API_KEY=...  MISTRAL_API_KEY=...  OPENROUTER_API_KEY=...  OPEN_ROUTER_API_KEY=...   # see "API access"
                                                 # Windows PowerShell:  $env:GOOGLE_API_KEY="..."  (one line per variable)
python run_all.py --dry-run                      # free: prints every command and the estimated number of API calls
python run_all.py --smoke --yes                  # cheap end-to-end test (1 strategy, 1 category, ~34 API calls), runs in ./smoke_run/
python run_all.py --yes                          # the full run
```
The full run is resumable: every script skips outputs that already exist, so after an interruption or a failed step run the same command again.
`python run_all.py --help` lists the options (`--strategies`, `--tracks`, `--workers`, `--gptoss-model`).

## Layout

```
run_all.py            executes everything (preflight, stages, logs, manifest)
requirements.txt      pinned dependencies
pipeline/
  01_run_inference_with_prompts.py   answers every question with a strategy's system prompt            -> answers/
  02_judge_answers.py                grades answers 1-10 (eval_prompts/eval_prompt.txt)                -> scores/
  03_improve_prompt.py               per improvement question: question + answer + score -> improved prompt -> improvements/
  04_merge_improvements.py           merges the improvements into one prompt per category              -> system_prompts/<S>_merged/
  05_pairwise_compare.py             original vs improved answer, 3 judges in parallel                 -> pairwise_scores/
  utils.py   providers/              rate limiting; clients for Google, Mistral, OpenRouter, DeepSeek
  system_prompts/    the 9 initial prompts (base = empty = no system prompt)
  eval_prompts/      eval_prompt.txt (grading) and pairwise_eval_prompt_v2.txt (pairwise comparison)
  benchmark_questions/   the 51 questions, one text file each
```
The improvement and merge prompts are written inside `03` and `04`.

## What `run_all.py` does

For each strategy `S` (all 9 by default), in three feedback tracks. The track name says which judge scored the answers whose
feedback was used to write the improved prompts. The main track must run first because the others reuse its improvement questions.

| Track | Steps per strategy |
|---|---|
| `mistral` (main; judge `mistral-large-2512`) | `01` answers (51 questions) -> `02` scores -> `03` improvements -> `04` merge to `S_merged/` -> `01` answers on the held-out test questions with the merged prompt -> `05` pairwise, original shown first -> `05` pairwise, improved shown first |
| `deepseek` (`deepseek-v4-pro:thinking`) | `02` scores by that judge -> `03` improvements on the same questions -> `04` merge to `S_merged_<track>/` -> `01` test answers -> `05` pairwise, both orders |
| `gptoss` (`openai/gpt-oss-120b:free`) | same as `deepseek` with the GPT-OSS judge |

The pairwise comparison is always judged by the same three judges (`deepseek-v4-pro:thinking`, `mistral-large-2512`, `openai/gpt-oss-120b`).
A step is reported as failed if it exits non-zero **or** its log shows errors (the scripts exit 0 even when single API calls failed);
the run continues with the other strategies and exits with code 1 at the end.

## Outputs (full run)

| Folder (under `pipeline/`) | Content | Count |
|---|---|---|
| `answers/` | answers: 9 strategies x 51 questions, plus 27 merged variants x 26 test questions | 459 + 702 |
| `scores/`, `<track>/scores/` | grades of the original answers | 9 x 51 per judge |
| `improvements/`, `<track>/improvements/` | one improved prompt per improvement question | 9 x 25 per track |
| `system_prompts/<S>_merged*/` | merged prompts, one file per category | 27 folders |
| `pairwise_scores/`, `pairwise_scores_reversed/` | verdict JSON per pair, judge and question (original first / improved first) | 2,106 + 2,106 |
| `logs/`, `run_manifest.json` | one log per step; the record of the run (below) | |

`run_manifest.json` records the command line, Python and package versions, OS family, models, settings, the SHA-256 of every script, initial prompt and evaluation
prompt, of the question set and of every generated prompt, and every step with its exit code, error count and duration. It contains no user or host name.
Every pairwise verdict also stores the SHA-256 of the evaluation prompt it was produced with (`eval_prompt_sha256`).

## Data

- **Questions:** the 51 text questions of PM-LLM-Benchmark v2.2 (<https://github.com/fit-alessandro-berti/pm-llm-benchmark>), in 7 categories
  (`cat01`-`cat06`, `cat08`: 8, 9, 8, 7, 7, 7, 5 questions; the benchmark's category 7 has no text questions). Question IDs and per-category counts match the upstream index;
  the files were not diffed against the upstream repository. The benchmark is licensed **GPL-3.0** (its `LICENSE` file), so the questions in `pipeline/benchmark_questions/` are third-party
  material under that licence (see `pipeline/benchmark_questions/NOTICE`). The upstream README points to the paper describing the benchmark (arXiv 2407.13244, version 1.0) and gives no other citation format.
  The upstream repository is archived and marked deprecated in favour of PM-LLM-Benchmark3, which this study does not use.
- **Split:** `03` draws half of the questions at random (`--seed 42`, 25 of 51) as the *improvement* questions; the other 26 are the *test* questions on which improved prompts are compared with the originals.
  The draw is random and not stratified by category, so a fresh run uses its own split. The scripts define "test" as "no improvement file exists for the base strategy".
- No other data is read. All results are generated by calling the models below.

## API access and settings

| Role | Model (as called) | Provider code | Environment variable |
|---|---|---|---|
| Generator, improvement, merge | `gemma-4-31b-it` | `providers/gemini.py` | `GOOGLE_API_KEY` |
| Judge / feedback, Mistral | `mistral-large-2512` | `providers/mistral.py` | `MISTRAL_API_KEY` (optional extra keys `MISTRAL_API_KEY2`-`4`, `MISTRAL_APL_KEY6`, `MISTRAL_APL_KEY7`, rotated) |
| Judge / feedback, GPT-OSS | `openai/gpt-oss-120b` (feedback track: `:free` route) | `providers/openrouter.py` | `OPENROUTER_API_KEY` (or `OPEN_AI_API_KEY_PAID`) |
| Judge / feedback, DeepSeek | `deepseek-v4-pro` with thinking enabled | `providers/deepseek.py` | `OPEN_ROUTER_API_KEY` (calls are routed through OpenRouter with its `reasoning` flag; set `ROUTE_VIA_OPENROUTER = False` and `DEEPSEEK_API_KEY` to call DeepSeek directly, then a thinking budget of 8192 tokens applies) |

Settings that matter for reproduction: no temperature, top-p, max-token or other generation parameter is set (provider defaults apply);
the only seed is the improvement-question draw (42); the Google client sends the system prompt in front of the question in one user message (the other providers send a system message);
every script retries failed calls with back-off. Models are accessed through hosted APIs: versions behind a name can change and routes (e.g. the free GPT-OSS route) can be withdrawn,
which is the main reason a rerun need not match the original data (see "Reproducibility").

## Compute

No GPU and no local model: the machine only orchestrates API calls. A full run makes about **7,600 API calls**
(1,161 generation, 864 improvement/merge, 1,377 grading, 4,212 pairwise; `python run_all.py --dry-run` prints the exact figures for your options).
Wall-clock time is dominated by API latency and the built-in throttle of 1 call per 6 s in steps `01`-`04`, which alone gives a lower bound of roughly 6 hours when run sequentially
(`--workers N` processes strategies in parallel; each worker throttles separately, so mind your providers' rate limits). Monetary cost depends on your providers' prices and is not estimated here.

## Reproducibility

- **Deterministic parts:** the code, prompts, questions and settings are fixed and hashed in the manifest.
- **Non-deterministic parts:** LLM outputs. Reruns produce similar, not identical, answers and verdicts; the improvement/test split also differs (random draw), so exact numbers should not be expected. Compare distributions (e.g. win/tie/loss rates per judge), not individual files.
- **Checks:** `--dry-run` (plan and call counts), `--smoke` (end-to-end on a small slice), the per-step error check, and the manifest.

## Relation to the data reported in the paper

The data reported in the paper was produced during the study with earlier versions of these scripts and prompts, which changed while the study ran.
This package contains the final versions only. The known differences:

- **Pairwise prompt.** The package has one pairwise evaluation prompt (`pairwise_eval_prompt_v2.txt`: five criteria, ties allowed). In the study data of the merged variant,
  1,610 of 2,106 swapped verdicts and 152 of 2,106 unswapped verdicts were produced with an earlier prompt (scores for A1/A2 and a forced choice, no tie).
- **Grading prompt.** The package has the final grading prompt. The scores that fed the improvement step of the Mistral track were graded with an earlier wording of it.
- **Judge routes.** The swapped GPT-OSS verdicts were judged on the free OpenRouter route, the unswapped ones on the paid route; 32 swapped "Mistral" verdicts came from `mistral-large-2407`.
- **Code.** The Azure client and the OpenRouter attribution headers were removed and the prompt hash was added; no other script changed.

A dataset produced entirely under the prompts and settings of this package therefore requires running the pipeline (or re-judging the affected verdicts).

## Reproducibility checklist (items asked by NeurIPS, ICML and ICLR)

| Item | Where |
|---|---|
| Code to reproduce the main results, with instructions | `run_all.py`, "Quick start"; analysis scripts are out of scope (see "Scope") |
| Dependencies and versions | `requirements.txt` (pinned), `run_manifest.json` |
| Hardware / compute / run time | "Compute" |
| Datasets: source, size, split, licence | "Data": PM-LLM-Benchmark v2.2, 51 questions, 25 / 26 split, GPL-3.0 |
| Hyperparameters and settings | "API access and settings" |
| Random seeds and sources of non-determinism | "API access and settings", "Reproducibility" |
| Prompts used | `pipeline/system_prompts/`, `pipeline/eval_prompts/`; improvement and merge prompts inside `03` and `04` |
| Models and access | "API access and settings" |
| Anonymity (double-blind) | no author names, e-mail addresses, institutions or personal paths in code, prompts or manifest; logs contain absolute paths, so `pipeline/logs/` is excluded (`.gitignore`) and must be deleted before sharing |
| Licence | none for the code in this anonymised version (ICML asks for licences and author names to be removed from submitted code); add at camera-ready. The benchmark questions keep their own GPL-3.0 licence (`pipeline/benchmark_questions/NOTICE`) |
| Statistical significance | not produced here (analysis code) |
| Limitations | hosted models change, routes are withdrawn, no seeds for generation, random non-stratified split, the analysis is not included |

## Draft text for a reproducibility statement

> The data-generation procedure of this paper is released as `run_all.py`. It fixes the 51 benchmark questions, the nine initial system prompts, the evaluation prompts, the library versions and the one random seed
> (improvement/test split); it records code, prompt and question hashes in a run manifest and stamps every judge verdict with the hash of its evaluation prompt. The models are accessed through hosted APIs without fixed
> sampling parameters, so reruns reproduce the experimental design, not individual model outputs, and exact numbers will differ. API call counts and the compute needs are listed in the README.
