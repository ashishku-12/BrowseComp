# Multihop BrowseComp-style Dataset Pipeline

Generates hard-to-find, easy-to-verify multihop questions of the form
`A -> B -> C -> D`, following the BrowseComp-style dataset:
seed an entity, discover independently-sourced hops, filter out
parametric-shortcut risk, construct an obfuscated backward-built question,
check the reasoning graph for shortcuts/redundancy, and verify with a
blind-solve contamination check plus an evidence-only re-derivation check.

## Setup

```bash
pip install -r requirements.txt
```

The requirements file uses the PyTorch CUDA 12.8 wheel index. If this virtual
environment already has the CPU-only build installed, reinstall PyTorch with:
```bash
python -m pip uninstall -y torch
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

## Run

```bash
# Full pipeline
python main.py --domains "regional sports coaches" "indie film composers" --n-per-domain 5

# Just one step (e.g. re-run filtering after tuning its prompt)
python main.py --only-step 3
```

## Resumability (important)

Every step:
1. Reads its OWN output JSON file first (if it exists).
2. Skips any record whose `id` is already present there.
3. Only does work for what's missing.
4. Writes to disk immediately after EVERY record (atomic write), not just
   at the end of the step.

This means:
- If the process crashes or you kill it mid-run, nothing already written
  to `data/*.json` is lost.
- Re-running `main.py` (full pipeline or `--only-step N`) always picks up
  exactly where it left off - it never restarts from scratch.
- If you want to force-redo a specific record, delete its entry from the
  relevant `data/stepN_*.json` file (or delete the whole file to redo
  that step for everything).

## Output files (in `data/`, one per step, for inspection at every stage)

| File | Produced by | Contents |
|---|---|---|
| `step1_seeds.json` | Seed Agent | Candidate entity A's + attributes |
| `step2_chains.json` | Chain Explorer | Full A->B->C chains (or dead-end records) |
| `step3_filtered.json` | Filtering Agent | Chains + pass/fail salience & credibility verdict |
| `step4_questions.json` | Question Constructor | Obfuscated question + canonical answer |
| `step5_graphchecked.json` | Redundancy Check | + reasoning graph, shortcut/redundancy verdict |
| `step6_verified.json` | Verifier | **Final file** - + blind-solve & evidence checks, `final_verdict` |

The usable dataset is every record in `step6_verified.json` where
`verification.final_verdict == "accept"`.

## Model assignment (why each step uses what)

| Step | Model | Cost |
|---|---|---|
| 1 Seed, 2 Explorer, 3 Filter, 4 Constructor, 6b Evidence re-derivation | DeepSeek API (`deepseek-chat`/`deepseek-reasoner`) | Paid, usage-based |
| 5 Redundancy/shortcut check, 6a Blind-solve check | Local Qwen2.5-14B-Instruct (4-bit) | **Free** - your GPU only |

Steps 5 and 6a are the two quality gates most likely to let an accidentally-easy
question through, so they deliberately use a model from a different family
than DeepSeek - running it locally means that safeguard costs nothing extra.

## Tuning knobs

See `config.py`:
- `MAX_RELATION_ATTEMPTS_PER_HOP` / `MAX_BACKTRACKS_PER_CHAIN` - dead-end retry budget in Step 2.
- `LOW_CREDIBILITY_DOMAINS` - quick pre-filter list used in Step 3 before the LLM call.
- `DEEPSEEK_MODEL_CHEAP` vs `DEEPSEEK_MODEL_STRONG` - which steps use the lighter vs. stronger DeepSeek tier.
- `LOCAL_MODEL_ID` / `LOCAL_MODEL_4BIT` - which local model backs Steps 5 & 6a, and whether to quantize it.
