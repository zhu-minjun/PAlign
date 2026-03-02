# Reproducing Table 1: Big Five Personality Alignment (PAS)

Step-by-step guide to reproducing the Big Five (OCEAN) results from
**"Personality Alignment of Large Language Models"** (Zhu et al., ICLR 2025)
using the PAS method on Llama-3-8B-Instruct.

**Target results (Table 1, MAE — lower is better):**

| A    | C    | E    | N    | O    |
|------|------|------|------|------|
| 0.94 | 0.91 | 0.86 | 0.98 | 0.72 |

---

## Prerequisites

- **GPU**: NVIDIA GPU with ≥24 GB VRAM (e.g. RTX 4090, A5000). The model loads in float16 and uses ~20.5 GB.
- **Python**: 3.10+
- **HuggingFace access**: Accepted the [Meta-Llama-3-8B-Instruct license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and logged in (`huggingface-cli login`).

---

## Step 1: Create a conda environment

```bash
conda create -n palign python=3.11 -y
conda activate palign
```

## Step 2: Install PyTorch

Install a CUDA-compatible PyTorch build. Check https://pytorch.org/get-started/locally/ for the command matching your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 3: Install the project and dependencies

From the repository root:

```bash
pip install .
```

This single command installs all required dependencies (including `baukit`,
`einops`, `transformers>=4.50`, `scikit-learn`, etc.).

## Step 4: Verify data files

The PAPI dataset is already included in the repository. Confirm the three required files exist:

```
PAPI/Test-set.json          # 300 clustered subjects with personality scores
PAPI/mpi_300_split.json     # Train/test question index split (120/180)
PAPI/IPIP-NEO-ItemKey.xls   # Question text lookup table
```

No additional data download is needed for the Big Five reproduction.

## Step 5: Quick sanity check (5 subjects)

Run on a small subset first to verify everything works:

```bash
python main.py \
  --modes PAS \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --num_subjects 5
```

This takes ~30–35 minutes. When it finishes you will see output like:

```
{'score': {'mean_A': ..., 'mean_C': ..., 'mean_E': ..., 'mean_N': ..., 'mean_O': ...}}
```

Check that:
- All `mean_*_abs` values are in the range [0, 4] (no NaN).
- The `UNK` count in each subject's result is low (< 15% of answers).
- Results are saved to `reproduction/PAS_Meta-Llama-3-8B-Instruct_OOD.json`.

## Step 6: Full reproduction run (300 subjects)

```bash
python main.py \
  --modes PAS \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --num_subjects 0
```

`--num_subjects 0` means "all 300 subjects".

**Expected runtime:** ~35 hours (each subject takes ~6–7 minutes: 6 alpha
values × 180 test questions in batches of 3).

### Running in the background

Use `nohup` or `tmux`/`screen` since this is a long run:

```bash
nohup python main.py \
  --modes PAS \
  --model_file meta-llama/Meta-Llama-3-8B-Instruct \
  --num_subjects 0 \
  > reproduction/full_run.log 2>&1 &
```

### Resuming after interruption

The script saves a pickle file per completed subject in
`reproduction/subject_results/subject_XXXX.pkl`. If the process is killed,
simply re-run the same command — it will detect completed subjects and skip
them automatically.

Monitor progress:

```bash
# Number of subjects completed so far
ls reproduction/subject_results/*.pkl | wc -l

# Latest progress entries
tail reproduction/pas_progress.jsonl
```

## Step 7: Read the results

Final results are written to:

```
reproduction/PAS_Meta-Llama-3-8B-Instruct_OOD.json
```

The key field is `score`. The values to compare against Table 1 are:

| Field         | Trait | Paper target |
|---------------|-------|--------------|
| `mean_A_abs`  | A     | 0.94         |
| `mean_C_abs`  | C     | 0.91         |
| `mean_E_abs`  | E     | 0.86         |
| `mean_N_abs`  | N     | 0.98         |
| `mean_O_abs`  | O     | 0.72         |

Print them quickly:

```bash
python -c "
import json
with open('reproduction/PAS_Meta-Llama-3-8B-Instruct_OOD.json') as f:
    r = json.load(f)
s = r['score']
print(f\"A={s['mean_A_abs']:.2f}  C={s['mean_C_abs']:.2f}  E={s['mean_E_abs']:.2f}  N={s['mean_N_abs']:.2f}  O={s['mean_O_abs']:.2f}\")
"
```

---

## Troubleshooting

**CUDA out of memory** — The batch size is hardcoded to 3 in
`main.py:generateAnswer()`. If you still OOM, reduce it to 1. This will
roughly triple runtime.

**`baukit` import error** — Re-run `pip install .` from the repository root.
`baukit` is installed automatically from its GitHub source.

**HuggingFace gated model error** — Run `huggingface-cli login` and ensure you
have accepted the Llama 3 license at
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.

**High UNK rate** — If > 20% of answers are unparseable, check that the
tokenizer and chat template are applied correctly. The answer parser
(`baseline_utils.py:process_answers`) looks for a letter A–E in the first 12
characters of the model output.

**Results differ from paper** — Small deviations are expected because the
logistic regression probe uses a random 60/40 train/val split
(`PAlign/pas.py:train_probes`, no fixed seed for the split itself). The paper
reports averages over 300 subjects, which smooths out per-subject variance.
