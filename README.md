# Predicting Chess Moves with a Compact Transformer

**MVA course — LLMs for Code and Proof**

Can a small language model learn the structure of legal chess moves? This project trains a compact causal Transformer on move sequences, with a square-level tokenizer and generation diagnostics. It includes model code, historical training logs and a matching saved model.

My coursework adaptations cover tokenization, model/configuration serialization, training settings and evaluator behavior. The original course framework is credited below.

## Implementation

- A GPT-style pre-normalized Transformer with learned token/position embeddings,
  causal self-attention, GELU feed-forward blocks and tied input/output weights.
- A modified square-level tokenizer: from/to squares, promotion destinations and
  special tokens. The current tokenizer has 132 entries.
- A Hugging Face Trainer pipeline with next-token loss and padded-label masking.
- Generation diagnostics and evaluation of first-attempt legality, retry legality
  and play against Stockfish. These are evaluation capabilities, not verified scores.

The course architecture and training/evaluation scaffolding are retained.
Local modifications relative to the checked-out course revision are documented in
[the source diff](docs/local_course_changes.patch): tokenizer replacement,
model/configuration and serialization changes, training settings, and evaluator
adjustments. This does not claim the whole framework was written from scratch.

## Contents

| Path | Contents |
|---|---|
| `src/` | Local model, tokenizer, data, training and evaluation code; tokenizer variants retained |
| `snapshots/final_model_v5/` | Matching TP2 configuration, tokenizer files and approximately 4 MB safetensors weights |
| `evidence/checkpoint_summary.csv` | Historical configuration, stored tensor counts and recorded validation loss |
| `evidence/runs/` | Small configuration/log exports from the TP2 checkpoint folders |
| `check.py`, `debug_generate.py` | Original exploratory parameter-budget and generation scripts |
| `submit.py` | Course submission utility; it can publish to the Hub and was not executed here |

## Model snapshot and historical evidence

The included `final_model_v5` has **6 layers, width 128, 4 heads, FFN width 339,
context 512 and vocabulary size 132**. Its safetensors header contains
**1,005,554 parameter elements**. It therefore exceeds the challenge's one-million
parameter cap by 5,554 (about 0.56%). It is published as a historical experiment,
not as proof of eligibility or a competition-winning model.

The included configuration agrees with its embedding dimensions and layer count.
The safe binary header was inspected without loading pickle-based training files.
Historical logs also contain smaller variants (812,800; 909,824; 993,856; 998,482
stored elements). Their validation losses are not directly comparable when
tokenizers, vocabularies or context lengths differ. No Elo, leaderboard position,
win rate or legality percentage is claimed without corresponding evidence.

## Setup and entry points

Use Python 3.10+ in an appropriate PyTorch environment:

```bash
python -m pip install -r requirements.txt
python -m src.train --help
python -m src.evaluate --help
```

The snapshot records Transformers 4.57.6. The dependency file bounds Transformers
to version 4 and includes TensorBoard used by the training code; it is not an exact
environment lockfile. Training uses `dlouapre/lichess_2025-01_1M` through Datasets;
data is downloaded separately and is not bundled in this repository.

A small training command, intended as a starting configuration rather than a
reproduction of a historical run:

```bash
python -m src.train --output_dir ./output --max_train_samples 1000 --val_samples 100 --n_layer 5 --n_embd 128 --n_head 4 --n_ctx 128 --n_inner 256 --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 2
```

Evaluate the included snapshot, starting with legality checks:

```bash
python -m src.evaluate --model_path ./snapshots/final_model_v5 --mode legal --n_positions 20 --seed 42
```

Full game evaluation requires a Stockfish executable supplied through the evaluator
options. Check outputs for fallback behavior when it is absent; do not report an
engine benchmark without identifying the opponent and settings.

## Preserved limitations

- Training CLI options for evaluation/saving frequency are parsed, but the current
  implementation hard-codes both intervals to 2,000 steps. Small runs may have no
  in-training evaluation; final weights are still saved by the training entry point.
- Model-class defaults, training CLI defaults and saved checkpoints differ. Always
  load the saved configuration when inspecting an existing model.
- `check.py` uses an older exploratory vocabulary setting; `debug_generate.py`
  has an original local checkpoint path. Adjust these exploratory scripts before use.
- No KV cache is implemented by this model; full-prefix generation has extra cost.
- Existing source and checkpoint files are preserved, not retrospectively rewritten
  into a claimed final competition submission. `TP2-redo` was not used.

Publication validation (2026-10-05): Python syntax, file integrity, basic checkpoint
header/configuration consistency and text secret-pattern checks. Training, model
loading and Stockfish evaluation were **not** rerun for this upload.

## Course provenance

Based on [ChessChallengeTemplate](https://github.com/nathanael-fijalkow/ChessChallengeTemplate)
and the [course arena](https://huggingface.co/spaces/LLM-course/Chess1MChallenge).
The original course metadata names Nathanaël Fijalkow and declares MIT;
it is preserved in `pyproject.toml`, with the original arena README in `docs/`.
See [provenance and exclusions](docs/PROVENANCE.md).


[Back to Ying Jin's project directory](https://github.com/PineappleBlowsnow#mva-projects)
