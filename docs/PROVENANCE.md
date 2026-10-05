# Source and publication scope

- Publication date: 2026-10-05.
- Source: MVA / 2, LLM for code and proof / TP2 / Chess1MChallenge only.
- Local course Git HEAD: 873a7e31cde349337050610a6dc68bff4a9a42cb.
- Original remote: https://huggingface.co/spaces/LLM-course/Chess1MChallenge.
- Course author attribution and MIT declaration remain in the original pyproject.
- All copied Python files are unchanged. `docs/local_course_changes.patch` describes
  the working-tree differences from the local course Git HEAD, not a claim about
  every line's individual authorship.
- README, dependency guidance, checkpoint audit and extracted numeric logs were
  added for portfolio publication. `training_log.json` exports only global step
  and the saved log history; no new training results were generated.
- Included weights: TP2 `my_model/final_model_v5/model.safetensors`, with matching
  configuration and tokenizer files. Other weight snapshots remain local.
- Excluded: TP2-redo, environments, Git internals, caches, datasets, the course arena
  app, optimizer/RNG state and pickle-based training arguments. These are not
  required to inspect the student's model work.
- No external course submission, inference, training or benchmark was run.

Copied-file hashes and original relative paths are in `source_manifest.json`.
