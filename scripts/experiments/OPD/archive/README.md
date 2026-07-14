# Archived OPD Recipes

These 1-node recipes are retained as historical smoke-test references. They
are not the active OPD baseline and are not current validation targets.

- `megatron_teacher_baseline/` loads the teacher in Megatron. It still uses
  `--rm-type math`, so mixed-reward groups train with task GRPO plus OPD rather
  than pure distillation despite the older inline comment.
- `sglang_teacher_baseline/` serves a TP=1 teacher on GPU 7 of the same node.
  Its resource and scoring constraints motivated the dedicated 3-node recipe.

If a historical recipe must be reproduced, submit it through its archived
path, for example `OPD/archive/sglang_teacher_baseline/qwen3-8B`.
