# Archived OPD Recipes

These recipes are retained as historical experiment references. They are not
the active OPD recipe and are not current validation targets.

- `megatron_teacher_baseline/` loads the teacher in Megatron. It still uses
  `--rm-type math`, so mixed-reward groups train with task GRPO plus OPD rather
  than pure distillation despite the older inline comment.
- `sglang_teacher_baseline/` serves a TP=1 teacher on GPU 7 of the same node.
  Its resource and scoring constraints motivated the dedicated 3-node recipe.
- `math_qwen3_32b_8b_3nodes/` is the former canonical 3-node sampled-RKLD
  recipe plus its persistent-HTTP A/B wrapper. It intentionally has no eval.
- `math_qwen3_32b_8b_3nodes_legacy_teacher/` preserves the bounded legacy
  teacher-top-k -> student-rescore characterization used by milestone 00.

If a historical recipe must be reproduced, submit it through its archived
path, for example `OPD/archive/sglang_teacher_baseline/qwen3-8B`.
