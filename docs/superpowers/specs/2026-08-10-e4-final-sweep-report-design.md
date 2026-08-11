# E4 GSM8K and Math Final Sweep Report Design

## Goal

Replace the partial GSM8K-only E4 report with one durable experiment record for
the completed GSM8K and Math FullFT-versus-LoRA learning-rate sweeps. The report
must make the positive and negative results equally easy to recover, while
remaining self-contained and reproducible from the repository and recorded
cluster evidence.

## Record shape

The final record will use the repository's existing HTML-report convention:

- Markdown source at
  `docs/reports/_src/2026-08-10-e4-gsm8k-math-panel.md`;
- self-contained rendered artifact at
  `docs/reports/2026-08-10-e4-gsm8k-math-panel.html`;
- regenerated `docs/reports/index.html`;
- removal of the superseded August 8 GSM8K-only source and artifact.

This is one experiment record rather than one report per dataset because both
datasets use the same E4 hypothesis, arm definitions, and learning-rate grid.
Keeping them together makes cross-dataset stability and optimum comparisons
explicit and avoids duplicate records for the same sweep.

## Evidence and provenance

The remote cluster remains the authoritative runtime. Collection will use one
bounded snapshot of the completed result ledgers and only the logs needed to
support reported trajectories or abnormal outcomes. Successful rows will be
deduplicated by arm; stale failed attempts will not replace later successful
rows. The report will record:

- local and remote branch and commit identity, plus clean or dirty state;
- SSH host alias, HTCondor scheduler, managed-session owner, job IDs, and GPU
  resource classes where they can be recovered;
- exact launcher commands, result-ledger paths, and relevant log paths;
- completion counts for each dataset and the two intentionally abandoned Math
  LoRA `1e-3` arms.

No run-store contents or raw cluster outputs will be committed. Only the
derived Markdown and self-contained HTML record belong in Git.

## Content

The report will be organized as follows:

1. **Executive result.** State the best FullFT and LoRA endpoint for each
   dataset, the LR ratio, and whether the sweep supports LoRA parity.
2. **Setup.** Record the shared model, training protocol, rank choices, dataset
   differences, learning-rate mapping, rollout budget, evaluation cadence,
   seed count, commands, and hardware.
3. **Results.** Show complete endpoint-accuracy matrices for GSM8K and Math,
   including the LoRA-only `2e-6` LR0 point. Add compact plots or trajectory
   tables only when the recovered evidence supports them.
4. **Interpretation.** Compare LR sensitivity, rank sensitivity, and
   cross-dataset robustness without presenting a single-seed sweep as a
   variance estimate.
5. **Negative results and run history.** Describe collapsed or degenerate arms,
   the intentionally abandoned Math LoRA `1e-3` runs, stale failed attempts,
   and any W&B synchronization limitation separately from training success.
6. **Limitations and next steps.** Call out the single-seed design and any
   missing checkpoint or trajectory evidence.

Missing evidence will be labeled as missing; it will not be inferred from an
endpoint or from a scheduler success state.

## Verification

Before publication:

- reconcile every reported endpoint against the deduplicated successful
  ledger rows and verify the expected arm counts;
- render with the `html-reports` tool, regenerate the index, and resolve every
  renderer warning;
- confirm there are no unparsed math blocks or external/local image sources;
- visually inspect the combined report at desktop and narrow widths;
- run `git diff --check` and review the complete report diff;
- commit and push only after the report and index pass these checks.

## Deliberate exclusions

- No new training, evaluation, or W&B synchronization is part of this task.
- No raw result ledgers, logs, checkpoints, credentials, or run-store snapshots
  will be added to Git.
- The report will not claim statistical significance from one seed.
