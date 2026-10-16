# Homework 8: assigned skill execution

The adjacent `SKILL.md` was extracted from the complete skill code block in the supplied `homework-07.html`, with HTML markup removed and character entities decoded. Its UTF-8 size is 2,683 bytes, matching the author's stated size. It is outside automatically loaded skill directories.

Explicit invocation used for this run:

> Follow the instructions in homework/week-08/peer-skill-run/SKILL.md to analyze homework/week-08/data/saheart.csv. Use chd as the response and all nine other variables as predictors, coding famhist as 1 for Absent and 2 for Present.

No additional analysis or report preferences were supplied. The skill requires ordinary LDA and QDA and makes additional candidates optional. This run used the two required candidates only. It does not reproduce the author's expanded candidate search or random seed.

Run `Rscript analysis.R` from this folder. The script uses MASS and base/recommended R packages. The completed run took under one second for the R process; chat token use and weekly quota were not available through the execution tools. No early stop or steer was needed. Re-running the script repeats the experiment; no repeat was used to revise model selection.

Outputs:

- `analysis-report.pdf`: fixed one-page statistical report.
- `analysis.R`: reproducible analysis and report generation.
- `cv-summary.csv`, `cv-fold-errors.csv`: training-only model comparisons.
- `split.csv`, `folds.csv`: observation assignments for audit.
- `fit-log.txt`: fitting failures, if any.
- `results.rds`: fitted model, transformations, selection and evaluation results.
- `test-confusion.csv`: final held-out confusion counts.

Seed: 43208. Training/test sizes: 323/139. Mean CV errors: LDA 0.2600, QDA 0.2815. Selected LDA; held-out test error 45/139 = 0.3237. No fitting failures occurred. This execution record is not a peer review. No GitHub Issue has been posted.
