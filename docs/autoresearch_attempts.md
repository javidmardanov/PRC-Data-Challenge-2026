# Three official attempts

User-authorized budget: at most three new official submissions after V3.
The parent alone uploads. Local experiments do not consume an official attempt.

Following the experiment/measure/retain loop from
https://github.com/karpathy/autoresearch, each bounded trial records its hypothesis,
parameters, matched complete-pipeline scores, failure or acceptance, and artifacts.
No automatic claim of winning or monotonic improvement is made.

Incumbent V3: local July/November 259.9769017; official 271.0449. Existing V3
validation and ranking files are frozen. New experiments use separate names.
December labels are excluded from new selection. Ranking departure labels are
unavailable and official scores are never used to fit parameters.

Selection checks: lower complete pooled validation RMSE, improvement on later
calibration-excluded dates in both July and November, and inspect strict forward
comparisons where available. Prior research on these months prevents claiming
untouched validation. Full chronological V3 needs independently refitted bases
and specialists; the earlier six diagnostic refits are not that baseline.

Initial lanes: timestamp residual reference (LOBT), missing-timestamp residual
experts, and constrained recalibration of cached models. CPU and GPU jobs are
bounded; only one worker owns the GPU at once. Keep rejected results.

Before each upload: freeze recipe; final refit; independently verify full
submission IDs, finite nonnegative outputs, correction arithmetic and protected
regimes; publish the produced source as required by the challenge; record hash
and upload receipt; retrieve official score and rank. A worse submission cannot
displace the team's best score, but its failure must remain in the report.
