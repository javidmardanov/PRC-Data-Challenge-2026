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

## Official attempt 1: V4, rejected

Published source: `5f7f293`. Local complete RMSE 258.0149003 versus V3
259.9769017. Strict forward checks improved the affected tail branch in both
months. Only three ranking rows changed. SHA256:
`f92c4411f49ae133449a342f63a771f7645a0d1390e391456f56c550c4ab23c8`.

Uploaded 2026-09-15 20:54:21 UTC; processed 20:54:59 UTC. Official RMSE
**271.5069**, all 344,841 pairs accepted. This is worse than V3's 271.0449;
the rare-tail change is rejected for subsequent combinations. No coefficient is
refitted to the official score. Two official attempts remain.

## Candidates for attempts 2 and 3 (not yet submitted)

Frozen candidate A2 predicts the residual from the supplied LOBT timestamp with
CatBoost: 3,500 trees, depth 10, learning rate 0.05, L2 20, seed 42. Its global
coefficient 0.630797588656986 was fitted on days 1–14 of July/November; later
dates improved in both months. Local complete RMSE is 257.7679019. A longer
5,000-tree alternative was worse overall and rejected. The initial trial using
validation early stopping was interrupted and excluded from selection.

An airport-specific coefficient variant, shrunk toward the global coefficient
with 1,000 pseudo-rows, has local RMSE 257.5250142. The missing-Rome flight
identity trial failed in summer; its separate winter-only hypothesis improved
later November dates. That winter correction is disjoint from the LOBT gate.
Neither planned candidate includes the rejected V4 tail change.

Complete chronological July refits: V3 297.3438985, global A2 293.8911651,
regional A2 293.7105852. The regional gain over global is uncertain under a
day-block bootstrap (95% SSE change interval -47.63M to +7.16M). These are
additional checks, not untouched
research holdouts: frozen coefficients were originally selected using these
months. Model fits use only preceding labels and fixed iteration counts.

Reweighting local errors to the ranking set's observed season, airport, and
missing-AOBT mix preserves the gains: V3 264.4347, global A2 262.1583, regional
A2 261.9169. The winter identity correction alone gives 263.6286. These are
diagnostics, not official score forecasts; November only approximates January.

Complete November checks now pass. Baseline V3 is 213.5902805; global A2 is
212.3631785; regional A2 is 212.2473897. Adding the winter identity correction
to regional A2 gives 211.2894892. The two corrections affect disjoint groups;
all other predictions remain bitwise unchanged. The combined November SSE gain
is 158.69M, with a descriptive day-bootstrap interval of 103.51M to 217.52M.
The frozen A2 model is therefore refitted on all 2025 labels for submission.

| Complete forward check | V3 | V5 recipe | V6 recipe |
| --- | ---: | ---: | ---: |
| July | 297.343899 | 293.867987 | 293.701634 |
| November | 213.590281 | 212.363178 | 211.289489 |

The final artifact review caught one negative ranking prediction in V5 and two
in V6. Restoring V3's nonnegative projection improved the affected local and
July forward checks; November was unchanged. No model or coefficient was
retuned. The table above includes this correction. Final local RMSE is
257.7546428 for V5 and 257.1286939 for V6, versus V3 259.9769017. Ranking-mix
weighted local RMSE is 262.1451119 and 261.0992930, respectively.

As of 2026-09-15 21:23 UTC, the best official score is still 271.0449, ranked
16 of 147 teams. Rank movement reflects new submissions from other teams.
