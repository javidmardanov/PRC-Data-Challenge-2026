# Challenge brief

Retrieved from the official challenge pages on 2026-09-08.

## Objective and evaluation

Predict departure taxi-out time, in seconds, for flights at ten major European airports. For a departure:

```text
TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt
```

The public ranking data covers January and July 2026. `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blank for every departure. The score is root mean square error (RMSE), and a team's best submission determines its rank.

The 2025 training set contains 4,167,797 arrival/departure movement rows. The downloaded ranking set contains 689,534 rows: 344,841 departures and 344,693 arrivals. The submission template has exactly the 344,841 departure IDs, in the same order as those departures in `ranking.parquet`.

As of the preserved 2026-09-08 leaderboard snapshot, the leading public score was 248.4818 seconds. The leaderboard is live; refresh it with `python scripts/fetch_leaderboard.py` before comparing results.

## Calendar and prize

- Competition: 2026-09-01 through 2026-10-11 23:59:59 CET, as stated by the organizers.
- Combined prize pool for the first three teams: EUR 5,000.

## Submission contract

- Fill only `TAXITIME_SEC_mvt` in the provided `submitting.parquet` template.
- Preserve every `MVT_ID_mvt`; do not add, remove, reorder, or mismatch rows.
- File name: `<team-name>_v<incremental integer>.parquet`.
- Upload to the team's own bucket. This credential set exposes `prc-2026-elegant-alligator`.
- Limit: five submissions per day and 1 GB total per bucket.
- The ranking service errors on mismatched IDs, missing rows, or extra rows.
- The organizers monitor attempts to learn from or exploit the ranking process.

## Prize-eligibility obligations

- Team must satisfy the participation/sanctions rules.
- Every external dataset must be openly accessible/usable, openly licensed, and documented.
- All produced source code must be publicly available on GitHub under GNU GPLv3.
- Documentation must be sufficient to understand and reproduce the result.
- The solution must be original. Existing implementations require usage rights and significant modification; merely changing I/O is insufficient.

## Data scope and caveats

Reporting airports: EDDF/FRA, EDDM/MUC, EGLL/LHR, EHAM/AMS, LEBL/BCN, LEMD/MAD, LFPG/CDG, LIRF/FCO, LTFM/IST, and LSZH/ZRH.

The official homepage says “11 airport movements,” while the data page, airport table, and files consistently describe ten airports; treat the homepage phrase as a typo unless the organizers clarify it.

Military, Head-of-State, and sensitive movements were removed. Movement records were left-joined to Network Manager flight records. The organizers explicitly warn that real-world matching inconsistencies remain and that the data are not synthetic.

## Important operational context

The organizers identify airline constraints, airport procedures/load, and Air Traffic Flow Management effects as interlinked sources of taxi-out variability. They frame the prediction as useful for post-operations analysis of constrained intervals and excess fuel/CO2 versus unconstrained conditions.

Important updates may appear first in the OpenSky Discord channel `#prc-data-competition`; the archived web pages are not a substitute for monitoring announcements.
