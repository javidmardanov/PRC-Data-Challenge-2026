# Official data dictionary

Columns ending in `_mvt` describe the reporting airport's movement record. Columns ending in `_flt` come from the EUROCONTROL Network Manager flight list. The two logical tables were left-joined by the organizers.

## Movement fields

| Column | Downloaded Arrow type | Meaning |
| --- | --- | --- |
| `MVT_ID_mvt` | `double` | Unique movement-record identifier. |
| `FLIGHT_ID_mvt` | `double` | Unique Network Manager flight identifier, if matched. |
| `FLIGHT_mvt` | `string` | Flight number shown to passengers. |
| `FLIGHT_RULE_mvt` | `string` | `I` IFR, `V` VFR, or `NA` unknown. |
| `ADEP_mvt` | `string` | ICAO departure aerodrome. |
| `ADES_mvt` | `string` | ICAO destination aerodrome. |
| `PHASE_mvt` | `string` | `DEP` departure or `ARR` arrival. |
| `MVT_TIME_UTC_mvt` | UTC timestamp | Best movement time: takeoff for departures, landing for arrivals. |
| `BLOCK_TIME_UTC_mvt` | UTC timestamp | Off-block for departures, in-block for arrivals. Blank for ranking departures. |
| `SCHED_TIME_UTC_mvt` | UTC timestamp | Scheduled departure or arrival time. |
| `AIRCRAFT_TYPE_mvt` | `string` | ICAO aircraft type, e.g. `A21N`. |
| `RUNWAY_mvt` | `string` | Departure/arrival runway ID. |
| `STAND_mvt` | `string` | Departure/arrival stand ID. |
| `TAXITIME_SEC_mvt` | `int32` | Taxi-out seconds for departures, taxi-in seconds for arrivals. Blank for ranking departures and the submission template. |

## Flight fields

| Column | Downloaded Arrow type | Meaning |
| --- | --- | --- |
| `LOBT_flt` | UTC timestamp | Last known off-block time. |
| `CALLSIGN_flt` | `string` | Flight callsign, e.g. `BAW6VB`. |
| `ADEP_flt` | `string` | ICAO departure aerodrome from NM. |
| `ADES_flt` | `string` | ICAO destination aerodrome from NM. |
| `ADES_FILED_flt` | `string` | Initially filed destination; a difference from `ADES_flt` indicates diversion. |
| `MARKET_SEGMENT_flt` | `string` | Mainline, Regional, Low-Cost, Business Aviation, All-Cargo, Charter, Military, Other, or Not classified. |
| `IOBT_flt` | UTC timestamp | Initial off-block time. |
| `FLIGHT_RULE_flt` | `string` | `I` IFR, `V` VFR, `Y` IFR then VFR, or `Z` VFR then IFR. |
| `FLIGHT_TYPE_flt` | `string` | `S` scheduled, `N` non-scheduled, `G` general aviation, `M` military (filtered), or `X` other. |
| `AIRCRAFT_TYPE_flt` | `string` | ICAO aircraft type. |
| `WK_TBL_CAT_flt` | `string` | Wake turbulence category: `L`, `M`, `H`, or `J`. |
| `AIRCRAFT_OPERATOR_flt` | `string` | Anonymized ICAO airline designator. |
| `EOBT_1_flt` | UTC timestamp | Estimated off-block time for the filed-plan (M1) trajectory. |
| `ARVT_1_flt` | UTC timestamp | Arrival time for the M1 trajectory. |
| `AOBT_3_flt` | UTC timestamp | Actual off-block time for the flown (M3) trajectory. |
| `ARVT_3_flt` | UTC timestamp | Arrival time for the M3 trajectory. |

## Files and verified row counts

| File | Rows |
| --- | ---: |
| `training_2025-01-01_2025-02-01.parquet` | 307,257 |
| `training_2025-02-01_2025-03-01.parquet` | 287,411 |
| `training_2025-03-01_2025-04-01.parquet` | 328,622 |
| `training_2025-04-01_2025-05-01.parquet` | 350,288 |
| `training_2025-05-01_2025-06-01.parquet` | 370,105 |
| `training_2025-06-01_2025-07-01.parquet` | 365,986 |
| `training_2025-07-01_2025-08-01.parquet` | 381,161 |
| `training_2025-08-01_2025-09-01.parquet` | 382,047 |
| `training_2025-09-01_2025-10-01.parquet` | 367,679 |
| `training_2025-10-01_2025-11-01.parquet` | 371,163 |
| `training_2025-11-01_2025-12-01.parquet` | 324,530 |
| `training_2025-12-01_2026-01-01.parquet` | 331,548 |
| **Training total** | **4,167,797** |
| `ranking.parquet` | 689,534 |
| `submitting.parquet` | 344,841 |

Every Parquet currently contains one row group. `ranking.parquet` has the same 30-column schema as training; `submitting.parquet` contains only `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.
