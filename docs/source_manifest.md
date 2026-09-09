# Source manifest

Retrieved 2026-09-08 unless noted otherwise.

## Primary challenge sources

- Home, scope, timeline, prize: <https://prc-data-challenge-2026.netlify.app/>
- Eligibility and open-source obligations: <https://prc-data-challenge-2026.netlify.app/eligibility.html>
- Problem rationale: <https://prc-data-challenge-2026.netlify.app/rationale.html>
- Dataset files and data dictionary: <https://prc-data-challenge-2026.netlify.app/data.html>
- Ranking and submission rules: <https://prc-data-challenge-2026.netlify.app/ranking.html>
- Live leaderboard: <https://prc-challenge-2026.vercel.app/>
- Full paginated leaderboard JSON: <https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard>
- Leaderboard OpenAPI document: <https://datacomp.opensky-network.org/api/openapi>
- Historical MinIO/S3 access instructions referenced by the 2026 organizers: <https://prc-data-challenge.netlify.app/data.html#using-minio-client>
- Data endpoint: <https://s3.opensky-network.org/> (authenticated S3 API; credentials are private).

Exact HTML snapshots of the five challenge pages and the OpenAPI file are under `docs/official/`. The current full leaderboard snapshot is `data/external/leaderboard_latest.json`.

## Organizer-linked definitions

- Flight callsigns: <https://skybrary.aero/articles/call-sign-confusion>
- EUROCONTROL market segments: <https://www.eurocontrol.int/publication/eurocontrol-market-segment-rules>
- ICAO flight-plan Items 8 and 9: <https://skybrary.aero/articles/flight-plan-completion>
- ICAO aircraft type designators: <https://www.icao.int/publications/doc8643/pages/search.aspx>

## Communications

- OpenSky Discord invitation: <https://discord.gg/opensky>
- Challenge channel after joining: `#prc-data-competition`

Any added training source must be recorded here (URL, retrieval date, version, license, and exact fields used) to preserve prize eligibility and reproducibility.

## Modeling sources added 2026-09-09 UTC

- Google TimesFM-3 source: https://github.com/google-research/timesfm/tree/v3.0.0, commit `331c6d33cb1ac2611de3056d0ac7164aab6301eb`, Apache-2.0 source license.
- Pretrained checkpoint: https://huggingface.co/google/timesfm-3.0-pytorch, revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. Separate TimesFM Non-Commercial License v1.0; prize eligibility unconfirmed. No weights redistributed. Used to generate hourly taxi forecast deciles from official challenge histories/covariates; see `timesfm_notes.md`.
- IEM ASOS/METAR archive: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=; public-domain statement https://mesonet.agron.iastate.edu/disclaimer.php. Ten airport stations, 2025 plus January/July 2026, with preceding-day context. Temperature, dewpoint, wind, gust, visibility, pressure, and weather codes; see `weather_notes.md`. Source URLs, SHA-256 hashes, retrieval timestamp and exact cached sizes are in `data/processed/weather_hourly.json`; total download 12,585,360 bytes.
