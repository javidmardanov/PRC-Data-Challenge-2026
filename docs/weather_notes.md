# Historical airport METAR weather

The Iowa Environmental Mesonet (IEM), Iowa State University, explicitly states
that the materials on its website are public domain and freely usable for lawful
purposes. Attribution is appreciated. This is the documented reuse basis for
this external dataset; no unstated third-party license is assumed.
[Official usage statement](https://mesonet.agron.iastate.edu/disclaimer.php).

`python scripts/weather_features.py` retrieves routine and special METAR reports
for EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, and LTFM using the
[official archive API](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=).
It caches two CSV requests: 2024-12-31 through 2026-01-31, and 2026-06-30 through
2026-07-31. The extra preceding day provides prior observations at period start.
Downloads use no credentials and are capped at 100 MB total. Source URLs,
retrieval time, SHA256 hashes, sizes, and coverage statistics are preserved in
`data/processed/weather_hourly.json`.
The executed download's metadata is also preserved in the published
[weather source manifest](weather_source_manifest.json).

Output `data/processed/weather_hourly.parquet` joins on `ADEP_mvt` and UTC `hour`,
the same keys as the TimesFM feature tables. For each airport-hour, the latest
weather report at or before the hour is used, with a maximum age of three hours.
This is stricter than using the average weather during that hour, which could
include observations after a flight. The script has a runnable check ensuring a
future report cannot enter an earlier hour, and checks every actual observation age.
No flight targets are read or used.

The completed download contains 206,374 observations and occupies 12,585,360
bytes. The output has 102,480 airport-hours. All 344,841 ranking departures match
a weather report and nonmissing temperature. Across all output hours, temperature
is missing for only three hours; snow-related codes appear in 739 hours and
freezing-weather codes in 392 hours. Gust is missing in roughly 98% of reports
and wind direction in roughly 10%; these remain missing rather than invented.

Features include Celsius temperature/dewpoint, wind direction and speed/gust in
knots, visibility in kilometers, pressure in hPa, reported snow/ice pellets,
freezing weather, rain, fog/mist, thunderstorms and hail, humidity/temperature
interactions, wind components, and observation age. The cold-and-humid interaction
is a modeling proxy, not an observed deicing procedure or a meteorological guarantee.
Reported-weather flags denote code presence; absence of a code is not proof of
no physical weather phenomenon. Missing numeric reports remain missing.

The IEM archive notes that precipitation quantities are unavailable for non-US
stations, so this feature set deliberately uses reported rain/snow weather codes
instead of treating unavailable precipitation quantities as zero. Snow depth and
ice accretion are also not assumed to be available. METAR visibility can be capped
by reporting conventions. IEM describes the archive as observations collected
largely as received, with limited quality control.
[Archive field definitions and limitations](https://mesonet.agron.iastate.edu/request/download.phtml).

NOAA's replacement GHCNh archive was also verified as a potential alternative:
its official metadata explicitly specifies CC0-1.0, but it was not downloaded for
this experiment because the IEM station-code query was smaller and more direct.
[NOAA GHCNh rights](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01688).
