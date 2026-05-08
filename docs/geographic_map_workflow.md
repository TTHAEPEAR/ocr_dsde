# Geographic Map QA Workflow

Use this workflow when building a map such as "this tambon / moo voted for which party".

## Why this extra step exists

Do not treat OCR geography as official truth. Some headers are read incorrectly, for example province, district, moo, or municipality names. The map should therefore use a separate location reference table and only draw a real map from rows that have been verified.

## Build the location reference

Run:

```powershell
python 05_analysis\analysis.py
python 05_analysis\build_location_reference.py
```

Outputs:

```text
data/reference/polling_unit_locations_draft.csv
data/reference/polling_unit_locations.csv
data/reference/tambon_reference.csv
outputs/figures/geographic_winner_summary.csv
outputs/figures/tambon_party_summary.csv
```

`polling_unit_locations_draft.csv` is generated from source paths plus OCR markdown headers. Use it for QA only.

`polling_unit_locations.csv` is the editable reference table. The script updates draft-derived columns but preserves manual official columns when it is rerun.

## Columns to verify

Fill these fields in `data/reference/polling_unit_locations.csv` from real documents or official reference data:

```text
official_province
official_district
official_subdistrict
official_municipality
official_moo
lat
lon
tambon_code
amphoe_code
province_code
location_verified
verification_source
verification_note
```

Set `location_verified=True` only when the row has been checked against real evidence. If `lat` and `lon` are not known yet, leave them blank; the dashboard will not draw that row on the real map.

## Dashboard

Open:

```powershell
streamlit run 06_dashboard\app.py
```

Use the `Geo QA / Map` tab:

- `Location QA table` shows rows that still need verification.
- `Draft geographic cluster by tambon/moo` is a review aid, not final geographic evidence.
- `Real map from verified coordinates` appears only for rows where `location_verified=True` and `lat/lon` are filled.

This keeps the visual map rigorous: no unverified OCR place name becomes a final geographic claim by accident.

## Tambon-only mode

For a cleaner report, start at tambon level instead of moo level. The editable mapping lives in:

```text
data/reference/tambon_reference.csv
```

Current columns:

```text
path_local_government
official_province
official_district
official_subdistrict
tambon_verified
tambon_lat
tambon_lon
verification_source
verification_note
```

The script aggregates every party's votes by `official_subdistrict`, then ranks parties within each tambon and ballot kind. This is stricter than counting polling-unit winners because it uses total party votes in the tambon.

If `tambon_lat` and `tambon_lon` are filled, the dashboard can draw a tambon-level point map. These coordinates are treated as approximate centroids/reference points unless the verification note states that they come from official GIS boundaries. Do not describe them as exact polling-place coordinates or official tambon boundaries.
