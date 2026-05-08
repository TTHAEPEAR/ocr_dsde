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
outputs/figures/geographic_winner_summary.csv
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
