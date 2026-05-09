# Semantic OCR Corrections

Some OCR errors pass arithmetic checks. A row-shift error can keep
`votes_sum == good_ballots` while assigning votes to the wrong party row.

## Reference-Driven Rules

Semantic corrections live in:

```text
data/reference/semantic_corrections.csv
```

The cleaner reads this file on every run. Rules are not hidden in raw data and
the original OCR CSV is never modified.

Current rule:

```text
klong_thai_to_democrat_block_26_34
```

This rule applies only to party-list rows where candidate 26, Klong Thai, has
more than 3 votes. Image review confirmed this is an off-by-one row shift inside
the party-list block from candidate 26 through candidate 34:

- candidate 26 is reset to 0
- candidate 27 receives the old candidate 26 value
- candidate 28 receives the old candidate 27 value
- this continues through candidate 34
- candidate 35 and later are not changed

## Audit Output

Every applied semantic correction is written to:

```text
data/cleaned/semantic_correction_audit.csv
```

The audit table includes source identifiers, the rule name, before/after values
for the affected candidate range, and an explicit check that the first
unaffected candidate after the block stayed unchanged.

Use this file when writing the report or when checking corrections against the
original images.

## Safe Workflow

After editing `semantic_corrections.csv`, rerun:

```powershell
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
python 05_analysis\build_location_reference.py
```

Then inspect:

```text
data/cleaned/semantic_correction_audit.csv
data/cleaned/review_queue.csv
outputs/figures/party_performance.csv
```
