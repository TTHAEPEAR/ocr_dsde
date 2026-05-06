"""Quick debug script: replay Stage B with the saved markdown to see what Gemini returns."""
from pathlib import Path
from config import GEMINI_API_KEY, OCR_RAW_DIR

# Load the markdown
md_path = OCR_RAW_DIR / "markdown" / "sample" / "อำเภอลอง_20260426T052535Z_3_001_อำเภอลอง_ต_ปากกาง_หน_วยเล_อกต_งท_9_ส_ส_5_18_บช.md"
raw_md = md_path.read_text(encoding="utf-8")

# Convert Thai digits to Arabic (same as pipeline does)
TABLE = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
converted_md = raw_md.translate(TABLE)

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=== CONVERTED MARKDOWN (first 500 chars) ===")
print(converted_md[:500])
print(f"\n... ({len(converted_md)} total chars)")
print("\n" + "="*60)

# Now send to Gemini
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)

form_type = "election"  # Use "election" instead of "sample"!

prompt = f"""You receive the OCR transcription of a Thai election form (Form {form_type}),
spanning 1 page(s) of the SAME polling unit. Merge them into a single record.

Return ONLY a JSON object — no markdown fences, no commentary.
Schema:
{{
  "station_id": int,
  "constituency_number": int,
  "province": str,
  "good_ballots": int,
  "bad_ballots": int,
  "no_vote_ballots": int,
  "total_ballots": int,
  "votes": {{ "candidate_<N>_votes": int, ... }},
  "parties": {{ "candidate_<N>_party": "<party name>", ... }}
}}
Rules:
- Extract vote counts from each row of the candidate table.
- The table columns are: (1) party NUMBER, (2) party NAME, (3) vote count DIGITS, (4) vote count in THAI WORDS.
- Extract ONLY the vote count from column 3 (digits). Ignore column 4.
- Do NOT mix the party number (column 1) with the vote count (column 3).
- Use 0 for missing/unreadable values.

OCR transcription:
---
{converted_md}
---
"""

import time
for attempt in range(3):
    try:
        resp = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
        )
        print("\n=== GEMINI RAW RESPONSE ===")
        print(resp.text)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(30)
