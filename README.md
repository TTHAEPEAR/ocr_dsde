# Thai Election OCR + Research Dashboard

โปรเจกต์นี้เป็น pipeline สำหรับแปลงไฟล์ PDF/รูปภาพรายงานผลการนับคะแนนเลือกตั้งไทยให้เป็นข้อมูลเชิงตาราง จากนั้น clean, validate, รวม batch ของหลายคน, วิเคราะห์เชิงสถิติ และเปิด dashboard แบบเล่าเรื่องเพื่อใช้พรีเซนต์งานวิจัย

ชุดงานล่าสุดออกแบบสำหรับข้อมูลเขต `กำแพงเพชร เขต 1` โดยรองรับ PDF ที่มีทั้งแบบแบ่งเขตและบัญชีรายชื่ออยู่ในไฟล์เดียวกัน ระบบจะสร้าง 2 records ต่อ 1 PDF คือ `constituency` และ `party_list`

## สิ่งสำคัญก่อนเริ่ม

- Workflow หลักไม่ใช้ `01_download` ใน final run แล้ว ไฟล์ PDF จริงให้วางเองในโฟลเดอร์ `3/`
- OCR default คือ `typhoon` ซึ่งทำ Stage A ด้วย Typhoon OCR และ Stage B extraction/checksum ด้วย Gemini
- ไฟล์ข้อมูลขนาดใหญ่ เช่น PDF, รูป, OCR output, cleaned output และ figures ถูก ignore จาก Git ต้องสร้างใหม่หรือส่งไฟล์แยกให้เพื่อน
- ห้าม commit `.env` เพราะมี API keys
- สำหรับ analysis ให้ใช้ไฟล์ split เป็นหลัก: `data/ocr_raw/raw_all_forms_split.csv`

## โครงสร้างโปรเจกต์

```text
01_download/       optional legacy downloader; ไม่ได้ใช้ใน final run
02_preprocess/     แปลง PDF ใน 3/ เป็น PNG
03_ocr/            OCR pipeline และตัว split/retry จาก markdown
04_clean/          clean, canonicalize, checksum, review queue
05_analysis/       รวม batch, วิเคราะห์, สร้างไฟล์ insight/figures
06_dashboard/      Streamlit dashboard สำหรับนำเสนอ
data/reference/    reference mapping และ manual corrections ที่ track ใน Git
outputs/figures/   ผลวิเคราะห์/กราฟที่สร้างตอนรัน analysis
config.py          path, engine, party mapping, constituency mapping
```

## ติดตั้ง

แนะนำ Python 3.11 ขึ้นไป

```powershell
cd D:\ocr\ocr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

ถ้าใช้ `streamlit` แล้ว PowerShell บอกว่าไม่รู้จักคำสั่ง ให้ใช้:

```powershell
python -m streamlit run 06_dashboard\app.py
```

## ตั้งค่า API keys

```powershell
Copy-Item .env.example .env
```

ใส่ key จริงใน `.env`

```env
GEMINI_API_KEY=...
TYPHOON_API_KEY=...
```

ถ้าใช้ `--engine easyocr` หรือ `tesseract` จะลด dependency API ได้ แต่คุณภาพ extraction และ checksum feedback จะด้อยกว่า pipeline `typhoon`

## ตั้งค่าเขตและ mapping

ค่าหลักอยู่ใน `config.py`

```python
PROVINCE = "กำแพงเพชร"
CONSTITUENCY_NUMBER = 1
OCR_ENGINE = "typhoon"
```

Mapping สำคัญ:

- `PARTY_NAMES`: เลขพรรคบัญชีรายชื่อ 1-57
- `CONSTITUENCY_CANDIDATE_PARTIES`: เลขผู้สมัครแบบแบ่งเขตของกำแพงเพชร เขต 1
- `data/reference/party_reference.csv`: reference ชื่อพรรคบัญชีรายชื่อ
- `data/reference/ocr_corrections.csv`: manual corrections ที่ยืนยันจากภาพจริง
- `data/reference/semantic_corrections.csv`: rules แก้ปัญหาเชิงความหมาย เช่นคะแนนเลื่อนแถว
- `data/reference/tambon_reference.csv`: พิกัดตำบลสำหรับ spatial dashboard

ข้อควรจำ: เลขแบบแบ่งเขตไม่ใช่เลขเดียวกับบัญชีรายชื่อ เช่น เลข `2` ในแบ่งเขตคือผู้สมัครพรรคกล้าธรรม แต่เลข `2` ในบัญชีรายชื่อคือพรรคเพื่อชาติไทย

## Quick Start สำหรับรันตั้งแต่ PDF

วาง PDF จริงไว้ใต้โฟลเดอร์ `3/` แล้วรันตามลำดับนี้

```powershell
python 02_preprocess\convert_pdfs.py
python 03_ocr\ocr_pipeline.py
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
python -m streamlit run 06_dashboard\app.py
```

ผลหลักที่ควรได้:

```text
data/images/election/
data/ocr_raw/raw_all_forms_split.csv
data/ocr_raw/raw_election_split.csv
data/ocr_raw/split_review_queue.csv
data/cleaned/election_results_cleaned.csv
data/cleaned/review_queue.csv
outputs/figures/
```

## OCR Workflow ละเอียด

### 1. แปลง PDF เป็นภาพ

```powershell
python 02_preprocess\convert_pdfs.py
```

สคริปต์จะอ่าน PDF จาก `3/` และเขียนรูปลง `data/images/election/`

### 2. OCR ใหม่เต็มชุด

```powershell
python 03_ocr\ocr_pipeline.py
```

หรือระบุ engine:

```powershell
python 03_ocr\ocr_pipeline.py --engine typhoon
python 03_ocr\ocr_pipeline.py --engine easyocr_gemini
python 03_ocr\ocr_pipeline.py --engine easyocr
```

รันแบบทดลอง sample:

```powershell
python 03_ocr\ocr_pipeline.py --sample
```

ถ้ารันค้างหรือ API ล่ม ให้รันคำสั่งเดิมซ้ำได้ ระบบมี checkpoint และจะข้าม record ที่สำเร็จแล้ว

### 3. Retry เฉพาะ record ที่ต้องตรวจ

ถ้ามี markdown OCR เดิมแล้ว และต้องการ retry เฉพาะ fail/review โดยไม่ OCR ภาพใหม่ทั้งหมด:

```powershell
python 03_ocr\split_dual_forms_from_markdown.py --retry-failed
```

ถ้าต้องเริ่ม split checkpoint ใหม่:

```powershell
python 03_ocr\split_dual_forms_from_markdown.py --reset-checkpoint
```

ถ้าต้องลองเฉพาะไฟล์เดียว:

```powershell
python 03_ocr\split_dual_forms_from_markdown.py --source-contains "หน่วยที่_13"
```

## รวมผล OCR จากหลายคน

วางไฟล์เพื่อนใน `data/ocr_raw/` ตามชื่อที่ script ใช้:

```text
data/ocr_raw/raw_all_forms_split.csv
data/ocr_raw/friend1.csv
data/ocr_raw/friend2.csv
data/ocr_raw/friend3.csv
```

รวม batch โดยยังไม่แทนไฟล์ active:

```powershell
python 05_analysis\combine_ocr_batches.py
```

ผลรวมจะอยู่ที่:

```text
data/ocr_raw/raw_all_forms_split_combined.csv
data/ocr_raw/combined_batch_summary.csv
data/ocr_raw/combined_missing_ballot_kind.csv
data/ocr_raw/combined_review_queue.csv
```

ถ้าตรวจแล้วพร้อมใช้ combined เป็นชุดหลัก ให้ activate:

```powershell
python 05_analysis\combine_ocr_batches.py --activate
```

คำสั่งนี้จะ backup ไฟล์ active เดิมก่อน แล้วแทนที่:

```text
data/ocr_raw/raw_all_forms_split.csv
data/ocr_raw/raw_election_split.csv
```

## Clean และ Validate

```powershell
python 04_clean\clean_data.py
python 04_clean\validate_data.py
```

Clean ทำหน้าที่หลัก:

- normalize party names
- ใช้ mapping แยก constituency/party-list
- apply manual/semantic corrections
- คำนวณ `votes_sum`, `vote_sum_match`, `ballot_sum_match`
- สร้างคอลัมน์สำหรับ analysis/dashboard

Validate ทำหน้าที่หลัก:

- ตรวจว่าทุก polling unit มีทั้ง `constituency` และ `party_list`
- ตรวจ checksum ของคะแนนและจำนวนบัตร
- สร้าง `data/cleaned/review_queue.csv`

Interpretation ของ flags:

- `vote_sum_match`: ผลรวมคะแนนผู้สมัคร/พรรคตรงกับ `good_ballots` หรือ `total_votes_sum`
- `ballot_sum_match`: `good_ballots + bad_ballots + no_vote_ballots == total_ballots`
- `needs_review`: record ที่ควรตรวจภาพจริงก่อนใช้เป็นหลักฐานแน่นอน
- `review_reason`: เหตุผลที่ควรตรวจ เช่น checksum fail, Klong Thai row shift, semantic warning

## Analysis

```powershell
python 05_analysis\analysis.py
```

ผลสำคัญอยู่ใน `outputs/figures/`

```text
data_quality_summary.csv
party_performance.csv
polling_unit_summary.csv
cross_ballot_consistency.csv
cross_ballot_winner_splits.csv
cross_ballot_winner_matrix.csv
unit_party_dominance.csv
stronghold_units.csv
unit_competitiveness.csv
evidence_fingerprint_map.csv
network_nodes.csv
network_edges.csv
review_records.csv
anomaly_records.csv
research_report.md
```

หลักการอ่านผล:

- ใช้ `confirmed` เป็นฐาน insight หลัก
- ใช้ `needs_review` เป็น uncertainty ไม่ใช่ข้อสรุปสุดท้าย
- เปรียบเทียบพรรคด้วย vote share ต่อหน่วย ไม่ใช้คะแนนดิบอย่างเดียว
- ใช้ `polling_unit_id` หรือ `ballot_record_id` เป็น key ห้ามใช้ `station_id` เดี่ยว ๆ เพราะเลขหน่วยซ้ำข้ามพื้นที่ได้

## Dashboard

เปิด dashboard:

```powershell
python -m streamlit run 06_dashboard\app.py --server.port 8502 --server.address 127.0.0.1
```

Dashboard ล่าสุดจัดเป็น 3 กลุ่ม:

- `Presentation Story`: flow เล่าเรื่องสำหรับนำเสนอ
- `Evidence`: evidence map, quality, overview, party performance, geo/spatial, network, anomalies
- `Drilldown`: ค้นหน่วย/ไฟล์เพื่อกลับไปตรวจภาพจริง

สีพรรคหลักใน dashboard:

- กล้าธรรม: เขียว
- ประชาชน: ส้ม
- ภูมิใจไทย: น้ำเงิน
- เพื่อไทย: แดง
- ประชาธิปัตย์: ฟ้า

## Insight สำคัญที่ dashboard รองรับ

1. แผนที่เชิงพื้นที่ระดับตำบล  
   ดูว่าพื้นที่ใดเอนเอียงไปทางพรรคใด และ cluster ทางภูมิศาสตร์สอดคล้องกับผลเลือกตั้งหรือไม่

2. Fingerprint map  
   ลดมิติ vote-share profile ของแต่ละหน่วย เพื่อดูว่าหน่วยไหนมี pattern คล้ายกันหรือเป็น outlier

3. Split-ticket / number-collision insight  
   เปรียบเทียบผลแบ่งเขตกับบัญชีรายชื่อ เช่น กล้าธรรมชนะเขต แต่บัญชีรายชื่อพรรคเพื่อชาติไทยเด่นผิดคาด เพราะเลขบัญชีรายชื่อ `2` ตรงกับเลขผู้สมัครเขตของกล้าธรรม

4. OCR evidence quality  
   แยก confirmed vs needs_review เพื่อไม่ให้ insight ถูกปนกับ uncertainty ของ OCR

## Manual Corrections

ถ้าตรวจภาพจริงแล้วพบ OCR ผิด ให้เพิ่ม correction ที่:

```text
data/reference/ocr_corrections.csv
```

ตัวอย่าง:

```csv
source_file,field,value,reason
image.pdf,candidate_27_votes,11,user_verified_partial_page_11_not_1_or_17
```

ถ้าเป็นปัญหาเชิง pattern เช่นคะแนนเลื่อนแถว ให้บันทึก/อธิบายใน:

```text
data/reference/semantic_corrections.csv
docs/semantic_corrections.md
```

หลังแก้ reference ให้รันใหม่:

```powershell
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
```

## `01_download` คืออะไร

`01_download` เป็น optional legacy downloader ที่เก็บไว้เพื่อความโปร่งใสของ source code เท่านั้น final run ไม่ได้ใช้ส่วนนี้

ถ้ารันปกติ:

```powershell
python 01_download\download_pdfs.py
```

สคริปต์จะไม่ crawl เว็บเอง จะเตือนและตรวจ local `data/raw_pdfs` เฉย ๆ

ถ้าจะใช้จริงในอนาคต ต้องตรวจ URL/selector ของเว็บ ECT ก่อน แล้วจึงรัน:

```powershell
python 01_download\download_pdfs.py --run-discovery
```

## ไฟล์ที่ควรส่งให้เพื่อนเพื่อเปิด dashboard

ถ้าเพื่อนไม่ต้อง OCR ใหม่ ให้ส่งอย่างน้อย:

```text
source code ทั้ง repo
data/ocr_raw/raw_all_forms_split.csv
data/cleaned/election_results_cleaned.csv
outputs/figures/
data/reference/
.env.example
```

ถ้าส่งผ่าน GitHub อย่างเดียว ข้อมูลที่ถูก ignore เช่น `data/ocr_raw`, `data/cleaned`, `outputs` จะไม่ติดไป ต้อง zip หรือส่งแยก

คำสั่งฝั่งเพื่อน:

```powershell
pip install -r requirements.txt
python -m streamlit run 06_dashboard\app.py
```

ถ้าต้อง regenerate analysis:

```powershell
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
python -m streamlit run 06_dashboard\app.py
```

## ส่ง source code ให้อาจารย์

ควรส่ง:

- source code ทั้ง repo
- README.md ฉบับนี้
- `data/reference/` เพราะเป็น mapping/corrections ที่จำเป็น
- ถ้าต้องให้อาจารย์เปิด dashboard ได้ทันที ให้แนบ `data/ocr_raw`, `data/cleaned`, `outputs/figures` แยกไปด้วย

ไม่ควรส่ง:

- `.env`
- API keys
- `.venv`
- ไฟล์ temporary/log ที่ไม่เกี่ยวกับงาน

## Troubleshooting

### `streamlit` is not recognized

ใช้:

```powershell
python -m streamlit run 06_dashboard\app.py
```

### Typhoon/Gemini API key หาย

เช็ก `.env`:

```env
GEMINI_API_KEY=...
TYPHOON_API_KEY=...
```

### Gemini busy / 500 INTERNAL / connection closed

เป็นปัญหาฝั่ง API หรือ network เป็นช่วง ๆ รันคำสั่งเดิมต่อได้ เพราะมี checkpoint

### อยากลด quota Gemini

ใช้ `--engine easyocr_gemini` เพื่อลดการเรียก Typhoon แต่ยังใช้ Gemini Stage B หรือใช้ `--engine easyocr` แบบ offline แต่คุณภาพ extraction จะลดลง

### Checksum ผ่านแต่ยังผิดได้ไหม

ได้ ถ้า OCR เลื่อนแถวแล้วผลรวมยังพอดี จึงต้องใช้ `review_queue.csv`, semantic corrections และการตรวจภาพจริงประกอบ

## Current Recommended Command Order

สำหรับรัน full pipeline จาก PDF:

```powershell
python 02_preprocess\convert_pdfs.py
python 03_ocr\ocr_pipeline.py
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
python -m streamlit run 06_dashboard\app.py --server.port 8502 --server.address 127.0.0.1
```

สำหรับใช้ข้อมูลรวมจากหลายคนที่มี OCR output แล้ว:

```powershell
python 05_analysis\combine_ocr_batches.py --activate
python 04_clean\clean_data.py
python 04_clean\validate_data.py
python 05_analysis\analysis.py
python -m streamlit run 06_dashboard\app.py --server.port 8502 --server.address 127.0.0.1
```
