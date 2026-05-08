# Thai Election OCR Pipeline

โปรเจกต์นี้เป็น pipeline สำหรับแปลงไฟล์ PDF/รูปภาพแบบรายงานผลการนับคะแนนเลือกตั้งไทย ให้กลายเป็น CSV ที่นำไป clean, validate, วิเคราะห์ และทำ dashboard ต่อได้

โค้ดถูกจัดเป็นขั้นตอนตั้งแต่ดาวน์โหลด/จัดไฟล์ PDF, แปลงเป็นรูป, OCR ด้วย Typhoon + Gemini, ทำความสะอาดข้อมูล, ตรวจความถูกต้อง และวิเคราะห์ผล

## ภาพรวมโฟลเดอร์

```text
01_download/       ดาวน์โหลดไฟล์ PDF
02_preprocess/     แปลง PDF เป็น PNG และเตรียมภาพ
03_ocr/            OCR + ดึง field เป็น structured CSV
04_clean/          clean และ validate ข้อมูล
05_analysis/       วิเคราะห์และสร้างกราฟ
06_dashboard/      Streamlit dashboard
data/reference/    ไฟล์อ้างอิง/กฎแก้ OCR แบบ manual
config.py          config หลักของโปรเจกต์
```

โฟลเดอร์ข้อมูลขนาดใหญ่ เช่น `data/images`, `data/raw_pdfs`, `data/ocr_raw`, `data/cleaned`, `outputs` ถูก ignore จาก Git เพื่อไม่ให้ repo หนักเกินไป เพื่อนที่ใช้งานต้องสร้างข้อมูลเองด้วยคำสั่งด้านล่าง

## สิ่งที่ต้องติดตั้ง

ใช้ Python 3.11 ขึ้นไป แนะนำให้ใช้ virtual environment

```powershell
cd D:\ocr\ocr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

ถ้าใช้ EasyOCR บนเครื่องไม่มี GPU อาจช้าหน่อย แต่ default ของโปรเจกต์นี้ใช้ `typhoon` เป็น OCR engine หลัก

## ตั้งค่า API keys

คัดลอกไฟล์ตัวอย่าง:

```powershell
Copy-Item .env.example .env
```

เปิด `.env` แล้วใส่ key จริง:

```env
GEMINI_API_KEY=...
TYPHOON_API_KEY=...
```

ห้าม commit `.env` ขึ้น GitHub เพราะมี secret key จริง

## ตั้งค่าเขต/จังหวัด

แก้ใน `config.py`

```python
PROVINCE = "กำแพงเพชร"
CONSTITUENCY_NUMBER = 1
OCR_ENGINE = "typhoon"
```

ตอนนี้ตั้ง default เป็น `typhoon` แล้ว ถ้ารัน `python 03_ocr/ocr_pipeline.py` โดยไม่ใส่ `--engine` ระบบจะใช้ Typhoon OCR + Gemini extraction อัตโนมัติ

ถ้าต้องการบังคับให้ clean เฉพาะจังหวัด/เขตใดเขตหนึ่ง ค่อยตั้ง `PROVINCE` และ `CONSTITUENCY_NUMBER` เป็นค่าจริง แต่ถ้าชุดข้อมูลมีหลายพื้นที่หรือยังไม่แน่ใจ ให้ปล่อยเป็น `None` เพื่อให้ OCR อ่านจากภาพเอง

## Workflow ใช้งานหลัก

### 1. เตรียม PDF

โปรเจกต์เดิมใช้โฟลเดอร์ `D:\ocr\ocr\3` เป็นแหล่ง PDF หลัก ถ้ามี PDF อยู่แล้ว ให้วางตามโครงสร้างเดิมได้เลย

ถ้าจะย้าย PDF เข้า `data/raw_pdfs` ให้ดูสคริปต์:

```powershell
python move_pdfs.py
```

### 2. แปลง PDF เป็นรูป

ถ้า PDF อยู่ในโฟลเดอร์ `3/`:

```powershell
python 02_preprocess\convert_pdfs.py
```

ผลลัพธ์จะไปที่:

```text
data/images/election/
```

สคริปต์นี้ข้ามหน้าสุดท้ายของ PDF ถ้าเป็นหน้าลายเซ็น/ไม่มีคะแนน

### 3. รัน OCR

รันทั้งชุด:

```powershell
python 03_ocr\ocr_pipeline.py
```

ผล OCR จะสร้าง `polling_unit_id` จากชื่อไฟล์/โฟลเดอร์ที่ convert มา เพื่อใช้เป็น key หน่วยเลือกตั้งแบบไม่ซ้ำข้ามตำบล/เทศบาล และจะให้โมเดลระบุ `ballot_kind` เป็น `constituency` หรือ `party_list` จากภาพด้วย ถ้าชื่อไฟล์ไม่มี `5_18` หรือ `บช` ระบบจะใช้ค่านี้ช่วยตั้ง `form_type`

หรือระบุ engine เอง:

```powershell
python 03_ocr\ocr_pipeline.py --engine typhoon
python 03_ocr\ocr_pipeline.py --engine easyocr_gemini
```

ถ้าต้องการเลือกแบบ interactive:

```powershell
python 03_ocr\ocr_pipeline.py --interactive
```

ผลลัพธ์จะอยู่ที่:

```text
data/ocr_raw/raw_all_forms.csv
data/ocr_raw/raw_election_checkpoint.csv
data/ocr_raw/markdown/
```

สำหรับ PDF ชุดเลือกตั้งจริงที่มีทั้ง 2 แบบฟอร์มในไฟล์เดียวกัน pipeline รุ่นล่าสุดจะเขียนผลหลักแบบ form-level แยกเป็น 2 records ต่อ PDF:

```text
data/ocr_raw/raw_all_forms_split.csv
data/ocr_raw/raw_election_split.csv
data/ocr_raw/raw_election_split_checkpoint.csv
```

โครงสร้างนี้ใช้ `ballot_record_id` เป็น key เช่น `...__constituency` และ `...__party_list` โดยแยกหน้า `1-2` เป็นแบบแบ่งเขต และหน้า `3-5` เป็นแบบบัญชีรายชื่อ ไฟล์ `raw_all_forms.csv` เดิมถือเป็น legacy output และไม่ควรใช้เป็น source หลักสำหรับ analysis ถ้า `raw_all_forms_split.csv` มีอยู่

ถ้ามี markdown จาก OCR รอบเก่าแล้ว และไม่อยาก OCR ใหม่ สามารถสร้างชุด split จาก markdown เดิมได้ด้วย:

```powershell
python 03_ocr\split_dual_forms_from_markdown.py
```

ระบบมี checkpoint ถ้ารันค้างหรือ API ล่ม สามารถรันซ้ำได้ โดยจะข้าม record ที่สำเร็จแล้ว และจะรันใหม่สำหรับ record ที่ `needs_review=True`

### 4. ทดลองกับ sample

ถ้ามีรูปใน:

```text
data/images/sample/
```

ให้รัน:

```powershell
python 03_ocr\ocr_pipeline.py --sample
```

ผลลัพธ์:

```text
data/ocr_raw/raw_sample.csv
data/ocr_raw/raw_sample_all.csv
```

ใน sample ล่าสุด โค้ดสามารถแยกได้ว่า:

- รูปเต็มฟอร์มที่มีหัวหน่วย + summary ผ่าน validation
- รูป partial page เช่นมีเฉพาะหน้ากลางของบัญชีรายชื่อ จะถูกตั้ง `partial_page=True` และ `needs_review=True`

### 5. Clean ข้อมูล

```powershell
python 04_clean\clean_data.py
```

ผลลัพธ์:

```text
data/cleaned/election_results_cleaned.csv
```

ขั้นนี้จะ:

- ลบ record OCR ที่ fail
- รวม page-level rows เป็น document-level rows
- แก้ numeric OCR เบื้องต้น
- normalize party names โดยเทียบกับ `data/reference/party_reference.csv`
- zero out vote ที่เกินจำนวนบัตร
- สร้าง derived columns เช่น turnout ratio

### 6. Validate

```powershell
python 04_clean\validate_data.py
```

ผลลัพธ์:

```text
data/cleaned/validation_report.txt
data/cleaned/review_queue.csv
```

`review_queue.csv` คือไฟล์สำคัญมาก เอาไว้ดูว่า row ไหนควร OCR ใหม่หรือตรวจมือ เพราะมีปัญหาเช่น:

- `ballot_sum_mismatch`
- `vote_sum_exceeds_good_ballots`
- `missing_station_id`

### 7. วิเคราะห์

```powershell
python 05_analysis\analysis.py
```

ผลกราฟ/CSV จะไปที่:

```text
outputs/figures/
```

### 8. เปิด dashboard

```powershell
streamlit run 06_dashboard\app.py
```

หรือ dashboard อีกเวอร์ชัน:

```powershell
streamlit run 05_analysis\election_dashboard.py
```

## รายชื่อพรรคอ้างอิง

ระบบมีไฟล์เลขพรรค-ชื่อพรรคจริงอยู่ที่:

```text
data/reference/party_reference.csv
```

ไฟล์นี้มาจาก PDF อ้างอิงรายชื่อพรรคที่ถูกต้อง และมีหมายเลข 1-57 ครบแล้ว ขั้น `04_clean\clean_data.py` จะอ่านไฟล์นี้ก่อนเพื่อ fuzzy match ชื่อพรรคที่ OCR อ่านได้ให้กลับเป็นชื่อจริง และสร้างคอลัมน์ `party_<ชื่อพรรค>_votes` สำหรับฟอร์มบัญชีรายชื่อ

ถ้าอนาคตมีชื่อพรรคหรือหมายเลขเปลี่ยน ให้แก้ CSV นี้เป็นหลัก รูปแบบคือ:

```csv
party_number,party_name
1,ไทยทรัพย์ทวี
2,เพื่อชาติไทย
```

หลังแก้ reference แล้วให้รัน clean/validate ใหม่:

```powershell
python 04_clean\clean_data.py
python 04_clean\validate_data.py
```

## การแก้ OCR แบบ manual correction

บางเคส OCR อ่านผิดแต่ผลรวมยังตรง เช่นอ่านแถวผิดแล้วไปชดเชยอีกแถว ระบบ sum-check จะจับไม่ได้ 100%

ให้เพิ่ม correction ที่:

```text
data/reference/ocr_corrections.csv
```

รูปแบบ:

```csv
source_file,field,value,reason
image.pdf,candidate_27_votes,11,user_verified_partial_page_11_not_1_or_17
```

เมื่อ OCR pipeline เจอ `source_file` และ `field` ตรงกัน จะ override ค่านั้น และ refresh `votes_sum`, `needs_review`, `partial_page` ให้อัตโนมัติ

## คอลัมน์สำคัญใน raw OCR

- `source_file`: ชื่อไฟล์/กลุ่ม PDF
- `station_id`: หมายเลขหน่วยเลือกตั้ง
- `good_ballots`: บัตรดี
- `bad_ballots`: บัตรเสีย
- `no_vote_ballots`: บัตรไม่เลือกผู้สมัคร
- `total_ballots`: บัตรที่ใช้ทั้งหมด
- `candidate_<N>_votes`: คะแนนผู้สมัคร/พรรคหมายเลข N
- `votes_sum`: ผลรวม candidate votes
- `vote_sum_match`: votes_sum ตรงกับ good_ballots หรือไม่
- `ballot_sum_match`: good + bad + no_vote ตรงกับ total หรือไม่
- `partial_page`: เป็นหน้าไม่สมบูรณ์ ไม่มีหัว/summary หรือไม่
- `needs_review`: ควรตรวจซ้ำหรือไม่

## คำแนะนำเวลาทำต่อ

1. เริ่มจากรัน `--sample` ทุกครั้งก่อนเปลี่ยน prompt หรือ logic OCR
2. เช็ก `raw_sample.csv` ว่าแถวที่รู้คำตอบจริงออกถูกไหม
3. ถ้าผลรวมถูกแต่รายแถวผิด ให้เพิ่ม prompt/crop หรือ manual correction
4. อย่าเชื่อ `ocr_confidence` อย่างเดียว ให้ดู `vote_sum_match`, `ballot_sum_match`, `partial_page`, `needs_review`
5. สำหรับข้อมูลชุดใหญ่ ให้ดู `review_queue.csv` เป็นรายการทำงานต่อ

## ปัญหาที่พบบ่อย

### API key หาย

ถ้าเจอ error ว่าไม่มี `GEMINI_API_KEY` หรือ `TYPHOON_API_KEY` ให้เช็ก `.env`

### ไฟล์ sample ไม่ถูกรันใหม่

OCR มี checkpoint ถ้าอยากบังคับรันใหม่ ให้ลบหรือ backup:

```powershell
Remove-Item data\ocr_raw\raw_sample*.csv
```

### รูปเป็น partial page

ถ้ารูปมีเฉพาะหน้ากลางของฟอร์ม ไม่มีหัวหน่วยหรือ summary ท้ายฟอร์ม ระบบจะตั้ง:

```text
partial_page=True
needs_review=True
```

ให้รวมกับหน้าอื่นของ PDF เดียวกัน หรือใช้ manual correction เฉพาะเมื่อรู้คำตอบแน่นอน

## หมายเหตุเรื่อง GitHub

Repo นี้ไม่เก็บ:

- API keys
- raw PDFs
- generated images
- OCR outputs
- cleaned outputs
- virtual environment

ทุกคนควรรัน pipeline บนเครื่องตัวเองเพื่อสร้างข้อมูลเหล่านี้ใหม่
