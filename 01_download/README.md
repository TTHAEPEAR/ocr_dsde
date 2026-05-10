# 01_download

This folder is optional / legacy.

The final project dataset did not use this downloader. PDFs were collected or
provided manually and placed under `3/`, then converted with:

```powershell
python 02_preprocess\convert_pdfs.py
```

`download_pdfs.py` is kept for transparency and future reuse if the ECT website
is crawlable. Public website markup can change, so verify selectors and URLs
before using it:

```powershell
python 01_download\download_pdfs.py --run-discovery
```

Running without `--run-discovery` only checks local `data/raw_pdfs`
completeness and prints a warning.
