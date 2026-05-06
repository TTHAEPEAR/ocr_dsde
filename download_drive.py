import gdown
import sys

# Fix encoding issues for printing Thai characters on Windows terminal
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

url = 'https://drive.google.com/drive/folders/1T_2uYkDRrBzh9mckLgkR9Fh0vrFWaVst'
print("Start downloading from Google Drive...")
# Download to a temporary staging folder
gdown.download_folder(url, output="data/drive_download", quiet=False)
print("Finished downloading!")
