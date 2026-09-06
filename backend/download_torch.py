import urllib.request
import sys
import os
import time

url = "https://download-r2.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl"
file_name = "torch-2.2.2+cu121-cp311-cp311-win_amd64.whl"

def get_file_size(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers['Content-Length'])

total_size = get_file_size(url)
print(f"Total size: {total_size / (1024*1024):.2f} MB")

downloaded = 0
if os.path.exists(file_name):
    downloaded = os.path.getsize(file_name)

last_percent = -1

while downloaded < total_size:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Range': f'bytes={downloaded}-'
        })
        print(f"Resuming download from {downloaded / (1024*1024):.2f} MB...")
        with urllib.request.urlopen(req) as resp, open(file_name, 'ab') as f:
            while True:
                chunk = resp.read(8192 * 4)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                percent = int(downloaded * 100 / total_size)
                if percent != last_percent and percent % 5 == 0:
                    print(f"Downloading PyTorch CUDA: {percent}% complete ({downloaded / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB)")
                    sys.stdout.flush()
                    last_percent = percent
                    
    except Exception as e:
        print(f"Connection dropped: {e}. Retrying in 3 seconds...")
        time.sleep(3)

print("Download finished completely!")
