import requests
import os

# CONFIG
user = "liuxuannan"
repo = "FAK-Owl"
commit = "1f6c13c7d0e4f77411edf5261623fdb8c5a0f7cc"
directory = "data/metadata_split"
local_dir = "metadata_split"  # Where to save

# GitHub API endpoint
api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{directory}?ref={commit}"

def download_file(url, dest):
    r = requests.get(url)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def download_dir(api_url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    r = requests.get(api_url)
    r.raise_for_status()
    for item in r.json():
        if item['type'] == 'file':
            print(f"Downloading {item['name']}")
            download_file(item['download_url'], os.path.join(local_dir, item['name']))
        elif item['type'] == 'dir':
            download_dir(item['url'], os.path.join(local_dir, item['name']))

if __name__ == "__main__":
    download_dir(api_url, local_dir)
    print("Done!")
