import requests 
import zipfile
import shutil
import os 
import concurrent.futures


os.makedirs('./hf_cache', exist_ok=True)
path = "hf_cache/data.zip"

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/DGM4', exist_ok=True)
os.makedirs('./data/DGM4/manipulation', exist_ok=True)
os.makedirs('./data/DGM4/origin', exist_ok=True)

origins = ['washington_post', 'bbc', 'guardian', 'usa_today']   # 'bbc', 'guardian', 'usa_today', 
manipulations = ['simswap', 'StyleCLIP', 'HFGI', 'infoswap']   # 'HFGI', 'StyleCLIP', 'infoswap', 

def download_and_extract(url, extract_path):
    response = requests.get(url, stream=True)
    with open(path, 'wb') as fp:
        for chunk in response.iter_content(chunk_size=1024):
            fp.write(chunk)
    with zipfile.ZipFile(path, "r") as zp:
        zp.extractall(extract_path)
    print(f"Downloaded and extracted {url}")

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for origin in origins:
        url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/origin/{origin}.zip'
        futures.append(executor.submit(download_and_extract, url, './data/DGM4/origin'))

    for manipulation in manipulations:
        url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/manipulation/{manipulation}.zip'
        futures.append(executor.submit(download_and_extract, url, './data/DGM4/manipulation'))

    for future in concurrent.futures.as_completed(futures):
        future.result()
