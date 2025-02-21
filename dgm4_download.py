import requests 
import zipfile
import shutil
import os 

os.makedirs('./hf_cache', exist_ok=True)
path = "hf_cache/data.zip"

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/DGM4', exist_ok=True)
os.makedirs('./data/DGM4/manipulation', exist_ok=True)
os.makedirs('./data/DGM4/origin', exist_ok=True)

origins = ['washington_post', 'bbc']   # 'bbc', 'guardian', 'usa_today', 
manipulations = ['simswap', 'StyleCLIP']   # 'HFGI', 'StyleCLIP', 'infoswap', 

for origin in origins:
    url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/origin/{origin}.zip'
    response = requests.get(url, stream=True)
    with open(path, 'wb') as fp:
        for chunk in response.iter_content(chunk_size=1024):
            fp.write(chunk)

    with zipfile.ZipFile(path, "r") as zp:
        zp.extractall(f'./data/DGM4/origin')

for manipulation in manipulations:
    url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/manipulation/{manipulation}.zip'
    response = requests.get(url, stream=True)
    with open(path, 'wb') as fp:
        for chunk in response.iter_content(chunk_size=1024):
            fp.write(chunk)

    with zipfile.ZipFile(path, "r") as zp:
        zp.extractall(f'./data/DGM4/manipulation')
