import requests 
import zipfile
import os 
import concurrent.futures


def download_dgm4(
        origins = ['washington_post', 'bbc', 'guardian', 'usa_today'], 
        manipulations = ['simswap', 'StyleCLIP', 'HFGI', 'infoswap'] 
    ):
    os.makedirs('./hf_cache', exist_ok=True)
    path = "hf_cache/"

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./data/DGM4', exist_ok=True)
    os.makedirs('./data/DGM4/manipulation', exist_ok=True)
    os.makedirs('./data/DGM4/origin', exist_ok=True)

    def download_and_extract(url, extract_path, name):
        response = requests.get(url, stream=True)
        with open(f"{path}/{name}.zip", 'wb') as fp:
            for chunk in response.iter_content(chunk_size=1024):
                fp.write(chunk)
        with zipfile.ZipFile(f"{path}/{name}.zip", "r") as zp:
            zp.extractall(extract_path)
        print(f"Downloaded and extracted {url}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for origin in origins:
            url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/origin/{origin}.zip'
            futures.append(executor.submit(download_and_extract, url, './data/DGM4/origin', origin))

        for manipulation in manipulations:
            url = f'https://huggingface.co/datasets/rshaojimmy/DGM4/resolve/main/manipulation/{manipulation}.zip'
            futures.append(executor.submit(download_and_extract, url, './data/DGM4/manipulation', manipulation))

        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__=="__main__":
    download_dgm4()