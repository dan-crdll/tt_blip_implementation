import os 
import zipfile
from tqdm.auto import tqdm
import threading


def unzip_dgm4():
    t = []
    for dirn in ['manipulation', 'origin']:
        t.append(threading.Thread(target=unzip_dirn, args=[dirn]))
        t[-1].start()
    for s in t:
        s.join()

    print("Files successfully unzipped")


def unzip_dirn(dirn):
    files = os.listdir(f'./DGM4/{dirn}')
    t = []
    for f in tqdm(files):
        t.append(threading.Thread(target=unzip_file, args=[f, dirn]))
        t[-1].start()
    for s in t:
        s.join()


def unzip_file(f, dirn):
    with zipfile.ZipFile(f'./DGM4/{dirn}/{f}', 'r') as zf:
        zf.extractall(f'./DGM4/{dirn}')


if __name__=="__main__":
    unzip_dgm4()