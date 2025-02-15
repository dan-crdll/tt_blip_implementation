from json import JSONDecoder
import json
import os 
import requests
from tqdm.auto import tqdm
import threading
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

"""
Utility script to download images and texts from the Gossipcop dataset (downloaded separately)
"""

# Initialize JSON decoder
decoder = JSONDecoder()

def process_element(path, d, idx, split):
    """
    Process a single element in the dataset.
    
    Args:
        path (str): Path to the dataset split.
        d (str): Directory name of the element.
        idx (int): Index of the element.
        split (str): Split type ('fake' or 'real').
    """
    try:
        # Get the first file in the directory
        el = os.listdir(f'{path}/{d}')[0]

        # Open and decode the JSON file
        with open(f"{path}/{d}/{el}", 'r') as f:
            j = decoder.decode(f.read())

            # Save text in a JSON file
            os.makedirs(f"./data/gossipcop/{split}/text", exist_ok=True)
            with open(f"./data/gossipcop/{split}/text/{idx}.json", "w") as f_out:
                f_out.write(json.dumps({'text': j['text']}))

            # Save images
            os.makedirs(f"./data/gossipcop/{split}/img", exist_ok=True)
            img_url = j['top_img']

            if img_url:
                try:
                    # Download and decode the image with a timeout
                    img_data = requests.get(img_url, timeout=2).content
                    img_array = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Save the image
                        cv2.imwrite(f"./data/gossipcop/{split}/img/{idx}.jpg", img)
                except Exception:
                    pass
    except Exception as e:
        pass

def process_split(split):
    """
    Process all elements in a dataset split.
    
    Args:
        split (str): Split type ('fake' or 'real').
    """
    path = f"../gossipcop/{split}"
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(process_element, path, d, idx, split): d for idx, d in enumerate(os.listdir(path))}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                # Wait for the task to complete with a timeout
                future.result(timeout=2)
            except TimeoutError:
                continue

def check_mismatches(split):
    """
    Check for mismatches between text and image files and rename remaining files in ascending order.
    
    Args:
        split (str): Split type ('fake' or 'real').
    """
    # Check for mismatches and delete unmatched files
    text_dir = f"./data/gossipcop/{split}/text"
    img_dir = f"./data/gossipcop/{split}/img"
    text_files = set(int(f.split('.')[0]) for f in os.listdir(text_dir))
    img_files = set(int(f.split('.')[0]) for f in os.listdir(img_dir))

    unmatched_texts = text_files - img_files
    unmatched_imgs = img_files - text_files

    for idx in unmatched_texts:
        os.remove(f"{text_dir}/{idx}.json")
    for idx in unmatched_imgs:
        os.remove(f"{img_dir}/{idx}.jpg")

    # Rename remaining files in ascending order
    text_files = sorted(int(f.split('.')[0]) for f in os.listdir(text_dir))
    img_files = sorted(int(f.split('.')[0]) for f in os.listdir(img_dir))

    for new_idx, old_idx in enumerate(text_files):
        os.rename(f"{text_dir}/{old_idx}.json", f"{text_dir}/{new_idx}.json")
    for new_idx, old_idx in enumerate(img_files):
        os.rename(f"{img_dir}/{old_idx}.jpg", f"{img_dir}/{new_idx}.jpg")

# Create and start threads for processing 'fake' and 'real' splits
threads = []
for split in ['fake', 'real']:
    thread = threading.Thread(target=process_split, args=(split,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Check for mismatches after all downloads are complete
for split in ['fake', 'real']:
    check_mismatches(split)
