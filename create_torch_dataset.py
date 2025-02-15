import cv2 as cv
import os
import torch
import json

PATH = './data/gossipcop'
def create_dataset():
    splits = [f'{PATH}/fake/', f'{PATH}/real/']

    for l, s in enumerate(splits):
        x_img = []
        x_txt = []
        y = []

        for img_path in os.listdir(f"{s}/img"):
            img = cv.imread(f"{PATH}{img_path}", cv.IMREAD_ANYCOLOR)
            img = cv.resize(img, (256, 256))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
            x_img.append(img)
            y.append(l)
        
        for txt_path in os.listdir("{s}/text"):
            with open(f"{PATH}{txt_path}", "r") as f:
                j = json.load(f)
                x_txt.append(j['text'])

        x_img = torch.vstack(x_img)
        y = torch.tensor(y, dtype=torch.float32)

        torch.save(x_img, f'./dataset/gossipcop/{l}_imgs.pth')
        torch.save(x_txt, f'./dataset/gossipcop/{l}_txts.pth')
        torch.save(y, f'./dataset/gossipcop/{l}_labels.pth')
        

if __name__=="__main__":
    create_dataset()