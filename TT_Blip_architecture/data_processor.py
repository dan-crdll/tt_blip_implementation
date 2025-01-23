import torch
from torch import nn
import cv2 as cv
from transformers import BertTokenizerFast, AutoImageProcessor, AutoTokenizer

class DataProcessor():
    def __init__(self):
        super().__init__()
        
        self.bert_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.blip_txt_processor = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_img_processor = AutoImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def process_func(self, batch):
        image = []
        text = []
        label = []

        for b in batch:
            img = cv.imread(b['image'], cv.IMREAD_ANYCOLOR)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            image.append(img)
            text.append(b['text'])
            label.append(1 if b['fake_cls'] == 'orig' else 0)

        vit_img = self.vit_processor(image, return_tensors='pt').pixel_values
        blip_img = self.blip_img_processor(image, return_tensors='pt').pixel_values
        blip_txt = self.blip_txt_processor(text, return_tensors='pt', padding=True)
        bert_txt = self.bert_tokenizer(text, return_tensors='pt', padding=True)

        blip_tokens = blip_txt.input_ids
        blip_attn = blip_txt.attention_mask

        bert_tokens = bert_txt.input_ids
        bert_attn = bert_txt.attention_mask


        label = torch.tensor(label).unsqueeze(-1).float()
        return vit_img, blip_img, (blip_tokens, blip_attn), (bert_tokens, bert_attn), label

