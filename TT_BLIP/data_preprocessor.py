import torch 
from torch import nn 
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import json


class DataPreprocessor():
    def __init__(self):
        # BLIP preprocessing utils
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

        self.empty_pixel_values = self.blip_processor(torch.zeros((3, 256, 256)), return_tensors='pt')['pixel_values']
        empty_txt = self.blip_tokenizer(["no text"], return_tensors='pt')
        self.empty_input_ids = empty_txt['input_ids']
        self.empty_attn_mask = empty_txt['attention_mask']

        # ViT preprocessing utils
        self.vit_processor = AutoProcessor.from_pretrained("WinKawaks/vit-small-patch16-224", use_fast = True)

        # BERT preprocessing utils
        self.bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


    def __call__(self, images, texts):
        # BLIP input extraction
        blip_tokens = self.blip_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        blip_pixel_values = self.blip_processor(images, return_tensors='pt')['pixel_values']
        blip_input_ids = blip_tokens['input_ids']
        blip_attn_mask = blip_tokens['attention_mask']

        # ViT input extraction
        vit_pixel_values = self.vit_processor(images, return_tensors='pt')['pixel_values']

        # BERT input extraction
        bert_tokens = self.bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        bert_input_ids = bert_tokens['input_ids']
        bert_attn_mask = bert_tokens['attention_mask']

        return (blip_pixel_values, blip_input_ids, blip_attn_mask, 
                vit_pixel_values, bert_input_ids, bert_attn_mask)
        
