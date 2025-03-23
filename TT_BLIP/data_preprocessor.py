import torch 
from transformers import ViTImageProcessor, BertTokenizer, BlipProcessor
from nltk.corpus import stopwords


class DataPreprocessor():
    def __init__(self):
        # BLIP preprocessing utils
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        empty = self.blip_processor(torch.zeros((3, 224, 224)), [""], return_tensors='pt')
        self.empty_pixel_values = empty['pixel_values']
        self.empty_input_ids = empty['input_ids']
        self.empty_attn_mask = empty['attention_mask']

        # ViT preprocessing utils
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast = True)

        # BERT preprocessing utils
        self.bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


    def __call__(self, images, texts):
        # BLIP input extraction
        blip_tokens = self.blip_processor(images, texts, return_tensors='pt', padding=True, truncation=True)

        blip_pixel_values = blip_tokens['pixel_values']
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
        
