import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch

class Dataset:
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        # self.max_length = max_length
        self.data = self.load_data()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['O', 'B-CREDIT_CARD', 'I-CREDIT_CARD', 
                                 'B-PHONE', 'I-PHONE', 
                                 'B-EMAIL', 'I-EMAIL', 
                                 'B-PERSON_NAME', 'I-PERSON_NAME', 
                                 'B-DATE', 'I-DATE', 
                                 'B-CITY', 'I-CITY', 
                                 'B-LOCATION', 'I-LOCATION'])
        self.encoded_data = self.prepare_data()

    def load_data(self):
        with open(self.file_path, 'r') as f:
            return [json.loads(line) for line in f]

    def prepare_data(self):
        """Prepare and encode all data at initialization"""
        encoded_data = []
        
        for entry in self.data:
            # Tokenize the text
            encoding = self.tokenizer(
                entry['text'],
                # max_length=self.max_length,
                # padding='max_length',
                truncation=True,
                return_offsets_mapping=True
            )
            
            # Create label sequence
            labels = [-100] * len(encoding['input_ids'])  # -100 is ignored in loss
            
            for entity in entry['entities']:
                start, end, label = entity['start'], entity['end'], entity['label']
                
                # Map character offsets to token indices
                for idx, (token_start, token_end) in enumerate(encoding['offset_mapping']):
                    if token_start < end and token_end > start:
                        if token_start >= start and token_end <= end:
                            # Determine if B- or I- tag
                            if token_start == start:
                                labels[idx] = self.label_encoder.transform([f'B-{label}'])[0]
                            else:
                                labels[idx] = self.label_encoder.transform([f'I-{label}'])[0]
                        elif labels[idx] == -100:  # Only set if not already labeled
                            labels[idx] = self.label_encoder.transform(['O'])[0]
            
            # Set remaining labels to 'O'
            for idx in range(len(labels)):
                if labels[idx] == -100:
                    labels[idx] = self.label_encoder.transform(['O'])[0]
            
            encoded_data.append({
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            })
        
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]