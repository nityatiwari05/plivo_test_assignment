from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch

def create_model(model_name='distilbert-base-uncased', num_labels=7):
    model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def create_tokenizer(model_name='distilbert-base-uncased'):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    return tokenizer

def load_model(model_path):
    model = DistilBertForTokenClassification.from_pretrained(model_path)
    return model

def save_model(model, output_dir):
    model.save_pretrained(output_dir)