import json
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from dataset import Dataset
from eval_span_f1 import calculate_metrics
from torch.optim import AdamW

def collate_fn(batch):
    """
    Pads input_ids, attention_mask, and labels to the max length in the batch.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100   # ignored in loss
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels
    }


def train_model(train_dataset, dev_dataset, model, tokenizer, output_dir, epochs=3, batch_size=16, lr=5e-5):
    """Train the token classification model"""
    
    train_loader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    shuffle=True,
    collate_fn=collate_fn
    )

    dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
    )


    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save the trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--dev', type=str, required=True, help='Path to development data')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    # Load tokenizer first
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    
    # Load datasets with tokenizer
    train_data = Dataset(args.train, tokenizer)
    dev_data = Dataset(args.dev, tokenizer)

    # Load model
    model = DistilBertForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=15  # 15 BIO labels (O + 7 entity types * 2)
    )

    # Create output directory 
    os.makedirs(args.out_dir, exist_ok=True)

    # Train the model
    train_model(train_data, dev_data, model, tokenizer, args.out_dir, 
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    

if __name__ == "__main__":
    main()