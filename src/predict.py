import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load JSONL data"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def decode_predictions_v2(text, logits, tokenizer, label_encoder, encoding):
    """
    Properly decode token-level predictions to character spans.
    
    CRITICAL: logits[0] has shape (seq_len,) after argmax.
    offset_mapping[0] also has shape (seq_len, 2).
    They MUST be aligned: logits[0][i] corresponds to offset_mapping[0][i].
    
    Args:
        text: original text string
        logits: (1, seq_len, num_labels) tensor
        tokenizer: HF tokenizer
        label_encoder: label encoder fitted with all BIO labels
        encoding: tokenizer output (contains input_ids, offset_mapping)
    
    Returns:
        list of entity dicts with {start, end, label, pii}
    """
    # Get predictions: argmax over label dimension
    preds = torch.argmax(logits, dim=-1)[0].tolist()  # Shape: (seq_len,)
    
    # Get offset mapping and input IDs
    offset_mapping = encoding['offset_mapping'][0].tolist()  # Shape: (seq_len, 2)
    input_ids = encoding['input_ids'][0].tolist()  # Shape: (seq_len,)
    special_ids = set(tokenizer.all_special_ids)
    
    # Create ID-to-label mapping
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # CRITICAL: Loop through ALL tokens, but skip special tokens
    # Ensure len(preds) == len(offset_mapping) == len(input_ids)
    assert len(preds) == len(offset_mapping) == len(input_ids), \
        f"Mismatch: preds={len(preds)}, offsets={len(offset_mapping)}, input_ids={len(input_ids)}"
    
    entities = []
    current_entity = None
    
    for idx in range(len(preds)):
        token_id = input_ids[idx]
        label_id = preds[idx]
        char_start, char_end = offset_mapping[idx]
        
        # Skip special tokens (CLS, SEP, PAD, etc.)
        if token_id in special_ids:
            # Finalize any ongoing entity
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Skip padding/empty spans
        if char_start == char_end:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Verify span is within text bounds
        if not (0 <= char_start < char_end <= len(text)):
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Get label name
        label_name = id2label.get(label_id, 'O')
        
        if label_name == 'O':
            # Outside any entity
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
        else:
            # Inside an entity: parse BIO format
            parts = label_name.split('-')
            if len(parts) != 2:
                # Malformed label, treat as O
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            tag_type, entity_type = parts[0], parts[1]
            is_beginning = (tag_type == 'B')
            
            if is_beginning or current_entity is None or current_entity['label'] != entity_type:
                # Start new entity
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    'start': char_start,
                    'end': char_end,
                    'label': entity_type,
                    'pii': entity_type in ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE']
                }
            else:
                # Continue current entity (I-tag, same type)
                current_entity['end'] = char_end
    
    # Finalize last entity
    if current_entity is not None:
        entities.append(current_entity)
    
    # Post-process: validate and deduplicate
    entities = validate_and_deduplicate_entities(text, entities)
    
    return entities

def validate_and_deduplicate_entities(text, entities):
    """
    Validate entity spans and remove overlaps.
    
    Returns:
        list of valid non-overlapping entities
    """
    if not entities:
        return []
    
    # Validate spans
    valid = []
    for entity in entities:
        start, end = entity['start'], entity['end']
        
        # Check bounds
        if not (0 <= start < end <= len(text)):
            continue
        
        # Extract and validate substring
        substring = text[start:end]
        if not substring or substring.isspace():
            continue
        
        valid.append(entity)
    
    if not valid:
        return []
    
    # Sort by start position, then by end (descending length)
    valid.sort(key=lambda e: (e['start'], -e['end']))
    
    # Remove overlaps (keep first)
    dedup = []
    for entity in valid:
        overlaps = False
        for existing in dedup:
            # Check if spans overlap
            if not (entity['end'] <= existing['start'] or entity['start'] >= existing['end']):
                overlaps = True
                break
        if not overlaps:
            dedup.append(entity)
    
    return dedup

def predict(model, tokenizer, label_encoder, input_data):
    """Run inference on all input examples"""
    model.eval()
    predictions = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for item in input_data:
        text = item['text']
        utterance_id = item['id']
        
        # Tokenize with offset mapping
        # CRITICAL: Use exact same settings as training
        encoding = tokenizer(
        text,
        truncation=True,               # no fixed max_length unless training used it
        padding=False,                 # match training!
        return_offsets_mapping=True,
        return_tensors='pt'
        )

        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Run model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, num_labels)
        
        # Decode predictions
        entities = decode_predictions_v2(text, logits, tokenizer, label_encoder, encoding)
        
        predictions[utterance_id] = entities
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run inference on the trained model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file.")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # Recreate label encoder (MUST match training exactly)
    label_encoder = LabelEncoder()
    label_encoder.fit([
        'O',
        'B-CREDIT_CARD', 'I-CREDIT_CARD',
        'B-PHONE', 'I-PHONE',
        'B-EMAIL', 'I-EMAIL',
        'B-PERSON_NAME', 'I-PERSON_NAME',
        'B-DATE', 'I-DATE',
        'B-CITY', 'I-CITY',
        'B-LOCATION', 'I-LOCATION'
    ])
    
    # Load input data
    input_data = load_data(args.input)
    
    # Run predictions
    predictions = predict(model, tokenizer, label_encoder, input_data)
    
    # Save predictions
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {args.output}")
    print(f"Total predictions: {len(predictions)}")
    for uid, entities in list(predictions.items())[:3]:
        print(f"  {uid}: {len(entities)} entities")
        for e in entities[:2]:
            print(f"    [{e['start']}, {e['end']}] {e['label']}: '{input_data[0]['text'][e['start']:e['end']]}'")

if __name__ == "__main__":
    main()