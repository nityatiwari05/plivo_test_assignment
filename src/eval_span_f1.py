import json
import argparse

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_predictions_json(file_path):
    """Load predictions from JSON dict format"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(gold, pred):
    """
    Calculate span-level precision, recall, and F1.
    
    Args:
        gold: dict {utt_id: [entities]}
        pred: dict {utt_id: [entities]}
    """
    tp = fp = fn = 0
    pii_tp = pii_fp = pii_fn = 0

    for utt_id in gold:
        gold_entities = {(ent['start'], ent['end'], ent['label']) for ent in gold[utt_id]}
        pred_entities = {(ent['start'], ent['end'], ent['label']) for ent in pred.get(utt_id, [])}

        # Overall metrics
        tp += len(gold_entities & pred_entities)
        fp += len(pred_entities - gold_entities)
        fn += len(gold_entities - pred_entities)
        
        # PII-specific metrics
        gold_pii = {e for e in gold_entities if e[2] in ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE']}
        pred_pii = {e for e in pred_entities if e[2] in ['CREDIT_CARD', 'PHONE', 'EMAIL', 'PERSON_NAME', 'DATE']}
        
        pii_tp += len(gold_pii & pred_pii)
        pii_fp += len(pred_pii - gold_pii)
        pii_fn += len(gold_pii - pred_pii)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    pii_precision = pii_tp / (pii_tp + pii_fp) if (pii_tp + pii_fp) > 0 else 0
    pii_recall = pii_tp / (pii_tp + pii_fn) if (pii_tp + pii_fn) > 0 else 0
    pii_f1 = 2 * (pii_precision * pii_recall) / (pii_precision + pii_recall) if (pii_precision + pii_recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pii_precision': pii_precision,
        'pii_recall': pii_recall,
        'pii_f1': pii_f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'pii_tp': pii_tp,
        'pii_fp': pii_fp,
        'pii_fn': pii_fn
    }

def main(gold_file, pred_file):
    # Load gold data (JSONL format)
    gold_data = load_jsonl(gold_file)
    gold = {item['id']: item['entities'] for item in gold_data}
    
    # Load predictions (JSON dict format)
    pred = load_predictions_json(pred_file)

    metrics = calculate_metrics(gold, pred)

    print(f"=== Overall Metrics ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print(f"\n=== PII-Only Metrics ===")
    print(f"PII Precision: {metrics['pii_precision']:.4f}")
    print(f"PII Recall: {metrics['pii_recall']:.4f}")
    print(f"PII F1 Score: {metrics['pii_f1']:.4f}")
    print(f"PII TP: {metrics['pii_tp']}, PII FP: {metrics['pii_fp']}, PII FN: {metrics['pii_fn']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate span-level NER predictions.")
    parser.add_argument("--gold", type=str, required=True, help="Gold JSONL file with entities.")
    parser.add_argument("--pred", type=str, required=True, help="Predicted JSON file with entities.")
    
    args = parser.parse_args()
    main(args.gold, args.pred)