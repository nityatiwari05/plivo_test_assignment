import time
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

def measure_latency(model_dir, input_file, runs=50):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # Load the input data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    latencies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for run in range(runs):
        for entry in data:
            text = entry['text']
            
            # Tokenize the text
            encoding = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            )
            
            # Move to device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                _ = model(**encoding)
            end_time = time.time()

            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Calculate p50 and p95 latencies
    latencies.sort()
    p50_idx = int(0.5 * len(latencies))
    p95_idx = int(0.95 * len(latencies))
    
    p50 = latencies[p50_idx]
    p95 = latencies[p95_idx]

    print(f"Total runs: {len(latencies)}")
    print(f"p50 latency: {p50:.2f} ms")
    print(f"p95 latency: {p95:.2f} ms")
    print(f"Mean latency: {sum(latencies) / len(latencies):.2f} ms")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure model inference latency.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--runs", type=int, default=50, help="Number of runs for latency measurement.")
    
    args = parser.parse_args()
    measure_latency(args.model_dir, args.input, args.runs)