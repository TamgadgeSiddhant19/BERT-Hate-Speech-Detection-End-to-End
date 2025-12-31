import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# --- CONFIGURATION ---
SEARCH_DIR = "./hate_speech_model"
TOKENIZER_NAME = "bert-base-uncased" # Download tokenizer from web to be safe
LABELS = ["Hate Speech", "Offensive Language", "Normal / Neither"]

def find_valid_model_path(root_dir):
    print(f"🕵️  Scanning '{root_dir}' for valid model weights...")
    
    candidates = []
    
    # Walk through all subfolders
    for dirpath, _, filenames in os.walk(root_dir):
        # Check if the weight file exists in this folder
        has_weights = "pytorch_model.bin" in filenames or "model.safetensors" in filenames
        
        if has_weights:
            print(f"    Found valid model in: {dirpath}")
            candidates.append(dirpath)
        else:
            # Only print if it looks like a model folder but is empty
            if "checkpoint" in dirpath:
                print(f"    Found checkpoint folder but NO weights in: {dirpath}")

    if not candidates:
        return None
    
    # Sort candidates to find the best one
    # Logic: We prefer the one with the highest number (latest training)
    def sort_key(path):
        if "checkpoint-" in path:
            try:
                return int(path.split("checkpoint-")[-1])
            except:
                return 0
        return 999999999 # Prefer the main folder (final save) if it exists
        
    candidates.sort(key=sort_key, reverse=True)
    best_path = candidates[0]
    
    print(f"\n Selected Best Model: {best_path}")
    return best_path

def load_model():
    # 1. FIND THE FILE
    model_path = find_valid_model_path(SEARCH_DIR)
    
    if model_path is None:
        print("\n" + "!"*60)
        print("CRITICAL FAILURE: No model weights found anywhere.")
        print("The training process did not finish saving the files.")
        print("YOU MUST RUN 'train.py' AGAIN.")
        print("!"*60 + "\n")
        return None, None, None

    # 2. LOAD IT
    print(" Loading model...")
    try:
        # Load Tokenizer from Web (Safe)
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # Load Model from Disk
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f" Success! Loaded on {device.upper()}")
        return tokenizer, model, device
        
    except Exception as e:
        print(f" Error loading: {e}")
        return None, None, None

def predict(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    idx = np.argmax(probs)
    return idx, probs[idx], probs

if __name__ == "__main__":
    tokenizer, model, device = load_model()
    
    if model:
        print("\n" + "="*50)
        print("HATE SPEECH DETECTOR - READY")
        print("="*50)
        while True:
            t = input("\nEnter text (or 'exit'): ")
            if t.lower() in ['exit', 'quit']: break
            if not t.strip(): continue
            
            idx, conf, probs = predict(t, tokenizer, model, device)
            print(f"Result: {LABELS[idx].upper()} ({conf*100:.1f}%)")