import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# --- CONFIGURATION ---
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./hate_speech_model"
EPOCHS = 2
# GTX 1050 Ti Specific Settings
BATCH_SIZE = 8  # Reduced to fit 4GB VRAM
GRAD_ACCUMULATION = 2 # Accumulate gradients to simulate batch_size=16

def compute_metrics(p):
    """
    Calculates accuracy and F1 score during evaluation.
    Weighted F1 is used because the dataset is imbalanced.
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    f1 = f1_score(labels, pred, average='weighted')
    acc = accuracy_score(labels, pred)
    return {"accuracy": acc, "f1": f1}

def main():
    # 1. Setup Hardware
    # Checks if CUDA (GPU) is available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")
    
    if device == "cuda":
        # Check specific GPU properties
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Load Data
    # Reads the clean CSV file created in the previous step.
    if not os.path.exists("train_data.csv"):
        print("Error: 'train_data.csv' not found. Run preprocess.py first.")
        return
        
    df = pd.read_csv("train_data.csv").dropna()
    
    # Stratified Split
    # Splits data 80% Train, 20% Validation.
    # 'stratify' ensures both sets have the same ratio of Hate/Offensive/Neither.
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    # Convert Pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 3. Tokenization
    # Loads the BERT tokenizer and processes the text.
    print("Tokenizing Data...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # Truncation=True cuts off texts longer than 128 tokens
        # Padding='max_length' adds zeros to shorter texts so all are length 128
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # 4. Model Setup
    # Loads the pre-trained BERT model with a classification head for 3 classes.
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # 5. Training Arguments
    # Defines how the model should be trained.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,               # Small learning rate to prevent destroying pre-trained weights
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE*2,
        gradient_accumulation_steps=GRAD_ACCUMULATION, # Updates weights every 2 steps (Effective Batch=16)
        num_train_epochs=EPOCHS,
        weight_decay=0.01,                # Regularization to prevent overfitting
        load_best_model_at_end=True,      # Restores the best version of the model after training
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),   # Mixed Precision (Crucial for 1050 Ti)
        logging_steps=50,
        dataloader_num_workers=0,         # Set to 0 for Windows compatibility
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Start Training
    print("\nStarting Training...")
    trainer.train()

    # 8. Evaluation
    print("\nEvaluating Final Model...")
    preds = trainer.predict(tokenized_val)
    y_pred = np.argmax(preds.predictions, axis=1)
    
    print("\n--- CLASSIFICATION REPORT ---")
    # Target names map to: 0=Hate, 1=Offensive, 2=Neither
    print(classification_report(val_df['label'], y_pred, target_names=["Hate", "Offensive", "Neither"]))

    # 9. Save Model
    # Saves the model logic and tokenizer vocabulary to disk.
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()