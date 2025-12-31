# 🛡️ Hate-AI Detector: End-to-End Content Moderation System

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **M.Tech Thesis Project** | *IIIT Bhubaneswar*

### 🔴 **Live Cloud Demo:** [Click Here to Launch App](https://huggingface.co/spaces/Sid1907/Hate_AI_Detector)

---

## 📌 Project Overview
**Hate-AI Detector** is a robust Machine Learning pipeline designed to classify social media text into three safety categories:
1.  **Hate Speech** (High Severity)
2.  **Offensive Language** (Medium Severity)
3.  **Normal / Safe** (Low Severity)

Unlike traditional keyword-based filters, this project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to understand the *contextual nuance* of toxic language. The system is fully deployed using a decoupled architecture (Model Registry + Inference App).

---

⚙️ Technical Architecture

1. Dataset
Source: Davidson et al. (2017) - Automated Hate Speech Detection and the Problem of Offensive Language.

Composition: ~24,000 labeled tweets.

Class Imbalance: Hate speech constitutes only ~5% of the data.

Handling Strategy: Implemented Weighted Cross-Entropy Loss in train.py to penalize the model heavily for missing hate speech samples.

2. Model Pipeline (train.py)
Base Architecture: bert-base-uncased (12 Layers, 110M Parameters).

Tokenizer: WordPiece tokenizer (max sequence length: 128 tokens).

Optimization: AdamW optimizer with a linear learning rate scheduler.

Training: Fine-tuned for 3 epochs on a GPU environment.

3. Inference Engine (app.py)
Framework: Streamlit (Python-based Web UI).

Logic: Instead of simple argmax, the app uses Probability Thresholding. If the confidence for "Hate Speech" exceeds 40%, it overrides "Normal" predictions to prioritize safety (High Recall approach).

📸 Screenshots
1. Real-Time Detection Interface
The deployed application detecting hate speech with confidence scores.

---

## 📂 Project Structure
This repository contains the full source code for training, testing, and deployment:

```bash
├── app.py                 # The Main Streamlit Application (Frontend & Inference Logic)
├── train.py               # Script for Fine-Tuning BERT on the dataset
├── test_model.py          # Script for evaluating model performance (Precision/Recall)
├── requirements.txt       # List of Python dependencies
├── training_data.csv      # The Davidson et al. Dataset
├── README.md              # Project Documentation
└── screenshot1.png        # UI Demo Screenshot
```

---

## 🛠️ Installation & Usage
Prerequisite
Ensure you have Python 3.8+ installed.

1. Clone the Repository

```bash
git clone https://github.com/TamgadgeSiddhant19/BERT-Hate-Speech-Detection-End-to-End.git
cd BERT-Hate-Speech-Detection-End-to-End
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the Application

```bash
streamlit run app.py
```

4. (Optional) Retrain the Model
If you want to train the model from scratch on your local machine (GPU recommended):

```bash
python train.py
```

---

## 📈 Evaluation
- Metrics: Precision, Recall, F1-score (per-class and macro-averaged)
- Validation Accuracy: ~90%
- Known limitations: The dataset is imbalanced and drawn from older Twitter data; model performance may degrade on newer platforms or different dialects without further fine-tuning.

Future Work: Integration of the HateXplain dataset to improve detection of implicit toxicity.

---

## 🧾 License
This project is released under the MIT License.

---

## 📚 References
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language.
