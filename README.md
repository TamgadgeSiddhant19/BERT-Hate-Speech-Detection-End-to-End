# 🛡️ Hate-AI Detector: End-to-End Content Moderation System

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **M.Tech Thesis Project** | *IIIT Bhubaneswar*

### 🔴 **Live Cloud Demo:** [Click Here to Launch App](https://huggingface.co/spaces/[YOUR_HF_USERNAME]/Hate_AI_Detector)

---

## 📌 Project Overview
**Hate-AI Detector** is a robust Machine Learning pipeline designed to classify social media text into three safety categories:
1.  **Hate Speech** (High Severity)
2.  **Offensive Language** (Medium Severity)
3.  **Normal / Safe** (Low Severity)

Unlike traditional keyword-based filters, this project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to understand the *contextual nuance* of toxic language. The system is fully deployed using a decoupled architecture (Model Registry + Inference App).

---

## 1. Dataset

- Source: Davidson et al. (2017) - "Automated Hate Speech Detection and the Problem of Offensive Language."
- Composition: Approximately 24,000 labeled tweets.
- Class Imbalance: Hate speech constitutes only ~5% of the dataset.
- Handling Strategy: Implemented a Weighted Cross-Entropy Loss in `train.py` to penalize the model more heavily for misclassifying hate speech samples and mitigate the class imbalance.

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

## 📈 Evaluation
- Metrics: Precision, Recall, F1-score (per-class and macro-averaged)
- Known limitations: The dataset is imbalanced and drawn from older Twitter data; model performance may degrade on newer platforms or different dialects without further fine-tuning.

---

## 🚀 Deployment
The app is built with Streamlit and intended to be hosted on Hugging Face Spaces. The model weights can be loaded from a model registry or a checkpoint in the repo depending on deployment setup.

---

## 🛠️ Usage
1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Unix/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```bash
streamlit run app.py
```

---

## 🧾 License
This project is released under the MIT License.

---

## 📚 References
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language.
