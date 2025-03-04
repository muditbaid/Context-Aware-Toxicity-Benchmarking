# Threat Detection

This folder contains datasets, notebooks, and results related to **Threat Detection** within the **Context-Aware Toxicity Benchmarking** project. The objective is to benchmark various **threat classification models** using multiple datasets and evaluation metrics.

## 📌 Folder Structure
```
📂 Threat
│── 📂 notebooks      # Jupyter Notebooks for training & evaluation
│── 📂 utils        # Utility function scripts
│── 📜 requirements.txt # Dependencies for running the models
```

## 📥 Dataset Information
The datasets used in this folder **are not stored directly in this repository** due to size limitations. You can access them from the following sources:

- **Combined Toxicity Profanity v2** ([Hugging Face](https://huggingface.co))
  - Labels: Toxic, Profane, Insult, Hate, Threat, Sexual, Offensive, Self-harm, Harassment
  - We do not know how the `Threat` label was assigned.
- **Suspicious Tweets Dataset** ([Kaggle](https://www.kaggle.com))
  - Collected from Twitter (~60k tweets labeled as **suspicious** or **non-suspicious**).
  - "Suspicious" posts include threat-like content (potentially harmful or malicious).
- **Toxic Comment Classification Dataset** ([Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge))
  - 159,571 Wikipedia comments labeled for toxicity, including **478 labeled as threats**.
- **Jigsaw Unintended Bias in Toxicity Classification** ([Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification))
  - 1,048,576 comments labeled for toxicity and bias.
- **Life-Threatening Comment Detection** ([GitHub Repository](https://github.com))
  - NLP-based model that detects life-threatening comments using keyword-based filtering.

Make sure to **download the datasets manually** and place them in the `dataset/` folder before running any experiments.

## 🏗️ Models Used
This benchmark evaluates multiple **Threat Detection Models**, including:

1. **Ensemble Model (Social Media Toxic Comments Classification)** ([GitHub](https://github.com))
   - Detects threats, obscenity, insults, and identity-based hate.
2. **Detoxify** ([GitHub](https://github.com/unitaryai/detoxify))
   - Uses **Jigsaw Toxic Comment datasets** for training.
   - Reported accuracy: **93.74%**.
3. **Toxic_Git Model (Comments Toxicity Detection)** ([GitHub](https://github.com))
   - Labels: **Toxic, Very Toxic, Obscene, Threat, Insult, Hate, Neutral**.
   - Reported accuracy: **94%**.
4. **BERT for Threatening Speech Detection** ([Hugging Face](https://huggingface.co/nvsl/bert-for-threatening))
   - Labels: **Threatening, Not Threatening**.
5. **NB-SVM (Baseline Model)** ([Kaggle](https://www.kaggle.com))
   - Uses **Toxic Comment Classification** dataset.
   - Reported accuracy: **97.6%**.
6. **Improved LSTM (GloVe + Dropout)** ([Kaggle](https://www.kaggle.com))
   - Uses **Toxic Comment Classification** dataset.
   - Reported accuracy: **97.6%**.

These models classify text based on **threat detection** in online comments and social media content.

## 🚀 Running the Experiments
### 1️⃣ Install Dependencies
Ensure that you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Model Evaluation
Execute the Jupyter notebooks in the `notebooks/` folder to:
- Load datasets
- Fine-tune and evaluate models
- Compute accuracy, F1-score, and bias metrics

Example command:
```bash
jupyter notebook notebooks/threat_detection_evaluation.ipynb
```

### 3️⃣ View Results
Results, including model performance metrics, confusion matrices, and bias evaluations, are saved in the `results/` folder.

## 📊 Evaluation Metrics
Models are compared based on:
- **Accuracy & F1-score** across different datasets
- **Bias Metrics**
  - **Subgroup AUC** (Detects biases for specific communities)
  - **BPSN AUC** (False positives for neutral speech)
  - **BNSP AUC** (False negatives for threat detection)
- **Generalized Mean of Bias (GMB) AUC**

## 📝 References
- [Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Jigsaw Unintended Bias Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Detoxify - Toxic Comment Detection](https://github.com/unitaryai/detoxify)
- [BERT for Threatening Speech Detection](https://huggingface.co/nvsl/bert-for-threatening)
- [NB-SVM Strong Linear Baseline](https://www.kaggle.com)
- [Life-Threatening Comment Detection](https://github.com)

---
For any issues, please open an **issue** or contribute via **pull request**! 🚀
