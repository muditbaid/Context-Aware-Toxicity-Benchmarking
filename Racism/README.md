# Racism Detection

This folder contains datasets, notebooks, and results related to **Racism Detection** within the **Context-Aware Toxicity Benchmarking** project. The objective is to benchmark various **racism classification models** using multiple datasets and evaluation metrics.

## 📌 Folder Structure
```
📂 Racism
│── 📂 notebooks      # Jupyter Notebooks for training & evaluation
│── 📂 dataset        # Datasets for racism classification
│── 📜 requirements.txt # Dependencies for running the models
```

## 📥 Dataset Information
The datasets used in this folder **are not stored directly in this repository** due to size limitations. You can access them from the following sources:

- **BiasCorp Dataset** ([Harvard Dataverse](https://dataverse.harvard.edu))
  - Extracted from online comments (Fox News, Breitbart, YouTube)
  - Labeled via sentiment analysis and MTurk annotations
- **Sakren Twitter Dataset** ([Hugging Face](https://huggingface.co))
- **Tweets Hate Speech Detection Dataset** ([Hugging Face](https://huggingface.co/datasets/tweets-hate-speech-detection))
  - Classifies tweets based on racist/sexist sentiment
- **Kaggle Racism Dataset** ([Kaggle](https://www.kaggle.com/datasets))
  - Uses GCR-NN for sentiment-based racism detection (not manually annotated)

Make sure to **download the datasets manually** and place them in the `dataset/` folder before running any experiments.

## 🏗️ Models Used
This benchmark evaluates multiple **Racism Detection Models**, including:

1. **DEBERTA (tasksource/deberta-small-long-nli)** ([Hugging Face](https://huggingface.co/tasksource/deberta-small-long-nli))
   - Reported accuracy: **70%**
2. **Cardiff NLP RoBERTa (twitter-roberta-base-hate-latest)** ([Hugging Face](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest))
   - Reported accuracy: **92%**
3. **XLM-R Racismo (xlm-r-racismo-es-v2-finetuned-detests)** ([Hugging Face](https://huggingface.co/Pablo94/xlm-r-racismo-es-v2-finetuned-detests))
   - Reported accuracy: **Unknown**

These models classify text based on racism detection, often in **multilingual** contexts, using **pretrained transformer architectures**.

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

### 3️⃣ View Results
Results, including model performance metrics, confusion matrices, and bias evaluations, are saved in the `results/` folder.

## 📊 Evaluation Metrics
Models are compared based on:
- **Accuracy & F1-score** across different datasets
- **Bias Metrics**
  - **Subgroup AUC** (Detects biases for specific communities)
  - **BPSN AUC** (False positives for neutral speech)
  - **BNSP AUC** (False negatives for racism detection)
- **Generalized Mean of Bias (GMB) AUC**

## 📝 References
- [BiasCorp Dataset](https://dataverse.harvard.edu)
- [DEBERTA Model for NLP Tasks](https://huggingface.co/tasksource/deberta-small-long-nli)
- [Cardiff NLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest)
- [XLM-R Racismo for Multilingual Hate Speech](https://huggingface.co/Pablo94/xlm-r-racismo-es-v2-finetuned-detests)
- [Hate Speech Detection in Tweets](https://huggingface.co/datasets/tweets-hate-speech-detection)

---
For any issues, please open an **issue** or contribute via **pull request**! 🚀

