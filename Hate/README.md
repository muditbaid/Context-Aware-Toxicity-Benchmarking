# Hate Speech Detection

This folder contains datasets, notebooks, and results related to **Hate Speech Detection** within the **Context-Aware Toxicity Benchmarking** project. The goal is to benchmark various **hate speech classification models** using multiple datasets and evaluation metrics.

## 📌 Folder Structure
```
📂 Hate
│── 📂 notebooks      # Jupyter Notebooks for training & evaluation
│── 📂 dataset        # Datasets for hate speech classification
│── 📂 results        # Model outputs, metrics, and comparisons
│── 📜 requirements.txt # Dependencies for running the models
```

## 📥 Dataset Information
The datasets used in this folder are **not stored directly in this repository** due to size limitations. You can access them from the following sources:

- **HateXplain**: [Hugging Face Dataset](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain)
- **Dynamically Generated Hate Speech Dataset** (Facebook RoBERTa): [GitHub Repository](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)
- **Badmatr11x Dataset**: [Hugging Face](https://huggingface.co/datasets/badmatr11x/hate-offensive-speech)
- **Super Tweet Eval Dataset**: [Hugging Face](https://huggingface.co/datasets/cardiffnlp/super_tweeteval)
- **TDavidson Hate Speech Dataset**: [ArXiv Paper](https://arxiv.org/abs/1703.04009)

Make sure to **download the datasets manually** and place them in the `dataset/` folder before running any experiments.

## 🏗️ Models Used
This benchmark evaluates multiple **Hate Speech Detection Models**, including:

1. **HateXplain** ([GitHub](https://github.com/hate-alert/HateXplain))
2. **Facebook RoBERTa (Dynabench R4 Target)** ([Hugging Face](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target))
3. **DistilRoBERTa (Offensive & Hate Speech)** ([Hugging Face](https://huggingface.co/badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification))
4. **Cardiff NLP RoBERTa (Hate Speech Classification)** ([Hugging Face](https://huggingface.co/cardiffnlp/twitter-roberta-large-hate-latest))
5. **DehateBERT Mono-English** ([Hugging Face](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english))

These models classify text into categories such as **Hate, Offensive, Neutral**, or more fine-grained categories like **race, gender, and religious-based hate**.

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
jupyter notebook notebooks/hate_speech_evaluation.ipynb
```

### 3️⃣ View Results
Results, including model performance metrics, confusion matrices, and bias evaluations, are saved in the `results/` folder.

## 📊 Evaluation Metrics
Models are compared based on:
- **Accuracy & F1-score** across different datasets
- **Bias Metrics**
  - **Subgroup AUC** (Detects biases for specific communities)
  - **BPSN AUC** (False positives for neutral speech)
  - **BNSP AUC** (False negatives for hateful speech)
- **Generalized Mean of Bias (GMB) AUC**

## 📝 References
- [HateXplain: Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289)
- [Dynamically Generated Hate Speech Dataset](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)
- [DistilBERT: A Faster & Lighter Model](https://arxiv.org/abs/1910.01108)
- [Cardiff NLP Hate Speech Research](https://huggingface.co/cardiffnlp)
- [Automated Hate Speech Detection & Offensive Language](https://arxiv.org/abs/1703.04009)

---
For any issues, please open an **issue** or contribute via **pull request**! 🚀
