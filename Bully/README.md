# Cyberbullying Detection

This folder contains datasets, notebooks, and results related to **Cyberbullying Detection** as part of the **Context-Aware Toxicity Benchmarking** project. The objective is to benchmark various cyberbullying classification models using multiple datasets and evaluation metrics.

## 📌 Folder Structure
```
📂 Bully
│── 📂 notebooks      # Jupyter Notebooks for training & evaluation
│── 📂 dataset        # Datasets for cyberbullying classification
│── 📜 requirements.txt # Dependencies for running the models
```

## 📥 Dataset Information
The datasets used in this folder **are not stored directly in this repository** due to size limitations. You can access them from the following sources:

- **Cyberbullying Classification Dataset (Kaggle)**: [Kaggle Link](https://www.kaggle.com/datasets)
- **Jigsaw Unintended Bias in Toxicity Classification**: [Kaggle Link](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- **Wikipedia Detox Dataset**: [GitHub](https://github.com/ewulczyn/wiki-detox)
- **Gab Hate Corpus**: [OSF Dataset](https://osf.io/)
- **Twitter Hate Speech Dataset**: [ACL Paper](https://aclanthology.org/P17-2089/)
- **Instagram Cyberbullying Dataset**: [Dataset Reference](https://journals.sagepub.com/doi/10.1177/0002764218778229)
- **Formspring Dataset**: [Dataset Reference](https://arxiv.org/abs/1610.08914)

Make sure to **download the datasets manually** and place them in the `dataset/` folder before running any experiments.

## 🏗️ Models Used
This benchmark evaluates multiple **Cyberbullying Detection Models**, including:

1. **BERT_PyTorch (Kaggle Notebook)** ([Kaggle](https://www.kaggle.com))
2. **fair_cyberbullying_detection (Fairness-Aware Model)** ([GitHub](https://github.com/ogencoglu/fair_cyberbullying_detection))
3. **DistilBERT Fine-Tuned Model** ([Hugging Face](https://huggingface.co/SSEF-HG-AC/distilbert-uncased-finetuned-cyberbullying))
4. **Word2Vec-based Cyberbullying Classification** ([GitHub](https://github.com/leoAshu/cyberbullying-classification))
5. **BERT Text Classification for Cyberbullying** ([Kaggle](https://www.kaggle.com))

These models classify text into **cyberbullying-related categories**, including:
- **Age-based bullying**
- **Ethnicity-based bullying**
- **Gender-based bullying**
- **Religion-based bullying**
- **Other cyberbullying**
- **Not cyberbullying**

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
  - **BNSP AUC** (False negatives for cyberbullying content)
- **Generalized Mean of Bias (GMB) AUC**

## 📝 References
- [Cyberbullying Classification Dataset](https://www.kaggle.com/datasets)
- [Cyberbullying Detection with Fairness Constraints](https://github.com/ogencoglu/fair_cyberbullying_detection)
- [Hate Speech and Cyberbullying on Twitter](https://aclanthology.org/P17-2089/)
- [Wikipedia Detox: Ex Machina](https://arxiv.org/abs/1610.08914)
- [Gab Hate Corpus](https://osf.io/)
- [Deep Learning for Cyberbullying Detection](https://journals.sagepub.com/doi/10.1177/0002764218778229)

---
For any issues, please open an **issue** or contribute via **pull request**! 🚀

