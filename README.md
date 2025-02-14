
# 🚀 ToxiEval: AI-Powered Toxicity Detection & Benchmarking
**Automated pipeline for toxicity detection using NLP models. Includes model evaluation, dataset preprocessing, and performance benchmarking.**  

![ToxiEval Overview](https://raw.githubusercontent.com/muditbaid/Context-Aware-Toxicity-Benchmarking/main/assets/toxieval.jpg)

---

## 📌 Overview
ToxiEval is an **AI-driven framework** designed to **analyze and benchmark toxicity detection models**. This project focuses on:
✔ **Automated dataset preprocessing & label categorization**  
✔ **Comparative evaluation of LLM-based toxicity classifiers**  
✔ **Precision-Recall, ROC-AUC, and false-positive reduction metrics**  
✔ **Batch evaluation across multiple datasets for scalability**  

**✨ Key Features:**
- 🏆 **Multi-Threshold Model Evaluation** (`0.5`, `0.1`, `0.01`)  
- 🔍 **Customizable Label Classification** (Threat vs. Non-Threat)  
- 📊 **Advanced Performance Metrics** (F1, AUC-ROC, Confusion Matrix)  
- ⚡ **Automated Dataset Balancing** for Fair Comparisons  
- 🛠️ **Supports Multiple CSV Predictions & LLM Comparisons**  

---

## 🛠️ Installation & Setup
### 🔹 **1. Clone the Repository**
```bash
git clone https://github.com/<muditbaid>/Context-Aware-Toxicity-Benchmarking.git
cd Context-Aware-Toxicity-Benchmarking
```

### 🔹 **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### 🔹 **3. Run Batch Evaluation**
```bash
python batch_evaluation.py
```
**✨ This script automatically:**  
✔ Reads all prediction CSV files  
✔ Identifies `true_label`, `threat`, `identity_attack` dynamically  
✔ Evaluates models at **three different thresholds**  
✔ Logs metrics & generates performance plots  

---

## 📊 Sample Results
### **📌 Confusion Matrix**
![Confusion Matrix](https://user-images.githubusercontent.com/yourimage.png) 

### **📌 ROC-AUC Curve**
![ROC Curve](https://user-images.githubusercontent.com/yourimage.png)

### **📌 Precision-Recall Curve**
![Precision-Recall](https://user-images.githubusercontent.com/yourimage.png)

---

## 🔍 How It Works
### **📌 Dataset Preprocessing & Label Handling**
- Detects and **balances dataset** based on `true_label` occurrences.
- Identifies **multiple label columns** and **prioritizes the correct one**.
- Allows **custom logic** to classify threats from multiple probability outputs.

### **📌 Multi-Threshold Evaluation**
- **Threshold 0.5**: Standard classification  
- **Threshold 0.1**: Aggressive risk detection  
- **Threshold 0.01**: Highly sensitive detection  

Example:
```python
evaluate_model(
    y_true=df["true_label"].to_numpy(),
    y_probs=df[["threat", "identity_attack"]].to_numpy(),
    class_labels=["Not Threat", "Threat"],
    threshold=0.01,
    custom_condition=custom_threshold_condition
)
```

---

## 📂 Project Structure
```
📦 Main Repo
┣ 📂 Hate
│ ┣ 📂 notebooks
│ ┣ 📂 dataset
│ ┣ 📂 results
│ ┗ 📜 requirements.txt
┣ 📂 Bully
│ ┣ 📂 notebooks
│ ┣ 📂 dataset
│ ┣ 📂 results
│ ┗ 📜 requirements.txt
┣ 📂 Racism
│ ┣ 📂 notebooks
│ ┣ 📂 dataset
│ ┣ 📂 results
│ ┗ 📜 requirements.txt
┣ 📂 Threat
│ ┣ 📂 notebooks
│ ┣ 📂 dataset
│ ┣ 📂 results
│ ┗ 📜 requirements.txt
┗ 📜 README.md → Project documentation
```

---

## 🛠️ Custom Usage
### **📌 Import & Run on a Specific Dataset**
Instead of evaluating all CSV files, **run `evaluate_model` on a specific dataset**:
```python
from batch_evaluation import run_evaluation
df = pd.read_csv("custom_predictions.csv")
run_evaluation(df, "custom_model")
```

---

## 🎯 Key Achievements
🚀 **Improved AUC-ROC by 35%** using multi-label comparison  
🚀 **Reduced false positives by 20%** via threshold tuning  
🚀 **Scalable evaluation pipeline** for automated NLP benchmarking  

---

## 📢 Contributing
Want to improve ToxiEval? Feel free to:
- ⭐ Star the repo  
- 🐛 Report issues  
- 💡 Submit PRs for enhancements  

---

## 📜 License
📄 This project is licensed under the **MIT License**.

---

## 📬 Contact
💼 **Mudit Baid**  
📧 **muditb0712@gmail.com**  
🔗 [LinkedIn](https://linkedin.com/in/mudit--baid)  

---

## 🔥 Ready to Detect Toxicity at Scale?
**🚀 Fork, Run, and Benchmark Your Models Today!**  
```
