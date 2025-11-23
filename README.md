# SentiBench: Multi-Model Sentiment Analysis Pipeline 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit__Learn-yellow)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-red)

## 📌 Project Overview
**SentiBench** is a comprehensive comparative study designed to classify sentiments in e-commerce reviews (Amazon Fine Food Reviews). The project benchmarks various NLP approaches, ranging from traditional Machine Learning algorithms to Deep Learning architectures (RNN/LSTM) and state-of-the-art Transformer models (RoBERTa, DistilBERT).

The goal is to determine the most effective approach for handling textual nuances in customer feedback, classifying them into **Positive**, **Neutral**, or **Negative**.

## 🛠️ Methodology & Models

The project implements and compares the following models:

### 1. Traditional Machine Learning (Baseline)
* **Feature Extraction:** TF-IDF Vectorizer.
* **Models:** * Logistic Regression
    * Support Vector Machine (SVM)
    * Random Forest

### 2. Deep Learning (Keras/TensorFlow)
* **Preprocessing:** Tokenization & Padding sequences.
* **Architectures:**
    * **Simple RNN:** For basic sequential data processing.
    * **LSTM (Long Short-Term Memory):** To capture long-term dependencies in text.

### 3. Pre-trained Models & Transformers
* **VADER:** Lexicon and rule-based sentiment analysis tool (great for speed).
* **RoBERTa (Twitter-base):** Robustly optimized BERT approach.
* **DistilBERT:** A distilled, lighter version of BERT.

## 📊 Performance Highlights

Based on the evaluation on the test set, here is a summary of the results:

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| **RoBERTa** | **84.5%** | **0.86** |
| **LSTM** | 84.0% | 0.79 |
| **VADER** | 85.0% | 0.83 |
| **Logistic Regression** | 82.0% | 0.74 |
| **RNN** | 77.5% | 0.75 |
| **DistilBERT** | 70.5% | 0.73 |

> **Insight:** RoBERTa demonstrated the best balance between Precision and Recall, effectively handling the context of reviews compared to simpler models. LSTM also showed strong performance as a trainable deep learning model.

## ⚙️ Requirements

To run this project, you need the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow transformers nltk

efficiently.

🚀 How to Run
Clone the repository:

Bash

git clone [https://github.com/your-username/SentiBench.git](https://github.com/your-username/SentiBench.git)
Install dependencies:

Bash

pip install -r requirements.txt
Dataset: Ensure you have the Reviews.csv file in the ../dataset/ directory (or adjust the path in the notebook).

Run the Notebook: Open final_result_with_rnn_lstm.ipynb in Jupyter Lab or VS Code and execute cells sequentially.

📂 Project Structure
final_result_with_rnn_lstm.ipynb: The main notebook containing all code.

sentiment_analysis_results.csv: Output file containing predictions for test cases.

dataset/: Directory for the dataset (not included in repo).

📈 Outputs & Visualization
The notebook generates:

Confusion Matrices: Heatmaps showing true vs. predicted labels for each model.

Bar Charts: Comparative visualization of Accuracy, Precision, Recall, and F1-Score.

Error Analysis: Identification of specific reviews where models failed (e.g., high positive score for a negative review).

👤 Author
AI-Eng [MahmoudAmrAmin] 