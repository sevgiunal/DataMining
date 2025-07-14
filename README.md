# 🧠 Data Analysis and Machine Learning Scripts

This repository contains standalone Python scripts for performing machine learning and data analysis on various datasets, including census income data, Twitter coronavirus sentiment data, and wholesale customer sales data.

---

## 📁 Project Contents

### 1. `adult.py`

**Dataset:** UCI Adult (Census Income)  
**Objective:** Predict whether a person earns >$50K/yr based on census features.

**Main Steps:**
- Data loading and preprocessing
- Label encoding and scaling
- Model training using classifiers (e.g., Logistic Regression, Random Forest)
- Accuracy and F1 evaluation
- Cross-validation and hyperparameter tuning

**Libraries Used:**
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`

---

### 2. `coronavirus_tweets.py`

**Dataset:** Tweets about COVID-19  
**Objective:** Sentiment analysis or classification based on tweet content.

**Main Steps:**
- Text preprocessing (stopwords removal, tokenization)
- Vectorization using TF-IDF or CountVectorizer
- Sentiment prediction with classifiers (e.g., Naive Bayes, SVM)
- Word cloud generation and exploratory visualizations

**Libraries Used:**
- `pandas`, `re`, `nltk`, `wordcloud`
- `scikit-learn`, `matplotlib`

---

### 3. `wholesale_customers.py`

**Dataset:** UCI Wholesale Customers  
**Objective:** Cluster customers based on spending habits.

**Main Steps:**
- Exploratory data analysis (EDA)
- Dimensionality reduction (PCA)
- Clustering using KMeans, DBSCAN, or Agglomerative Clustering
- Visualization of cluster results and feature relationships

**Libraries Used:**
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 🚀 Running the Scripts

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
