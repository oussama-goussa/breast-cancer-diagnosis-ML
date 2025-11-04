# 🧬 Breast Cancer Diagnosis Classification using Machine Learning

*Predicting benign or malignant breast tumors using real clinical data and AI models.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/pandas-data--analysis-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-green)

---

## 📖 Overview

This project implements a **machine learning system for breast cancer diagnosis classification**.  
It predicts whether a tumor is **benign** or **malignant** using the **Breast Cancer Wisconsin Diagnostic Dataset**.  
Through preprocessing, model training, and evaluation, the system compares multiple ML algorithms to identify the most accurate model for medical diagnosis assistance.

---

## 🎯 Key Features

- **📊 Data Analysis & Visualization** — Exploratory Data Analysis (EDA) to understand correlations between features.  
- **⚙️ Data Preprocessing** — Cleaning, handling missing values, and normalization.  
- **🤖 Multi-Model Training** — Logistic Regression, KNN, Decision Tree, and SVM.  
- **📈 Performance Evaluation** — Precision, Recall, F1-score, and Confusion Matrix.  
- **💡 Model Interpretability** — Identifying key clinical features influencing predictions.  
- **🩺 Practical Application** — Can be integrated into a simple doctor’s interface for real-time diagnostic support.

---

## 🧩 Dataset

**Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Samples:** 569  
- **Features:** 30 numeric attributes  
- **Classes:**  
  - `M` → Malignant  
  - `B` → Benign  

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.8+ |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Environment | Jupyter Notebook |

---

## 🧠 Model Training Pipeline

```
Data Collection → Preprocessing → Feature Scaling → Model Training 
       ↓                                   ↓
  Visualization                      Model Evaluation
       ↓                                   ↓
  Best Model Selection → Deployment / Prediction
```

---

## 📁 Project Structure

```
breast-cancer-diagnosis-ML/
│
├── 📓 breast-cancer-diagnosis-ML.ipynb      # Main Jupyter notebook
├── 📄 README.md                         # Project documentation
│
├── 📁 data/
│   └── breast_cancer_data.csv          
│
└── 📄 requirements.txt                  
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/oussama-goussa/breast-cancer-diagnosis-ML.git
cd breast-cancer-diagnosis-ML
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook Projet_IA_IISE_GOUSSA.ipynb
```

---

## ⚙️ Example Workflow

```python
# Load Dataset
import pandas as pd
data = pd.read_csv('data/breast_cancer_data.csv')

# Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.iloc[:, 2:32])

# Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_scaled, data['diagnosis'])

# Prediction
sample = X_scaled[0].reshape(1, -1)
print(model.predict(sample))
```

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|--------|-----------|
| Logistic Regression | 97.3% | 97% | 97% | 97% |
| KNN | 96.8% | 96% | 96% | 96% |
| Decision Tree | 94.5% | 94% | 94% | 94% |
| SVM | **98.2%** | **98%** | **98%** | **98%** |

✅ **Best Model:** Support Vector Machine (SVM)

---

## 🧠 Feature Importance Visualization

```text
Top contributing features:
1️⃣ mean concavity
2️⃣ worst radius
3️⃣ mean perimeter
4️⃣ mean texture
5️⃣ worst smoothness
```

---

## 🧾 References
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)

---

## 🧑‍💻 Author

**Oussama GOUSSA**  
🎓 *Filière : IISE*  
🏫 *Université Ibn Zohr, Faculté des Sciences d’Agadir*  
📅 *Année Universitaire : 2023–2024*

---

<div align="center">

**Made with ❤️ using Machine Learning for Healthcare**

If this project inspired you, please give it a ⭐ on GitHub!  

[![GitHub stars](https://img.shields.io/github/stars/oussama-goussa/breast-cancer-diagnosis-ML?style=social)](https://github.com/oussama-goussa/breast-cancer-diagnosis-ML)

</div>
