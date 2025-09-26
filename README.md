# Title : Machine Learning Approaches for Early Detection and Severity Prediction of Parkinson's Disease from Voice Features
## 🌟 Overview

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting millions worldwide. This project implements a robust machine learning pipeline for PD detection using vocal biomarkers extracted from sustained phonations. Our framework addresses critical challenges in medical AI including data leakage prevention, class imbalance handling, and model interpretability.
Key Achievements:
- ✅ **90% Accuracy** with XGBoost classifier
- ✅ **93.33% F1-Score** demonstrating balanced performance
- ✅ **Patient-level splitting** preventing data leakage
- ✅ **SHAP interpretability** for clinical transparency
- ✅ **Comprehensive model comparison** (6 algorithms)

### Live Demo & Resources
-🌐 Live Demo: [Google Colab Notebook]([https://colab.research.google.com/drive/your-notebook-id](https://colab.research.google.com/drive/1UBGqgTrBXnwl9yGFK8cW1mmsGm9Pwctx?usp=sharing))
-📄 Research Paper: [Overleaf Document]([https://www.overleaf.com/read/abc123xyz456](https://www.overleaf.com/read/nmcdpqymcwtm#fcd055))
-📊 Dataset: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification)

## 🚀 Key Features
### 🔬 Scientific Rigor
- **Patient-level data splitting** ensuring clinical validity
- **SMOTE oversampling** handling class imbalance (75.4% PD vs 24.6% Healthy)
- **Mutual Information feature selection** identifying key biomarkers
- **5-fold cross-validation** assessing model robustness

### 🤖 Machine Learning Models
- **XGBoost** - Best performing algorithm (90% accuracy)
- **LightGBM** - Efficient gradient boosting (86.67% accuracy)
- **Random Forest** - Ensemble method (80% accuracy)
- **Support Vector Machine** - Maximum margin classifier (80% accuracy)
- **Logistic Regression** - Linear baseline (80% accuracy)
- **Gradient Boosting** - Sequential ensemble (86.67% accuracy)

### 📊 Model Interpretability
- **SHAP analysis** for feature importance visualization
- **Confusion matrices** for error analysis
- **ROC curves** for discriminatory capability assessment
- **Feature importance rankings** for biomarker identification

## 📁 Dataset

### UCI Parkinson's Voice Dataset
- **Samples**: 195 voice recordings
- **Patients**: 32 unique individuals
- **Classes**: 147 Parkinson's vs 48 Healthy controls
- **Features**: 22 acoustic parameters per sample

#### Key Feature Categories:
- **Fundamental Frequency**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- **Jitter Measures**: Cycle-to-cycle frequency variation
- **Shimmer Measures**: Cycle-to-cycle amplitude variation  
- **Nonlinear Dynamics**: RPDE, DFA, Spread1, Spread2, PPE

**Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification)

## 🔬 Methodology

### Data Preprocessing Pipeline
1. **Data Cleaning**: Handle missing values, sanitize column names
2. **Feature Scaling**: StandardScaler (Z-score normalization)
3. **Class Balancing**: SMOTE oversampling for minority class
4. **Feature Selection**: Mutual Information based selection (top 25 features)

### Patient-Level Splitting
- **Training**: 70% patients (22 patients, 134 samples)
- **Validation**: 15% patients (5 patients, 30 samples)
- **Testing**: 15% patients (5 patients, 31 samples)

### Model Training & Evaluation
```python
# Example training pipeline
models = {
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Evaluation metrics
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
```

## 📈 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 90.00% | 91.30% | 95.45% | 93.33% | 89.77% |
| LightGBM | 86.67% | 87.50% | 95.45% | 91.30% | 93.18% |
| Gradient Boosting | 86.67% | 87.50% | 95.45% | 91.30% | 95.45% |
| Random Forest | 80.00% | 90.00% | 81.82% | 85.71% | 90.91% |
| SVM | 80.00% | 100.00% | 72.73% | 84.21% | 91.48% |
| Logistic Regression | 80.00% | 94.44% | 77.27% | 85.00% | 89.20% |

### Key Biomarkers Identified
1. **PPE (Pitch Period Entropy)** - 0.2540 importance
2. **spread1** - 0.2214 importance  
3. **MDVP:Fo(Hz)** - 0.2066 importance
4. **spread2** - 0.1920 importance
5. **MDVP:Jitter(Abs)** - 0.1902 importance

## 👥 Authors

- **Md Mehedi Hasan** - *Lead Researcher* - [@mehedihasan](https://github.com/mehedihasan)
- **Most. Sonia Islam** - *Co-Researcher* - [@soniaislam](https://github.com/soniaislam)  
- **Lamia Muntaha** - *Co-Researcher* - [@lamiamuntaha](https://github.com/lamiamuntaha)

## 🌟 Acknowledgments

We would like to acknowledge:
- **UCI Machine Learning Repository** for providing the Parkinson's Voice Dataset
- **Open Source Community** for the excellent machine learning libraries
- **Research Advisors** for their guidance and support

---

<div align="center">

**⭐️ Don't forget to star this repository if you find it useful!**

[![Star History Chart](https://api.star-history.com/svg?repos=/0mehedihasan/Machine-Learning-Course)](https://star-history.com/0mehedihasan/Machine-Learning-Course)

</div>
```
