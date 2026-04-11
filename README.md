# Credit Card Fraud Detection

This project uses machine learning and deep learning to detect fraudulent credit card transactions. The dataset is highly imbalanced (only 0.17% fraud), so I used techniques like SMOTE and proper evaluation metrics instead of just accuracy.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **284,807 transactions**, 492 are fraud
- Features V1-V28 are PCA-transformed for privacy, only Time and Amount are original

## What I Did

1. **Data Cleaning** — checked for missing values, duplicates, and outliers
2. **EDA** — visualised class distribution, amounts, time patterns, correlations
3. **PCA Analysis** — explored variance and created 2D/3D projections
4. **Feature Engineering** — extracted hour from time, log-transformed amount
5. **Scaling** — StandardScaler fitted on training data only (to avoid data leakage)
6. **SMOTE** — oversampled fraud cases in training data only (test data untouched)
7. **Clustering** — K-Means, DBSCAN, and Isolation Forest
8. **Trained 11 models** — compared classical ML, ensemble, and deep learning
9. **Hyperparameter Tuning** — used Optuna for XGBoost and LightGBM
10. **Evaluation** — confusion matrices, ROC curves, PR curves, SHAP explainability

## Models Used

**Classical ML:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN

**Ensemble:** XGBoost, LightGBM

**Deep Learning:** Neural Network (MLP), Autoencoder, LSTM, 1D-CNN

## Key Takeaways

- Accuracy alone is misleading — a model predicting "no fraud" gets 99.8% accuracy but catches nothing
- Recall and F1-Score are more important for fraud detection
- SMOTE must only be applied to training data to avoid data leakage
- Ensemble methods (XGBoost, LightGBM) generally perform best on tabular data

## How to Run

**Note:** Requires Python 3.10–3.12 (TensorFlow doesn't support 3.13+)

```bash
git clone https://github.com/YOUR_USERNAME/CreditCard_Fraud_Detection.git
cd CreditCard_Fraud_Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in the `Data/` folder. Then open the notebook in Jupyter or VS Code.

## Libraries

NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, TensorFlow, Optuna, SHAP

## Author

Rachana — MSc Artificial Intelligence
