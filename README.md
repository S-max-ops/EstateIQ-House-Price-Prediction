# 🏡 EstateIQ – AI-Powered House Price Prediction

An end-to-end Machine Learning application that predicts residential property prices using an XGBoost Regressor. Built with Python, Scikit-learn and Streamlit.

## 📌 Project Overview

EstateIQ is a Machine Learning-based property valuation application developed using the Kaggle House Prices: Advanced Regression Techniques dataset.

The project uses property characteristics such as overall quality, living area, bedrooms, bathrooms, garage capacity and neighborhood to estimate residential property prices.

It combines data preprocessing, feature engineering, model training and an interactive Streamlit interface.

## ✨ Features

- Interactive property price prediction.
- XGBoost regression model.
- Automated numerical feature scaling.
- Categorical feature encoding.
- Feature engineering.
- Model performance comparison.
- Interactive analytics dashboard.
- Saved model and preprocessing pipeline.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Scikit-learn | Preprocessing and evaluation |
| XGBoost | Machine Learning model |
| Streamlit | Interactive web application |
| Plotly | Interactive charts |
| Joblib | Model serialization |

## 🤖 Model Performance

Four regression algorithms were evaluated during the project.

| Algorithm | Validation R² Score |
|---|---:|
| Linear Regression | 0.4440 |
| Random Forest | 0.8874 |
| Gradient Boosting | 0.8933 |
| XGBoost | **0.9228** |

### Kaggle Performance

- Competition: House Prices – Advanced Regression Techniques
- Model: XGBoost Regressor
- Public Leaderboard RMSLE: **0.12952**

The Kaggle leaderboard score and local validation metrics are calculated on different datasets.

## 📂 Project Structure

```text
EstateIQ-House-Price-Prediction/
│
├── Data/
│
├── models/
│   ├── input_template.pkl
│   ├── preprocessor.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/S-max-ops/EstateIQ-House-Price-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd EstateIQ-House-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

The project uses the Kaggle competition dataset:

[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Download the dataset separately and place the required CSV files inside the `Data/` directory if you want to reproduce the notebook analysis.

The trained model and preprocessing artifacts are stored in the `models/` directory.

## 🚀 Future Improvements

- Add more property input features.
- Improve prediction accuracy through feature engineering and model optimization.
- Add explainable AI using SHAP.
- Add INR currency conversion.
- Deploy the application publicly.
- Improve the user interface and analytics dashboard.

## 👨‍💻 Author

**Sujal Goswami**

GitHub: [@S-max-ops](https://github.com/S-max-ops)
# 🏡 EstateIQ – House Price Prediction

🚀 **Live Demo:** https://estateiq-house-price-prediction-afni8lkgarmwxj9nlgbeha.streamlit.app/

💻 **GitHub Repository:** https://github.com/S-max-ops/EstateIQ-House-Price-Prediction

---

⭐ If you find this project interesting, consider giving it a star!
