# 🏡 Real Estate Investment Advisor  
### ML-Powered Investment Classification + 5-Year Price Prediction  
<img src="https://img.shields.io/badge/Status-Completed-brightgreen"> <img src="https://img.shields.io/badge/Python-3.10-blue"> <img src="https://img.shields.io/badge/MLflow-Enabled-orange"> <img src="https://img.shields.io/badge/Streamlit-App-red">

---

## 📌 Project Summary

The Real Estate Investment Advisor project is an end-to-end Machine Learning system designed to assist investors and home buyers in making data-driven real estate decisions.

The project addresses two core business problems:

Classification Problem – Identify whether a property is a Good Investment based on price trends, locality factors, and infrastructure indicators.

Regression Problem – Predict the Future Property Price (5-Year Horizon) to estimate long-term appreciation.

The solution combines robust data preprocessing, feature engineering, Exploratory Data Analysis (EDA), multiple ML models, MLflow experiment tracking, and Streamlit deployment, making it suitable for real-world production use.

## 🔧 Model Development & Evaluation (Detailed)
### 1️⃣ Problem Formulation

| Task                | Type           | Target Variable           |
| ------------------- | -------------- | ------------------------- |
| Investment Decision | Classification | `Good_Investment` (0 / 1) |
| Price Forecasting   | Regression     | `Future_Price_5Y`         |

--

## 2️⃣ Feature Engineering

Key engineered features include:

Age_of_Property = Current Year − Year Built

Price_per_SqFt = Price / Size

Infrastructure Score (derived from transport, schools, hospitals)

Investment Label (Good_Investment)

Based on appreciation threshold, locality quality, and pricing metrics

These features significantly improved model stability and interpretability.

### Preprocessing Pipeline

✔ Missing value imputation
✔ Scaling of numerical features
✔ One-hot encoding of categorical features
✔ Consistent pipeline reused for training & inference

This ensured no data leakage and seamless deployment.

### Models Trained
🔹 Classification Models
| Model                    | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| Logistic Regression      | Baseline & explainability                |
| Random Forest Classifier | Non-linear patterns & feature importance |

🔹 Regression Models

| Model                   | Purpose                          |
| ----------------------- | -------------------------------- |
| Linear Regression       | Baseline comparison              |
| Random Forest Regressor | Capturing complex price dynamics |


--

## 📌 **Project Overview**

The **Real Estate Investment Advisor** is an end-to-end Machine Learning project that analyzes residential property data to:

### ✔ Predict whether a property is a **Good Investment** (Classification)  
### ✔ Predict the **Estimated Price After 5 Years** (Regression)  
### ✔ Provide data-driven insights using EDA & visualizations  
### ✔ Track experiments and register the best models using **MLflow**  
### ✔ Serve predictions through an interactive **Streamlit App**

This system combines Data Science, Machine Learning, MLflow model tracking, and Streamlit deployment to deliver a full production-ready real estate analytics solution.

---

## 🎯 **Business Objective**

Help buyers, investors, and agencies evaluate property investment potential and forecast future prices using historical trends and property characteristics.

---

## 🧱 **Tech Stack**

| Component | Technology |
|----------|------------|
| Programming | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Experiment Tracking | MLflow |
| Deployment (UI) | Streamlit |
| Model Storage | Joblib |
| Logging | JSON, MLflow Tracking |

---

## 📂 **Project Structure**

DSRealEstateAdvisorPrediction/
│── data/
│── models/
│ ├── best_investment_classifier.pkl
│ ├── best_future_price_regressor.pkl
│ ├── metadata.json
│── mlruns/ # MLflow Experiments
│── plots/ # Saved EDA Visuals
│── streamlit/
│ └── app.py # Streamlit Application
│── RealEstatePricePredictionMLClassificationAndRegressionProject.ipynb
│── RealEstatePricePredictionMLClassificationAndRegressionProject.py
│── requirements.txt
│── README.md



---

## 📊 **Exploratory Data Analysis (EDA)**

Key insights from the dataset:

### **Univariate Analysis**
- Price distribution is right-skewed with large variation across cities.
- Most properties fall within 2BHK–3BHK range.
- Age_of_Property clusters strongly around 10–25 years.

### **Bivariate Analysis**
- Price increases with Size_in_SqFt but varies heavily across cities.
- Price_per_SqFt influenced by locality and city.
- BHK count vs price shows linear increase but inconsistent across markets.

### **Multivariate Analysis**
- Correlation heatmap reveals:
  - Strong relationship between Price and Size_in_SqFt  
  - Negative correlation between Age_of_Property and Price  
  - Minimal effect from Furnishing Status on pricing

### **General Market Observations**
- Metropolitan cities dominate premium pricing.
- Locality-level variance is a major contributor to price fluctuations.
- Schools/Hospitals have mild influence but add context to investment scoring.

---

## 🤖 **Model Development**

### **1️⃣ Classification Model**
**Objective:** Predict whether a property is a *Good Investment*.

Models Evaluated:
- Logistic Regression  
- Random Forest Classifier ✔ *(Best Model)*

**Final Model Performance:**
- Accuracy: ~1.0  
- Precision/Recall/F1: 1.0  
- ROC-AUC: 1.0  
*(on the optimized/clean dataset)*

---

### **2️⃣ Regression Model**
**Objective:** Predict the **Future Price After 5 Years**

Models Evaluated:
- Linear Regression  
- Random Forest Regressor ✔ *(Best Model)*

**Final Model Performance:**
- RMSE: Low (excellent predictive capability)
- R² Score: High (strong model fit)

Both final models are saved inside the `models/` directory.

---

## 🔥 **MLflow Tracking & Model Registry**

The project uses **MLflow** to track:
- Model parameters  
- Metrics (accuracy, RMSE, F1-score, etc.)  
- Trained models as artifacts  
- Best-run selection and metadata storage

All experiment runs are stored under:

mlruns / 

You can visualize them via:

mlflow ui

---

## 🌐 **Streamlit Application**

The interactive Streamlit app allows users to:

### 🧾 **Enter Property Details**
✔ City  
✔ Locality  
✔ Size in SqFt  
✔ BHK  
✔ Property Type  
✔ Furnishing Status  
✔ Floors & Age  
✔ Nearby amenities  

### 📌 **Outputs**
- **Good Investment?** → Yes / No  
- **Confidence Score**  
- **Predicted Future Price (5 Years)**  

### 📉 **Visual Insights**
- Feature importance  
- Market distribution charts  
- Example heatmaps  

### 🚀 Run Streamlit App

cd streamlit
streamlit run app.py


---

## 🛠 **How to Run the Project Locally**

### 1️⃣ Create Environment
pip install -r requirements.txt


### 2️⃣ Run Training Script
python RealEstatePricePredictionMLClassificationAndRegressionProject.py


### 3️⃣ Start MLflow UI (optional)
mlflow ui 

### 4️⃣ Launch Streamlit Dashboard
cd streamlit
streamlit run app.py


---

## 📁 **Models Saved**

### ✔ best_investment_classifier.pkl  
Random Forest Classifier trained for investment scoring.

### ✔ best_future_price_regressor.pkl  
Random Forest Regressor for 5-year price forecasting.

### ✔ metadata.json  
Stores feature structure and model metadata for the Streamlit app.

---

## 🚀 **Future Enhancements**

- Add SHAP-based interpretability  
- Include geospatial mapping for city/locality  
- Integrate real-time property listings API  
- Deploy Streamlit app to cloud (Streamlit Cloud / AWS / Azure)  

---

## 📜 **License**
This project is licensed under the MIT License.

---

## 🤝 **Contributions**
Pull requests, suggestions, and improvements are welcome!

---

## ⭐ If you like this project, don’t forget to star the repo!
