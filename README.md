# Bengaluru House Price Prediction 

A machine learning project that predicts house prices in Bengaluru using property features such as location, square footage, BHK, and bathrooms.

### Tech Stack
Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn

### Features
- Data cleaning and feature engineering
- Outlier handling and location processing
- Random Forest Regression
- Hyperparameter tuning using GridSearchCV
- Interactive price prediction using Streamlit

### Model Performance

| Metric | R² Score |
|---|---:|
| Cross-validation | 0.7095 |
| Test set | 0.7631 |

**Best model:** Random Forest Regressor  
**Tuned parameters:** 200 estimators, maximum depth of 10.

### Run Locally

```bash
git clone https://github.com/nihairam35/Bengaluru-House-Price-Prediction.git
cd Bengaluru-House-Price-Prediction
pip install -r requirements.txt
streamlit run app.py
```

### Future Improvements
- Evaluate MAE and RMSE
- Improve generalization to unseen locations
- Retrain using updated housing data

**Note:** Predictions are estimates for educational purposes and are not official property valuations.
