
# Housing Price Prediction with Machine Learning

## Project Overview

This project focuses on predicting house prices using supervised machine learning techniques.  
Beyond achieving predictive performance, the main goal is to **understand model behavior, analyze errors, and iteratively improve results through data-driven decisions**.

The project follows a structured, end-to-end ML workflow similar to real-world machine learning projects.

---

## Problem Statement

Given a dataset of 546 houses with structural and qualitative features, the objective is to predict the house price as accurately as possible.

This is a **regression problem**, where the target variable is continuous (`price`).

---

## Dataset Description

Each row represents a house, and each column represents a feature.

### Target
- `price`: House price

### Numerical Features
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

### Categorical Features
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus`

---

## Project Approach

The project was developed following these steps:

1. **Data Loading and Inspection**
   - Dataset exploration
   - Identification of numerical and categorical features

2. **Preprocessing**
   - One-hot encoding for categorical variables
   - Numerical features passed without scaling
   - All preprocessing handled through **scikit-learn Pipelines** to prevent data leakage

3. **Model Training**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor

4. **Evaluation Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score

5. **Error Analysis**
   - Visualization of real vs predicted prices
   - Identification of cases with large prediction errors
   - Analysis of systematic failures

6. **Feature Engineering**
   - Error analysis revealed poor performance on houses with:
     - 3 or more bedrooms
     - 2 or more stories
   - New features were engineered to explicitly capture these patterns:
     - `many_bedrooms`
     - `multi_stories`
     - `large_and_multi`

7. **Re-evaluation**
   - Models retrained using engineered features
   - Focused evaluation on previously problematic cases

---

## Results Summary

- Linear Regression proved to be a strong baseline, outperforming more complex models in initial experiments.
- Error analysis revealed systematic underperformance on larger, multi-story houses.
- Targeted feature engineering improved predictions for these specific cases.
- The project highlights that **model understanding and feature engineering can be more impactful than model complexity**.

---

## Key Learnings

- Error analysis is essential to understand where and why models fail.
- Feature engineering guided by domain insights can significantly improve performance.
- Pipelines ensure consistent preprocessing and prevent data leakage.
- Simpler models can outperform complex ones when properly engineered.
- Metrics must always be interpreted in the context of the business problem.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

---

## Project Structure

```
├── data/
│   └── housing.csv
├── src/
│   ├── metrics_housing.py
│   └── feature_engineering.py
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Hyperparameter tuning using GridSearch and cross-validation
- Evaluation using relative error metrics such as MAPE
- Additional feature engineering based on economic or location data
- Model deployment as an API for real-time predictions

---

## Conclusion

This project demonstrates a complete machine learning workflow, emphasizing **understanding model behavior and improving performance through informed decisions**, rather than blindly increasing model complexity.

---

## Author

**EderrekLab**
