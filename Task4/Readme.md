# **Loan Prediction System**

## **Project Overview**

This project builds a **predictive model** to determine whether a loan will be approved or not based on applicant information.
It is a **binary classification problem** solved using multiple machine learning algorithms, hyperparameter tuning, and model evaluation metrics.

---

## **Motivation**

Banks and financial institutions face risks in loan approval.
Automating the process with predictive models can **reduce risk**, **save time**, and **improve decision-making**.

---

## **Dataset**

* **Source:** [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
* **Train set:** 614 records, 12 features
* **Test set:** 367 records, 11 features
* **Key Features:**

  * `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` → Categorical applicant info
  * `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term` → Numerical financial info
  * `Credit_History`, `Property_Area` → Loan-related history
  * `Loan_Status` → Target variable (Y/N)

---

## **Libraries & Tools**

* **Python 3.x**
* **Pandas, NumPy** → Data manipulation
* **Matplotlib, Seaborn** → Visualization
* **Scikit-learn** → Modeling, preprocessing, metrics, hyperparameter tuning
* **Joblib** → Model persistence

---

## **Project Workflow**

### **1. Data Cleaning & Preprocessing**

* Removed `Loan_ID` column
* Filled **missing values**:

  * Numerical features → median
  * Categorical features → mode
* Converted categorical variables to numerical using **Label Encoding**

### **2. Exploratory Data Analysis (EDA)**

* Visualized **numerical feature distributions** using histograms
* Checked for **data imbalance** and missing values

### **3. Feature Scaling**

* Scaled numerical features using **StandardScaler** for models sensitive to feature scale (Logistic Regression, SVM, KNN)

### **4. Model Training & Hyperparameter Tuning**

* Trained multiple models with **GridSearchCV** for hyperparameter optimization:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * Decision Tree
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
  * Naive Bayes

* **Evaluation Metrics:**

  * Accuracy
  * F1-Score (primary metric)
  * AUC Score

### **5. Best Model Selection**

* **Best Model:** Logistic Regression
* **Best Params:** `{'C': 0.1, 'penalty': 'l2'}`
* F1-Score: 0.908
* Accuracy: 0.862
* AUC: 0.801

### **6. Model Persistence**

* Saved the trained model using **Joblib** for future predictions:

```python
joblib.dump(best_model, "best_model.pkl")
```

### **7. Predictions on Test Data**

* Loaded the saved model and made predictions on unseen test data:

```python
pkl_model = joblib.load("best_model.pkl")
prediction = pkl_model.predict(test)
```

---

## **Results**

| Model               | F1-Score | Accuracy | AUC   |
| ------------------- | -------- | -------- | ----- |
| Logistic Regression | 0.908    | 0.862    | 0.801 |
| Random Forest       | 0.903    | 0.854    | 0.783 |
| SVM                 | 0.903    | 0.854    | 0.809 |
| KNN                 | 0.897    | 0.846    | 0.776 |
| Naive Bayes         | 0.896    | 0.846    | 0.765 |
| Gradient Boosting   | 0.894    | 0.837    | 0.726 |
| Decision Tree       | 0.879    | 0.821    | 0.726 |

**Best Model:** Logistic Regression (saved as `best_model.pkl`)

---

## **Key Features & Strengths**

* Handles **missing data** effectively
* Works with both **categorical and numerical features**
* Scalable to larger datasets
* **Automated hyperparameter tuning** for optimal performance
* Saved model allows **fast deployment** without retraining

---

## **Potential Improvements**

* Use **SMOTE or other resampling techniques** to handle class imbalance
* Try **ensemble methods** like XGBoost or LightGBM for higher accuracy
* Feature engineering: combine applicant income and co-applicant income
* Deploy as a **web API** using Flask/FastAPI for real-time predictions

---

## **Conclusion**

This project demonstrates **data cleaning, preprocessing, machine learning, and model deployment** for a practical **loan prediction problem**.
It’s a strong portfolio project showcasing **end-to-end ML workflow** for binary classification tasks.
