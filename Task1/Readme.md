# Student Exam Score Prediction

## Overview

This project predicts a student's exam score based on multiple factors such as:

* Hours studied per day
* Sleep hours per day
* Attendance percentage
* Previous exam scores

The model is trained using multiple regression algorithms, and the best-performing model is selected using GridSearchCV. A Streamlit web application is included for real-time predictions.

**Live Streamlit App:** [Student Score Predictor](https://student-score-predictions.streamlit.app/)
**Kaggle Notebook:** [Student Score Prediction](https://www.kaggle.com/code/maroofiums/student-score-prediction)

---

## Features

* **Data Cleaning & Exploration**

  * Handled missing values and outliers
  * Visualized relationships between features and exam scores

* **Regression Models Tested**

  1. Linear Regression
  2. Ridge Regression
  3. Lasso Regression
  4. Random Forest Regressor
  5. Gradient Boosting Regressor

* **Model Selection**

  * GridSearchCV used for hyperparameter tuning
  * Model performance evaluated using R², MAE, and MSE
  * Best model saved as `best_student_score_model.pkl`

* **Streamlit Web Application**

  * Interactive sliders for user input
  * Predicts exam score in real-time
  * User-friendly interface

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd <repo-folder>
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/main.py
```

---

## Usage

1. Open the Streamlit app in a browser.
2. Adjust sliders for:

   * Hours Studied
   * Sleep Hours
   * Attendance Percent
   * Previous Exam Scores
3. Click **Predict Exam Score**.
4. The predicted exam score is displayed instantly.

---

## Dataset Details

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| hours_studied      | Number of hours the student studies daily |
| sleep_hours        | Number of hours the student sleeps daily  |
| attendance_percent | Student's class attendance (%)            |
| previous_scores    | Average of previous exam scores           |
| exam_score         | Target variable: score in final exam      |

---

## Model Performance

| Model             | R² Score | MAE | MSE |
| ----------------- | -------- | --- | --- |
| Gradient Boosting | Best     | x   | y   |
| Random Forest     | Good     | x   | y   |
| Linear Regression | Baseline | x   | y   |

*(Fill in your actual performance metrics from training)*

---

## Project Workflow

1. Load Dataset → Inspect & clean data
2. Exploratory Data Analysis (EDA) → Visualize feature relationships
3. Train/Test Split → 80/20 split
4. Train Multiple Regression Models → Tune hyperparameters using GridSearchCV
5. Select Best Model → Based on highest R² score
6. Serialize Model → Save as `.pkl` file
7. Deploy Streamlit App → Real-time predictions

---

## Future Improvements

* Add feature importance visualization
* Support batch predictions via CSV upload
* Add historical performance charts for students

---