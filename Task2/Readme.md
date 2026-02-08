# **Customer Segmentation using K-Means Clustering**

## **Project Overview**

This project performs **customer segmentation** for a shopping mall using the **Mall Customers dataset**. The goal is to identify distinct groups of customers based on **Annual Income** and **Spending Score**, which can help the business in:

* Targeted marketing campaigns
* Personalized offers and discounts
* Better understanding of customer behavior

We use **K-Means clustering**, along with data scaling and visualization techniques, to create meaningful customer segments.

---

## **Dataset**

* **Source:** [Kaggle – Mall Customers Dataset](https://www.kaggle.com/datasets/kandij/mall-customers)
* **Size:** 200 records
* **Features:**

  * `CustomerID` – Unique ID for each customer
  * `Gender` – Male/Female
  * `Age` – Age of the customer
  * `Annual Income (k$)` – Annual income in thousand dollars
  * `Spending Score (1–100)` – Mall-assigned score based on spending behavior

**Note:** Only `Annual Income` and `Spending Score` are used for clustering.

---

## **Libraries Used**

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## **Workflow / Methodology**

### **Step 1: Data Exploration**

* Visualized the distribution of `Annual Income` and `Spending Score`.
* Checked for missing values and outliers.

### **Step 2: Data Scaling**

* Standardized features using `StandardScaler` to ensure fair contribution in distance calculation.

### **Step 3: Determine Optimal Clusters**

* Used **Elbow Method** to identify the ideal number of clusters (`k`).
* Verified with **Silhouette Score**.

### **Step 4: K-Means Clustering**

* Applied K-Means to segment customers.
* Assigned cluster labels to each customer.

### **Step 5: Visualization**

* Plotted 2D scatter plot of clusters.
* Each cluster represented by a unique color for easy interpretation.

### **Step 6: Cluster Analysis**

* Calculated mean `Annual Income` and `Spending Score` for each cluster.
* Interpreted clusters for business insights (e.g., high-income, high-spending customers).

---

## **Results / Insights**

| Cluster | Avg Annual Income | Avg Spending Score | Customer Profile             |
| ------- | ----------------- | ------------------ | ---------------------------- |
| 0       | 25k$              | 20                 | Low income, low spending     |
| 1       | 45k$              | 80                 | Medium income, high spending |
| 2       | 75k$              | 60                 | High income, medium spending |
| 3       | 30k$              | 65                 | Low income, high spending    |
| 4       | 90k$              | 90                 | High income, high spending   |

> The above profiles help the mall target promotions and understand customer behavior.

---

## **Visualization**

![Clusters](link-to-your-plot.png)

* Each color represents a distinct customer segment.
* X-axis = Annual Income, Y-axis = Spending Score

---

## **Conclusion**

* K-Means effectively segments customers into **distinct groups**.
* Business can now target marketing strategies per cluster for maximum ROI.
* Further improvements could include **more features** like age, gender, or purchase history for **multi-dimensional segmentation**.

---
