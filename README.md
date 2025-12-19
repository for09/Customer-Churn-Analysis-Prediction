# Machine Learning Approach to Bank Customer Retention

## Overview
Customer churn is a major financial risk for banks, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project implements an end-to-end machine learning workflow to predict customer churn and identify high-risk customer segments, enabling proactive and cost-effective retention strategies.

The solution combines Exploratory Data Analysis (EDA), unsupervised customer segmentation using K-Means and PCA, and supervised classification models (Random Forest and XGBoost), with final model selection driven by business impact rather than accuracy alone.

---

## Problem Statement
The objective is to accurately identify customers with a high probability of churning based on demographic, financial, and behavioral attributes, while handling class imbalance and non-linear feature interactions common in real-world banking data.

---

## Objectives
- Perform exploratory analysis to uncover key churn drivers  
- Segment customers using PCA + K-Means and interpret churn behavior per segment  
- Build and compare Random Forest and XGBoost classifiers  
- Optimize models for imbalanced data using business-driven evaluation metrics  
- Translate predictions into actionable retention insights  

---

## Dataset
- **Source:** Kaggle – Bank Customer Churn Dataset  
- **Size:** 10,000 customers  
- **Target Variable:** `Churn` (1 = Exited, 0 = Stayed)

### Features
- **Demographic:** Age, Gender, Geography  
- **Financial:** CreditScore, Balance, EstimatedSalary  
- **Behavioral:** NumOfProducts, Tenure, IsActiveMember, HasCrCard  

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Techniques:**  
  - Exploratory Data Analysis (EDA)  
  - PCA (Dimensionality Reduction)  
  - K-Means Clustering  
  - Random Forest Classification  
  - XGBoost Classification  

---

## Data Preprocessing
- Removed duplicate and non-informative identifier columns  
- One-Hot Encoding for categorical variables  
- Feature scaling using StandardScaler  
- Addressed class imbalance (~20% churn rate) using:
  - `class_weight='balanced'` for Random Forest  
  - Probability threshold optimization for XGBoost  
- Implemented end-to-end pipelines using `ColumnTransformer` and `Pipeline` to prevent data leakage  

---

## Exploratory Data Analysis (Key Findings)
- Overall churn rate: **20.37%**  
- Churn probability increases significantly after age 40  
- Active members and customers with more products are less likely to churn  
- Customers with high balances and low engagement show elevated churn risk  
- German customers exhibit the highest regional churn rate  

---

## Customer Segmentation (Unsupervised Learning)
- Applied K-Means clustering on standardized numerical features  
- Used PCA to preserve ~90% variance while reducing dimensionality  
- Optimal number of clusters determined using the Elbow Method (k = 4)

### Segment Summary
- **Cluster 0:** High balance, highly active customers (Low churn – 13%)  
- **Cluster 1:** Multiple products, low balance customers (Low churn – 12%)  
- **Cluster 2:** High balance but inactive customers (Medium churn – 29%)  
- **Cluster 3:** Older customers with medium balances (High churn – 36%)  

These segments provide clear targets for personalized retention strategies.

---

## Model Development & Evaluation

### Random Forest Classifier (Final Model)
- ROC-AUC: **0.847**  
- Churn Recall: **66.83%**  
- F1-Score: **0.61**  
- F₂-Score: **0.64**  

The model prioritizes recall, ensuring that a larger proportion of churners are identified, which is critical given the higher cost of missed churners.

---

### XGBoost Classifier
- ROC-AUC: **0.854**  
- Churn Recall (optimized threshold = 0.35): **59%**  
- F1-Score: **0.60**  

XGBoost achieved higher accuracy and precision, but lower recall compared to Random Forest.

---

## Business Impact Analysis
Assumed cost framework:
- False Negative (missed churner): $1,000 loss  
- False Positive (unnecessary offer): $50 cost  

### Net Value (per 2,000 customers)
- **Random Forest:** ~$256,200 net savings  
- **XGBoost:** ~$173,150 net savings  

Random Forest delivers approximately **48% higher ROI**, making it the preferred model for deployment.

---

## Key Insights
- Age is the strongest predictor of churn, peaking between 50–60  
- Customers with exactly two products show the lowest churn  
- High-balance customers are more likely to churn, likely due to competitive alternatives  
- Geographic differences play a significant role in churn behavior  

---

## Recommendations
- Deploy Random Forest with a tiered intervention strategy:
  - High risk (>70%): Direct relationship manager outreach  
  - Medium risk (50–70%): Targeted offers and engagement campaigns  
  - Low risk (30–50%): Minimal monitoring  
- Focus retention efforts on older customers, high-balance inactive users, and customers with 3+ products  

---

## Future Work
- Incorporate behavioral time-series data (transaction frequency, engagement trends)  
- Further optimize decision thresholds to increase recall  
- Validate model performance over multiple time periods to detect drift  

---

## References
- Pedregosa et al., Scikit-learn: Machine Learning in Python  
- Chen & Guestrin, XGBoost: A Scalable Tree Boosting System  
- He & Garcia, Learning from Imbalanced Data  
- Kaggle – Bank Customer Churn Dataset  
