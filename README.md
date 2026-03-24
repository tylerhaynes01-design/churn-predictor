 📊 Customer Churn Predictor

A full end-to-end machine learning application that predicts whether a
telecom customer is likely to churn based on their account details and 
usage behavior.

 🔍 Project Overview
Customer churn is one of the most costly problems in subscription-based 
businesses. This project builds a predictive model to identify at-risk 
customers before they leave, enabling proactive retention strategies.

🛠️ Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Streamlit (web app deployment)
- Jupyter Notebook (exploration and modeling)

## 📈 Models Compared
| Model | AUC Score |
|-------|-----------|
| Logistic Regression | 0.836 |
| Random Forest | 0.814 |
| XGBoost (tuned) | 0.836 |

 🔑 Key Findings
- 26.5% of customers churned in the dataset
- Top churn predictors: Total Charges, Tenure, and Monthly Charges
- Month-to-month customers churn at significantly higher rates
- New customers are at the highest risk of churning
  
 🚀 How to Run Locally
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run app.py
