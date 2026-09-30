# advml-capstone
## Project Title
Predicting Motor Insurance Claim Amounts in Kenya
________________________________________
## Introduction
Insurance companies receive claims from policyholders after accidents or other insured events. The amount paid for each claim can be different depending on factors such as the vehicle, driver, previous claims and the type of policy.
This project will use machine learning to predict the amount of money that may be paid for a motor insurance claim using a Kenyan motor insurance dataset.
My previous Machine Learning project focused on classifying high-risk policyholders. For this Advanced Machine Learning project, I will focus on predicting the actual claim amount, making it a regression problem.
________________________________________
## Problem Statement
Insurance companies need to estimate how much they may have to pay for insurance claims. If the claim amount can be predicted more accurately, it can help with financial planning and claims management.
The dataset contains information about policyholders, vehicles, drivers, policies, premiums and previous claims. I will use these variables to investigate whether machine learning can predict the total claim amount.
________________________________________
## Main Objective
To develop a machine learning model that can predict motor insurance claim amounts in Kenya.
________________________________________
## Specific Objectives
1.	To clean and prepare the motor insurance dataset for machine learning.
2.	To explore the factors that may be related to insurance claim amounts.
3.	To create useful features from the available data.
4.	To develop different machine learning regression models for predicting claim amounts.
5.	To compare the performance of the different models.
6.	To improve the selected models using cross-validation and hyperparameter tuning.
7.	To identify the variables that contribute most to the predicted claim amounts.

________________________________________
## Research Questions
1.	What factors are related to motor insurance claim amounts?
2.	How accurately can machine learning predict motor insurance claim amounts?
3.	Which machine learning model gives the best prediction performance?
4.	Does hyperparameter tuning improve model performance?
5.	Which features are most important when predicting claim amounts?
6.	What types of claims are difficult for the model to predict accurately?
________________________________________
##  Dataset
I will use the Kenyan Motor Insurance 2023–2024 dataset.
The dataset contains:
•	2,000 records
•	19 variables
Some of the variables include:
•	Customer Age
•	Gender
•	Region
•	Vehicle Type
•	Vehicle Age
•	Vehicle Value
•	Engine Capacity
•	Use Purpose
•	Annual Premium
•	Previous Claims Count
•	Driver Experience
•	No-Claim Bonus
•	Policy Term
•	Claims Frequency
•	Total Claim Amount
•	Accident Cause
________________________________________
## Target Variable
The target variable will be:
Total_Claim_Amount_KES
This is the amount of money associated with the insurance claim.
Since the target is a continuous numerical value, this will be treated as a regression problem.
For example, instead of predicting:
High Risk / Low Risk
the model will predict something like:
Predicted Claim Amount = KES 180,000
The predicted amount will then be compared with the actual claim amount.
________________________________________
## Variables to be Used
Potential input variables include:
•	Customer age
•	Gender
•	Region
•	Vehicle type
•	Vehicle age
•	Vehicle value
•	Engine capacity
•	Use purpose
•	Annual premium
•	Previous claims
•	Driver experience
•	No-claim bonus
•	Policy term
•	Third-party-only status
The variables will be examined first to determine which ones are appropriate for prediction.
________________________________________
## Machine Learning Models
I will start with a simple model and then move to more advanced models.
1. Linear Regression
This will be used as the baseline model.
2. Random Forest Regressor
This will allow the project to model nonlinear relationships.
3. Gradient Boosting Regressor
This will be used to improve predictions by learning from previous errors.
4. XGBoost Regressor
This will be used as an advanced boosting model.
The models will then be compared to determine how well they predict claim amounts.
________________________________________
## Advanced Machine Learning Techniques
The following techniques will be used:
Feature Engineering
Creating useful variables from the existing data where necessary.
Cross-Validation
Using different portions of the training data to check whether the model performs consistently.
Hyperparameter Tuning
Testing different model settings to find better-performing models.
Ensemble Learning
Using models such as Random Forest that combine multiple decision trees.
Explainable Machine Learning
Using feature importance and possibly SHAP to understand why the model makes certain predictions.
Error Analysis
Examining cases where the predicted claim amount is very different from the actual amount.
________________________________________
## Project Process
The project will follow these main steps:
1. Data Collection
Obtain and load the Kenyan motor insurance dataset.
↓
2. Data Cleaning
Check for missing values, duplicates, incorrect data types and unusual values.
↓
3. Exploratory Data Analysis
Use graphs and statistics to understand the dataset and claim amounts.
↓
4. Feature Engineering
Create useful features where necessary.
↓
5. Data Preprocessing
Encode categorical variables and prepare the data for modelling.
↓
6. Train-Test Split
Separate the data into training and testing sets.
↓
7. Baseline Model
Train Linear Regression.
↓
8. Advanced Models
Train Random Forest, Gradient Boosting and XGBoost.
↓
9. Cross-Validation
Check model performance across different training/validation samples.
↓
10. Hyperparameter Tuning
Improve the selected models by testing different parameters.
↓
11. Model Evaluation
Compare the models using:
•	MAE
•	RMSE
•	R²
↓
12. Feature Importance
Identify the variables that contribute most to predictions.
↓
13. Error Analysis
Investigate large prediction errors and understand where the model struggles.
↓
14. Final Model
Select the most suitable model based on the results.
↓
15. Conclusions
Answer the research questions and discuss the findings.
________________________________________
12. Evaluation Metrics
MAE — Mean Absolute Error
Shows the average difference between the predicted and actual claim amounts.
RMSE — Root Mean Squared Error
Shows prediction error while giving more importance to larger errors.
R² — R-Squared
Shows how much of the variation in claim amounts is explained by the model.
________________________________________
## Tools
Programming Language
Python
Libraries
•	Pandas — data cleaning and manipulation
•	NumPy — numerical calculations
•	Matplotlib — visualization
•	Seaborn — data visualization
•	Scikit-learn — machine learning and evaluation
•	XGBoost — advanced boosting model
•	SHAP — model explanation
________________________________________
## Expected Results
At the end of the project, I expect to:
•	Develop a model that predicts insurance claim amounts.
•	Identify factors that are useful in predicting claim amounts.
•	Compare different regression models.
•	Determine whether advanced models improve prediction performance.
•	Identify the most important features.
•	Understand where the final model makes large errors.
•	Use explainable machine learning to understand model predictions.
________________________________________
________________________________________
## Project Significance
This project will demonstrate how machine learning can be applied to a real-world problem in the Kenyan insurance industry.
Predicting claim amounts can provide useful information for understanding potential claim costs and can support areas such as insurance planning, claims management and financial analysis.
The project will also demonstrate the use of advanced machine-learning techniques beyond the classification model used in my previous project.
________________________________________

