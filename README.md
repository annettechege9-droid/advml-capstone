
# ADVANCED MACHINE LEARNING PROJECT PROPOSAL

## Title

Advanced Machine Learning for Motor Insurance Claim Prediction and Policyholder Profiling in Kenya

### 1. Introduction

Motor insurance companies need to understand their customers and estimate potential claim costs to improve pricing, risk management, and decision-making. This project will use a Kenyan motor insurance dataset to identify different policyholder groups and predict the amount of money that may be claimed.

Unlike a basic regression project, this study will combine **unsupervised learning, dimensionality reduction, supervised learning, and neural networks** to demonstrate different Advanced Machine Learning techniques.

### 2. Problem Statement

Insurance companies need accurate ways of identifying different types of policyholders and estimating claim amounts. Traditional approaches may not fully capture patterns within customer and vehicle data.

This project will therefore use Advanced Machine Learning techniques to:

* Discover natural groups of policyholders.
* Explore the structure of the data.
* Predict total claim amounts.
* Compare traditional machine learning models with a neural network.
* Identify the factors that contribute most to prediction errors and claim amounts.

### 3. Main Objective

To apply Advanced Machine Learning techniques to predict motor insurance claim amounts and identify different policyholder profiles in Kenya.

### 4. Specific Objectives

1. To clean and explore the Kenyan motor insurance dataset.
2. To use **clustering** to identify groups of similar policyholders.
3. To use **PCA (Principal Component Analysis)** to reduce dimensions and visualize patterns in the data.
4. To develop regression models for predicting `Total_Claim_Amount_KES`.
5. To develop and compare a **Neural Network** with traditional machine learning models.
6. To evaluate and compare the performance of the different models.
7. To identify the most important factors influencing claim predictions.

### 5. Research Questions

1. What natural groups of policyholders can be identified from the insurance data?
2. What characteristics distinguish the different policyholder groups?
3. Which factors are most useful for predicting total claim amounts?
4. How well can machine learning models predict motor insurance claim amounts?
5. Does a neural network perform better than traditional and ensemble models?
6. What types of cases produce the largest prediction errors?

### 6. Dataset

The project will use a Kenyan motor insurance dataset containing approximately **2,000 records and 19 variables**.

Important variables include:

* Customer age and gender
* Region
* Vehicle type and vehicle age
* Vehicle value
* Engine capacity
* Use purpose
* Annual premium
* Previous claims
* Driver experience
* No-claim bonus
* Claims frequency
* Accident cause
* Total claim amount

The main prediction target will be:

**`Total_Claim_Amount_KES`**

### 7. Advanced Machine Learning Techniques

**Clustering / Unsupervised Learning**
K-Means clustering will be used to identify natural groups of policyholders based on characteristics such as vehicle value, customer age, driver experience, previous claims, and premiums. The resulting groups will then be compared based on their claim characteristics.

**Dimensionality Reduction**
PCA will be used to reduce the number of dimensions and visualize the main patterns in the dataset. It will also help us understand whether the identified clusters are clearly separated.

**Supervised Machine Learning**
The claim amount will be treated as a regression problem. The following models will be considered:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting
* XGBoost

**Neural Network**
A feed-forward Artificial Neural Network will also be developed using TensorFlow/Keras and compared with the other regression models.

### 8. Methodology

The project will follow these main steps:

**Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Clustering → PCA → Regression Models → Neural Network → Model Tuning → Evaluation → Explainability → Error Analysis → Conclusion**

The dataset will be divided into training and testing sets. Cross-validation and hyperparameter tuning will be used where appropriate to improve model performance.

### 9. Model Evaluation

Regression models will be evaluated using:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **R²** – Coefficient of Determination

For clustering, **Silhouette Score** will be used to evaluate the quality of the clusters.

The final models will also be compared based on their prediction errors.

### 10. Tools and Technologies

The project will use:

* **Python** – main programming language
* **Pandas** – data cleaning and manipulation
* **NumPy** – numerical calculations
* **Matplotlib & Seaborn** – data visualization
* **Scikit-learn** – clustering, PCA, regression, preprocessing, evaluation and tuning
* **XGBoost** – gradient boosting model
* **TensorFlow/Keras** – neural network development
* **SHAP** – model explainability

### 11. Expected Results

The project is expected to:

* Identify meaningful groups of motor insurance policyholders.
* Show the major patterns in the insurance dataset using PCA.
* Produce models capable of predicting claim amounts.
* Determine which model gives the best prediction performance.
* Identify important factors affecting claim predictions.
* Provide useful insights that could support insurance risk assessment and decision-making.

### 12. Conclusion

This project will demonstrate how different Advanced Machine Learning techniques can be combined to solve a practical insurance problem. It will go beyond simple regression by using **clustering, dimensionality reduction, ensemble learning, and neural networks** to both profile policyholders and predict insurance claim amounts.
