# Telemetry Solutions Recruitment Challenge -Submission
Platform | Build Status |
-------- | ------------ |
JUPYTER| [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)

## African Credit Scoring Challenge - Documentation and Usage

## Exploratory Data Analysis (EDA) - Experiment_EDA_001
# Overview
This notebook performs an Exploratory Data Analysis (EDA) on the loan dataset to uncover key patterns, detect data quality issues, and visualize relationships between features. The analysis helps inform subsequent modeling decisions by providing a comprehensive understanding of the data distribution, missing values, and feature interactions.

# EDA Approach
* Data Loading: Load data from the specified source using relative paths.

* Data Cleaning: Identify and handle missing values and incorrect data types.

* Univariate Analysis: Visualize distributions of key numeric and categorical variables.

* Bivariate Analysis: Examine relationships between features and the target variable (loan default) using bar plots, box plots, and correlation heatmaps.

* Insights & Summary: Highlight important trends and potential predictors for default risk.

# How to Run
* Clone the repository.

* Open the notebook located at Notebooks/Experiment_EDA_001.ipynb.

* Run all cells sequentially. The notebook uses relative paths to load data from the Data folder (e.g., ../Data/Train.csv).

* Install required dependencies listed in requirements.txt if necessary (e.g., pandas, matplotlib, seaborn).

# Dependencies
```bash
Python 3.x
pandas
matplotlib
seaborn
```
## Model Selection - Experiment_Model_Selection_002
# Model Selection Summary
To identify the most effective model for our classification task, we evaluated several algorithms including Logistic Regression, Random Forest, and XGBoost. Each model was trained on the training dataset and assessed on a validation set using key metrics such as accuracy, precision, recall, and F1-score.

Despite tuning and optimizing the Random Forest and XGBoost models, the K-Nearest Neighbors (KNN) classifier consistently demonstrated superior performance on the validation data in terms of predictive accuracy and balanced precision/recall.

The selected KNN model was further fine-tuned using hyperparameter optimization (e.g., number of neighbors, distance metrics) to enhance its robustness and generalization capability.

Key points:

* Multiple models were compared systematically on validation metrics.

* KNN showed the best balance of accuracy and generalization.

* Preprocessing (such as feature scaling) is critical to KNN’s performance.

* The final saved model (best_KNN.pkl) reflects the optimized KNN ready for deployment.

# How to Run
* Open the notebook located at Notebooks/Experiment_Model_Selection_002.ipynb.



