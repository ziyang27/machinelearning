# House Price Prediction - Machine Learning Project

## Project Overview
This project focuses on predicting house prices using machine learning techniques in Python. The implementation follows a complete machine learning pipeline from data exploration to model training and evaluation.

## Video Reference
This project is based on the tutorial: [House Price Prediction in Python - Full Machine Learning Project](insert-video-link-here)

## Project Steps

### 1. Data Preparation
- Import necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Read and inspect the dataset
- Handle null values by dropping rows with missing data

### 2. Data Analysis & Visualization
- Created histograms to understand feature distributions
- Generated heatmaps and correlation matrices to identify relationships
- Plotted scatterplots with longitude and latitude for geographical insights

### 3. Feature Engineering
- Applied log transformations to normalize right-skewed distributions
- Performed one-hot encoding for categorical features using `pd.get_dummies()`
- Created new features:
  - `bedroom_ratio = total_bedrooms / total_rooms`
  - `household_rooms = total_rooms / households`

### 4. Data Scaling
- Standardized features using `StandardScaler` from `sklearn.preprocessing`

### 5. Model Training & Evaluation
- Implemented Linear Regression (`sklearn.linear_model`)
- Implemented Random Forest Regressor (`sklearn.ensemble`)
- Conducted hyperparameter tuning using GridSearchCV (`sklearn.model_selection`)

## Why Feature Transformation?
Feature transformation to a Gaussian (normal) distribution improves performance for models that assume or perform better with normally distributed data. Many statistical methods, including linear and logistic regression, work best with approximately normal distributions. These transformations enhance model accuracy and reliability.

## Usage
1. Clone this repository
2. Install required packages
3. Run the Jupyter notebook or Python script

## Future Improvements
- Experiment with additional feature engineering
- Try different machine learning algorithms
- Implement more advanced hyperparameter tuning techniques
- Add cross-validation for more robust evaluation

## Acknowledgments
Credit to the original tutorial video for the project structure and inspiration.