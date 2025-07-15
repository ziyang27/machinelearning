# Titanic Survival Prediction

## Overview
This project uses machine learning to predict passenger survival on the Titanic based on various features such as age, gender, class, and fare. The Titanic dataset is a classic beginner's project that demonstrates binary classification, data preprocessing, and feature engineering techniques in a real-world historical context.

## Video Reference
This project is based on the tutorial: [Project 15. Titanic Survival Prediction using Machine Learning in Python | Machine Learning Project](https://www.youtube.com/watch?v=QlbyGPVaRSE&ab_channel=InfiniteCodes)

## Objectives
- Predict passenger survival using machine learning classification
- Demonstrate comprehensive data preprocessing techniques
- Apply feature engineering and categorical encoding
- Analyze survival patterns and extract meaningful insights
- Evaluate model performance using appropriate metrics

## Methodology

### 1. Environment Setup
**Dependencies:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `matplotlib` - Data visualization
- `seaborn` - Statistical plotting
- `scikit-learn` - Machine learning algorithms

### 2. Data Collection & Processing
**Comprehensive data preprocessing pipeline:**

#### Missing Value Treatment:
- **Quantitative Variables**: Replaced null values with mean
  - `Age`: Filled with average age
  - `Fare`: Filled with mean fare (if missing)
  
- **Qualitative Variables**: Replaced null values with mode
  - `Embarked`: Filled with most common port
  - `Cabin`: Handled missing cabin information

#### Data Cleaning:
- Removed or handled irrelevant columns
- Standardized data formats
- Addressed outliers where necessary

### 3. Data Analysis & Visualization
**Key Insights Discovered:**

#### Primary Finding:
- **Gender Impact**: Females had significantly higher survival rates than males
- **Statistical Evidence**: [Include specific percentages from your analysis]

#### Additional Patterns:
- **Class Effect**: Higher class passengers had better survival rates
- **Age Distribution**: Survival patterns across different age groups
- **Fare Analysis**: Relationship between ticket price and survival
- **Embarkation Point**: Survival rates by port of departure

#### Visualizations Created:
- Survival rate by gender (bar plots)
- Age distribution of survivors vs non-survivors
- Survival by passenger class
- Correlation heatmap of numerical features

### 4. Feature Engineering
**Categorical Encoding:**
- **Binary Encoding**: Sex (male=0, female=1)
- **Label Encoding**: Embarked port (C=0, Q=1, S=2)
- **Ordinal Encoding**: Pclass (already numerical: 1st, 2nd, 3rd class)
- **Feature Creation**: [Any new features derived from existing ones]

### 5. Model Pipeline
**Data Preparation:**
- **Feature Selection**: Separated independent variables (X) from target (y)
- **Train-Test Split**: Divided data for training and validation
- **Feature Scaling**: [If applied, mention scaling techniques]

### 6. Model Training
**Algorithm Selection:**
- **Primary Model**: [Specify algorithm used - e.g., Logistic Regression, Random Forest, SVM]
- **Reasoning**: [Why this algorithm was chosen]
- **Training Process**: Fitted model on training data

### 7. Model Evaluation
**Performance Metrics:**
- **Accuracy**: [Overall prediction accuracy]
- **Precision**: [Precision score for survival prediction]
- **Recall**: [Recall score for survival prediction]
- **F1-Score**: [Harmonic mean of precision and recall]
- **Confusion Matrix**: Detailed classification results

## Technical Skills Demonstrated
- **Data Preprocessing**: Handling missing values systematically
- **Feature Engineering**: Categorical encoding and feature selection
- **Data Visualization**: Creating insightful plots for analysis
- **Statistical Analysis**: Extracting meaningful patterns from data
- **Machine Learning**: Binary classification implementation
- **Model Evaluation**: Comprehensive performance assessment

## Key Learnings
- **Data Quality**: Importance of systematic missing value treatment
- **Feature Impact**: Understanding which variables most influence survival
- **Historical Context**: Social dynamics reflected in survival patterns
- **Preprocessing Pipeline**: Building robust data cleaning workflows
- **Model Selection**: Choosing appropriate algorithms for binary classification

## Challenges Faced
- **Missing Data**: Significant missing values in Age and Cabin columns
- **Categorical Encoding**: Deciding optimal encoding strategies
- **Feature Selection**: Determining which variables to include/exclude
- **Class Imbalance**: Handling uneven distribution of survival outcomes
- **Overfitting**: Ensuring model generalizes well to unseen data

## Data Processing Pipeline
```python
# Missing Value Treatment
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Categorical Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Feature Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']
```

## Usage
1. **Setup environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   ```bash
   jupyter notebook notebook.ipynb
   ```

3. **Make predictions:**
   ```python
   import pickle
   model = pickle.load(open('models/titanic_model.pkl', 'rb'))
   prediction = model.predict(new_passenger_data)
   ```

## Historical Context
The Titanic disaster provides a compelling case study in:
- **Social Class**: Clear survival differences by passenger class
- **Gender Norms**: "Women and children first" policy impact
- **Economic Factors**: Wealth correlation with survival chances
- **Age Demographics**: Survival patterns across age groups

## Conclusion
This project successfully demonstrates the complete machine learning pipeline from data preprocessing to model evaluation. The analysis reveals important historical patterns about survival on the Titanic, with gender being the strongest predictor of survival. The systematic approach to handling missing data and categorical encoding provides a solid foundation for more complex classification problems.

The insights gained not only showcase technical ML skills but also demonstrate the ability to extract meaningful patterns from historical data.

---

**Project Status**: ✅ Completed  
**Last Updated**: 15/7/2025
**Time Invested**: 3 Hours