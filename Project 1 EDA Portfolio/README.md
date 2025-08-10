# EDA Portfolio Project

## Overview
This project demonstrates comprehensive Exploratory Data Analysis (EDA) techniques using Python and pandas. The focus is on showcasing systematic data exploration, cleaning, and visualization skills that form the foundation of any data science project.

## Video Reference
This project is based on the tutorial: [Data Analyst Portfolio Project (Exploratory Data Analysis With Python Pandas)](https://www.youtube.com/watch?v=QlbyGPVaRSE&ab_channel=InfiniteCodes)

## Objectives
- Demonstrate proficiency in data manipulation with pandas
- Showcase data cleaning and preprocessing techniques
- Create meaningful visualizations to understand data patterns
- Apply systematic EDA methodology to real-world datasets

## Methodology

### 1. Environment Setup
**Libraries Used:**
- `pandas` - Data manipulation and analysis
- `seaborn` - Statistical data visualization

### 2. Data Loading
- Used `pd.read_csv()` to load the dataset
- Initial data inspection with basic pandas methods

### 3. Data Inspection
**Initial Assessment:**
- `df.head()` - First 5 rows overview
- `df.shape` - Dataset dimensions
- `df.info()` - Data types and memory usage
- `df.describe()` - Statistical summary

### 4. Data Cleaning 
**Systematic cleaning approach:**

1. **Filtering** - Removed irrelevant or out-of-scope records
2. **Feature Selection** - Identified and kept relevant columns
3. **Feature Engineering** - Created new features (e.g., age from date of birth)
4. **Null Value Handling**:
   - `df.isna().sum()` - Identified missing values
   - `df.dropna()` - Removed or imputed missing data
5. **Duplicate Removal** - Identified and handled duplicate records
6. **Index Reset** - Clean index after data modifications
7. **Data Type Conversion** - Ensured appropriate data types
8. **Column Renaming** - Improved column names for clarity
9. **Column Reordering** - Logical arrangement of features

### 5. Exploratory Visualizations
**Chart Types Used:**
- **Histograms** - Distribution analysis of numerical variables
- **Violin Plots** - Distribution shape and density visualization
- **Scatter Plots** - Relationship analysis between variables

### 6. Data Querying
- Applied various pandas DataFrame methods for data exploration
- Used filtering, grouping, and aggregation techniques
- Extracted meaningful insights through targeted queries

## Technical Skills Demonstrated
- **Data Manipulation**: pandas DataFrame operations
- **Data Cleaning**: Systematic approach to data quality
- **Feature Engineering**: Creating meaningful variables
- **Data Visualization**: Statistical plotting with seaborn
- **Exploratory Analysis**: Hypothesis-driven data exploration

## Key Learnings
- **Data Quality**: Understanding the importance of thorough data inspection
- **Systematic Approach**: Following a structured EDA methodology
- **Visualization Selection**: Choosing appropriate charts for different data types
- **Feature Engineering**: Creating valuable insights from existing data
- **Documentation**: Importance of clear, reproducible analysis

## Conclusion
This project successfully demonstrates fundamental EDA skills and establishes a solid foundation for data analysis projects. The systematic approach to data exploration, cleaning, and visualization provides a template for future data science endeavors.

---

**Project Status**: ✅ Completed  
**Last Updated**: 15/7/25 
**Time Invested**: 4 Hours