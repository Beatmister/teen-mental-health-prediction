# Teen Mental Health Prediction 
## Overview 
This project explores the relationship between social meadia usage, lifestyle factors and depression among teenagers. The goal was to identify influential factors associated with depression and develop a machine learning model capable of predicting depression labels based on behavioral and mental health indicators.

The project covers the complete data science workflow, including data exploration, preprocessing, feature analysis, model development, hyperparameter tuning and evaluation.

## Dataset
Source: https://www.kaggle.com/datasets/algozee/teenager-menthal-healy?select=Teen_Mental_Health_Dataset.csv<br/>
The dataset contains information about teenagers, including:
- Demographic characteristics
- Social media usage behavior
- Sleep habits
- Academic performance
- Physical activity
- Stress and anxiety levels
- Depression label (target variable)

### Target Variable
0 = No Depression
1 = Depression

## Key Findings
The exploratory analysis revealed several factors that appear to be associated with depression among teenagers.

### Strong Indicators
The following variavles showd stronger relationships with the target variable:
- Daily social media usage
- Sleep duration
- Stress level
- Anxiety Level

These features displayed noticeable differenced between the depression and non-depression groups.

## Result
The machine learning pipeline successfully learned patterns within the dataset and achieved strong predictive performance on the test data.
Due to the strong class imbalance and the relatively small number of positive samples, performance metrics should be interpreted carefully. The project should therefore be viewed primarily as a demonstration of a complete machine learing workflow rather than as evidence of a production-ready depression prediction system.

