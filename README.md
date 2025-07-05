# Titanic Survival Analysis

This repository explores the classic Titanic dataset from Kaggle to identify key factors that influenced passenger survival.  
It is part of an end-to-end machine learning pipeline consisting of data exploration, preprocessing, model building and evaluation.

## Project Overview

The project is divided into multiple stages:

1. **Exploratory Data Analysis (EDA)**  
   Identify patterns and correlations, clean the data, and engineer useful features.

2. **Modeling**  
   Train and evaluate machine learning models (e.g., logistic regression, random forest).

## Dataset

- Source: [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Goal: Predict survival outcome based on personal and travel-related features.

## Key Insights from EDA

| Feature       | Insight                                                                 |
|---------------|-------------------------------------------------------------------------|
| Sex           | Strong predictor: females had a significantly higher survival rate      |
| Pclass        | Higher class correlated with better survival chances                    |
| Fare          | Higher fare generally meant better odds of survival (after transformation) |
| Embarked      | Passengers from Cherbourg survived more often (likely linked to class)  |
| Deck          | Upper decks had higher survival rates                                   |
| FamilySize    | Small families (2–4 members) had the highest survival rate              |

## Features Created

- `LogFare`: log-transformed fare to reduce skew
- `FamilySize`: combined from SibSp and Parch
- `Deck`: extracted from the Cabin variable
- `CabinNumber`: numeric part of Cabin, tested but showed no clear signal
- `FareBin`, `LogFareBin`: binned versions for visual inspection

## Modeling Summary

Both a Logistic Regression and a Random Forest model were trained using a preprocessing pipeline (imputation, scaling, encoding).  
The models were evaluated on a test set using metrics such as accuracy, F1-score, and ROC AUC.  
Logistic Regression performed slightly better overall.

| Metric                | Logistic Regression | Random Forest     |
|-----------------------|---------------------|--------------------|
| Accuracy              | 0.79                | 0.78               |
| F1-Score (Survived)   | 0.71                | 0.70               |
| ROC AUC               | 0.8472              | 0.8354             |

## Repository Structure

- `titanic_eda_modeling.ipynb` – EDA and feature exploration  
- `titanic_model_training.ipynb` – Model building and evaluation  
- `README.md` – Project description  
- `train.csv` – Training data

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn

## Author

Richard Theus
richardtheus7@gmail.com  